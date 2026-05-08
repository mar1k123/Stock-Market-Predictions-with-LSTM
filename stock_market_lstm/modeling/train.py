import numpy as np
import tensorflow as tf
from loguru import logger
from tqdm import trange
from tqdm.keras import TqdmCallback

from stock_market_lstm.features import all_mid_data, train_data
from stock_market_lstm.config import configure_logging

configure_logging()


class DataGeneratorSeq:
    def __init__(self, prices: np.ndarray, batch_size: int, num_unroll: int):
        self._prices = prices
        self._prices_length = len(self._prices) - num_unroll
        self._batch_size = batch_size
        self._num_unroll = num_unroll
        self._segments = self._prices_length // self._batch_size
        self._cursor = [offset * self._segments for offset in range(self._batch_size)]

    def next_batch(self) -> tuple[np.ndarray, np.ndarray]:
        batch_data = np.zeros((self._batch_size), dtype=np.float32)
        batch_labels = np.zeros((self._batch_size), dtype=np.float32)

        for b in range(self._batch_size):
            if self._cursor[b] + 1 >= self._prices_length:
                self._cursor[b] = np.random.randint(0, (b + 1) * self._segments)

            batch_data[b] = self._prices[self._cursor[b]]
            batch_labels[b] = self._prices[self._cursor[b] + np.random.randint(0, 5)]
            self._cursor[b] = (self._cursor[b] + 1) % self._prices_length

        return batch_data, batch_labels

    def unroll_batches(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        unroll_data, unroll_labels = [], []
        for _ in range(self._num_unroll):
            data, labels = self.next_batch()
            unroll_data.append(data)
            unroll_labels.append(labels)
        return unroll_data, unroll_labels


def build_sequences(series: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for idx in range(seq_len, len(series)):
        x.append(series[idx - seq_len : idx])
        y.append(series[idx])
    x_arr = np.array(x, dtype=np.float32).reshape(-1, seq_len, 1)
    y_arr = np.array(y, dtype=np.float32).reshape(-1, 1)
    return x_arr, y_arr


def build_model(seq_len: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(seq_len, 1)),
            tf.keras.layers.LSTM(200, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(200, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(150),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.MeanSquaredError(),
    )
    return model


def recursive_forecast(
    model: tf.keras.Model, series: np.ndarray, start_index: int, context: int, steps: int
) -> tuple[np.ndarray, float]:
    history = list(series[start_index - context : start_index].astype(np.float32))
    preds = []
    mse = 0.0

    for step in range(steps):
        x_input = np.array(history[-context:], dtype=np.float32).reshape(1, context, 1)
        pred = float(model.predict(x_input, verbose=0)[0, 0])
        preds.append(pred)
        history.append(pred)
        target = float(series[start_index + step])
        mse += 0.5 * ((pred - target) ** 2)

    return np.array(preds, dtype=np.float32), mse / steps


if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    seq_len = 50
    batch_size = 500
    epochs = 30
    n_predict_once = 50

    dg = DataGeneratorSeq(train_data, 5, 5)
    u_data, u_labels = dg.unroll_batches()
    for ui, (dat, lbl) in enumerate(zip(u_data, u_labels)):
        logger.debug("Unrolled index {}", ui)
        logger.debug("Inputs: {}", dat)
        logger.debug("Output: {}", lbl)

    x_train, y_train = build_sequences(train_data, seq_len)
    model = build_model(seq_len)

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        verbose=0,
        callbacks=[lr_scheduler, TqdmCallback(verbose=1)],
    )

    train_mse_ot = [float(loss) for loss in history.history["loss"]]
    test_mse_ot = []
    predictions_over_time = []
    x_axis_seq = []

    max_forecast_horizon = max(1, len(all_mid_data) - seq_len - 1)
    forecast_horizon = min(n_predict_once, max_forecast_horizon)
    max_start_index = len(all_mid_data) - forecast_horizon

    test_points_seq = np.arange(seq_len, max_start_index, 5).tolist()
    if not test_points_seq and seq_len < max_start_index:
        # Ensure at least one evaluation window for short datasets.
        test_points_seq = [seq_len]

    progress = trange(epochs, desc="Evaluating epochs", unit="epoch")
    for ep in progress:
        predictions_seq = []
        mse_test_loss_seq = []

        for w_i in test_points_seq:
            preds, mse_loss = recursive_forecast(
                model=model,
                series=all_mid_data,
                start_index=w_i,
                context=seq_len,
                steps=forecast_horizon,
            )
            predictions_seq.append(preds)
            mse_test_loss_seq.append(mse_loss)

            if ep == 0:
                x_axis_seq.append(list(range(w_i, w_i + forecast_horizon)))

        if mse_test_loss_seq:
            current_test_mse = float(np.mean(mse_test_loss_seq))
        else:
            logger.warning(
                "Skipping test MSE evaluation: not enough points (len={}, seq_len={}, horizon={})",
                len(all_mid_data),
                seq_len,
                forecast_horizon,
            )
            current_test_mse = float("nan")
        test_mse_ot.append(current_test_mse)
        predictions_over_time.append(predictions_seq)
        progress.set_postfix(
            train_mse=f"{train_mse_ot[ep]:.6f}",
            test_mse=f"{current_test_mse:.6f}",
        )

