# stock_market_lstm/modeling/train.py
"""
LSTM model for stock price time series prediction.
Implements training, recursive forecasting, and model persistence.
"""

import numpy as np
import tensorflow as tf
from loguru import logger
from tqdm.keras import TqdmCallback
from pathlib import Path


class LSTMPredictor:
    """
    LSTM-based stock price predictor.

    Architecture:
        LSTM(200) -> Dropout(0.2) -> LSTM(200) -> Dropout(0.2) -> LSTM(150) -> Dense(1)

    Workflow:
        1. Build model: model.build()
        2. Train: model.train(data, epochs=30)
        3. Predict: predictions = model.forecast(data, start, context, steps)
        4. Save: model.save("model.h5")
        5. Load: model.load("model.h5")

    Example:
        model = LSTMPredictor(seq_len=50)
        model.build()
        model.train(train_data, epochs=30)
        predictions = model.recursive_forecast(full_data, start_idx=1000, context=50, steps=30)
        model.save("trained_model.h5")
    """

    def __init__(self, seq_len: int = 50):
        """
        Initialize LSTM predictor.

        Args:
            seq_len: Number of past time steps to use for each prediction.
                     Larger = more context, but slower training.
        """
        self.seq_len = seq_len
        self.model = None
        self.history = None

    def build(self) -> "LSTMPredictor":
        """
        Build the LSTM model architecture.
        Must be called before training.

        Returns:
            self for method chaining
        """
        self.model = tf.keras.Sequential([
            # Input: (batch, seq_len, 1) — time series window
            tf.keras.layers.Input(shape=(self.seq_len, 1)),

            # First LSTM layer with 200 units
            # return_sequences=True passes full sequence to next LSTM
            tf.keras.layers.LSTM(200, return_sequences=True),
            tf.keras.layers.Dropout(0.2),  # Prevent overfitting

            # Second LSTM layer
            tf.keras.layers.LSTM(200, return_sequences=True),
            tf.keras.layers.Dropout(0.2),

            # Third LSTM layer
            # return_sequences=False (default) — returns only last output
            tf.keras.layers.LSTM(150),

            # Output: single predicted value
            tf.keras.layers.Dense(1),
        ])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=tf.keras.losses.MeanSquaredError(),
        )

        logger.info(f"Model built: seq_len={self.seq_len}")
        return self

    def train(
            self,
            data: np.ndarray,
            epochs: int = 30,
            batch_size: int = 500,
            validation_split: float = 0.0,
            verbose: int = 1
    ) -> tf.keras.callbacks.History:
        """
        Train the LSTM model on prepared data.

        Args:
            data: 1D array of preprocessed stock prices (EMA smoothed)
            epochs: Number of training epochs. More = potentially better fit
            batch_size: Samples per gradient update. Larger = faster but more memory
            validation_split: Fraction of data to use for validation (0.0 to 1.0)
            verbose: 0 = silent, 1 = progress bar

        Returns:
            Training history object (contains loss values per epoch)

        Raises:
            RuntimeError: If model hasn't been built yet
        """
        if self.model is None:
            raise RuntimeError(
                "Model not built! Call model.build() before training."
            )

        # Convert time series into supervised learning format
        x_train, y_train = self._build_sequences(data)

        logger.info(
            f"Training on {len(x_train)} sequences "
            f"({epochs} epochs, batch_size={batch_size})"
        )

        # Learning rate scheduler: reduce LR when loss plateaus
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,  # Multiply LR by 0.5 when triggered
            patience=2,  # Wait 2 epochs before reducing
            min_lr=1e-6,  # Don't go below this
            verbose=1
        )

        callbacks = [lr_scheduler]
        if verbose:
            callbacks.append(TqdmCallback(verbose=1))

        # Train the model
        self.history = self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            shuffle=False,  # Keep chronological order for time series
            verbose=0,
            callbacks=callbacks
        )

        final_loss = self.history.history['loss'][-1]
        logger.info(f"Training complete. Final loss: {final_loss:.6f}")

        return self.history

    def recursive_forecast(
            self,
            series: np.ndarray,
            start_index: int,
            context: int = None,
            steps: int = 50
    ) -> np.ndarray:
        """
        Generate multi-step predictions using recursive forecasting.

        At each step, the model uses its own previous prediction as input
        to predict the next value. This is also called "autoregressive" prediction.

        Args:
            series: Full time series data (train + test)
            start_index: Position to start predicting from
            context: How many past values to use as initial context.
                     Defaults to self.seq_len.
            steps: How many steps to predict into the future

        Returns:
            Array of predicted values (length = steps)

        Raises:
            RuntimeError: If model hasn't been trained
        """
        if self.model is None:
            raise RuntimeError("Model not trained! Call train() first.")

        context = context or self.seq_len

        # Use recent history as initial context for prediction
        history = list(series[start_index - context: start_index].astype(np.float32))
        predictions = []

        for _ in range(steps):
            # Take the last 'context' values and reshape for model input
            x_input = np.array(history[-context:], dtype=np.float32).reshape(1, context, 1)

            # Predict next value
            pred = float(self.model.predict(x_input, verbose=0)[0, 0])
            predictions.append(pred)

            # Add prediction to history for next step
            history.append(pred)

        return np.array(predictions, dtype=np.float32)

    def evaluate_forecast(
            self,
            series: np.ndarray,
            start_index: int,
            context: int = None,
            steps: int = 50
    ) -> tuple[np.ndarray, float]:
        """
        Make predictions and calculate Mean Squared Error.

        Args:
            series: Full time series data
            start_index: Position to start predicting from
            context: Context window size
            steps: Prediction horizon

        Returns:
            Tuple of (predictions_array, mean_squared_error)
        """
        predictions = self.recursive_forecast(series, start_index, context, steps)

        # Calculate MSE against actual values
        targets = series[start_index: start_index + steps]
        mse = np.mean((predictions - targets) ** 2)

        return predictions, mse

    def save(self, filepath: str) -> None:
        """
        Save trained model to disk.

        Args:
            filepath: Path to save the model (e.g., "models/my_model.h5")
        """
        if self.model is None:
            raise RuntimeError("No model to save! Build and train a model first.")

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(filepath)
        logger.info(f"Model saved to: {filepath}")

    def load(self, filepath: str) -> "LSTMPredictor":
        """
        Load a saved model from disk.
        The model can be used immediately for prediction without retraining.

        Args:
            filepath: Path to the saved model (.h5 file)

        Returns:
            self for method chaining

        Raises:
            FileNotFoundError: If the model file doesn't exist
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        self.model = tf.keras.models.load_model(filepath)
        logger.info(f"Model loaded from: {filepath}")
        return self

    def _build_sequences(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert time series into supervised learning format.

        For each position i, creates:
            X[i] = data[i-seq_len : i]     (input window)
            y[i] = data[i]                  (target value)

        Args:
            data: 1D array of time series values

        Returns:
            Tuple of (X, y) where:
                X shape = (n_samples, seq_len, 1)
                y shape = (n_samples, 1)
        """
        x, y = [], []

        for i in range(self.seq_len, len(data)):
            x.append(data[i - self.seq_len: i])  # Past window
            y.append(data[i])  # Current value

        x_arr = np.array(x, dtype=np.float32).reshape(-1, self.seq_len, 1)
        y_arr = np.array(y, dtype=np.float32).reshape(-1, 1)

        return x_arr, y_arr