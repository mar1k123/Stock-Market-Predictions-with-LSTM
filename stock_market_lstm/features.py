# stock_market_lstm/features.py
"""
Feature engineering and data preprocessing for stock price prediction.
Handles scaling, smoothing, and sequence preparation.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from loguru import logger


class StockPreprocessor:
    """
    Prepare stock data for LSTM model training.

    Pipeline:
        1. Calculate mid price from High/Low
        2. Split into train/test sets
        3. Scale data with MinMaxScaler (using sliding windows)
        4. Apply Exponential Moving Average (EMA) smoothing to training data

    Example:
        processor = StockPreprocessor(test_size=0.2, ema_gamma=0.1)
        result = processor.process(df)

        # Access results
        train_data = result['train']      # Training data (smoothed)
        test_data = result['test']        # Test data (scaled only)
        full_data = result['full']        # Concatenated train + test
        original_df = result['original']  # Original DataFrame
    """

    def __init__(
            self,
            test_size: float = 0.2,
            smoothing_window: int = 2500,
            ema_gamma: float = 0.1
    ):
        """
        Initialize the preprocessor.

        Args:
            test_size: Fraction of data to use for testing (0.0 to 1.0)
            smoothing_window: Size of window for MinMaxScaler fitting.
                              Smaller = faster, larger = better scaling for long series.
            ema_gamma: Smoothing factor for EMA (0.0 to 1.0).
                       Higher = more weight on recent values = less smoothing.
        """
        self.test_size = test_size
        self.smoothing_window = smoothing_window
        self.ema_gamma = ema_gamma
        self.scaler = MinMaxScaler()

    def process(self, df: pd.DataFrame) -> dict:
        """
        Run the full preprocessing pipeline.

        Args:
            df: DataFrame with columns [Date, High, Low, Close]

        Returns:
            dict with keys:
                - 'train': np.ndarray — training data (EMA smoothed, scaled)
                - 'test': np.ndarray — test data (scaled, no smoothing)
                - 'full': np.ndarray — concatenated train + test
                - 'original': pd.DataFrame — original DataFrame (sorted)

        Raises:
            ValueError: If there are fewer than 100 data points
        """
        # Sort by date to ensure chronological order
        df = df.sort_values('Date').reset_index(drop=True)

        # Calculate mid price: average of daily high and low
        high_prices = df['High'].to_numpy()
        low_prices = df['Low'].to_numpy()
        mid_prices = (high_prices + low_prices) / 2.0

        # Safety check: need enough data for meaningful training
        if len(mid_prices) < 100:
            raise ValueError(
                f"Not enough data: {len(mid_prices)} rows found.\n"
                f"Minimum required: 100 rows.\n"
                f"Tip: Ensure your CSV has at least 100 trading days of data."
            )

        # Split: first part = training, last part = testing
        split_idx = int(len(mid_prices) * (1 - self.test_size))

        train_raw = mid_prices[:split_idx].reshape(-1, 1)
        test_raw = mid_prices[split_idx:].reshape(-1, 1)

        logger.info(f"Split data: {len(train_raw)} train + {len(test_raw)} test points")

        # Scale training data using sliding windows
        train_scaled = self._scale_with_windows(train_raw)

        # Scale test data using the same scaler (fitted on training data)
        test_scaled = self._scale_test(test_raw)

        # Apply EMA smoothing to training data only
        train_smoothed = self._apply_ema(train_scaled.flatten())

        # Build full dataset (smoothed train + raw test)
        full_data = np.concatenate([train_smoothed, test_scaled.flatten()])

        logger.info(
            f"Preprocessing complete: "
            f"train={len(train_smoothed)}, test={len(test_scaled)}, "
            f"total={len(full_data)}"
        )

        return {
            'train': train_smoothed,
            'test': test_scaled.flatten(),
            'full': full_data,
            'original': df,
        }

    def _scale_with_windows(self, data: np.ndarray) -> np.ndarray:
        """
        Scale data using sliding windows.

        This approach handles long time series better than fitting on
        the entire dataset at once. Each window is scaled independently
        to capture local patterns.

        Args:
            data: Raw data array of shape (n, 1)

        Returns:
            Scaled data array of same shape
        """
        # Use smaller window if dataset is smaller than smoothing_window
        window_size = min(self.smoothing_window, len(data))

        for start in range(0, len(data), window_size):
            end = start + window_size
            window = data[start:end]

            if len(window) == 0:
                continue

            # Fit scaler on current window, then transform it
            self.scaler.fit(window)
            data[start:end] = self.scaler.transform(window)

        # Handle remaining data if any
        last_window_start = (len(data) // window_size) * window_size
        if last_window_start < len(data):
            self.scaler.fit(data[last_window_start:])
            data[last_window_start:] = self.scaler.transform(data[last_window_start:])

        return data

    def _scale_test(self, data: np.ndarray) -> np.ndarray:
        """
        Scale test data using the already-fitted scaler.
        Test data must be transformed with the same scaler as training data.

        Args:
            data: Raw test data array of shape (n, 1)

        Returns:
            Scaled test data array
        """
        if len(data) > 0:
            return self.scaler.transform(data)
        return np.array([])

    def _apply_ema(self, data: np.ndarray) -> np.ndarray:
        """
        Apply Exponential Moving Average smoothing.

        EMA formula: EMA_t = gamma * x_t + (1 - gamma) * EMA_{t-1}

        This reduces noise in the training data, helping the LSTM
        learn the underlying trend rather than daily fluctuations.

        Args:
            data: 1D array of scaled values

        Returns:
            Smoothed 1D array of same length
        """
        ema = 0.0
        result = data.copy()

        for i in range(len(result)):
            ema = self.ema_gamma * result[i] + (1 - self.ema_gamma) * ema
            result[i] = ema

        return result