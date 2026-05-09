# tests/test_preprocessor.py
"""
Tests for the StockPreprocessor class.
Validates data preprocessing pipeline: scaling, splitting, EMA smoothing.
"""

import pytest
import pandas as pd
import numpy as np

from stock_market_lstm.features import StockPreprocessor


class TestStockPreprocessor:
    """Test suite for StockPreprocessor class."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create a synthetic DataFrame with 500 days of stock data."""
        dates = pd.date_range('2020-01-01', periods=500, freq='B')
        np.random.seed(42)

        base = 100
        close = base + np.random.randn(len(dates)).cumsum() * 0.3

        return pd.DataFrame({
            'Date': dates.strftime('%Y-%m-%d'),
            'Open': close + np.random.randn(len(dates)) * 0.1,
            'High': close + np.abs(np.random.randn(len(dates))) * 0.3,
            'Low': close - np.abs(np.random.randn(len(dates))) * 0.3,
            'Close': close,
        })

    @pytest.fixture
    def preprocessor(self):
        """Create a StockPreprocessor with default settings."""
        return StockPreprocessor(test_size=0.2)

    def test_process_returns_expected_keys(self, preprocessor, sample_dataframe):
        """Output dict should contain all expected keys."""
        result = preprocessor.process(sample_dataframe)

        assert 'train' in result
        assert 'test' in result
        assert 'full' in result
        assert 'original' in result

    def test_train_test_split_ratio(self, preprocessor, sample_dataframe):
        """Train/test split should match the specified ratio."""
        result = preprocessor.process(sample_dataframe)

        total = len(sample_dataframe)
        expected_train = int(total * 0.8)

        # Allow ±1 due to rounding
        assert abs(len(result['train']) - expected_train) <= 1
        assert abs(len(result['test']) - (total - expected_train)) <= 1

    def test_full_equals_train_plus_test(self, preprocessor, sample_dataframe):
        """Full dataset should be concatenation of train and test."""
        result = preprocessor.process(sample_dataframe)

        assert len(result['full']) == len(result['train']) + len(result['test'])

    def test_rejects_insufficient_data(self, preprocessor):
        """Should raise ValueError if fewer than 100 data points."""
        tiny_df = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=50),
            'Open': np.random.randn(50) + 100,
            'High': np.random.randn(50) + 102,
            'Low': np.random.randn(50) + 98,
            'Close': np.random.randn(50) + 100,
        })

        with pytest.raises(ValueError, match="Not enough data"):
            preprocessor.process(tiny_df)

    def test_output_values_are_numpy_arrays(self, preprocessor, sample_dataframe):
        """Train, test, and full data should be numpy arrays."""
        result = preprocessor.process(sample_dataframe)

        assert isinstance(result['train'], np.ndarray)
        assert isinstance(result['test'], np.ndarray)
        assert isinstance(result['full'], np.ndarray)

    def test_scaled_values_are_in_range(self, preprocessor, sample_dataframe):
        """After MinMax scaling, values should be between 0 and 1 (or close)."""
        result = preprocessor.process(sample_dataframe)

        # Test data should be scaled between 0 and 1
        assert np.all(result['test'] >= -0.1)  # Small tolerance
        assert np.all(result['test'] <= 1.1)

    def test_different_test_sizes(self, sample_dataframe):
        """Preprocessor should handle different test_size values."""
        for test_size in [0.1, 0.3, 0.5]:
            preprocessor = StockPreprocessor(test_size=test_size)
            result = preprocessor.process(sample_dataframe)

            expected_train = int(len(sample_dataframe) * (1 - test_size))
            assert abs(len(result['train']) - expected_train) <= 1

    def test_sorting_is_applied(self, preprocessor):
        """Data should be sorted by date even if input is shuffled."""
        shuffled_df = pd.DataFrame({
            'Date': ['2020-03-15', '2020-01-10', '2020-02-20'],
            'Open': [101, 100, 102],
            'High': [103, 102, 104],
            'Low': [99, 98, 100],
            'Close': [102, 101, 103],
        })

        result = preprocessor.process(shuffled_df)

        # Original DataFrame should now be sorted
        sorted_dates = result['original']['Date'].tolist()
        assert sorted_dates == ['2020-01-10', '2020-02-20', '2020-03-15']