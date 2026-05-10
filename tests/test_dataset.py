# tests/test_dataset.py
"""
Tests for data loading functionality.
Ensures CSV loading works correctly and handles edge cases gracefully.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from stock_market_lstm.dataset import load_from_csv


class TestLoadFromCSV:
    """Test suite for load_from_csv function."""

    def test_loads_valid_csv(self, sample_csv_file):
        """Should successfully load a properly formatted CSV file."""
        df = load_from_csv(sample_csv_file)

        assert df is not None
        assert len(df) > 0
        assert list(df.columns) == ['Date', 'Open', 'High', 'Low', 'Close']

    def test_raises_on_missing_file(self):
        """Should raise FileNotFoundError if file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_from_csv("nonexistent_file.csv")

    def test_raises_on_missing_columns(self, tmp_path):
        """Should raise ValueError if required columns are missing."""
        # Create a CSV with wrong columns
        bad_csv = tmp_path / "bad_data.csv"
        df = pd.DataFrame({
            'Date': ['2020-01-01'],
            'Price': [100.0],  # Wrong column name
        })
        df.to_csv(bad_csv, index=False)

        with pytest.raises(ValueError, match="missing required columns"):
            load_from_csv(str(bad_csv))

    def test_handles_extra_columns(self, tmp_path):
        """Should work even if CSV has extra columns beyond required ones."""
        csv_path = tmp_path / "extra_columns.csv"
        df = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=100),
            'Open': np.random.randn(100).cumsum() + 100,
            'High': np.random.randn(100).cumsum() + 102,
            'Low': np.random.randn(100).cumsum() + 98,
            'Close': np.random.randn(100).cumsum() + 100,
            'Volume': np.random.randint(1000, 10000, 100),  # Extra column
            'Adj Close': np.random.randn(100).cumsum() + 100,  # Extra column
        })
        df.to_csv(csv_path, index=False)

        # Should not raise — extra columns are fine
        result = load_from_csv(str(csv_path))
        assert len(result) == 100


# Fixtures shared across test files
@pytest.fixture
def sample_csv_file(tmp_path):
    """
    Create a temporary CSV file with valid stock data for testing.
    This fixture is reused across multiple test files.
    """
    dates = pd.date_range('2020-01-01', periods=200, freq='B')  # Business days
    np.random.seed(42)

    base_price = 100
    close = base_price + np.random.randn(len(dates)).cumsum() * 0.5

    df = pd.DataFrame({
        'Date': dates.strftime('%Y-%m-%d'),
        'Open': close + np.random.randn(len(dates)) * 0.2,
        'High': close + np.abs(np.random.randn(len(dates))) * 0.5,
        'Low': close - np.abs(np.random.randn(len(dates))) * 0.5,
        'Close': close,
    })

    csv_path = tmp_path / "sample_stock_data.csv"
    df.to_csv(csv_path, index=False)

    return str(csv_path)