# tests/conftest.py
"""
Shared test fixtures.
Fixtures defined here are automatically available to all test files.
"""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_dataframe():
    """
    Create a realistic synthetic stock DataFrame for testing.
    Shared across test_dataset.py and test_preprocessor.py.
    """
    dates = pd.date_range('2021-01-01', periods=300, freq='B')
    np.random.seed(42)

    base = 100
    trend = np.linspace(0, 20, len(dates))
    noise = np.random.randn(len(dates)).cumsum() * 0.4
    close = base + trend + noise

    return pd.DataFrame({
        'Date': dates.strftime('%Y-%m-%d'),
        'Open': close + np.random.randn(len(dates)) * 0.1,
        'High': close + np.abs(np.random.randn(len(dates))) * 0.3,
        'Low': close - np.abs(np.random.randn(len(dates))) * 0.3,
        'Close': close,
    })