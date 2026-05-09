# scripts/generate_sample_data.py
"""
Generate synthetic stock data for demos and testing.
Run once: python scripts/generate_sample_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_sample_data(
        output_path: str = "examples/sample_data.csv",
        n_days: int = 1000,
        seed: int = 42
):
    """
    Generate realistic-looking stock price data.

    Args:
        output_path: Where to save the CSV
        n_days: Number of trading days to generate
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)

    # Generate business days
    dates = pd.date_range('2019-01-01', periods=n_days * 2, freq='B')[:n_days]

    # Create price series with trend + noise
    base_price = 100
    trend = np.linspace(0, 50, n_days)  # Gradual upward trend
    cycles = 15 * np.sin(np.linspace(0, 8 * np.pi, n_days))  # Market cycles
    noise = np.random.randn(n_days).cumsum() * 0.5  # Random walk noise

    close = base_price + trend + cycles + noise
    close = np.maximum(close, 10)  # Price can't be negative

    # Generate OHLC data around close price
    daily_range = np.abs(np.random.randn(n_days)) * 1.5 + 0.5

    df = pd.DataFrame({
        'Date': dates.strftime('%Y-%m-%d'),
        'Open': close + np.random.randn(n_days) * 0.2,
        'High': close + daily_range,
        'Low': close - daily_range * 0.8,
        'Close': close,
    })

    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} days of sample data -> {output_path}")
    print(f"Price range: ${close.min():.2f} - ${close.max():.2f}")


if __name__ == "__main__":
    generate_sample_data()