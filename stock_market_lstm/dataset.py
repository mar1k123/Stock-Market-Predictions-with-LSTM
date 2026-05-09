# stock_market_lstm/dataset.py
"""
Data loading utilities for stock market data.
Supports loading from CSV files and AlphaVantage API.
"""

import pandas as pd
import datetime as dt
import urllib.request
import json
import os
from pathlib import Path
from loguru import logger


def load_from_csv(filepath: str) -> pd.DataFrame:
    """
    Load stock data from a user's CSV file.

    Expected CSV format:
        Date,Open,High,Low,Close
        2020-01-02,100.0,102.0,99.0,101.0

    Args:
        filepath: Path to the CSV file

    Returns:
        DataFrame with columns: Date, Open, High, Low, Close

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If required columns are missing

    Example:
        df = load_from_csv("my_stocks/AAPL.csv")
        df = load_from_csv("data/raw/GOOGL.csv")
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath)

    # Validate that all required columns exist
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close']
    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"CSV file is missing required columns: {missing_columns}\n"
            f"Your columns: {list(df.columns)}\n"
            f"Required columns: {required_columns}\n\n"
            f"Your CSV should look like:\n"
            f"Date,Open,High,Low,Close\n"
            f"2020-01-02,100.0,102.0,99.0,101.0"
        )

    logger.info(f"Loaded {len(df)} rows from {filepath}")
    return df


def load_from_alphavantage(
        ticker: str,
        api_key: str,
        save_to: str = None
) -> pd.DataFrame:
    """
    Download stock data from AlphaVantage API.

    Args:
        ticker: Stock symbol (e.g., "AAPL", "GOOGL", "MSFT")
        api_key: Your AlphaVantage API key (free from alphavantage.co)
        save_to: Optional path to save the downloaded data as CSV

    Returns:
        DataFrame with columns: Date, Open, High, Low, Close

    Raises:
        ValueError: If API returns an error (invalid ticker, rate limit, etc.)

    Example:
        df = load_from_alphavantage("AAPL", "YOUR_API_KEY")
        df = load_from_alphavantage("MSFT", "YOUR_KEY", save_to="data/MSFT.csv")
    """
    url_string = (
        "https://www.alphavantage.co/query"
        "?function=TIME_SERIES_DAILY"
        f"&symbol={ticker}"
        "&outputsize=compact"
        f"&apikey={api_key}"
    )

    logger.info(f"Downloading data for {ticker} from AlphaVantage...")

    with urllib.request.urlopen(url_string) as url:
        data = json.loads(url.read().decode())

    # AlphaVantage returns error messages in the JSON itself
    if "Time Series (Daily)" not in data:
        error_msg = data.get("Error Message", str(data))
        raise ValueError(
            f"AlphaVantage API error for ticker '{ticker}': {error_msg}\n"
            f"Common issues:\n"
            f"  - Invalid ticker symbol\n"
            f"  - Invalid API key\n"
            f"  - Rate limit exceeded (5 calls/min on free tier)"
        )

    # Parse the time series data
    time_series = data["Time Series (Daily)"]
    df = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close'])

    for date_str, values in time_series.items():
        date = dt.datetime.strptime(date_str, '%Y-%m-%d')
        df.loc[len(df)] = [
            date.date(),
            float(values['1. open']),
            float(values['2. high']),
            float(values['3. low']),
            float(values['4. close'])
        ]

    # Save to CSV if requested
    if save_to:
        save_path = Path(save_to)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info(f"Saved {len(df)} rows to {save_path}")

    logger.info(f"Downloaded {len(df)} days of data for {ticker}")
    return df


def load_from_csv_or_api(
        csv_path: str = None,
        ticker: str = None,
        api_key: str = None
) -> pd.DataFrame:
    """
    Smart loader: tries CSV first, falls back to API.
    This is the recommended entry point for most users.

    Args:
        csv_path: Path to local CSV file (optional)
        ticker: Stock ticker for API download (optional)
        api_key: AlphaVantage API key (optional)

    Returns:
        DataFrame with stock data

    Example:
        # From CSV
        df = load_from_csv_or_api(csv_path="data/AAPL.csv")

        # From API
        df = load_from_csv_or_api(ticker="MSFT", api_key="YOUR_KEY")

        # From CSV with API fallback
        df = load_from_csv_or_api(csv_path="data/AAPL.csv", ticker="AAPL", api_key="KEY")
    """
    # Try CSV first if provided
    if csv_path and Path(csv_path).exists():
        return load_from_csv(csv_path)

    # Fall back to API
    if ticker and api_key:
        save_path = f"data/raw/stock_market_data-{ticker}.csv" if csv_path else None
        return load_from_alphavantage(ticker, api_key, save_to=save_path)

    raise ValueError(
        "No valid data source provided.\n"
        "Options:\n"
        "  1. Provide csv_path to existing CSV file\n"
        "  2. Provide ticker and api_key to download from AlphaVantage"
    )