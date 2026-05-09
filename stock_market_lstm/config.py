# stock_market_lstm/config.py
"""
Configuration and utilities for stock_market_lstm.
Loads environment variables and provides logging setup.
"""

import sys
import os
from loguru import logger
from dotenv import load_dotenv

# Load .env file if it exists (API keys, etc.)
load_dotenv()


def get_api_key() -> str:
    """
    Get AlphaVantage API key from environment variables.
    User should set ALPHAVANTAGE_API_KEY in .env file.

    Returns:
        str: API key or empty string if not set
    """
    return os.getenv("ALPHAVANTAGE_API_KEY", "")


def configure_logging(level: str = "INFO") -> None:
    """
    Set up logging configuration.

    Args:
        level: Log level - "DEBUG", "INFO", "WARNING", "ERROR"
               Use "DEBUG" for verbose output during development.

    Example:
        configure_logging("DEBUG")  # Show all debug messages
        configure_logging("INFO")   # Show only info and above
    """
    logger.remove()
    logger.add(sys.stderr, level=level)

 