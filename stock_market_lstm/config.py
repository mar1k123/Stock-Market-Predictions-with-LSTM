import sys
import os
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


DATA_SOURCE = "alphavantage"
TICKER = "AAL"
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

def configure_logging(level: str = "INFO") -> None:
    logger.remove()
    logger.add(sys.stderr, level=level)
