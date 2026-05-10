# stock_market_lstm/cli.py
"""
Command-line interface for Stock Market LSTM Predictor.

Provides easy access to training, prediction, and evaluation
without writing any Python code.

Usage:
    stock-predict train my_data.csv --epochs 30
    stock-predict predict new_data.csv model.h5
    stock-predict evaluate data.csv model.h5
"""

import typer
from pathlib import Path
from typing import Optional
from loguru import logger

from stock_market_lstm.config import configure_logging

# Create the main app
app = typer.Typer(
    name="stock-predict",
    help="LSTM-based stock price prediction tool",
    add_completion=False
)


@app.command()
def train(
        data: Path = typer.Argument(
            ...,
            help="Path to CSV file with stock data. Required columns: Date, Open, High, Low, Close",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
        epochs: int = typer.Option(
            30,
            "--epochs", "-e",
            help="Number of training epochs. More = potentially better, but slower",
            min=1,
        ),
        seq_len: int = typer.Option(
            50,
            "--seq-len", "-s",
            help="Sequence length - how many past days to use for each prediction",
            min=5,
        ),
        save: Optional[Path] = typer.Option(
            None,
            "--save",
            help="Path to save the trained model (e.g., models/my_model.h5)",
        ),
        verbose: bool = typer.Option(
            False,
            "--verbose", "-v",
            help="Show detailed debug information",
        ),
):
    """
    Train an LSTM model on your stock data.

    Example:
        stock-predict train data/AAPL.csv --epochs 50 --save models/aapl.h5
        stock-predict train GOOGL.csv -e 100 -v
    """
    import numpy as np
    import tensorflow as tf

    configure_logging("DEBUG" if verbose else "INFO")

    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    logger.info(f"Loading data from: {data}")

    from stock_market_lstm.dataset import load_from_csv
    from stock_market_lstm.features import StockPreprocessor
    from stock_market_lstm.models import LSTMPredictor

    # Step 1: Load the data
    try:
        df = load_from_csv(str(data))
        logger.success(f"Loaded {len(df)} trading days")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise typer.Exit(code=1)

    # Step 2: Preprocess
    logger.info("Preprocessing data...")
    preprocessor = StockPreprocessor(test_size=0.2)
    processed = preprocessor.process(df)
    logger.success(
        f"Data ready: {len(processed['train'])} train + {len(processed['test'])} test"
    )

    # Step 3: Build model
    logger.info(f"Building LSTM model (seq_len={seq_len})...")
    model = LSTMPredictor(seq_len=seq_len)
    model.build()

    # Step 4: Train
    logger.info(f"Training for {epochs} epochs...")
    history = model.train(processed['train'], epochs=epochs, verbose=1)

    # Step 5: Report results
    final_loss = history.history['loss'][-1]
    logger.success(f"Training complete! Final loss: {final_loss:.6f}")

    # Step 6: Save model if requested
    if save:
        model.save(str(save))
        logger.info(f"Model saved to: {save}")


@app.command()
def predict(
        data: Path = typer.Argument(
            ...,
            help="Path to CSV file with data for prediction",
            exists=True,
        ),
        model_path: Path = typer.Argument(
            ...,
            help="Path to trained model (.h5 file)",
            exists=True,
        ),
        context: int = typer.Option(
            50,
            "--context",
            help="Number of past days to use as context for prediction",
        ),
        steps: int = typer.Option(
            50,
            "--steps",
            help="How many days to predict into the future",
        ),
        verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """
    Make predictions using a trained model.

    Example:
        stock-predict predict new_data.csv models/aapl.h5 --steps 30
    """
    configure_logging("DEBUG" if verbose else "INFO")

    from stock_market_lstm.dataset import load_from_csv
    from stock_market_lstm.features import StockPreprocessor
    from stock_market_lstm.models import LSTMPredictor

    logger.info(f"Loading data from: {data}")
    df = load_from_csv(str(data))

    # Preprocess to get the full data series
    preprocessor = StockPreprocessor()
    processed = preprocessor.process(df)

    # Load the trained model
    logger.info(f"Loading model from: {model_path}")
    model = LSTMPredictor()
    model.load(str(model_path))

    # Generate predictions
    n = len(processed['full'])
    start_index = max(context, n - steps - 1)

    logger.info(f"Predicting {steps} steps ahead...")
    predictions = model.recursive_forecast(
        processed['full'],
        start_index=start_index,
        context=context,
        steps=steps
    )

    # Display predictions
    logger.success("Predictions generated!")
    for i, pred in enumerate(predictions[:10], 1):
        logger.info(f"  Day {i}: {pred:.6f}")
    if len(predictions) > 10:
        logger.info(f"  ... and {len(predictions) - 10} more predictions")


@app.command()
def evaluate(
        data: Path = typer.Argument(
            ...,
            help="Path to CSV file with stock data",
            exists=True,
        ),
        model_path: Path = typer.Argument(
            ...,
            help="Path to trained model (.h5 file)",
            exists=True,
        ),
        verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """
    Evaluate a trained model on your data.
    Reports Mean Squared Error (MSE) as the performance metric.

    Example:
        stock-predict evaluate data/AAPL.csv models/aapl.h5
    """
    configure_logging("DEBUG" if verbose else "INFO")

    from stock_market_lstm.dataset import load_from_csv
    from stock_market_lstm.features import StockPreprocessor
    from stock_market_lstm.models import LSTMPredictor
    import numpy as np

    logger.info(f"Loading data from: {data}")
    df = load_from_csv(str(data))
    preprocessor = StockPreprocessor()
    processed = preprocessor.process(df)

    logger.info(f"Loading model from: {model_path}")
    model = LSTMPredictor()
    model.load(str(model_path))

    # Evaluate at multiple points and average
    full_data = processed['full']
    context = model.seq_len
    step = max(1, len(full_data) // 10)

    mse_scores = []
    for start in range(context, len(full_data) - 50, step):
        _, mse = model.evaluate_forecast(
            full_data, start_index=start, context=context, steps=50
        )
        mse_scores.append(mse)

    avg_mse = np.mean(mse_scores)
    logger.success(f"Average MSE over {len(mse_scores)} evaluation windows: {avg_mse:.6f}")


@app.command()
def info():
    """
    Show information about this tool and how to use it.
    """
    typer.echo("""
    Stock Market LSTM Predictor

    A machine learning tool for stock price prediction using LSTM networks.

    Quick Start:
      1. Prepare your data as CSV with columns: Date, Open, High, Low, Close
      2. Train a model:
         $ stock-predict train your_data.csv --save model.h5
      3. Make predictions:
         $ stock-predict predict new_data.csv model.h5
      4. Evaluate performance:
         $ stock-predict evaluate data.csv model.h5

    For more help on any command:
      $ stock-predict COMMAND --help

    Example dataset format:
      Date,Open,High,Low,Close
      2020-01-02,100.0,102.0,99.0,101.0
      2020-01-03,101.0,103.0,100.0,102.0
      ...
    """)


if __name__ == "__main__":
    app()