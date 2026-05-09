# stock_market_lstm/plots.py
"""
Visualization utilities for stock price analysis and model evaluation.
All functions take data as parameters — no global state.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_raw_prices(df: pd.DataFrame, figsize: tuple = (18, 9)) -> None:
    """
    Plot the mid price (average of High and Low) over the entire dataset.
    Use this to inspect your raw data before any processing.

    Args:
        df: DataFrame with columns [Date, High, Low]
        figsize: Figure size as (width, height) in inches
    """
    mid_prices = (df['Low'] + df['High']) / 2.0

    plt.figure(figsize=figsize)
    plt.plot(range(len(df)), mid_prices, color='blue', linewidth=1)

    # Show date labels on x-axis (spaced evenly)
    num_ticks = min(10, len(df))
    tick_positions = range(0, len(df), max(1, len(df) // num_ticks))
    tick_labels = df['Date'].iloc[tick_positions]

    plt.xticks(tick_positions, tick_labels, rotation=45)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Mid Price', fontsize=18)
    plt.title('Stock Mid Price Over Time', fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_predictions_vs_true(
        true_values: np.ndarray,
        predictions: np.ndarray,
        offset: int = 0,
        title: str = 'Predictions vs True Values',
        true_label: str = 'True',
        pred_label: str = 'Prediction',
        figsize: tuple = (18, 9)
) -> None:
    """
    Compare predicted values against actual values on the same plot.

    Args:
        true_values: Array of actual values (full dataset)
        predictions: Array of predicted values
        offset: Starting index for plotting predictions (align with true data)
        title: Plot title
        true_label: Legend label for true values
        pred_label: Legend label for predictions
        figsize: Figure size (width, height) in inches
    """
    plt.figure(figsize=figsize)

    # Plot full true series as blue line
    plt.plot(
        range(len(true_values)),
        true_values,
        color='blue',
        label=true_label,
        linewidth=1
    )

    # Plot predictions as orange line, starting from offset
    plt.plot(
        range(offset, offset + len(predictions)),
        predictions,
        color='orange',
        label=pred_label,
        linewidth=1.5
    )

    plt.xlabel('Time', fontsize=18)
    plt.ylabel('Price (scaled)', fontsize=18)
    plt.title(title, fontsize=20)
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_training_history(
        history: dict,
        figsize: tuple = (10, 6)
) -> None:
    """
    Plot training loss over epochs to diagnose learning progress.

    A good training curve should:
        - Decrease rapidly at first (model is learning)
        - Gradually flatten out (converging)
        - Not increase (sign of instability)

    Args:
        history: Training history object from model.fit().
                 Use history.history dict directly.
        figsize: Figure size (width, height) in inches
    """
    plt.figure(figsize=figsize)

    plt.plot(history.get('loss', []), label='Training Loss', color='blue')

    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss', color='orange')

    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss (MSE)', fontsize=14)
    plt.title('Model Training History', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_forecast_evolution(
        predictions_over_time: list,
        x_axis_sequences: list,
        true_data: np.ndarray,
        best_epoch: int = None,
        figsize: tuple = (18, 18)
) -> None:
    """
    Visualize how predictions change during training.

    Top plot: All predictions from different epochs (faded red lines).
              Newer predictions are more opaque.
    Bottom plot: Predictions from the best epoch (solid red line).

    Args:
        predictions_over_time: List of lists — predictions per epoch
        x_axis_sequences: List of x-axis position lists
        true_data: Full array of actual values
        best_epoch: Index of the best epoch to highlight (default: last)
        figsize: Figure size (width, height) in inches
    """
    if best_epoch is None:
        best_epoch = len(predictions_over_time) - 1

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    # --- Top plot: Evolution of predictions ---
    ax1.plot(range(len(true_data)), true_data, color='blue', label='True Data')

    # Create alpha progression: older = more transparent
    n_epochs_to_show = len(predictions_over_time[::3])
    alphas = np.linspace(0.25, 1.0, n_epochs_to_show)

    # Plot every 3rd epoch to avoid clutter
    for i, (preds, alpha) in enumerate(zip(predictions_over_time[::3], alphas)):
        for x_vals, y_vals in zip(x_axis_sequences, preds):
            ax1.plot(x_vals, y_vals, color='red', alpha=alpha, linewidth=0.8)

    ax1.set_title('Evolution of Test Predictions Over Training', fontsize=18)
    ax1.set_xlabel('Time Index', fontsize=14)
    ax1.set_ylabel('Price (scaled)', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # --- Bottom plot: Best epoch ---
    ax2.plot(range(len(true_data)), true_data, color='blue', label='True Data')

    for x_vals, y_vals in zip(x_axis_sequences, predictions_over_time[best_epoch]):
        ax2.plot(x_vals, y_vals, color='red', linewidth=1, label='Best Predictions')

    ax2.set_title(f'Best Test Predictions (Epoch {best_epoch})', fontsize=18)
    ax2.set_xlabel('Time Index', fontsize=14)
    ax2.set_ylabel('Price (scaled)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_baseline_comparison(
        true_values: np.ndarray,
        sma_predictions: np.ndarray = None,
        ema_predictions: np.ndarray = None,
        lstm_predictions: np.ndarray = None,
        offset: int = 0,
        figsize: tuple = (18, 9)
) -> None:
    """
    Compare multiple prediction methods on the same plot.
    Useful for evaluating if LSTM outperforms simple baselines.

    Args:
        true_values: Actual values
        sma_predictions: Simple Moving Average predictions (optional)
        ema_predictions: Exponential Moving Average predictions (optional)
        lstm_predictions: LSTM model predictions (optional)
        offset: Starting index for predictions
        figsize: Figure size (width, height) in inches
    """
    plt.figure(figsize=figsize)

    plt.plot(range(len(true_values)), true_values, color='blue', label='True', linewidth=1)

    colors = {'SMA': 'green', 'EMA': 'purple', 'LSTM': 'red'}

    for name, preds in [('SMA', sma_predictions), ('EMA', ema_predictions), ('LSTM', lstm_predictions)]:
        if preds is not None:
            plt.plot(
                range(offset, offset + len(preds)),
                preds,
                color=colors.get(name, 'gray'),
                label=name,
                linewidth=1.5,
                alpha=0.8
            )

    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Price (scaled)', fontsize=14)
    plt.title('Model Comparison: Simple Baselines vs LSTM', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()