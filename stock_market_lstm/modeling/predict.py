from stock_market_lstm.features import df, train_data
import numpy as np
from loguru import logger
from stock_market_lstm.config import configure_logging

configure_logging()

N = train_data.size
window_size = min(100, max(1, N - 1))
std_avg_predictions = []
std_avg_x = []
mse_errors = []

for pred_idx in range(window_size,N):
    date = df.loc[pred_idx,'Date']
    std_avg_predictions.append(np.mean(train_data[pred_idx-window_size:pred_idx]))
    mse_errors.append((std_avg_predictions[-1]-train_data[pred_idx])**2)
    std_avg_x.append(date)

if mse_errors:
    logger.info("MSE error for standard averaging: {:.5f}", 0.5 * np.mean(mse_errors))
else:
    logger.warning("Skipping standard averaging MSE: not enough data points (N={})", N)

N = train_data.size
run_avg_predictions = []
run_avg_x = []
mse_errors = []

running_mean = 0.0
run_avg_predictions.append(running_mean)

decay = 0.5

for pred_idx in range(1,N):
    running_mean = running_mean*decay + (1.0-decay)*train_data[pred_idx-1]
    run_avg_predictions.append(running_mean)
    mse_errors.append((run_avg_predictions[-1]-train_data[pred_idx])**2)
    date = df.loc[pred_idx,'Date']
    run_avg_x.append(date)

if mse_errors:
    logger.info("MSE error for EMA averaging: {:.5f}", 0.5 * np.mean(mse_errors))
else:
    logger.warning("Skipping EMA averaging MSE: not enough data points (N={})", N)
