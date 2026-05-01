import numpy as np
from sklearn.preprocessing import MinMaxScaler
from stock_market_lstm.dataset import load_data

df = load_data()
df = df.sort_values('Date')

# Mid price
high_prices = df['High'].to_numpy()
low_prices = df['Low'].to_numpy()
mid_prices = (high_prices + low_prices) / 2.0

# ---- SAFETY CHECK ----
if len(mid_prices) < 100:
    raise ValueError(f"Слишком мало данных: {len(mid_prices)} строк")

# Split (без хардкода 11000)
split_idx = int(len(mid_prices) * 0.8)

train_data = mid_prices[:split_idx].reshape(-1, 1)
test_data = mid_prices[split_idx:].reshape(-1, 1)

# ---- SCALING ----
scaler = MinMaxScaler()

smoothing_window_size = min(2500, len(train_data))  # защита от пустых срезов

for di in range(0, len(train_data), smoothing_window_size):
    end = di + smoothing_window_size
    window = train_data[di:end]

    if len(window) == 0:
        continue

    scaler.fit(window)
    train_data[di:end] = scaler.transform(window)

# Final part scaling
last_part_start = (len(train_data) // smoothing_window_size) * smoothing_window_size
if last_part_start < len(train_data):
    scaler.fit(train_data[last_part_start:])
    train_data[last_part_start:] = scaler.transform(train_data[last_part_start:])

# ---- FLATTEN ----
train_data = train_data.reshape(-1)

# Scale test safely
test_data = scaler.transform(test_data).reshape(-1) if len(test_data) > 0 else np.array([])

# ---- EMA smoothing ----
EMA = 0.0
gamma = 0.1

for ti in range(len(train_data)):
    EMA = gamma * train_data[ti] + (1 - gamma) * EMA
    train_data[ti] = EMA

# Combine
all_mid_data = np.concatenate([train_data, test_data], axis=0)