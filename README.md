# Stock Market Prediction with LSTM



<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>
<a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.11+-blue?logo=python" />
</a>
<a href="https://www.tensorflow.org/">
    <img src="https://img.shields.io/badge/TensorFlow-LSTM-orange?logo=tensorflow" />
</a>
<a href="https://github.com/astral-sh/ruff">
    <img src="https://img.shields.io/badge/Code%20style-ruff-purple" />
</a>
<a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green" />
</a>

You would like to model stock prices correctly so as a stock buyer, you can reasonably decide when to buy stocks and when to sell them to make a profit. This is where time series modeling comes in. You need good machine learning models that can look at the history of a sequence of data and correctly predict.

---

## Quick Start

### Installation

```bash
git clone https://github.com/mar1k123/Stock-Market-Predictions-with-LSTM.git
cd Stock-Market-Predictions-with-LSTM
pip install -e .
```

After installation, the CLI becomes available:

```bash
stock-predict --help
```

---

## Dataset Format

Prepare a CSV file with the following columns:

```csv
Date,Open,High,Low,Close
2020-01-02,300.0,302.5,298.0,301.0
2020-01-03,301.0,305.0,300.5,304.5
```

Requirements:

- Required columns: `Date`, `Open`, `High`, `Low`, `Close`
- Minimum dataset size: `100+ rows`

---

## CLI Usage

### Train a Model

```bash
stock-predict train your_data.csv --epochs 30 --save model.h5
```

### Predict Future Prices

```bash
stock-predict predict your_data.csv model.h5 --steps 30
```

### Evaluate Model Performance

```bash
stock-predict evaluate your_data.csv model.h5
```

---

## Demo (No Real Data Required)

Generate synthetic stock data:

```bash
python scripts/generate_sample_data.py
```

Run the full demo pipeline:

```bash
make demo
```

---

## Python API

### Train and Forecast

```python
from stock_market_lstm.dataset import load_from_csv
from stock_market_lstm.features import StockPreprocessor
from stock_market_lstm.modeling import LSTMPredictor
from stock_market_lstm import plots

df = load_from_csv("my_data.csv")

data = StockPreprocessor(test_size=0.2).process(df)

model = LSTMPredictor(seq_len=50).build()

model.train(
    data["train"],
    epochs=30
)

predictions = model.recursive_forecast(
    data["full"],
    start_index=len(data["full"]) - 60,
    context=50,
    steps=30
)

plots.plot_predictions_vs_true(
    data["full"],
    predictions,
    offset=len(data["full"]) - 60
)

model.save("models/my_model.h5")
```

### Load Existing Model

```python
from stock_market_lstm.dataset import load_from_csv
from stock_market_lstm.features import StockPreprocessor
from stock_market_lstm.modeling import LSTMPredictor

# Load and preprocess data first
df = load_from_csv("my_data.csv")
data = StockPreprocessor(test_size=0.2).process(df)

# Load saved model
model = LSTMPredictor().load("models/my_model.h5")

# Predict
predictions = model.recursive_forecast(
    data["full"],
    start_index=1000,
    context=50,
    steps=30
)
```

---

## Jupyter Notebook

Start notebook environment:

```bash
make notebook
```

Or open manually:

```text
notebooks/01-quick-start.ipynb
```

Update the dataset path and run all cells.

---

## How It Works

### Prediction Pipeline

```text
CSV Data
   ↓
Mid Price Calculation
   ↓
Train/Test Split (80/20)
   ↓
MinMax Scaling
   ↓
EMA Smoothing
   ↓
LSTM Training
   ↓
Predictions
```

### Model Architecture

```text
Input (50 days)
        ↓
LSTM(200)
        ↓
Dropout(0.2)
        ↓
LSTM(200)
        ↓
Dropout(0.2)
        ↓
LSTM(150)
        ↓
Dense(1)
        ↓
Predicted Price
```

---

## AlphaVantage API (Optional)

Get a free API key:

```text
https://www.alphavantage.co/support/#api-key
```

Create environment file:

```bash
cp .env.example .env
```

Add your API key:

```text
ALPHAVANTAGE_API_KEY=your_key
```

Example usage:

```python
from stock_market_lstm.dataset import load_from_alphavantage

df = load_from_alphavantage(
    "AAPL",
    api_key="YOUR_KEY"
)
```

---

## Development

### Run Tests

```bash
make test
```

### Run Tests with Coverage

```bash
make test-cov
```

### Clean Temporary Files

```bash
make clean
```

### Show Available Commands

```bash
make help
```

---

## Common Issues

| Problem | Solution |
|---|---|
| `Not enough data` | Add more rows (minimum 100+) |
| `Missing required columns` | CSV must contain `Date`, `Open`, `High`, `Low`, `Close` |
| Training is slow | Reduce `--epochs` or `--seq-len` |
| Predictions are flat | Increase epochs or sequence length |
| `Loss is NaN` | Remove missing values or outliers |
| Import errors | Run `pip install -e .` and mark Sources Root in PyCharm |

---

## Project Organization

```text
├├── LICENSE            <- Open-source license
├── Makefile           <- Convenience commands (`make test`, `make demo`, etc.)
├── README.md          <- This file
│
├── data
│   ├── external       <- Data from third party sources
│   ├── interim        <- Intermediate transformed data
│   ├── processed      <- Final datasets for modeling
│   └── raw            <- Original immutable data dump
│
├── examples
│   └── sample_data.csv <- Demo data for quick start
│
├── models             <- Trained .h5 model files
│
├── notebooks
│   └── 01-quick-start.ipynb <- Interactive getting-started guide
│
├── pyproject.toml     <- Package metadata and dependencies
│
├── scripts
│   └── generate_sample_data.py <- Generate demo CSV
│
├── tests              <- Unit tests
│   ├── conftest.py
│   ├── test_dataset.py
│   ├── test_preprocessor.py
│   └── test_model.py
│
└── stock_market_lstm  <- Source code
    ├── __init__.py
    ├── cli.py         <- Command-line interface (train, predict, evaluate)
    ├── config.py      <- Logging and env vars
    ├── dataset.py     <- CSV & AlphaVantage API loading
    ├── features.py    <- Preprocessing (scaling, EMA, split)
    ├── plots.py       <- Visualization functions
    │
    └── modeling
        ├── __init__.py
        ├── predict.py <- Model inference
        └── train.py   <- LSTMPredictor class
```

---

## Example Workflow

### Generate Sample Data

```bash
python scripts/generate_sample_data.py
```

### Train Model

```bash
stock-predict train examples/sample_data.csv --epochs 30 --save models/demo.h5
```

### Predict Future Prices

```bash
stock-predict predict examples/sample_data.csv models/demo.h5 --steps 30
```

### Evaluate Model

```bash
stock-predict evaluate examples/sample_data.csv models/demo.h5
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Deep Learning | TensorFlow / Keras |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib |
| Testing | Pytest |
| Packaging | setuptools |

---

## License

MIT — see `LICENSE`