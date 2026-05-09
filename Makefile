# Makefile
# Convenience commands for development and usage

.PHONY: install install-dev train predict test test-cov clean notebook demo help

# Default Python interpreter
PYTHON = python
PIP = pip

install:
	@echo "Installing stock_market_lstm..."
	$(PIP) install -e .
	@echo "Done! Use 'stock-predict --help' to get started."
	@echo "Or run 'make demo' for a quick demo."

install-dev:
	@echo "Installing with development dependencies..."
	$(PIP) install -e ".[dev]"
	@echo "Dev installation complete. Tests and notebooks are available."

train:
	@echo "Training model on sample data..."
	stock-predict train examples/sample_data.csv --epochs 10 --save models/demo.h5

predict:
	@echo "Making predictions with trained model..."
	stock-predict predict examples/sample_data.csv models/demo.h5 --steps 30

demo: train predict
	@echo ""
	@echo "Demo complete! Check models/ for the trained model."

test:
	@echo "Running tests..."
	pytest tests/ -v

test-cov:
	@echo "Running tests with coverage report..."
	pytest tests/ -v --cov=stock_market_lstm --cov-report=term-missing

clean:
	@echo "Cleaning up..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name *.egg-info -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
	rm -rf dist build
	@echo "Clean!"

notebook:
	@echo "Starting Jupyter Lab..."
	jupyter lab notebooks/

help:
	@echo "Available commands:"
	@echo "  make install      - Install the package"
	@echo "  make install-dev  - Install with development tools"
	@echo "  make train        - Train a demo model"
	@echo "  make predict      - Make predictions with demo model"
	@echo "  make demo         - Run full demo (train + predict)"
	@echo "  make test         - Run all tests"
	@echo "  make test-cov     - Run tests with coverage"
	@echo "  make clean        - Remove generated files"
	@echo "  make notebook     - Launch Jupyter Lab"