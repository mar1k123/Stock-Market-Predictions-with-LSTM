# tests/test_model.py
"""
Tests for the LSTMPredictor class.
Validates model building, training, prediction, and save/load functionality.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

from stock_market_lstm.modeling import LSTMPredictor

class TestLSTMPredictor:
    """Test suite for LSTMPredictor class."""

    @pytest.fixture
    def sample_data(self):
        """Generate synthetic time series data for testing."""
        np.random.seed(42)
        # Simple sine wave with noise
        t = np.linspace(0, 10 * np.pi, 300)
        data = np.sin(t) + np.random.randn(len(t)) * 0.1
        return data.astype(np.float32)

    @pytest.fixture
    def model(self):
        """Create an LSTMPredictor with short sequence length for fast tests."""
        return LSTMPredictor(seq_len=20)

    def test_build_creates_model(self, model):
        """After build(), the model attribute should not be None."""
        model.build()
        assert model.model is not None

    def test_train_runs_without_error(self, model, sample_data):
        """Training should complete without raising exceptions."""
        model.build()
        history = model.train(sample_data, epochs=2, batch_size=32, verbose=0)

        assert history is not None
        assert 'loss' in history.history
        assert len(history.history['loss']) == 2  # 2 epochs

    def test_training_loss_decreases(self, model, sample_data):
        """Loss should generally decrease after training."""
        model.build()
        history = model.train(sample_data, epochs=5, batch_size=32, verbose=0)

        losses = history.history['loss']
        # Final loss should be lower than initial loss
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"

    def test_raises_if_train_without_build(self, model, sample_data):
        """Calling train() before build() should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="not built"):
            model.train(sample_data, epochs=1, verbose=0)

    def test_recursive_forecast_shape(self, model, sample_data):
        """Predictions should have the expected shape."""
        model.build()
        model.train(sample_data, epochs=2, verbose=0)

        predictions = model.recursive_forecast(
            sample_data,
            start_index=150,
            context=20,
            steps=30
        )

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 30
        assert predictions.dtype == np.float32

    def test_evaluate_forecast_returns_mse(self, model, sample_data):
        """evaluate_forecast should return predictions and MSE score."""
        model.build()
        model.train(sample_data, epochs=2, verbose=0)

        predictions, mse = model.evaluate_forecast(
            sample_data,
            start_index=150,
            context=20,
            steps=30
        )

        assert len(predictions) == 30
        assert isinstance(mse, float)
        assert mse >= 0  # MSE is always non-negative

    def test_save_and_load(self, model, sample_data, tmp_path):
        """Model should work identically after save/load cycle."""
        model.build()
        model.train(sample_data, epochs=2, verbose=0)

        # Generate predictions before saving
        preds_before = model.recursive_forecast(
            sample_data, start_index=150, context=20, steps=10
        )

        # Save to temporary file
        save_path = tmp_path / "test_model.h5"
        model.save(str(save_path))
        assert save_path.exists()

        # Load into a new model instance
        new_model = LSTMPredictor()
        new_model.load(str(save_path))

        # Generate predictions after loading
        preds_after = new_model.recursive_forecast(
            sample_data, start_index=150, context=20, steps=10
        )

        # Predictions should be identical
        np.testing.assert_array_almost_equal(preds_before, preds_after, decimal=5)

    def test_raises_if_save_without_model(self, model):
        """Calling save() before build() should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="No model to save"):
            model.save("dummy.h5")

    def test_raises_if_load_nonexistent_file(self, model):
        """Loading nonexistent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            model.load("nonexistent_model.h5")