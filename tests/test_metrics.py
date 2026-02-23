"""Tests for evaluation metrics."""

import numpy as np
import pytest

from reserves_project.eval.metrics import (
    compute_metrics,
    naive_mae_scale,
    asymmetric_loss,
)


class TestComputeMetrics:
    """Tests for compute_metrics function."""

    def test_perfect_forecast(self):
        """Perfect forecast should have zero error."""
        actual = np.array([100, 200, 300, 400, 500])
        forecast = actual.copy()

        metrics = compute_metrics(actual, forecast)

        assert metrics["mae"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["mape"] == 0.0
        assert metrics["smape"] == 0.0

    def test_constant_error(self):
        """Constant error should be predictable."""
        actual = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        forecast = actual + 10.0  # Constant over-prediction

        metrics = compute_metrics(actual, forecast)

        assert metrics["mae"] == 10.0
        assert metrics["rmse"] == 10.0
        # MAPE = mean(|error / actual|) * 100 = mean([0.1, 0.05, 0.033, 0.025, 0.02]) * 100
        expected_mape = 100 * np.mean(10 / actual)
        assert pytest.approx(metrics["mape"], rel=0.01) == expected_mape

    def test_rmse_penalizes_large_errors(self):
        """RMSE should be higher than MAE when errors vary."""
        actual = np.array([100.0, 100.0, 100.0, 100.0])
        forecast = np.array([100.0, 100.0, 100.0, 140.0])  # One large error

        metrics = compute_metrics(actual, forecast)

        assert metrics["mae"] == 10.0  # (0 + 0 + 0 + 40) / 4
        assert metrics["rmse"] == 20.0  # sqrt((0 + 0 + 0 + 1600) / 4)
        assert metrics["rmse"] > metrics["mae"]

    def test_handles_nan_values(self):
        """Should handle NaN values gracefully."""
        actual = np.array([100.0, np.nan, 300.0, 400.0])
        forecast = np.array([110.0, 200.0, np.nan, 410.0])

        metrics = compute_metrics(actual, forecast)

        # Should compute on valid pairs only (indices 0 and 3)
        expected_mae = (10 + 10) / 2
        assert metrics["mae"] == expected_mae

    def test_all_nan_returns_nan(self):
        """All NaN input should return NaN metrics."""
        actual = np.array([np.nan, np.nan])
        forecast = np.array([np.nan, np.nan])

        metrics = compute_metrics(actual, forecast)

        assert np.isnan(metrics["mae"])
        assert np.isnan(metrics["rmse"])

    def test_mase_with_scale(self):
        """MASE should be computed when scale is provided."""
        actual = np.array([100.0, 200.0, 300.0, 400.0])
        forecast = np.array([110.0, 210.0, 310.0, 410.0])
        mase_scale = 50.0  # Naive MAE

        metrics = compute_metrics(actual, forecast, mase_scale=mase_scale)

        assert metrics["mase"] == 10.0 / 50.0  # MAE / scale

    def test_mase_without_scale_is_nan(self):
        """MASE should be NaN when no scale provided."""
        actual = np.array([100.0, 200.0, 300.0])
        forecast = np.array([110.0, 210.0, 310.0])

        metrics = compute_metrics(actual, forecast)

        assert np.isnan(metrics["mase"])


class TestNaiveMaeScale:
    """Tests for naive_mae_scale function."""

    def test_constant_series(self):
        """Constant series should have zero scale."""
        series = np.array([100.0, 100.0, 100.0, 100.0])
        scale = naive_mae_scale(series)
        assert scale == 0.0

    def test_linear_series(self):
        """Linear series should have constant differences."""
        series = np.array([100.0, 110.0, 120.0, 130.0])
        scale = naive_mae_scale(series)
        assert scale == 10.0

    def test_handles_nan(self):
        """Should handle NaN values."""
        series = np.array([100.0, np.nan, 120.0, 130.0])
        scale = naive_mae_scale(series)
        # After removing NaN: [100, 120, 130], diffs: [20, 10]
        assert scale == 15.0

    def test_single_value_returns_nan(self):
        """Single value should return NaN."""
        series = np.array([100.0])
        scale = naive_mae_scale(series)
        assert np.isnan(scale)


class TestAsymmetricLoss:
    """Tests for asymmetric_loss function."""

    def test_symmetric_errors(self):
        """With equal weights, should equal MAE."""
        actual = np.array([100.0, 200.0, 300.0])
        forecast = np.array([90.0, 210.0, 290.0])  # Under, over, under

        loss = asymmetric_loss(actual, forecast, under_weight=1.0, over_weight=1.0)
        expected_mae = (10 + 10 + 10) / 3

        assert loss == expected_mae

    def test_penalizes_under_prediction(self):
        """Under-prediction should be penalized more."""
        actual = np.array([100.0, 100.0])
        # One under-prediction (-10), one over-prediction (+10)
        forecast = np.array([90.0, 110.0])

        loss = asymmetric_loss(actual, forecast, under_weight=2.0, over_weight=1.0)
        # Error: forecast - actual = [-10, +10]
        # Under-prediction (error < 0): 2 * 10 = 20
        # Over-prediction (error >= 0): 1 * 10 = 10
        expected = (20 + 10) / 2

        assert loss == expected

    def test_all_under_prediction(self):
        """All under-predictions should use under_weight."""
        actual = np.array([100.0, 200.0, 300.0])
        forecast = np.array([90.0, 190.0, 290.0])

        loss = asymmetric_loss(actual, forecast, under_weight=2.0, over_weight=1.0)
        expected = 2.0 * 10.0  # All errors are 10, all under-predicted

        assert loss == expected

    def test_handles_nan(self):
        """Should handle NaN gracefully."""
        actual = np.array([100.0, np.nan, 300.0])
        forecast = np.array([90.0, 200.0, 290.0])

        loss = asymmetric_loss(actual, forecast, under_weight=2.0, over_weight=1.0)
        # Valid pairs: (100, 90) and (300, 290)
        expected = (2 * 10 + 2 * 10) / 2

        assert loss == expected
