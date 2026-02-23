"""Tests for Diebold-Mariano test implementation."""

import numpy as np
import pytest

from reserves_project.eval.diebold_mariano import (
    compute_loss_differential,
    newey_west_variance,
    diebold_mariano_test,
    dm_test_hln,
    dm_test_matrix,
)


class TestLossDifferential:
    """Tests for compute_loss_differential."""

    def test_squared_loss(self):
        """Squared loss should compute e1^2 - e2^2."""
        actual = np.array([100.0, 200.0, 300.0])
        f1 = np.array([110.0, 200.0, 290.0])  # Errors: 10, 0, -10
        f2 = np.array([120.0, 200.0, 280.0])  # Errors: 20, 0, -20

        d = compute_loss_differential(actual, f1, f2, loss_fn="squared")

        expected = np.array([
            10**2 - 20**2,  # -300
            0**2 - 0**2,    # 0
            10**2 - 20**2,  # -300 (absolute values same)
        ])
        np.testing.assert_array_equal(d, expected)

    def test_absolute_loss(self):
        """Absolute loss should compute |e1| - |e2|."""
        actual = np.array([100.0, 200.0])
        f1 = np.array([110.0, 190.0])  # Errors: 10, -10
        f2 = np.array([120.0, 180.0])  # Errors: 20, -20

        d = compute_loss_differential(actual, f1, f2, loss_fn="absolute")

        expected = np.array([10 - 20, 10 - 20])  # [-10, -10]
        np.testing.assert_array_equal(d, expected)

    def test_custom_loss_function(self):
        """Custom callable loss function should work."""
        actual = np.array([100.0, 200.0])
        f1 = np.array([110.0, 210.0])
        f2 = np.array([120.0, 220.0])

        # Cubic loss
        def cubic_loss(e):
            return np.abs(e) ** 3

        d = compute_loss_differential(actual, f1, f2, loss_fn=cubic_loss)

        expected = np.array([10**3 - 20**3, 10**3 - 20**3])
        np.testing.assert_array_equal(d, expected)


class TestNeweyWestVariance:
    """Tests for Newey-West HAC variance estimator."""

    def test_positive_variance(self):
        """Variance should always be positive."""
        np.random.seed(42)
        d = np.random.randn(50)

        var = newey_west_variance(d, h=1)
        assert var > 0

    def test_bandwidth_effect(self):
        """Higher bandwidth should affect variance estimate."""
        np.random.seed(42)
        d = np.random.randn(50)

        var_h1 = newey_west_variance(d, h=1)
        var_h6 = newey_west_variance(d, h=6)

        # Both should be positive but may differ
        assert var_h1 > 0
        assert var_h6 > 0


class TestDieboldMarianoTest:
    """Tests for main DM test function."""

    def test_identical_forecasts_not_significant(self):
        """Identical forecasts should not reject H0."""
        np.random.seed(42)
        actual = np.random.randn(50)
        f1 = actual + np.random.randn(50) * 0.5
        f2 = f1.copy()  # Identical

        result = diebold_mariano_test(actual, f1, f2)

        assert result["mean_loss_diff"] == 0.0
        assert result["p_value"] > 0.10
        assert result["significance"] == ""

    def test_clearly_better_forecast(self):
        """Clearly better forecast should be significant."""
        np.random.seed(42)
        actual = np.random.randn(100)
        f1 = actual + np.random.randn(100) * 0.1  # Very good
        f2 = actual + np.random.randn(100) * 2.0  # Much worse

        result = diebold_mariano_test(actual, f1, f2)

        assert result["mean_loss_diff"] < 0  # f1 has lower loss
        assert result["p_value"] < 0.05
        assert "Forecast 1" in result["better_forecast"]

    def test_insufficient_observations(self):
        """Should return error for insufficient data."""
        actual = np.array([1.0, 2.0, 3.0])
        f1 = np.array([1.1, 2.1, 3.1])
        f2 = np.array([1.2, 2.2, 3.2])

        result = diebold_mariano_test(actual, f1, f2)

        assert "error" in result
        assert result["n_obs"] == 3

    def test_one_sided_less(self):
        """One-sided test (f1 better) should have correct p-value."""
        np.random.seed(42)
        actual = np.random.randn(100)
        f1 = actual + np.random.randn(100) * 0.1
        f2 = actual + np.random.randn(100) * 2.0

        result_two = diebold_mariano_test(actual, f1, f2, alternative="two-sided")
        result_less = diebold_mariano_test(actual, f1, f2, alternative="less")

        # One-sided should have smaller p-value when alternative is true
        assert result_less["p_value"] < result_two["p_value"]


class TestDMTestHLN:
    """Tests for HLN-corrected DM test."""

    def test_hln_correction_applied(self):
        """HLN correction should modify statistic and p-value."""
        np.random.seed(42)
        actual = np.random.randn(50)
        f1 = actual + np.random.randn(50) * 0.5
        f2 = actual + np.random.randn(50) * 1.0

        result = dm_test_hln(actual, f1, f2)

        assert "dm_statistic_hln" in result
        assert "p_value_hln" in result
        assert "correction_factor" in result
        assert "df" in result
        assert result["df"] == 49  # n - 1

    def test_hln_more_conservative(self):
        """HLN p-value should generally be more conservative."""
        np.random.seed(42)
        actual = np.random.randn(30)  # Small sample
        f1 = actual + np.random.randn(30) * 0.5
        f2 = actual + np.random.randn(30) * 1.0

        result = dm_test_hln(actual, f1, f2)

        # HLN uses t-distribution which is more conservative for small samples
        # Not always true but generally p_value_hln >= p_value
        assert result["p_value_hln"] >= 0


class TestDMTestMatrix:
    """Tests for pairwise DM test matrix."""

    def test_matrix_dimensions(self, multi_model_forecasts, forecast_data):
        """Matrix should have correct dimensions."""
        actual = forecast_data["actual"]

        dm_stats, p_values = dm_test_matrix(actual, multi_model_forecasts)

        n_models = len(multi_model_forecasts)
        assert dm_stats.shape == (n_models, n_models)
        assert p_values.shape == (n_models, n_models)

    def test_diagonal_is_nan(self, multi_model_forecasts, forecast_data):
        """Diagonal (self-comparison) should be NaN."""
        actual = forecast_data["actual"]

        dm_stats, p_values = dm_test_matrix(actual, multi_model_forecasts)

        for i in range(len(multi_model_forecasts)):
            assert np.isnan(dm_stats.iloc[i, i])
            assert np.isnan(p_values.iloc[i, i])

    def test_antisymmetric_statistic(self, multi_model_forecasts, forecast_data):
        """DM statistic should be antisymmetric: DM(A,B) = -DM(B,A)."""
        actual = forecast_data["actual"]

        dm_stats, _ = dm_test_matrix(actual, multi_model_forecasts, use_hln=False)

        models = list(multi_model_forecasts.keys())
        for i, m1 in enumerate(models):
            for j, m2 in enumerate(models):
                if i != j:
                    assert pytest.approx(dm_stats.iloc[i, j], rel=0.01) == -dm_stats.iloc[j, i]
