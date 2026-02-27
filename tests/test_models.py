"""Smoke tests for model implementations."""

import numpy as np
import pandas as pd
import pytest


class TestMarkovSwitchingVAR:
    """Smoke tests for MS-VAR model."""

    @pytest.fixture
    def msvar_data(self):
        """Generate simple VAR-like data."""
        np.random.seed(42)
        n = 100
        k = 3  # 3 variables

        # Generate differenced data (stationary)
        data = np.random.randn(n, k) * 10
        return data

    def test_msvar_import(self):
        """MS-VAR should be importable."""
        try:
            from reserves_project.models.ms_switching_var import MarkovSwitchingVAR
            assert MarkovSwitchingVAR is not None
        except ImportError:
            pytest.skip("MS-VAR model not available")

    def test_msvar_fit(self, msvar_data):
        """MS-VAR should fit without error."""
        try:
            from reserves_project.models.ms_switching_var import MarkovSwitchingVAR
        except ImportError:
            pytest.skip("MS-VAR model not available")

        model = MarkovSwitchingVAR(n_regimes=2, ar_order=1)
        model.fit(msvar_data)

        # Check transition matrix exists (attribute name: transition_)
        assert model.transition_ is not None
        assert model.transition_.shape == (2, 2)

    def test_msvar_forecast(self, msvar_data):
        """MS-VAR should produce forecasts."""
        try:
            from reserves_project.models.ms_switching_var import MarkovSwitchingVAR
        except ImportError:
            pytest.skip("MS-VAR model not available")

        model = MarkovSwitchingVAR(n_regimes=2, ar_order=1)
        model.fit(msvar_data)

        horizon = 12
        # MS-VAR forecast requires y_history
        forecast = model.forecast(y_history=msvar_data, steps=horizon)

        assert forecast.shape == (horizon, msvar_data.shape[1])
        assert not np.any(np.isnan(forecast))

    def test_msvar_transition_matrix_valid(self, msvar_data):
        """Transition matrix rows should sum to 1."""
        try:
            from reserves_project.models.ms_switching_var import MarkovSwitchingVAR
        except ImportError:
            pytest.skip("MS-VAR model not available")

        model = MarkovSwitchingVAR(n_regimes=2, ar_order=1)
        model.fit(msvar_data)

        row_sums = model.transition_.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, [1.0, 1.0])

    def test_msvar_fit_diagnostics_present(self, msvar_data):
        """Fit diagnostics should be populated after estimation."""
        try:
            from reserves_project.models.ms_switching_var import MarkovSwitchingVAR
        except ImportError:
            pytest.skip("MS-VAR model not available")

        model = MarkovSwitchingVAR(n_regimes=2, ar_order=1, max_iter=30, tol=1e-6)
        model.fit(msvar_data)

        assert isinstance(model.loglik_path_, list)
        assert len(model.loglik_path_) >= 1
        assert model.n_iter_ == len(model.loglik_path_)
        assert isinstance(model.converged_, bool)
        assert np.isfinite(model.loglik_path_).all()
        assert model.loglik_ == pytest.approx(model.loglik_path_[-1])

        # EM should not materially degrade likelihood from start to finish.
        assert model.loglik_path_[-1] >= model.loglik_path_[0] - 1e-4

    def test_msvar_expected_durations_positive(self, msvar_data):
        """Expected regime durations should be positive."""
        try:
            from reserves_project.models.ms_switching_var import MarkovSwitchingVAR
        except ImportError:
            pytest.skip("MS-VAR model not available")

        model = MarkovSwitchingVAR(n_regimes=2, ar_order=1)
        model.fit(msvar_data)
        durations = model.expected_durations()

        assert durations.shape == (2,)
        assert np.all(durations > 0)
        assert np.all(np.isfinite(durations))

    def test_msvar_classification_certainty_bounds(self, msvar_data):
        """Classification certainty summary should be well-formed and bounded."""
        try:
            from reserves_project.models.ms_switching_var import MarkovSwitchingVAR
        except ImportError:
            pytest.skip("MS-VAR model not available")

        model = MarkovSwitchingVAR(n_regimes=2, ar_order=1)
        model.fit(msvar_data)
        certainty = model.classification_certainty()

        assert certainty["n_obs"] > 0
        assert 0.0 <= certainty["mean_max_probability"] <= 1.0
        assert 0.0 <= certainty["median_max_probability"] <= 1.0
        assert 0.0 <= certainty["share_max_prob_ge_0_8"] <= 1.0
        assert "regime_assignment_counts" in certainty
        assert "regime_assignment_shares" in certainty


class TestBVAR:
    """Smoke tests for Bayesian VAR model."""

    @pytest.fixture
    def bvar_data(self):
        """Generate VAR data for BVAR."""
        np.random.seed(42)
        n = 100
        k = 3
        data = np.random.randn(n, k) * 10
        return data

    def test_bvar_import(self):
        """BVAR should be importable."""
        try:
            from reserves_project.models.bvar import BayesianVAR
            assert BayesianVAR is not None
        except ImportError:
            pytest.skip("BVAR model not available")

    def test_bvar_fit(self, bvar_data):
        """BVAR should fit without error."""
        try:
            from reserves_project.models.bvar import BayesianVAR
        except ImportError:
            pytest.skip("BVAR model not available")

        # BVAR uses n_lags parameter
        model = BayesianVAR(n_lags=2, n_draws=100, n_burn=50)
        model.fit(bvar_data)

        # BVAR uses coef_mean (no underscore)
        assert model.coef_mean is not None

    def test_bvar_forecast(self, bvar_data):
        """BVAR should produce forecasts."""
        try:
            from reserves_project.models.bvar import BayesianVAR
        except ImportError:
            pytest.skip("BVAR model not available")

        model = BayesianVAR(n_lags=2, n_draws=100, n_burn=50)
        model.fit(bvar_data)

        horizon = 6
        # forecast_point returns array directly
        forecast = model.forecast_point(h=horizon)

        assert forecast.shape[0] == horizon
        assert not np.any(np.isnan(forecast))


class TestMLModels:
    """Smoke tests for ML models (optional)."""

    def test_xgboost_available(self):
        """Check if XGBoost is available."""
        try:
            import xgboost
            assert xgboost is not None
        except ImportError:
            pytest.skip("XGBoost not installed")

    def test_tensorflow_available(self):
        """Check if TensorFlow is available."""
        try:
            import tensorflow
            assert tensorflow is not None
        except ImportError:
            pytest.skip("TensorFlow not installed")


class TestLegacyHistoryInitialization:
    """Regression tests for legacy wrapper history seeding."""

    def test_regime_var_wrapper_uses_train_end_history(self, monkeypatch):
        from reserves_project.forecasting_models import regime_var_model as wrapper

        captured = {}

        class DummyMSVAR:
            def __init__(self, n_regimes=2, ar_order=1):
                self.n_regimes = n_regimes
                self.ar_order = ar_order

            def fit(self, y, init_states=None):
                return self

            def forecast(self, y_history, steps, **kwargs):
                captured["history"] = y_history.copy()
                return np.ones((steps, 1))

        monkeypatch.setattr(wrapper, "MarkovSwitchingVAR", DummyMSVAR)

        dates = pd.date_range("2024-01-01", periods=6, freq="MS")
        raw_df = pd.DataFrame(
            {
                "date": dates,
                "gross_reserves_usd_m": [1.0, 2.0, 3.0, 40.0, 50.0, 60.0],
                "split": ["train", "train", "train", "validation", "validation", "test"],
            }
        )
        regime_df = pd.DataFrame({"date": dates, "regime_init_high_vol": [0, 0, 1, 1, 1, 1]})
        level_series = pd.Series([100, 101, 102, 103, 104, 105], index=dates)

        wrapper.run_regime_var_forecast(raw_df, regime_df, level_series, ar_order=1)

        assert captured["history"].shape == (1, 1)
        # Must come from end of training window (2024-03), not end-of-sample (2024-06).
        assert captured["history"][0, 0] == 3.0
        assert captured["history"][0, 0] != 60.0

    def test_ms_vecm_wrapper_uses_train_end_history(self, monkeypatch):
        from reserves_project.forecasting_models import ms_vecm_model as wrapper

        captured = {}

        class DummyMSVAR:
            def __init__(self, n_regimes=2, ar_order=1):
                self.n_regimes = n_regimes
                self.ar_order = ar_order

            def fit(self, y, exog=None, init_states=None):
                return self

            def forecast(self, y_history, steps, exog_future=None, **kwargs):
                captured["history"] = y_history.copy()
                return np.ones((steps, y_history.shape[1]))

        monkeypatch.setattr(wrapper, "MarkovSwitchingVAR", DummyMSVAR)

        dates = pd.date_range("2024-01-01", periods=6, freq="MS")
        state_df = pd.DataFrame(
            {
                "date": dates,
                "d_gross_reserves_usd_m": [1.0, 2.0, 3.0, 40.0, 50.0, 60.0],
                "d_exports_usd_m": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                "ect_lag1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "regime_init_high_vol": [0, 0, 1, 1, 1, 1],
                "split": ["train", "train", "train", "validation", "validation", "test"],
            }
        )
        level_series = pd.Series([100, 101, 102, 103, 104, 105], index=dates)

        wrapper.run_ms_vecm_forecast(state_df, level_series, ar_order=1)

        assert captured["history"].shape[0] == 1
        # First y column is d_gross_reserves_usd_m and should come from 2024-03 (train end).
        assert captured["history"][0, 0] == 3.0
        assert captured["history"][0, 0] != 60.0
