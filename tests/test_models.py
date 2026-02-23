"""Smoke tests for model implementations."""

import numpy as np
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
