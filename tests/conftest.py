"""Shared pytest fixtures for reserves_project tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_time_series():
    """Generate a simple time series for testing."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2015-01-01", periods=n, freq="MS")
    values = 5000 + np.cumsum(np.random.randn(n) * 100)
    return pd.Series(values, index=dates, name="gross_reserves_usd_m")


@pytest.fixture
def sample_panel():
    """Generate a sample panel dataset matching expected structure."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2015-01-01", periods=n, freq="MS")

    df = pd.DataFrame({
        "date": dates,
        "gross_reserves_usd_m": 5000 + np.cumsum(np.random.randn(n) * 100),
        "exports_usd_m": 1000 + np.random.randn(n) * 50,
        "imports_usd_m": 1500 + np.random.randn(n) * 75,
        "remittances_usd_m": 500 + np.random.randn(n) * 25,
        "tourism_usd_m": 200 + np.random.randn(n) * 20,
        "usd_lkr": 150 + np.cumsum(np.random.randn(n) * 2),
        "m2_usd_m": 8000 + np.cumsum(np.random.randn(n) * 50),
        "trade_balance_usd_m": -500 + np.random.randn(n) * 100,
    })
    df = df.set_index("date")
    return df


@pytest.fixture
def forecast_data():
    """Generate actual/forecast pairs for testing metrics."""
    np.random.seed(42)
    n = 50
    actual = 5000 + np.random.randn(n) * 500

    # Good forecast (low error)
    forecast_good = actual + np.random.randn(n) * 100

    # Bad forecast (high error)
    forecast_bad = actual + np.random.randn(n) * 500

    # Biased forecast (systematic under-prediction)
    forecast_biased = actual - 200 + np.random.randn(n) * 100

    return {
        "actual": actual,
        "forecast_good": forecast_good,
        "forecast_bad": forecast_bad,
        "forecast_biased": forecast_biased,
    }


@pytest.fixture
def multi_model_forecasts(forecast_data):
    """Multiple model forecasts for pairwise testing."""
    return {
        "Model_A": forecast_data["forecast_good"],
        "Model_B": forecast_data["forecast_bad"],
        "Model_C": forecast_data["forecast_biased"],
        "Naive": forecast_data["actual"] + np.random.randn(len(forecast_data["actual"])) * 300,
    }
