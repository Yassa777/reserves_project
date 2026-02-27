"""Tests for temporal leakage guardrails."""

import pandas as pd
import pytest

from reserves_project.eval.leakage_checks import assert_no_future_in_history, history_debug_info
from reserves_project.pipelines import run_rolling_backtests as rolling


def test_assert_no_future_in_history_passes_on_valid_window():
    history = pd.date_range("2024-01-01", periods=3, freq="MS")
    origin = pd.Timestamp("2024-03-01")
    assert_no_future_in_history(history, origin, context="unit-test")


def test_assert_no_future_in_history_raises_on_leakage():
    history = pd.date_range("2024-01-01", periods=4, freq="MS")
    origin = pd.Timestamp("2024-03-01")
    with pytest.raises(ValueError, match="Temporal leakage detected"):
        assert_no_future_in_history(history, origin, context="unit-test")


def test_history_debug_info_reports_origin_and_history_end():
    history = pd.date_range("2024-01-01", periods=3, freq="MS")
    origin = pd.Timestamp("2024-03-01")
    debug = history_debug_info(history, origin)
    assert debug["origin_date"] == origin
    assert debug["history_end_date"] == pd.Timestamp("2024-03-01")


def test_rolling_arima_detects_unsorted_future_leakage(monkeypatch):
    class DummyResult:
        def forecast(self, steps=1, exog=None):
            return pd.Series([0.0])

    class DummySARIMAX:
        def __init__(self, y, order, exog=None, enforce_stationarity=False, enforce_invertibility=False):
            self.y = y

        def fit(self, disp=False):
            return DummyResult()

    monkeypatch.setattr(rolling, "SARIMAX", DummySARIMAX)
    monkeypatch.setattr(rolling, "_select_arima_order", lambda y, exog, d=1: (1, d, 0))

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-02-01",
                    "2024-03-01",
                    "2024-05-01",  # validation (comes before April in row order)
                    "2024-04-01",  # validation (later row has earlier origin date)
                ]
            ),
            "gross_reserves_usd_m": [100.0, 101.0, 102.0, 103.0, 104.0],
            "split": ["train", "train", "train", "validation", "validation"],
        }
    )

    with pytest.raises(ValueError, match="Temporal leakage detected"):
        rolling._rolling_arima(df, exog_vars=[], refit_interval=1)


def test_rolling_arima_optional_debug_columns(monkeypatch):
    class DummyResult:
        def forecast(self, steps=1, exog=None):
            return pd.Series([0.0])

    class DummySARIMAX:
        def __init__(self, y, order, exog=None, enforce_stationarity=False, enforce_invertibility=False):
            self.y = y

        def fit(self, disp=False):
            return DummyResult()

    monkeypatch.setattr(rolling, "SARIMAX", DummySARIMAX)
    monkeypatch.setattr(rolling, "_select_arima_order", lambda y, exog, d=1: (1, d, 0))

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]),
            "gross_reserves_usd_m": [100.0, 101.0, 102.0, 103.0],
            "split": ["train", "train", "train", "validation"],
        }
    )

    out = rolling._rolling_arima(df, exog_vars=[], refit_interval=1, include_debug_cols=True)
    assert "origin_date" in out.columns
    assert "history_end_date" in out.columns
    assert out.loc[0, "origin_date"] == pd.Timestamp("2024-04-01")
    assert out.loc[0, "history_end_date"] == pd.Timestamp("2024-03-01")
