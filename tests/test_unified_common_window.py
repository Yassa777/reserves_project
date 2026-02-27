"""Tests for common-window summaries in unified evaluator."""

import numpy as np
import pandas as pd
import pytest

from reserves_project.eval.unified_evaluator import summarize_results


def _results_frame() -> pd.DataFrame:
    dates = pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"])
    rows = []

    # Model A: Jan-Mar
    for i, date in enumerate(dates[:3], start=1):
        rows.append({
            "model": "A",
            "split": "test",
            "horizon": 1,
            "forecast_origin": date - pd.DateOffset(months=1),
            "forecast_date": date,
            "actual": 100.0 + i,
            "forecast": 100.0 + i + 1.0,
            "std": 1.0,
            "lower_80": 99.0,
            "upper_80": 103.0,
            "lower_95": 98.0,
            "upper_95": 104.0,
            "crps": np.nan,
            "log_score": np.nan,
        })

    # Model B: Feb-Apr
    for i, date in enumerate(dates[1:], start=1):
        rows.append({
            "model": "B",
            "split": "test",
            "horizon": 1,
            "forecast_origin": date - pd.DateOffset(months=1),
            "forecast_date": date,
            "actual": 100.0 + i,
            "forecast": 100.0 + i + 2.0,
            "std": 1.0,
            "lower_80": 99.0,
            "upper_80": 103.0,
            "lower_95": 98.0,
            "upper_95": 104.0,
            "crps": np.nan,
            "log_score": np.nan,
        })

    return pd.DataFrame(rows)


def test_summarize_results_common_window_equalizes_n():
    results = _results_frame()
    train_series = pd.Series(np.linspace(50.0, 100.0, 24))

    summary_full = summarize_results(results, train_series, window_mode="full")
    summary_common = summarize_results(results, train_series, window_mode="common_dates")

    full_counts = dict(zip(summary_full["model"], summary_full["n"]))
    common_counts = dict(zip(summary_common["model"], summary_common["n"]))

    assert full_counts["A"] == 3
    assert full_counts["B"] == 3
    assert common_counts["A"] == 2
    assert common_counts["B"] == 2
    assert (summary_common["n_common_dates"] == 2).all()

    starts = dict(zip(summary_common["model"], pd.to_datetime(summary_common["effective_start"])))
    ends = dict(zip(summary_common["model"], pd.to_datetime(summary_common["effective_end"])))
    assert starts["A"] == pd.Timestamp("2024-02-01")
    assert starts["B"] == pd.Timestamp("2024-02-01")
    assert ends["A"] == pd.Timestamp("2024-03-01")
    assert ends["B"] == pd.Timestamp("2024-03-01")


def test_summarize_results_invalid_window_mode_raises():
    results = _results_frame()
    train_series = pd.Series(np.linspace(50.0, 100.0, 24))
    with pytest.raises(ValueError, match="Unsupported window_mode"):
        summarize_results(results, train_series, window_mode="invalid")
