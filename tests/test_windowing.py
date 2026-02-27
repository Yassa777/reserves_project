"""Tests for common-window date filtering utilities."""

import pandas as pd

from reserves_project.eval.windowing import compute_common_dates_by_group, filter_to_common_window


def _sample_results() -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=5, freq="MS")
    rows = []

    # Group 1: split=test, horizon=1
    for date in dates[:3]:  # A: Jan, Feb, Mar
        rows.append({"model": "A", "split": "test", "horizon": 1, "forecast_date": date, "actual": 1.0, "forecast": 1.0})
    for date in dates[1:4]:  # B: Feb, Mar, Apr
        rows.append({"model": "B", "split": "test", "horizon": 1, "forecast_date": date, "actual": 1.0, "forecast": 1.0})
    for date in dates[2:5]:  # C: Mar, Apr, May
        rows.append({"model": "C", "split": "test", "horizon": 1, "forecast_date": date, "actual": 1.0, "forecast": 1.0})

    # Group 2: split=test, horizon=3 (A/B overlap on Jan-Feb)
    for date in dates[:2]:
        rows.append({"model": "A", "split": "test", "horizon": 3, "forecast_date": date, "actual": 1.0, "forecast": 1.0})
        rows.append({"model": "B", "split": "test", "horizon": 3, "forecast_date": date, "actual": 1.0, "forecast": 1.0})

    return pd.DataFrame(rows)


def test_compute_common_dates_by_group():
    results = _sample_results()
    common = compute_common_dates_by_group(results)

    key_h1 = ("test", 1)
    key_h3 = ("test", 3)

    assert key_h1 in common
    assert key_h3 in common
    assert list(common[key_h1]) == [pd.Timestamp("2023-03-01")]
    assert list(common[key_h3]) == [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-02-01")]


def test_filter_to_common_window_keeps_only_intersection_dates():
    results = _sample_results()
    filtered = filter_to_common_window(results)

    h1 = filtered[filtered["horizon"] == 1]
    assert set(h1["forecast_date"]) == {pd.Timestamp("2023-03-01")}

    h3 = filtered[filtered["horizon"] == 3]
    assert set(h3["forecast_date"]) == {pd.Timestamp("2023-01-01"), pd.Timestamp("2023-02-01")}

    # No group/model should introduce new dates or reorder model date progression.
    for (split, horizon, model), subset in filtered.groupby(["split", "horizon", "model"]):
        original = results[
            (results["split"] == split)
            & (results["horizon"] == horizon)
            & (results["model"] == model)
        ]
        assert set(subset["forecast_date"]).issubset(set(original["forecast_date"]))
        assert list(subset["forecast_date"]) == sorted(subset["forecast_date"].tolist())
