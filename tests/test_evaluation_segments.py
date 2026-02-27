"""Tests for evaluation segment configuration and segment summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reserves_project.config.evaluation_segments import normalize_segment_keys, segment_date_mask
from reserves_project.eval.unified_evaluator import summarize_results


def test_segment_date_mask_boundaries():
    dates = pd.to_datetime(
        [
            "2020-02-01",
            "2020-03-01",
            "2023-12-01",
            "2024-01-01",
        ]
    )

    crisis = segment_date_mask(dates, "crisis")
    tranquil = segment_date_mask(dates, "tranquil")
    all_mask = segment_date_mask(dates, "all")

    assert crisis.tolist() == [False, True, True, False]
    assert tranquil.tolist() == [True, False, False, True]
    assert all_mask.tolist() == [True, True, True, True]


def test_normalize_segment_keys_rejects_unknown():
    with pytest.raises(KeyError):
        normalize_segment_keys(["all", "unknown"])


def test_summarize_results_segment_weighted_mae_identity():
    dates = pd.to_datetime(["2020-03-01", "2020-04-01", "2024-01-01", "2024-02-01"])
    actual = np.array([10.0, 20.0, 30.0, 40.0])
    forecast = np.array([12.0, 19.0, 33.0, 36.0])  # abs errors: 2,1,3,4

    rows = []
    for d, a, f in zip(dates, actual, forecast):
        rows.append(
            {
                "model": "A",
                "split": "test",
                "horizon": 1,
                "forecast_origin": d - pd.DateOffset(months=1),
                "forecast_date": d,
                "actual": a,
                "forecast": f,
                "std": np.nan,
                "lower_80": np.nan,
                "upper_80": np.nan,
                "lower_95": np.nan,
                "upper_95": np.nan,
                "crps": np.nan,
                "log_score": np.nan,
            }
        )
    results = pd.DataFrame(rows)

    summary = summarize_results(
        results,
        train_series=pd.Series(np.linspace(1.0, 50.0, 24)),
        window_mode="full",
        segment_keys=["all", "crisis", "tranquil"],
    )

    assert set(summary["segment"]) == {"all", "crisis", "tranquil"}

    all_row = summary[summary["segment"] == "all"].iloc[0]
    crisis_row = summary[summary["segment"] == "crisis"].iloc[0]
    tranquil_row = summary[summary["segment"] == "tranquil"].iloc[0]

    assert all_row["n"] == crisis_row["n"] + tranquil_row["n"]
    weighted_mae = (
        crisis_row["mae"] * crisis_row["n"] + tranquil_row["mae"] * tranquil_row["n"]
    ) / all_row["n"]
    assert np.isclose(all_row["mae"], weighted_mae)
