"""Tests for formal information-loss under aggregation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from reserves_project.eval.information_loss import (
    compute_cancellation_index,
    evaluate_information_loss_by_segment,
    load_aligned_aggregation_forecasts,
)
from reserves_project.pipelines import run_information_loss_tests


def _forecast_rows_for_varset(
    dates: pd.DatetimeIndex,
    actual: np.ndarray,
    model_to_forecast: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for model, forecast in model_to_forecast.items():
        for d, a, f in zip(dates, actual, forecast):
            rows.append(
                {
                    "model": model,
                    "forecast_origin": d - pd.DateOffset(months=1),
                    "forecast_date": d,
                    "horizon": 1,
                    "split": "test",
                    "actual": float(a),
                    "forecast": float(f),
                    "std": np.nan,
                    "lower_80": np.nan,
                    "upper_80": np.nan,
                    "lower_95": np.nan,
                    "upper_95": np.nan,
                    "crps": np.nan,
                    "log_score": np.nan,
                }
            )
    return pd.DataFrame(rows)


def test_cancellation_index_detects_offsetting_components():
    dates = pd.date_range("2024-01-01", periods=4, freq="MS")
    df = pd.DataFrame(
        {
            "exports_usd_m": [100, 100, 100, 100],
            "imports_usd_m": [100, 50, 120, 90],
            "remittances_usd_m": [0, 0, 0, 0],
            "tourism_usd_m": [0, 0, 0, 0],
        },
        index=dates,
    )
    signs = {
        "exports_usd_m": 1.0,
        "imports_usd_m": -1.0,
        "remittances_usd_m": 1.0,
        "tourism_usd_m": 1.0,
    }
    out = compute_cancellation_index(df, component_signs=signs)
    assert len(out) == 4
    # First row: exact offset => cancellation index 0, info-loss potential 1.
    assert np.isclose(out.loc[0, "cancellation_index"], 0.0)
    assert np.isclose(out.loc[0, "info_loss_potential"], 1.0)


def test_information_loss_detects_crisis_aggregation_degradation():
    rng = np.random.default_rng(7)
    dates = pd.date_range("2019-01-01", periods=72, freq="MS")
    actual = 100 + np.linspace(0, 20, len(dates))

    crisis_mask = (dates >= pd.Timestamp("2020-03-01")) & (dates <= pd.Timestamp("2023-12-01"))
    noise_dis = rng.normal(0, 1.0, len(dates))
    noise_agg = np.where(crisis_mask, rng.normal(0, 5.0, len(dates)), rng.normal(0, 1.0, len(dates)))

    aligned = pd.DataFrame(
        {
            "forecast_date": dates,
            "model": "MS-VAR",
            "actual": actual,
            "forecast_aggregated": actual + noise_agg,
            "forecast_disaggregated": actual + noise_dis,
        }
    )
    out = evaluate_information_loss_by_segment(aligned, segment_keys=["all", "crisis", "tranquil"])
    assert set(out["segment"]) == {"all", "crisis", "tranquil"}

    crisis = out[out["segment"] == "crisis"].iloc[0]
    tranquil = out[out["segment"] == "tranquil"].iloc[0]
    assert crisis["rmse_aggregated"] > crisis["rmse_disaggregated"]
    assert crisis["mean_loss_diff_agg_minus_disagg"] > tranquil["mean_loss_diff_agg_minus_disagg"]
    assert crisis["p_value_one_sided"] < 0.10


def test_information_loss_pipeline_smoke(tmp_path, monkeypatch):
    rng = np.random.default_rng(11)
    dates = pd.date_range("2019-01-01", periods=72, freq="MS")
    actual = 200 + np.linspace(0, 20, len(dates))
    crisis_mask = (dates >= pd.Timestamp("2020-03-01")) & (dates <= pd.Timestamp("2023-12-01"))

    dis_model_fc = {
        "MS-VAR": actual + rng.normal(0, 1.2, len(dates)),
        "XGBoost": actual + rng.normal(0, 1.8, len(dates)),
    }
    agg_model_fc = {
        "MS-VAR": actual + np.where(crisis_mask, rng.normal(0, 4.5, len(dates)), rng.normal(0, 1.4, len(dates))),
        "XGBoost": actual + np.where(crisis_mask, rng.normal(0, 5.0, len(dates)), rng.normal(0, 2.0, len(dates))),
    }

    input_dir = tmp_path / "forecast_results_unified"
    input_dir.mkdir(parents=True, exist_ok=True)
    _forecast_rows_for_varset(dates, actual, agg_model_fc).to_csv(
        input_dir / "rolling_origin_forecasts_parsimonious.csv", index=False
    )
    _forecast_rows_for_varset(dates, actual, dis_model_fc).to_csv(
        input_dir / "rolling_origin_forecasts_bop.csv", index=False
    )

    # Synthetic component panel for cancellation index.
    levels = pd.DataFrame(
        {
            "gross_reserves_usd_m": 5000 + np.cumsum(rng.normal(0, 5, len(dates))),
            "exports_usd_m": 100 + rng.normal(0, 10, len(dates)),
            "imports_usd_m": 100 + rng.normal(0, 10, len(dates)),
            "remittances_usd_m": 20 + rng.normal(0, 3, len(dates)),
            "tourism_usd_m": 15 + rng.normal(0, 2, len(dates)),
            "cse_net_usd_m": rng.normal(0, 2, len(dates)),
        },
        index=dates,
    )

    out_dir = tmp_path / "info_loss_out"
    monkeypatch.setattr(run_information_loss_tests, "load_varset_levels", lambda varset: levels.copy())
    monkeypatch.setattr(
        "sys.argv",
        [
            "reserves-information-loss",
            "--aggregated-varset=parsimonious",
            "--disaggregated-varset=bop",
            "--models=MS-VAR,XGBoost",
            "--horizon=1",
            "--split=test",
            "--segments=all,crisis,tranquil",
            "--min-obs-per-model=24",
            f"--input-dir={input_dir}",
            f"--output-dir={out_dir}",
        ],
    )
    run_information_loss_tests.main()

    expected = [
        out_dir / "information_loss_aligned_forecasts.csv",
        out_dir / "information_loss_model_segment_tests.csv",
        out_dir / "information_loss_cancellation_index.csv",
        out_dir / "information_loss_segment_summary.csv",
        out_dir / "information_loss_diagnostics.json",
        out_dir / "run_manifest.json",
    ]
    for path in expected:
        assert path.exists(), f"Missing output file: {path}"

    tests_df = pd.read_csv(out_dir / "information_loss_model_segment_tests.csv")
    crisis = tests_df[tests_df["segment"] == "crisis"]
    assert (crisis["rmse_aggregated"] > crisis["rmse_disaggregated"]).all()

    with open(out_dir / "information_loss_diagnostics.json", "r") as f:
        diagnostics = json.load(f)
    assert diagnostics["aggregated_varset"] == "parsimonious"
    assert diagnostics["disaggregated_varset"] == "bop"
    assert set(diagnostics["models_tested"]) == {"MS-VAR", "XGBoost"}


def test_load_aligned_aggregation_forecasts_model_intersection(tmp_path):
    dates = pd.date_range("2024-01-01", periods=6, freq="MS")
    actual = np.linspace(10, 20, len(dates))
    base = tmp_path / "unified"
    base.mkdir(parents=True, exist_ok=True)

    agg = _forecast_rows_for_varset(
        dates,
        actual,
        {"MS-VAR": actual + 1.0, "XGBoost": actual + 2.0},
    )
    dis = _forecast_rows_for_varset(
        dates,
        actual,
        {"MS-VAR": actual + 0.5, "VECM": actual + 0.8},
    )
    agg.to_csv(base / "rolling_origin_forecasts_parsimonious.csv", index=False)
    dis.to_csv(base / "rolling_origin_forecasts_bop.csv", index=False)

    aligned = load_aligned_aggregation_forecasts(
        input_dir=base,
        aggregated_varset="parsimonious",
        disaggregated_varset="bop",
        horizon=1,
        split="test",
    )
    assert set(aligned["model"].unique()) == {"MS-VAR"}
    assert len(aligned) == len(dates)
