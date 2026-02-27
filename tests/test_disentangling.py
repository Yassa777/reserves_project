"""Tests for model-vs-information disentangling analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from reserves_project.eval.disentangling import (
    build_2x2_aligned_panel,
    compute_rmse_matrix,
    compute_two_by_two_effects,
)
from reserves_project.pipelines import run_disentangling_analysis


def _make_synthetic_long() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=10, freq="MS")
    actual = np.linspace(100.0, 109.0, len(dates))
    model_bias = {"MS-VAR": 0.0, "XGBoost": 2.0}
    info_bias = {"parsimonious": 1.0, "bop": 4.0}

    rows = []
    for varset, ibias in info_bias.items():
        for model, mbias in model_bias.items():
            for d, a in zip(dates, actual):
                rows.append(
                    {
                        "forecast_date": d,
                        "model": model,
                        "varset": varset,
                        "actual": a,
                        "forecast": a + mbias + ibias,
                    }
                )
    return pd.DataFrame(rows)


def _write_unified_files(base_dir: Path):
    long = _make_synthetic_long()
    base_dir.mkdir(parents=True, exist_ok=True)

    for varset in ["parsimonious", "bop"]:
        subset = long[long["varset"] == varset].copy()
        subset = subset.rename(columns={"forecast_date": "forecast_date"})
        subset["forecast_origin"] = subset["forecast_date"] - pd.DateOffset(months=1)
        subset["horizon"] = 1
        subset["split"] = "test"
        subset["std"] = np.nan
        subset["lower_80"] = np.nan
        subset["upper_80"] = np.nan
        subset["lower_95"] = np.nan
        subset["upper_95"] = np.nan
        subset["crps"] = np.nan
        subset["log_score"] = np.nan
        cols = [
            "model",
            "forecast_origin",
            "forecast_date",
            "horizon",
            "split",
            "actual",
            "forecast",
            "std",
            "lower_80",
            "upper_80",
            "lower_95",
            "upper_95",
            "crps",
            "log_score",
        ]
        subset[cols].to_csv(base_dir / f"rolling_origin_forecasts_{varset}.csv", index=False)


def test_two_by_two_effects_zero_interaction_in_additive_case():
    long = _make_synthetic_long()
    aligned = build_2x2_aligned_panel(long, models=["MS-VAR", "XGBoost"], varsets=["parsimonious", "bop"])
    rmse = compute_rmse_matrix(aligned, models=["MS-VAR", "XGBoost"], varsets=["parsimonious", "bop"])
    effects = compute_two_by_two_effects(rmse, models=["MS-VAR", "XGBoost"], varsets=["parsimonious", "bop"])
    effect_map = dict(zip(effects["effect"], effects["value"]))

    assert np.isclose(effect_map["interaction_did"], 0.0)
    assert np.isclose(effect_map["architecture_effect_avg"], 2.0)
    assert np.isclose(effect_map["information_effect_avg"], 3.0)


def test_disentangling_pipeline_cli_smoke(tmp_path, monkeypatch):
    input_dir = tmp_path / "forecast_results_unified"
    output_dir = tmp_path / "disentangling_out"
    _write_unified_files(input_dir)

    monkeypatch.setattr(
        run_disentangling_analysis,
        "DATA_DIR",
        tmp_path,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "reserves-disentangling",
            f"--input-dir={input_dir}",
            f"--output-dir={output_dir}",
            "--varsets=parsimonious,bop",
            "--models=MS-VAR,XGBoost",
            "--horizon=1",
            "--split=test",
            "--min-obs=5",
            "--bootstrap-reps=100",
            "--random-seed=7",
        ],
    )
    run_disentangling_analysis.main()

    expected = [
        output_dir / "disentangling_aligned_panel.csv",
        output_dir / "disentangling_rmse_long.csv",
        output_dir / "disentangling_rmse_matrix.csv",
        output_dir / "disentangling_effects.csv",
        output_dir / "disentangling_dm_tests.csv",
        output_dir / "disentangling_summary.json",
        output_dir / "run_manifest.json",
    ]
    for path in expected:
        assert path.exists(), f"Missing output file: {path}"

    effects = pd.read_csv(output_dir / "disentangling_effects.csv")
    effect_map = dict(zip(effects["effect"], effects["value"]))
    assert np.isclose(effect_map["interaction_did"], 0.0)

    with open(output_dir / "disentangling_summary.json", "r") as f:
        summary = json.load(f)
    assert summary["n_aligned_obs"] == 10
    assert summary["models"] == ["MS-VAR", "XGBoost"]
    assert summary["varsets"] == ["parsimonious", "bop"]
