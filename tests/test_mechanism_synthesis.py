"""Tests for mechanism synthesis pipeline and join integrity."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from reserves_project.eval.mechanism_synthesis import load_disentangling_metrics
from reserves_project.pipelines import run_mechanism_synthesis


def _write_disentangling_artifacts(base):
    rmse = pd.DataFrame(
        [
            {"model": "MS-VAR", "varset": "parsimonious", "rmse": 300.0, "n": 24},
            {"model": "MS-VAR", "varset": "bop", "rmse": 260.0, "n": 24},
            {"model": "XGBoost", "varset": "parsimonious", "rmse": 520.0, "n": 24},
            {"model": "XGBoost", "varset": "bop", "rmse": 430.0, "n": 24},
        ]
    )
    effects = pd.DataFrame(
        [
            {"effect": "architecture_effect_avg", "value": 195.0, "definition": "test"},
            {"effect": "interaction_did", "value": 50.0, "definition": "test"},
        ]
    )
    summary = {
        "varsets": ["parsimonious", "bop"],
        "models": ["MS-VAR", "XGBoost"],
        "horizon": 1,
        "split": "test",
    }
    rmse.to_csv(base / "disentangling_rmse_long.csv", index=False)
    effects.to_csv(base / "disentangling_effects.csv", index=False)
    with open(base / "disentangling_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def _write_regime_artifacts(base):
    durations = pd.DataFrame(
        [
            {"regime": 0, "persistence_probability": 0.85, "expected_duration_months": 6.67},
            {"regime": 1, "persistence_probability": 0.95, "expected_duration_months": 20.0},
        ]
    )
    certainty = pd.DataFrame(
        [
            {"metric": "mean_max_probability", "value": 0.78},
            {"metric": "share_max_prob_ge_0_8", "value": 0.62},
        ]
    )
    durations.to_csv(base / "regime_durations.csv", index=False)
    certainty.to_csv(base / "regime_classification_certainty.csv", index=False)


def _write_irf_artifacts(base):
    comp = pd.DataFrame(
        [
            {
                "shock_variable": "exports_usd_m",
                "response_variable": "gross_reserves_usd_m",
                "peak_abs_response_regime0": 10.0,
                "peak_abs_response_regime1": 18.0,
                "peak_abs_delta_regime1_minus_regime0": 8.0,
                "half_life_horizon_regime0": 2.0,
                "half_life_horizon_regime1": 4.0,
                "half_life_delta_regime1_minus_regime0": 2.0,
            },
            {
                "shock_variable": "imports_usd_m",
                "response_variable": "gross_reserves_usd_m",
                "peak_abs_response_regime0": 12.0,
                "peak_abs_response_regime1": 20.0,
                "peak_abs_delta_regime1_minus_regime0": 8.0,
                "half_life_horizon_regime0": 3.0,
                "half_life_horizon_regime1": 5.0,
                "half_life_delta_regime1_minus_regime0": 2.0,
            },
        ]
    )
    diag = {"target_col": "gross_reserves_usd_m"}
    comp.to_csv(base / "msvar_irf_regime_comparison.csv", index=False)
    with open(base / "msvar_irf_diagnostics.json", "w") as f:
        json.dump(diag, f, indent=2)


def _write_information_loss_artifacts(base):
    seg_summary = pd.DataFrame(
        [
            {
                "segment": "all",
                "n_models": 2,
                "mean_rmse_improvement_pct_disagg_vs_agg": 12.0,
                "share_models_disagg_better_rmse": 1.0,
                "share_models_significant_info_loss_10pct": 0.5,
                "mean_loss_diff_agg_minus_disagg": 50.0,
                "mean_cancellation_index": 0.4,
                "mean_info_loss_potential": 0.6,
            },
            {
                "segment": "crisis",
                "n_models": 2,
                "mean_rmse_improvement_pct_disagg_vs_agg": 22.0,
                "share_models_disagg_better_rmse": 1.0,
                "share_models_significant_info_loss_10pct": 1.0,
                "mean_loss_diff_agg_minus_disagg": 120.0,
                "mean_cancellation_index": 0.25,
                "mean_info_loss_potential": 0.75,
            },
            {
                "segment": "tranquil",
                "n_models": 2,
                "mean_rmse_improvement_pct_disagg_vs_agg": 5.0,
                "share_models_disagg_better_rmse": 0.5,
                "share_models_significant_info_loss_10pct": 0.0,
                "mean_loss_diff_agg_minus_disagg": 20.0,
                "mean_cancellation_index": 0.55,
                "mean_info_loss_potential": 0.45,
            },
        ]
    )
    model_tests = pd.DataFrame(
        [
            {"model": "MS-VAR", "segment": "crisis", "n_obs": 24},
            {"model": "XGBoost", "segment": "crisis", "n_obs": 24},
        ]
    )
    diag = {"aggregated_varset": "parsimonious", "disaggregated_varset": "bop"}
    seg_summary.to_csv(base / "information_loss_segment_summary.csv", index=False)
    model_tests.to_csv(base / "information_loss_model_segment_tests.csv", index=False)
    with open(base / "information_loss_diagnostics.json", "w") as f:
        json.dump(diag, f, indent=2)


def test_load_disentangling_metrics_infers_disaggregation_gain(tmp_path):
    dis_dir = tmp_path / "disentangling"
    dis_dir.mkdir(parents=True, exist_ok=True)
    _write_disentangling_artifacts(dis_dir)

    metrics, detail = load_disentangling_metrics(
        dis_dir,
        aggregated_varset="parsimonious",
        disaggregated_varset="bop",
    )
    assert metrics["architecture_gain_primary_vs_challenger"] > 0
    assert metrics["disaggregation_gain_vs_aggregation"] > 0
    assert len(detail) >= 3


def test_mechanism_synthesis_pipeline_smoke(tmp_path, monkeypatch):
    dis_dir = tmp_path / "disentangling"
    reg_dir = tmp_path / "regime"
    irf_dir = tmp_path / "irf"
    info_dir = tmp_path / "information_loss"
    out_dir = tmp_path / "mechanism_out"
    for d in [dis_dir, reg_dir, irf_dir, info_dir]:
        d.mkdir(parents=True, exist_ok=True)

    _write_disentangling_artifacts(dis_dir)
    _write_regime_artifacts(reg_dir)
    _write_irf_artifacts(irf_dir)
    _write_information_loss_artifacts(info_dir)

    monkeypatch.setattr(
        "sys.argv",
        [
            "reserves-mechanism-synthesis",
            f"--disentangling-dir={dis_dir}",
            f"--regime-dir={reg_dir}",
            f"--irf-dir={irf_dir}",
            f"--information-loss-dir={info_dir}",
            f"--output-dir={out_dir}",
        ],
    )
    run_mechanism_synthesis.main()

    expected = [
        out_dir / "mechanism_synthesis_table.csv",
        out_dir / "mechanism_synthesis_detail.csv",
        out_dir / "mechanism_synthesis_diagnostics.json",
        out_dir / "run_manifest.json",
    ]
    for path in expected:
        assert path.exists(), f"Missing output file: {path}"

    table = pd.read_csv(out_dir / "mechanism_synthesis_table.csv")
    required_cols = {
        "mechanism_id",
        "channel",
        "mechanism_metric",
        "mechanism_value",
        "linked_forecast_gain_metric",
        "linked_forecast_gain_value",
        "evidence_available",
        "supports_direction",
    }
    assert required_cols.issubset(set(table.columns))
    assert table["mechanism_id"].nunique() == 5
    assert table["mechanism_value"].notna().all()
    assert table["linked_forecast_gain_value"].notna().all()

    with open(out_dir / "mechanism_synthesis_diagnostics.json", "r") as f:
        diag = json.load(f)
    assert diag["missing_mechanism_values"] == 0
    assert diag["missing_linked_gain_values"] == 0
