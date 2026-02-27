"""Tests for robustness table pipeline integration of mechanism synthesis artifacts."""

from __future__ import annotations

import json

import pandas as pd

import reserves_project.pipelines.generate_robustness_tables as robustness_pipeline


def _write_mechanism_artifacts(base_dir):
    base_dir.mkdir(parents=True, exist_ok=True)

    table = pd.DataFrame(
        [
            {
                "mechanism_id": "regime_duration_separation",
                "channel": "architecture",
                "mechanism_metric": "regime_duration_spread_months",
                "mechanism_value": 13.33,
                "linked_forecast_gain_metric": "architecture_gain_primary_vs_challenger",
                "linked_forecast_gain_value": 120.0,
                "expected_relation": "positive",
                "evidence_available": True,
                "supports_direction": True,
            }
        ]
    )
    detail = pd.DataFrame([{"source": "regime", "metric": "regime_mean_max_probability", "value": 0.81}])
    diagnostics = {"n_synthesis_rows": 1, "n_detail_rows": 1}

    table.to_csv(base_dir / "mechanism_synthesis_table.csv", index=False)
    detail.to_csv(base_dir / "mechanism_synthesis_detail.csv", index=False)
    with open(base_dir / "mechanism_synthesis_diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)


def test_load_mechanism_synthesis_results_prefers_configured_dir(tmp_path, monkeypatch):
    mechanism_dir = tmp_path / "mechanism_synthesis"
    _write_mechanism_artifacts(mechanism_dir)

    monkeypatch.setattr(robustness_pipeline, "MECHANISM_SYNTHESIS_DIR", mechanism_dir)
    loaded = robustness_pipeline.load_mechanism_synthesis_results()

    assert loaded["source_dir"] == mechanism_dir
    assert "table" in loaded
    assert "detail" in loaded
    assert "diagnostics" in loaded


def test_generate_robustness_main_bundles_mechanism_artifacts(tmp_path, monkeypatch):
    output_root = tmp_path / "run_bundle"
    mechanism_dir = output_root / "mechanism_synthesis"
    _write_mechanism_artifacts(mechanism_dir)

    monkeypatch.setattr(
        "sys.argv",
        [
            "reserves-tables",
            f"--output-root={output_root}",
        ],
    )

    monkeypatch.setattr(robustness_pipeline, "load_statistical_test_results", lambda: {})
    monkeypatch.setattr(robustness_pipeline, "load_combination_forecasts", lambda: {})
    monkeypatch.setattr(robustness_pipeline, "load_bvar_results", lambda: {})
    monkeypatch.setattr(robustness_pipeline, "load_favar_results", lambda: {})
    monkeypatch.setattr(robustness_pipeline, "load_dma_results", lambda: {})
    monkeypatch.setattr(robustness_pipeline, "load_split_robustness_results", lambda: None)
    monkeypatch.setattr(robustness_pipeline, "load_unified_results", lambda: ({}, {}, {}))
    monkeypatch.setattr(robustness_pipeline, "load_baseline_forecasts", lambda: {})
    monkeypatch.setattr(robustness_pipeline, "run_subsample_analysis", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(robustness_pipeline, "run_horizon_analysis", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(robustness_pipeline, "run_variable_set_analysis", lambda *_args, **_kwargs: None)

    monkeypatch.setattr(robustness_pipeline, "compile_paper_statistics", lambda **_kwargs: {})
    monkeypatch.setattr(robustness_pipeline, "generate_statistics_summary", lambda _stats: "ok")

    def _save_statistics(stats, path):
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)

    monkeypatch.setattr(robustness_pipeline, "save_statistics", _save_statistics)

    robustness_pipeline.main()

    summary_dir = output_root / "robustness" / "summary"
    tables_dir = output_root / "robustness" / "tables"

    assert (summary_dir / "mechanism_synthesis_table.csv").exists()
    assert (summary_dir / "mechanism_synthesis_detail.csv").exists()
    assert (summary_dir / "mechanism_synthesis_diagnostics.json").exists()
    assert (tables_dir / "table7_mechanism_synthesis.tex").exists()

    table_tex = (tables_dir / "table7_mechanism_synthesis.tex").read_text()
    assert "Mechanism Evidence Synthesis" in table_tex
    assert "regime\\_duration\\_separation" in table_tex


def test_generate_robustness_main_generates_split_table(tmp_path, monkeypatch):
    output_root = tmp_path / "run_bundle"
    mechanism_dir = output_root / "mechanism_synthesis"
    _write_mechanism_artifacts(mechanism_dir)

    split_df = pd.DataFrame(
        [
            {
                "split_label": "baseline",
                "train_end": "2019-12-01",
                "valid_end": "2022-12-01",
                "varset": "bop",
                "model": "MS-VAR",
                "rmse": 300.0,
                "mae": 200.0,
                "mape": 5.0,
                "rank_within_split_varset": 1.0,
            },
            {
                "split_label": "early",
                "train_end": "2018-12-01",
                "valid_end": "2021-12-01",
                "varset": "bop",
                "model": "MS-VAR",
                "rmse": 320.0,
                "mae": 220.0,
                "mape": 5.5,
                "rank_within_split_varset": 1.0,
            },
        ]
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "reserves-tables",
            f"--output-root={output_root}",
        ],
    )

    monkeypatch.setattr(robustness_pipeline, "load_statistical_test_results", lambda: {})
    monkeypatch.setattr(robustness_pipeline, "load_combination_forecasts", lambda: {})
    monkeypatch.setattr(robustness_pipeline, "load_bvar_results", lambda: {})
    monkeypatch.setattr(robustness_pipeline, "load_favar_results", lambda: {})
    monkeypatch.setattr(robustness_pipeline, "load_dma_results", lambda: {})
    monkeypatch.setattr(robustness_pipeline, "load_split_robustness_results", lambda: split_df)
    monkeypatch.setattr(robustness_pipeline, "load_unified_results", lambda: ({}, {}, {}))
    monkeypatch.setattr(robustness_pipeline, "load_baseline_forecasts", lambda: {})
    monkeypatch.setattr(robustness_pipeline, "run_subsample_analysis", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(robustness_pipeline, "run_horizon_analysis", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(robustness_pipeline, "run_variable_set_analysis", lambda *_args, **_kwargs: None)

    monkeypatch.setattr(robustness_pipeline, "compile_paper_statistics", lambda **_kwargs: {})
    monkeypatch.setattr(robustness_pipeline, "generate_statistics_summary", lambda _stats: "ok")

    def _save_statistics(stats, path):
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)

    monkeypatch.setattr(robustness_pipeline, "save_statistics", _save_statistics)

    robustness_pipeline.main()

    tables_dir = output_root / "robustness" / "tables"
    summary_dir = output_root / "robustness" / "summary"

    assert (tables_dir / "table8_split_robustness.tex").exists()
    assert (summary_dir / "split_robustness_summary.csv").exists()
