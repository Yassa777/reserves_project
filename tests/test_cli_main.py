"""CLI wiring tests for main command handlers."""

from __future__ import annotations

import argparse
import sys

import reserves_project.cli.main as cli_main
import reserves_project.pipelines.generate_robustness_tables as pipeline_tables
import reserves_project.pipelines.run_disentangling_analysis as pipeline_disentangle
import reserves_project.pipelines.run_unified_evaluations as pipeline_eval
import reserves_project.pipelines.run_statistical_tests as pipeline_tests
from reserves_project.robustness.paper_statistics import generate_statistics_summary


def test_cmd_tables_forwards_run_id_and_output_root(monkeypatch):
    captured = {}

    def _fake_main():
        captured["argv"] = list(sys.argv)

    monkeypatch.setattr(pipeline_tables, "main", _fake_main)

    args = argparse.Namespace(run_id="demo_run", output_root="/tmp/custom_out")
    ret = cli_main.cmd_tables(args)

    assert ret == 0
    assert captured["argv"][0] == "reserves-tables"
    assert "--run-id=demo_run" in captured["argv"]
    assert "--output-root=/tmp/custom_out" in captured["argv"]


def test_cmd_disentangle_forwards_output_root(monkeypatch):
    captured = {}

    def _fake_main():
        captured["argv"] = list(sys.argv)

    monkeypatch.setattr(pipeline_disentangle, "main", _fake_main)

    args = argparse.Namespace(
        varsets="parsimonious,bop",
        models="MS-VAR,XGBoost",
        horizon=1,
        split="test",
        input_dir="data/forecast_results_unified",
        output_dir="data/disentangling",
        min_obs=12,
        bootstrap_reps=1000,
        random_seed=42,
        skip_dm=False,
        run_id="demo_run",
        output_root="/tmp/custom_out",
    )
    ret = cli_main.cmd_disentangle(args)

    assert ret == 0
    assert "--output-root=/tmp/custom_out" in captured["argv"]
    assert "--run-id=demo_run" in captured["argv"]


def test_cmd_tests_forwards_output_root(monkeypatch):
    captured = {}

    def _fake_main():
        captured["argv"] = list(sys.argv)

    monkeypatch.setattr(pipeline_tests, "main", _fake_main)

    args = argparse.Namespace(
        varset="parsimonious",
        horizon=1,
        split="test",
        run_id="demo_run",
        output_root="/tmp/custom_out",
    )
    ret = cli_main.cmd_tests(args)

    assert ret == 0
    assert "--output-root=/tmp/custom_out" in captured["argv"]
    assert "--run-id=demo_run" in captured["argv"]


def test_cmd_evaluate_forwards_split_and_dma_args(monkeypatch):
    captured = {}

    def _fake_main():
        captured["argv"] = list(sys.argv)

    monkeypatch.setattr(pipeline_eval, "main", _fake_main)

    args = argparse.Namespace(
        varsets="parsimonious,bop",
        horizons="1,3",
        refit_interval=12,
        exog_mode="forecast",
        exog_forecast="naive",
        window_mode="full",
        segments="all,crisis",
        train_end="2019-12-01",
        valid_end="2022-12-01",
        split_specs="baseline:2019-12-01:2022-12-01,early:2018-12-01:2021-12-01",
        include_ms=True,
        include_lstm=False,
        include_llsv=True,
        include_bop=True,
        include_dma=True,
        dma_alpha=0.99,
        dma_warmup_periods=12,
        dma_variance_window=24,
        dma_min_model_obs=24,
        dma_model_pool="Naive,ARIMA,MS-VAR",
        exclude_bvar=False,
        run_id="demo_run",
        output_root="/tmp/custom_out",
    )
    ret = cli_main.cmd_evaluate(args)

    assert ret == 0
    argv = captured["argv"]
    assert "--split-specs=baseline:2019-12-01:2022-12-01,early:2018-12-01:2021-12-01" in argv
    assert "--include-dma" in argv
    assert "--dma-model-pool=Naive,ARIMA,MS-VAR" in argv
    assert "--output-root=/tmp/custom_out" in argv


def test_generate_statistics_summary_handles_missing_numeric_values():
    stats = {
        "sample": {},
        "accuracy": {},
        "statistical_tests": {"mcs_members": []},
        "robustness": {"horizon": {}},
    }

    summary = generate_statistics_summary(stats)
    assert "Best RMSE: N/A" in summary
    assert "Improvement vs Naive: N/A%" in summary
    assert "Deterioration ratio (h12/h1): N/A" in summary
