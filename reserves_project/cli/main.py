#!/usr/bin/env python3
"""Unified CLI entry point for the Sri Lanka Reserves Forecasting project.

Usage:
    slreserves <command> [options]

Commands:
    diagnostics    Run time-series diagnostics (stationarity, breaks, cointegration)
    prep           Prepare forecasting datasets for all models
    evaluate       Run rolling-origin model evaluation
    scenarios      Run MS-VARX conditional scenario analysis
    tests          Run statistical tests (Diebold-Mariano, MCS)
    tables         Generate publication-ready robustness tables
    tune           Tune ML model hyperparameters
    dashboard      Launch Streamlit diagnostics dashboard
    replicate      Run full replication pipeline (diagnostics -> evaluate -> tests)

Examples:
    slreserves evaluate --varset parsimonious --include-ms
    slreserves scenarios --scenarios combined_adverse,combined_upside
    slreserves replicate --run-id 2026-02-23_full
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def cmd_diagnostics(args: argparse.Namespace) -> int:
    """Run time-series diagnostics pipeline."""
    from reserves_project.pipelines.run_diagnostics import run_all_diagnostics

    run_all_diagnostics(verbose=args.verbose, write_manifest=True)
    return 0


def cmd_prep(args: argparse.Namespace) -> int:
    """Prepare forecasting datasets."""
    from reserves_project.pipelines.prepare_forecasting_data import run_forecasting_prep

    run_forecasting_prep(varset=args.varset, verbose=True)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Run rolling-origin model evaluation."""
    from reserves_project.pipelines.run_unified_evaluations import main

    sys.argv = ["reserves-unified"]
    if args.varsets:
        sys.argv.append(f"--varsets={args.varsets}")
    if args.horizons:
        sys.argv.append(f"--horizons={args.horizons}")
    if args.refit_interval:
        sys.argv.append(f"--refit-interval={args.refit_interval}")
    if args.exog_mode:
        sys.argv.append(f"--exog-mode={args.exog_mode}")
    if args.exog_forecast:
        sys.argv.append(f"--exog-forecast={args.exog_forecast}")
    if args.include_ms:
        sys.argv.append("--include-ms")
    if args.include_lstm:
        sys.argv.append("--include-lstm")
    if args.include_llsv:
        sys.argv.append("--include-llsv")
    if args.include_bop:
        sys.argv.append("--include-bop")
    if args.exclude_bvar:
        sys.argv.append("--exclude-bvar")
    if args.run_id:
        sys.argv.append(f"--run-id={args.run_id}")
    main()
    return 0


def cmd_scenarios(args: argparse.Namespace) -> int:
    """Run MS-VARX conditional scenario analysis."""
    from reserves_project.pipelines.run_scenario_analysis import main

    sys.argv = ["reserves-scenarios"]
    if args.varset:
        sys.argv.append(f"--varset={args.varset}")
    if args.horizon:
        sys.argv.append(f"--horizon={args.horizon}")
    if args.scenarios:
        sys.argv.append(f"--scenarios={args.scenarios}")
    if args.regime_mode:
        sys.argv.append(f"--regime-mode={args.regime_mode}")
    if args.run_id:
        sys.argv.append(f"--run-id={args.run_id}")
    if args.no_figures:
        sys.argv.append("--no-figures")
    main()
    return 0


def cmd_tests(args: argparse.Namespace) -> int:
    """Run statistical tests (DM, MCS)."""
    from reserves_project.pipelines.run_statistical_tests import main

    sys.argv = ["reserves-stat-tests", "--use-unified"]
    if args.varset:
        sys.argv.append(f"--unified-varset={args.varset}")
    if args.horizon:
        sys.argv.append(f"--unified-horizon={args.horizon}")
    if args.split:
        sys.argv.append(f"--unified-split={args.split}")
    if args.run_id:
        sys.argv.append(f"--run-id={args.run_id}")
    main()
    return 0


def cmd_tables(args: argparse.Namespace) -> int:
    """Generate publication-ready robustness tables."""
    from reserves_project.pipelines.generate_robustness_tables import main

    sys.argv = ["reserves-tables"]
    if args.run_id:
        sys.argv.append(f"--run-id={args.run_id}")
    main()
    return 0


def cmd_tune(args: argparse.Namespace) -> int:
    """Tune ML model hyperparameters."""
    from reserves_project.pipelines.tune_ml_models import main

    sys.argv = ["reserves-tune-ml"]
    if args.varset:
        sys.argv.append(f"--varset={args.varset}")
    if args.model:
        sys.argv.append(f"--model={args.model}")
    main()
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    """Launch Streamlit diagnostics dashboard."""
    app_path = Path(__file__).parent.parent / "apps" / "reserves_diagnostics" / "main.py"
    if not app_path.exists():
        app_path = Path(__file__).parent.parent.parent / "apps" / "run_diagnostics.py"

    cmd = ["streamlit", "run", str(app_path)]
    if args.port:
        cmd.extend(["--server.port", str(args.port)])

    return subprocess.call(cmd)


def cmd_replicate(args: argparse.Namespace) -> int:
    """Run full replication pipeline."""
    run_id = args.run_id

    print("=" * 60)
    print("SRI LANKA RESERVES FORECASTING - FULL REPLICATION")
    print("=" * 60)

    # Step 1: Diagnostics
    print("\n[1/4] Running diagnostics...")
    from reserves_project.pipelines.run_diagnostics import run_all_diagnostics
    run_all_diagnostics(verbose=False, write_manifest=True)

    # Step 2: Model evaluation
    print("\n[2/4] Running model evaluation across all variable sets...")
    from reserves_project.pipelines.run_unified_evaluations import main as eval_main
    sys.argv = [
        "reserves-unified",
        "--include-ms",
        "--include-llsv",
        "--include-bop",
        f"--run-id={run_id}" if run_id else "",
    ]
    sys.argv = [a for a in sys.argv if a]  # Remove empty
    eval_main()

    # Step 3: Statistical tests
    print("\n[3/4] Running statistical tests...")
    from reserves_project.pipelines.run_statistical_tests import main as test_main
    for varset in ["parsimonious", "bop", "monetary", "pca", "full"]:
        sys.argv = [
            "reserves-stat-tests",
            "--use-unified",
            f"--unified-varset={varset}",
            "--unified-horizon=1",
            f"--run-id={run_id}" if run_id else "",
        ]
        sys.argv = [a for a in sys.argv if a]
        try:
            test_main()
        except Exception as e:
            print(f"  Warning: Tests for {varset} failed: {e}")

    # Step 4: Scenario analysis
    print("\n[4/4] Running scenario analysis...")
    from reserves_project.pipelines.run_scenario_analysis import main as scenario_main
    sys.argv = [
        "reserves-scenarios",
        "--varset=parsimonious",
        f"--run-id={run_id}" if run_id else "",
    ]
    sys.argv = [a for a in sys.argv if a]
    scenario_main()

    print("\n" + "=" * 60)
    print("REPLICATION COMPLETE")
    print("=" * 60)
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="slreserves",
        description="Sri Lanka Reserves Forecasting - Unified CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  slreserves evaluate --include-ms --varsets parsimonious
  slreserves scenarios --scenarios combined_adverse,combined_upside
  slreserves replicate --run-id 2026-02-23_full
  slreserves dashboard --port 8501
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # diagnostics
    p_diag = subparsers.add_parser("diagnostics", help="Run time-series diagnostics")
    p_diag.add_argument("-v", "--verbose", action="store_true", default=True)
    p_diag.set_defaults(func=cmd_diagnostics)

    # prep
    p_prep = subparsers.add_parser("prep", help="Prepare forecasting datasets")
    p_prep.add_argument("--varset", default="baseline")
    p_prep.add_argument("--run-id", default=None)
    p_prep.set_defaults(func=cmd_prep)

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Run rolling-origin model evaluation")
    p_eval.add_argument("--varsets", default="parsimonious,bop,monetary,pca,full")
    p_eval.add_argument("--horizons", default="1,3,6,12")
    p_eval.add_argument("--refit-interval", type=int, default=12)
    p_eval.add_argument("--exog-mode", choices=["forecast", "actual"], default="forecast")
    p_eval.add_argument("--exog-forecast", choices=["naive", "arima"], default="naive")
    p_eval.add_argument("--include-ms", action="store_true", help="Include MS-VAR model")
    p_eval.add_argument("--include-lstm", action="store_true", help="Include LSTM model")
    p_eval.add_argument("--include-llsv", action="store_true", help="Include LLSV model")
    p_eval.add_argument("--include-bop", action="store_true", help="Include BoP identity model")
    p_eval.add_argument("--exclude-bvar", action="store_true")
    p_eval.add_argument("--run-id", default=None)
    p_eval.set_defaults(func=cmd_evaluate)

    # scenarios
    p_scen = subparsers.add_parser("scenarios", help="Run MS-VARX scenario analysis")
    p_scen.add_argument("--varset", default="parsimonious")
    p_scen.add_argument("--horizon", type=int, default=12)
    p_scen.add_argument("--scenarios", default="all", help="Comma-separated scenario keys or 'all'")
    p_scen.add_argument("--regime-mode", choices=["free", "locked", "path"], default="free")
    p_scen.add_argument("--no-figures", action="store_true")
    p_scen.add_argument("--run-id", default=None)
    p_scen.set_defaults(func=cmd_scenarios)

    # tests
    p_test = subparsers.add_parser("tests", help="Run statistical tests (DM, MCS)")
    p_test.add_argument("--varset", default="parsimonious")
    p_test.add_argument("--horizon", type=int, default=1)
    p_test.add_argument("--split", choices=["train", "valid", "test"], default="test")
    p_test.add_argument("--run-id", default=None)
    p_test.set_defaults(func=cmd_tests)

    # tables
    p_tbl = subparsers.add_parser("tables", help="Generate publication-ready tables")
    p_tbl.add_argument("--run-id", default=None)
    p_tbl.set_defaults(func=cmd_tables)

    # tune
    p_tune = subparsers.add_parser("tune", help="Tune ML model hyperparameters")
    p_tune.add_argument("--varset", default="parsimonious")
    p_tune.add_argument("--model", choices=["xgb", "lstm", "all"], default="all")
    p_tune.set_defaults(func=cmd_tune)

    # dashboard
    p_dash = subparsers.add_parser("dashboard", help="Launch Streamlit diagnostics dashboard")
    p_dash.add_argument("--port", type=int, default=8501)
    p_dash.set_defaults(func=cmd_dashboard)

    # replicate
    p_repl = subparsers.add_parser("replicate", help="Run full replication pipeline")
    p_repl.add_argument("--run-id", default=None, help="Tag outputs with this run ID")
    p_repl.set_defaults(func=cmd_replicate)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
