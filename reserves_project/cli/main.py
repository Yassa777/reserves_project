#!/usr/bin/env python3
"""Unified CLI entry point for the Sri Lanka Reserves Forecasting project.

Usage:
    slreserves <command> [options]

Commands:
    diagnostics    Run time-series diagnostics (stationarity, breaks, cointegration)
    prep           Prepare forecasting datasets for all models
    evaluate       Run rolling-origin model evaluation
    regimes        Run MS-VAR regime characterization diagnostics
    irf            Run regime-conditional MS-VAR impulse responses
    info-loss      Run formal aggregation information-loss tests
    mechanisms     Synthesize mechanism evidence into one table
    disentangle    Run 2x2 model-vs-information disentangling analysis
    scenarios      Run MS-VARX conditional scenario analysis
    tests          Run statistical tests (Diebold-Mariano, MCS)
    tables         Generate publication-ready robustness tables
    tune           Tune ML model hyperparameters
    dashboard      Launch Streamlit diagnostics dashboard
    replicate      Run full replication pipeline (diagnostics -> eval -> tests -> mechanisms -> tables)

Examples:
    slreserves evaluate --varset parsimonious --include-ms
    slreserves regimes --varset bop --fit-split train_valid
    slreserves irf --varset bop --max-horizon 24
    slreserves info-loss --aggregated-varset parsimonious --disaggregated-varset bop
    slreserves mechanisms --run-id my_run
    slreserves disentangle --varsets parsimonious,bop --models MS-VAR,XGBoost
    slreserves scenarios --scenarios combined_adverse,combined_upside
    slreserves replicate --run-id 2026-02-23_full
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from reserves_project.config.paths import DATA_DIR


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
    if args.window_mode:
        sys.argv.append(f"--window-mode={args.window_mode}")
    if args.segments:
        sys.argv.append(f"--segments={args.segments}")
    if getattr(args, "train_end", None):
        sys.argv.append(f"--train-end={args.train_end}")
    if getattr(args, "valid_end", None):
        sys.argv.append(f"--valid-end={args.valid_end}")
    if getattr(args, "split_specs", None):
        sys.argv.append(f"--split-specs={args.split_specs}")
    if args.include_ms:
        sys.argv.append("--include-ms")
    if args.include_lstm:
        sys.argv.append("--include-lstm")
    if args.include_llsv:
        sys.argv.append("--include-llsv")
    if args.include_bop:
        sys.argv.append("--include-bop")
    if getattr(args, "include_dma", False):
        sys.argv.append("--include-dma")
    if getattr(args, "dma_alpha", None) is not None:
        sys.argv.append(f"--dma-alpha={args.dma_alpha}")
    if getattr(args, "dma_warmup_periods", None) is not None:
        sys.argv.append(f"--dma-warmup-periods={args.dma_warmup_periods}")
    if getattr(args, "dma_variance_window", None) is not None:
        sys.argv.append(f"--dma-variance-window={args.dma_variance_window}")
    if getattr(args, "dma_min_model_obs", None) is not None:
        sys.argv.append(f"--dma-min-model-obs={args.dma_min_model_obs}")
    if getattr(args, "dma_model_pool", None):
        sys.argv.append(f"--dma-model-pool={args.dma_model_pool}")
    if args.exclude_bvar:
        sys.argv.append("--exclude-bvar")
    if args.run_id:
        sys.argv.append(f"--run-id={args.run_id}")
    if args.output_root:
        sys.argv.append(f"--output-root={args.output_root}")
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
    if args.output_root:
        sys.argv.append(f"--output-root={args.output_root}")
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
    if args.output_root:
        sys.argv.append(f"--output-root={args.output_root}")
    main()
    return 0


def cmd_tables(args: argparse.Namespace) -> int:
    """Generate publication-ready robustness tables."""
    from reserves_project.pipelines.generate_robustness_tables import main

    sys.argv = ["reserves-tables"]
    if args.run_id:
        sys.argv.append(f"--run-id={args.run_id}")
    if args.output_root:
        sys.argv.append(f"--output-root={args.output_root}")
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


def cmd_disentangle(args: argparse.Namespace) -> int:
    """Run disentangling analysis pipeline."""
    from reserves_project.pipelines.run_disentangling_analysis import main

    sys.argv = ["reserves-disentangling"]
    if args.varsets:
        sys.argv.append(f"--varsets={args.varsets}")
    if args.models:
        sys.argv.append(f"--models={args.models}")
    if args.horizon:
        sys.argv.append(f"--horizon={args.horizon}")
    if args.split:
        sys.argv.append(f"--split={args.split}")
    if args.input_dir:
        sys.argv.append(f"--input-dir={args.input_dir}")
    if args.output_dir:
        sys.argv.append(f"--output-dir={args.output_dir}")
    if args.min_obs:
        sys.argv.append(f"--min-obs={args.min_obs}")
    if args.bootstrap_reps is not None:
        sys.argv.append(f"--bootstrap-reps={args.bootstrap_reps}")
    if args.random_seed is not None:
        sys.argv.append(f"--random-seed={args.random_seed}")
    if args.skip_dm:
        sys.argv.append("--skip-dm")
    if args.run_id:
        sys.argv.append(f"--run-id={args.run_id}")
    if args.output_root:
        sys.argv.append(f"--output-root={args.output_root}")
    main()
    return 0


def cmd_regimes(args: argparse.Namespace) -> int:
    """Run regime characterization diagnostics."""
    from reserves_project.pipelines.run_regime_characterization import main

    sys.argv = ["reserves-regime-characterization"]
    if args.varset:
        sys.argv.append(f"--varset={args.varset}")
    if args.n_regimes is not None:
        sys.argv.append(f"--n-regimes={args.n_regimes}")
    if args.ar_order is not None:
        sys.argv.append(f"--ar-order={args.ar_order}")
    if args.fit_split:
        sys.argv.append(f"--fit-split={args.fit_split}")
    if args.train_end:
        sys.argv.append(f"--train-end={args.train_end}")
    if args.valid_end:
        sys.argv.append(f"--valid-end={args.valid_end}")
    if args.fit_start:
        sys.argv.append(f"--fit-start={args.fit_start}")
    if args.fit_end:
        sys.argv.append(f"--fit-end={args.fit_end}")
    if args.output_dir:
        sys.argv.append(f"--output-dir={args.output_dir}")
    if args.run_id:
        sys.argv.append(f"--run-id={args.run_id}")
    if args.output_root:
        sys.argv.append(f"--output-root={args.output_root}")
    main()
    return 0


def cmd_irf(args: argparse.Namespace) -> int:
    """Run regime-conditional MS-VAR IRF analysis."""
    from reserves_project.pipelines.run_msvar_irf_analysis import main

    sys.argv = ["reserves-msvar-irf"]
    if args.varset:
        sys.argv.append(f"--varset={args.varset}")
    if args.n_regimes is not None:
        sys.argv.append(f"--n-regimes={args.n_regimes}")
    if args.ar_order is not None:
        sys.argv.append(f"--ar-order={args.ar_order}")
    if args.max_horizon is not None:
        sys.argv.append(f"--max-horizon={args.max_horizon}")
    if args.fit_split:
        sys.argv.append(f"--fit-split={args.fit_split}")
    if args.train_end:
        sys.argv.append(f"--train-end={args.train_end}")
    if args.valid_end:
        sys.argv.append(f"--valid-end={args.valid_end}")
    if args.fit_start:
        sys.argv.append(f"--fit-start={args.fit_start}")
    if args.fit_end:
        sys.argv.append(f"--fit-end={args.fit_end}")
    if args.difference_irf:
        sys.argv.append("--difference-irf")
    if args.output_dir:
        sys.argv.append(f"--output-dir={args.output_dir}")
    if args.run_id:
        sys.argv.append(f"--run-id={args.run_id}")
    if args.output_root:
        sys.argv.append(f"--output-root={args.output_root}")
    main()
    return 0


def cmd_info_loss(args: argparse.Namespace) -> int:
    """Run formal information-loss tests."""
    from reserves_project.pipelines.run_information_loss_tests import main

    sys.argv = ["reserves-information-loss"]
    if args.aggregated_varset:
        sys.argv.append(f"--aggregated-varset={args.aggregated_varset}")
    if args.disaggregated_varset:
        sys.argv.append(f"--disaggregated-varset={args.disaggregated_varset}")
    if args.models:
        sys.argv.append(f"--models={args.models}")
    if args.horizon is not None:
        sys.argv.append(f"--horizon={args.horizon}")
    if args.split:
        sys.argv.append(f"--split={args.split}")
    if args.loss:
        sys.argv.append(f"--loss={args.loss}")
    if args.segments:
        sys.argv.append(f"--segments={args.segments}")
    if args.min_obs_per_model is not None:
        sys.argv.append(f"--min-obs-per-model={args.min_obs_per_model}")
    if args.component_signs:
        sys.argv.append(f"--component-signs={args.component_signs}")
    if args.input_dir:
        sys.argv.append(f"--input-dir={args.input_dir}")
    if args.output_dir:
        sys.argv.append(f"--output-dir={args.output_dir}")
    if args.run_id:
        sys.argv.append(f"--run-id={args.run_id}")
    if args.output_root:
        sys.argv.append(f"--output-root={args.output_root}")
    main()
    return 0


def cmd_mechanisms(args: argparse.Namespace) -> int:
    """Run mechanism synthesis pipeline."""
    from reserves_project.pipelines.run_mechanism_synthesis import main

    sys.argv = ["reserves-mechanism-synthesis"]
    if args.disentangling_dir:
        sys.argv.append(f"--disentangling-dir={args.disentangling_dir}")
    if args.regime_dir:
        sys.argv.append(f"--regime-dir={args.regime_dir}")
    if args.irf_dir:
        sys.argv.append(f"--irf-dir={args.irf_dir}")
    if args.information_loss_dir:
        sys.argv.append(f"--information-loss-dir={args.information_loss_dir}")
    if args.output_dir:
        sys.argv.append(f"--output-dir={args.output_dir}")
    if args.run_id:
        sys.argv.append(f"--run-id={args.run_id}")
    if args.output_root:
        sys.argv.append(f"--output-root={args.output_root}")
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
    output_root = Path(args.output_root) if args.output_root else (DATA_DIR / "outputs" / run_id if run_id else None)

    def _add_shared(base_args: list[str]) -> list[str]:
        out = list(base_args)
        if run_id:
            out.append(f"--run-id={run_id}")
        if output_root is not None:
            out.append(f"--output-root={output_root}")
        return out

    def _print_mechanism_stack_summary(root: Path | None):
        import pandas as pd

        base = root if root is not None else Path("data")
        dis_dir = base / "disentangling"
        reg_dir = base / "regime_characterization"
        irf_dir = base / "msvar_irf"
        info_dir = base / "information_loss"
        mech_dir = base / "mechanism_synthesis"
        rob_dir = base / "robustness"

        print("\nKey Results:")

        dis_effects = dis_dir / "disentangling_effects.csv"
        if dis_effects.exists():
            df = pd.read_csv(dis_effects)
            keep = df[df["effect"].isin(["architecture_effect_avg", "information_effect_avg", "interaction_did"])]
            if not keep.empty:
                stats = ", ".join(f"{r.effect}={r.value:.3f}" for r in keep.itertuples())
                print(f"  Disentangling: {stats}")

        durations = reg_dir / "regime_durations.csv"
        if durations.exists():
            df = pd.read_csv(durations)
            if "expected_duration_months" in df.columns and not df.empty:
                min_dur = float(df["expected_duration_months"].min())
                max_dur = float(df["expected_duration_months"].max())
                print(f"  Regimes: expected duration range = {min_dur:.2f} to {max_dur:.2f} months")

        irf_comp = irf_dir / "msvar_irf_regime_comparison.csv"
        if irf_comp.exists():
            df = pd.read_csv(irf_comp)
            if "peak_abs_delta_regime1_minus_regime0" in df.columns and not df.empty:
                mean_abs_peak = float(df["peak_abs_delta_regime1_minus_regime0"].abs().mean())
                print(f"  IRF: mean |peak delta| across regimes = {mean_abs_peak:.3f}")

        info_seg = info_dir / "information_loss_segment_summary.csv"
        if info_seg.exists():
            df = pd.read_csv(info_seg)
            crisis = df[df["segment"] == "crisis"]
            if not crisis.empty:
                gain = float(crisis.iloc[0]["mean_rmse_improvement_pct_disagg_vs_agg"])
                pot = float(crisis.iloc[0]["mean_info_loss_potential"])
                print(f"  Information loss (crisis): RMSE gain={gain:.2f}%, potential={pot:.3f}")

        mech_table = mech_dir / "mechanism_synthesis_table.csv"
        if mech_table.exists():
            df = pd.read_csv(mech_table)
            if "supports_direction" in df.columns and len(df) > 0:
                support_share = float(df["supports_direction"].fillna(False).astype(bool).mean())
                print(f"  Mechanism synthesis: support share={support_share:.2%} ({len(df)} channels)")

        table_dir = rob_dir / "tables"
        figure_dir = rob_dir / "figures"
        n_tex = len(list(table_dir.glob("*.tex"))) if table_dir.exists() else 0
        n_fig = len(list(figure_dir.glob("*.png"))) if figure_dir.exists() else 0
        print(f"  Publication artifacts: {n_tex} LaTeX tables, {n_fig} figures")

        summary_md = rob_dir / "summary" / "statistics_summary.md"
        if summary_md.exists():
            print(f"  Summary markdown: {summary_md}")

    print("=" * 60)
    print("SRI LANKA RESERVES FORECASTING - FULL REPLICATION")
    print("=" * 60)

    # Step 1: Diagnostics
    print("\n[1/10] Running diagnostics...")
    from reserves_project.pipelines.run_diagnostics import run_all_diagnostics
    run_all_diagnostics(verbose=False, write_manifest=True)

    # Step 2: Model evaluation
    print("\n[2/10] Running model evaluation across all variable sets...")
    from reserves_project.pipelines.run_unified_evaluations import main as eval_main
    sys.argv = _add_shared([
        "reserves-unified",
        "--include-ms",
        "--include-llsv",
        "--include-bop",
        "--include-dma",
    ])
    if getattr(args, "split_specs", None):
        sys.argv.append(f"--split-specs={args.split_specs}")
    eval_main()

    # Step 3: Statistical tests
    print("\n[3/10] Running statistical tests...")
    from reserves_project.pipelines.run_statistical_tests import main as test_main
    for varset in ["parsimonious", "bop", "monetary", "pca", "full"]:
        sys.argv = _add_shared([
            "reserves-stat-tests",
            "--use-unified",
            f"--unified-varset={varset}",
            "--unified-horizon=1",
        ])
        try:
            test_main()
        except Exception as e:
            print(f"  Warning: Tests for {varset} failed: {e}")

    # Step 4: Disentangling analysis
    print("\n[4/10] Running disentangling analysis...")
    from reserves_project.pipelines.run_disentangling_analysis import main as dis_main
    sys.argv = _add_shared([
        "reserves-disentangling",
        "--varsets=parsimonious,bop",
        "--models=MS-VAR,XGBoost",
        "--horizon=1",
        "--split=test",
    ])
    dis_main()

    # Step 5: Regime characterization
    print("\n[5/10] Running regime characterization...")
    from reserves_project.pipelines.run_regime_characterization import main as reg_main
    sys.argv = _add_shared([
        "reserves-regime-characterization",
        "--varset=bop",
        "--fit-split=full",
    ])
    reg_main()

    # Step 6: IRF analysis
    print("\n[6/10] Running MS-VAR IRF analysis...")
    from reserves_project.pipelines.run_msvar_irf_analysis import main as irf_main
    sys.argv = _add_shared([
        "reserves-msvar-irf",
        "--varset=bop",
        "--fit-split=full",
    ])
    irf_main()

    # Step 7: Information-loss tests
    print("\n[7/10] Running information-loss tests...")
    from reserves_project.pipelines.run_information_loss_tests import main as info_main
    sys.argv = _add_shared([
        "reserves-information-loss",
        "--aggregated-varset=parsimonious",
        "--disaggregated-varset=bop",
        "--horizon=1",
        "--split=test",
    ])
    info_main()

    # Step 8: Mechanism synthesis
    print("\n[8/10] Running mechanism synthesis...")
    from reserves_project.pipelines.run_mechanism_synthesis import main as mech_main
    sys.argv = _add_shared(["reserves-mechanism-synthesis"])
    mech_main()

    # Step 9: Scenario analysis
    print("\n[9/10] Running scenario analysis...")
    from reserves_project.pipelines.run_scenario_analysis import main as scenario_main
    sys.argv = _add_shared([
        "reserves-scenarios",
        "--varset=parsimonious",
    ])
    scenario_main()

    # Step 10: Publication tables/figures
    print("\n[10/10] Generating publication tables and figures...")
    from reserves_project.pipelines.generate_robustness_tables import main as tables_main
    sys.argv = _add_shared(["reserves-tables"])
    tables_main()

    _print_mechanism_stack_summary(output_root)

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
  slreserves regimes --varset bop --fit-split train_valid
  slreserves irf --varset bop --max-horizon 24
  slreserves info-loss --aggregated-varset parsimonious --disaggregated-varset bop
  slreserves mechanisms --run-id 2026-02-26_mech
  slreserves disentangle --varsets parsimonious,bop --models MS-VAR,XGBoost
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
    p_eval.add_argument("--window-mode", choices=["full", "common_dates"], default="full")
    p_eval.add_argument("--segments", default="all,crisis,tranquil")
    p_eval.add_argument("--train-end", default="2019-12-01")
    p_eval.add_argument("--valid-end", default="2022-12-01")
    p_eval.add_argument(
        "--split-specs",
        default=None,
        help="label:train_end:valid_end comma list for split robustness runs",
    )
    p_eval.add_argument("--include-ms", action="store_true", help="Include MS-VAR model")
    p_eval.add_argument("--include-lstm", action="store_true", help="Include LSTM model")
    p_eval.add_argument("--include-llsv", action="store_true", help="Include LLSV model")
    p_eval.add_argument("--include-bop", action="store_true", help="Include BoP identity model")
    p_eval.add_argument("--include-dma", action="store_true", help="Append DMA/DMS combinations")
    p_eval.add_argument("--dma-alpha", type=float, default=0.99)
    p_eval.add_argument("--dma-warmup-periods", type=int, default=12)
    p_eval.add_argument("--dma-variance-window", type=int, default=24)
    p_eval.add_argument("--dma-min-model-obs", type=int, default=24)
    p_eval.add_argument("--dma-model-pool", default=None)
    p_eval.add_argument("--exclude-bvar", action="store_true")
    p_eval.add_argument("--run-id", default=None)
    p_eval.add_argument("--output-root", default=None)
    p_eval.set_defaults(func=cmd_evaluate)

    # regimes
    p_reg = subparsers.add_parser("regimes", help="Run MS-VAR regime characterization")
    p_reg.add_argument("--varset", default="parsimonious")
    p_reg.add_argument("--n-regimes", type=int, default=2)
    p_reg.add_argument("--ar-order", type=int, default=1)
    p_reg.add_argument("--fit-split", choices=["train", "train_valid", "full"], default="full")
    p_reg.add_argument("--train-end", default="2019-12-01")
    p_reg.add_argument("--valid-end", default="2022-12-01")
    p_reg.add_argument("--fit-start", default=None)
    p_reg.add_argument("--fit-end", default=None)
    p_reg.add_argument("--output-dir", default="data/regime_characterization")
    p_reg.add_argument("--run-id", default=None)
    p_reg.add_argument("--output-root", default=None)
    p_reg.set_defaults(func=cmd_regimes)

    # irf
    p_irf = subparsers.add_parser("irf", help="Run regime-conditional MS-VAR impulse responses")
    p_irf.add_argument("--varset", default="parsimonious")
    p_irf.add_argument("--n-regimes", type=int, default=2)
    p_irf.add_argument("--ar-order", type=int, default=1)
    p_irf.add_argument("--max-horizon", type=int, default=24)
    p_irf.add_argument("--fit-split", choices=["train", "train_valid", "full"], default="full")
    p_irf.add_argument("--train-end", default="2019-12-01")
    p_irf.add_argument("--valid-end", default="2022-12-01")
    p_irf.add_argument("--fit-start", default=None)
    p_irf.add_argument("--fit-end", default=None)
    p_irf.add_argument("--difference-irf", action="store_true")
    p_irf.add_argument("--output-dir", default="data/msvar_irf")
    p_irf.add_argument("--run-id", default=None)
    p_irf.add_argument("--output-root", default=None)
    p_irf.set_defaults(func=cmd_irf)

    # info-loss
    p_info = subparsers.add_parser("info-loss", help="Run formal aggregation information-loss tests")
    p_info.add_argument("--aggregated-varset", default="parsimonious")
    p_info.add_argument("--disaggregated-varset", default="bop")
    p_info.add_argument("--models", default=None)
    p_info.add_argument("--horizon", type=int, default=1)
    p_info.add_argument("--split", default="test")
    p_info.add_argument("--loss", choices=["squared", "absolute"], default="squared")
    p_info.add_argument("--segments", default="all,crisis,tranquil")
    p_info.add_argument("--min-obs-per-model", type=int, default=12)
    p_info.add_argument("--component-signs", default=None)
    p_info.add_argument("--input-dir", default="data/forecast_results_unified")
    p_info.add_argument("--output-dir", default="data/information_loss")
    p_info.add_argument("--run-id", default=None)
    p_info.add_argument("--output-root", default=None)
    p_info.set_defaults(func=cmd_info_loss)

    # mechanisms
    p_mech = subparsers.add_parser("mechanisms", help="Synthesize mechanism evidence into one table")
    p_mech.add_argument("--disentangling-dir", default="data/disentangling")
    p_mech.add_argument("--regime-dir", default="data/regime_characterization")
    p_mech.add_argument("--irf-dir", default="data/msvar_irf")
    p_mech.add_argument("--information-loss-dir", default="data/information_loss")
    p_mech.add_argument("--output-dir", default="data/mechanism_synthesis")
    p_mech.add_argument("--run-id", default=None)
    p_mech.add_argument("--output-root", default=None)
    p_mech.set_defaults(func=cmd_mechanisms)

    # disentangle
    p_dis = subparsers.add_parser("disentangle", help="Run 2x2 model-vs-information disentangling")
    p_dis.add_argument("--varsets", default="parsimonious,bop")
    p_dis.add_argument("--models", default="MS-VAR,XGBoost")
    p_dis.add_argument("--horizon", type=int, default=1)
    p_dis.add_argument("--split", default="test")
    p_dis.add_argument("--input-dir", default="data/forecast_results_unified")
    p_dis.add_argument("--output-dir", default="data/disentangling")
    p_dis.add_argument("--min-obs", type=int, default=12)
    p_dis.add_argument("--bootstrap-reps", type=int, default=1000)
    p_dis.add_argument("--random-seed", type=int, default=42)
    p_dis.add_argument("--skip-dm", action="store_true")
    p_dis.add_argument("--run-id", default=None)
    p_dis.add_argument("--output-root", default=None)
    p_dis.set_defaults(func=cmd_disentangle)

    # scenarios
    p_scen = subparsers.add_parser("scenarios", help="Run MS-VARX scenario analysis")
    p_scen.add_argument("--varset", default="parsimonious")
    p_scen.add_argument("--horizon", type=int, default=12)
    p_scen.add_argument("--scenarios", default="all", help="Comma-separated scenario keys or 'all'")
    p_scen.add_argument("--regime-mode", choices=["free", "locked", "path"], default="free")
    p_scen.add_argument("--no-figures", action="store_true")
    p_scen.add_argument("--run-id", default=None)
    p_scen.add_argument("--output-root", default=None)
    p_scen.set_defaults(func=cmd_scenarios)

    # tests
    p_test = subparsers.add_parser("tests", help="Run statistical tests (DM, MCS)")
    p_test.add_argument("--varset", default="parsimonious")
    p_test.add_argument("--horizon", type=int, default=1)
    p_test.add_argument("--split", choices=["train", "validation", "test"], default="test")
    p_test.add_argument("--run-id", default=None)
    p_test.add_argument("--output-root", default=None)
    p_test.set_defaults(func=cmd_tests)

    # tables
    p_tbl = subparsers.add_parser("tables", help="Generate publication-ready tables")
    p_tbl.add_argument("--run-id", default=None)
    p_tbl.add_argument("--output-root", default=None)
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
    p_repl.add_argument("--output-root", default=None, help="Override output root directory")
    p_repl.add_argument(
        "--split-specs",
        default="baseline:2019-12-01:2022-12-01,early:2018-12-01:2021-12-01,late:2020-12-01:2023-12-01",
        help="Split robustness axis for evaluation stage",
    )
    p_repl.set_defaults(func=cmd_replicate)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
