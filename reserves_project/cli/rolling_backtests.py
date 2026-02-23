"""Run rolling backtests."""

from reserves_project.pipelines.run_rolling_backtests import run_backtests


def main() -> None:
    import argparse
    from pathlib import Path
    from reserves_project.config.paths import DATA_DIR

    parser = argparse.ArgumentParser(description="Run rolling backtests.")
    parser.add_argument("--varset", default=None, help="Variable set to use (baseline or expanded).")
    parser.add_argument("--refit-interval", type=int, default=12)
    parser.add_argument("--run-id", default=None, help="Optional run ID to nest outputs in data/outputs/<run-id>/.")
    parser.add_argument("--output-root", default=None, help="Optional output root (overrides --run-id).")
    args = parser.parse_args()

    output_root = None
    if args.output_root:
        output_root = Path(args.output_root)
    elif args.run_id:
        output_root = DATA_DIR / "outputs" / args.run_id

    run_backtests(refit_interval=args.refit_interval, varset=args.varset, output_root=output_root)
