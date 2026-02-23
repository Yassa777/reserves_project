"""Run baseline forecasting models."""

from reserves_project.pipelines.run_forecasting_models import run_forecasts


def main() -> None:
    import argparse
    from pathlib import Path
    from reserves_project.config.paths import DATA_DIR

    parser = argparse.ArgumentParser(description="Run forecasting baselines.")
    parser.add_argument("--varset", default=None, help="Variable set to use (baseline or expanded).")
    parser.add_argument("--run-id", default=None, help="Optional run ID to nest outputs in data/outputs/<run-id>/.")
    parser.add_argument("--output-root", default=None, help="Optional output root (overrides --run-id).")
    args = parser.parse_args()

    output_root = None
    if args.output_root:
        output_root = Path(args.output_root)
    elif args.run_id:
        output_root = DATA_DIR / "outputs" / args.run_id

    run_forecasts(varset=args.varset, verbose=True, output_root=output_root)
