"""Run baseline forecasting models."""

from reserves_project.pipelines.run_forecasting_models import run_forecasts


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run forecasting baselines.")
    parser.add_argument("--varset", default=None, help="Variable set to use (baseline or expanded).")
    args = parser.parse_args()

    run_forecasts(varset=args.varset, verbose=True)
