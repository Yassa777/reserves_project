"""Run rolling backtests."""

from reserves_project.pipelines.run_rolling_backtests import run_backtests


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run rolling backtests.")
    parser.add_argument("--varset", default=None, help="Variable set to use (baseline or expanded).")
    parser.add_argument("--refit-interval", type=int, default=12)
    args = parser.parse_args()

    run_backtests(refit_interval=args.refit_interval, varset=args.varset)
