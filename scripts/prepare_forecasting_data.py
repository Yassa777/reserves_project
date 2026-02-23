"""Compatibility shim for forecasting prep pipeline."""

from reserves_project.pipelines.prepare_forecasting_data import run_forecasting_prep

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare forecasting datasets.")
    parser.add_argument("--varset", default=None, help="Variable set to use (baseline or expanded).")
    args = parser.parse_args()
    run_forecasting_prep(varset=args.varset, verbose=True)
