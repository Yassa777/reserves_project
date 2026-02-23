#!/usr/bin/env python3
"""Prepare model-ready datasets for ARIMA, VECM, MS-VAR, and MS-VECM forecasting."""

from __future__ import annotations

from datetime import datetime
import argparse
import pandas as pd

from reserves_project.diagnostics.io_utils import load_panel
from reserves_project.forecasting_prep import (
    build_arima_dataset,
    build_model_readiness,
    build_ms_var_dataset,
    build_vecm_datasets,
    save_dataframe,
    save_metadata,
)
from reserves_project.forecasting_prep.config import TRAIN_END, VALID_END, get_output_dir, get_varset_name, MISSING_STRATEGY
from reserves_project.utils.run_manifest import write_run_manifest


def run_forecasting_prep(varset: str | None = None, verbose: bool = True):
    if verbose:
        print("=" * 70)
        print("FORECASTING DATA PREPARATION")
        print("=" * 70)
        print(f"Started: {datetime.now()}")
        print(f"Varset: {get_varset_name(varset)}")
        print(f"Missing strategy: {MISSING_STRATEGY}")

    panel = load_panel()
    if verbose:
        print(f"Loaded panel: {panel.shape[0]} rows x {panel.shape[1]} columns")
        print(f"Date range: {panel.index.min().date()} to {panel.index.max().date()}")
        print(f"Splits: train <= {TRAIN_END.date()}, validation <= {VALID_END.date()}, test > {VALID_END.date()}")

    arima_df, arima_meta = build_arima_dataset(panel, varset=varset, missing_strategy=MISSING_STRATEGY)
    vecm_levels_df, vecm_state_df, vecm_meta = build_vecm_datasets(panel, varset=varset, missing_strategy=MISSING_STRATEGY)
    ms_var_raw_df, ms_var_scaled_df, ms_var_meta = build_ms_var_dataset(panel, varset=varset, missing_strategy=MISSING_STRATEGY)

    metadata = {
        "timestamp": str(datetime.now()),
        "source": "data/merged/reserves_forecasting_panel.csv",
        "varset": get_varset_name(varset),
        "missing_strategy": MISSING_STRATEGY,
        "splits": {
            "train_end": str(TRAIN_END.date()),
            "validation_end": str(VALID_END.date()),
            "test_start": str((VALID_END + pd.offsets.MonthBegin(1)).date()),
        },
        "arima": arima_meta,
        "vecm": vecm_meta,
        "ms_var": ms_var_meta,
    }

    readiness_df = build_model_readiness(metadata)

    output_dir = get_output_dir(varset)
    outputs = {
        "arima": save_dataframe(arima_df, "arima_prep_dataset.csv", output_dir=output_dir),
        "vecm_levels": save_dataframe(vecm_levels_df, "vecm_levels_dataset.csv", output_dir=output_dir),
        "vecm_state": save_dataframe(vecm_state_df, "ms_vecm_state_dataset.csv", output_dir=output_dir),
        "ms_var_raw": save_dataframe(ms_var_raw_df, "ms_var_raw_dataset.csv", output_dir=output_dir),
        "ms_var_scaled": save_dataframe(ms_var_scaled_df, "ms_var_scaled_dataset.csv", output_dir=output_dir),
        "readiness": save_dataframe(readiness_df, "model_readiness_summary.csv", output_dir=output_dir),
        "metadata": save_metadata(metadata, output_dir=output_dir),
    }

    if verbose:
        print("\nSaved artifacts:")
        for key, path in outputs.items():
            print(f"  - {key}: {path}")
        print(f"\nOutput directory: {output_dir}")
        print("=" * 70)
        print(f"Completed: {datetime.now()}")

    write_run_manifest(output_dir, metadata)
    return outputs, metadata, readiness_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare forecasting datasets.")
    parser.add_argument("--varset", default=None, help="Variable set to use (baseline or expanded).")
    args = parser.parse_args()
    run_forecasting_prep(varset=args.varset, verbose=True)
