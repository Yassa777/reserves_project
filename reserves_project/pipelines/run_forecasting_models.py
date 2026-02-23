#!/usr/bin/env python3
"""Run forecasting baselines for ARIMA, VECM, MS-VAR, and MS-VECM."""

from __future__ import annotations

import json
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from reserves_project.config.paths import DATA_DIR
from reserves_project.forecasting_models.arima_model import run_arima_forecast
from reserves_project.forecasting_models.vecm_model import run_vecm_forecast
from reserves_project.forecasting_models.regime_var_model import run_regime_var_forecast
from reserves_project.forecasting_models.ms_vecm_model import run_ms_vecm_forecast
from reserves_project.forecasting_models.data_loader import (
    get_results_dir,
    estimate_johansen_rank,
    estimate_k_ar_diff,
    load_prep_csv,
    load_prep_metadata,
)
from reserves_project.eval.metrics import compute_metrics, naive_mae_scale
from reserves_project.utils.run_manifest import write_run_manifest


def _save_forecast(df: pd.DataFrame, filename: str, results_dir: Path) -> str:
    path = results_dir / filename
    df.to_csv(path, index=False)
    return str(path)


def _build_naive_forecast(
    actual_df: pd.DataFrame,
    train_last_value: float,
) -> pd.DataFrame:
    out = actual_df.copy()
    out["forecast"] = train_last_value
    out = out.reset_index().rename(columns={"index": "date", "gross_reserves_usd_m": "actual"})
    return out


def run_forecasts(
    varset: str | None = None,
    verbose: bool = True,
    output_root: Path | None = None,
):
    if verbose:
        print("=" * 70)
        print("FORECASTING MODELS")
        print("=" * 70)
        print(f"Started: {datetime.now()}")
        if varset:
            print(f"Varset: {varset}")

    arima_df = load_prep_csv("arima_prep_dataset.csv", varset)
    vecm_levels = load_prep_csv("vecm_levels_dataset.csv", varset)
    vecm_state = load_prep_csv("ms_vecm_state_dataset.csv", varset)
    ms_var_raw = load_prep_csv("ms_var_raw_dataset.csv", varset)
    ms_var_scaled = load_prep_csv("ms_var_scaled_dataset.csv", varset)

    meta = load_prep_metadata(varset)
    train_end = pd.Timestamp(meta["splits"]["train_end"])
    k_ar_diff = estimate_k_ar_diff(vecm_levels, train_end=train_end)
    joh_rank = estimate_johansen_rank(vecm_levels, train_end=train_end, k_ar_diff=k_ar_diff)

    results_dir = get_results_dir(varset, output_root=output_root)

    # ARIMA
    arima_out, arima_summary = run_arima_forecast(arima_df, meta["arima"]["arima_exog_vars"])
    # VECM
    vecm_out, vecm_summary = run_vecm_forecast(vecm_levels, joh_rank, k_ar_diff)
    # MS-VAR / MS-VECM level reference from full target series
    level_series = arima_df.set_index("date")["gross_reserves_usd_m"]
    reg_var_out, reg_var_summary = run_regime_var_forecast(ms_var_raw, ms_var_scaled, level_series)
    # MS-VECM
    ms_vecm_out, ms_vecm_summary = run_ms_vecm_forecast(vecm_state, level_series)

    outputs = {
        "arima": _save_forecast(arima_out, "arima_forecast.csv", results_dir),
        "vecm": _save_forecast(vecm_out, "vecm_forecast.csv", results_dir),
        "ms_var": _save_forecast(reg_var_out, "ms_var_forecast.csv", results_dir),
        "ms_vecm": _save_forecast(ms_vecm_out, "ms_vecm_forecast.csv", results_dir),
    }

    # Naive benchmark (last observation carried forward from train)
    arima_df_idx = arima_df.set_index("date")
    train = arima_df_idx[arima_df_idx["split"] == "train"]
    future = arima_df_idx[arima_df_idx["split"].isin(["validation", "test"])]
    train_last = float(train["gross_reserves_usd_m"].iloc[-1])
    naive_out = _build_naive_forecast(
        future[["gross_reserves_usd_m", "split"]],
        train_last,
    )
    outputs["naive"] = _save_forecast(naive_out, "naive_forecast.csv", results_dir)

    mase_scale = naive_mae_scale(train["gross_reserves_usd_m"].values)
    naive_val = naive_out[naive_out["split"] == "validation"]
    naive_test = naive_out[naive_out["split"] == "test"]
    naive_summary = {
        "metrics_validation": compute_metrics(
            naive_val["actual"].values,
            naive_val["forecast"].values,
            mase_scale=mase_scale,
        ),
        "metrics_test": compute_metrics(
            naive_test["actual"].values,
            naive_test["forecast"].values,
            mase_scale=mase_scale,
        ),
    }

    summary = {
        "timestamp": str(datetime.now()),
        "varset": meta.get("varset", varset or "baseline"),
        "missing_strategy": meta.get("missing_strategy"),
        "output_root": str(output_root) if output_root is not None else None,
        "arima": arima_summary,
        "vecm": vecm_summary,
        "ms_var": reg_var_summary,
        "ms_vecm": ms_vecm_summary,
        "naive": naive_summary,
    }

    summary_path = results_dir / "forecast_model_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    write_run_manifest(results_dir, summary)

    if verbose:
        print("\nSaved forecasts:")
        for key, path in outputs.items():
            print(f"  - {key}: {path}")
        print(f"  - summary: {summary_path}")
        print("=" * 70)
        print(f"Completed: {datetime.now()}")

    return outputs, summary


if __name__ == "__main__":
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
