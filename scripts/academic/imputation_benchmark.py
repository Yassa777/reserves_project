#!/usr/bin/env python3
"""Benchmark imputation strategies with artificial masking + sensitivity analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from reserves_project.scripts.academic.variable_sets.config import VARIABLE_SETS

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data" / "merged" / "reserves_forecasting_panel.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "imputation_benchmark"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def mask_data(df: pd.DataFrame, columns: List[str], rate: float) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    masked = df.copy()
    for col in columns:
        series = masked[col].copy()
        idx = series.dropna().index
        n_mask = max(1, int(len(idx) * rate))
        mask_idx = rng.choice(idx, size=n_mask, replace=False)
        series.loc[mask_idx] = np.nan
        masked[col] = series
    return masked


def impute_ffill_limit(df: pd.DataFrame, columns: List[str], limit: int = 3) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        out[col] = out[col].ffill(limit=limit)
    return out


def impute_linear(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        out[col] = out[col].interpolate(method="time")
    return out


def impute_seasonal_naive(df: pd.DataFrame, columns: List[str], lag: int = 12) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        series = out[col].copy()
        for i in range(lag, len(series)):
            if pd.isna(series.iloc[i]):
                series.iloc[i] = series.iloc[i - lag]
        out[col] = series
    return out


def impute_mean(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        out[col] = out[col].fillna(out[col].mean())
    return out


def evaluate_imputation(original: pd.DataFrame, imputed: pd.DataFrame, mask: pd.DataFrame, columns: List[str]) -> Dict:
    results = {}
    for col in columns:
        mask_idx = mask[col].isna()
        if mask_idx.sum() == 0:
            continue
        true = original.loc[mask_idx, col]
        pred = imputed.loc[mask_idx, col]
        rmse = float(np.sqrt(np.mean((true - pred) ** 2)))
        mae = float(np.mean(np.abs(true - pred)))
        results[col] = {"rmse": rmse, "mae": mae, "n": int(mask_idx.sum())}
    return results


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"], index_col="date")
    df = df.sort_index()

    # Use union of variables across varsets (excluding target-only duplicates)
    var_cols = set()
    for varset in VARIABLE_SETS.values():
        var_cols.update(varset.get("arima_exog", []))
        var_cols.update(varset.get("vecm_system", []))
        var_cols.update(varset.get("var_system", []))
    var_cols = [c for c in var_cols if c in df.columns]

    rates = [0.05, 0.1, 0.2]
    methods = {
        "ffill_limit": lambda d: impute_ffill_limit(d, var_cols, limit=3),
        "linear": lambda d: impute_linear(d, var_cols),
        "seasonal_naive": lambda d: impute_seasonal_naive(d, var_cols, lag=12),
        "mean": lambda d: impute_mean(d, var_cols),
    }

    all_rows = []

    for rate in rates:
        masked = mask_data(df, var_cols, rate)
        for name, imputer in methods.items():
            imputed = imputer(masked)
            scores = evaluate_imputation(df, imputed, masked, var_cols)
            for col, metrics in scores.items():
                all_rows.append({
                    "method": name,
                    "missing_rate": rate,
                    "variable": col,
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "n_masked": metrics["n"],
                })

    results_df = pd.DataFrame(all_rows)
    results_path = OUTPUT_DIR / "imputation_benchmark.csv"
    results_df.to_csv(results_path, index=False)

    summary = results_df.groupby(["method", "missing_rate"]).agg({
        "rmse": "mean",
        "mae": "mean",
        "n_masked": "sum",
    }).reset_index().sort_values(["missing_rate", "rmse"])

    summary_path = OUTPUT_DIR / "imputation_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Saved: {results_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
