#!/usr/bin/env python3
"""Rolling backtests for ARIMA, VECM, MS-VAR, and MS-VECM."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.vecm import VECM
import warnings

from reserves_project.config.paths import DATA_DIR
from reserves_project.forecasting_models.data_loader import (
    get_results_dir,
    estimate_johansen_rank,
    estimate_k_ar_diff,
    load_prep_csv,
    load_prep_metadata,
)
from reserves_project.eval.leakage_checks import assert_no_future_in_history, history_debug_info
from reserves_project.eval.metrics import compute_metrics, naive_mae_scale
from reserves_project.models.ms_switching_var import MarkovSwitchingVAR
from reserves_project.utils.run_manifest import write_run_manifest, write_latest_pointer


TARGET_VAR = "gross_reserves_usd_m"

def _select_arima_order(y: pd.Series, exog: pd.DataFrame | None, d: int = 1):
    candidates = [(p, d, q) for p in range(0, 4) for q in range(0, 4)]
    best = None
    for order in candidates:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SARIMAX(
                    y,
                    order=order,
                    exog=exog,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                res = model.fit(disp=False)
            if best is None or res.aic < best[0]:
                best = (res.aic, order)
        except Exception:
            continue
    return best[1] if best else (1, d, 1)


def _rolling_arima(
    df: pd.DataFrame,
    exog_vars: list[str],
    refit_interval: int = 12,
    include_debug_cols: bool = False,
):
    df = df.set_index("date")
    df = df.dropna(subset=[TARGET_VAR] + exog_vars)

    train = df[df["split"] == "train"]
    future = df[df["split"].isin(["validation", "test"])]

    forecasts = []
    last_refit = -refit_interval
    model = None
    order = None

    for i, (date, row) in enumerate(future.iterrows()):
        hist = pd.concat([train, future.iloc[:i]])
        assert_no_future_in_history(hist.index, date, context="ARIMA")

        if i - last_refit >= refit_interval or model is None:
            y = hist[TARGET_VAR]
            exog = hist[exog_vars] if exog_vars else None
            if order is None:
                order = _select_arima_order(y, exog, d=1)
            model = SARIMAX(y, order=order, exog=exog, enforce_stationarity=False, enforce_invertibility=False)
            model = model.fit(disp=False)
            last_refit = i

        exog_next = row[exog_vars].to_frame().T.astype(float) if exog_vars else None
        pred = model.forecast(steps=1, exog=exog_next).iloc[0]
        payload = {
            "date": date,
            "forecast": pred,
            "actual": row[TARGET_VAR],
            "split": row["split"],
            "model": "ARIMA",
        }
        if include_debug_cols:
            payload.update(history_debug_info(hist.index, date))
        forecasts.append(payload)

    return pd.DataFrame(forecasts)


def _rolling_vecm(
    df: pd.DataFrame,
    coint_rank: int,
    k_ar_diff: int,
    refit_interval: int = 12,
    include_debug_cols: bool = False,
):
    df = df.set_index("date")
    train = df[df["split"] == "train"]
    future = df[df["split"].isin(["validation", "test"])]

    variables = [c for c in df.columns if c not in {"split", "regime_init_high_vol", "regime_threshold"}]

    forecasts = []
    last_refit = -refit_interval
    model = None

    for i, (date, row) in enumerate(future.iterrows()):
        hist = pd.concat([train, future.iloc[:i]])
        assert_no_future_in_history(hist.index, date, context="VECM")

        if i - last_refit >= refit_interval or model is None:
            levels = hist[variables]
            vecm = VECM(levels, k_ar_diff=max(1, k_ar_diff), coint_rank=max(1, min(coint_rank, len(variables) - 1)), deterministic="co")
            model = vecm.fit()
            last_refit = i

        pred = model.predict(steps=1)[0][variables.index(TARGET_VAR)]
        payload = {
            "date": date,
            "forecast": pred,
            "actual": row[TARGET_VAR],
            "split": row["split"],
            "model": "VECM",
        }
        if include_debug_cols:
            payload.update(history_debug_info(hist.index, date))
        forecasts.append(payload)

    return pd.DataFrame(forecasts)


def _rolling_msvar(
    diff_df: pd.DataFrame,
    level_series: pd.Series,
    refit_interval: int = 12,
    include_debug_cols: bool = False,
):
    df = diff_df.set_index("date")
    train = df[df["split"] == "train"]
    future = df[df["split"].isin(["validation", "test"])]

    variables = [c for c in df.columns if c not in {"split", "regime_init_high_vol", "regime_threshold"}]

    forecasts = []
    last_refit = -refit_interval
    model = None
    for i, (date, row) in enumerate(future.iterrows()):
        hist = pd.concat([train, future.iloc[:i]])
        assert_no_future_in_history(hist.index, date, context="MS-VAR")

        if i - last_refit >= refit_interval or model is None:
            init_states = hist["regime_init_high_vol"].values if "regime_init_high_vol" in hist.columns else None
            model = MarkovSwitchingVAR(n_regimes=2, ar_order=1)
            model.fit(hist[variables].values, init_states=init_states)
            last_refit = i

        hist_y = hist[variables].values
        pred_diff = model.forecast(hist_y[-1:], steps=1)[0][variables.index(TARGET_VAR)]
        last_level = level_series.loc[level_series.index < date].iloc[-1]
        level_pred = last_level + pred_diff

        actual = level_series.reindex([date]).iloc[0]
        payload = {
            "date": date,
            "forecast": level_pred,
            "actual": actual,
            "split": row["split"],
            "model": "MS-VAR",
        }
        if include_debug_cols:
            payload.update(history_debug_info(hist.index, date))
        forecasts.append(payload)

    return pd.DataFrame(forecasts)


def _rolling_msvecm(
    state_df: pd.DataFrame,
    level_series: pd.Series,
    refit_interval: int = 12,
    include_debug_cols: bool = False,
):
    df = state_df.set_index("date")
    train = df[df["split"] == "train"]
    future = df[df["split"].isin(["validation", "test"])]

    y_cols = [c for c in df.columns if c.startswith("d_")]
    diff_target = "d_gross_reserves_usd_m"

    forecasts = []
    last_refit = -refit_interval
    model = None

    for i, (date, row) in enumerate(future.iterrows()):
        hist = pd.concat([train, future.iloc[:i]])
        assert_no_future_in_history(hist.index, date, context="MS-VECM")

        if i - last_refit >= refit_interval or model is None:
            init_states = hist["regime_init_high_vol"].values if "regime_init_high_vol" in hist.columns else None
            exog = hist[["ect_lag1"]].values if "ect_lag1" in hist.columns else None
            model = MarkovSwitchingVAR(n_regimes=2, ar_order=1)
            model.fit(hist[y_cols].values, exog=exog, init_states=init_states)
            last_refit = i

        exog_next = row[["ect_lag1"]].to_frame().T.values if "ect_lag1" in df.columns else None
        pred_diff = model.forecast(hist[y_cols].values[-1:], steps=1, exog_future=exog_next)[0][y_cols.index(diff_target)]
        last_level = level_series.loc[level_series.index < date].iloc[-1]
        level_pred = last_level + pred_diff

        actual = level_series.reindex([date]).iloc[0]
        payload = {
            "date": date,
            "forecast": level_pred,
            "actual": actual,
            "split": row["split"],
            "model": "MS-VECM",
        }
        if include_debug_cols:
            payload.update(history_debug_info(hist.index, date))
        forecasts.append(payload)

    return pd.DataFrame(forecasts)


def _rolling_naive(df: pd.DataFrame):
    df = df.set_index("date")
    future = df[df["split"].isin(["validation", "test"])]
    prev_values = df[TARGET_VAR].shift(1)
    forecasts = []
    for date, row in future.iterrows():
        forecast = float(prev_values.loc[date])
        forecasts.append({"date": date, "forecast": forecast, "actual": row[TARGET_VAR], "split": row["split"], "model": "Naive"})
    return pd.DataFrame(forecasts)


def run_backtests(
    refit_interval: int = 12,
    varset: str | None = None,
    output_root: Path | None = None,
    run_id: str | None = None,
    include_debug_cols: bool = False,
):
    print("=" * 70)
    print("ROLLING BACKTESTS")
    print("=" * 70)
    print(f"Started: {datetime.now()}")
    if varset:
        print(f"Varset: {varset}")

    arima_df = load_prep_csv("arima_prep_dataset.csv", varset)
    vecm_levels = load_prep_csv("vecm_levels_dataset.csv", varset)
    ms_var_raw = load_prep_csv("ms_var_raw_dataset.csv", varset)
    ms_var_scaled = load_prep_csv("ms_var_scaled_dataset.csv", varset)
    ms_vecm_state = load_prep_csv("ms_vecm_state_dataset.csv", varset)

    meta = load_prep_metadata(varset)
    train_end = pd.Timestamp(meta["splits"]["train_end"])
    k_ar_diff = estimate_k_ar_diff(vecm_levels, train_end=train_end)
    joh_rank = estimate_johansen_rank(vecm_levels, train_end=train_end, k_ar_diff=k_ar_diff)

    level_series = arima_df.set_index("date")[TARGET_VAR]

    arima_bt = _rolling_arima(
        arima_df,
        meta["arima"]["arima_exog_vars"],
        refit_interval,
        include_debug_cols=include_debug_cols,
    )
    vecm_bt = _rolling_vecm(
        vecm_levels,
        joh_rank,
        k_ar_diff,
        refit_interval,
        include_debug_cols=include_debug_cols,
    )
    msvar_bt = _rolling_msvar(
        ms_var_raw.join(ms_var_scaled["regime_init_high_vol"]),
        level_series,
        refit_interval,
        include_debug_cols=include_debug_cols,
    )
    msvecm_bt = _rolling_msvecm(
        ms_vecm_state,
        level_series,
        refit_interval,
        include_debug_cols=include_debug_cols,
    )
    naive_bt = _rolling_naive(arima_df)

    all_bt = pd.concat([arima_bt, vecm_bt, msvar_bt, msvecm_bt, naive_bt], ignore_index=True)
    results_dir = get_results_dir(varset, output_root=output_root)
    bt_path = results_dir / "rolling_backtests.csv"
    all_bt.to_csv(bt_path, index=False)

    train_end = arima_df[arima_df["split"] == "train"]["date"].max()
    train_values = level_series.loc[level_series.index <= train_end].values
    mase_scale = naive_mae_scale(train_values)

    summary_rows = []
    for model in all_bt["model"].unique():
        for split in ["validation", "test"]:
            subset = all_bt[(all_bt["model"] == model) & (all_bt["split"] == split)]
            metrics = compute_metrics(subset["actual"].values, subset["forecast"].values, mase_scale=mase_scale)
            summary_rows.append({"model": model, "split": split, **metrics})

    summary_df = pd.DataFrame(summary_rows)
    summary_path = results_dir / "rolling_backtest_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    meta_path = results_dir / "rolling_backtest_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(
            {"timestamp": str(datetime.now()), "refit_interval": refit_interval, "varset": meta.get("varset", varset or "baseline")},
            f,
            indent=2,
        )

    write_run_manifest(
        results_dir,
        {
            "refit_interval": refit_interval,
            "varset": meta.get("varset", varset or "baseline"),
            "output_root": str(output_root) if output_root is not None else None,
            "include_debug_cols": include_debug_cols,
        },
    )
    if run_id and output_root is not None:
        write_latest_pointer(DATA_DIR / "outputs", run_id, output_root)

    print("Saved:")
    print(f"  - {bt_path}")
    print(f"  - {summary_path}")
    print(f"  - {meta_path}")
    print("=" * 70)

    return all_bt, summary_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run rolling backtests.")
    parser.add_argument("--varset", default=None, help="Variable set to use (baseline or expanded).")
    parser.add_argument("--refit-interval", type=int, default=12)
    parser.add_argument("--run-id", default=None, help="Optional run ID to nest outputs in data/outputs/<run-id>/.")
    parser.add_argument("--output-root", default=None, help="Optional output root (overrides --run-id).")
    parser.add_argument("--include-debug-cols", action="store_true", help="Include origin/history debug columns.")
    args = parser.parse_args()
    output_root = None
    if args.output_root:
        output_root = Path(args.output_root)
    elif args.run_id:
        output_root = DATA_DIR / "outputs" / args.run_id
    run_backtests(
        refit_interval=args.refit_interval,
        varset=args.varset,
        output_root=output_root,
        run_id=args.run_id,
        include_debug_cols=args.include_debug_cols,
    )
