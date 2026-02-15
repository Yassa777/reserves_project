"""Markov-switching VECM (true MS-VECM via MS-VAR on differenced system)."""

from __future__ import annotations

import pandas as pd

from .metrics import compute_metrics, naive_mae_scale
from .ms_switching_var import MarkovSwitchingVAR


TARGET_VAR = "d_gross_reserves_usd_m"


def run_ms_vecm_forecast(state_df: pd.DataFrame, level_series: pd.Series, ar_order: int = 1):
    df = state_df.set_index("date")

    train = df[df["split"] == "train"]
    validation = df[df["split"] == "validation"]
    test = df[df["split"] == "test"]

    exog_cols = [c for c in df.columns if c.startswith("d_") and c != TARGET_VAR]
    if "ect_lag1" in df.columns:
        exog_cols.append("ect_lag1")

    if "regime_init_high_vol" not in df.columns:
        raise ValueError("regime_init_high_vol not found in ms_vecm_state dataset")

    y_cols = [c for c in df.columns if c.startswith("d_") and c != "split"]
    y_train = train[y_cols].values
    init_states = train["regime_init_high_vol"].values

    exog_train = train[["ect_lag1"]] if "ect_lag1" in df.columns else None

    msvar = MarkovSwitchingVAR(n_regimes=2, ar_order=ar_order)
    msvar.fit(y_train, exog=exog_train.values if exog_train is not None else None, init_states=init_states)

    combined = pd.concat([validation, test])
    exog_future = combined[["ect_lag1"]].values if "ect_lag1" in df.columns else None
    history = df[y_cols].iloc[-ar_order:].values
    forecast = msvar.forecast(history, steps=len(combined), exog_future=exog_future)
    forecast_df = pd.DataFrame(forecast, index=combined.index, columns=y_cols)

    start_date = validation.index.min()
    last_level = level_series.loc[level_series.index < start_date].iloc[-1]
    level_forecast = last_level + forecast_df["d_gross_reserves_usd_m"].cumsum()
    actual_levels = level_series.reindex(combined.index)

    mase_scale = naive_mae_scale(level_series.loc[level_series.index < start_date].values)
    metrics_val = compute_metrics(
        actual_levels.loc[validation.index].values,
        level_forecast.loc[validation.index].values,
        mase_scale=mase_scale,
    )
    metrics_test = compute_metrics(
        actual_levels.loc[test.index].values,
        level_forecast.loc[test.index].values,
        mase_scale=mase_scale,
    )

    out = combined[[TARGET_VAR, "split"]].copy()
    out["forecast"] = level_forecast.values
    out = out.reset_index().rename(columns={"index": "date", TARGET_VAR: "actual"})
    out["actual"] = actual_levels.values

    summary = {
        "regime_model": "ms_vecm",
        "train_rows": int(len(train)),
        "metrics_validation": metrics_val,
        "metrics_test": metrics_test,
    }

    return out, summary
