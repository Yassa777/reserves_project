"""Univariate ARIMA (SARIMAX) forecasting."""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .metrics import compute_metrics, naive_mae_scale


TARGET_VAR = "gross_reserves_usd_m"


def _select_order(y: pd.Series, exog: pd.DataFrame | None, d: int = 1):
    candidates = list(itertools.product([0, 1, 2], [d], [0, 1, 2]))
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


def run_arima_forecast(df: pd.DataFrame, exog_vars: list[str]):
    df = df.set_index("date")

    if exog_vars:
        # Keep only rows with complete exogenous coverage.
        keep_cols = [TARGET_VAR] + exog_vars
        df = df.dropna(subset=keep_cols)
    train = df[df["split"] == "train"]
    validation = df[df["split"] == "validation"]
    test = df[df["split"] == "test"]

    y_train = train[TARGET_VAR]
    exog_train = train[exog_vars] if exog_vars else None

    order = _select_order(y_train, exog_train, d=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            y_train,
            order=order,
            exog=exog_train,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False)

    # Forecast validation + test in one pass
    horizon = len(validation) + len(test)
    exog_future = None
    if exog_vars:
        exog_future = pd.concat([validation[exog_vars], test[exog_vars]])

    forecast = res.forecast(steps=horizon, exog=exog_future)
    forecast.index = pd.concat([validation, test]).index

    combined = pd.concat([validation, test])
    actual = combined[TARGET_VAR]

    mase_scale = naive_mae_scale(y_train.values)
    metrics_val = compute_metrics(
        validation[TARGET_VAR].values,
        forecast.loc[validation.index].values,
        mase_scale=mase_scale,
    )
    metrics_test = compute_metrics(
        test[TARGET_VAR].values,
        forecast.loc[test.index].values,
        mase_scale=mase_scale,
    )

    out = combined[[TARGET_VAR, "split"]].copy()
    out["forecast"] = forecast.values
    out = out.reset_index().rename(columns={"index": "date", TARGET_VAR: "actual"})

    summary = {
        "order": order,
        "aic": float(res.aic),
        "metrics_validation": metrics_val,
        "metrics_test": metrics_test,
    }

    return out, summary
