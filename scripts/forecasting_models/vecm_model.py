"""VECM forecasting using prepared levels dataset."""

from __future__ import annotations

import warnings

import pandas as pd
from statsmodels.tsa.vector_ar.vecm import VECM

from .metrics import compute_metrics, naive_mae_scale


TARGET_VAR = "gross_reserves_usd_m"


def run_vecm_forecast(df: pd.DataFrame, coint_rank: int, k_ar_diff: int):
    df = df.set_index("date")
    train = df[df["split"] == "train"]
    validation = df[df["split"] == "validation"]
    test = df[df["split"] == "test"]

    variables = [c for c in df.columns if c not in {"split"}]
    variables = [c for c in variables if c != "split"]

    train_levels = train[variables]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VECM(
            train_levels,
            k_ar_diff=max(1, k_ar_diff),
            coint_rank=max(1, min(coint_rank, len(variables) - 1)),
            deterministic="co",
        )
        res = model.fit()

    horizon = len(validation) + len(test)
    fc = res.predict(steps=horizon)
    fc_index = pd.concat([validation, test]).index
    fc_df = pd.DataFrame(fc, index=fc_index, columns=variables)

    actual_val = validation[TARGET_VAR].values
    actual_test = test[TARGET_VAR].values
    pred_val = fc_df.loc[validation.index, TARGET_VAR].values
    pred_test = fc_df.loc[test.index, TARGET_VAR].values

    mase_scale = naive_mae_scale(train[TARGET_VAR].values)
    metrics_val = compute_metrics(actual_val, pred_val, mase_scale=mase_scale)
    metrics_test = compute_metrics(actual_test, pred_test, mase_scale=mase_scale)

    out = pd.concat([validation, test])[[TARGET_VAR, "split"]].copy()
    out["forecast"] = fc_df[TARGET_VAR].values
    out = out.reset_index().rename(columns={"index": "date", TARGET_VAR: "actual"})

    summary = {
        "coint_rank": coint_rank,
        "k_ar_diff": k_ar_diff,
        "metrics_validation": metrics_val,
        "metrics_test": metrics_test,
    }

    return out, summary
