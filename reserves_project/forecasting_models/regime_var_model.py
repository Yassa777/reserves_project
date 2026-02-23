"""Markov-switching VAR forecasting (true MS-VAR)."""

from __future__ import annotations

import pandas as pd

from reserves_project.eval.metrics import compute_metrics, naive_mae_scale
from reserves_project.models.ms_switching_var import MarkovSwitchingVAR


TARGET_VAR = "gross_reserves_usd_m"


def run_regime_var_forecast(raw_df: pd.DataFrame, regime_df: pd.DataFrame, level_series: pd.Series, ar_order: int = 1):
    raw_df = raw_df.set_index("date")
    regime_df = regime_df.set_index("date")

    joined = raw_df.join(regime_df[["regime_init_high_vol"]], how="inner")

    train = joined[joined["split"] == "train"]
    validation = joined[joined["split"] == "validation"]
    test = joined[joined["split"] == "test"]

    variables = [c for c in raw_df.columns if c not in {"split", "regime_init_high_vol", "regime_threshold"}]

    y_train = train[variables].values
    init_states = train["regime_init_high_vol"].values

    msvar = MarkovSwitchingVAR(n_regimes=2, ar_order=ar_order)
    msvar.fit(y_train, init_states=init_states)

    horizon = len(validation) + len(test)
    history = joined[variables].iloc[-ar_order:].values
    fc = msvar.forecast(history, steps=horizon)
    fc_index = pd.concat([validation, test]).index
    fc_df = pd.DataFrame(fc, index=fc_index, columns=variables)

    # Convert differenced forecast to level forecast for target.
    start_date = validation.index.min()
    last_level = level_series.loc[level_series.index < start_date].iloc[-1]
    level_forecast = last_level + fc_df[TARGET_VAR].cumsum()

    actual_levels = level_series.reindex(fc_index)
    pred_val = level_forecast.loc[validation.index].values
    pred_test = level_forecast.loc[test.index].values

    mase_scale = naive_mae_scale(level_series.loc[level_series.index < start_date].values)
    metrics_val = compute_metrics(actual_levels.loc[validation.index].values, pred_val, mase_scale=mase_scale)
    metrics_test = compute_metrics(actual_levels.loc[test.index].values, pred_test, mase_scale=mase_scale)

    out = pd.concat([validation, test])[[TARGET_VAR, "split"]].copy()
    out["forecast"] = level_forecast.values
    out = out.reset_index().rename(columns={"index": "date", TARGET_VAR: "actual"})
    out["actual"] = actual_levels.values

    summary = {
        "regime_model": "ms_var",
        "train_rows": int(len(train)),
        "metrics_validation": metrics_val,
        "metrics_test": metrics_test,
    }

    return out, summary
