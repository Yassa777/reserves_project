"""MS-VARX helpers for scenario analysis."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from reserves_project.models.ms_switching_var import MarkovSwitchingVAR
from reserves_project.scenarios.paths import diff_from_last


def compute_init_states(
    diff_target: pd.Series,
    window: int = 6,
) -> np.ndarray:
    rolling_vol = diff_target.rolling(window).std()
    threshold = rolling_vol.median()
    # Regime 0 = high volatility (crisis), Regime 1 = low volatility (recovery)
    init_states = (rolling_vol < threshold).astype(int).fillna(0).values
    return init_states.astype(int)


def fit_msvarx(
    levels_df: pd.DataFrame,
    endog_cols: list[str],
    exog_cols: list[str],
    ar_order: int = 2,
    max_iter: int = 200,
) -> tuple[MarkovSwitchingVAR, pd.DataFrame]:
    cols = endog_cols + exog_cols
    levels = levels_df[cols].dropna().copy()
    diff_data = levels.diff().dropna()
    y = diff_data[endog_cols]
    x = diff_data[exog_cols] if exog_cols else None

    init_states = compute_init_states(y[endog_cols[0]])

    msvar = MarkovSwitchingVAR(n_regimes=2, ar_order=ar_order, max_iter=max_iter)
    msvar.fit(y.values, exog=x.values if x is not None else None, init_states=init_states)
    return msvar, diff_data


def forecast_msvarx(
    msvar: MarkovSwitchingVAR,
    diff_data: pd.DataFrame,
    levels_df: pd.DataFrame,
    target_col: str,
    endog_cols: list[str],
    exog_cols: list[str],
    exog_future_levels: Optional[pd.DataFrame],
    horizon: int,
    regime_mode: str = "free",
    regime_path: Optional[np.ndarray] = None,
    regime_probs: Optional[np.ndarray] = None,
) -> pd.Series:
    p = msvar.ar_order
    y_history = diff_data[endog_cols].values[-p:]

    exog_future = None
    if exog_cols and exog_future_levels is not None and not exog_future_levels.empty:
        last_exog = levels_df[exog_cols].iloc[-1]
        exog_future = diff_from_last(last_exog, exog_future_levels).values

    lock_regime = regime_mode == "locked"
    steps = horizon if horizon is not None else (len(exog_future_levels) if exog_future_levels is not None else 0)
    forecasts = msvar.forecast(
        y_history,
        steps=steps,
        exog_future=exog_future,
        regime_probs=regime_probs,
        lock_regime=lock_regime,
        regime_path=regime_path,
    )

    target_idx = endog_cols.index(target_col)
    last_level = float(levels_df[target_col].iloc[-1])
    level_forecast = last_level + np.cumsum(forecasts[:, target_idx])

    if exog_future_levels is not None and not exog_future_levels.empty:
        forecast_index = exog_future_levels.index
    else:
        last_date = levels_df.index.max()
        forecast_index = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=steps,
            freq="MS",
        )
    return pd.Series(level_forecast, index=forecast_index, name=target_col)
