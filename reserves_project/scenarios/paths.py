"""Build baseline and scenario exogenous paths."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from reserves_project.eval.unified_evaluator import ARIMAExogForecaster, NaiveExogForecaster
from reserves_project.scenarios.definitions import Scenario


def build_baseline_exog_path(
    history_df: pd.DataFrame,
    exog_cols: list[str],
    horizon: int,
    method: str = "naive",
) -> pd.DataFrame:
    if not exog_cols:
        return pd.DataFrame()
    history_df = history_df.sort_index()
    forecast_index = pd.date_range(
        start=history_df.index.max() + pd.DateOffset(months=1),
        periods=horizon,
        freq="MS",
    )
    forecaster = ARIMAExogForecaster() if method == "arima" else NaiveExogForecaster()
    forecaster.fit(history_df, exog_cols)
    return forecaster.forecast(horizon, forecast_index)


def _profile_multipliers(profile: str, shock: float, horizon: int) -> np.ndarray:
    profile = (profile or "ramp").lower()
    if profile == "step":
        return np.full(horizon, shock)
    if profile == "impulse":
        multipliers = np.ones(horizon)
        multipliers[0] = shock
        return multipliers
    return np.linspace(1.0, shock, horizon)


def build_scenario_exog_path(
    baseline_path: pd.DataFrame,
    scenario: Scenario,
    variables: Iterable[str] | None = None,
) -> pd.DataFrame:
    if baseline_path.empty:
        return baseline_path
    horizon = len(baseline_path)
    shock_vars = list(variables) if variables is not None else list(baseline_path.columns)
    shocks = scenario.normalized_shocks(shock_vars)
    profile = scenario.normalized_profile()

    scenario_path = baseline_path.copy()
    for col in baseline_path.columns:
        shock = shocks.get(col, 1.0)
        multipliers = _profile_multipliers(profile, shock, horizon)
        scenario_path[col] = baseline_path[col].values * multipliers

    return scenario_path


def diff_from_last(last_values: pd.Series, future_levels: pd.DataFrame) -> pd.DataFrame:
    if future_levels.empty:
        return future_levels
    last = last_values.reindex(future_levels.columns).astype(float)
    diffs = []
    prev = last.values
    for _, row in future_levels.iterrows():
        cur = row.values.astype(float)
        diffs.append(cur - prev)
        prev = cur
    return pd.DataFrame(diffs, index=future_levels.index, columns=future_levels.columns)

