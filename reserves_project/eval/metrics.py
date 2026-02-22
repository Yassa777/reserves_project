"""Forecast evaluation metrics."""

from __future__ import annotations

import numpy as np


def compute_metrics(
    actual: np.ndarray,
    forecast: np.ndarray,
    mase_scale: float | None = None,
) -> dict[str, float]:
    mask = ~np.isnan(actual) & ~np.isnan(forecast)
    if mask.sum() == 0:
        return {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "smape": np.nan, "mase": np.nan}

    actual = actual[mask]
    forecast = forecast[mask]

    mae = float(np.mean(np.abs(actual - forecast)))
    rmse = float(np.sqrt(np.mean((actual - forecast) ** 2)))
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = float(np.mean(np.abs((actual - forecast) / actual)) * 100)
        smape = float(
            np.mean(
                200.0 * np.abs(actual - forecast) / (np.abs(actual) + np.abs(forecast))
            )
        )

    mase = np.nan
    if mase_scale is not None and mase_scale > 0:
        mase = float(mae / mase_scale)

    return {"mae": mae, "rmse": rmse, "mape": mape, "smape": smape, "mase": mase}


def naive_mae_scale(series: np.ndarray) -> float:
    series = np.asarray(series)
    series = series[~np.isnan(series)]
    if len(series) < 2:
        return np.nan
    diffs = np.abs(np.diff(series))
    scale = float(np.mean(diffs)) if len(diffs) else np.nan
    return scale


def asymmetric_loss(
    actual: np.ndarray,
    forecast: np.ndarray,
    under_weight: float = 2.0,
    over_weight: float = 1.0,
) -> float:
    """Asymmetric loss: penalize under-forecasting more than over-forecasting."""
    actual = np.asarray(actual)
    forecast = np.asarray(forecast)
    mask = ~np.isnan(actual) & ~np.isnan(forecast)
    if mask.sum() == 0:
        return np.nan
    actual = actual[mask]
    forecast = forecast[mask]
    errors = forecast - actual
    loss = np.where(errors < 0, under_weight * np.abs(errors), over_weight * np.abs(errors))
    return float(np.mean(loss))
