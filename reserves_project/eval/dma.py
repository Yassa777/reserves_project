"""Dynamic Model Averaging / Selection utilities for unified forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class DMAConfig:
    """Configuration for DMA/DMS post-processing."""

    alpha: float = 0.99
    variance_window: int = 24
    min_variance: float = 1e-6
    warmup_periods: int = 12
    min_model_obs: int = 24


def _normal_pdf(x: float, mu: float, sigma2: float, min_variance: float) -> float:
    var = max(float(sigma2), float(min_variance))
    sigma = sqrt(var)
    z = (x - mu) / sigma
    density = (1.0 / sqrt(2.0 * pi * var)) * np.exp(-0.5 * z**2)
    if not np.isfinite(density) or density <= 0:
        return 1e-300
    return float(density)


def _estimate_variances(
    forecast_matrix: np.ndarray,
    actuals: np.ndarray,
    window: int,
    min_variance: float,
) -> np.ndarray:
    """Estimate rolling forecast-error variances per model."""

    t_obs, k_models = forecast_matrix.shape
    variances = np.full((t_obs, k_models), np.nan)
    errors = actuals.reshape(-1, 1) - forecast_matrix

    for k in range(k_models):
        err_k = errors[:, k]
        global_var = np.nanvar(err_k)
        if not np.isfinite(global_var) or global_var < min_variance:
            global_var = 1.0

        for t in range(t_obs):
            if t < 2:
                var_t = global_var
            elif t < window:
                var_t = np.nanvar(err_k[:t])
            else:
                var_t = np.nanvar(err_k[t - window : t])
            if not np.isfinite(var_t) or var_t < min_variance:
                var_t = min_variance
            variances[t, k] = var_t

    return variances


def _parse_model_pool(
    available_models: Sequence[str],
    model_pool: Optional[Sequence[str]],
) -> List[str]:
    if not model_pool:
        return list(available_models)
    pool_set = {m.strip() for m in model_pool if m and m.strip()}
    return [m for m in available_models if m in pool_set]


def _dma_for_horizon(
    horizon_df: pd.DataFrame,
    horizon: int,
    config: DMAConfig,
    model_pool: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Compute DMA/DMS rows for a single forecast horizon."""

    if horizon_df.empty:
        return pd.DataFrame(), None

    pivot_fc = horizon_df.pivot_table(
        index="forecast_date",
        columns="model",
        values="forecast",
        aggfunc="mean",
    ).sort_index()

    actual = (
        horizon_df.groupby("forecast_date")["actual"]
        .mean()
        .reindex(pivot_fc.index)
    )
    split = (
        horizon_df.groupby("forecast_date")["split"]
        .first()
        .reindex(pivot_fc.index)
    )
    origin = (
        horizon_df.groupby("forecast_date")["forecast_origin"]
        .first()
        .reindex(pivot_fc.index)
    )

    candidate_models = [m for m in pivot_fc.columns if pivot_fc[m].notna().sum() >= config.min_model_obs]
    candidate_models = _parse_model_pool(candidate_models, model_pool)

    if len(candidate_models) < 2:
        return pd.DataFrame(), None

    fc = pivot_fc[candidate_models].to_numpy(dtype=float)
    y = actual.to_numpy(dtype=float)
    t_obs, k_models = fc.shape

    weights = np.zeros((t_obs, k_models), dtype=float)
    post = np.ones(k_models, dtype=float) / k_models
    dma_forecast = np.full(t_obs, np.nan, dtype=float)
    dms_forecast = np.full(t_obs, np.nan, dtype=float)

    variances = _estimate_variances(
        forecast_matrix=fc,
        actuals=y,
        window=config.variance_window,
        min_variance=config.min_variance,
    )

    for t in range(t_obs):
        fc_t = fc[t]
        available = np.isfinite(fc_t)
        if not available.any():
            continue

        if t < config.warmup_periods:
            w_t = np.zeros(k_models, dtype=float)
            w_t[available] = 1.0 / available.sum()
        else:
            prior = np.power(np.clip(post, 1e-12, None), config.alpha)
            prior_sum = prior.sum()
            if prior_sum <= 0 or not np.isfinite(prior_sum):
                prior = np.ones(k_models, dtype=float) / k_models
            else:
                prior = prior / prior_sum
            w_t = prior * available.astype(float)
            w_sum = w_t.sum()
            if w_sum <= 0 or not np.isfinite(w_sum):
                w_t = np.zeros(k_models, dtype=float)
                w_t[available] = 1.0 / available.sum()
            else:
                w_t = w_t / w_sum

        weights[t] = w_t
        dma_forecast[t] = float(np.dot(w_t, np.nan_to_num(fc_t, nan=0.0)))

        masked_w = np.where(available, w_t, -np.inf)
        best_idx = int(np.argmax(masked_w))
        dms_forecast[t] = float(fc_t[best_idx]) if np.isfinite(fc_t[best_idx]) else np.nan

        if t < config.warmup_periods:
            post = w_t if w_t.sum() > 0 else post
            continue

        y_t = y[t]
        if not np.isfinite(y_t):
            post = w_t if w_t.sum() > 0 else post
            continue

        prior = np.power(np.clip(post, 1e-12, None), config.alpha)
        prior_sum = prior.sum()
        if prior_sum <= 0 or not np.isfinite(prior_sum):
            prior = np.ones(k_models, dtype=float) / k_models
        else:
            prior = prior / prior_sum

        likelihood = np.ones(k_models, dtype=float)
        for k in range(k_models):
            if np.isfinite(fc_t[k]):
                likelihood[k] = _normal_pdf(
                    x=float(y_t),
                    mu=float(fc_t[k]),
                    sigma2=float(variances[t, k]),
                    min_variance=config.min_variance,
                )

        posterior = prior * likelihood
        post_sum = posterior.sum()
        if post_sum <= 0 or not np.isfinite(post_sum):
            post = prior
        else:
            post = posterior / post_sum

    rows = []
    for dt in pivot_fc.index:
        dt_loc = pivot_fc.index.get_loc(dt)
        rows.append(
            {
                "model": "DMA",
                "forecast_origin": origin.iloc[dt_loc],
                "forecast_date": dt,
                "horizon": int(horizon),
                "split": split.iloc[dt_loc],
                "actual": y[dt_loc],
                "forecast": dma_forecast[dt_loc],
                "std": np.nan,
                "lower_80": np.nan,
                "upper_80": np.nan,
                "lower_95": np.nan,
                "upper_95": np.nan,
                "crps": np.nan,
                "log_score": np.nan,
            }
        )
        rows.append(
            {
                "model": "DMS",
                "forecast_origin": origin.iloc[dt_loc],
                "forecast_date": dt,
                "horizon": int(horizon),
                "split": split.iloc[dt_loc],
                "actual": y[dt_loc],
                "forecast": dms_forecast[dt_loc],
                "std": np.nan,
                "lower_80": np.nan,
                "upper_80": np.nan,
                "lower_95": np.nan,
                "upper_95": np.nan,
                "crps": np.nan,
                "log_score": np.nan,
            }
        )

    dma_rows = pd.DataFrame(rows)
    weight_df = pd.DataFrame(weights, columns=candidate_models)
    weight_df.insert(0, "date", pivot_fc.index)
    weight_df.insert(1, "horizon", int(horizon))
    return dma_rows, weight_df


def augment_with_dma_dms(
    results: pd.DataFrame,
    alpha: float = 0.99,
    variance_window: int = 24,
    min_variance: float = 1e-6,
    warmup_periods: int = 12,
    min_model_obs: int = 24,
    model_pool: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Append DMA/DMS forecasts to unified rolling-origin outputs.

    Parameters
    ----------
    results : pd.DataFrame
        Unified evaluator output with per-model forecasts.
    alpha : float
        Forgetting factor in (0, 1].
    variance_window : int
        Rolling window for predictive-variance estimation.
    min_variance : float
        Lower bound for forecast error variance.
    warmup_periods : int
        Periods with equal weights before adaptation.
    min_model_obs : int
        Minimum non-missing forecasts required for model entry into DMA pool.
    model_pool : iterable[str], optional
        Restrict DMA pool to this model subset.
    """

    if results is None or results.empty:
        return results, pd.DataFrame()

    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1].")

    config = DMAConfig(
        alpha=float(alpha),
        variance_window=int(variance_window),
        min_variance=float(min_variance),
        warmup_periods=int(warmup_periods),
        min_model_obs=int(min_model_obs),
    )

    base = results[~results["model"].isin(["DMA", "DMS"])].copy()
    if base.empty:
        return results, pd.DataFrame()

    horizons = sorted(base["horizon"].dropna().unique().astype(int).tolist())
    pool = list(model_pool) if model_pool is not None else None

    dma_blocks: List[pd.DataFrame] = []
    weight_blocks: List[pd.DataFrame] = []
    for horizon in horizons:
        horizon_df = base[base["horizon"] == horizon].copy()
        dma_rows, weight_rows = _dma_for_horizon(
            horizon_df=horizon_df,
            horizon=int(horizon),
            config=config,
            model_pool=pool,
        )
        if not dma_rows.empty:
            dma_blocks.append(dma_rows)
        if weight_rows is not None and not weight_rows.empty:
            weight_blocks.append(weight_rows)

    if not dma_blocks:
        return results, pd.DataFrame()

    dma_df = pd.concat(dma_blocks, ignore_index=True)
    out = pd.concat([results, dma_df], ignore_index=True, sort=False)
    out = out.sort_values(["forecast_date", "horizon", "model"]).reset_index(drop=True)

    weights = pd.concat(weight_blocks, ignore_index=True) if weight_blocks else pd.DataFrame()
    return out, weights


__all__ = [
    "DMAConfig",
    "augment_with_dma_dms",
]

