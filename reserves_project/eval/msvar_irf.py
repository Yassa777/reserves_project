"""Regime-conditional generalized impulse responses for MS-VAR."""

from __future__ import annotations

from itertools import product
from typing import Sequence

import numpy as np
import pandas as pd

from reserves_project.models.ms_switching_var import MarkovSwitchingVAR


def _lag_matrices_column_form(model: MarkovSwitchingVAR, regime: int) -> list[np.ndarray]:
    if regime < 0 or regime >= model.n_regimes:
        raise ValueError(f"Invalid regime index: {regime}")
    if not model.coefs_:
        raise RuntimeError("Model coefficients unavailable; run fit() first.")

    coef = np.asarray(model.coefs_[regime], dtype=float)
    p = int(model.ar_order)
    k = coef.shape[1]
    needed_rows = 1 + p * k  # intercept + lag blocks
    if coef.shape[0] < needed_rows:
        raise RuntimeError(
            f"Coefficient matrix has insufficient rows for lag extraction: "
            f"need at least {needed_rows}, got {coef.shape[0]}."
        )

    lag_mats = []
    for lag in range(p):
        start = 1 + lag * k
        end = start + k
        # model is estimated in row form, convert to conventional column form.
        lag_block_row = coef[start:end, :]
        lag_block_col = lag_block_row.T
        lag_mats.append(lag_block_col)
    return lag_mats


def _ma_matrices(lag_mats: Sequence[np.ndarray], max_horizon: int) -> np.ndarray:
    if max_horizon < 0:
        raise ValueError("max_horizon must be >= 0")
    if not lag_mats:
        raise ValueError("At least one lag matrix is required.")

    p = len(lag_mats)
    k = lag_mats[0].shape[0]
    psi = np.zeros((max_horizon + 1, k, k))
    psi[0] = np.eye(k)

    for h in range(1, max_horizon + 1):
        acc = np.zeros((k, k))
        for lag in range(1, p + 1):
            if h - lag < 0:
                continue
            acc += lag_mats[lag - 1] @ psi[h - lag]
        psi[h] = acc

    return psi


def generalized_irf(
    model: MarkovSwitchingVAR,
    regime: int,
    max_horizon: int = 24,
    cumulative: bool = True,
) -> np.ndarray:
    """Return GIRF tensor with shape (horizon+1, n_response, n_shock)."""
    if model.cov_ is None or len(model.cov_) == 0:
        raise RuntimeError("Model covariance matrices unavailable; run fit() first.")

    sigma = np.asarray(model.cov_[regime], dtype=float)
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise RuntimeError("Invalid covariance matrix shape for GIRF.")
    if not np.all(np.isfinite(sigma)):
        raise RuntimeError("Covariance matrix contains non-finite values.")

    k = sigma.shape[0]
    lag_mats = _lag_matrices_column_form(model, regime=regime)
    psi = _ma_matrices(lag_mats, max_horizon=max_horizon)

    girf = np.zeros((max_horizon + 1, k, k))
    for shock_idx in range(k):
        shock_var = float(sigma[shock_idx, shock_idx])
        denom = np.sqrt(max(shock_var, 1e-12))
        shock_vec = sigma[:, shock_idx] / denom
        for h in range(max_horizon + 1):
            girf[h, :, shock_idx] = psi[h] @ shock_vec

    if cumulative:
        girf = np.cumsum(girf, axis=0)
    return girf


def girf_to_long_df(
    girf: np.ndarray,
    var_names: Sequence[str],
    regime: int,
    cumulative: bool = True,
) -> pd.DataFrame:
    """Convert GIRF tensor to long-form DataFrame."""
    if girf.ndim != 3:
        raise ValueError("GIRF tensor must be 3-dimensional.")
    h_plus_one, k_resp, k_shock = girf.shape
    if k_resp != len(var_names) or k_shock != len(var_names):
        raise ValueError("Variable-name length mismatch with GIRF dimensions.")

    rows = []
    for h, response_idx, shock_idx in product(range(h_plus_one), range(k_resp), range(k_shock)):
        rows.append(
            {
                "regime": int(regime),
                "horizon": int(h),
                "response_variable": var_names[response_idx],
                "shock_variable": var_names[shock_idx],
                "response": float(girf[h, response_idx, shock_idx]),
                "response_type": "cumulative" if cumulative else "difference",
            }
        )
    return pd.DataFrame(rows)


def _half_life_horizon(resp: np.ndarray) -> float:
    if len(resp) == 0:
        return np.nan
    baseline = abs(float(resp[0]))
    if baseline <= 1e-12:
        return 0.0
    threshold = 0.5 * baseline
    for h, value in enumerate(resp):
        if abs(float(value)) <= threshold:
            return float(h)
    return float(len(resp) - 1)


def summarize_regime_comparison(
    regime0_df: pd.DataFrame,
    regime1_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create pairwise persistence/amplitude comparison between two regimes."""
    required = {"horizon", "response_variable", "shock_variable", "response"}
    if missing := required.difference(regime0_df.columns):
        raise KeyError(f"regime0_df missing required columns: {sorted(missing)}")
    if missing := required.difference(regime1_df.columns):
        raise KeyError(f"regime1_df missing required columns: {sorted(missing)}")

    rows = []
    keys = ["shock_variable", "response_variable"]
    g0 = regime0_df.sort_values("horizon").groupby(keys)
    g1 = regime1_df.sort_values("horizon").groupby(keys)

    shared = sorted(set(g0.groups.keys()).intersection(set(g1.groups.keys())))
    for key in shared:
        s0 = g0.get_group(key)["response"].to_numpy(dtype=float)
        s1 = g1.get_group(key)["response"].to_numpy(dtype=float)
        peak0 = float(np.max(np.abs(s0))) if len(s0) else np.nan
        peak1 = float(np.max(np.abs(s1))) if len(s1) else np.nan
        hl0 = _half_life_horizon(s0)
        hl1 = _half_life_horizon(s1)
        rows.append(
            {
                "shock_variable": key[0],
                "response_variable": key[1],
                "peak_abs_response_regime0": peak0,
                "peak_abs_response_regime1": peak1,
                "peak_abs_delta_regime1_minus_regime0": float(peak1 - peak0),
                "half_life_horizon_regime0": hl0,
                "half_life_horizon_regime1": hl1,
                "half_life_delta_regime1_minus_regime0": float(hl1 - hl0),
            }
        )
    return pd.DataFrame(rows)
