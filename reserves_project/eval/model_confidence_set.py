"""
Model Confidence Set (MCS) Procedure
====================================

Implements the Model Confidence Set procedure of Hansen, Lunde & Nason (2011)
for simultaneous comparison of multiple forecasting models.

The MCS procedure:
1. Starts with a set of all candidate models
2. Tests if all models have equal predictive ability
3. If rejected, eliminates the worst model
4. Repeats until the null cannot be rejected
5. Returns the surviving models as the "confidence set"

References:
-----------
- Hansen, P.R., Lunde, A., & Nason, J.M. (2011). The Model Confidence Set.
  Econometrica, 79(2), 453-497.

Author: Academic Forecasting Pipeline
Date: 2026-02-10
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats


def compute_loss_series(
    actual: np.ndarray,
    forecasts_dict: Dict[str, np.ndarray],
    loss_fn: str = "squared"
) -> pd.DataFrame:
    """
    Compute loss series for all models.

    Parameters
    ----------
    actual : np.ndarray
        Actual values
    forecasts_dict : dict
        {model_name: forecast_array}
    loss_fn : str
        "squared" for MSE or "absolute" for MAE

    Returns
    -------
    losses : pd.DataFrame
        DataFrame with loss series for each model
    """
    actual = np.asarray(actual, dtype=float)
    losses = {}

    for model_name, forecast in forecasts_dict.items():
        forecast = np.asarray(forecast, dtype=float)
        error = actual - forecast

        if loss_fn == "squared":
            losses[model_name] = error**2
        elif loss_fn == "absolute":
            losses[model_name] = np.abs(error)
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")

    return pd.DataFrame(losses)


def compute_relative_loss(
    loss_df: pd.DataFrame,
    model_i: str,
    model_j: str
) -> np.ndarray:
    """
    Compute relative loss d_ij = L_i - L_j.

    Positive values mean model_i has higher loss (worse).
    """
    return loss_df[model_i].values - loss_df[model_j].values


def bootstrap_statistic_range(
    loss_matrix: np.ndarray,
    n_bootstrap: int = 1000,
    block_length: Optional[int] = None,
    seed: int = 42
) -> np.ndarray:
    """
    Bootstrap the range statistic T_R = max(d_bar) - min(d_bar).

    Uses stationary block bootstrap to preserve serial correlation.

    Parameters
    ----------
    loss_matrix : np.ndarray
        T x M matrix of losses (rows=time, cols=models)
    n_bootstrap : int
        Number of bootstrap replications
    block_length : int, optional
        Expected block length for stationary bootstrap
    seed : int
        Random seed

    Returns
    -------
    boot_stats : np.ndarray
        Bootstrap distribution of range statistic
    """
    rng = np.random.RandomState(seed)
    T, M = loss_matrix.shape

    if block_length is None:
        # Rule of thumb: T^(1/3)
        block_length = max(1, int(T**(1/3)))

    # Probability of starting new block
    p = 1.0 / block_length

    # Center the losses
    mean_losses = np.nanmean(loss_matrix, axis=0)
    centered_losses = loss_matrix - mean_losses

    boot_stats = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        # Stationary block bootstrap
        boot_idx = np.zeros(T, dtype=int)
        t = 0

        while t < T:
            # Start new block with random starting point
            if t == 0 or rng.rand() < p:
                start = rng.randint(0, T)

            boot_idx[t] = start % T
            start += 1
            t += 1

        boot_losses = centered_losses[boot_idx, :]
        boot_means = np.nanmean(boot_losses, axis=0)

        # Range statistic
        boot_stats[b] = np.nanmax(boot_means) - np.nanmin(boot_means)

    return boot_stats


def bootstrap_statistic_tmax(
    loss_matrix: np.ndarray,
    n_bootstrap: int = 1000,
    block_length: Optional[int] = None,
    seed: int = 42
) -> np.ndarray:
    """
    Bootstrap the semi-quadratic statistic T_max.

    T_max = max_i { sqrt(T) * d_bar_i / se(d_i) }

    Parameters
    ----------
    loss_matrix : np.ndarray
        T x M matrix of losses
    n_bootstrap : int
        Number of bootstrap replications
    block_length : int, optional
        Expected block length
    seed : int
        Random seed

    Returns
    -------
    boot_stats : np.ndarray
        Bootstrap distribution of T_max statistic
    """
    rng = np.random.RandomState(seed)
    T, M = loss_matrix.shape

    if block_length is None:
        block_length = max(1, int(T**(1/3)))

    p = 1.0 / block_length

    # Compute relative losses for each model vs average
    mean_loss_per_t = np.nanmean(loss_matrix, axis=1, keepdims=True)
    relative_losses = loss_matrix - mean_loss_per_t

    # Center
    mean_relative = np.nanmean(relative_losses, axis=0)
    centered_relative = relative_losses - mean_relative

    boot_stats = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        # Stationary block bootstrap
        boot_idx = np.zeros(T, dtype=int)
        t = 0

        while t < T:
            if t == 0 or rng.rand() < p:
                start = rng.randint(0, T)
            boot_idx[t] = start % T
            start += 1
            t += 1

        boot_losses = centered_relative[boot_idx, :]
        boot_means = np.nanmean(boot_losses, axis=0)
        boot_vars = np.nanvar(boot_losses, axis=0, ddof=1)
        boot_vars = np.maximum(boot_vars, 1e-10)

        # t-statistics
        t_stats = np.sqrt(T) * boot_means / np.sqrt(boot_vars)
        boot_stats[b] = np.nanmax(t_stats)

    return boot_stats


def model_confidence_set(
    actual: np.ndarray,
    forecasts_dict: Dict[str, np.ndarray],
    loss_fn: str = "squared",
    alpha: float = 0.10,
    bootstrap_reps: int = 1000,
    statistic: str = "range",
    block_length: Optional[int] = None,
    seed: int = 42
) -> Dict:
    """
    Model Confidence Set procedure.

    Sequentially eliminates inferior models until we cannot reject
    the null hypothesis that all remaining models have equal predictive ability.

    Parameters
    ----------
    actual : np.ndarray
        Actual values
    forecasts_dict : dict
        {model_name: forecast_array}
    loss_fn : str
        Loss function ("squared" or "absolute")
    alpha : float
        Significance level (default 0.10 for 90% MCS)
    bootstrap_reps : int
        Number of bootstrap replications
    statistic : str
        "range" for T_R or "tmax" for semi-quadratic statistic
    block_length : int, optional
        Block length for bootstrap
    seed : int
        Random seed

    Returns
    -------
    result : dict
        Contains:
        - mcs: List of models in the confidence set
        - eliminated: List of eliminated models (in order)
        - p_values: List of p-values at each elimination step
        - mean_losses: Average loss for each model
        - alpha: Significance level used
    """
    # Compute losses
    loss_df = compute_loss_series(actual, forecasts_dict, loss_fn)

    # Remove observations with any NaN
    valid_mask = ~loss_df.isna().any(axis=1)
    loss_df = loss_df[valid_mask]
    T = len(loss_df)

    if T < 20:
        return {
            "error": "Insufficient observations for MCS",
            "n_obs": T,
            "mcs": list(forecasts_dict.keys()),
        }

    remaining = list(forecasts_dict.keys())
    eliminated = []
    p_values = []
    elimination_p_values = {}

    while len(remaining) > 1:
        # Get losses for remaining models
        loss_matrix = loss_df[remaining].values

        # Compute sample mean losses
        mean_losses = np.nanmean(loss_matrix, axis=0)

        # Compute test statistic
        if statistic == "range":
            t_stat = np.max(mean_losses) - np.min(mean_losses)
            boot_stats = bootstrap_statistic_range(
                loss_matrix, n_bootstrap=bootstrap_reps,
                block_length=block_length, seed=seed + len(eliminated)
            )
        else:  # tmax
            # Compute t-statistics
            vars_losses = np.nanvar(loss_matrix, axis=0, ddof=1)
            vars_losses = np.maximum(vars_losses, 1e-10)
            t_stats_sample = np.sqrt(T) * (mean_losses - np.mean(mean_losses)) / np.sqrt(vars_losses)
            t_stat = np.max(t_stats_sample)
            boot_stats = bootstrap_statistic_tmax(
                loss_matrix, n_bootstrap=bootstrap_reps,
                block_length=block_length, seed=seed + len(eliminated)
            )

        # Compute p-value
        p_value = np.mean(boot_stats >= t_stat)
        p_values.append(p_value)

        if p_value >= alpha:
            # Cannot reject null: remaining models form MCS
            break
        else:
            # Eliminate worst model (highest average loss)
            worst_idx = np.argmax(mean_losses)
            worst_model = remaining[worst_idx]
            elimination_p_values[worst_model] = p_value
            eliminated.append(worst_model)
            remaining.remove(worst_model)

    # Compute final statistics
    final_mean_losses = {
        model: np.nanmean(loss_df[model])
        for model in forecasts_dict.keys()
    }

    return {
        "mcs": remaining,
        "eliminated": eliminated,
        "p_values": p_values,
        "elimination_p_values": elimination_p_values,
        "mean_losses": final_mean_losses,
        "alpha": alpha,
        "n_obs": T,
        "statistic": statistic,
        "bootstrap_reps": bootstrap_reps,
    }


def mcs_summary_table(
    mcs_result: Dict,
    forecasts_dict: Dict[str, np.ndarray],
    actual: np.ndarray
) -> pd.DataFrame:
    """
    Create publication-quality summary table of MCS results.

    Parameters
    ----------
    mcs_result : dict
        Output from model_confidence_set()
    forecasts_dict : dict
        Original forecasts
    actual : np.ndarray
        Actual values

    Returns
    -------
    summary : pd.DataFrame
        Summary table sorted by performance
    """
    actual = np.asarray(actual, dtype=float)
    rows = []

    for model_name, forecast in forecasts_dict.items():
        forecast = np.asarray(forecast, dtype=float)

        # Compute metrics
        valid = ~(np.isnan(actual) | np.isnan(forecast))
        if valid.sum() < 1:
            continue

        errors = actual[valid] - forecast[valid]
        rmse = np.sqrt(np.mean(errors**2))
        mae = np.mean(np.abs(errors))
        mse = np.mean(errors**2)

        # MCS membership
        in_mcs = model_name in mcs_result["mcs"]

        # Elimination order
        if model_name in mcs_result["eliminated"]:
            elim_order = mcs_result["eliminated"].index(model_name) + 1
            elim_p_value = mcs_result["elimination_p_values"].get(model_name, np.nan)
        else:
            elim_order = None
            elim_p_value = None

        rows.append({
            "Model": model_name,
            "RMSE": rmse,
            "MAE": mae,
            "MSE": mse,
            "In_MCS": in_mcs,
            "MCS_Rank": None if in_mcs else elim_order,
            "Elim_p_value": elim_p_value,
        })

    df = pd.DataFrame(rows)

    # Sort by RMSE
    df = df.sort_values("RMSE").reset_index(drop=True)

    # Add rank
    df["Rank"] = range(1, len(df) + 1)

    return df


def format_mcs_table_for_paper(
    summary: pd.DataFrame,
    include_mse: bool = False
) -> pd.DataFrame:
    """
    Format MCS summary for publication.

    Parameters
    ----------
    summary : pd.DataFrame
        Output from mcs_summary_table()
    include_mse : bool
        Whether to include MSE column

    Returns
    -------
    formatted : pd.DataFrame
        Publication-ready table
    """
    cols = ["Rank", "Model", "RMSE", "MAE"]
    if include_mse:
        cols.append("MSE")

    formatted = summary[cols].copy()

    # Add MCS indicator
    formatted["In_MCS"] = summary["In_MCS"].apply(
        lambda x: "Yes" if x else "No"
    )

    # Round numbers
    for col in ["RMSE", "MAE", "MSE"]:
        if col in formatted.columns:
            formatted[col] = formatted[col].round(2)

    return formatted


def compute_mcs_pvalue(
    model_name: str,
    mcs_result: Dict
) -> float:
    """
    Compute the MCS p-value for a specific model.

    This is the smallest alpha at which the model would be included in the MCS.

    Parameters
    ----------
    model_name : str
        Model to check
    mcs_result : dict
        Output from model_confidence_set()

    Returns
    -------
    p_value : float
        MCS p-value for this model
    """
    if model_name in mcs_result["mcs"]:
        # Model is in MCS at current alpha
        # Its MCS p-value is at least alpha
        if mcs_result["p_values"]:
            return mcs_result["p_values"][-1]
        else:
            return 1.0

    # Model was eliminated
    if model_name in mcs_result["elimination_p_values"]:
        return mcs_result["elimination_p_values"][model_name]

    return 0.0


def mcs_with_pvalues(
    actual: np.ndarray,
    forecasts_dict: Dict[str, np.ndarray],
    loss_fn: str = "squared",
    bootstrap_reps: int = 1000,
    seed: int = 42
) -> pd.DataFrame:
    """
    Compute MCS p-values for all models.

    The MCS p-value is the smallest significance level at which
    the model would be included in the MCS.

    Parameters
    ----------
    actual : np.ndarray
        Actual values
    forecasts_dict : dict
        Model forecasts
    loss_fn : str
        Loss function
    bootstrap_reps : int
        Number of bootstrap replications
    seed : int
        Random seed

    Returns
    -------
    results : pd.DataFrame
        Table with MCS p-values for each model
    """
    # Run MCS at very low alpha to get full elimination ordering
    mcs_result = model_confidence_set(
        actual, forecasts_dict,
        loss_fn=loss_fn,
        alpha=0.001,  # Very low to eliminate all but one
        bootstrap_reps=bootstrap_reps,
        seed=seed
    )

    # Compute p-values
    rows = []
    for model_name in forecasts_dict.keys():
        forecast = np.asarray(forecasts_dict[model_name], dtype=float)
        actual_arr = np.asarray(actual, dtype=float)

        valid = ~(np.isnan(actual_arr) | np.isnan(forecast))
        errors = actual_arr[valid] - forecast[valid]
        rmse = np.sqrt(np.mean(errors**2))

        p_value = compute_mcs_pvalue(model_name, mcs_result)

        rows.append({
            "Model": model_name,
            "RMSE": rmse,
            "MCS_pvalue": p_value,
            "In_MCS_90": p_value >= 0.10,
            "In_MCS_75": p_value >= 0.25,
            "In_MCS_50": p_value >= 0.50,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("RMSE").reset_index(drop=True)

    return df


if __name__ == "__main__":
    # Simple test
    np.random.seed(42)
    n = 100

    actual = np.random.randn(n)
    forecasts = {
        "Good": actual + np.random.randn(n) * 0.3,
        "Medium": actual + np.random.randn(n) * 0.5,
        "Poor": actual + np.random.randn(n) * 1.0,
        "Bad": actual + np.random.randn(n) * 1.5,
    }

    result = model_confidence_set(
        actual, forecasts,
        alpha=0.10,
        bootstrap_reps=500,
        seed=42
    )

    print("MCS Result:")
    print(f"  Models in MCS: {result['mcs']}")
    print(f"  Eliminated: {result['eliminated']}")
    print(f"  P-values: {result['p_values']}")

    summary = mcs_summary_table(result, forecasts, actual)
    print("\nSummary Table:")
    print(summary)
