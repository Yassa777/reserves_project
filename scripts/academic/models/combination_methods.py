"""
Forecast Combination Methods

This module implements static forecast combination methods following the
academic literature (Bates & Granger 1969, Granger & Ramanathan 1984,
Timmermann 2006).

Methods implemented:
1. Equal weights (simple average)
2. Inverse MSE weights
3. Granger-Ramanathan regression (3 variants)
4. Trimmed mean (robust)
5. Median combination (robust)
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, Tuple, Optional


def equal_weight_combination(forecasts: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Simple average of all model forecasts.

    Parameters
    ----------
    forecasts : dict
        {model_name: forecast_array} for each model

    Returns
    -------
    combined : np.ndarray
        Equal-weighted average forecast

    References
    ----------
    Bates, J.M. & Granger, C.W.J. (1969). The Combination of Forecasts.
    """
    forecast_matrix = np.column_stack(list(forecasts.values()))
    return np.mean(forecast_matrix, axis=1)


def mse_weight_combination(
    forecasts: Dict[str, np.ndarray],
    actuals: np.ndarray,
    train_end_idx: int
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Weight inversely proportional to historical MSE.

    Parameters
    ----------
    forecasts : dict
        {model_name: forecast_array}
    actuals : np.ndarray
        Actual values
    train_end_idx : int
        Index separating training from evaluation

    Returns
    -------
    combined : np.ndarray
        MSE-weighted forecast
    weights : dict
        Optimal weights for each model
    """
    # Compute MSE on training period
    mse = {}
    for name, fc in forecasts.items():
        errors = fc[:train_end_idx] - actuals[:train_end_idx]
        # Handle NaN values
        valid_mask = ~np.isnan(errors)
        if valid_mask.sum() > 0:
            mse[name] = np.mean(errors[valid_mask] ** 2)
        else:
            mse[name] = np.inf

    # Handle zero MSE (perfect forecast) or very small values
    min_mse = 1e-10
    for name in mse:
        if mse[name] < min_mse:
            mse[name] = min_mse

    # Inverse MSE weights (normalized)
    inv_mse = {name: 1.0 / m for name, m in mse.items() if m != np.inf}

    if len(inv_mse) == 0:
        # Fallback to equal weights if all MSEs are infinite
        n = len(forecasts)
        weights = {name: 1.0 / n for name in forecasts.keys()}
    else:
        total = sum(inv_mse.values())
        weights = {name: w / total for name, w in inv_mse.items()}
        # Models with infinite MSE get zero weight
        for name in forecasts.keys():
            if name not in weights:
                weights[name] = 0.0

    # Apply weights
    combined = np.zeros(len(actuals))
    for name, fc in forecasts.items():
        combined += weights[name] * fc

    return combined, weights


def granger_ramanathan_combination(
    forecasts: Dict[str, np.ndarray],
    actuals: np.ndarray,
    train_end_idx: int,
    constraint: str = "none"
) -> Tuple[np.ndarray, Dict[str, float], float]:
    """
    Optimal combination via regression (Granger & Ramanathan, 1984).

    Three variants:
    - "none": Unconstrained OLS (allows bias correction via intercept)
    - "sum_to_one": Weights sum to 1 (no intercept)
    - "convex": Weights sum to 1 and are non-negative

    Parameters
    ----------
    forecasts : dict
        {model_name: forecast_array}
    actuals : np.ndarray
        Actual target values
    train_end_idx : int
        Training/evaluation split index
    constraint : str
        One of "none", "sum_to_one", or "convex"

    Returns
    -------
    combined : np.ndarray
        Combined forecast
    weights : dict
        Estimated weights for each model
    intercept : float
        Intercept (only non-zero for constraint="none")

    References
    ----------
    Granger, C.W.J. & Ramanathan, R. (1984). Improved Methods of
    Combining Forecasts. Journal of Forecasting.
    """
    # Prepare data
    model_names = list(forecasts.keys())
    n_models = len(model_names)

    X_train = np.column_stack([forecasts[m][:train_end_idx] for m in model_names])
    y_train = actuals[:train_end_idx]

    # Handle NaN values - use only complete cases
    valid_mask = ~np.isnan(y_train) & ~np.any(np.isnan(X_train), axis=1)
    X_train = X_train[valid_mask]
    y_train = y_train[valid_mask]

    if len(y_train) < n_models + 1:
        # Not enough data - fall back to equal weights
        n = len(model_names)
        weights = {m: 1.0 / n for m in model_names}
        intercept = 0.0
        X_full = np.column_stack([forecasts[m] for m in model_names])
        combined = np.mean(X_full, axis=1)
        return combined, weights, intercept

    if constraint == "none":
        # OLS with intercept
        X_with_const = np.column_stack([np.ones(len(y_train)), X_train])
        try:
            beta = np.linalg.lstsq(X_with_const, y_train, rcond=None)[0]
            intercept = beta[0]
            weights = {m: beta[i + 1] for i, m in enumerate(model_names)}
        except np.linalg.LinAlgError:
            # Singular matrix - fall back to equal weights
            n = len(model_names)
            weights = {m: 1.0 / n for m in model_names}
            intercept = 0.0

    elif constraint == "sum_to_one":
        # OLS without intercept, weights sum to 1
        def objective(w):
            pred = X_train @ w
            return np.sum((y_train - pred) ** 2)

        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        w0 = np.ones(n_models) / n_models

        result = minimize(
            objective, w0,
            constraints=constraints,
            method='SLSQP',
            options={'maxiter': 1000}
        )

        intercept = 0.0
        weights = {m: result.x[i] for i, m in enumerate(model_names)}

    elif constraint == "convex":
        # Non-negative weights summing to 1
        def objective(w):
            pred = X_train @ w
            return np.sum((y_train - pred) ** 2)

        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_models)]
        w0 = np.ones(n_models) / n_models

        result = minimize(
            objective, w0,
            constraints=constraints,
            bounds=bounds,
            method='SLSQP',
            options={'maxiter': 1000}
        )

        intercept = 0.0
        weights = {m: result.x[i] for i, m in enumerate(model_names)}

    else:
        raise ValueError(f"Unknown constraint: {constraint}. "
                        f"Use 'none', 'sum_to_one', or 'convex'.")

    # Apply to full sample
    X_full = np.column_stack([forecasts[m] for m in model_names])
    weight_array = np.array([weights[m] for m in model_names])
    combined = intercept + X_full @ weight_array

    return combined, weights, intercept


def trimmed_mean_combination(
    forecasts: Dict[str, np.ndarray],
    trim_pct: float = 0.1
) -> np.ndarray:
    """
    Trimmed mean - remove extreme forecasts before averaging.

    Robust to outlier models.

    Parameters
    ----------
    forecasts : dict
        {model_name: forecast_array}
    trim_pct : float
        Proportion of forecasts to trim from each end (default 0.1)

    Returns
    -------
    combined : np.ndarray
        Trimmed mean forecast
    """
    forecast_matrix = np.column_stack(list(forecasts.values()))
    n_models = forecast_matrix.shape[1]
    n_trim = max(1, int(n_models * trim_pct))

    # Need at least 1 model after trimming
    if n_models - 2 * n_trim < 1:
        # Fall back to median
        return np.median(forecast_matrix, axis=1)

    combined = np.zeros(len(forecast_matrix))

    for t in range(len(combined)):
        row = forecast_matrix[t]
        # Handle NaN values
        valid = row[~np.isnan(row)]
        if len(valid) <= 2 * n_trim:
            combined[t] = np.nanmean(row)
        else:
            sorted_fc = np.sort(valid)
            # Trim from both ends
            end_idx = len(sorted_fc) - n_trim if n_trim > 0 else len(sorted_fc)
            combined[t] = np.mean(sorted_fc[n_trim:end_idx])

    return combined


def median_combination(forecasts: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Median of all forecasts (robust to outliers).

    Parameters
    ----------
    forecasts : dict
        {model_name: forecast_array}

    Returns
    -------
    combined : np.ndarray
        Median forecast
    """
    forecast_matrix = np.column_stack(list(forecasts.values()))
    return np.nanmedian(forecast_matrix, axis=1)


def get_combination_weights(
    forecasts: Dict[str, np.ndarray],
    actuals: np.ndarray,
    train_end_idx: int,
    method: str
) -> Tuple[Optional[Dict[str, float]], float]:
    """
    Convenience function to get weights for any combination method.

    Parameters
    ----------
    forecasts : dict
        {model_name: forecast_array}
    actuals : np.ndarray
        Actual values
    train_end_idx : int
        Training period end index
    method : str
        Combination method name

    Returns
    -------
    weights : dict or None
        Weights for each model (None for trimmed/median)
    intercept : float
        Intercept term (0 for most methods)
    """
    if method == "equal":
        n = len(forecasts)
        weights = {m: 1.0 / n for m in forecasts.keys()}
        return weights, 0.0

    elif method == "mse":
        _, weights = mse_weight_combination(forecasts, actuals, train_end_idx)
        return weights, 0.0

    elif method == "gr_none":
        _, weights, intercept = granger_ramanathan_combination(
            forecasts, actuals, train_end_idx, constraint="none"
        )
        return weights, intercept

    elif method == "gr_sum":
        _, weights, intercept = granger_ramanathan_combination(
            forecasts, actuals, train_end_idx, constraint="sum_to_one"
        )
        return weights, intercept

    elif method == "gr_convex":
        _, weights, intercept = granger_ramanathan_combination(
            forecasts, actuals, train_end_idx, constraint="convex"
        )
        return weights, intercept

    elif method in ["trimmed", "median"]:
        return None, 0.0

    else:
        raise ValueError(f"Unknown method: {method}")
