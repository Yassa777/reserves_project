"""
CUSUM and CUSUMSQ Tests for Parameter Stability

Implements the Brown, Durbin, and Evans (1975) CUSUM and CUSUM-of-squares
tests for detecting structural instability in regression coefficients.

References:
    - Brown, R.L., Durbin, J., & Evans, J.M. (1975). Techniques for Testing
      the Constancy of Regression Relationships Over Time. JRSS-B, 37, 149-192.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Any, Tuple
import warnings


def compute_recursive_residuals(
    y: np.ndarray,
    X: Optional[np.ndarray] = None,
    start_obs: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute recursive residuals for CUSUM test.

    Recursive residuals are one-step-ahead forecast errors from
    models estimated on progressively larger samples.

    Parameters
    ----------
    y : np.ndarray
        Dependent variable
    X : np.ndarray, optional
        Exogenous regressors
    start_obs : int, optional
        First observation for recursive estimation (default: k+1)

    Returns
    -------
    tuple
        (recursive_residuals, indices) where indices are the observation
        numbers corresponding to each residual
    """
    n = len(y)

    if X is None:
        X_const = np.ones((n, 1))
    else:
        X_const = np.column_stack([np.ones(n), np.asarray(X)])

    k = X_const.shape[1]

    # Start after minimum observations for estimation
    if start_obs is None:
        start_obs = k + 1

    if start_obs >= n:
        raise ValueError(f"Insufficient observations: need at least {start_obs + 1}")

    recursive_residuals = []
    indices = []

    for t in range(start_obs, n):
        # Estimate on observations 0 to t-1
        y_est = y[:t]
        X_est = X_const[:t]

        try:
            beta = np.linalg.lstsq(X_est, y_est, rcond=None)[0]

            # One-step-ahead forecast
            y_forecast = X_const[t] @ beta
            residual = y[t] - y_forecast

            # Standardize by forecast variance
            # f_t = 1 + x_t' (X'X)^{-1} x_t
            try:
                XtX_inv = np.linalg.inv(X_est.T @ X_est)
                f_t = 1 + X_const[t] @ XtX_inv @ X_const[t]
                sigma2 = np.sum((y_est - X_est @ beta) ** 2) / (t - k)
                if sigma2 > 0 and f_t > 0:
                    std_residual = residual / np.sqrt(sigma2 * f_t)
                else:
                    std_residual = residual
            except np.linalg.LinAlgError:
                std_residual = residual

            recursive_residuals.append(std_residual)
            indices.append(t)
        except np.linalg.LinAlgError:
            continue

    return np.array(recursive_residuals), np.array(indices)


def cusum_test(
    y: np.ndarray,
    X: Optional[np.ndarray] = None,
    significance_level: float = 0.05
) -> Dict[str, Any]:
    """
    CUSUM test for parameter stability.

    The CUSUM statistic is the cumulative sum of recursive residuals.
    Under the null of parameter stability, the CUSUM should stay within
    critical boundaries derived from Brownian motion.

    Parameters
    ----------
    y : np.ndarray
        Dependent variable
    X : np.ndarray, optional
        Exogenous regressors
    significance_level : float
        Significance level for critical bounds

    Returns
    -------
    dict
        Results containing:
        - cusum: CUSUM path
        - upper_bound: upper critical boundary
        - lower_bound: lower critical boundary
        - stable: whether null of stability is not rejected
        - max_deviation: maximum deviation from zero
        - first_crossing_idx: first index where bounds are crossed
        - crossing_date_idx: index in original series where crossing occurs
    """
    y_clean = np.asarray(y).flatten()

    # Handle NaN values
    valid_mask = ~np.isnan(y_clean)
    if not np.all(valid_mask):
        y_clean = pd.Series(y_clean).interpolate().fillna(method='bfill').fillna(method='ffill').values

    # Compute recursive residuals
    try:
        rec_resid, indices = compute_recursive_residuals(y_clean, X)
    except ValueError as e:
        return {
            "cusum": np.array([]),
            "upper_bound": np.array([]),
            "lower_bound": np.array([]),
            "stable": None,
            "error": str(e),
            "valid": False
        }

    if len(rec_resid) == 0:
        return {
            "cusum": np.array([]),
            "upper_bound": np.array([]),
            "lower_bound": np.array([]),
            "stable": None,
            "error": "No recursive residuals computed",
            "valid": False
        }

    n_resid = len(rec_resid)

    # Estimate standard deviation
    sigma = np.std(rec_resid, ddof=1) if n_resid > 1 else 1.0
    if sigma == 0:
        sigma = 1.0

    # Cumulative sum of standardized recursive residuals
    cusum = np.cumsum(rec_resid) / sigma

    # Critical boundaries from Brownian bridge approximation
    # B(t) = a + 2*a*t where a depends on significance level
    critical_values = {
        0.01: 1.143,
        0.05: 0.948,
        0.10: 0.850
    }
    a = critical_values.get(significance_level, 0.948)

    # Time proportion
    t = np.arange(1, n_resid + 1) / n_resid

    # Critical bounds (two-sided)
    upper_bound = a * np.sqrt(n_resid) * (1 + 2 * t)
    lower_bound = -a * np.sqrt(n_resid) * (1 + 2 * t)

    # Alternative: straight-line boundaries
    # These are more commonly used in practice
    upper_bound_linear = a * np.sqrt(n_resid) + 2 * a * np.arange(1, n_resid + 1) / np.sqrt(n_resid)
    lower_bound_linear = -upper_bound_linear

    # Check stability
    exceeds_curved = np.any((cusum > upper_bound) | (cusum < lower_bound))
    exceeds_linear = np.any((cusum > upper_bound_linear) | (cusum < lower_bound_linear))

    # Use linear bounds (more standard)
    exceeds_bounds = exceeds_linear
    upper_bound_used = upper_bound_linear
    lower_bound_used = lower_bound_linear

    # Find first crossing
    if exceeds_bounds:
        crossing_mask = (cusum > upper_bound_used) | (cusum < lower_bound_used)
        first_crossing = np.argmax(crossing_mask)
        first_crossing_idx = int(first_crossing)
        crossing_date_idx = int(indices[first_crossing])
    else:
        first_crossing_idx = None
        crossing_date_idx = None

    # Maximum deviation from zero
    max_deviation = float(np.max(np.abs(cusum)))
    max_deviation_idx = int(np.argmax(np.abs(cusum)))

    return {
        "cusum": cusum.tolist(),
        "upper_bound": upper_bound_used.tolist(),
        "lower_bound": lower_bound_used.tolist(),
        "indices": indices.tolist(),
        "stable": not exceeds_bounds,
        "exceeds_bounds": exceeds_bounds,
        "max_deviation": max_deviation,
        "max_deviation_idx": max_deviation_idx,
        "max_deviation_original_idx": int(indices[max_deviation_idx]) if len(indices) > 0 else None,
        "first_crossing_idx": first_crossing_idx,
        "crossing_date_idx": crossing_date_idx,
        "significance_level": significance_level,
        "critical_value": a,
        "n_recursive_residuals": n_resid,
        "interpretation": "Parameter stability cannot be rejected" if not exceeds_bounds else "Parameter instability detected",
        "valid": True
    }


def cusumsq_test(
    y: np.ndarray,
    X: Optional[np.ndarray] = None,
    significance_level: float = 0.05
) -> Dict[str, Any]:
    """
    CUSUM of Squares test for variance stability.

    Tests for stability of error variance over time using cumulative
    sum of squared recursive residuals.

    Parameters
    ----------
    y : np.ndarray
        Dependent variable
    X : np.ndarray, optional
        Exogenous regressors
    significance_level : float
        Significance level for critical bounds

    Returns
    -------
    dict
        Results containing CUSUMSQ path and stability assessment
    """
    y_clean = np.asarray(y).flatten()

    # Handle NaN
    valid_mask = ~np.isnan(y_clean)
    if not np.all(valid_mask):
        y_clean = pd.Series(y_clean).interpolate().fillna(method='bfill').fillna(method='ffill').values

    # Compute recursive residuals
    try:
        rec_resid, indices = compute_recursive_residuals(y_clean, X)
    except ValueError as e:
        return {
            "cusumsq": np.array([]),
            "upper_bound": np.array([]),
            "lower_bound": np.array([]),
            "stable": None,
            "error": str(e),
            "valid": False
        }

    if len(rec_resid) == 0:
        return {
            "cusumsq": np.array([]),
            "upper_bound": np.array([]),
            "lower_bound": np.array([]),
            "stable": None,
            "error": "No recursive residuals computed",
            "valid": False
        }

    n_resid = len(rec_resid)

    # Squared residuals
    rec_resid_sq = rec_resid ** 2

    # Cumulative sum of squares
    cumsum_sq = np.cumsum(rec_resid_sq)
    total_sq = cumsum_sq[-1]

    # Normalize to [0, 1]
    if total_sq > 0:
        cusumsq = cumsum_sq / total_sq
    else:
        cusumsq = np.zeros(n_resid)

    # Time proportion (expected value under null)
    t = np.arange(1, n_resid + 1) / n_resid

    # Critical bounds for CUSUMSQ (based on Brownian bridge)
    # c_alpha values from Brown, Durbin, Evans (1975)
    c_values = {
        0.01: 1.63,
        0.05: 1.36,
        0.10: 1.22
    }
    c = c_values.get(significance_level, 1.36)

    # Bounds: t +/- c * sqrt(t(1-t)/n) approximately
    # Simplified straight-line bounds often used
    upper_bound = t + c * np.sqrt(n_resid) / n_resid
    lower_bound = t - c * np.sqrt(n_resid) / n_resid

    # Ensure bounds are in [0, 1]
    upper_bound = np.minimum(upper_bound, 1.0)
    lower_bound = np.maximum(lower_bound, 0.0)

    # Alternative: parallel line bounds (more common in practice)
    c_parallel = {
        0.01: 0.184,
        0.05: 0.146,
        0.10: 0.122
    }
    c_alt = c_parallel.get(significance_level, 0.146)

    upper_bound_parallel = t + c_alt
    lower_bound_parallel = t - c_alt
    upper_bound_parallel = np.minimum(upper_bound_parallel, 1.0)
    lower_bound_parallel = np.maximum(lower_bound_parallel, 0.0)

    # Use parallel bounds
    exceeds_bounds = np.any((cusumsq > upper_bound_parallel) | (cusumsq < lower_bound_parallel))

    # Find first crossing
    if exceeds_bounds:
        crossing_mask = (cusumsq > upper_bound_parallel) | (cusumsq < lower_bound_parallel)
        first_crossing = np.argmax(crossing_mask)
        first_crossing_idx = int(first_crossing)
        crossing_date_idx = int(indices[first_crossing])
    else:
        first_crossing_idx = None
        crossing_date_idx = None

    # Maximum deviation from expected value (45-degree line)
    deviations = cusumsq - t
    max_deviation = float(np.max(np.abs(deviations)))
    max_deviation_idx = int(np.argmax(np.abs(deviations)))

    return {
        "cusumsq": cusumsq.tolist(),
        "expected_line": t.tolist(),
        "upper_bound": upper_bound_parallel.tolist(),
        "lower_bound": lower_bound_parallel.tolist(),
        "indices": indices.tolist(),
        "stable": not exceeds_bounds,
        "exceeds_bounds": exceeds_bounds,
        "max_deviation": max_deviation,
        "max_deviation_idx": max_deviation_idx,
        "max_deviation_original_idx": int(indices[max_deviation_idx]) if len(indices) > 0 else None,
        "first_crossing_idx": first_crossing_idx,
        "crossing_date_idx": crossing_date_idx,
        "significance_level": significance_level,
        "critical_value": c_alt,
        "n_recursive_residuals": n_resid,
        "interpretation": "Variance stability cannot be rejected" if not exceeds_bounds else "Variance instability detected",
        "valid": True
    }


def cusum_test_with_dates(
    series: pd.Series,
    X: Optional[pd.DataFrame] = None,
    significance_level: float = 0.05
) -> Dict[str, Any]:
    """
    CUSUM test with date information.

    Parameters
    ----------
    series : pd.Series
        Time series with DatetimeIndex
    X : pd.DataFrame, optional
        Exogenous regressors
    significance_level : float
        Significance level

    Returns
    -------
    dict
        CUSUM results with date annotations
    """
    result = cusum_test(
        y=series.values,
        X=X.values if X is not None else None,
        significance_level=significance_level
    )

    if result.get("valid", False) and hasattr(series, 'index'):
        dates = series.index

        if result.get("crossing_date_idx") is not None:
            idx = result["crossing_date_idx"]
            if idx < len(dates):
                result["crossing_date"] = dates[idx].strftime('%Y-%m-%d')

        if result.get("max_deviation_original_idx") is not None:
            idx = result["max_deviation_original_idx"]
            if idx < len(dates):
                result["max_deviation_date"] = dates[idx].strftime('%Y-%m-%d')

    result["variable"] = series.name if hasattr(series, 'name') and series.name else "series"
    return result


def cusumsq_test_with_dates(
    series: pd.Series,
    X: Optional[pd.DataFrame] = None,
    significance_level: float = 0.05
) -> Dict[str, Any]:
    """
    CUSUMSQ test with date information.

    Parameters
    ----------
    series : pd.Series
        Time series with DatetimeIndex
    X : pd.DataFrame, optional
        Exogenous regressors
    significance_level : float
        Significance level

    Returns
    -------
    dict
        CUSUMSQ results with date annotations
    """
    result = cusumsq_test(
        y=series.values,
        X=X.values if X is not None else None,
        significance_level=significance_level
    )

    if result.get("valid", False) and hasattr(series, 'index'):
        dates = series.index

        if result.get("crossing_date_idx") is not None:
            idx = result["crossing_date_idx"]
            if idx < len(dates):
                result["crossing_date"] = dates[idx].strftime('%Y-%m-%d')

        if result.get("max_deviation_original_idx") is not None:
            idx = result["max_deviation_original_idx"]
            if idx < len(dates):
                result["max_deviation_date"] = dates[idx].strftime('%Y-%m-%d')

    result["variable"] = series.name if hasattr(series, 'name') and series.name else "series"
    return result


def combined_stability_test(
    series: pd.Series,
    X: Optional[pd.DataFrame] = None,
    significance_level: float = 0.05
) -> Dict[str, Any]:
    """
    Run both CUSUM and CUSUMSQ tests and combine results.

    Parameters
    ----------
    series : pd.Series
        Time series with DatetimeIndex
    X : pd.DataFrame, optional
        Exogenous regressors
    significance_level : float
        Significance level

    Returns
    -------
    dict
        Combined results from both tests
    """
    cusum_result = cusum_test_with_dates(series, X, significance_level)
    cusumsq_result = cusumsq_test_with_dates(series, X, significance_level)

    # Overall stability assessment
    cusum_stable = cusum_result.get("stable", None)
    cusumsq_stable = cusumsq_result.get("stable", None)

    if cusum_stable is None or cusumsq_stable is None:
        overall_stable = None
        overall_interpretation = "Tests could not be completed"
    elif cusum_stable and cusumsq_stable:
        overall_stable = True
        overall_interpretation = "Both mean and variance appear stable"
    elif not cusum_stable and not cusumsq_stable:
        overall_stable = False
        overall_interpretation = "Both mean and variance show instability"
    elif not cusum_stable:
        overall_stable = False
        overall_interpretation = "Mean instability detected (CUSUM rejects)"
    else:
        overall_stable = False
        overall_interpretation = "Variance instability detected (CUSUMSQ rejects)"

    return {
        "variable": series.name if hasattr(series, 'name') and series.name else "series",
        "cusum": cusum_result,
        "cusumsq": cusumsq_result,
        "overall_stable": overall_stable,
        "overall_interpretation": overall_interpretation,
        "significance_level": significance_level
    }
