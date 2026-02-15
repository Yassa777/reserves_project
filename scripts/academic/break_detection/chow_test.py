"""
Chow Test for Structural Breaks at Known Dates

Implements the Chow (1960) test for parameter stability when the
break date is known a priori.

References:
    - Chow, G.C. (1960). Tests of Equality Between Sets of Coefficients
      in Two Linear Regressions. Econometrica, 28(3), 591-605.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Any, Union
import warnings


def chow_test(
    y: np.ndarray,
    X: Optional[np.ndarray],
    break_date_idx: int,
    significance_level: float = 0.05
) -> Dict[str, Any]:
    """
    Chow test for structural break at a known date.

    Tests whether the regression coefficients are stable across
    the proposed break date.

    H0: Coefficients are the same before and after break
    H1: Coefficients differ (structural break exists)

    Parameters
    ----------
    y : np.ndarray
        Dependent variable
    X : np.ndarray, optional
        Exogenous regressors (if None, uses constant-only model)
    break_date_idx : int
        Index of the proposed break date
    significance_level : float
        Significance level for hypothesis testing

    Returns
    -------
    dict
        Results containing:
        - f_statistic: Chow F-statistic
        - p_value: p-value for the test
        - reject_null: whether to reject null of no break
        - df1, df2: degrees of freedom
        - ssr_full: SSR from pooled regression
        - ssr_unrestricted: SSR from separate regressions
        - interpretation: text interpretation of result
    """
    n = len(y)

    # Validate break date
    if break_date_idx <= 0 or break_date_idx >= n:
        return {
            "f_statistic": np.nan,
            "p_value": np.nan,
            "reject_null": False,
            "interpretation": f"Invalid break index: {break_date_idx}",
            "valid": False
        }

    # Add constant to X
    if X is None:
        X_const = np.ones((n, 1))
    else:
        X_const = np.column_stack([np.ones(n), np.asarray(X)])

    k = X_const.shape[1]  # Number of parameters

    # Check for sufficient observations
    n1 = break_date_idx
    n2 = n - break_date_idx

    if n1 < k or n2 < k:
        return {
            "f_statistic": np.nan,
            "p_value": np.nan,
            "reject_null": False,
            "interpretation": f"Insufficient observations: n1={n1}, n2={n2}, k={k}",
            "valid": False
        }

    # Full sample (restricted) regression
    try:
        beta_full = np.linalg.lstsq(X_const, y, rcond=None)[0]
        y_hat_full = X_const @ beta_full
        ssr_full = np.sum((y - y_hat_full) ** 2)
    except np.linalg.LinAlgError:
        return {
            "f_statistic": np.nan,
            "p_value": np.nan,
            "reject_null": False,
            "interpretation": "Numerical error in full regression",
            "valid": False
        }

    # Pre-break (unrestricted) regression
    y1, X1 = y[:break_date_idx], X_const[:break_date_idx]
    try:
        beta1 = np.linalg.lstsq(X1, y1, rcond=None)[0]
        ssr1 = np.sum((y1 - X1 @ beta1) ** 2)
    except np.linalg.LinAlgError:
        return {
            "f_statistic": np.nan,
            "p_value": np.nan,
            "reject_null": False,
            "interpretation": "Numerical error in pre-break regression",
            "valid": False
        }

    # Post-break (unrestricted) regression
    y2, X2 = y[break_date_idx:], X_const[break_date_idx:]
    try:
        beta2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
        ssr2 = np.sum((y2 - X2 @ beta2) ** 2)
    except np.linalg.LinAlgError:
        return {
            "f_statistic": np.nan,
            "p_value": np.nan,
            "reject_null": False,
            "interpretation": "Numerical error in post-break regression",
            "valid": False
        }

    # Unrestricted SSR
    ssr_unrestricted = ssr1 + ssr2

    # Chow F-statistic
    df1 = k
    df2 = n - 2 * k

    if df2 <= 0:
        return {
            "f_statistic": np.nan,
            "p_value": np.nan,
            "reject_null": False,
            "interpretation": f"Insufficient degrees of freedom: df2={df2}",
            "valid": False
        }

    if ssr_unrestricted <= 0:
        ssr_unrestricted = 1e-10

    f_stat = ((ssr_full - ssr_unrestricted) / df1) / (ssr_unrestricted / df2)
    p_value = 1 - stats.f.cdf(f_stat, df1, df2)

    # Interpretation
    reject_null = p_value < significance_level
    if reject_null:
        interpretation = f"Structural break detected at index {break_date_idx} (p={p_value:.4f})"
    else:
        interpretation = f"No structural break at index {break_date_idx} (p={p_value:.4f})"

    return {
        "f_statistic": float(f_stat),
        "p_value": float(p_value),
        "reject_null": reject_null,
        "df1": df1,
        "df2": df2,
        "ssr_full": float(ssr_full),
        "ssr_unrestricted": float(ssr_unrestricted),
        "n_pre_break": n1,
        "n_post_break": n2,
        "interpretation": interpretation,
        "valid": True,
        "significance_level": significance_level
    }


def chow_test_with_dates(
    series: pd.Series,
    break_date: Union[str, pd.Timestamp],
    X: Optional[pd.DataFrame] = None,
    significance_level: float = 0.05
) -> Dict[str, Any]:
    """
    Chow test with date-based break specification.

    Parameters
    ----------
    series : pd.Series
        Time series with DatetimeIndex
    break_date : str or pd.Timestamp
        Date of proposed break
    X : pd.DataFrame, optional
        Exogenous regressors with matching index
    significance_level : float
        Significance level

    Returns
    -------
    dict
        Chow test results with date information
    """
    # Convert break_date to Timestamp
    break_dt = pd.Timestamp(break_date)

    # Find index of break date
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series must have DatetimeIndex")

    # Find closest date if exact date not in index
    if break_dt not in series.index:
        idx_loc = series.index.get_indexer([break_dt], method='nearest')[0]
        if idx_loc < 0:
            raise ValueError(f"Break date {break_date} not found in series")
        break_date_idx = idx_loc
        actual_date = series.index[break_date_idx]
        warnings.warn(f"Exact date {break_date} not found, using nearest: {actual_date}")
    else:
        break_date_idx = series.index.get_loc(break_dt)
        actual_date = break_dt

    # Prepare X if provided
    if X is not None:
        X_array = X.values
    else:
        X_array = None

    # Run Chow test
    result = chow_test(
        y=series.values,
        X=X_array,
        break_date_idx=break_date_idx,
        significance_level=significance_level
    )

    # Add date information
    result["break_date"] = actual_date.strftime('%Y-%m-%d')
    result["break_index"] = int(break_date_idx)
    if result["valid"]:
        result["pre_break_period"] = f"{series.index[0].strftime('%Y-%m-%d')} to {series.index[break_date_idx-1].strftime('%Y-%m-%d')}"
        result["post_break_period"] = f"{series.index[break_date_idx].strftime('%Y-%m-%d')} to {series.index[-1].strftime('%Y-%m-%d')}"

    return result


def multiple_chow_tests(
    series: pd.Series,
    break_dates: List[Union[str, pd.Timestamp]],
    event_names: Optional[List[str]] = None,
    X: Optional[pd.DataFrame] = None,
    significance_level: float = 0.05
) -> Dict[str, Any]:
    """
    Run Chow tests for multiple known break dates.

    Parameters
    ----------
    series : pd.Series
        Time series with DatetimeIndex
    break_dates : list
        List of proposed break dates
    event_names : list, optional
        Names of events corresponding to break dates
    X : pd.DataFrame, optional
        Exogenous regressors
    significance_level : float
        Significance level

    Returns
    -------
    dict
        Results for all Chow tests
    """
    if event_names is None:
        event_names = [f"Break_{i+1}" for i in range(len(break_dates))]

    results = {
        "variable": series.name if hasattr(series, 'name') and series.name else "series",
        "n_tests": len(break_dates),
        "significance_level": significance_level,
        "tests": [],
        "summary": {
            "n_significant": 0,
            "significant_breaks": []
        }
    }

    for break_date, event_name in zip(break_dates, event_names):
        try:
            test_result = chow_test_with_dates(
                series=series,
                break_date=break_date,
                X=X,
                significance_level=significance_level
            )
            test_result["event_name"] = event_name

            results["tests"].append(test_result)

            if test_result.get("reject_null", False):
                results["summary"]["n_significant"] += 1
                results["summary"]["significant_breaks"].append({
                    "event": event_name,
                    "date": test_result.get("break_date"),
                    "p_value": test_result.get("p_value"),
                    "f_statistic": test_result.get("f_statistic")
                })
        except Exception as e:
            results["tests"].append({
                "event_name": event_name,
                "break_date": str(break_date),
                "valid": False,
                "error": str(e)
            })

    return results


def predictive_chow_test(
    y: np.ndarray,
    X: Optional[np.ndarray],
    break_date_idx: int,
    significance_level: float = 0.05
) -> Dict[str, Any]:
    """
    Predictive (forecast) Chow test.

    Tests whether the model estimated on pre-break data can
    accurately predict post-break observations.

    Parameters
    ----------
    y : np.ndarray
        Dependent variable
    X : np.ndarray, optional
        Exogenous regressors
    break_date_idx : int
        Index of proposed break
    significance_level : float
        Significance level

    Returns
    -------
    dict
        Predictive Chow test results
    """
    n = len(y)

    if X is None:
        X_const = np.ones((n, 1))
    else:
        X_const = np.column_stack([np.ones(n), np.asarray(X)])

    k = X_const.shape[1]
    n1 = break_date_idx
    n2 = n - break_date_idx

    if n1 < k or n2 < 1:
        return {
            "f_statistic": np.nan,
            "p_value": np.nan,
            "reject_null": False,
            "interpretation": "Insufficient observations",
            "valid": False
        }

    # Estimate on pre-break sample
    y1, X1 = y[:break_date_idx], X_const[:break_date_idx]
    try:
        beta1 = np.linalg.lstsq(X1, y1, rcond=None)[0]
        ssr1 = np.sum((y1 - X1 @ beta1) ** 2)
    except np.linalg.LinAlgError:
        return {
            "f_statistic": np.nan,
            "p_value": np.nan,
            "reject_null": False,
            "interpretation": "Numerical error",
            "valid": False
        }

    # Forecast post-break
    y2, X2 = y[break_date_idx:], X_const[break_date_idx:]
    y2_forecast = X2 @ beta1
    forecast_errors = y2 - y2_forecast
    ssr_forecast = np.sum(forecast_errors ** 2)

    # Predictive F-statistic
    df1 = n2
    df2 = n1 - k

    if df2 <= 0:
        return {
            "f_statistic": np.nan,
            "p_value": np.nan,
            "reject_null": False,
            "interpretation": "Insufficient degrees of freedom",
            "valid": False
        }

    s2 = ssr1 / df2
    f_stat = (ssr_forecast / n2) / s2 if s2 > 0 else np.inf
    p_value = 1 - stats.f.cdf(f_stat, df1, df2)

    reject_null = p_value < significance_level

    return {
        "f_statistic": float(f_stat),
        "p_value": float(p_value),
        "reject_null": reject_null,
        "df1": df1,
        "df2": df2,
        "ssr_estimation": float(ssr1),
        "ssr_forecast": float(ssr_forecast),
        "mean_forecast_error": float(np.mean(forecast_errors)),
        "rmse_forecast": float(np.sqrt(np.mean(forecast_errors ** 2))),
        "interpretation": f"Predictive failure detected" if reject_null else "No predictive failure",
        "valid": True
    }


def qlr_test(
    y: np.ndarray,
    X: Optional[np.ndarray] = None,
    trimming: float = 0.15,
    significance_level: float = 0.05
) -> Dict[str, Any]:
    """
    Quandt Likelihood Ratio (QLR) test for unknown break date.

    Also known as Andrews-Quandt or sup-Wald test. Computes Chow
    statistics for all possible break dates and takes the supremum.

    Parameters
    ----------
    y : np.ndarray
        Dependent variable
    X : np.ndarray, optional
        Exogenous regressors
    trimming : float
        Fraction of sample to trim from each end
    significance_level : float
        Significance level

    Returns
    -------
    dict
        QLR test results with supremum statistic and break date
    """
    n = len(y)
    trim = int(n * trimming)

    if X is None:
        X_const = np.ones((n, 1))
    else:
        X_const = np.column_stack([np.ones(n), np.asarray(X)])

    k = X_const.shape[1]

    # Compute Chow statistics for all possible break dates
    f_stats = []
    break_indices = []

    for t in range(trim, n - trim):
        result = chow_test(y, X_const[:, 1:] if X is not None else None, t)
        if result.get("valid", False):
            f_stats.append(result["f_statistic"])
            break_indices.append(t)

    if len(f_stats) == 0:
        return {
            "sup_f_statistic": np.nan,
            "optimal_break_idx": None,
            "p_value": np.nan,
            "reject_null": False,
            "valid": False
        }

    # Supremum F-statistic
    sup_f = max(f_stats)
    optimal_idx = break_indices[np.argmax(f_stats)]

    # Critical values from Andrews (1993) tables (approximate)
    # These depend on k and trimming fraction
    # Using approximation for common cases
    critical_values = {
        0.10: 7.04,
        0.05: 8.85,
        0.01: 12.29
    }
    critical_value = critical_values.get(significance_level, 8.85)

    # Approximate p-value using exponential approximation
    # This is a rough approximation; exact p-values require simulation
    p_value = np.exp(-0.5 * (sup_f - 5)) if sup_f > 5 else 1.0
    p_value = min(max(p_value, 0.0), 1.0)

    reject_null = sup_f > critical_value

    return {
        "sup_f_statistic": float(sup_f),
        "optimal_break_idx": int(optimal_idx),
        "all_f_statistics": [float(f) for f in f_stats],
        "all_break_indices": break_indices,
        "critical_value": critical_value,
        "p_value_approx": float(p_value),
        "reject_null": reject_null,
        "trimming": trimming,
        "interpretation": f"Break detected at index {optimal_idx}" if reject_null else "No break detected",
        "valid": True
    }
