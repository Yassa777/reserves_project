"""
Bai-Perron Multiple Structural Break Detection

Implements the Bai-Perron (1998, 2003) methodology for detecting multiple
structural breaks with unknown break dates using the ruptures library.

References:
    - Bai, J. & Perron, P. (1998). Estimating and Testing Linear Models with
      Multiple Structural Changes. Econometrica, 66(1), 47-78.
    - Bai, J. & Perron, P. (2003). Computation and Analysis of Multiple
      Structural Change Models. Journal of Applied Econometrics, 18(1), 1-22.
"""

import numpy as np
import pandas as pd
import ruptures as rpt
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
import warnings


def compute_bic(
    y: np.ndarray,
    X: Optional[np.ndarray],
    break_points: List[int],
    model_type: str = "l2"
) -> float:
    """
    Compute BIC (Bayesian Information Criterion) for a given segmentation.

    Parameters
    ----------
    y : np.ndarray
        Target series
    X : np.ndarray, optional
        Exogenous regressors
    break_points : list
        List of break point indices (including end point)
    model_type : str
        Model type for residual computation

    Returns
    -------
    float
        BIC value (lower is better)
    """
    n = len(y)
    segments = [0] + list(break_points)

    ssr = 0.0
    k = 0  # number of parameters

    for i in range(len(segments) - 1):
        start, end = segments[i], segments[i + 1]
        y_seg = y[start:end]

        if len(y_seg) == 0:
            continue

        if X is not None:
            X_seg = X[start:end]
            # OLS regression with constant
            X_with_const = np.column_stack([np.ones(len(y_seg)), X_seg])
            try:
                beta = np.linalg.lstsq(X_with_const, y_seg, rcond=None)[0]
                resid = y_seg - X_with_const @ beta
                k += X_with_const.shape[1]
            except np.linalg.LinAlgError:
                resid = y_seg - np.mean(y_seg)
                k += 1
        else:
            # Mean model
            resid = y_seg - np.mean(y_seg)
            k += 1

        ssr += np.sum(resid ** 2)

    # Avoid log of zero
    if ssr <= 0:
        ssr = 1e-10

    # BIC formula: n * log(SSR/n) + k * log(n)
    bic = n * np.log(ssr / n) + k * np.log(n)
    return bic


def compute_lwz(
    y: np.ndarray,
    X: Optional[np.ndarray],
    break_points: List[int]
) -> float:
    """
    Compute LWZ (Liu-Wu-Zidek) criterion for break selection.

    The LWZ criterion is more conservative than BIC and tends to select
    fewer breaks.

    Parameters
    ----------
    y : np.ndarray
        Target series
    X : np.ndarray, optional
        Exogenous regressors
    break_points : list
        List of break point indices

    Returns
    -------
    float
        LWZ value (lower is better)
    """
    n = len(y)
    segments = [0] + list(break_points)
    n_breaks = len(break_points) - 1  # Exclude end point

    ssr = 0.0
    k = 0

    for i in range(len(segments) - 1):
        start, end = segments[i], segments[i + 1]
        y_seg = y[start:end]

        if len(y_seg) == 0:
            continue

        if X is not None:
            X_seg = X[start:end]
            X_with_const = np.column_stack([np.ones(len(y_seg)), X_seg])
            try:
                beta = np.linalg.lstsq(X_with_const, y_seg, rcond=None)[0]
                resid = y_seg - X_with_const @ beta
                k += X_with_const.shape[1]
            except np.linalg.LinAlgError:
                resid = y_seg - np.mean(y_seg)
                k += 1
        else:
            resid = y_seg - np.mean(y_seg)
            k += 1

        ssr += np.sum(resid ** 2)

    if ssr <= 0:
        ssr = 1e-10

    # LWZ formula with stronger penalty
    c = 0.299  # Asymptotic constant
    lwz = n * np.log(ssr / n) + c * k * (np.log(n)) ** 2.1
    return lwz


def bai_perron_test(
    y: np.ndarray,
    X: Optional[np.ndarray] = None,
    max_breaks: int = 5,
    min_segment_length: int = 24,
    significance_level: float = 0.05,
    trimming_fraction: float = 0.15,
    selection_criterion: str = "bic"
) -> Dict[str, Any]:
    """
    Bai-Perron structural break detection using dynamic programming.

    Uses the ruptures library for efficient computation of optimal
    segmentation via dynamic programming.

    Parameters
    ----------
    y : np.ndarray
        Target series (e.g., reserves)
    X : np.ndarray, optional
        Exogenous regressors (not used in current implementation)
    max_breaks : int
        Maximum number of breaks to consider
    min_segment_length : int
        Minimum observations between breaks (h parameter in Bai-Perron)
    significance_level : float
        Significance level for hypothesis testing
    trimming_fraction : float
        Fraction of sample to trim from each end (epsilon in Bai-Perron)
    selection_criterion : str
        Criterion for selecting number of breaks: "bic" or "lwz"

    Returns
    -------
    dict
        Results containing:
        - n_breaks: optimal number of breaks
        - break_indices: list of break indices
        - break_dates: list of break dates (if dates provided)
        - bic_values: BIC for each number of breaks
        - lwz_values: LWZ for each number of breaks
        - optimal_bic: BIC at optimal number of breaks
        - segment_means: mean value in each segment
        - segment_stats: detailed statistics for each segment
    """
    # Clean data
    y_clean = np.asarray(y).flatten()

    # Handle NaN values
    valid_mask = ~np.isnan(y_clean)
    if not np.all(valid_mask):
        warnings.warn(f"Found {np.sum(~valid_mask)} NaN values. Using interpolation.")
        y_clean = pd.Series(y_clean).interpolate(method='linear').fillna(method='bfill').fillna(method='ffill').values

    n = len(y_clean)

    # Validate min_segment_length
    effective_min_seg = max(min_segment_length, int(n * trimming_fraction))

    # Ensure we don't exceed data limits
    max_possible_breaks = (n // effective_min_seg) - 1
    max_breaks = min(max_breaks, max_possible_breaks)

    if max_breaks <= 0:
        return {
            "n_breaks": 0,
            "break_indices": [],
            "bic_values": {0: compute_bic(y_clean, X, [n])},
            "lwz_values": {0: compute_lwz(y_clean, X, [n])},
            "optimal_bic": compute_bic(y_clean, X, [n]),
            "segment_means": [np.mean(y_clean)],
            "segment_stats": [_compute_segment_stats(y_clean, 0, n)]
        }

    # Prepare signal for ruptures
    signal = y_clean.reshape(-1, 1)

    # Use dynamic programming for optimal segmentation
    # l2 model assumes piecewise constant mean (appropriate for level shifts)
    algo = rpt.Dynp(model="l2", min_size=effective_min_seg, jump=1)
    algo.fit(signal)

    # Compute information criteria for each number of breaks
    bic_values = {}
    lwz_values = {}
    break_results = {}

    for n_bkps in range(0, max_breaks + 1):
        if n_bkps == 0:
            result = [n]
        else:
            try:
                result = algo.predict(n_bkps=n_bkps)
            except Exception as e:
                warnings.warn(f"Could not compute {n_bkps} breaks: {e}")
                continue

        bic_values[n_bkps] = compute_bic(y_clean, X, result)
        lwz_values[n_bkps] = compute_lwz(y_clean, X, result)
        break_results[n_bkps] = result

    # Select optimal number of breaks
    if selection_criterion == "lwz":
        criterion_values = lwz_values
    else:
        criterion_values = bic_values

    optimal_n = min(criterion_values, key=criterion_values.get)

    # Get break points for optimal model
    if optimal_n == 0:
        break_indices = []
        final_breaks = [n]
    else:
        final_breaks = break_results[optimal_n]
        break_indices = final_breaks[:-1]  # Exclude end point

    # Compute segment statistics
    segments = [0] + list(final_breaks)
    segment_means = []
    segment_stats = []

    for i in range(len(segments) - 1):
        start, end = segments[i], segments[i + 1]
        segment_data = y_clean[start:end]
        segment_means.append(float(np.mean(segment_data)))
        segment_stats.append(_compute_segment_stats(y_clean, start, end))

    return {
        "n_breaks": optimal_n,
        "break_indices": break_indices,
        "bic_values": bic_values,
        "lwz_values": lwz_values,
        "optimal_bic": bic_values.get(optimal_n, np.nan),
        "optimal_lwz": lwz_values.get(optimal_n, np.nan),
        "segment_means": segment_means,
        "segment_stats": segment_stats,
        "selection_criterion": selection_criterion,
        "min_segment_length": effective_min_seg,
        "trimming_fraction": trimming_fraction
    }


def _compute_segment_stats(y: np.ndarray, start: int, end: int) -> Dict[str, float]:
    """Compute statistics for a segment."""
    segment = y[start:end]
    return {
        "start_idx": int(start),
        "end_idx": int(end),
        "n_obs": int(len(segment)),
        "mean": float(np.mean(segment)),
        "std": float(np.std(segment, ddof=1)) if len(segment) > 1 else 0.0,
        "min": float(np.min(segment)),
        "max": float(np.max(segment)),
        "range": float(np.max(segment) - np.min(segment))
    }


def bai_perron_with_dates(
    series: pd.Series,
    max_breaks: int = 5,
    min_segment_length: int = 24,
    significance_level: float = 0.05,
    trimming_fraction: float = 0.15,
    selection_criterion: str = "bic"
) -> Dict[str, Any]:
    """
    Bai-Perron test with date index support.

    Parameters
    ----------
    series : pd.Series
        Time series with DatetimeIndex
    max_breaks : int
        Maximum number of breaks
    min_segment_length : int
        Minimum segment length in observations
    significance_level : float
        Significance level
    trimming_fraction : float
        Trimming fraction
    selection_criterion : str
        Selection criterion ("bic" or "lwz")

    Returns
    -------
    dict
        Results with break dates included
    """
    # Run basic test
    results = bai_perron_test(
        y=series.values,
        max_breaks=max_breaks,
        min_segment_length=min_segment_length,
        significance_level=significance_level,
        trimming_fraction=trimming_fraction,
        selection_criterion=selection_criterion
    )

    # Convert indices to dates
    if hasattr(series, 'index') and isinstance(series.index, pd.DatetimeIndex):
        dates = series.index
        break_dates = [dates[idx].strftime('%Y-%m-%d') for idx in results["break_indices"]]
        results["break_dates"] = break_dates

        # Add date ranges to segment stats
        segments = [0] + results["break_indices"] + [len(series)]
        for i, stats in enumerate(results["segment_stats"]):
            start_idx = segments[i]
            end_idx = segments[i + 1] - 1 if segments[i + 1] < len(dates) else segments[i + 1] - 1
            stats["start_date"] = dates[start_idx].strftime('%Y-%m-%d')
            stats["end_date"] = dates[min(end_idx, len(dates) - 1)].strftime('%Y-%m-%d')

    return results


def sequential_bai_perron(
    y: np.ndarray,
    max_breaks: int = 5,
    min_segment_length: int = 24,
    significance_level: float = 0.05
) -> Dict[str, Any]:
    """
    Sequential Bai-Perron test (sup-F tests).

    Tests sequentially for 0 vs 1 break, 1 vs 2 breaks, etc.
    This approach follows Bai-Perron (1998) sequential procedure.

    Parameters
    ----------
    y : np.ndarray
        Target series
    max_breaks : int
        Maximum number of breaks to test
    min_segment_length : int
        Minimum segment length
    significance_level : float
        Significance level for sup-F tests

    Returns
    -------
    dict
        Results with sequential test statistics
    """
    y_clean = np.asarray(y).flatten()
    valid_mask = ~np.isnan(y_clean)
    if not np.all(valid_mask):
        y_clean = pd.Series(y_clean).interpolate().fillna(method='bfill').fillna(method='ffill').values

    n = len(y_clean)
    signal = y_clean.reshape(-1, 1)

    # Use Pelt algorithm for efficient change point detection
    algo = rpt.Pelt(model="l2", min_size=min_segment_length, jump=1)
    algo.fit(signal)

    # Get optimal segmentation using penalty
    # Penalty calibrated for 5% significance
    pen = np.log(n) * 2  # BIC-like penalty

    try:
        result = algo.predict(pen=pen)
        break_indices = result[:-1]
    except Exception:
        break_indices = []

    # Compute F-statistics for each potential break
    f_statistics = []
    for n_bkps in range(1, min(max_breaks + 1, len(break_indices) + 2)):
        algo_dynp = rpt.Dynp(model="l2", min_size=min_segment_length, jump=1)
        algo_dynp.fit(signal)

        try:
            if n_bkps == 0:
                ssr_restricted = np.sum((y_clean - np.mean(y_clean)) ** 2)
            else:
                breaks = algo_dynp.predict(n_bkps=n_bkps)
                segments = [0] + list(breaks)
                ssr_unrestricted = 0
                for i in range(len(segments) - 1):
                    seg = y_clean[segments[i]:segments[i + 1]]
                    ssr_unrestricted += np.sum((seg - np.mean(seg)) ** 2)

            if n_bkps > 0:
                # Previous model SSR
                if n_bkps == 1:
                    ssr_restricted = np.sum((y_clean - np.mean(y_clean)) ** 2)
                else:
                    prev_breaks = algo_dynp.predict(n_bkps=n_bkps - 1)
                    prev_segments = [0] + list(prev_breaks)
                    ssr_restricted = 0
                    for i in range(len(prev_segments) - 1):
                        seg = y_clean[prev_segments[i]:prev_segments[i + 1]]
                        ssr_restricted += np.sum((seg - np.mean(seg)) ** 2)

                # F-statistic
                df1 = 1  # Change in number of parameters
                df2 = n - 2 * n_bkps
                if df2 > 0 and ssr_unrestricted > 0:
                    f_stat = ((ssr_restricted - ssr_unrestricted) / df1) / (ssr_unrestricted / df2)
                    p_value = 1 - stats.f.cdf(f_stat, df1, df2)
                    f_statistics.append({
                        "test": f"{n_bkps-1} vs {n_bkps} breaks",
                        "f_statistic": float(f_stat),
                        "p_value": float(p_value),
                        "reject_null": p_value < significance_level
                    })
        except Exception:
            continue

    return {
        "n_breaks": len(break_indices),
        "break_indices": break_indices,
        "sequential_tests": f_statistics,
        "penalty_used": pen
    }


def compute_confidence_intervals(
    y: np.ndarray,
    break_indices: List[int],
    confidence_level: float = 0.95,
    method: str = "bootstrap",
    n_bootstrap: int = 1000
) -> Dict[int, Tuple[int, int]]:
    """
    Compute confidence intervals for break dates.

    Parameters
    ----------
    y : np.ndarray
        Target series
    break_indices : list
        Detected break indices
    confidence_level : float
        Confidence level (e.g., 0.95 for 95% CI)
    method : str
        Method for CI computation: "bootstrap" or "asymptotic"
    n_bootstrap : int
        Number of bootstrap replications

    Returns
    -------
    dict
        Dictionary mapping break index to (lower, upper) CI bounds
    """
    if len(break_indices) == 0:
        return {}

    y_clean = np.asarray(y).flatten()
    n = len(y_clean)

    confidence_intervals = {}

    if method == "bootstrap":
        # Bootstrap confidence intervals
        alpha = 1 - confidence_level

        for break_idx in break_indices:
            bootstrap_breaks = []

            for _ in range(n_bootstrap):
                # Resample with replacement
                idx = np.random.choice(n, size=n, replace=True)
                y_boot = y_clean[idx]

                # Detect breaks in bootstrap sample
                signal = y_boot.reshape(-1, 1)
                algo = rpt.Dynp(model="l2", min_size=max(10, n // 10), jump=1)
                algo.fit(signal)

                try:
                    result = algo.predict(n_bkps=len(break_indices))
                    boot_breaks = result[:-1]

                    # Find closest break to original
                    if len(boot_breaks) > 0:
                        distances = [abs(b - break_idx) for b in boot_breaks]
                        closest = boot_breaks[np.argmin(distances)]
                        bootstrap_breaks.append(closest)
                except Exception:
                    continue

            if len(bootstrap_breaks) > 0:
                lower = int(np.percentile(bootstrap_breaks, alpha / 2 * 100))
                upper = int(np.percentile(bootstrap_breaks, (1 - alpha / 2) * 100))
                confidence_intervals[break_idx] = (max(0, lower), min(n - 1, upper))
            else:
                # Default to +/- 10% of segment length
                margin = max(3, n // 20)
                confidence_intervals[break_idx] = (
                    max(0, break_idx - margin),
                    min(n - 1, break_idx + margin)
                )
    else:
        # Asymptotic approximation
        for break_idx in break_indices:
            # Approximate standard error based on local volatility
            window = min(24, n // 4)
            start = max(0, break_idx - window // 2)
            end = min(n, break_idx + window // 2)
            local_std = np.std(y_clean[start:end])

            # Approximate CI width (conservative)
            se = local_std / np.sqrt(window)
            z = stats.norm.ppf((1 + confidence_level) / 2)
            margin = int(np.ceil(z * se * window / local_std)) if local_std > 0 else 3

            confidence_intervals[break_idx] = (
                max(0, break_idx - margin),
                min(n - 1, break_idx + margin)
            )

    return confidence_intervals
