"""
Density Forecast Evaluation
===========================

Implements evaluation metrics for probabilistic/density forecasts:
- CRPS (Continuous Ranked Probability Score)
- Log Score (negative log predictive density)
- PIT (Probability Integral Transform) tests

References:
-----------
- Gneiting, T. & Raftery, A.E. (2007). Strictly Proper Scoring Rules,
  Prediction, and Estimation. JASA, 102(477), 359-378.
- Diebold, F.X., Gunther, T.A., & Tay, A.S. (1998). Evaluating Density
  Forecasts with Applications to Financial Risk Management. IER, 39, 863-883.

Author: Academic Forecasting Pipeline
Date: 2026-02-10
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Optional, Tuple, Union, List


# =============================================================================
# CRPS (Continuous Ranked Probability Score)
# =============================================================================

def crps_normal(
    actual: float,
    mean: float,
    std: float
) -> float:
    """
    CRPS for Gaussian predictive distribution (closed-form).

    The CRPS is a proper scoring rule that measures the quality of
    probabilistic forecasts. Lower is better.

    CRPS = std * { z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) }

    where z = (actual - mean) / std

    Parameters
    ----------
    actual : float
        Realized value
    mean : float
        Forecast mean
    std : float
        Forecast standard deviation

    Returns
    -------
    crps : float
        CRPS value (lower is better)
    """
    if std <= 0:
        return np.inf if actual != mean else 0.0

    z = (actual - mean) / std
    crps = std * (
        z * (2 * stats.norm.cdf(z) - 1) +
        2 * stats.norm.pdf(z) -
        1 / np.sqrt(np.pi)
    )
    return crps


def crps_empirical(
    actual: float,
    samples: np.ndarray
) -> float:
    """
    CRPS from ensemble/MCMC samples.

    Uses the empirical CDF from samples to compute CRPS.

    CRPS = E|X - y| - 0.5 * E|X - X'|

    where X, X' are independent draws from the forecast distribution.

    Parameters
    ----------
    actual : float
        Realized value
    samples : np.ndarray
        Samples from predictive distribution

    Returns
    -------
    crps : float
        Empirical CRPS value
    """
    samples = np.asarray(samples)
    samples = samples[~np.isnan(samples)]

    if len(samples) == 0:
        return np.nan

    # First term: E|X - y|
    term1 = np.mean(np.abs(samples - actual))

    # Second term: 0.5 * E|X - X'|
    # More efficient computation using sorted samples
    n = len(samples)
    sorted_samples = np.sort(samples)

    # E|X - X'| = 2 * sum_{i<j} |X_i - X_j| / (n*(n-1))
    # This can be computed more efficiently
    cumsum = np.cumsum(sorted_samples)
    term2 = 2 * np.sum(
        (2 * np.arange(1, n + 1) - n - 1) * sorted_samples
    ) / (n * (n - 1))

    crps = term1 - 0.5 * np.abs(term2)
    return crps


def compute_crps_series(
    actuals: np.ndarray,
    forecast_means: np.ndarray,
    forecast_stds: np.ndarray
) -> np.ndarray:
    """
    Compute CRPS for a time series of Gaussian forecasts.

    Parameters
    ----------
    actuals : np.ndarray
        Realized values
    forecast_means : np.ndarray
        Point forecasts (means)
    forecast_stds : np.ndarray
        Forecast standard deviations

    Returns
    -------
    crps_values : np.ndarray
        CRPS at each time point
    """
    n = len(actuals)
    crps_values = np.full(n, np.nan)

    for t in range(n):
        if (np.isnan(actuals[t]) or
            np.isnan(forecast_means[t]) or
            np.isnan(forecast_stds[t])):
            continue

        crps_values[t] = crps_normal(
            actuals[t],
            forecast_means[t],
            forecast_stds[t]
        )

    return crps_values


def compare_crps(
    actuals: np.ndarray,
    forecasts_dict: Dict[str, Tuple[np.ndarray, np.ndarray]]
) -> pd.DataFrame:
    """
    Compare CRPS across multiple models.

    Parameters
    ----------
    actuals : np.ndarray
        Realized values
    forecasts_dict : dict
        {model_name: (means, stds)}

    Returns
    -------
    summary : pd.DataFrame
        CRPS statistics for each model
    """
    results = []

    for model_name, (means, stds) in forecasts_dict.items():
        crps_series = compute_crps_series(actuals, means, stds)
        valid_crps = crps_series[~np.isnan(crps_series)]

        if len(valid_crps) > 0:
            results.append({
                "Model": model_name,
                "Mean_CRPS": np.mean(valid_crps),
                "Median_CRPS": np.median(valid_crps),
                "Std_CRPS": np.std(valid_crps),
                "N_obs": len(valid_crps),
            })

    df = pd.DataFrame(results)
    if len(df) > 0:
        df = df.sort_values("Mean_CRPS").reset_index(drop=True)
        df["Rank"] = range(1, len(df) + 1)

    return df


# =============================================================================
# Log Score
# =============================================================================

def log_score_normal(
    actual: float,
    mean: float,
    std: float
) -> float:
    """
    Log score (negative log predictive density) for Gaussian distribution.

    Lower is better.

    Parameters
    ----------
    actual : float
        Realized value
    mean : float
        Forecast mean
    std : float
        Forecast standard deviation

    Returns
    -------
    log_score : float
        Negative log density (lower is better)
    """
    if std <= 0:
        return np.inf

    return -stats.norm.logpdf(actual, loc=mean, scale=std)


def compute_log_score_series(
    actuals: np.ndarray,
    forecast_means: np.ndarray,
    forecast_stds: np.ndarray
) -> np.ndarray:
    """
    Compute log scores for a series of forecasts.

    Parameters
    ----------
    actuals : np.ndarray
        Realized values
    forecast_means : np.ndarray
        Forecast means
    forecast_stds : np.ndarray
        Forecast standard deviations

    Returns
    -------
    log_scores : np.ndarray
        Log score at each time point
    """
    n = len(actuals)
    log_scores = np.full(n, np.nan)

    for t in range(n):
        if (np.isnan(actuals[t]) or
            np.isnan(forecast_means[t]) or
            np.isnan(forecast_stds[t])):
            continue

        log_scores[t] = log_score_normal(
            actuals[t],
            forecast_means[t],
            forecast_stds[t]
        )

    return log_scores


def compare_log_scores(
    actuals: np.ndarray,
    forecasts_dict: Dict[str, Tuple[np.ndarray, np.ndarray]]
) -> pd.DataFrame:
    """
    Compare log scores across multiple models.

    Parameters
    ----------
    actuals : np.ndarray
        Realized values
    forecasts_dict : dict
        {model_name: (means, stds)}

    Returns
    -------
    summary : pd.DataFrame
        Log score statistics for each model
    """
    results = []

    for model_name, (means, stds) in forecasts_dict.items():
        ls_series = compute_log_score_series(actuals, means, stds)
        valid_ls = ls_series[~np.isnan(ls_series)]

        if len(valid_ls) > 0:
            results.append({
                "Model": model_name,
                "Mean_LogScore": np.mean(valid_ls),
                "Median_LogScore": np.median(valid_ls),
                "Std_LogScore": np.std(valid_ls),
                "N_obs": len(valid_ls),
            })

    df = pd.DataFrame(results)
    if len(df) > 0:
        df = df.sort_values("Mean_LogScore").reset_index(drop=True)
        df["Rank"] = range(1, len(df) + 1)

    return df


# =============================================================================
# PIT (Probability Integral Transform)
# =============================================================================

def compute_pit(
    actuals: np.ndarray,
    forecast_means: np.ndarray,
    forecast_stds: np.ndarray
) -> np.ndarray:
    """
    Compute Probability Integral Transform values.

    If the density forecasts are correctly specified, PIT values
    should be uniformly distributed on [0, 1].

    Parameters
    ----------
    actuals : np.ndarray
        Realized values
    forecast_means : np.ndarray
        Forecast means
    forecast_stds : np.ndarray
        Forecast standard deviations

    Returns
    -------
    pit : np.ndarray
        PIT values (should be uniform if calibrated)
    """
    pit = stats.norm.cdf(actuals, loc=forecast_means, scale=forecast_stds)
    return pit


def pit_histogram_test(
    pit_values: np.ndarray,
    n_bins: int = 10
) -> Dict:
    """
    Chi-squared test for PIT uniformity.

    Tests H0: PIT values are uniformly distributed.

    Parameters
    ----------
    pit_values : np.ndarray
        PIT values
    n_bins : int
        Number of bins for histogram

    Returns
    -------
    result : dict
        Test results including chi-squared statistic and p-value
    """
    # Remove NaN
    pit_clean = pit_values[~np.isnan(pit_values)]
    n = len(pit_clean)

    if n < 20:
        return {
            "error": "Insufficient observations",
            "n_obs": n,
        }

    # Observed frequencies
    observed, bin_edges = np.histogram(pit_clean, bins=n_bins, range=(0, 1))

    # Expected frequencies (uniform)
    expected = n / n_bins

    # Chi-squared statistic
    chi2 = np.sum((observed - expected)**2 / expected)
    df = n_bins - 1
    p_value = 1 - stats.chi2.cdf(chi2, df=df)

    return {
        "chi2_statistic": chi2,
        "p_value": p_value,
        "df": df,
        "n_obs": n,
        "n_bins": n_bins,
        "reject_uniformity": p_value < 0.05,
        "observed_frequencies": observed.tolist(),
        "expected_frequency": expected,
    }


def pit_ks_test(pit_values: np.ndarray) -> Dict:
    """
    Kolmogorov-Smirnov test for PIT uniformity.

    An alternative to the chi-squared test.

    Parameters
    ----------
    pit_values : np.ndarray
        PIT values

    Returns
    -------
    result : dict
        KS test results
    """
    pit_clean = pit_values[~np.isnan(pit_values)]
    n = len(pit_clean)

    if n < 10:
        return {
            "error": "Insufficient observations",
            "n_obs": n,
        }

    # KS test against uniform distribution
    ks_stat, p_value = stats.kstest(pit_clean, 'uniform')

    return {
        "ks_statistic": ks_stat,
        "p_value": p_value,
        "n_obs": n,
        "reject_uniformity": p_value < 0.05,
    }


def pit_berkowitz_test(pit_values: np.ndarray) -> Dict:
    """
    Berkowitz (2001) test for density forecast evaluation.

    Transforms PIT to normal and tests for iid N(0,1).

    Parameters
    ----------
    pit_values : np.ndarray
        PIT values

    Returns
    -------
    result : dict
        Berkowitz test results
    """
    pit_clean = pit_values[~np.isnan(pit_values)]
    pit_clean = np.clip(pit_clean, 1e-10, 1-1e-10)  # Avoid infinities
    n = len(pit_clean)

    if n < 30:
        return {
            "error": "Insufficient observations",
            "n_obs": n,
        }

    # Transform to normal
    z = stats.norm.ppf(pit_clean)

    # Test for N(0,1)
    mean_z = np.mean(z)
    std_z = np.std(z, ddof=1)

    # Joint test: mean = 0, variance = 1
    # Under null, (n-1)*var ~ chi2(n-1)
    chi2_var = (n - 1) * std_z**2
    p_var = 1 - stats.chi2.cdf(chi2_var, df=n-1)

    # Mean test
    t_mean = mean_z / (std_z / np.sqrt(n))
    p_mean = 2 * (1 - stats.t.cdf(np.abs(t_mean), df=n-1))

    # Test for autocorrelation
    z_centered = z - mean_z
    autocorr = np.corrcoef(z_centered[:-1], z_centered[1:])[0, 1]
    # Fisher transformation for autocorrelation
    z_fisher = 0.5 * np.log((1 + autocorr) / (1 - autocorr + 1e-10))
    se_fisher = 1 / np.sqrt(n - 3)
    p_autocorr = 2 * (1 - stats.norm.cdf(np.abs(z_fisher / se_fisher)))

    return {
        "mean_z": mean_z,
        "std_z": std_z,
        "t_mean": t_mean,
        "p_mean": p_mean,
        "chi2_var": chi2_var,
        "p_var": p_var,
        "autocorr": autocorr,
        "p_autocorr": p_autocorr,
        "n_obs": n,
        "reject_iid_normal": (p_mean < 0.05) or (p_var < 0.05) or (p_autocorr < 0.05),
    }


def evaluate_density_forecasts(
    actuals: np.ndarray,
    forecast_means: np.ndarray,
    forecast_stds: np.ndarray,
    model_name: str = "Model"
) -> Dict:
    """
    Comprehensive density forecast evaluation for a single model.

    Parameters
    ----------
    actuals : np.ndarray
        Realized values
    forecast_means : np.ndarray
        Forecast means
    forecast_stds : np.ndarray
        Forecast standard deviations
    model_name : str
        Model identifier

    Returns
    -------
    result : dict
        Comprehensive evaluation results
    """
    # CRPS
    crps_series = compute_crps_series(actuals, forecast_means, forecast_stds)
    valid_crps = crps_series[~np.isnan(crps_series)]

    # Log Score
    ls_series = compute_log_score_series(actuals, forecast_means, forecast_stds)
    valid_ls = ls_series[~np.isnan(ls_series)]

    # PIT
    pit = compute_pit(actuals, forecast_means, forecast_stds)
    pit_chi2 = pit_histogram_test(pit)
    pit_ks = pit_ks_test(pit)

    # Coverage at different levels
    z_scores = (actuals - forecast_means) / forecast_stds
    coverage_90 = np.mean(np.abs(z_scores[~np.isnan(z_scores)]) < 1.645)
    coverage_95 = np.mean(np.abs(z_scores[~np.isnan(z_scores)]) < 1.96)

    return {
        "model": model_name,
        "crps": {
            "mean": np.mean(valid_crps) if len(valid_crps) > 0 else np.nan,
            "median": np.median(valid_crps) if len(valid_crps) > 0 else np.nan,
            "std": np.std(valid_crps) if len(valid_crps) > 0 else np.nan,
            "series": crps_series,
        },
        "log_score": {
            "mean": np.mean(valid_ls) if len(valid_ls) > 0 else np.nan,
            "median": np.median(valid_ls) if len(valid_ls) > 0 else np.nan,
            "std": np.std(valid_ls) if len(valid_ls) > 0 else np.nan,
            "series": ls_series,
        },
        "pit": {
            "values": pit,
            "chi2_test": pit_chi2,
            "ks_test": pit_ks,
        },
        "coverage": {
            "90_pct": coverage_90,
            "95_pct": coverage_95,
        },
        "n_obs": len(valid_crps),
    }


def density_evaluation_summary(
    actuals: np.ndarray,
    forecasts_dict: Dict[str, Tuple[np.ndarray, np.ndarray]]
) -> pd.DataFrame:
    """
    Create summary table of density forecast evaluation.

    Parameters
    ----------
    actuals : np.ndarray
        Realized values
    forecasts_dict : dict
        {model_name: (means, stds)}

    Returns
    -------
    summary : pd.DataFrame
        Summary table for publication
    """
    rows = []

    for model_name, (means, stds) in forecasts_dict.items():
        eval_result = evaluate_density_forecasts(
            actuals, means, stds, model_name
        )

        rows.append({
            "Model": model_name,
            "Mean_CRPS": eval_result["crps"]["mean"],
            "Mean_LogScore": eval_result["log_score"]["mean"],
            "Coverage_90": eval_result["coverage"]["90_pct"],
            "Coverage_95": eval_result["coverage"]["95_pct"],
            "PIT_uniform": not eval_result["pit"]["chi2_test"].get(
                "reject_uniformity", True
            ),
            "N_obs": eval_result["n_obs"],
        })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values("Mean_CRPS").reset_index(drop=True)
        df["Rank"] = range(1, len(df) + 1)

    return df


if __name__ == "__main__":
    # Simple test
    np.random.seed(42)
    n = 100

    # Generate test data
    actual = np.random.randn(n) * 2 + 10

    # Good forecast (well-calibrated)
    good_mean = actual + np.random.randn(n) * 0.3
    good_std = np.ones(n) * 0.5

    # Poor forecast (under-confident)
    poor_mean = actual + np.random.randn(n) * 0.5
    poor_std = np.ones(n) * 0.2

    print("Good forecast evaluation:")
    result_good = evaluate_density_forecasts(actual, good_mean, good_std, "Good")
    print(f"  Mean CRPS: {result_good['crps']['mean']:.4f}")
    print(f"  Mean Log Score: {result_good['log_score']['mean']:.4f}")
    print(f"  Coverage 90%: {result_good['coverage']['90_pct']:.2%}")
    print(f"  PIT chi2 p-value: {result_good['pit']['chi2_test'].get('p_value', 'N/A')}")

    print("\nPoor forecast evaluation:")
    result_poor = evaluate_density_forecasts(actual, poor_mean, poor_std, "Poor")
    print(f"  Mean CRPS: {result_poor['crps']['mean']:.4f}")
    print(f"  Mean Log Score: {result_poor['log_score']['mean']:.4f}")
    print(f"  Coverage 90%: {result_poor['coverage']['90_pct']:.2%}")
    print(f"  PIT chi2 p-value: {result_poor['pit']['chi2_test'].get('p_value', 'N/A')}")
