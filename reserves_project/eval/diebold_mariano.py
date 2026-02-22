"""
Diebold-Mariano Test for Equal Predictive Accuracy
===================================================

Implements the Diebold-Mariano (1995) test for comparing forecast accuracy,
with Harvey-Leybourne-Newbold (1997) small-sample correction.

References:
-----------
- Diebold, F.X. & Mariano, R.S. (1995). Comparing Predictive Accuracy.
  Journal of Business & Economic Statistics, 13(3), 253-263.
- Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the Equality of
  Prediction Mean Squared Errors. International Journal of Forecasting, 13, 281-291.

Author: Academic Forecasting Pipeline
Date: 2026-02-10
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Optional, Union, Callable, Tuple, List


def compute_loss_differential(
    actual: np.ndarray,
    forecast1: np.ndarray,
    forecast2: np.ndarray,
    loss_fn: Union[str, Callable] = "squared"
) -> np.ndarray:
    """
    Compute the loss differential series d_t = L(e1_t) - L(e2_t).

    Parameters
    ----------
    actual : np.ndarray
        Actual/realized values
    forecast1 : np.ndarray
        First forecast series
    forecast2 : np.ndarray
        Second forecast series
    loss_fn : str or callable
        Loss function: "squared" (MSE), "absolute" (MAE), or custom callable

    Returns
    -------
    d : np.ndarray
        Loss differential series
    """
    e1 = actual - forecast1
    e2 = actual - forecast2

    if loss_fn == "squared":
        d = e1**2 - e2**2
    elif loss_fn == "absolute":
        d = np.abs(e1) - np.abs(e2)
    elif callable(loss_fn):
        d = loss_fn(e1) - loss_fn(e2)
    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")

    return d


def newey_west_variance(
    d: np.ndarray,
    h: int = 1,
    bandwidth: Optional[int] = None
) -> float:
    """
    Compute Newey-West HAC variance estimator.

    For h-step ahead forecasts, the loss differentials may be serially
    correlated up to order (h-1).

    Parameters
    ----------
    d : np.ndarray
        Loss differential series (with NaN removed)
    h : int
        Forecast horizon
    bandwidth : int, optional
        Bandwidth for kernel. If None, uses (h-1) as per DM recommendation

    Returns
    -------
    var_d : float
        HAC-consistent variance estimate
    """
    n = len(d)
    mean_d = np.mean(d)

    # Set bandwidth
    if bandwidth is None:
        bandwidth = max(h - 1, 0)

    # Autocovariances
    gamma = np.zeros(bandwidth + 1)
    for j in range(bandwidth + 1):
        if j == 0:
            gamma[j] = np.mean((d - mean_d)**2)
        else:
            gamma[j] = np.mean((d[j:] - mean_d) * (d[:-j] - mean_d))

    # HAC variance with Bartlett kernel
    var_d = gamma[0]
    for j in range(1, bandwidth + 1):
        weight = 1 - j / (bandwidth + 1)  # Bartlett kernel
        var_d += 2 * weight * gamma[j]

    # Ensure positive variance
    var_d = max(var_d, 1e-10)

    return var_d


def diebold_mariano_test(
    actual: np.ndarray,
    forecast1: np.ndarray,
    forecast2: np.ndarray,
    loss_fn: Union[str, Callable] = "squared",
    h: int = 1,
    alternative: str = "two-sided"
) -> Dict:
    """
    Diebold-Mariano test for equal predictive accuracy.

    Tests H0: E[d_t] = 0 where d_t = L(e1_t) - L(e2_t)

    Parameters
    ----------
    actual : np.ndarray
        Actual/realized values
    forecast1 : np.ndarray
        First forecast series
    forecast2 : np.ndarray
        Second forecast series
    loss_fn : str or callable
        "squared" (MSE), "absolute" (MAE), or custom function
    h : int
        Forecast horizon (for HAC standard error)
    alternative : str
        "two-sided", "less" (f1 better), or "greater" (f2 better)

    Returns
    -------
    result : dict
        Contains:
        - dm_statistic: DM test statistic
        - p_value: p-value (asymptotic)
        - mean_loss_diff: Mean loss differential
        - se: Standard error
        - n_obs: Number of observations
        - better_forecast: Which forecast is better (if significant)
        - significance: Stars for significance level
    """
    # Convert to numpy arrays and align
    actual = np.asarray(actual, dtype=float)
    forecast1 = np.asarray(forecast1, dtype=float)
    forecast2 = np.asarray(forecast2, dtype=float)

    # Compute loss differential
    d = compute_loss_differential(actual, forecast1, forecast2, loss_fn)

    # Remove NaN values
    valid_mask = ~np.isnan(d)
    d = d[valid_mask]
    n = len(d)

    if n < 10:
        return {
            "error": "Insufficient observations",
            "n_obs": n,
            "dm_statistic": np.nan,
            "p_value": np.nan,
        }

    # Mean loss differential
    mean_d = np.mean(d)

    # HAC variance
    var_d = newey_west_variance(d, h=h)

    # Standard error of the mean
    se = np.sqrt(var_d / n)

    # DM statistic
    dm_stat = mean_d / se

    # p-value using normal distribution
    if alternative == "two-sided":
        p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    elif alternative == "less":
        # H1: E[d] < 0, i.e., forecast1 is better
        p_value = stats.norm.cdf(dm_stat)
    elif alternative == "greater":
        # H1: E[d] > 0, i.e., forecast2 is better
        p_value = 1 - stats.norm.cdf(dm_stat)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    # Significance stars
    if p_value < 0.01:
        significance = "***"
    elif p_value < 0.05:
        significance = "**"
    elif p_value < 0.10:
        significance = "*"
    else:
        significance = ""

    # Interpretation
    if mean_d < 0:
        better = "Forecast 1"
    else:
        better = "Forecast 2"

    return {
        "dm_statistic": dm_stat,
        "p_value": p_value,
        "mean_loss_diff": mean_d,
        "se": se,
        "n_obs": n,
        "better_forecast": better if p_value < 0.10 else "No significant difference",
        "significance": significance,
    }


def dm_test_hln(
    actual: np.ndarray,
    forecast1: np.ndarray,
    forecast2: np.ndarray,
    loss_fn: Union[str, Callable] = "squared",
    h: int = 1,
    alternative: str = "two-sided"
) -> Dict:
    """
    Diebold-Mariano test with Harvey-Leybourne-Newbold small-sample correction.

    The HLN correction:
    1. Applies a finite-sample correction to the DM statistic
    2. Uses t-distribution instead of normal for inference

    Parameters
    ----------
    actual : np.ndarray
        Actual/realized values
    forecast1 : np.ndarray
        First forecast series
    forecast2 : np.ndarray
        Second forecast series
    loss_fn : str or callable
        Loss function
    h : int
        Forecast horizon
    alternative : str
        Test alternative

    Returns
    -------
    result : dict
        Extended result dict with HLN-corrected values
    """
    # Get base DM result
    result = diebold_mariano_test(
        actual, forecast1, forecast2,
        loss_fn=loss_fn, h=h, alternative=alternative
    )

    if "error" in result:
        return result

    n = result["n_obs"]
    dm_stat = result["dm_statistic"]

    # HLN correction factor
    # Adjusts for small-sample bias in variance estimation
    correction = np.sqrt((n + 1 - 2*h + h*(h-1)/n) / n)
    dm_corrected = dm_stat * correction

    # Use t-distribution for inference
    df = n - 1
    if alternative == "two-sided":
        p_value_hln = 2 * (1 - stats.t.cdf(np.abs(dm_corrected), df=df))
    elif alternative == "less":
        p_value_hln = stats.t.cdf(dm_corrected, df=df)
    else:
        p_value_hln = 1 - stats.t.cdf(dm_corrected, df=df)

    # Update significance based on HLN p-value
    if p_value_hln < 0.01:
        significance_hln = "***"
    elif p_value_hln < 0.05:
        significance_hln = "**"
    elif p_value_hln < 0.10:
        significance_hln = "*"
    else:
        significance_hln = ""

    result["dm_statistic_hln"] = dm_corrected
    result["p_value_hln"] = p_value_hln
    result["significance_hln"] = significance_hln
    result["correction_factor"] = correction
    result["df"] = df

    return result


def dm_test_matrix(
    actual: np.ndarray,
    forecasts_dict: Dict[str, np.ndarray],
    loss_fn: Union[str, Callable] = "squared",
    h: int = 1,
    use_hln: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute pairwise DM tests for all model pairs.

    Parameters
    ----------
    actual : np.ndarray
        Actual values
    forecasts_dict : dict
        {model_name: forecast_array}
    loss_fn : str or callable
        Loss function
    h : int
        Forecast horizon
    use_hln : bool
        Whether to use HLN correction

    Returns
    -------
    dm_stats : pd.DataFrame
        Matrix of DM statistics (row vs column)
        Positive value means column model has lower loss
    p_values : pd.DataFrame
        Matrix of p-values
    """
    model_names = list(forecasts_dict.keys())
    n_models = len(model_names)

    dm_stats = np.full((n_models, n_models), np.nan)
    p_values = np.full((n_models, n_models), np.nan)

    for i, m1 in enumerate(model_names):
        for j, m2 in enumerate(model_names):
            if i == j:
                continue

            # Test m1 vs m2
            if use_hln:
                result = dm_test_hln(
                    actual, forecasts_dict[m1], forecasts_dict[m2],
                    loss_fn=loss_fn, h=h
                )
                stat_key = "dm_statistic_hln"
                pval_key = "p_value_hln"
            else:
                result = diebold_mariano_test(
                    actual, forecasts_dict[m1], forecasts_dict[m2],
                    loss_fn=loss_fn, h=h
                )
                stat_key = "dm_statistic"
                pval_key = "p_value"

            if "error" not in result:
                dm_stats[i, j] = result.get(stat_key, np.nan)
                p_values[i, j] = result.get(pval_key, np.nan)

    dm_stats_df = pd.DataFrame(dm_stats, index=model_names, columns=model_names)
    p_values_df = pd.DataFrame(p_values, index=model_names, columns=model_names)

    return dm_stats_df, p_values_df


def format_dm_table_for_paper(
    dm_stats: pd.DataFrame,
    p_values: pd.DataFrame,
    latex: bool = False
) -> Union[pd.DataFrame, str]:
    """
    Format DM test results for publication.

    Parameters
    ----------
    dm_stats : pd.DataFrame
        DM statistics matrix
    p_values : pd.DataFrame
        p-values matrix
    latex : bool
        If True, return LaTeX table string

    Returns
    -------
    formatted : pd.DataFrame or str
        Formatted table with significance stars
    """
    model_names = dm_stats.index.tolist()
    n_models = len(model_names)

    # Create formatted matrix
    formatted = np.empty((n_models, n_models), dtype=object)

    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                formatted[i, j] = "-"
            else:
                stat = dm_stats.iloc[i, j]
                pval = p_values.iloc[i, j]

                if np.isnan(stat):
                    formatted[i, j] = "."
                else:
                    if pval < 0.01:
                        stars = "***"
                    elif pval < 0.05:
                        stars = "**"
                    elif pval < 0.10:
                        stars = "*"
                    else:
                        stars = ""
                    formatted[i, j] = f"{stat:.2f}{stars}"

    formatted_df = pd.DataFrame(formatted, index=model_names, columns=model_names)

    if latex:
        return formatted_df.to_latex()

    return formatted_df


def dm_test_vs_benchmark(
    actual: np.ndarray,
    forecasts_dict: Dict[str, np.ndarray],
    benchmark_name: str,
    loss_fn: str = "squared",
    h: int = 1
) -> pd.DataFrame:
    """
    Test all models against a single benchmark.

    Parameters
    ----------
    actual : np.ndarray
        Actual values
    forecasts_dict : dict
        All forecasts including benchmark
    benchmark_name : str
        Name of benchmark model
    loss_fn : str
        Loss function
    h : int
        Forecast horizon

    Returns
    -------
    results : pd.DataFrame
        Summary table of DM tests vs benchmark
    """
    if benchmark_name not in forecasts_dict:
        raise ValueError(f"Benchmark '{benchmark_name}' not found in forecasts")

    benchmark = forecasts_dict[benchmark_name]
    results = []

    for model_name, forecast in forecasts_dict.items():
        if model_name == benchmark_name:
            continue

        dm_result = dm_test_hln(
            actual, benchmark, forecast,
            loss_fn=loss_fn, h=h
        )

        if "error" not in dm_result:
            results.append({
                "model": model_name,
                "benchmark": benchmark_name,
                "dm_statistic": dm_result["dm_statistic_hln"],
                "p_value": dm_result["p_value_hln"],
                "significance": dm_result["significance_hln"],
                "mean_loss_diff": dm_result["mean_loss_diff"],
                "better": dm_result["better_forecast"],
                "n_obs": dm_result["n_obs"],
            })

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Simple test
    np.random.seed(42)
    n = 100
    actual = np.random.randn(n)
    f1 = actual + np.random.randn(n) * 0.5  # Better forecast
    f2 = actual + np.random.randn(n) * 1.0  # Worse forecast

    result = dm_test_hln(actual, f1, f2)
    print("DM Test Result (HLN corrected):")
    for k, v in result.items():
        print(f"  {k}: {v}")
