"""
Forecast Encompassing Tests
===========================

Tests whether one forecast contains all the useful information
from another forecast.

Theory:
-------
If forecast 1 "encompasses" forecast 2, then adding forecast 2
to a regression of actual on forecast 1 should not improve the fit.

The regression-based test:
    y_t = alpha + lambda1 * f1_t + lambda2 * f2_t + epsilon_t

Tests:
- H0: lambda2 = 0 (f1 encompasses f2)
- H0: lambda1 = 0 (f2 encompasses f1)
- Joint test for optimal combination weights

References:
-----------
- Fair, R.C. & Shiller, R.J. (1990). Comparing Information in
  Forecasts from Econometric Models. American Economic Review.
- Chong, Y.Y. & Hendry, D.F. (1986). Econometric Evaluation of
  Linear Macro-Economic Models. RES.

Author: Academic Forecasting Pipeline
Date: 2026-02-10
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Optional, Tuple, List, Union


def forecast_encompassing_test(
    actual: np.ndarray,
    forecast1: np.ndarray,
    forecast2: np.ndarray,
    hac_lags: int = 0
) -> Dict:
    """
    Test whether forecast1 encompasses forecast2.

    Runs the regression:
        y_t = alpha + lambda1 * f1_t + lambda2 * f2_t + epsilon_t

    And tests H0: lambda2 = 0

    Parameters
    ----------
    actual : np.ndarray
        Actual/realized values
    forecast1 : np.ndarray
        First forecast (the one potentially encompassing)
    forecast2 : np.ndarray
        Second forecast (being tested for additional info)
    hac_lags : int
        Number of lags for HAC standard errors (0 for OLS)

    Returns
    -------
    result : dict
        Contains regression results and test statistics
    """
    actual = np.asarray(actual, dtype=float)
    forecast1 = np.asarray(forecast1, dtype=float)
    forecast2 = np.asarray(forecast2, dtype=float)

    # Handle missing values
    valid = ~(np.isnan(actual) | np.isnan(forecast1) | np.isnan(forecast2))
    y = actual[valid]
    f1 = forecast1[valid]
    f2 = forecast2[valid]
    n = len(y)

    if n < 10:
        return {"error": "Insufficient observations", "n_obs": n}

    # Build design matrix: [1, f1, f2]
    X = np.column_stack([np.ones(n), f1, f2])

    # OLS regression
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return {"error": "Singular matrix in regression", "n_obs": n}

    residuals = y - X @ beta

    # Compute variance-covariance matrix
    if hac_lags > 0:
        # Newey-West HAC standard errors
        vcov = _newey_west_vcov(X, residuals, hac_lags)
    else:
        # Standard OLS
        sigma2 = np.sum(residuals**2) / (n - 3)
        try:
            vcov = sigma2 * np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            return {"error": "Singular matrix", "n_obs": n}

    se = np.sqrt(np.diag(vcov))

    # t-statistics
    t_stats = beta / se
    df = n - 3

    # p-values (two-sided)
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=df))

    # R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - ss_res / ss_tot

    # Individual encompassing tests
    # H0: lambda2 = 0 (f1 encompasses f2)
    t_lambda2 = t_stats[2]
    p_lambda2 = p_values[2]
    f1_encompasses_f2 = p_lambda2 > 0.05

    # H0: lambda1 = 0 (f2 encompasses f1)
    t_lambda1 = t_stats[1]
    p_lambda1 = p_values[1]
    f2_encompasses_f1 = p_lambda1 > 0.05

    # Significance
    def get_stars(p):
        if p < 0.01:
            return "***"
        elif p < 0.05:
            return "**"
        elif p < 0.10:
            return "*"
        return ""

    return {
        "n_obs": n,
        "alpha": beta[0],
        "lambda1": beta[1],
        "lambda2": beta[2],
        "se_alpha": se[0],
        "se_lambda1": se[1],
        "se_lambda2": se[2],
        "t_lambda1": t_lambda1,
        "t_lambda2": t_lambda2,
        "p_lambda1": p_lambda1,
        "p_lambda2": p_lambda2,
        "r_squared": r_squared,
        "f1_encompasses_f2": f1_encompasses_f2,
        "f2_encompasses_f1": f2_encompasses_f1,
        "significance_lambda1": get_stars(p_lambda1),
        "significance_lambda2": get_stars(p_lambda2),
        "residual_std": np.sqrt(ss_res / (n - 3)),
    }


def _newey_west_vcov(X: np.ndarray, residuals: np.ndarray, lags: int) -> np.ndarray:
    """
    Compute Newey-West HAC variance-covariance matrix.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    residuals : np.ndarray
        Regression residuals
    lags : int
        Number of lags

    Returns
    -------
    vcov : np.ndarray
        HAC variance-covariance matrix
    """
    n, k = X.shape

    # Score matrix
    u = residuals.reshape(-1, 1)
    G = X * u  # n x k matrix of scores

    # Meat of the sandwich
    S = G.T @ G  # k x k

    for j in range(1, lags + 1):
        weight = 1 - j / (lags + 1)  # Bartlett kernel
        Gj = G[j:].T @ G[:-j]
        S += weight * (Gj + Gj.T)

    # Bread
    try:
        XX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return np.full((k, k), np.nan)

    # Sandwich
    vcov = XX_inv @ S @ XX_inv

    return vcov


def fair_shiller_test(
    actual: np.ndarray,
    forecast1: np.ndarray,
    forecast2: np.ndarray,
    hac_lags: int = 0
) -> Dict:
    """
    Fair-Shiller information content test.

    A generalization of the encompassing test that allows for
    nested and non-nested model comparison.

    Parameters
    ----------
    actual : np.ndarray
        Actual values
    forecast1 : np.ndarray
        First forecast
    forecast2 : np.ndarray
        Second forecast
    hac_lags : int
        HAC lags for standard errors

    Returns
    -------
    result : dict
        Test results with information content metrics
    """
    # Get basic encompassing results
    enc = forecast_encompassing_test(actual, forecast1, forecast2, hac_lags)

    if "error" in enc:
        return enc

    # Information content interpretation
    lambda1 = enc["lambda1"]
    lambda2 = enc["lambda2"]
    p1 = enc["p_lambda1"]
    p2 = enc["p_lambda2"]

    # Categorize results
    if p1 < 0.05 and p2 < 0.05:
        interpretation = "Both forecasts contain unique information"
        optimal_combination = True
    elif p1 < 0.05 and p2 >= 0.05:
        interpretation = "Forecast 1 encompasses Forecast 2"
        optimal_combination = False
    elif p1 >= 0.05 and p2 < 0.05:
        interpretation = "Forecast 2 encompasses Forecast 1"
        optimal_combination = False
    else:
        interpretation = "Neither forecast has significant information"
        optimal_combination = False

    # Relative information content
    total_weight = abs(lambda1) + abs(lambda2)
    if total_weight > 0:
        info_share_f1 = abs(lambda1) / total_weight
        info_share_f2 = abs(lambda2) / total_weight
    else:
        info_share_f1 = 0.5
        info_share_f2 = 0.5

    enc.update({
        "interpretation": interpretation,
        "optimal_combination": optimal_combination,
        "info_share_f1": info_share_f1,
        "info_share_f2": info_share_f2,
    })

    return enc


def pairwise_encompassing_matrix(
    actual: np.ndarray,
    forecasts_dict: Dict[str, np.ndarray],
    hac_lags: int = 0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute pairwise encompassing tests for all model pairs.

    Parameters
    ----------
    actual : np.ndarray
        Actual values
    forecasts_dict : dict
        {model_name: forecast}
    hac_lags : int
        HAC lags

    Returns
    -------
    lambda_matrix : pd.DataFrame
        Matrix of lambda2 coefficients (column model's weight when added to row model)
    p_value_matrix : pd.DataFrame
        Matrix of p-values for lambda2 = 0 test
    """
    model_names = list(forecasts_dict.keys())
    n_models = len(model_names)

    lambda_matrix = np.full((n_models, n_models), np.nan)
    p_value_matrix = np.full((n_models, n_models), np.nan)

    for i, m1 in enumerate(model_names):
        for j, m2 in enumerate(model_names):
            if i == j:
                continue

            result = forecast_encompassing_test(
                actual,
                forecasts_dict[m1],
                forecasts_dict[m2],
                hac_lags=hac_lags
            )

            if "error" not in result:
                lambda_matrix[i, j] = result["lambda2"]
                p_value_matrix[i, j] = result["p_lambda2"]

    lambda_df = pd.DataFrame(lambda_matrix, index=model_names, columns=model_names)
    p_value_df = pd.DataFrame(p_value_matrix, index=model_names, columns=model_names)

    return lambda_df, p_value_df


def format_encompassing_table(
    lambda_matrix: pd.DataFrame,
    p_value_matrix: pd.DataFrame
) -> pd.DataFrame:
    """
    Format encompassing results for publication.

    Shows lambda2 values with significance stars.
    Entry (i,j) shows the weight on forecast j when added to forecast i.

    Parameters
    ----------
    lambda_matrix : pd.DataFrame
        Lambda2 coefficients
    p_value_matrix : pd.DataFrame
        P-values for encompassing test

    Returns
    -------
    formatted : pd.DataFrame
        Formatted table
    """
    model_names = lambda_matrix.index.tolist()
    n_models = len(model_names)

    formatted = np.empty((n_models, n_models), dtype=object)

    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                formatted[i, j] = "-"
            else:
                lam = lambda_matrix.iloc[i, j]
                pval = p_value_matrix.iloc[i, j]

                if np.isnan(lam):
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
                    formatted[i, j] = f"{lam:.3f}{stars}"

    return pd.DataFrame(formatted, index=model_names, columns=model_names)


def encompassing_summary(
    actual: np.ndarray,
    forecasts_dict: Dict[str, np.ndarray],
    hac_lags: int = 0
) -> pd.DataFrame:
    """
    Create summary table of encompassing test results.

    Parameters
    ----------
    actual : np.ndarray
        Actual values
    forecasts_dict : dict
        Model forecasts
    hac_lags : int
        HAC lags

    Returns
    -------
    summary : pd.DataFrame
        Summary showing which models encompass which
    """
    model_names = list(forecasts_dict.keys())
    n_models = len(model_names)

    results = []

    for i, m1 in enumerate(model_names):
        for j, m2 in enumerate(model_names):
            if i >= j:
                continue

            enc_result = fair_shiller_test(
                actual,
                forecasts_dict[m1],
                forecasts_dict[m2],
                hac_lags=hac_lags
            )

            if "error" not in enc_result:
                results.append({
                    "Model_1": m1,
                    "Model_2": m2,
                    "lambda1": enc_result["lambda1"],
                    "lambda2": enc_result["lambda2"],
                    "p_lambda1": enc_result["p_lambda1"],
                    "p_lambda2": enc_result["p_lambda2"],
                    "R_squared": enc_result["r_squared"],
                    "Interpretation": enc_result["interpretation"],
                    "n_obs": enc_result["n_obs"],
                })

    return pd.DataFrame(results)


def optimal_combination_weights(
    actual: np.ndarray,
    forecasts_dict: Dict[str, np.ndarray],
    constrained: bool = True
) -> Dict:
    """
    Estimate optimal forecast combination weights.

    Parameters
    ----------
    actual : np.ndarray
        Actual values
    forecasts_dict : dict
        Model forecasts
    constrained : bool
        If True, weights sum to 1 and are non-negative

    Returns
    -------
    result : dict
        Optimal weights and diagnostics
    """
    actual = np.asarray(actual, dtype=float)
    model_names = list(forecasts_dict.keys())
    n_models = len(model_names)

    # Stack forecasts
    forecasts = np.column_stack([
        np.asarray(forecasts_dict[m], dtype=float)
        for m in model_names
    ])

    # Handle missing values
    valid = ~(np.isnan(actual) | np.any(np.isnan(forecasts), axis=1))
    y = actual[valid]
    X = forecasts[valid]
    n = len(y)

    if n < n_models + 5:
        return {"error": "Insufficient observations", "n_obs": n}

    if constrained:
        # Use scipy.optimize for constrained optimization
        from scipy.optimize import minimize

        def objective(w):
            pred = X @ w
            return np.mean((y - pred)**2)

        # Constraints: sum(w) = 1, w >= 0
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        ]
        bounds = [(0, 1) for _ in range(n_models)]
        x0 = np.ones(n_models) / n_models

        result = minimize(
            objective, x0,
            method="SLSQP",
            constraints=constraints,
            bounds=bounds
        )

        if result.success:
            weights = result.x
        else:
            # Fall back to equal weights
            weights = np.ones(n_models) / n_models

        # Compute R-squared
        pred = X @ weights
        ss_res = np.sum((y - pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - ss_res / ss_tot

        return {
            "weights": dict(zip(model_names, weights)),
            "mse": ss_res / n,
            "r_squared": r_squared,
            "n_obs": n,
            "constrained": True,
        }

    else:
        # Unconstrained OLS with intercept
        X_with_const = np.column_stack([np.ones(n), X])

        try:
            beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            return {"error": "Singular matrix", "n_obs": n}

        intercept = beta[0]
        weights = beta[1:]

        pred = X_with_const @ beta
        ss_res = np.sum((y - pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - ss_res / ss_tot

        return {
            "intercept": intercept,
            "weights": dict(zip(model_names, weights)),
            "mse": ss_res / n,
            "r_squared": r_squared,
            "n_obs": n,
            "constrained": False,
        }


if __name__ == "__main__":
    # Simple test
    np.random.seed(42)
    n = 100

    actual = np.random.randn(n) + 5
    f1 = actual + np.random.randn(n) * 0.5  # Good forecast
    f2 = actual + np.random.randn(n) * 0.8  # Okay forecast
    f3 = np.random.randn(n) + 5  # Random forecast

    print("Testing f1 vs f2:")
    result = fair_shiller_test(actual, f1, f2)
    print(f"  lambda1 = {result['lambda1']:.3f} (p={result['p_lambda1']:.3f})")
    print(f"  lambda2 = {result['lambda2']:.3f} (p={result['p_lambda2']:.3f})")
    print(f"  Interpretation: {result['interpretation']}")

    print("\nTesting f1 vs f3 (random):")
    result = fair_shiller_test(actual, f1, f3)
    print(f"  lambda1 = {result['lambda1']:.3f} (p={result['p_lambda1']:.3f})")
    print(f"  lambda2 = {result['lambda2']:.3f} (p={result['p_lambda2']:.3f})")
    print(f"  Interpretation: {result['interpretation']}")

    print("\nOptimal combination weights:")
    forecasts = {"F1": f1, "F2": f2, "F3": f3}
    weights = optimal_combination_weights(actual, forecasts, constrained=True)
    for model, w in weights["weights"].items():
        print(f"  {model}: {w:.3f}")
