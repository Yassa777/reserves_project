"""
Linearity Tests for Threshold VAR Models.

Implements tests for threshold nonlinearity:
1. Sup-Wald test (Hansen, 1996)
2. Bootstrap linearity test (handles Davies problem)
3. Threshold confidence interval via likelihood ratio inversion

Reference: Specification 06 - Threshold VAR
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from scipy import stats
from statsmodels.tsa.api import VAR
import warnings


def linearity_test(
    Y: pd.DataFrame,
    z: pd.Series,
    n_lags: int = 2,
    delay: int = 1,
    trim: float = 0.15,
    min_obs_per_regime: int = 24
) -> Dict[str, Any]:
    """
    Test for threshold nonlinearity vs linear VAR using sup-Wald test.

    H0: Linear VAR (no threshold effect)
    H1: Threshold VAR (regime-switching at threshold tau)

    Note: Critical values require bootstrap due to Davies problem
    (threshold not identified under null hypothesis).

    Parameters
    ----------
    Y : pd.DataFrame
        Multivariate system
    z : pd.Series
        Threshold variable
    n_lags : int, default=2
        VAR lag order
    delay : int, default=1
        Threshold delay
    trim : float, default=0.15
        Trimming fraction
    min_obs_per_regime : int, default=24
        Minimum observations per regime

    Returns
    -------
    dict with:
        - f_statistic: Sup-Wald F-statistic
        - df1, df2: Degrees of freedom
        - ssr_linear: SSR from linear VAR
        - ssr_tvar: SSR from threshold VAR
        - optimal_threshold: Threshold that minimizes SSR
        - asymptotic_p_value: Approximate p-value (use bootstrap for accuracy)
    """
    from .tvar import ThresholdVAR

    # Align data
    common_idx = Y.index.intersection(z.index)
    Y = Y.loc[common_idx].copy()
    z = z.loc[common_idx].copy()

    # Create lagged threshold
    z_lagged = z.shift(delay).dropna()
    valid_idx = z_lagged.index
    Y = Y.loc[valid_idx]

    T = len(Y)
    k = Y.shape[1]

    # Fit linear VAR
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        var_linear = VAR(Y).fit(n_lags)

    ssr_linear = np.sum(var_linear.resid.values ** 2)

    # Fit threshold VAR
    try:
        tvar = ThresholdVAR(
            n_lags=n_lags,
            delay=delay,
            trim=trim,
            min_obs_per_regime=min_obs_per_regime
        )
        tvar.fit(Y, z)

        ssr_tvar = (
            np.sum(tvar.var_regime1.resid.values ** 2) +
            np.sum(tvar.var_regime2.resid.values ** 2)
        )
        optimal_threshold = tvar.threshold

    except ValueError as e:
        # Could not fit TVAR (insufficient data per regime)
        return {
            "f_statistic": np.nan,
            "df1": np.nan,
            "df2": np.nan,
            "ssr_linear": float(ssr_linear),
            "ssr_tvar": np.nan,
            "optimal_threshold": np.nan,
            "asymptotic_p_value": np.nan,
            "error": str(e),
        }

    # Compute F-statistic
    n_params = k * n_lags + 1  # Parameters per equation
    df1 = n_params * k  # Extra parameters in TVAR
    df2 = T - 2 * n_params * k

    if df2 <= 0:
        df2 = T - n_params * k  # Fallback

    f_stat = ((ssr_linear - ssr_tvar) / df1) / (ssr_tvar / max(df2, 1))

    # Asymptotic p-value (approximate - bootstrap is more accurate)
    # Under H0, the sup-Wald follows a non-standard distribution
    # We use F-distribution as rough approximation
    try:
        p_value = 1 - stats.f.cdf(f_stat, df1, max(df2, 1))
    except Exception:
        p_value = np.nan

    return {
        "f_statistic": float(f_stat),
        "df1": int(df1),
        "df2": int(df2),
        "ssr_linear": float(ssr_linear),
        "ssr_tvar": float(ssr_tvar),
        "ssr_reduction_pct": float((ssr_linear - ssr_tvar) / ssr_linear * 100),
        "optimal_threshold": float(optimal_threshold),
        "asymptotic_p_value": float(p_value) if not np.isnan(p_value) else None,
        "note": "Use bootstrap_linearity_test for accurate p-value (Davies problem)",
    }


def bootstrap_linearity_test(
    Y: pd.DataFrame,
    z: pd.Series,
    n_lags: int = 2,
    delay: int = 1,
    trim: float = 0.15,
    min_obs_per_regime: int = 24,
    n_bootstrap: int = 500,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Bootstrap test for linearity (handles Davies problem).

    The Davies problem arises because the threshold parameter is not
    identified under the null hypothesis (linear VAR). Standard asymptotic
    theory does not apply, so we use bootstrap to obtain the null distribution.

    Parameters
    ----------
    Y : pd.DataFrame
        Multivariate system
    z : pd.Series
        Threshold variable
    n_lags : int, default=2
        VAR lag order
    delay : int, default=1
        Threshold delay
    trim : float, default=0.15
        Trimming fraction
    min_obs_per_regime : int, default=24
        Minimum observations per regime
    n_bootstrap : int, default=500
        Number of bootstrap replications
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    dict with:
        - f_statistic: Observed sup-Wald F-statistic
        - bootstrap_p_value: Bootstrap p-value
        - reject_linearity: True if H0 rejected at 5% level
        - bootstrap_critical_values: 90%, 95%, 99% critical values
    """
    from .tvar import ThresholdVAR

    if random_state is not None:
        np.random.seed(random_state)

    # Align data
    common_idx = Y.index.intersection(z.index)
    Y = Y.loc[common_idx].copy()
    z = z.loc[common_idx].copy()

    # Create lagged threshold
    z_lagged = z.shift(delay).dropna()
    valid_idx = z_lagged.index
    Y = Y.loc[valid_idx]
    z = z_lagged.loc[valid_idx]

    # Get observed F-statistic
    test_result = linearity_test(
        Y, z,
        n_lags=n_lags,
        delay=delay,
        trim=trim,
        min_obs_per_regime=min_obs_per_regime
    )

    f_obs = test_result["f_statistic"]

    if np.isnan(f_obs):
        return {
            "f_statistic": float(f_obs),
            "bootstrap_p_value": np.nan,
            "reject_linearity": False,
            "error": test_result.get("error", "Could not compute F-statistic"),
        }

    # Fit linear VAR under null
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        var_linear = VAR(Y).fit(n_lags)

    # Bootstrap
    f_bootstrap = np.zeros(n_bootstrap)
    successful_boots = 0

    for b in range(n_bootstrap):
        try:
            # Generate data under null (linear VAR)
            # Use residual resampling
            resid = var_linear.resid
            boot_idx = np.random.choice(len(resid), size=len(resid), replace=True)
            resid_boot = resid[boot_idx]

            # Reconstruct series
            Y_boot_values = np.zeros_like(Y.values)
            Y_boot_values[:n_lags] = Y.values[:n_lags]

            for t in range(n_lags, len(Y)):
                Y_lagged = Y_boot_values[t - n_lags:t].flatten()
                Y_boot_values[t] = var_linear.params[0] + var_linear.coefs.flatten() @ Y_lagged
                Y_boot_values[t] += resid_boot[t - n_lags] if t - n_lags < len(resid_boot) else 0

            Y_boot = pd.DataFrame(Y_boot_values, columns=Y.columns, index=Y.index)

            # Compute F-statistic on bootstrap sample
            boot_result = linearity_test(
                Y_boot, z,
                n_lags=n_lags,
                delay=delay,
                trim=trim,
                min_obs_per_regime=min_obs_per_regime
            )

            if not np.isnan(boot_result["f_statistic"]):
                f_bootstrap[successful_boots] = boot_result["f_statistic"]
                successful_boots += 1

        except Exception:
            continue

    if successful_boots < n_bootstrap * 0.5:
        # Too few successful bootstraps
        return {
            "f_statistic": float(f_obs),
            "bootstrap_p_value": np.nan,
            "reject_linearity": False,
            "n_successful_boots": successful_boots,
            "error": f"Only {successful_boots}/{n_bootstrap} bootstraps succeeded",
        }

    # Trim to successful bootstraps
    f_bootstrap = f_bootstrap[:successful_boots]

    # Bootstrap p-value
    p_value = np.mean(f_bootstrap >= f_obs)

    # Critical values
    critical_values = {
        "90%": float(np.percentile(f_bootstrap, 90)),
        "95%": float(np.percentile(f_bootstrap, 95)),
        "99%": float(np.percentile(f_bootstrap, 99)),
    }

    return {
        "f_statistic": float(f_obs),
        "bootstrap_p_value": float(p_value),
        "reject_linearity": p_value < 0.05,
        "reject_at_10pct": p_value < 0.10,
        "reject_at_1pct": p_value < 0.01,
        "bootstrap_critical_values": critical_values,
        "n_bootstrap": n_bootstrap,
        "n_successful_boots": successful_boots,
    }


def threshold_confidence_interval(
    Y: pd.DataFrame,
    z: pd.Series,
    n_lags: int = 2,
    delay: int = 1,
    trim: float = 0.15,
    min_obs_per_regime: int = 24,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Construct confidence interval for threshold parameter.

    Based on likelihood ratio inversion: the CI includes all tau values
    where the LR statistic is below the chi-square critical value.

    Parameters
    ----------
    Y : pd.DataFrame
        Multivariate system
    z : pd.Series
        Threshold variable
    n_lags : int, default=2
        VAR lag order
    delay : int, default=1
        Threshold delay
    trim : float, default=0.15
        Trimming fraction
    min_obs_per_regime : int, default=24
        Minimum observations per regime
    alpha : float, default=0.05
        Significance level (0.05 gives 95% CI)

    Returns
    -------
    dict with:
        - point_estimate: Optimal threshold
        - lower: CI lower bound
        - upper: CI upper bound
        - confidence_level: 1 - alpha
    """
    from .tvar import ThresholdVAR

    # Fit optimal TVAR first
    try:
        tvar = ThresholdVAR(
            n_lags=n_lags,
            delay=delay,
            trim=trim,
            min_obs_per_regime=min_obs_per_regime
        )
        tvar.fit(Y, z)
    except ValueError as e:
        return {
            "point_estimate": np.nan,
            "lower": np.nan,
            "upper": np.nan,
            "error": str(e),
        }

    T = len(tvar.Y)
    k = tvar.Y.shape[1]

    # Best SSR
    ssr_best = (
        np.sum(tvar.var_regime1.resid.values ** 2) +
        np.sum(tvar.var_regime2.resid.values ** 2)
    )

    # Chi-square critical value
    cv = stats.chi2.ppf(1 - alpha, df=1)

    # Grid search for CI bounds - use same approach as TVAR fit
    sorted_z = np.sort(tvar.z.values)
    n_grid = len(sorted_z)
    min_obs = max(min_obs_per_regime, (k * n_lags + 1) * 2)

    # Only consider thresholds where both regimes have >= min_obs
    valid_thresholds = []
    for i, tau in enumerate(sorted_z):
        n_below = i + 1
        n_above = n_grid - i - 1
        if n_below >= min_obs and n_above >= min_obs:
            valid_thresholds.append(tau)

    if len(valid_thresholds) > 20:
        trim_n = int(len(valid_thresholds) * trim)
        threshold_grid = valid_thresholds[trim_n:-trim_n] if trim_n > 0 else valid_thresholds
    else:
        threshold_grid = valid_thresholds

    in_ci = []

    for tau in threshold_grid:
        # Compute SSR at this tau
        regime1_mask = tvar.z <= tau
        regime2_mask = tvar.z > tau

        n1, n2 = regime1_mask.sum(), regime2_mask.sum()

        if n1 < min_obs or n2 < min_obs:
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                var1 = VAR(tvar.Y[regime1_mask]).fit(n_lags)
                var2 = VAR(tvar.Y[regime2_mask]).fit(n_lags)

            ssr_tau = np.sum(var1.resid.values ** 2) + np.sum(var2.resid.values ** 2)

            # LR statistic
            if ssr_tau > 0 and ssr_best > 0:
                lr_stat = T * np.log(ssr_tau / ssr_best)

                if lr_stat <= cv:
                    in_ci.append(tau)
        except Exception:
            continue

    if in_ci:
        lower = min(in_ci)
        upper = max(in_ci)
    else:
        # CI could not be computed, use point estimate
        lower = tvar.threshold
        upper = tvar.threshold

    return {
        "point_estimate": float(tvar.threshold),
        "lower": float(lower),
        "upper": float(upper),
        "confidence_level": 1 - alpha,
        "n_grid_points_in_ci": len(in_ci),
    }


def compare_tvar_msvar(
    tvar_regimes: pd.Series,
    msvar_probs: pd.Series,
    dates: Optional[pd.DatetimeIndex] = None
) -> Dict[str, Any]:
    """
    Compare TVAR regime indicators with MS-VAR smoothed probabilities.

    Parameters
    ----------
    tvar_regimes : pd.Series
        TVAR regime indicators (0 = stable, 1 = crisis)
    msvar_probs : pd.Series
        MS-VAR smoothed probability of crisis regime
    dates : pd.DatetimeIndex, optional
        Common date index (if not provided, uses tvar_regimes index)

    Returns
    -------
    dict with:
        - concordance: Fraction of periods where regimes agree
        - correlation: Correlation between TVAR indicator and MS-VAR probability
        - tvar_switch_dates: Dates of TVAR regime switches
        - msvar_switch_dates: Dates of MS-VAR regime switches
    """
    # Align series
    if dates is not None:
        tvar_regimes = tvar_regimes.reindex(dates)
        msvar_probs = msvar_probs.reindex(dates)

    common_idx = tvar_regimes.dropna().index.intersection(msvar_probs.dropna().index)

    if len(common_idx) == 0:
        return {
            "concordance": np.nan,
            "correlation": np.nan,
            "error": "No overlapping observations",
        }

    tvar_aligned = tvar_regimes.loc[common_idx].values
    msvar_aligned = msvar_probs.loc[common_idx].values

    # MS-VAR regime = 1 if P(crisis) > 0.5
    msvar_regimes = (msvar_aligned > 0.5).astype(int)

    # Concordance measure
    concordance = np.mean(tvar_aligned == msvar_regimes)

    # Correlation
    correlation = np.corrcoef(tvar_aligned, msvar_aligned)[0, 1]

    # Timing differences - find regime switches
    tvar_switches = np.where(np.diff(tvar_aligned) != 0)[0]
    msvar_switches = np.where(np.diff(msvar_regimes) != 0)[0]

    tvar_switch_dates = [str(common_idx[i + 1].date()) for i in tvar_switches]
    msvar_switch_dates = [str(common_idx[i + 1].date()) for i in msvar_switches]

    return {
        "concordance": float(concordance),
        "concordance_pct": float(concordance * 100),
        "correlation": float(correlation),
        "n_tvar_switches": len(tvar_switches),
        "n_msvar_switches": len(msvar_switches),
        "tvar_switch_dates": tvar_switch_dates,
        "msvar_switch_dates": msvar_switch_dates,
        "n_observations": len(common_idx),
    }


def regime_persistence_test(regime_indicators: pd.Series) -> Dict[str, Any]:
    """
    Test for regime persistence (are regimes sticky?).

    Computes transition probabilities and tests if they differ
    from 0.5 (random switching).

    Parameters
    ----------
    regime_indicators : pd.Series
        Regime indicators (0 or 1)

    Returns
    -------
    dict with:
        - p_00: P(stay in regime 0 | in regime 0)
        - p_11: P(stay in regime 1 | in regime 1)
        - p_01: P(switch to regime 1 | in regime 0)
        - p_10: P(switch to regime 0 | in regime 1)
        - persistence_index: Average of p_00 and p_11
        - test_statistic: Chi-square test for independence
        - p_value: P-value for test
    """
    regimes = regime_indicators.values
    T = len(regimes)

    # Count transitions
    n_00 = np.sum((regimes[:-1] == 0) & (regimes[1:] == 0))
    n_01 = np.sum((regimes[:-1] == 0) & (regimes[1:] == 1))
    n_10 = np.sum((regimes[:-1] == 1) & (regimes[1:] == 0))
    n_11 = np.sum((regimes[:-1] == 1) & (regimes[1:] == 1))

    # Transition probabilities
    n_from_0 = n_00 + n_01
    n_from_1 = n_10 + n_11

    p_00 = n_00 / n_from_0 if n_from_0 > 0 else np.nan
    p_01 = n_01 / n_from_0 if n_from_0 > 0 else np.nan
    p_10 = n_10 / n_from_1 if n_from_1 > 0 else np.nan
    p_11 = n_11 / n_from_1 if n_from_1 > 0 else np.nan

    # Persistence index
    if not np.isnan(p_00) and not np.isnan(p_11):
        persistence_index = (p_00 + p_11) / 2
    else:
        persistence_index = np.nan

    # Chi-square test for independence
    observed = np.array([[n_00, n_01], [n_10, n_11]])
    try:
        chi2, p_value, dof, expected = stats.chi2_contingency(observed)
    except Exception:
        chi2, p_value = np.nan, np.nan

    return {
        "p_00": float(p_00) if not np.isnan(p_00) else None,
        "p_01": float(p_01) if not np.isnan(p_01) else None,
        "p_10": float(p_10) if not np.isnan(p_10) else None,
        "p_11": float(p_11) if not np.isnan(p_11) else None,
        "persistence_index": float(persistence_index) if not np.isnan(persistence_index) else None,
        "test_statistic": float(chi2) if not np.isnan(chi2) else None,
        "p_value": float(p_value) if not np.isnan(p_value) else None,
        "transition_counts": {
            "n_00": int(n_00),
            "n_01": int(n_01),
            "n_10": int(n_10),
            "n_11": int(n_11),
        },
    }
