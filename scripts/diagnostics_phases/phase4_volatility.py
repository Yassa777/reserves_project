"""Phase 4: volatility and heteroskedasticity diagnostics."""

from __future__ import annotations

import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.stats.diagnostic import het_arch

from .config import CRISIS_START


def run_arch_lm_test(series, name, nlags=12):
    series_clean = series.dropna()
    if len(series_clean) < 50:
        return {"variable": name, "error": "Insufficient observations"}

    try:
        y = series_clean.values[1:]
        y_lag = series_clean.values[:-1]
        model = OLS(y, add_constant(y_lag)).fit()
        residuals = model.resid

        if np.isclose(np.std(residuals), 0.0):
            return {"variable": name, "error": "AR(1) residual variance is zero"}

        lm_stat, lm_pval, f_stat, f_pval = het_arch(residuals, nlags=nlags)
        return {
            "variable": name,
            "arch_lm_stat": round(float(lm_stat), 4),
            "arch_lm_pvalue": round(float(lm_pval), 4),
            "arch_f_stat": round(float(f_stat), 4),
            "arch_f_pvalue": round(float(f_pval), 4),
            "has_arch_effects": bool(lm_pval < 0.05),
            "effective_nobs": int(len(series_clean)),
        }
    except Exception as exc:
        return {"variable": name, "error": str(exc)}


def compute_rolling_volatility(series, name, windows=(3, 6, 12)):
    series_clean = series.dropna()
    if len(series_clean) < max(windows) + 10:
        return {"variable": name, "error": "Insufficient observations"}

    try:
        results = {"variable": name, "effective_nobs": int(len(series_clean))}

        pre_crisis = series_clean[(series_clean.index >= "2015-01-01") & (series_clean.index < CRISIS_START)]
        crisis = series_clean[(series_clean.index >= CRISIS_START) & (series_clean.index <= "2022-12-31")]
        post_crisis = series_clean[series_clean.index >= "2023-01-01"]

        for window in windows:
            rolling_std = series_clean.rolling(window=window).std()
            results[f"rolling_vol_{window}m"] = rolling_std.tolist()

            if len(pre_crisis) > window:
                results[f"pre_crisis_vol_{window}m"] = round(float(pre_crisis.std()), 2)
            if len(crisis) > window:
                results[f"crisis_vol_{window}m"] = round(float(crisis.std()), 2)
            if len(post_crisis) > window:
                results[f"post_crisis_vol_{window}m"] = round(float(post_crisis.std()), 2)

        pre_v = results.get("pre_crisis_vol_12m")
        crisis_v = results.get("crisis_vol_12m")
        results["vol_change_ratio"] = round(crisis_v / pre_v, 2) if pre_v and crisis_v else None

        return results
    except Exception as exc:
        return {"variable": name, "error": str(exc)}


def run_phase4(df, variables, verbose=True):
    arch_results = []
    rolling_results = []

    for var in variables:
        series = df[var]

        arch = run_arch_lm_test(series, var)
        rolling = compute_rolling_volatility(series, var)

        arch_results.append(arch)
        rolling_results.append(rolling)

        if verbose:
            print(f"\n  Analyzing {var}...")
            print(f"    ARCH-LM: stat={arch.get('arch_lm_stat', 'N/A')}, p={arch.get('arch_lm_pvalue', 'N/A')}")
            print(f"    Vol ratio (crisis/pre): {rolling.get('vol_change_ratio', 'N/A')}")

    return {
        "arch_lm": arch_results,
        "rolling_volatility": rolling_results,
    }
