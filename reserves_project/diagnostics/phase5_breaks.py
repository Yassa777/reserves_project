"""Phase 5: structural break diagnostics."""

from __future__ import annotations

import numpy as np
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from .config import DEFAULT_DATE


def run_chow_test(series, name, break_date=DEFAULT_DATE):
    series_clean = series.dropna()
    if len(series_clean) < 50:
        return {"variable": name, "error": "Insufficient observations"}

    if break_date not in series_clean.index:
        idx = series_clean.index.get_indexer([break_date], method="nearest")[0]
        if idx < 0 or idx >= len(series_clean):
            return {"variable": name, "error": "Break date outside data range"}
        break_date = series_clean.index[idx]

    pre = series_clean[series_clean.index < break_date]
    post = series_clean[series_clean.index >= break_date]
    if len(pre) < 20 or len(post) < 20:
        return {"variable": name, "error": "Insufficient observations in one regime"}

    try:
        def fit_ar1(s):
            y = s.values[1:]
            x = add_constant(s.values[:-1])
            return OLS(y, x).fit()

        model_full = fit_ar1(series_clean)
        model_pre = fit_ar1(pre)
        model_post = fit_ar1(post)

        rss_full = model_full.ssr
        rss_pre = model_pre.ssr
        rss_post = model_post.ssr

        n = len(series_clean) - 1
        k = 2
        denom = (rss_pre + rss_post) / (n - 2 * k)
        if np.isclose(denom, 0.0):
            return {"variable": name, "error": "Degenerate variance for Chow denominator"}

        f_stat = ((rss_full - rss_pre - rss_post) / k) / denom
        p_value = 1 - stats.f.cdf(f_stat, k, n - 2 * k)

        return {
            "variable": name,
            "break_date": str(break_date.date()),
            "f_statistic": round(float(f_stat), 4),
            "p_value": round(float(p_value), 4),
            "break_confirmed": bool(p_value < 0.05),
            "pre_period_obs": int(len(pre)),
            "post_period_obs": int(len(post)),
            "effective_nobs": int(len(series_clean)),
        }
    except Exception as exc:
        return {"variable": name, "error": str(exc)}


def run_cusum_test(series, name):
    series_clean = series.dropna()
    if len(series_clean) < 50:
        return {"variable": name, "error": "Insufficient observations"}

    try:
        y = series_clean.values
        n = len(y)
        min_obs = 20
        recursive_residuals = []

        for t in range(min_obs, n):
            y_t = y[1 : t + 1]
            x_t = add_constant(y[:t])
            model = OLS(y_t, x_t).fit()

            x_new = np.array([1, y[t - 1]])
            y_pred = model.predict(x_new.reshape(1, -1))[0]
            resid = y[t] - y_pred

            h = x_new @ np.linalg.inv(x_t.T @ x_t) @ x_new.T
            denom = (model.mse_resid * (1 + h)) ** 0.5
            if np.isclose(denom, 0.0):
                continue
            recursive_residuals.append(resid / denom)

        if len(recursive_residuals) < 10:
            return {"variable": name, "error": "Insufficient recursive residuals"}

        rr = np.array(recursive_residuals)
        rr_std = np.std(rr)
        if np.isclose(rr_std, 0.0):
            return {"variable": name, "error": "CUSUM residual std is zero"}

        cusum = np.cumsum(rr) / rr_std

        k = len(rr)
        a = 0.948
        critical_bound = a * np.sqrt(k) + 2 * a * np.arange(1, k + 1) / np.sqrt(k)

        exceeds = bool(np.any(cusum > critical_bound) or np.any(cusum < -critical_bound))

        return {
            "variable": name,
            "cusum": cusum.tolist(),
            "upper_bound": critical_bound.tolist(),
            "lower_bound": (-critical_bound).tolist(),
            "max_cusum": round(float(np.max(np.abs(cusum))), 4),
            "exceeds_bounds": exceeds,
            "instability_detected": exceeds,
            "effective_nobs": int(len(series_clean)),
        }
    except Exception as exc:
        return {"variable": name, "error": str(exc)}


def run_phase5(df, variables, verbose=True):
    chow_results = []
    cusum_results = []

    for var in variables:
        series = df[var]

        chow = run_chow_test(series, var)
        cusum = run_cusum_test(series, var)

        chow_results.append(chow)
        cusum_results.append(cusum)

        if verbose:
            print(f"\n  Testing {var}...")
            print(
                f"    Chow: F={chow.get('f_statistic', 'N/A')}, p={chow.get('p_value', 'N/A')}, "
                f"Break={chow.get('break_confirmed', 'N/A')}"
            )
            print(f"    CUSUM: Max={cusum.get('max_cusum', 'N/A')}, Exceeds={cusum.get('exceeds_bounds', 'N/A')}")

    return {
        "chow": chow_results,
        "cusum": cusum_results,
    }
