"""Phase 3: temporal dependence and seasonality diagnostics."""

from __future__ import annotations

import numpy as np
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox


def compute_acf_pacf(series, name, nlags=36):
    series_clean = series.dropna()
    if len(series_clean) < nlags + 10:
        return {"variable": name, "error": "Insufficient observations"}

    try:
        acf_values = acf(series_clean, nlags=nlags, fft=True)
        pacf_values = pacf(series_clean, nlags=nlags, method="ywm")
        conf_int = 1.96 / np.sqrt(len(series_clean))

        sig_acf_lags = [i for i, v in enumerate(acf_values[1:], 1) if abs(v) > conf_int]
        sig_pacf_lags = [i for i, v in enumerate(pacf_values[1:], 1) if abs(v) > conf_int]

        acf1 = float(acf_values[1])
        persistence = "High" if acf1 > 0.9 else "Medium" if acf1 > 0.5 else "Low"

        return {
            "variable": name,
            "acf_values": acf_values.tolist(),
            "pacf_values": pacf_values.tolist(),
            "confidence_interval": conf_int,
            "significant_acf_lags": sig_acf_lags[:10],
            "significant_pacf_lags": sig_pacf_lags[:10],
            "acf_lag1": round(acf1, 4),
            "acf_lag12": round(float(acf_values[12]), 4) if len(acf_values) > 12 else None,
            "pacf_lag1": round(float(pacf_values[1]), 4),
            "persistence": persistence,
            "effective_nobs": int(len(series_clean)),
        }
    except Exception as exc:
        return {"variable": name, "error": str(exc)}


def run_ljungbox_test(series, name, lags=(12, 24, 36)):
    series_clean = series.dropna()
    if len(series_clean) < max(lags) + 10:
        return {"variable": name, "error": "Insufficient observations"}

    try:
        results = {}
        for lag in lags:
            lb_result = acorr_ljungbox(series_clean, lags=[lag], return_df=True)
            results[f"Q_{lag}"] = round(float(lb_result["lb_stat"].values[0]), 4)
            results[f"p_{lag}"] = round(float(lb_result["lb_pvalue"].values[0]), 4)

        return {
            "variable": name,
            **results,
            "autocorrelated": any(results[f"p_{lag}"] < 0.05 for lag in lags),
            "effective_nobs": int(len(series_clean)),
        }
    except Exception as exc:
        return {"variable": name, "error": str(exc)}


def run_stl_decomposition(series, name, period=12):
    series_clean = series.dropna()
    if len(series_clean) < 2 * period:
        return {"variable": name, "error": "Insufficient observations for STL"}

    try:
        stl = STL(series_clean, period=period, robust=True)
        result = stl.fit()

        var_resid = float(np.var(result.resid))
        var_detrend = float(np.var(result.seasonal + result.resid))
        var_deseas = float(np.var(result.trend + result.resid))

        trend_strength = max(0, 1 - var_resid / var_deseas) if var_deseas > 0 else 0
        seasonal_strength = max(0, 1 - var_resid / var_detrend) if var_detrend > 0 else 0

        return {
            "variable": name,
            "trend_strength": round(trend_strength, 4),
            "seasonal_strength": round(seasonal_strength, 4),
            "period": period,
            "trend": result.trend.tolist(),
            "seasonal": result.seasonal.tolist(),
            "resid": result.resid.tolist(),
            "has_strong_trend": trend_strength > 0.7,
            "has_seasonality": seasonal_strength > 0.3,
            "effective_nobs": int(len(series_clean)),
        }
    except Exception as exc:
        return {"variable": name, "error": str(exc)}


def run_phase3(df, variables, verbose=True):
    acf_results = []
    lb_results = []
    stl_results = []

    for var in variables:
        series = df[var]

        acf_res = compute_acf_pacf(series, var)
        lb_res = run_ljungbox_test(series, var)
        stl_res = run_stl_decomposition(series, var)

        acf_results.append(acf_res)
        lb_results.append(lb_res)
        stl_results.append(stl_res)

        if verbose:
            print(f"\n  Analyzing {var}...")
            print(f"    ACF(1)={acf_res.get('acf_lag1', 'N/A')}, Persistence={acf_res.get('persistence', 'N/A')}")
            print(f"    Ljung-Box: Q(12) p={lb_res.get('p_12', 'N/A')}, Autocorrelated={lb_res.get('autocorrelated', 'N/A')}")
            print(f"    STL: Trend={stl_res.get('trend_strength', 'N/A')}, Seasonal={stl_res.get('seasonal_strength', 'N/A')}")

    return {
        "acf_pacf": acf_results,
        "ljungbox": lb_results,
        "stl": stl_results,
    }
