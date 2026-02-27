"""Phase 2: stationarity and integration diagnostics."""

from __future__ import annotations

import numpy as np
import warnings
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews


def run_adf_test(series, name, maxlag=None, regression="ct"):
    series_clean = series.dropna()
    if len(series_clean) < 20:
        return {"variable": name, "error": "Insufficient observations"}
    if np.isclose(series_clean.std(), 0.0):
        return {"variable": name, "error": "Constant series"}

    try:
        stat, p_value, lags, nobs, crit_vals, _ = adfuller(
            series_clean,
            maxlag=maxlag,
            regression=regression,
            autolag="AIC",
        )
        return {
            "variable": name,
            "adf_statistic": round(stat, 4),
            "p_value": round(p_value, 4),
            "lags_used": int(lags),
            "nobs": int(nobs),
            "critical_1pct": round(crit_vals["1%"], 4),
            "critical_5pct": round(crit_vals["5%"], 4),
            "critical_10pct": round(crit_vals["10%"], 4),
            "stationary_5pct": bool(p_value < 0.05),
            "regression": regression,
        }
    except Exception as exc:
        return {"variable": name, "error": str(exc)}


def run_kpss_test(series, name, regression="ct", nlags="auto"):
    series_clean = series.dropna()
    if len(series_clean) < 20:
        return {"variable": name, "error": "Insufficient observations"}
    if np.isclose(series_clean.std(), 0.0):
        return {"variable": name, "error": "Constant series"}

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, p_value, lags, crit_vals = kpss(series_clean, regression=regression, nlags=nlags)
        return {
            "variable": name,
            "kpss_statistic": round(stat, 4),
            "p_value": round(p_value, 4),
            "lags_used": int(lags),
            "critical_1pct": round(crit_vals["1%"], 4),
            "critical_5pct": round(crit_vals["5%"], 4),
            "critical_10pct": round(crit_vals["10%"], 4),
            "stationary_5pct": bool(p_value > 0.05),
            "regression": regression,
        }
    except Exception as exc:
        return {"variable": name, "error": str(exc)}


def run_zivot_andrews_test(series, name, regression="ct", maxlag=None):
    series_clean = series.dropna()
    if len(series_clean) < 50:
        return {"variable": name, "error": "Insufficient observations for ZA test"}
    if np.isclose(series_clean.std(), 0.0):
        return {"variable": name, "error": "Constant series"}

    try:
        stat, p_value, crit_vals, baselag, break_idx = zivot_andrews(
            series_clean,
            maxlag=maxlag,
            regression=regression,
            autolag="AIC",
        )
        break_date = series_clean.index[int(break_idx)]
        return {
            "variable": name,
            "za_statistic": round(float(stat), 4),
            "p_value": round(float(p_value), 4),
            "lags_used": int(baselag),
            "break_date": str(break_date.date()),
            "critical_1pct": round(float(crit_vals["1%"]), 4),
            "critical_5pct": round(float(crit_vals["5%"]), 4),
            "critical_10pct": round(float(crit_vals["10%"]), 4),
            "stationary_5pct": bool(float(stat) < float(crit_vals["5%"])),
            "regression": regression,
        }
    except Exception as exc:
        return {"variable": name, "error": str(exc)}


def determine_integration_order(adf_result, kpss_result):
    if "error" in adf_result or "error" in kpss_result:
        return "Error"

    adf_stationary = adf_result.get("stationary_5pct", False)
    kpss_stationary = kpss_result.get("stationary_5pct", False)

    if adf_stationary and kpss_stationary:
        return "I(0) - Stationary"
    if not adf_stationary and not kpss_stationary:
        return "I(1) or higher - Non-stationary"
    if adf_stationary and not kpss_stationary:
        return "I(0) - Trend-stationary"
    return "Inconclusive - Possible structural break"


def run_phase2(df, variables, verbose=True):
    adf_results = []
    kpss_results = []
    za_results = []
    integration_summary = []

    for var in variables:
        series = df[var]

        adf = run_adf_test(series, var)
        kpss_res = run_kpss_test(series, var)
        za = run_zivot_andrews_test(series, var)

        adf_results.append(adf)
        kpss_results.append(kpss_res)
        za_results.append(za)

        integration = determine_integration_order(adf, kpss_res)
        integration_summary.append(
            {
                "variable": var,
                "adf_p_value": adf.get("p_value"),
                "adf_stationary": adf.get("stationary_5pct"),
                "kpss_stationary": kpss_res.get("stationary_5pct"),
                "za_stationary": za.get("stationary_5pct"),
                "za_break_date": za.get("break_date"),
                "integration_order": integration,
                "effective_nobs": int(series.dropna().shape[0]),
            }
        )

        if verbose:
            print(f"\n  Testing {var}...")
            print(f"    ADF: stat={adf.get('adf_statistic', 'N/A')}, p={adf.get('p_value', 'N/A')}")
            print(f"    KPSS: stat={kpss_res.get('kpss_statistic', 'N/A')}, p={kpss_res.get('p_value', 'N/A')}")
            print(f"    ZA: stat={za.get('za_statistic', 'N/A')}, break={za.get('break_date', 'N/A')}")
            print(f"    Integration: {integration}")

    return {
        "adf": adf_results,
        "kpss": kpss_results,
        "zivot_andrews": za_results,
        "integration_summary": integration_summary,
    }
