"""Phase 6: cross-variable relationship diagnostics."""

from __future__ import annotations

import io
import sys
import warnings
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests


def _align_pair(series1, series2):
    s1 = series1.dropna()
    s2 = series2.dropna()
    common_idx = s1.index.intersection(s2.index)
    if len(common_idx) == 0:
        return s1.iloc[0:0], s2.iloc[0:0]
    return s1.loc[common_idx], s2.loc[common_idx]


def compute_cross_correlation(series1, series2, name1, name2, max_lag=12):
    s1, s2 = _align_pair(series1, series2)
    if len(s1) < 30:
        return {"pair": f"{name1} vs {name2}", "error": "Insufficient common observations"}

    try:
        ccf_values = []
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                corr = s1.iloc[-lag:].corr(s2.iloc[:lag])
            elif lag > 0:
                corr = s1.iloc[:-lag].corr(s2.iloc[lag:])
            else:
                corr = s1.corr(s2)
            ccf_values.append({"lag": lag, "ccf": round(float(corr), 4) if not np.isnan(corr) else 0.0})

        max_ccf = max(ccf_values, key=lambda x: abs(x["ccf"]))

        if max_ccf["lag"] == 0:
            interp = "Contemporaneous"
        else:
            relation = "leads" if max_ccf["lag"] < 0 else "lags"
            interp = f"{name1} {relation} {name2} by {abs(max_ccf['lag'])} periods"

        return {
            "pair": f"{name1} vs {name2}",
            "ccf_values": ccf_values,
            "max_ccf": max_ccf["ccf"],
            "max_ccf_lag": max_ccf["lag"],
            "interpretation": interp,
            "effective_nobs": int(len(s1)),
            "transformation": "level",
        }
    except Exception as exc:
        return {"pair": f"{name1} vs {name2}", "error": str(exc)}


def run_granger_causality_test(series1, series2, name1, name2, max_lag=4, transform="diff"):
    s1, s2 = _align_pair(series1, series2)
    if transform == "diff":
        s1 = s1.diff().dropna()
        s2 = s2.diff().dropna()
        s1, s2 = _align_pair(s1, s2)

    min_obs_needed = max(50, max_lag * 12)
    if len(s1) < min_obs_needed:
        return {"test": f"{name1} → {name2}", "error": "Insufficient common observations"}

    try:
        data = np.column_stack([s2.values, s1.values])

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = grangercausalitytests(data, maxlag=max_lag, verbose=False)
        sys.stdout = old_stdout

        test_results = {}
        min_pvalue = 1.0
        best_lag = 1

        for lag in range(1, max_lag + 1):
            f_stat = float(result[lag][0]["ssr_ftest"][0])
            p_value = float(result[lag][0]["ssr_ftest"][1])
            test_results[f"lag_{lag}_f"] = round(f_stat, 4)
            test_results[f"lag_{lag}_p"] = round(p_value, 4)
            if p_value < min_pvalue:
                min_pvalue = p_value
                best_lag = lag

        return {
            "test": f"{name1} → {name2}",
            **test_results,
            "best_lag": best_lag,
            "best_p_value": round(min_pvalue, 4),
            "granger_causes": bool(min_pvalue < 0.05),
            "effective_nobs": int(len(s1)),
            "transformation": "first_difference" if transform == "diff" else "level",
        }
    except Exception as exc:
        return {"test": f"{name1} → {name2}", "error": str(exc)}


def run_phase6(df, target, predictors, verbose=True):
    ccf_results = []
    granger_results = []

    for pred in predictors:
        ccf = compute_cross_correlation(df[pred], df[target], pred, target)
        gc = run_granger_causality_test(df[pred], df[target], pred, target, transform="diff")

        ccf_results.append(ccf)
        granger_results.append(gc)

        if verbose:
            print(f"\n  {pred} → {target}...")
            print(f"    CCF: Max={ccf.get('max_ccf', 'N/A')} at lag={ccf.get('max_ccf_lag', 'N/A')}")
            print(f"    Granger ({gc.get('transformation', 'N/A')}): p={gc.get('best_p_value', 'N/A')}, Causes={gc.get('granger_causes', 'N/A')}")

    return {
        "cross_correlation": ccf_results,
        "granger_causality": granger_results,
    }
