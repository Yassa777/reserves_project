"""Phase 7: cointegration and ECM/VECM suitability diagnostics."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen

from .config import MIN_OBS_COINTEGRATION, PHASE7_COINTEGRATION_VARS, TARGET_VARIABLE


def _get_i1_flags(phase2_results: dict) -> dict[str, bool]:
    i1_map: dict[str, bool] = {}
    for row in phase2_results.get("integration_summary", []):
        order = str(row.get("integration_order", ""))
        i1_map[row.get("variable", "")] = "I(1)" in order
    return i1_map


def run_engle_granger_pair(y: pd.Series, x: pd.Series, y_name: str, x_name: str):
    common = pd.concat([y, x], axis=1).dropna()
    if len(common) < MIN_OBS_COINTEGRATION:
        return {
            "pair": f"{x_name} -> {y_name}",
            "error": "Insufficient common observations",
            "effective_nobs": int(len(common)),
        }

    try:
        stat, p_value, crit_vals = coint(common.iloc[:, 0], common.iloc[:, 1], trend="c", autolag="aic")
        return {
            "pair": f"{x_name} -> {y_name}",
            "coint_t_stat": round(float(stat), 4),
            "p_value": round(float(p_value), 4),
            "critical_1pct": round(float(crit_vals[0]), 4),
            "critical_5pct": round(float(crit_vals[1]), 4),
            "critical_10pct": round(float(crit_vals[2]), 4),
            "cointegrated_5pct": bool(float(p_value) < 0.05),
            "effective_nobs": int(len(common)),
        }
    except Exception as exc:
        return {"pair": f"{x_name} -> {y_name}", "error": str(exc), "effective_nobs": int(len(common))}


def run_ecm_suitability(y: pd.Series, x: pd.Series, y_name: str, x_name: str):
    common = pd.concat([y, x], axis=1).dropna()
    if len(common) < MIN_OBS_COINTEGRATION:
        return {
            "pair": f"{x_name} -> {y_name}",
            "error": "Insufficient common observations",
            "effective_nobs": int(len(common)),
        }

    try:
        y_level = common.iloc[:, 0]
        x_level = common.iloc[:, 1]

        coint_ols = OLS(y_level.values, add_constant(x_level.values)).fit()
        ect = pd.Series(coint_ols.resid, index=common.index, name="ect")

        dy = y_level.diff().rename("dy")
        dx = x_level.diff().rename("dx")
        ect_lag = ect.shift(1).rename("ect_lag")
        ecm_df = pd.concat([dy, dx, ect_lag], axis=1).dropna()

        if len(ecm_df) < 50:
            return {
                "pair": f"{x_name} -> {y_name}",
                "error": "Insufficient observations for ECM",
                "effective_nobs": int(len(ecm_df)),
            }

        model = OLS(ecm_df["dy"].values, add_constant(ecm_df[["dx", "ect_lag"]].values)).fit()
        ect_coef = float(model.params[2])
        ect_p = float(model.pvalues[2])

        return {
            "pair": f"{x_name} -> {y_name}",
            "ect_coef": round(ect_coef, 4),
            "ect_p_value": round(ect_p, 4),
            "dx_coef": round(float(model.params[1]), 4),
            "dx_p_value": round(float(model.pvalues[1]), 4),
            "r_squared": round(float(model.rsquared), 4),
            "ecm_viable": bool((ect_coef < 0) and (ect_p < 0.05)),
            "effective_nobs": int(len(ecm_df)),
        }
    except Exception as exc:
        return {"pair": f"{x_name} -> {y_name}", "error": str(exc), "effective_nobs": int(len(common))}


def run_johansen_system(system_df: pd.DataFrame):
    if len(system_df) < MIN_OBS_COINTEGRATION:
        return {
            "error": "Insufficient observations for Johansen",
            "effective_nobs": int(len(system_df)),
        }

    try:
        # Approximate lag choice using VAR on levels, then convert to VECM lag-diff order.
        maxlags = min(12, max(2, len(system_df) // 10))
        best_lag = 2
        try:
            lag_res = VAR(system_df).select_order(maxlags=maxlags)
            if lag_res.aic is not None and not np.isnan(lag_res.aic):
                best_lag = max(1, int(lag_res.aic))
        except Exception:
            best_lag = 2

        k_ar_diff = max(1, best_lag - 1)
        joh = coint_johansen(system_df, det_order=0, k_ar_diff=k_ar_diff)

        trace_stats = joh.lr1.tolist()
        trace_cv_95 = joh.cvt[:, 1].tolist()
        maxeig_stats = joh.lr2.tolist()
        maxeig_cv_95 = joh.cvm[:, 1].tolist()

        rank_trace = int(sum(stat > cv for stat, cv in zip(trace_stats, trace_cv_95)))
        rank_maxeig = int(sum(stat > cv for stat, cv in zip(maxeig_stats, maxeig_cv_95)))

        return {
            "variables": system_df.columns.tolist(),
            "effective_nobs": int(len(system_df)),
            "k_ar_diff": int(k_ar_diff),
            "trace_stats": [round(float(v), 4) for v in trace_stats],
            "trace_cv_95": [round(float(v), 4) for v in trace_cv_95],
            "maxeig_stats": [round(float(v), 4) for v in maxeig_stats],
            "maxeig_cv_95": [round(float(v), 4) for v in maxeig_cv_95],
            "rank_trace_5pct": rank_trace,
            "rank_maxeig_5pct": rank_maxeig,
            "cointegration_detected": bool(rank_trace >= 1 or rank_maxeig >= 1),
        }
    except Exception as exc:
        return {
            "error": str(exc),
            "effective_nobs": int(len(system_df)),
            "variables": system_df.columns.tolist(),
        }


def run_vecm_suitability(system_df: pd.DataFrame, johansen_result: dict):
    if "error" in johansen_result:
        return {
            "error": "Johansen step failed; skipping VECM suitability",
            "effective_nobs": int(len(system_df)),
        }

    rank = int(johansen_result.get("rank_trace_5pct", 0))
    if rank < 1:
        return {
            "effective_nobs": int(len(system_df)),
            "vecm_viable": False,
            "reason": "No cointegration rank at 5%",
        }

    try:
        k_ar_diff = int(johansen_result.get("k_ar_diff", 1))
        coint_rank = min(rank, len(system_df.columns) - 1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vecm = VECM(system_df, k_ar_diff=k_ar_diff, coint_rank=coint_rank, deterministic="co")
            fit = vecm.fit()

        target_idx = list(system_df.columns).index(TARGET_VARIABLE)
        alpha_target = float(fit.alpha[target_idx, 0])
        alpha_target_p = float(fit.pvalues_alpha[target_idx, 0])

        return {
            "effective_nobs": int(len(system_df)),
            "coint_rank_used": int(coint_rank),
            "k_ar_diff": int(k_ar_diff),
            "target_alpha_loading": round(alpha_target, 4),
            "target_alpha_p_value": round(alpha_target_p, 4),
            "vecm_viable": bool((alpha_target < 0) and (alpha_target_p < 0.1)),
            "log_likelihood": round(float(fit.llf), 4),
        }
    except Exception as exc:
        return {
            "error": str(exc),
            "effective_nobs": int(len(system_df)),
            "vecm_viable": False,
        }


def run_phase7(df: pd.DataFrame, usable_vars: list[str], phase2_results: dict, verbose=True):
    i1_map = _get_i1_flags(phase2_results)

    candidate_vars = [v for v in PHASE7_COINTEGRATION_VARS if v in usable_vars]
    if TARGET_VARIABLE not in candidate_vars:
        return {
            "engle_granger": [],
            "ecm_suitability": [],
            "johansen": {"error": "Target variable not available"},
            "vecm_suitability": {"error": "Target variable not available"},
        }

    system_df = df[candidate_vars].dropna()

    engle_granger_results = []
    ecm_results = []

    for pred in candidate_vars:
        if pred == TARGET_VARIABLE:
            continue

        eg = run_engle_granger_pair(df[TARGET_VARIABLE], df[pred], TARGET_VARIABLE, pred)
        engle_granger_results.append(eg)

        is_i1_pair = i1_map.get(TARGET_VARIABLE, False) and i1_map.get(pred, False)
        if is_i1_pair and ("error" not in eg) and eg.get("cointegrated_5pct", False):
            ecm = run_ecm_suitability(df[TARGET_VARIABLE], df[pred], TARGET_VARIABLE, pred)
        else:
            ecm = {
                "pair": f"{pred} -> {TARGET_VARIABLE}",
                "ecm_viable": False,
                "reason": "Pair not I(1)-I(1) cointegrated at 5%",
                "effective_nobs": int(pd.concat([df[TARGET_VARIABLE], df[pred]], axis=1).dropna().shape[0]),
            }
        ecm_results.append(ecm)

        if verbose:
            print(f"\n  Engle-Granger {pred} -> {TARGET_VARIABLE}: p={eg.get('p_value', 'N/A')}, coint={eg.get('cointegrated_5pct', 'N/A')}")
            print(f"    ECM viable: {ecm.get('ecm_viable', 'N/A')}")

    johansen_result = run_johansen_system(system_df)
    vecm_result = run_vecm_suitability(system_df, johansen_result)

    if verbose:
        print("\n  Johansen system:")
        print(
            "    Rank(trace 5%)="
            f"{johansen_result.get('rank_trace_5pct', 'N/A')}, "
            f"Rank(maxeig 5%)={johansen_result.get('rank_maxeig_5pct', 'N/A')}"
        )
        print(f"    VECM viable: {vecm_result.get('vecm_viable', 'N/A')}")

    return {
        "engle_granger": engle_granger_results,
        "ecm_suitability": ecm_results,
        "johansen": johansen_result,
        "vecm_suitability": vecm_result,
    }
