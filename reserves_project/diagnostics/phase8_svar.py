"""Phase 8: exogeneity and SVAR short-run/sign-restriction diagnostics."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.svar_model import SVAR
from statsmodels.tsa.vector_ar.var_model import VAR

from .config import MIN_OBS_SVAR, PHASE8_SVAR_VARS, TARGET_VARIABLE


def _integration_map(phase2_results: dict) -> dict[str, str]:
    return {
        row.get("variable", ""): str(row.get("integration_order", ""))
        for row in phase2_results.get("integration_summary", [])
    }


def _transform_for_var(df: pd.DataFrame, variables: list[str], integ_map: dict[str, str]):
    transformed = {}
    transform_meta = {}
    for var in variables:
        series = df[var]
        if "I(1)" in integ_map.get(var, ""):
            transformed[var] = series.diff()
            transform_meta[var] = "first_difference"
        else:
            transformed[var] = series
            transform_meta[var] = "level"

    x = pd.DataFrame(transformed).dropna()
    return x, transform_meta


def _select_var_lag(data: pd.DataFrame) -> int:
    maxlags = min(8, max(2, len(data) // 10))
    try:
        lag_res = VAR(data).select_order(maxlags=maxlags)
        if lag_res.aic is not None and not np.isnan(lag_res.aic):
            return max(1, int(lag_res.aic))
    except Exception:
        pass
    return 2


def _run_exogeneity_tests(var_fit, variables: list[str]):
    rows = []
    for var in variables:
        causing = [v for v in variables if v != var]
        if not causing:
            continue
        try:
            test = var_fit.test_causality(caused=var, causing=causing, kind="f")
            p_value = float(test.pvalue)
            rows.append(
                {
                    "variable": var,
                    "caused_by": ", ".join(causing),
                    "p_value": round(p_value, 4),
                    "weakly_exogenous_5pct": bool(p_value >= 0.05),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "variable": var,
                    "caused_by": ", ".join(causing),
                    "error": str(exc),
                    "weakly_exogenous_5pct": None,
                }
            )
    return rows


def _build_recursive_a_matrix(k: int):
    a = np.empty((k, k), dtype=object)
    for i in range(k):
        for j in range(k):
            if i == j:
                a[i, j] = 1.0
            elif j > i:
                a[i, j] = 0.0
            else:
                a[i, j] = "E"
    return a


def _evaluate_sign_restrictions(irfs: np.ndarray, variables: list[str]):
    checks = []
    horizon_slice = slice(0, min(4, irfs.shape[0]))

    constraints = [
        ("usd_lkr", TARGET_VARIABLE, -1),
        ("imports_usd_m", TARGET_VARIABLE, -1),
        ("exports_usd_m", TARGET_VARIABLE, 1),
    ]

    idx = {v: i for i, v in enumerate(variables)}

    for shock_var, response_var, sign in constraints:
        if shock_var not in idx or response_var not in idx:
            checks.append(
                {
                    "restriction": f"{shock_var} shock -> {response_var}",
                    "status": "not_tested",
                    "reason": "Variable not in SVAR system",
                }
            )
            continue

        resp = irfs[horizon_slice, idx[response_var], idx[shock_var]]
        if len(resp) == 0:
            checks.append(
                {
                    "restriction": f"{shock_var} shock -> {response_var}",
                    "status": "not_tested",
                    "reason": "No IRF horizons",
                }
            )
            continue

        sign_hits = np.sum(np.sign(resp) == sign)
        sign_score = float(sign_hits / len(resp))
        checks.append(
            {
                "restriction": f"{shock_var} shock -> {response_var}",
                "expected_sign": "positive" if sign > 0 else "negative",
                "horizon_0_3_match_ratio": round(sign_score, 3),
                "passes_75pct_rule": bool(sign_score >= 0.75),
            }
        )

    return checks


def run_phase8(df: pd.DataFrame, usable_vars: list[str], phase2_results: dict, verbose=True):
    vars_for_svar = [v for v in PHASE8_SVAR_VARS if v in usable_vars]
    if len(vars_for_svar) < 3:
        return {
            "metadata": {"error": "Insufficient variables for SVAR", "variables": vars_for_svar},
            "exogeneity_tests": [],
            "svar_model": {"error": "Insufficient variables for SVAR"},
            "sign_restriction_checks": [],
        }

    integ_map = _integration_map(phase2_results)
    model_df, transform_meta = _transform_for_var(df, vars_for_svar, integ_map)

    if len(model_df) < MIN_OBS_SVAR:
        return {
            "metadata": {
                "error": "Insufficient observations after transformations",
                "variables": vars_for_svar,
                "effective_nobs": int(len(model_df)),
                "transformations": transform_meta,
            },
            "exogeneity_tests": [],
            "svar_model": {"error": "Insufficient observations"},
            "sign_restriction_checks": [],
        }

    try:
        lag_order = _select_var_lag(model_df)
        var_fit = VAR(model_df).fit(lag_order)
        exogeneity_tests = _run_exogeneity_tests(var_fit, vars_for_svar)
    except Exception as exc:
        return {
            "metadata": {
                "error": f"VAR fit failed: {exc}",
                "variables": vars_for_svar,
                "effective_nobs": int(len(model_df)),
                "transformations": transform_meta,
            },
            "exogeneity_tests": [],
            "svar_model": {"error": str(exc)},
            "sign_restriction_checks": [],
        }

    a_restr = _build_recursive_a_matrix(len(vars_for_svar))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            svar_model = SVAR(model_df.values, svar_type="A", A=a_restr)
            svar_fit = svar_model.fit(maxlags=lag_order, ic=None, trend="c", solver="bfgs", maxiter=300)

        irf_obj = svar_fit.irf(8)
        irfs = irf_obj.irfs
        sign_checks = _evaluate_sign_restrictions(irfs, vars_for_svar)

        a_est = np.array(svar_fit.A, dtype=float)
        impact = np.linalg.inv(a_est)
        converged = bool(getattr(svar_fit, "mle_retvals", {}).get("converged", True))

        svar_summary = {
            "variables": vars_for_svar,
            "effective_nobs": int(len(model_df)),
            "lag_order": int(lag_order),
            "converged": converged,
            "a_matrix": np.round(a_est, 4).tolist(),
            "impact_matrix": np.round(impact, 4).tolist(),
            "target_variable": TARGET_VARIABLE,
        }
    except Exception as exc:
        svar_summary = {
            "variables": vars_for_svar,
            "effective_nobs": int(len(model_df)),
            "lag_order": int(lag_order),
            "error": str(exc),
        }
        sign_checks = []

    if verbose:
        print("\n  SVAR exogeneity tests:")
        for row in exogeneity_tests:
            print(f"    {row.get('variable')}: p={row.get('p_value', 'N/A')} exogenous={row.get('weakly_exogenous_5pct', 'N/A')}")
        print(f"  SVAR model status: {'ok' if 'error' not in svar_summary else 'error'}")

    return {
        "metadata": {
            "variables": vars_for_svar,
            "effective_nobs": int(len(model_df)),
            "transformations": transform_meta,
        },
        "exogeneity_tests": exogeneity_tests,
        "svar_model": svar_summary,
        "sign_restriction_checks": sign_checks,
    }
