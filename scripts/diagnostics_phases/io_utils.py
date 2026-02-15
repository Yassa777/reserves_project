"""I/O helpers and data-quality checks for diagnostics."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Dict

import numpy as np
import pandas as pd

from .config import MERGED_DIR, OUTPUT_DIR


def _normalize_to_month_start(index: pd.Index) -> pd.DatetimeIndex:
    dt_index = pd.to_datetime(index)
    return dt_index.to_period("M").to_timestamp(how="start")


def load_panel() -> pd.DataFrame:
    """Load forecasting panel and repair known alignment gaps."""
    path = MERGED_DIR / "reserves_forecasting_panel.csv"
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    df.index = _normalize_to_month_start(df.index)

    # Known issue: usd_lkr can be entirely null when merged with month-end indexes.
    if "usd_lkr" in df.columns and df["usd_lkr"].notna().sum() == 0:
        fx_path = MERGED_DIR / "slfsi_monthly_panel.csv"
        if fx_path.exists():
            fx = pd.read_csv(fx_path, parse_dates=["date"]).set_index("date").sort_index()
            if "usd_lkr" in fx.columns:
                fx.index = _normalize_to_month_start(fx.index)
                df["usd_lkr"] = df["usd_lkr"].combine_first(fx["usd_lkr"])

    # Recompute m2_usd_m if missing and inputs are available.
    if {"m2_lkr_m", "usd_lkr", "m2_usd_m"}.issubset(df.columns):
        missing_mask = df["m2_usd_m"].isna()
        safe_rate = df["usd_lkr"].replace(0, np.nan)
        df.loc[missing_mask, "m2_usd_m"] = df.loc[missing_mask, "m2_lkr_m"] / safe_rate.loc[missing_mask]

    return df


def build_variable_quality(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """Profile coverage/quality for each candidate variable."""
    rows = []
    total = len(df)

    for var in variables:
        if var not in df.columns:
            rows.append(
                {
                    "variable": var,
                    "available": False,
                    "non_null_obs": 0,
                    "coverage_pct": 0.0,
                    "std_dev": np.nan,
                    "is_constant": False,
                    "is_usable": False,
                    "status": "missing_column",
                }
            )
            continue

        series = df[var].dropna()
        n = len(series)
        std = float(series.std()) if n > 1 else np.nan
        is_constant = bool(n > 1 and np.isclose(std, 0.0, atol=1e-12))

        if n == 0:
            status = "all_missing"
            usable = False
        elif n < 20:
            status = "insufficient_obs"
            usable = False
        elif is_constant:
            status = "constant_series"
            usable = False
        else:
            status = "ok"
            usable = True

        rows.append(
            {
                "variable": var,
                "available": True,
                "non_null_obs": int(n),
                "coverage_pct": round((n / total) * 100, 1) if total else 0.0,
                "std_dev": round(std, 6) if not np.isnan(std) else np.nan,
                "is_constant": is_constant,
                "is_usable": usable,
                "status": status,
            }
        )

    return pd.DataFrame(rows)


def make_serializable(obj: Any) -> Any:
    """Convert diagnostics result tree into JSON-safe types."""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj


def save_outputs(results: Dict[str, Any], quality_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Persist JSON + summary CSV outputs."""
    summary_dfs: Dict[str, pd.DataFrame] = {}

    json_path = OUTPUT_DIR / "diagnostic_results.json"
    with json_path.open("w") as f:
        json.dump(make_serializable(results), f, indent=2)

    quality_path = OUTPUT_DIR / "variable_quality_summary.csv"
    quality_df.to_csv(quality_path, index=False)
    summary_dfs["quality"] = quality_df

    int_summary = pd.DataFrame(results["phase2_stationarity"]["integration_summary"])
    int_summary.to_csv(OUTPUT_DIR / "integration_summary.csv", index=False)
    summary_dfs["integration"] = int_summary

    arch_rows = [
        {
            "variable": r["variable"],
            "arch_lm_stat": r.get("arch_lm_stat"),
            "arch_lm_pvalue": r.get("arch_lm_pvalue"),
            "has_arch_effects": r.get("has_arch_effects"),
        }
        for r in results["phase4_volatility"]["arch_lm"]
        if "error" not in r
    ]
    arch_summary = pd.DataFrame(arch_rows)
    arch_summary.to_csv(OUTPUT_DIR / "arch_summary.csv", index=False)
    summary_dfs["arch"] = arch_summary

    chow_rows = [
        {
            "variable": r["variable"],
            "break_date": r.get("break_date"),
            "f_statistic": r.get("f_statistic"),
            "p_value": r.get("p_value"),
            "break_confirmed": r.get("break_confirmed"),
        }
        for r in results["phase5_breaks"]["chow"]
        if "error" not in r
    ]
    chow_summary = pd.DataFrame(chow_rows)
    chow_summary.to_csv(OUTPUT_DIR / "chow_test_summary.csv", index=False)
    summary_dfs["chow"] = chow_summary

    gc_rows = [
        {
            "test": r["test"],
            "best_lag": r.get("best_lag"),
            "best_p_value": r.get("best_p_value"),
            "granger_causes": r.get("granger_causes"),
            "effective_nobs": r.get("effective_nobs"),
        }
        for r in results["phase6_relationships"]["granger_causality"]
        if "error" not in r
    ]
    gc_summary = pd.DataFrame(gc_rows)
    gc_summary.to_csv(OUTPUT_DIR / "granger_causality_summary.csv", index=False)
    summary_dfs["granger"] = gc_summary

    # Phase 7 summaries
    eg_rows = [
        {
            "pair": r.get("pair"),
            "coint_t_stat": r.get("coint_t_stat"),
            "p_value": r.get("p_value"),
            "cointegrated_5pct": r.get("cointegrated_5pct"),
            "effective_nobs": r.get("effective_nobs"),
        }
        for r in results.get("phase7_cointegration", {}).get("engle_granger", [])
        if "error" not in r
    ]
    eg_summary = pd.DataFrame(eg_rows)
    eg_summary.to_csv(OUTPUT_DIR / "cointegration_engle_granger_summary.csv", index=False)
    summary_dfs["cointegration_engle_granger"] = eg_summary

    ecm_rows = [
        {
            "pair": r.get("pair"),
            "ect_coef": r.get("ect_coef"),
            "ect_p_value": r.get("ect_p_value"),
            "ecm_viable": r.get("ecm_viable"),
            "effective_nobs": r.get("effective_nobs"),
        }
        for r in results.get("phase7_cointegration", {}).get("ecm_suitability", [])
        if "error" not in r
    ]
    ecm_summary = pd.DataFrame(ecm_rows)
    ecm_summary.to_csv(OUTPUT_DIR / "ecm_suitability_summary.csv", index=False)
    summary_dfs["ecm_suitability"] = ecm_summary

    joh = results.get("phase7_cointegration", {}).get("johansen", {})
    joh_summary = pd.DataFrame([joh]) if joh else pd.DataFrame()
    joh_summary.to_csv(OUTPUT_DIR / "johansen_summary.csv", index=False)
    summary_dfs["johansen"] = joh_summary

    vecm = results.get("phase7_cointegration", {}).get("vecm_suitability", {})
    vecm_summary = pd.DataFrame([vecm]) if vecm else pd.DataFrame()
    vecm_summary.to_csv(OUTPUT_DIR / "vecm_suitability_summary.csv", index=False)
    summary_dfs["vecm_suitability"] = vecm_summary

    # Phase 8 summaries
    exog_rows = [
        {
            "variable": r.get("variable"),
            "p_value": r.get("p_value"),
            "weakly_exogenous_5pct": r.get("weakly_exogenous_5pct"),
        }
        for r in results.get("phase8_svar", {}).get("exogeneity_tests", [])
        if "error" not in r
    ]
    exog_summary = pd.DataFrame(exog_rows)
    exog_summary.to_csv(OUTPUT_DIR / "svar_exogeneity_summary.csv", index=False)
    summary_dfs["svar_exogeneity"] = exog_summary

    sign_rows = [
        {
            "restriction": r.get("restriction"),
            "expected_sign": r.get("expected_sign"),
            "horizon_0_3_match_ratio": r.get("horizon_0_3_match_ratio"),
            "passes_75pct_rule": r.get("passes_75pct_rule"),
        }
        for r in results.get("phase8_svar", {}).get("sign_restriction_checks", [])
        if r.get("status") != "not_tested"
    ]
    sign_summary = pd.DataFrame(sign_rows)
    sign_summary.to_csv(OUTPUT_DIR / "svar_sign_restriction_summary.csv", index=False)
    summary_dfs["svar_signs"] = sign_summary

    svar_model = results.get("phase8_svar", {}).get("svar_model", {})
    svar_model_summary = pd.DataFrame([svar_model]) if svar_model else pd.DataFrame()
    svar_model_summary.to_csv(OUTPUT_DIR / "svar_model_summary.csv", index=False)
    summary_dfs["svar_model"] = svar_model_summary

    # Phase 9 summary
    bp_rows = []
    for r in results.get("phase9_multiple_breaks", {}).get("bai_perron", []):
        if "error" in r:
            continue
        bp_rows.append(
            {
                "variable": r.get("variable"),
                "effective_nobs": r.get("effective_nobs"),
                "optimal_break_count": r.get("optimal_break_count"),
                "break_dates": ", ".join(r.get("break_dates", [])),
                "bic_no_break": r.get("bic_no_break"),
                "bic_optimal": r.get("bic_optimal"),
                "rss_reduction_pct": r.get("rss_reduction_pct"),
                "multiple_breaks_detected": r.get("multiple_breaks_detected"),
            }
        )
    bp_summary = pd.DataFrame(bp_rows)
    bp_summary.to_csv(OUTPUT_DIR / "bai_perron_summary.csv", index=False)
    summary_dfs["bai_perron"] = bp_summary

    return summary_dfs
