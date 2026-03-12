#!/usr/bin/env python3
"""Generate predictor-screening, diagnostics, and robustness appendix artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from reserves_project.apps.reserves_diagnostics.config import RESERVE_DATA_SOURCES
from reserves_project.config.paths import DATA_DIR
from reserves_project.config.varsets import TARGET_VAR, TRAIN_END, VALID_END, VARIABLE_SETS
from reserves_project.diagnostics.io_utils import build_variable_quality, load_panel
from reserves_project.diagnostics.phase2_stationarity import run_phase2
from reserves_project.diagnostics.phase3_temporal import run_phase3
from reserves_project.eval.dma import augment_with_dma_dms
from reserves_project.eval.unified_evaluator import (
    NaiveExogForecaster,
    RollingOriginEvaluator,
    build_models,
    summarize_results,
)
from reserves_project.utils.run_manifest import write_latest_pointer, write_run_manifest

DEFAULT_OUTPUT_DIR = DATA_DIR / "predictor_robustness"
DEFAULT_MODEL_POOL = ["Naive", "ARIMA", "VECM", "BVAR", "XGBoost", "MS-VAR", "MS-VECM"]

DISPLAY_NAMES = {
    "gross_reserves_usd_m": "Gross reserves",
    "import_cover_months": "Import cover",
    "fx_reserves_usd_m": "FX reserves",
    "imf_position_usd_m": "IMF reserve position",
    "sdrs_usd_m": "SDRs",
    "gold_usd_m": "Gold holdings",
    "exports_usd_m": "Exports",
    "imports_usd_m": "Imports",
    "tourism_usd_m": "Tourism earnings",
    "remittances_usd_m": "Remittances",
    "cse_net_usd_m": "CSE net flows",
    "m0_lkr_m": "Reserve money (M0)",
    "m2_lkr_m": "Broad money (M2, LKR)",
    "usd_lkr": "USD/LKR",
    "inflation_yoy_pct": "Inflation (YoY)",
    "govt_short_term_usd_m": "Govt short-term debt (USD)",
    "total_short_term_usd_m": "Total short-term debt (USD)",
    "portfolio_liabilities_usd_b": "Portfolio liabilities",
    "total_short_term_debt_lkr_m": "Short-term debt (LKR)",
    "trade_balance_usd_m": "Trade balance",
    "ca_proxy_usd_m": "Current-account proxy",
    "net_usable_reserves_usd_m": "Net usable reserves",
    "gg_ratio": "Guidotti-Greenspan ratio",
    "m2_usd_m": "Broad money (M2, USD)",
    "reserve_change_usd_m": "Reserve change",
    "reserve_change_pct": "Reserve change (%)",
    "total_short_term_debt_usd_m": "Short-term debt (USD converted)",
    "portfolio_liabilities_usd_m": "Portfolio liabilities (USD m)",
}

SOURCE_OVERRIDES = {
    "fx_reserves_usd_m": "Reserve Assets (CBSL Official)",
    "imf_position_usd_m": "Reserve Assets (CBSL Official)",
    "sdrs_usd_m": "Reserve Assets (CBSL Official)",
    "gold_usd_m": "Reserve Assets (CBSL Official)",
    "tourism_usd_m": "Tourism Earnings",
    "m0_lkr_m": "Monetary Aggregates (M0, M2)",
    "m2_lkr_m": "Monetary Aggregates (M0, M2)",
    "inflation_yoy_pct": "NCPI Inflation",
    "portfolio_liabilities_usd_b": "Int'l Investment Position",
    "total_short_term_debt_lkr_m": "Central Government Debt",
    "trade_balance_usd_m": "Derived: exports minus imports",
    "ca_proxy_usd_m": "Derived: exports + remittances + tourism - imports",
    "net_usable_reserves_usd_m": "Derived: gross reserves less PBOC swap",
    "gg_ratio": "Derived: gross reserves / short-term debt",
    "m2_usd_m": "Derived: M2 in LKR / USDLKR",
    "reserve_change_usd_m": "Derived: first difference of gross reserves",
    "reserve_change_pct": "Derived: pct change of gross reserves",
    "total_short_term_debt_usd_m": "Derived: short-term debt in LKR / USDLKR",
    "portfolio_liabilities_usd_m": "Derived: portfolio liabilities (USD b x 1000)",
}

SCREENING_DECISIONS = {
    "gross_reserves_usd_m": ("Target", "Dependent variable in every specification."),
    "import_cover_months": ("Excluded", "Reserve-adequacy ratio, not a forecasting regressor."),
    "fx_reserves_usd_m": ("Excluded", "Reserve subcomponent with short 2013+ coverage and mechanical overlap with the target."),
    "imf_position_usd_m": ("Excluded", "Reserve subcomponent with short coverage and mechanical overlap with the target."),
    "sdrs_usd_m": ("Excluded", "Reserve subcomponent with short coverage and mechanical overlap with the target."),
    "gold_usd_m": ("Excluded", "Reserve subcomponent dominated by valuation effects and short 2013+ coverage."),
    "exports_usd_m": ("Main text", "Core gross-flow inflow; retained in BoP, PCA, and Full sets."),
    "imports_usd_m": ("Main text", "Core gross-flow outflow; retained in BoP, PCA, and Full sets."),
    "tourism_usd_m": ("Main text", "Sri Lanka-specific external revenue channel; retained in BoP and Full sets."),
    "remittances_usd_m": ("Main text", "Stable FX inflow; retained in BoP, PCA, and Full sets."),
    "cse_net_usd_m": ("Appendix only", "Monthly capital-flow proxy used in omitted-variable robustness; excluded from the main horse race to preserve the longer baseline window."),
    "m0_lkr_m": ("Excluded", "Reserve money alternative considered but M2 aligns more directly with the reserve-adequacy literature."),
    "m2_lkr_m": ("Excluded", "Converted to USD and represented by m2_usd_m in the main specifications."),
    "usd_lkr": ("Main text", "Exchange-rate intervention channel; retained in Parsimonious, Monetary, and Full sets."),
    "inflation_yoy_pct": ("Excluded", "Only 71 usable observations in the merged panel, too short for the baseline horse race."),
    "govt_short_term_usd_m": ("Excluded", "All missing in the merged monthly panel."),
    "total_short_term_usd_m": ("Excluded", "All missing in the merged monthly panel."),
    "portfolio_liabilities_usd_b": ("Appendix only", "Liability proxy from the IIP used in the omitted-variable appendix; shorter 2012+ coverage keeps it out of the main text sets."),
    "total_short_term_debt_lkr_m": ("Appendix only", "Debt-pressure proxy used in omitted-variable robustness after USD conversion; shorter coverage keeps it out of the main text sets."),
    "trade_balance_usd_m": ("Main text", "Aggregate current-account proxy retained in Parsimonious and Full sets."),
    "ca_proxy_usd_m": ("Excluded", "Linear combination of BoP components already tested separately; excluded to avoid double counting."),
    "net_usable_reserves_usd_m": ("Excluded", "Alternative reserve target, not an explanatory predictor."),
    "gg_ratio": ("Excluded", "Unavailable because the USD short-term debt denominator is missing in the merged panel."),
    "m2_usd_m": ("Main text", "Monetary/liquidity proxy retained in Monetary, PCA, and Full sets."),
    "reserve_change_usd_m": ("Excluded", "Transformation of the dependent variable used for diagnostics and state equations, not as a standalone predictor."),
    "reserve_change_pct": ("Excluded", "Target transformation only."),
}

VARSET_LABELS = {
    "parsimonious": "Parsimonious",
    "bop": "BoP",
    "monetary": "Monetary",
    "pca": "PCA",
    "full": "Full",
}


@dataclass(frozen=True)
class EvaluationSpec:
    key: str
    label: str
    column_map: Dict[str, str]
    family: str
    note: str

    @property
    def model_columns(self) -> List[str]:
        return list(self.column_map.keys())


def _latex_escape(value: Any) -> str:
    text = "--" if value is None else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _format_number(value: Any, decimals: int = 1) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{decimals}f}"


def _longtable_latex(
    df: pd.DataFrame,
    columns: List[str],
    headers: List[str],
    caption: str,
    label: str,
    colspec: str,
    notes: str,
    decimals: Dict[str, int] | None = None,
) -> str:
    decimals = decimals or {}
    lines = [
        rf"\begin{{longtable}}{{{colspec}}}",
        rf"\caption{{{caption}}}\label{{{label}}}\\",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
        r"\endfirsthead",
        rf"\multicolumn{{{len(columns)}}}{{l}}{{\textit{{Continued from previous page}}}}\\",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        rf"\multicolumn{{{len(columns)}}}{{r}}{{\textit{{Continued on next page}}}}\\",
        r"\endfoot",
        r"\bottomrule",
        rf"\multicolumn{{{len(columns)}}}{{p{{0.94\textwidth}}}}{{\footnotesize {notes}}}\\",
        r"\endlastfoot",
    ]

    for _, row in df.iterrows():
        cells = []
        for col in columns:
            value = row[col]
            if col in decimals:
                rendered = _format_number(value, decimals[col])
            elif isinstance(value, (float, np.floating)):
                rendered = _format_number(value, 1)
            else:
                rendered = "--" if pd.isna(value) else str(value)
            cells.append(_latex_escape(rendered))
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\end{longtable}")
    return "\n".join(lines)


def _simple_table_latex(
    df: pd.DataFrame,
    columns: List[str],
    headers: List[str],
    caption: str,
    label: str,
    colspec: str,
    notes: str,
) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        cells = [_latex_escape("--" if pd.isna(row[col]) else row[col]) for col in columns]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{tablenotes}",
            r"\footnotesize",
            rf"\item Notes: {notes}",
            r"\end{tablenotes}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def _build_source_map() -> Dict[str, str]:
    source_map: Dict[str, str] = {}
    for source_name, payload in RESERVE_DATA_SOURCES.items():
        for column in payload.get("value_cols") or []:
            source_map.setdefault(column, source_name)
    source_map.update(SOURCE_OVERRIDES)
    return source_map


def _varset_membership(variable: str) -> str:
    memberships: List[str] = []
    for key, cfg in VARIABLE_SETS.items():
        listed: set[str] = set()
        for field in ("arima_exog", "vecm_system", "var_system", "source_vars"):
            values = cfg.get(field) or []
            listed.update(values)
        if variable == cfg.get("target"):
            memberships.append(f"{VARSET_LABELS.get(key, key)} (target)")
        elif variable in listed:
            memberships.append(VARSET_LABELS.get(key, key))
    return ", ".join(memberships)


def _predictor_screening(panel: pd.DataFrame) -> pd.DataFrame:
    variables = [col for col in panel.columns if col != "date"]
    quality = build_variable_quality(panel, variables).set_index("variable")
    source_map = _build_source_map()
    rows = []

    for var in variables:
        series = panel[var]
        non_null = series.dropna()
        status, reason = SCREENING_DECISIONS.get(var, ("Excluded", "Not retained after predictor screening."))
        memberships = _varset_membership(var)
        if status == "Appendix only":
            if var == "cse_net_usd_m":
                memberships = "Appendix robustness: BoP + CSE"
            elif var == "portfolio_liabilities_usd_b":
                memberships = "Appendix robustness: BoP + debt + portfolio"
            elif var == "total_short_term_debt_lkr_m":
                memberships = "Appendix robustness: BoP + debt"
        elif status == "Excluded" and memberships:
            status = "Main text"

        rows.append(
            {
                "variable": var,
                "Variable": DISPLAY_NAMES.get(var, var),
                "Source": source_map.get(var, "Merged/derived"),
                "First": non_null.index.min().strftime("%Y:%m") if len(non_null) else "--",
                "Last": non_null.index.max().strftime("%Y:%m") if len(non_null) else "--",
                "Obs": int(non_null.shape[0]),
                "Coverage": float(quality.loc[var, "coverage_pct"]),
                "Screening_Status": status,
                "Included_In": memberships or "--",
                "Reason": reason,
                "is_usable": bool(quality.loc[var, "is_usable"]),
                "quality_status": quality.loc[var, "status"],
            }
        )

    return pd.DataFrame(rows)


def _collect_phase_results(rows: Iterable[Dict[str, Any]], prefix: str) -> pd.DataFrame:
    out_rows = []
    for row in rows:
        if row.get("error"):
            continue
        record = {f"{prefix}_{k}": v for k, v in row.items() if k != "variable"}
        record["variable"] = row["variable"]
        out_rows.append(record)
    return pd.DataFrame(out_rows)


def _candidate_diagnostics(panel: pd.DataFrame, screening_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    usable_vars = screening_df.loc[screening_df["is_usable"], "variable"].tolist()
    phase2 = run_phase2(panel, usable_vars, verbose=False)
    phase3 = run_phase3(panel, usable_vars, verbose=False)

    integration_df = pd.DataFrame(phase2["integration_summary"])
    kpss_df = _collect_phase_results(phase2["kpss"], "kpss")

    stationarity_df = integration_df.merge(
        kpss_df[["variable", "kpss_p_value"]],
        on="variable",
        how="left",
    )
    stationarity_df = stationarity_df.merge(
        screening_df[["variable", "Variable", "Screening_Status", "Included_In"]],
        on="variable",
        how="left",
    )
    stationarity_df = stationarity_df[
        [
            "variable",
            "Variable",
            "effective_nobs",
            "adf_p_value",
            "kpss_p_value",
            "za_break_date",
            "integration_order",
            "Screening_Status",
            "Included_In",
        ]
    ].sort_values("Variable")

    acf_df = _collect_phase_results(phase3["acf_pacf"], "acf")
    lb_df = _collect_phase_results(phase3["ljungbox"], "lb")
    stl_df = _collect_phase_results(phase3["stl"], "stl")

    temporal_df = acf_df.merge(lb_df, on="variable", how="left")
    temporal_df = temporal_df.merge(stl_df, on="variable", how="left")
    temporal_df = temporal_df.merge(
        screening_df[["variable", "Variable", "Screening_Status", "Included_In"]],
        on="variable",
        how="left",
    )
    temporal_df = temporal_df[
        [
            "variable",
            "Variable",
            "acf_effective_nobs",
            "acf_acf_lag1",
            "lb_p_12",
            "stl_trend_strength",
            "stl_seasonal_strength",
            "stl_has_seasonality",
            "Screening_Status",
            "Included_In",
        ]
    ].sort_values("Variable")

    diagnostics_payload = {
        "phase2": phase2,
        "phase3": phase3,
        "usable_variables": usable_vars,
    }
    return stationarity_df, temporal_df, diagnostics_payload


def _estimate_train_month_effect(series: pd.Series, train_end: pd.Timestamp, period: int = 12) -> pd.Series:
    train_series = series.loc[series.index <= train_end].dropna()
    if len(train_series) < 2 * period:
        return pd.Series(0.0, index=range(1, 13))
    stl = STL(train_series, period=period, robust=True)
    result = stl.fit()
    seasonal = pd.Series(result.seasonal, index=train_series.index)
    month_effect = seasonal.groupby(seasonal.index.month).mean()
    return month_effect.reindex(range(1, 13)).fillna(0.0)


def _appendix_panel(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = panel.copy()
    factor_rows = []

    safe_rate = out["usd_lkr"].replace(0, np.nan)
    if {"total_short_term_debt_lkr_m", "usd_lkr"}.issubset(out.columns):
        out["total_short_term_debt_usd_m"] = out["total_short_term_debt_lkr_m"] / safe_rate
    if "portfolio_liabilities_usd_b" in out.columns:
        out["portfolio_liabilities_usd_m"] = out["portfolio_liabilities_usd_b"] * 1000.0

    seasonal_specs = {
        "trade_balance_usd_m_stl_adj": "trade_balance_usd_m",
        "exports_usd_m_stl_adj": "exports_usd_m",
        "imports_usd_m_stl_adj": "imports_usd_m",
        "tourism_usd_m_stl_adj": "tourism_usd_m",
    }
    for adjusted_col, source_col in seasonal_specs.items():
        if source_col not in out.columns:
            continue
        month_effect = _estimate_train_month_effect(out[source_col], TRAIN_END)
        adjustment = pd.Series(out.index.month, index=out.index).map(month_effect).astype(float)
        out[adjusted_col] = out[source_col] - adjustment
        for month, factor in month_effect.items():
            factor_rows.append(
                {
                    "adjusted_column": adjusted_col,
                    "source_column": source_col,
                    "month": int(month),
                    "seasonal_factor": float(factor),
                    "train_end": TRAIN_END.strftime("%Y-%m-%d"),
                }
            )

    return out, pd.DataFrame(factor_rows)


EVALUATION_SPECS = [
    EvaluationSpec(
        key="bop",
        label="BoP baseline",
        column_map={
            TARGET_VAR: TARGET_VAR,
            "exports_usd_m": "exports_usd_m",
            "imports_usd_m": "imports_usd_m",
            "remittances_usd_m": "remittances_usd_m",
            "tourism_usd_m": "tourism_usd_m",
        },
        family="omitted_variable",
        note="Baseline balance-of-payments specification.",
    ),
    EvaluationSpec(
        key="bop_cse",
        label="BoP + CSE flows",
        column_map={
            TARGET_VAR: TARGET_VAR,
            "exports_usd_m": "exports_usd_m",
            "imports_usd_m": "imports_usd_m",
            "remittances_usd_m": "remittances_usd_m",
            "tourism_usd_m": "tourism_usd_m",
            "cse_net_usd_m": "cse_net_usd_m",
        },
        family="omitted_variable",
        note="Adds the CSE portfolio-flow proxy available in the merged monthly panel.",
    ),
    EvaluationSpec(
        key="bop_debt",
        label="BoP + short-term debt",
        column_map={
            TARGET_VAR: TARGET_VAR,
            "exports_usd_m": "exports_usd_m",
            "imports_usd_m": "imports_usd_m",
            "remittances_usd_m": "remittances_usd_m",
            "tourism_usd_m": "tourism_usd_m",
            "total_short_term_debt_usd_m": "total_short_term_debt_usd_m",
        },
        family="omitted_variable",
        note="Adds short-term debt converted from LKR using the monthly USD/LKR rate.",
    ),
    EvaluationSpec(
        key="bop_debt_portfolio",
        label="BoP + debt + portfolio liabilities",
        column_map={
            TARGET_VAR: TARGET_VAR,
            "exports_usd_m": "exports_usd_m",
            "imports_usd_m": "imports_usd_m",
            "remittances_usd_m": "remittances_usd_m",
            "tourism_usd_m": "tourism_usd_m",
            "total_short_term_debt_usd_m": "total_short_term_debt_usd_m",
            "portfolio_liabilities_usd_m": "portfolio_liabilities_usd_m",
        },
        family="omitted_variable",
        note="Adds both the converted short-term debt proxy and portfolio liabilities from the IIP.",
    ),
    EvaluationSpec(
        key="parsimonious",
        label="Parsimonious baseline",
        column_map={
            TARGET_VAR: TARGET_VAR,
            "trade_balance_usd_m": "trade_balance_usd_m",
            "usd_lkr": "usd_lkr",
        },
        family="transformation",
        note="Baseline parsimonious specification.",
    ),
    EvaluationSpec(
        key="parsimonious_stl_adjusted",
        label="Parsimonious + STL-adjusted trade balance",
        column_map={
            TARGET_VAR: TARGET_VAR,
            "trade_balance_usd_m": "trade_balance_usd_m_stl_adj",
            "usd_lkr": "usd_lkr",
        },
        family="transformation",
        note="Uses a train-period STL month-of-year adjustment for the trade balance.",
    ),
    EvaluationSpec(
        key="bop_stl_adjusted",
        label="BoP + STL-adjusted seasonal flows",
        column_map={
            TARGET_VAR: TARGET_VAR,
            "exports_usd_m": "exports_usd_m_stl_adj",
            "imports_usd_m": "imports_usd_m_stl_adj",
            "remittances_usd_m": "remittances_usd_m",
            "tourism_usd_m": "tourism_usd_m_stl_adj",
        },
        family="transformation",
        note="Uses train-period STL month-of-year adjustments for exports, imports, and tourism.",
    ),
]


def _build_eval_dataset(panel: pd.DataFrame, spec: EvaluationSpec) -> pd.DataFrame:
    source_cols = list(dict.fromkeys(spec.column_map.values()))
    missing = [col for col in source_cols if col not in panel.columns]
    if missing:
        raise KeyError(f"Missing columns for {spec.key}: {missing}")

    data = panel[source_cols].copy().rename(columns={v: k for k, v in spec.column_map.items()})
    ordered_cols = spec.model_columns
    data = data[ordered_cols]

    exog_cols = [c for c in ordered_cols if c != TARGET_VAR]
    for col in exog_cols:
        data[col] = data[col].ffill(limit=3)
    data = data.dropna(subset=ordered_cols).sort_index()
    return data


def _evaluate_spec(
    panel: pd.DataFrame,
    spec: EvaluationSpec,
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = _build_eval_dataset(panel, spec)
    exog_cols = [col for col in data.columns if col != TARGET_VAR]
    models = build_models(
        TARGET_VAR,
        exog_cols,
        include_bvar=not args.exclude_bvar,
        include_ms=True,
        include_lstm=False,
        include_xgb=not args.exclude_xgb,
        include_xgb_quantile=False,
        include_llsv=False,
        include_bop=False,
        xgb_params=None,
        lstm_params=None,
    )
    models = [model for model in models if model.name in args.model_pool]

    evaluator = RollingOriginEvaluator(
        data=data,
        target_col=TARGET_VAR,
        exog_cols=exog_cols,
        models=models,
        horizons=[1],
        train_end=TRAIN_END,
        valid_end=VALID_END,
        refit_interval=args.refit_interval,
        exog_mode="forecast",
        exog_forecaster=NaiveExogForecaster(),
    )
    results = evaluator.run()
    dma_weights = pd.DataFrame()
    if args.include_dma:
        results, dma_weights = augment_with_dma_dms(
            results,
            alpha=args.dma_alpha,
            variance_window=args.dma_variance_window,
            warmup_periods=args.dma_warmup_periods,
            min_model_obs=args.dma_min_model_obs,
            model_pool=args.model_pool,
        )

    train_series = data.loc[data.index <= TRAIN_END, TARGET_VAR]
    summary = summarize_results(results, train_series, window_mode="common_dates", segment_keys=["all"])
    summary["spec_key"] = spec.key
    summary["spec_label"] = spec.label
    summary["family"] = spec.family
    summary["spec_note"] = spec.note
    summary["n_model_vars"] = len(data.columns)
    summary["data_start"] = data.index.min()
    summary["data_end"] = data.index.max()

    results_path = output_dir / f"horse_race_forecasts_{spec.key}.csv"
    summary_path = output_dir / f"horse_race_summary_{spec.key}.csv"
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    if not dma_weights.empty:
        dma_weights.to_csv(output_dir / f"dma_weights_{spec.key}.csv", index=False)

    return data, results, summary


def _rank_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    test_df = summary_df[(summary_df["split"] == "test") & (summary_df["horizon"] == 1)].copy()
    test_df = test_df.sort_values(["rmse", "mae", "model"], kind="mergesort").reset_index(drop=True)
    test_df["rank"] = np.arange(1, len(test_df) + 1)
    return test_df


def _build_rank_matrix(detail_df: pd.DataFrame, spec_order: List[str]) -> pd.DataFrame:
    matrix = detail_df.copy()
    matrix["cell"] = matrix.apply(
        lambda row: f"{int(row['rank'])} [{row['rmse']:.1f}]",
        axis=1,
    )
    pivot = matrix.pivot(index="model", columns="spec_label", values="cell")
    model_order = (
        detail_df.sort_values(["spec_key", "rank", "model"])
        .groupby("model")["rank"]
        .mean()
        .sort_values()
        .index.tolist()
    )
    ordered_labels = detail_df.drop_duplicates("spec_key").set_index("spec_key")["spec_label"].reindex(spec_order).tolist()
    pivot = pivot.reindex(index=model_order, columns=ordered_labels)
    return pivot.reset_index().rename(columns={"model": "Model"})


def _sample_note(detail_df: pd.DataFrame, spec_order: List[str]) -> str:
    rows = []
    for spec_key in spec_order:
        subset = detail_df[detail_df["spec_key"] == spec_key]
        if subset.empty:
            continue
        first = subset.iloc[0]
        start = pd.Timestamp(first["effective_start"]).strftime("%Y:%m") if pd.notna(first["effective_start"]) else "--"
        end = pd.Timestamp(first["effective_end"]).strftime("%Y:%m") if pd.notna(first["effective_end"]) else "--"
        rows.append(f"{first['spec_label']}: n={int(first['n'])}, window={start}--{end}")
    return "; ".join(rows)


def _rank_stability(detail_df: pd.DataFrame, baseline_key: str) -> List[Dict[str, Any]]:
    stats = []
    baseline = detail_df[detail_df["spec_key"] == baseline_key].set_index("model")["rank"]
    for spec_key, subset in detail_df.groupby("spec_key"):
        current = subset.set_index("model")["rank"]
        common = baseline.index.intersection(current.index)
        if len(common) < 2:
            corr = np.nan
        else:
            corr = float(baseline.loc[common].corr(current.loc[common], method="spearman"))
        stats.append(
            {
                "baseline_spec": baseline_key,
                "spec_key": spec_key,
                "spec_label": subset["spec_label"].iloc[0],
                "top_model": subset.sort_values("rank").iloc[0]["model"],
                "top_rmse": float(subset.sort_values("rank").iloc[0]["rmse"]),
                "spearman_rank_corr_vs_baseline": corr,
            }
        )
    return stats


def _write_table_files(
    screening_df: pd.DataFrame,
    stationarity_df: pd.DataFrame,
    temporal_df: pd.DataFrame,
    omitted_detail: pd.DataFrame,
    transform_detail: pd.DataFrame,
    factor_df: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, str]:
    screening_latex = _longtable_latex(
        screening_df,
        columns=["Variable", "Source", "First", "Last", "Obs", "Screening_Status", "Reason"],
        headers=["Variable", "Source", "First", "Last", "Obs", "Status", "Reason"],
        caption="Predictor screening across all candidate panel columns",
        label="tab:predictor_screening",
        colspec="p{2.8cm}p{2.9cm}cccp{1.9cm}p{6.0cm}",
        notes=(
            "Candidate columns are the 26 non-date fields in the merged forecasting panel. "
            "Status distinguishes dependent variables retained in the main horse race, appendix-only robustness variables, and excluded series. "
            "Quarterly debt and IIP series are forward-filled in the merged panel before the forecasting-stage missing-data rule is applied."
        ),
    )

    stationarity_latex = _longtable_latex(
        stationarity_df,
        columns=["Variable", "effective_nobs", "adf_p_value", "kpss_p_value", "za_break_date", "integration_order", "Screening_Status"],
        headers=["Variable", "Obs", "ADF p", "KPSS p", "ZA break", "Integration order", "Status"],
        caption="Stationarity diagnostics for usable candidate series",
        label="tab:stationarity_outputs",
        colspec="p{3.0cm}ccccp{5.0cm}p{1.9cm}",
        notes=(
            "Reported tests use the same 5\% decision rules as the diagnostics pipeline. "
            "ADF and KPSS are run on levels; ZA denotes the Zivot-Andrews single-break unit-root test. "
            "Series with insufficient observations or all-missing values are documented in Table~\\ref{tab:predictor_screening} instead."
        ),
        decimals={"adf_p_value": 4, "kpss_p_value": 4},
    )

    temporal_out = temporal_df.copy()
    temporal_out["Seasonal_Flag"] = np.where(temporal_out["stl_has_seasonality"], "Yes", "No")
    temporal_latex = _longtable_latex(
        temporal_out,
        columns=["Variable", "acf_effective_nobs", "acf_acf_lag1", "lb_p_12", "stl_trend_strength", "stl_seasonal_strength", "Seasonal_Flag", "Screening_Status"],
        headers=["Variable", "Obs", "ACF(1)", "Q(12) p", "Trend", "Seasonal", "Seasonal?", "Status"],
        caption="Temporal-dependence and seasonality diagnostics for usable candidate series",
        label="tab:seasonality_outputs",
        colspec="p{3.0cm}ccccccp{1.9cm}",
        notes=(
            "STL seasonality is flagged when seasonal strength exceeds 0.3, which is the threshold implemented in the codebase. "
            "Q(12) reports the Ljung-Box p-value at lag 12. "
            "The appendix seasonal-adjustment robustness uses train-period month-of-year factors for the seasonal BoP variables."
        ),
        decimals={
            "acf_acf_lag1": 3,
            "lb_p_12": 4,
            "stl_trend_strength": 3,
            "stl_seasonal_strength": 3,
        },
    )

    omitted_specs = ["bop", "bop_cse", "bop_debt", "bop_debt_portfolio"]
    omitted_matrix = _build_rank_matrix(omitted_detail, omitted_specs)
    omitted_latex = _simple_table_latex(
        omitted_matrix,
        columns=list(omitted_matrix.columns),
        headers=list(omitted_matrix.columns),
        caption="Appendix omitted-variable horse race: RMSE ranks by BoP augmentation",
        label="tab:omitted_variable_robustness",
        colspec="lcccc",
        notes=(
            "Cells report test-period rank with common-dates RMSE in brackets for the one-step-ahead forecast. "
            "The added appendix predictors are cse\\_net\\_usd\\_m, total short-term debt converted to USD (total\\_short\\_term\\_debt\\_lkr\\_m / usd\\_lkr), and portfolio liabilities converted to USD millions (portfolio\\_liabilities\\_usd\\_b $\\times$ 1000). "
            + _sample_note(omitted_detail, omitted_specs)
        ),
    )

    transform_specs = ["parsimonious", "parsimonious_stl_adjusted", "bop", "bop_stl_adjusted"]
    transform_matrix = _build_rank_matrix(transform_detail, transform_specs)
    adjusted_vars = sorted(factor_df["source_column"].unique().tolist())
    transformed_latex = _simple_table_latex(
        transform_matrix,
        columns=list(transform_matrix.columns),
        headers=list(transform_matrix.columns),
        caption="Alternative seasonal-transformation robustness",
        label="tab:transformation_robustness",
        colspec="lcccc",
        notes=(
            "Cells report test-period rank with common-dates RMSE in brackets for the one-step-ahead forecast. "
            "STL month-of-year factors are estimated on the training sample only (through 2019-12) and applied to the seasonal variables "
            + ", ".join(_latex_escape(DISPLAY_NAMES.get(v, v)) for v in adjusted_vars)
            + ". "
            + _sample_note(transform_detail, transform_specs)
        ),
    )

    outputs = {
        "screening_tex": output_dir / "table_a11_predictor_screening.tex",
        "stationarity_tex": output_dir / "table_a12_stationarity_outputs.tex",
        "temporal_tex": output_dir / "table_a13_seasonality_outputs.tex",
        "omitted_tex": output_dir / "table_a14_omitted_variable_robustness.tex",
        "transform_tex": output_dir / "table_a15_transformation_robustness.tex",
    }
    outputs["screening_tex"].write_text(screening_latex)
    outputs["stationarity_tex"].write_text(stationarity_latex)
    outputs["temporal_tex"].write_text(temporal_latex)
    outputs["omitted_tex"].write_text(omitted_latex)
    outputs["transform_tex"].write_text(transformed_latex)
    return {key: str(path) for key, path in outputs.items()}


def _serialize_timestamp(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def run_appendix(args: argparse.Namespace) -> Dict[str, Any]:
    output_root = None
    if args.output_root:
        output_root = Path(args.output_root)
    elif args.run_id:
        output_root = DATA_DIR / "outputs" / args.run_id

    output_dir = Path(args.output_dir)
    if output_root is not None:
        output_dir = output_root / "predictor_robustness"
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = load_panel()
    appendix_panel, factor_df = _appendix_panel(panel)
    screening_df = _predictor_screening(panel)
    stationarity_df, temporal_df, diagnostics_payload = _candidate_diagnostics(panel, screening_df)

    screening_df.to_csv(output_dir / "predictor_screening.csv", index=False)
    stationarity_df.to_csv(output_dir / "candidate_stationarity_summary.csv", index=False)
    temporal_df.to_csv(output_dir / "candidate_temporal_summary.csv", index=False)
    factor_df.to_csv(output_dir / "seasonal_adjustment_factors.csv", index=False)

    all_summaries: List[pd.DataFrame] = []
    detail_rows: List[pd.DataFrame] = []
    data_window_rows: List[Dict[str, Any]] = []

    for spec in EVALUATION_SPECS:
        data, results, summary = _evaluate_spec(appendix_panel, spec, args, output_dir)
        summary_detail = _rank_summary(summary)
        detail_rows.append(summary_detail)
        all_summaries.append(summary)
        data_window_rows.append(
            {
                "spec_key": spec.key,
                "spec_label": spec.label,
                "family": spec.family,
                "spec_note": spec.note,
                "n_model_vars": len(data.columns),
                "effective_start": data.index.min().strftime("%Y:%m"),
                "effective_end": data.index.max().strftime("%Y:%m"),
                "train_obs": int((data.index <= TRAIN_END).sum()),
                "validation_obs": int(((data.index > TRAIN_END) & (data.index <= VALID_END)).sum()),
                "test_obs": int((data.index > VALID_END).sum()),
                "models_run": ", ".join(sorted(results["model"].unique())),
            }
        )

    detail_df = pd.concat(detail_rows, ignore_index=True)
    full_summary_df = pd.concat(all_summaries, ignore_index=True)
    sample_windows_df = pd.DataFrame(data_window_rows)

    omitted_detail = detail_df[detail_df["family"] == "omitted_variable"].copy()
    transform_detail = detail_df[detail_df["spec_key"].isin(["parsimonious", "parsimonious_stl_adjusted", "bop", "bop_stl_adjusted"])].copy()

    detail_df.to_csv(output_dir / "horse_race_ranks_detail.csv", index=False)
    full_summary_df.to_csv(output_dir / "horse_race_summary_all.csv", index=False)
    sample_windows_df.to_csv(output_dir / "horse_race_sample_windows.csv", index=False)

    table_outputs = _write_table_files(
        screening_df=screening_df,
        stationarity_df=stationarity_df,
        temporal_df=temporal_df,
        omitted_detail=omitted_detail,
        transform_detail=transform_detail,
        factor_df=factor_df,
        output_dir=output_dir,
    )

    omitted_stability = _rank_stability(omitted_detail, baseline_key="bop")
    transform_stability = _rank_stability(transform_detail, baseline_key="bop")
    transform_stability.extend(_rank_stability(transform_detail, baseline_key="parsimonious"))

    seasonal_vars = temporal_df.loc[temporal_df["stl_has_seasonality"], "variable"].tolist()
    summary_json = {
        "screening": {
            "n_candidate_columns": int(screening_df.shape[0]),
            "n_main_text_predictors": int((screening_df["Screening_Status"] == "Main text").sum()),
            "n_appendix_only_predictors": int((screening_df["Screening_Status"] == "Appendix only").sum()),
            "n_excluded": int((screening_df["Screening_Status"] == "Excluded").sum()),
        },
        "diagnostics": {
            "n_usable_series": int(stationarity_df.shape[0]),
            "seasonal_variables_threshold_0_3": seasonal_vars,
        },
        "omitted_variable_stability": omitted_stability,
        "transformation_stability": transform_stability,
        "table_outputs": table_outputs,
    }
    with (output_dir / "predictor_robustness_summary.json").open("w") as f:
        json.dump(summary_json, f, indent=2, default=_serialize_timestamp)
    with (output_dir / "candidate_diagnostics_raw.json").open("w") as f:
        json.dump(diagnostics_payload, f, indent=2, default=_serialize_timestamp)

    config = {
        "output_dir": str(output_dir),
        "include_dma": bool(args.include_dma),
        "exclude_bvar": bool(args.exclude_bvar),
        "exclude_xgb": bool(args.exclude_xgb),
        "model_pool": args.model_pool,
        "refit_interval": int(args.refit_interval),
        "dma_alpha": float(args.dma_alpha),
        "dma_warmup_periods": int(args.dma_warmup_periods),
        "dma_variance_window": int(args.dma_variance_window),
        "dma_min_model_obs": int(args.dma_min_model_obs),
        "evaluation_specs": [spec.key for spec in EVALUATION_SPECS],
    }
    write_run_manifest(output_dir, config, extra={"tables": table_outputs})
    if output_root is not None and args.run_id:
        write_latest_pointer(DATA_DIR / "outputs", args.run_id, output_root, extra={"predictor_robustness": str(output_dir)})

    return {
        "output_dir": str(output_dir),
        "table_outputs": table_outputs,
        "summary": summary_json,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate predictor-selection appendix artifacts")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--refit-interval", type=int, default=12)
    parser.add_argument("--include-dma", action="store_true")
    parser.add_argument("--exclude-bvar", action="store_true")
    parser.add_argument("--exclude-xgb", action="store_true")
    parser.add_argument("--dma-alpha", type=float, default=0.99)
    parser.add_argument("--dma-warmup-periods", type=int, default=12)
    parser.add_argument("--dma-variance-window", type=int, default=24)
    parser.add_argument("--dma-min-model-obs", type=int, default=24)
    parser.add_argument(
        "--model-pool",
        default=",".join(DEFAULT_MODEL_POOL),
        help="Comma-separated model pool for evaluation and DMA/DMS construction.",
    )
    return parser


def main() -> Dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args()
    args.model_pool = [m.strip() for m in args.model_pool.split(",") if m.strip()]
    return run_appendix(args)


if __name__ == "__main__":
    main()
