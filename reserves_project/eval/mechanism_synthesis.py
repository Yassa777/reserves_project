"""Mechanism synthesis across disentangling, regime, IRF, and information-loss outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def _as_path(path_like: Path | str) -> Path:
    return path_like if isinstance(path_like, Path) else Path(path_like)


def _get_metric(df: pd.DataFrame, name: str) -> float:
    if df.empty or "metric" not in df.columns or "value" not in df.columns:
        return np.nan
    match = df[df["metric"] == name]
    if match.empty:
        return np.nan
    return float(match.iloc[0]["value"])


def load_disentangling_metrics(
    disentangling_dir: Path | str,
    aggregated_varset: str | None = None,
    disaggregated_varset: str | None = None,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Load forecast-gain metrics from disentangling outputs."""
    disentangling_dir = _as_path(disentangling_dir)
    rmse_long = pd.read_csv(disentangling_dir / "disentangling_rmse_long.csv")
    effects = pd.read_csv(disentangling_dir / "disentangling_effects.csv")

    summary_path = disentangling_dir / "disentangling_summary.json"
    if summary_path.exists():
        import json

        with open(summary_path, "r") as f:
            summary = json.load(f)
        models = summary.get("models") or sorted(rmse_long["model"].unique().tolist())
        varsets = summary.get("varsets") or sorted(rmse_long["varset"].unique().tolist())
    else:
        models = sorted(rmse_long["model"].unique().tolist())
        varsets = sorted(rmse_long["varset"].unique().tolist())

    if len(models) < 2 or len(varsets) < 2:
        raise RuntimeError("Disentangling artifacts do not contain a 2x2 design.")

    primary_model = models[0]
    challenger_model = models[1]

    pivot = rmse_long.pivot(index="model", columns="varset", values="rmse")
    architecture_gain = float(np.mean([pivot.loc[challenger_model, v] - pivot.loc[primary_model, v] for v in varsets]))
    interaction = float(
        effects.loc[effects["effect"] == "interaction_did", "value"].iloc[0]
        if (effects["effect"] == "interaction_did").any()
        else np.nan
    )

    disagg_gain = np.nan
    if aggregated_varset and disaggregated_varset:
        if aggregated_varset in pivot.columns and disaggregated_varset in pivot.columns:
            model_gains = [
                float(pivot.loc[m, aggregated_varset] - pivot.loc[m, disaggregated_varset])
                for m in pivot.index
                if pd.notna(pivot.loc[m, aggregated_varset]) and pd.notna(pivot.loc[m, disaggregated_varset])
            ]
            if model_gains:
                disagg_gain = float(np.mean(model_gains))

    metrics = {
        "primary_model": primary_model,
        "challenger_model": challenger_model,
        "architecture_gain_primary_vs_challenger": architecture_gain,
        "interaction_did": interaction,
        "disaggregation_gain_vs_aggregation": disagg_gain,
    }

    detail_rows = [
        {"source": "disentangling", "metric": "architecture_gain_primary_vs_challenger", "value": architecture_gain},
        {"source": "disentangling", "metric": "interaction_did", "value": interaction},
        {"source": "disentangling", "metric": "disaggregation_gain_vs_aggregation", "value": disagg_gain},
    ]
    return metrics, pd.DataFrame(detail_rows)


def load_regime_metrics(regime_dir: Path | str) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Load regime-persistence and certainty metrics."""
    regime_dir = _as_path(regime_dir)
    durations = pd.read_csv(regime_dir / "regime_durations.csv")
    certainty = pd.read_csv(regime_dir / "regime_classification_certainty.csv")

    duration_spread = float(durations["expected_duration_months"].max() - durations["expected_duration_months"].min())
    mean_duration = float(durations["expected_duration_months"].mean())
    mean_max_prob = _get_metric(certainty, "mean_max_probability")
    high_cert_share = _get_metric(certainty, "share_max_prob_ge_0_8")

    metrics = {
        "regime_duration_spread_months": duration_spread,
        "regime_mean_duration_months": mean_duration,
        "regime_mean_max_probability": mean_max_prob,
        "regime_share_max_prob_ge_0_8": high_cert_share,
    }

    rows = [{"source": "regime", "metric": k, "value": v} for k, v in metrics.items()]
    return metrics, pd.DataFrame(rows)


def load_irf_metrics(irf_dir: Path | str) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Load IRF asymmetry metrics."""
    irf_dir = _as_path(irf_dir)
    comp = pd.read_csv(irf_dir / "msvar_irf_regime_comparison.csv")

    diag_path = irf_dir / "msvar_irf_diagnostics.json"
    target_col = None
    if diag_path.exists():
        import json

        with open(diag_path, "r") as f:
            diag = json.load(f)
        target_col = diag.get("target_col")

    mean_abs_peak_delta = float(np.mean(np.abs(comp["peak_abs_delta_regime1_minus_regime0"])))
    mean_abs_half_life_delta = float(np.mean(np.abs(comp["half_life_delta_regime1_minus_regime0"])))

    target_mean_abs_peak_delta = np.nan
    target_mean_abs_half_life_delta = np.nan
    if target_col and "response_variable" in comp.columns:
        tcomp = comp[comp["response_variable"] == target_col]
        if not tcomp.empty:
            target_mean_abs_peak_delta = float(np.mean(np.abs(tcomp["peak_abs_delta_regime1_minus_regime0"])))
            target_mean_abs_half_life_delta = float(np.mean(np.abs(tcomp["half_life_delta_regime1_minus_regime0"])))

    metrics = {
        "irf_mean_abs_peak_delta": mean_abs_peak_delta,
        "irf_mean_abs_half_life_delta": mean_abs_half_life_delta,
        "irf_target_mean_abs_peak_delta": target_mean_abs_peak_delta,
        "irf_target_mean_abs_half_life_delta": target_mean_abs_half_life_delta,
    }
    rows = [{"source": "irf", "metric": k, "value": v} for k, v in metrics.items()]
    return metrics, pd.DataFrame(rows)


def load_information_loss_metrics(info_loss_dir: Path | str) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Load information-loss metrics and varset mapping."""
    info_loss_dir = _as_path(info_loss_dir)
    seg_summary = pd.read_csv(info_loss_dir / "information_loss_segment_summary.csv")
    pd.read_csv(info_loss_dir / "information_loss_model_segment_tests.csv")

    diag_path = info_loss_dir / "information_loss_diagnostics.json"
    if diag_path.exists():
        import json

        with open(diag_path, "r") as f:
            diag = json.load(f)
    else:
        diag = {}

    def _seg_val(df: pd.DataFrame, segment: str, col: str) -> float:
        if df.empty or segment not in df["segment"].values or col not in df.columns:
            return np.nan
        return float(df[df["segment"] == segment][col].iloc[0])

    crisis_rmse_gain = _seg_val(seg_summary, "crisis", "mean_rmse_improvement_pct_disagg_vs_agg")
    crisis_info_potential = _seg_val(seg_summary, "crisis", "mean_info_loss_potential")
    crisis_sig_share = _seg_val(seg_summary, "crisis", "share_models_significant_info_loss_10pct")
    all_rmse_gain = _seg_val(seg_summary, "all", "mean_rmse_improvement_pct_disagg_vs_agg")

    metrics: Dict[str, Any] = {
        "aggregated_varset": diag.get("aggregated_varset"),
        "disaggregated_varset": diag.get("disaggregated_varset"),
        "info_loss_crisis_rmse_gain_pct": crisis_rmse_gain,
        "info_loss_crisis_mean_info_loss_potential": crisis_info_potential,
        "info_loss_crisis_share_significant_10pct": crisis_sig_share,
        "info_loss_all_rmse_gain_pct": all_rmse_gain,
    }

    rows = [
        {"source": "information_loss", "metric": "info_loss_crisis_rmse_gain_pct", "value": crisis_rmse_gain},
        {"source": "information_loss", "metric": "info_loss_crisis_mean_info_loss_potential", "value": crisis_info_potential},
        {"source": "information_loss", "metric": "info_loss_crisis_share_significant_10pct", "value": crisis_sig_share},
        {"source": "information_loss", "metric": "info_loss_all_rmse_gain_pct", "value": all_rmse_gain},
    ]
    return metrics, pd.DataFrame(rows)


def build_mechanism_synthesis_table(
    disentangling_metrics: Dict[str, Any],
    regime_metrics: Dict[str, Any],
    irf_metrics: Dict[str, Any],
    information_loss_metrics: Dict[str, Any],
) -> pd.DataFrame:
    """Map mechanism evidence to forecast-gain channels."""
    architecture_gain = float(disentangling_metrics.get("architecture_gain_primary_vs_challenger", np.nan))
    disagg_gain = float(disentangling_metrics.get("disaggregation_gain_vs_aggregation", np.nan))
    interaction = float(disentangling_metrics.get("interaction_did", np.nan))

    rows = [
        {
            "mechanism_id": "regime_duration_separation",
            "channel": "architecture",
            "mechanism_metric": "regime_duration_spread_months",
            "mechanism_value": float(regime_metrics.get("regime_duration_spread_months", np.nan)),
            "linked_forecast_gain_metric": "architecture_gain_primary_vs_challenger",
            "linked_forecast_gain_value": architecture_gain,
            "expected_relation": "positive",
        },
        {
            "mechanism_id": "regime_classification_certainty",
            "channel": "architecture",
            "mechanism_metric": "regime_mean_max_probability",
            "mechanism_value": float(regime_metrics.get("regime_mean_max_probability", np.nan)),
            "linked_forecast_gain_metric": "architecture_gain_primary_vs_challenger",
            "linked_forecast_gain_value": architecture_gain,
            "expected_relation": "positive",
        },
        {
            "mechanism_id": "regime_irf_asymmetry",
            "channel": "interaction",
            "mechanism_metric": "irf_target_mean_abs_peak_delta",
            "mechanism_value": float(irf_metrics.get("irf_target_mean_abs_peak_delta", np.nan)),
            "linked_forecast_gain_metric": "interaction_did_abs",
            "linked_forecast_gain_value": float(abs(interaction)) if np.isfinite(interaction) else np.nan,
            "expected_relation": "positive",
        },
        {
            "mechanism_id": "cancellation_information_loss",
            "channel": "information",
            "mechanism_metric": "info_loss_crisis_mean_info_loss_potential",
            "mechanism_value": float(information_loss_metrics.get("info_loss_crisis_mean_info_loss_potential", np.nan)),
            "linked_forecast_gain_metric": "disaggregation_gain_vs_aggregation",
            "linked_forecast_gain_value": disagg_gain,
            "expected_relation": "positive",
        },
        {
            "mechanism_id": "significant_loss_share",
            "channel": "information",
            "mechanism_metric": "info_loss_crisis_share_significant_10pct",
            "mechanism_value": float(information_loss_metrics.get("info_loss_crisis_share_significant_10pct", np.nan)),
            "linked_forecast_gain_metric": "disaggregation_gain_vs_aggregation",
            "linked_forecast_gain_value": disagg_gain,
            "expected_relation": "positive",
        },
    ]

    out = pd.DataFrame(rows)
    out["evidence_available"] = out["mechanism_value"].notna() & out["linked_forecast_gain_value"].notna()
    out["supports_direction"] = np.where(
        out["evidence_available"],
        (out["mechanism_value"] > 0) & (out["linked_forecast_gain_value"] > 0),
        np.nan,
    )
    return out


def build_mechanism_detail_table(*details: pd.DataFrame) -> pd.DataFrame:
    """Combine detailed metrics from all sources."""
    frames = [df for df in details if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame(columns=["source", "metric", "value"])
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["source", "metric"]).reset_index(drop=True)
