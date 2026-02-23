"""Forecast comparison dashboard."""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..config import DATA_DIR


def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def render(selected_categories):
    st.title("Forecast Comparison")
    st.markdown("*Rolling backtests and baseline forecast comparisons*")

    results_dir = DATA_DIR / "forecast_results"
    latest = _load_json(DATA_DIR / "outputs" / "latest.json")
    if latest and latest.get("output_root"):
        candidate = Path(latest["output_root"]) / "forecast_results"
        if candidate.exists():
            results_dir = candidate
    summary_json = _load_json(results_dir / "forecast_model_summary.json")
    rolling_summary_path = results_dir / "rolling_backtest_summary.csv"
    rolling_path = results_dir / "rolling_backtests.csv"

    if summary_json is None or not rolling_summary_path.exists() or not rolling_path.exists():
        st.warning("Run `reserves-forecast-baselines` and `reserves-rolling-backtests` (optionally with --run-id).")
        return

    # Baseline summary table
    st.subheader("Baseline Forecast Metrics")
    rows = []
    for model, info in summary_json.items():
        if model in {"timestamp", "varset", "missing_strategy"}:
            continue
        if not isinstance(info, dict):
            continue
        row = {"model": model}
        for split in ["metrics_validation", "metrics_test"]:
            if split in info:
                row[f"{split}_mae"] = info[split]["mae"]
                row[f"{split}_rmse"] = info[split]["rmse"]
                row[f"{split}_mape"] = info[split]["mape"]
                row[f"{split}_smape"] = info[split].get("smape")
                row[f"{split}_mase"] = info[split].get("mase")
        rows.append(row)
    baseline_df = pd.DataFrame(rows)
    st.dataframe(baseline_df, hide_index=True, use_container_width=True)

    st.subheader("Rolling Backtest Metrics")
    rolling_summary = pd.read_csv(rolling_summary_path)
    st.dataframe(rolling_summary, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("Forecast vs Actual")

    rolling_df = pd.read_csv(rolling_path, parse_dates=["date"])
    model = st.selectbox("Select model", sorted(rolling_df["model"].unique()))
    subset = rolling_df[rolling_df["model"] == model].sort_values("date")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=subset["date"], y=subset["actual"], mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(x=subset["date"], y=subset["forecast"], mode="lines", name="Forecast"))
    fig.update_layout(height=450, xaxis_title="Date", yaxis_title="Gross Reserves (USD M)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Rolling Absolute Error")
    subset["abs_error"] = (subset["actual"] - subset["forecast"]).abs()
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=subset["date"], y=subset["abs_error"], name="Absolute Error", marker_color="#e67e22"))
    fig2.update_layout(height=300, xaxis_title="Date", yaxis_title="Abs Error")
    st.plotly_chart(fig2, use_container_width=True)
