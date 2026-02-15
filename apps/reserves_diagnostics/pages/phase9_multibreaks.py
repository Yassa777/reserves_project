"""Phase 9: Bai-Perron multiple structural breaks page."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from ..config import DATA_DIR
from ..data_loader import create_merged_reserves_panel


def _read_csv(path):
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def render(selected_categories):
    """Render Phase 9 page."""
    st.title("Phase 9: Bai-Perron Multiple Breaks")
    st.markdown("*BIC-selected multiple breakpoints using piecewise trend regressions*")

    bp_df = _read_csv(DATA_DIR / "diagnostics" / "bai_perron_summary.csv")
    if bp_df is None:
        st.warning("Run `python scripts/run_diagnostics.py` first to generate phase 9 results.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Variables Tested", len(bp_df))
    if "optimal_break_count" in bp_df.columns:
        col2.metric("Mean Break Count", f"{bp_df['optimal_break_count'].mean():.2f}")
    else:
        col2.metric("Mean Break Count", "N/A")

    if "multiple_breaks_detected" in bp_df.columns:
        col3.metric("Multi-Break Variables", int(bp_df["multiple_breaks_detected"].sum()))
    else:
        col3.metric("Multi-Break Variables", "N/A")

    st.markdown("---")
    st.subheader("Bai-Perron Summary")
    st.dataframe(bp_df, hide_index=True, use_container_width=True)

    panel = create_merged_reserves_panel()
    if panel is None:
        return

    st.subheader("Visualize Break Dates")
    variables = [c for c in panel.columns if c != "date" and pd.api.types.is_numeric_dtype(panel[c])]
    selected_var = st.selectbox("Select variable", variables, index=variables.index("gross_reserves_usd_m") if "gross_reserves_usd_m" in variables else 0)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=panel["date"],
            y=panel[selected_var],
            mode="lines",
            name=selected_var,
            line=dict(color="#2c7fb8", width=2),
        )
    )

    row = bp_df[bp_df["variable"] == selected_var]
    if not row.empty and "break_dates" in row.columns:
        raw = str(row["break_dates"].iloc[0])
        if raw and raw.lower() != "nan":
            for date_str in [d.strip() for d in raw.split(",") if d.strip()]:
                try:
                    dt = pd.to_datetime(date_str)
                    fig.add_vline(x=dt, line_dash="dash", line_color="red")
                except Exception:
                    continue

    fig.update_layout(
        title=f"{selected_var} with Bai-Perron Break Candidates",
        xaxis_title="Date",
        yaxis_title=selected_var,
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)
