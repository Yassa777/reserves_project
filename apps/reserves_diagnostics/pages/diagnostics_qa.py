"""Diagnostics QA page."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from ..data_loader import load_diagnostic_results, load_variable_quality_summary


def render(selected_categories):
    """Render the Diagnostics QA page."""
    st.title("Diagnostics Quality Assurance")
    st.markdown("*Check test validity, coverage, and skipped variables before modeling*")

    diag_json = load_diagnostic_results()
    quality_df = load_variable_quality_summary()

    if diag_json is None or quality_df is None:
        st.warning("Run `python scripts/run_diagnostics.py` first to generate QA artifacts.")
    else:
        meta = diag_json.get("metadata", {})
        skipped = pd.DataFrame(meta.get("variables_skipped", []))
        tested = meta.get("variables_tested", [])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Variables Requested", len(meta.get("variables_requested", [])))
        col2.metric("Variables Tested", len(tested))
        col3.metric("Variables Skipped", len(skipped))
        col4.metric("Run Timestamp", str(meta.get("timestamp", "N/A"))[:19])

        st.markdown("---")
        st.subheader("Variable Coverage & Readiness")
        st.dataframe(quality_df, hide_index=True, use_container_width=True)

        usable = quality_df[quality_df["is_usable"] == True] if "is_usable" in quality_df.columns else pd.DataFrame()
        if not usable.empty:
            st.subheader("Coverage of Usable Variables")
            fig = go.Figure(
                go.Bar(
                    x=usable["variable"],
                    y=usable["coverage_pct"],
                    marker_color="#2ecc71",
                    text=usable["coverage_pct"].astype(str) + "%",
                    textposition="auto",
                )
            )
            fig.update_layout(
                title="Non-null Coverage by Variable",
                xaxis_title="Variable",
                yaxis_title="Coverage %",
                xaxis_tickangle=-45,
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

        if not skipped.empty:
            st.subheader("Skipped Variables")
            st.dataframe(skipped, hide_index=True, use_container_width=True)

        st.info(
            "Granger causality is now run on first-differenced series to reduce spurious inference from non-stationary levels."
        )
