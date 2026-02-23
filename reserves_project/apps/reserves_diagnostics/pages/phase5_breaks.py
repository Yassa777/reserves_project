"""Phase 5: Structural Breaks page."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from ..config import DATA_DIR, DEFAULT_DATE, PBOC_SWAP_DATE
from ..data_loader import create_merged_reserves_panel


def render(selected_categories):
    """Render the Phase 5 Breaks page."""
    st.title("Phase 5: Structural Break Detection")
    st.markdown("*Chow test and CUSUM analysis for regime changes*")

    # Load Chow test results
    chow_path = DATA_DIR / "diagnostics" / "chow_test_summary.csv"
    if chow_path.exists():
        chow_df = pd.read_csv(chow_path)

        st.subheader("Chow Test Results (Break Date: April 2022)")
        st.dataframe(chow_df, hide_index=True, use_container_width=True)

        # Visualize
        st.subheader("Structural Break Detection")

        fig = go.Figure()
        colors = ['#e74c3c' if x else '#2ecc71' for x in chow_df['break_confirmed']]

        fig.add_trace(go.Bar(
            x=chow_df['variable'],
            y=chow_df['f_statistic'],
            marker_color=colors,
            text=chow_df['p_value'].apply(lambda x: f'p={x:.3f}'),
            textposition='auto'
        ))

        fig.add_hline(y=3.0, line_dash="dash", line_color="orange",
                      annotation_text="Approximate 5% critical value")

        fig.update_layout(
            title="Chow Test F-Statistics (Red = Significant Break)",
            xaxis_title="Variable",
            yaxis_title="F-Statistic",
            height=450,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)

        # Key finding
        st.subheader("Surprising Finding")
        st.warning("""
        **The Chow test does NOT detect a structural break in reserves at April 2022.**

        This suggests:
        1. The crisis affected the **level** but not the **autoregressive dynamics**
        2. The reserve decline was a **gradual process** rather than an abrupt regime shift
        3. The PBOC swap may have masked the break in gross reserves

        However, **exports** and **trade balance** DO show clear structural breaks.
        """)

    # Time series with break
    st.subheader("Reserve Levels with Potential Break Points")

    panel = create_merged_reserves_panel()
    if panel is not None and 'gross_reserves_usd_m' in panel.columns:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=panel['date'],
            y=panel['gross_reserves_usd_m'],
            mode='lines',
            name='Gross Reserves',
            line=dict(color='#3498db', width=2)
        ))

        # Add break lines
        fig.add_vline(x=DEFAULT_DATE, line_dash="dash", line_color="red",
                      annotation_text="Default (Apr 2022)")
        fig.add_vline(x=PBOC_SWAP_DATE, line_dash="dash", line_color="orange",
                      annotation_text="PBOC Swap (Mar 2021)")

        fig.update_layout(
            title="Reserve Levels with Key Dates",
            xaxis_title="Date",
            yaxis_title="USD Millions",
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)
