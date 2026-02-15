"""Phase 4: Volatility page."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from ..config import DATA_DIR, CRISIS_START, CRISIS_END, DEFAULT_DATE
from ..data_loader import create_merged_reserves_panel


def render(selected_categories):
    """Render the Phase 4 Volatility page."""
    st.title("Phase 4: Volatility & Heteroskedasticity")
    st.markdown("*ARCH effects and volatility regime analysis*")

    # Load ARCH results
    arch_path = DATA_DIR / "diagnostics" / "arch_summary.csv"
    if arch_path.exists():
        arch_df = pd.read_csv(arch_path)

        st.subheader("ARCH-LM Test Results")
        st.dataframe(arch_df, hide_index=True, use_container_width=True)

        # Visualize ARCH effects
        st.subheader("ARCH Effects by Variable")

        fig = go.Figure()
        colors = ['#e74c3c' if x else '#2ecc71' for x in arch_df['has_arch_effects']]

        fig.add_trace(go.Bar(
            x=arch_df['variable'],
            y=arch_df['arch_lm_stat'],
            marker_color=colors,
            text=arch_df['arch_lm_pvalue'].apply(lambda x: f'p={x:.3f}'),
            textposition='auto'
        ))

        fig.add_hline(y=21.03, line_dash="dash", line_color="orange",
                      annotation_text="5% critical value")

        fig.update_layout(
            title="ARCH-LM Test Statistics (Red = Significant ARCH effects)",
            xaxis_title="Variable",
            yaxis_title="ARCH-LM Statistic",
            height=450,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)

    # Rolling volatility
    st.subheader("Rolling Volatility Analysis")

    panel = create_merged_reserves_panel()
    if panel is not None:
        fig = go.Figure()

        # Calculate rolling volatility for reserves
        for var, color in [('gross_reserves_usd_m', '#3498db'), ('reserve_change_usd_m', '#e74c3c')]:
            if var in panel.columns:
                rolling_vol = panel.set_index('date')[var].rolling(12).std()
                fig.add_trace(go.Scatter(
                    x=rolling_vol.index,
                    y=rolling_vol.values,
                    mode='lines',
                    name=f'{var} (12m rolling std)',
                    line=dict(color=color)
                ))

        fig.add_vrect(x0=CRISIS_START, x1=CRISIS_END,
                      fillcolor="rgba(255,0,0,0.1)", layer="below", line_width=0)
        fig.add_vline(x=DEFAULT_DATE, line_dash="dash", line_color="red",
                      annotation_text="Default")

        fig.update_layout(
            title="Rolling 12-Month Volatility",
            xaxis_title="Date",
            yaxis_title="Standard Deviation",
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)

        # Volatility regime comparison
        st.subheader("Volatility Regime Comparison")

        if 'gross_reserves_usd_m' in panel.columns:
            pre_crisis = panel[(panel['date'] >= '2015-01-01') & (panel['date'] < CRISIS_START)]['gross_reserves_usd_m'].std()
            crisis = panel[(panel['date'] >= CRISIS_START) & (panel['date'] <= '2022-12-31')]['gross_reserves_usd_m'].std()
            post_crisis = panel[panel['date'] >= '2023-01-01']['gross_reserves_usd_m'].std()

            col1, col2, col3 = st.columns(3)
            col1.metric("Pre-Crisis Vol (2015-2019)", f"{pre_crisis:.0f}")
            col2.metric("Crisis Vol (2020-2022)", f"{crisis:.0f}", f"{crisis/pre_crisis:.1f}x")
            col3.metric("Post-Crisis Vol (2023+)", f"{post_crisis:.0f}", f"{post_crisis/pre_crisis:.1f}x")
