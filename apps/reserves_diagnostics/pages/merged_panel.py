"""Merged Panel page."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from ..config import CRISIS_START, CRISIS_END, DEFAULT_DATE
from ..data_loader import create_merged_reserves_panel


def render(selected_categories):
    """Render the Merged Panel page."""
    st.title("Merged Reserves Forecasting Panel")
    st.markdown("*Combined monthly panel for reserve level forecasting*")

    with st.spinner("Creating merged panel..."):
        panel = create_merged_reserves_panel()

    if panel is not None:
        st.success(f"Created panel with **{len(panel)} rows** and **{len(panel.columns)} columns**")

        # Panel overview
        st.subheader("Panel Overview")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Date Range", f"{panel['date'].min().strftime('%Y-%m')} to {panel['date'].max().strftime('%Y-%m')}")
        col2.metric("Total Rows", f"{len(panel):,}")
        col3.metric("Total Columns", len(panel.columns))
        col4.metric("Target Variable", "gross_reserves_usd_m")

        st.markdown("---")

        # Column categories
        st.subheader("Panel Variables by Category")

        var_categories = {
            "Target": ["gross_reserves_usd_m", "net_usable_reserves_usd_m", "reserve_change_usd_m", "reserve_change_pct"],
            "Reserve Components": ["fx_reserves_usd_m", "imf_position_usd_m", "sdrs_usd_m", "gold_usd_m"],
            "Trade Flows": ["exports_usd_m", "imports_usd_m", "trade_balance_usd_m"],
            "External Revenue": ["tourism_usd_m", "remittances_usd_m"],
            "Capital Flows": ["cse_net_usd_m", "govt_short_term_usd_m", "total_short_term_usd_m", "portfolio_liabilities_usd_b"],
            "Monetary": ["m0_lkr_m", "m2_lkr_m", "m2_usd_m"],
            "Exchange Rate": ["usd_lkr"],
            "Prices": ["inflation_yoy_pct"],
            "Derived Metrics": ["import_cover_months", "gg_ratio", "ca_proxy_usd_m"]
        }

        for cat, vars in var_categories.items():
            available = [v for v in vars if v in panel.columns]
            if available:
                with st.expander(f"**{cat}** ({len(available)} variables)"):
                    for var in available:
                        non_null = panel[var].notna().sum()
                        pct = (non_null / len(panel)) * 100
                        st.markdown(f"- `{var}`: {non_null:,} values ({pct:.1f}% coverage)")

        st.markdown("---")

        # Data preview
        st.subheader("Data Preview")
        st.dataframe(panel.tail(20), use_container_width=True)

        # Correlation matrix
        st.subheader("Correlation Matrix (Key Variables)")

        key_vars = ['gross_reserves_usd_m', 'exports_usd_m', 'imports_usd_m',
                    'remittances_usd_m', 'tourism_usd_m', 'cse_net_usd_m', 'usd_lkr']
        key_vars = [v for v in key_vars if v in panel.columns]

        if len(key_vars) > 2:
            corr_df = panel[key_vars].corr()

            fig = px.imshow(
                corr_df,
                x=key_vars,
                y=key_vars,
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1,
                title="Correlation Matrix"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

        # Time series plot
        st.subheader("Key Variables Over Time")

        plot_vars = st.multiselect(
            "Select variables to plot:",
            [col for col in panel.columns if col != 'date'],
            default=['gross_reserves_usd_m']
        )

        if plot_vars:
            fig = go.Figure()
            for var in plot_vars:
                fig.add_trace(go.Scatter(
                    x=panel['date'],
                    y=panel[var],
                    mode='lines',
                    name=var
                ))

            fig.add_vrect(x0=CRISIS_START, x1=CRISIS_END,
                          fillcolor="rgba(255,0,0,0.1)", layer="below", line_width=0)
            fig.add_vline(x=DEFAULT_DATE, line_dash="dash", line_color="red")

            fig.update_layout(
                title="Selected Variables Over Time",
                xaxis_title="Date",
                yaxis_title="Value",
                height=500,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Could not create merged panel. Check that core reserve data is available.")
