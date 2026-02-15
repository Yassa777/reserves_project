"""Phase 3: Temporal Dependence page."""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import STL

from ..data_loader import create_merged_reserves_panel


def render(selected_categories):
    """Render the Phase 3 Temporal page."""
    st.title("Phase 3: Temporal Dependence Structure")
    st.markdown("*Autocorrelation, seasonality, and persistence analysis*")

    panel = create_merged_reserves_panel()

    if panel is not None:
        # Variable selector
        num_cols = [c for c in panel.columns if c != 'date' and panel[c].dtype in ['float64', 'int64']]
        selected_var = st.selectbox("Select variable:", num_cols, index=0)

        series = panel[selected_var].dropna()

        if len(series) > 30:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("ACF (Autocorrelation)")
                acf_vals = acf(series, nlags=24)

                fig = go.Figure()
                fig.add_trace(go.Bar(x=list(range(len(acf_vals))), y=acf_vals, marker_color='#3498db'))
                conf = 1.96 / np.sqrt(len(series))
                fig.add_hline(y=conf, line_dash="dash", line_color="red")
                fig.add_hline(y=-conf, line_dash="dash", line_color="red")
                fig.update_layout(title=f"ACF of {selected_var}", xaxis_title="Lag", yaxis_title="ACF", height=350)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("PACF (Partial Autocorrelation)")
                pacf_vals = pacf(series, nlags=24)

                fig = go.Figure()
                fig.add_trace(go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals, marker_color='#e74c3c'))
                fig.add_hline(y=conf, line_dash="dash", line_color="red")
                fig.add_hline(y=-conf, line_dash="dash", line_color="red")
                fig.update_layout(title=f"PACF of {selected_var}", xaxis_title="Lag", yaxis_title="PACF", height=350)
                st.plotly_chart(fig, use_container_width=True)

            # STL decomposition
            st.subheader("Seasonal Decomposition (STL)")

            if len(series) >= 24:
                try:
                    stl = STL(series, period=12, robust=True)
                    result = stl.fit()

                    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                        subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'))

                    fig.add_trace(go.Scatter(y=series.values, mode='lines', name='Original'), row=1, col=1)
                    fig.add_trace(go.Scatter(y=result.trend, mode='lines', name='Trend', line=dict(color='#e74c3c')), row=2, col=1)
                    fig.add_trace(go.Scatter(y=result.seasonal, mode='lines', name='Seasonal', line=dict(color='#2ecc71')), row=3, col=1)
                    fig.add_trace(go.Scatter(y=result.resid, mode='lines', name='Residual', line=dict(color='#9b59b6')), row=4, col=1)

                    fig.update_layout(height=700, showlegend=False, title=f"STL Decomposition of {selected_var}")
                    st.plotly_chart(fig, use_container_width=True)

                    # Strength metrics
                    var_resid = np.var(result.resid)
                    var_deseas = np.var(result.trend + result.resid)
                    var_detrend = np.var(result.seasonal + result.resid)

                    trend_strength = max(0, 1 - var_resid / var_deseas) if var_deseas > 0 else 0
                    seasonal_strength = max(0, 1 - var_resid / var_detrend) if var_detrend > 0 else 0

                    col1, col2 = st.columns(2)
                    col1.metric("Trend Strength", f"{trend_strength:.2%}")
                    col2.metric("Seasonal Strength", f"{seasonal_strength:.2%}")
                except Exception as e:
                    st.warning(f"Could not perform STL decomposition: {e}")
        else:
            st.warning("Insufficient data for temporal analysis")
