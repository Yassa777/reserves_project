"""Phase 6: Relationships page."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from ..config import DATA_DIR
from ..data_loader import create_merged_reserves_panel


def render(selected_categories):
    """Render the Phase 6 Relationships page."""
    st.title("Phase 6: Relationship Analysis")
    st.markdown("*Cross-correlation and Granger causality tests*")

    # Load Granger results
    gc_path = DATA_DIR / "diagnostics" / "granger_causality_summary.csv"
    if gc_path.exists():
        gc_df = pd.read_csv(gc_path)

        st.subheader("Granger Causality Test Results")
        st.markdown("*H0: X does not Granger-cause gross_reserves_usd_m*")
        st.dataframe(gc_df, hide_index=True, use_container_width=True)

        # Visualize
        fig = go.Figure()
        colors = ['#e74c3c' if x else '#95a5a6' for x in gc_df['granger_causes']]

        fig.add_trace(go.Bar(
            x=gc_df['test'],
            y=gc_df['best_p_value'],
            marker_color=colors,
            text=gc_df['best_p_value'].apply(lambda x: f'p={x:.3f}'),
            textposition='auto'
        ))

        fig.add_hline(y=0.05, line_dash="dash", line_color="red",
                      annotation_text="5% significance")

        fig.update_layout(
            title="Granger Causality p-values (Lower = More Evidence of Causality)",
            xaxis_title="Test",
            yaxis_title="p-value",
            height=400,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info("""
        **Key Finding:** Granger tests are evaluated on first-differenced series.

        This suggests:
        - Inference is less sensitive to spurious level relationships
        - Contemporaneous dynamics may dominate lagged effects
        - Consider VAR/SVAR designs with simultaneous channels
        """)

    # Cross-correlation visualization
    st.subheader("Cross-Correlation Analysis")

    panel = create_merged_reserves_panel()
    if panel is not None:
        target = 'gross_reserves_usd_m'
        predictors = ['exports_usd_m', 'imports_usd_m', 'remittances_usd_m', 'cse_net_usd_m']
        predictors = [p for p in predictors if p in panel.columns]

        selected_pred = st.selectbox("Select predictor:", predictors)

        if selected_pred and target in panel.columns:
            common = panel[[target, selected_pred]].dropna()

            if len(common) > 30:
                # Compute CCF manually
                max_lag = 12
                ccf_vals = []
                for lag in range(-max_lag, max_lag + 1):
                    if lag < 0:
                        corr = common[target].iloc[-lag:].corr(common[selected_pred].iloc[:lag])
                    elif lag > 0:
                        corr = common[target].iloc[:-lag].corr(common[selected_pred].iloc[lag:])
                    else:
                        corr = common[target].corr(common[selected_pred])
                    ccf_vals.append(corr if not np.isnan(corr) else 0)

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=list(range(-max_lag, max_lag + 1)),
                    y=ccf_vals,
                    marker_color='#3498db'
                ))

                conf = 1.96 / np.sqrt(len(common))
                fig.add_hline(y=conf, line_dash="dash", line_color="red")
                fig.add_hline(y=-conf, line_dash="dash", line_color="red")

                fig.update_layout(
                    title=f"Cross-Correlation: {selected_pred} vs {target}",
                    xaxis_title="Lag (negative = predictor leads)",
                    yaxis_title="Correlation",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

                max_idx = np.argmax(np.abs(ccf_vals))
                max_lag_val = list(range(-max_lag, max_lag + 1))[max_idx]
                st.metric("Max Correlation", f"{ccf_vals[max_idx]:.3f}", f"at lag {max_lag_val}")
