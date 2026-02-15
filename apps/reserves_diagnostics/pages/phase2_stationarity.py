"""Phase 2: Stationarity page."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.stattools import acf

from ..config import DATA_DIR
from ..data_loader import load_diagnostic_results, load_variable_quality_summary, create_merged_reserves_panel


def render(selected_categories):
    """Render the Phase 2 Stationarity page."""
    st.title("Phase 2: Stationarity & Integration Order")

    # Load diagnostic results
    diag_path = DATA_DIR / "diagnostics" / "integration_summary.csv"
    if diag_path.exists():
        int_df = pd.read_csv(diag_path)
        quality_df = load_variable_quality_summary()
        diag_json = load_diagnostic_results()

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        stationary = int_df[int_df['integration_order'].str.contains('I\\(0\\)', na=False)]
        non_stationary = int_df[int_df['integration_order'].str.contains('I\\(1\\)', na=False)]

        col1.metric("Stationary I(0)", len(stationary))
        col2.metric("Non-Stationary I(1)", len(non_stationary))
        col3.metric("Target Variable", "I(1)")
        if quality_df is not None and "is_usable" in quality_df.columns:
            col4.metric("Usable Variables", int(quality_df["is_usable"].sum()))
        else:
            col4.metric("Usable Variables", "N/A")

        st.markdown("---")

        # Results table
        st.subheader("Integration Order Summary")
        st.dataframe(int_df, hide_index=True, use_container_width=True)

        if diag_json is not None:
            za_rows = pd.DataFrame(diag_json.get("phase2_stationarity", {}).get("zivot_andrews", []))
            if not za_rows.empty and {"variable", "za_statistic", "break_date"}.issubset(za_rows.columns):
                st.subheader("Zivot-Andrews Break Candidates")
                st.dataframe(
                    za_rows[["variable", "za_statistic", "break_date", "stationary_5pct"]],
                    hide_index=True,
                    use_container_width=True,
                )

        # Interpretation
        st.subheader("Key Findings")
        st.info("""
        **Target Variable (`gross_reserves_usd_m`):**
        - ADF: p=0.549 -> Cannot reject unit root
        - KPSS: p<0.01 -> Reject stationarity
        - **Conclusion:** I(1) - First differencing required

        **Implication:** Use `reserve_change_usd_m` (first difference) or apply differencing before modeling.
        """)

        # ACF visualization for target
        st.subheader("ACF of Reserves (showing persistence)")

        panel = create_merged_reserves_panel()
        if panel is not None and 'gross_reserves_usd_m' in panel.columns:
            series = panel['gross_reserves_usd_m'].dropna()
            acf_vals = acf(series, nlags=24)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(range(len(acf_vals))),
                y=acf_vals,
                name='ACF',
                marker_color='#3498db'
            ))

            conf = 1.96 / np.sqrt(len(series))
            fig.add_hline(y=conf, line_dash="dash", line_color="red")
            fig.add_hline(y=-conf, line_dash="dash", line_color="red")

            fig.update_layout(
                title="ACF of gross_reserves_usd_m (Slow decay = Non-stationary)",
                xaxis_title="Lag",
                yaxis_title="Autocorrelation",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Run `python scripts/run_diagnostics.py` first to generate results.")
