"""Phase 7: Cointegration and ECM/VECM suitability page."""

import streamlit as st
import pandas as pd

from ..config import DATA_DIR


def _read_csv(path):
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def render(selected_categories):
    """Render Phase 7 page."""
    st.title("Phase 7: Cointegration & ECM/VECM")
    st.markdown("*Johansen / Engle-Granger diagnostics plus ECM/VECM suitability checks*")

    eg_df = _read_csv(DATA_DIR / "diagnostics" / "cointegration_engle_granger_summary.csv")
    ecm_df = _read_csv(DATA_DIR / "diagnostics" / "ecm_suitability_summary.csv")
    joh_df = _read_csv(DATA_DIR / "diagnostics" / "johansen_summary.csv")
    vecm_df = _read_csv(DATA_DIR / "diagnostics" / "vecm_suitability_summary.csv")

    if eg_df is None or ecm_df is None:
        st.warning("Run `python scripts/run_diagnostics.py` first to generate phase 7 results.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Engle-Granger Pairs", len(eg_df))
    col2.metric("Cointegrated @5%", int(eg_df["cointegrated_5pct"].sum()) if "cointegrated_5pct" in eg_df.columns else 0)
    col3.metric("ECM Viable Pairs", int(ecm_df["ecm_viable"].sum()) if "ecm_viable" in ecm_df.columns else 0)

    if joh_df is not None and not joh_df.empty and "rank_trace_5pct" in joh_df.columns:
        col4.metric("Johansen Rank (trace)", int(joh_df["rank_trace_5pct"].iloc[0]))
    else:
        col4.metric("Johansen Rank (trace)", "N/A")

    st.markdown("---")

    st.subheader("Engle-Granger Pairwise Results")
    st.dataframe(eg_df, hide_index=True, use_container_width=True)

    st.subheader("ECM Suitability")
    st.dataframe(ecm_df, hide_index=True, use_container_width=True)

    if joh_df is not None and not joh_df.empty:
        st.subheader("Johansen System Test")
        st.dataframe(joh_df, hide_index=True, use_container_width=True)

    if vecm_df is not None and not vecm_df.empty:
        st.subheader("VECM Suitability")
        st.dataframe(vecm_df, hide_index=True, use_container_width=True)

    st.info(
        "Use VECM when Johansen rank >= 1 and target alpha loading is negative/significant; "
        "otherwise prefer differenced VAR/ARIMAX style specifications."
    )
