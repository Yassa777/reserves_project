"""Phase 8: Exogeneity and SVAR diagnostics page."""

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
    """Render Phase 8 page."""
    st.title("Phase 8: Exogeneity & SVAR Identification")
    st.markdown("*Block exogeneity, recursive short-run restrictions, and sign-pattern checks*")

    exog_df = _read_csv(DATA_DIR / "diagnostics" / "svar_exogeneity_summary.csv")
    sign_df = _read_csv(DATA_DIR / "diagnostics" / "svar_sign_restriction_summary.csv")
    model_df = _read_csv(DATA_DIR / "diagnostics" / "svar_model_summary.csv")

    if exog_df is None or model_df is None:
        st.warning("Run `python scripts/run_diagnostics.py` first to generate phase 8 results.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Exogeneity Tests", len(exog_df))
    if "weakly_exogenous_5pct" in exog_df.columns:
        col2.metric("Weakly Exogenous @5%", int(exog_df["weakly_exogenous_5pct"].sum()))
    else:
        col2.metric("Weakly Exogenous @5%", "N/A")

    if "converged" in model_df.columns:
        col3.metric("SVAR Converged", str(model_df["converged"].iloc[0]))
    else:
        col3.metric("SVAR Converged", "N/A")

    st.markdown("---")
    st.subheader("Block Exogeneity Results")
    st.dataframe(exog_df, hide_index=True, use_container_width=True)

    st.subheader("SVAR Model Summary")
    st.dataframe(model_df, hide_index=True, use_container_width=True)

    if sign_df is not None and not sign_df.empty:
        st.subheader("Sign-Pattern Checks (Horizon 0-3)")
        st.dataframe(sign_df, hide_index=True, use_container_width=True)

    st.info(
        "These checks validate whether the recursive short-run identification is economically plausible. "
        "For publication-grade structural analysis, augment with externally justified sign/zero restrictions."
    )
