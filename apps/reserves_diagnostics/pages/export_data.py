"""Export Data page."""

import streamlit as st
import pandas as pd

from ..config import RESERVE_DATA_SOURCES, MERGED_DIR
from ..data_loader import load_source_data, create_merged_reserves_panel
from ..utils import compute_missing_analysis, compute_distribution_stats


def render(selected_categories):
    """Render the Export Data page."""
    st.title("Export Data")
    st.markdown("*Download merged panel and diagnostic reports*")

    # Merged panel export
    st.subheader("Merged Reserves Forecasting Panel")

    with st.spinner("Creating merged panel..."):
        panel = create_merged_reserves_panel()

    if panel is not None:
        st.success(f"Panel ready: {len(panel)} rows x {len(panel.columns)} columns")

        # Preview
        st.dataframe(panel.head(), use_container_width=True)

        # Download button
        csv = panel.to_csv(index=False)
        st.download_button(
            label="Download Merged Panel (CSV)",
            data=csv,
            file_name="reserves_forecasting_panel.csv",
            mime="text/csv"
        )

        # Save to data/merged
        if st.button("Save to data/merged/reserves_forecasting_panel.csv"):
            output_path = MERGED_DIR / "reserves_forecasting_panel.csv"
            panel.to_csv(output_path, index=False)
            st.success(f"Saved to {output_path}")

    st.markdown("---")

    # Diagnostic reports
    st.subheader("Diagnostic Reports")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Missing Value Report**")

        all_missing = []
        for name, info in RESERVE_DATA_SOURCES.items():
            df = load_source_data(name)
            if df is not None and info["value_cols"]:
                missing_df = compute_missing_analysis(df, info["date_col"], info["value_cols"])
                if missing_df is not None:
                    missing_df["Source"] = name
                    all_missing.append(missing_df)

        if all_missing:
            combined_missing = pd.concat(all_missing, ignore_index=True)
            csv_missing = combined_missing.to_csv(index=False)
            st.download_button(
                label="Download Missing Value Report",
                data=csv_missing,
                file_name="missing_value_report.csv",
                mime="text/csv"
            )

    with col2:
        st.markdown("**Distribution Statistics Report**")

        all_dist = []
        for name, info in RESERVE_DATA_SOURCES.items():
            df = load_source_data(name)
            if df is not None and info["value_cols"]:
                dist_df = compute_distribution_stats(df, info["value_cols"])
                if dist_df is not None:
                    dist_df["Source"] = name
                    all_dist.append(dist_df)

        if all_dist:
            combined_dist = pd.concat(all_dist, ignore_index=True)
            csv_dist = combined_dist.to_csv(index=False)
            st.download_button(
                label="Download Distribution Report",
                data=csv_dist,
                file_name="distribution_report.csv",
                mime="text/csv"
            )
