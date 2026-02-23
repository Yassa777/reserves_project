"""Overview page."""

import streamlit as st
import pandas as pd

from ..config import RESERVE_DATA_SOURCES, CRISIS_START, CRISIS_END
from ..data_loader import load_source_data
from ..utils import get_coverage_info


def render(selected_categories):
    """Render the Overview page."""
    st.title("Reserves Data Dictionary & Diagnostics")

    st.markdown("---")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    sources_loaded = 0
    total_records = 0
    crisis_coverage = 0

    for name, info in RESERVE_DATA_SOURCES.items():
        df = load_source_data(name)
        if df is not None:
            sources_loaded += 1
            total_records += len(df)
            if info["date_col"] in df.columns:
                df[info["date_col"]] = pd.to_datetime(df[info["date_col"]])
                crisis_df = df[(df[info["date_col"]] >= CRISIS_START) & (df[info["date_col"]] <= CRISIS_END)]
                if len(crisis_df) > 0:
                    crisis_coverage += 1

    col1.metric("Sources Available", f"{sources_loaded}/{len(RESERVE_DATA_SOURCES)}")
    col2.metric("Total Records", f"{total_records:,}")
    col3.metric("Crisis Coverage", f"{crisis_coverage} sources")
    col4.metric("Target Variable", "gross_reserves_usd_m")

    st.markdown("---")

    phases = pd.DataFrame([
        {"Phase": "1. Data Quality", "Tests": "Missing values, outliers, cross-source validation",
         "When": "BEFORE merge", "Status": "Available"},
        {"Phase": "2. Distributions", "Tests": "Normality, skewness, kurtosis",
         "When": "BEFORE merge", "Status": "Available"},
        {"Phase": "3. Coverage", "Tests": "Date ranges, gaps, frequency alignment",
         "When": "BEFORE merge", "Status": "Available"},
        {"Phase": "4. Stationarity", "Tests": "ADF, KPSS, Phillips-Perron",
         "When": "AFTER merge", "Status": "Next step"},
        {"Phase": "5. Relationships", "Tests": "Cross-correlation, Granger causality",
         "When": "AFTER merge", "Status": "Next step"},
    ])

    st.dataframe(phases, hide_index=True, use_container_width=True)

    st.markdown("---")

    # Data source summary table
    st.subheader("Reserve-Related Data Sources")

    summary_rows = []
    for name, info in RESERVE_DATA_SOURCES.items():
        if info["category"] not in selected_categories:
            continue

        df = load_source_data(name)
        coverage = get_coverage_info(df, info["date_col"]) if df is not None else None

        summary_rows.append({
            "Source": name,
            "Category": info["category"],
            "Frequency": info["frequency"],
            "File": info["file"],
            "Records": coverage["records"] if coverage else 0,
            "Start": coverage["start"].strftime("%Y-%m") if coverage else "N/A",
            "End": coverage["end"].strftime("%Y-%m") if coverage else "N/A",
            "Status": "OK" if df is not None else "Missing"
        })

    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)
