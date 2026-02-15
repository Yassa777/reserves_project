"""Phase 1: Data Quality page."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ..config import RESERVE_DATA_SOURCES, CRISIS_START, CRISIS_END
from ..data_loader import load_source_data
from ..utils import compute_missing_analysis


def compute_coverage_analysis(df, date_col, frequency="Monthly"):
    """Compute date coverage and identify missing months/rows."""
    if df is None or date_col not in df.columns:
        return None

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    start_date = df[date_col].min()
    end_date = df[date_col].max()

    # Generate expected date range based on frequency
    if frequency == "Monthly":
        expected_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
        # Normalize actual dates to month start for comparison
        actual_dates = df[date_col].dt.to_period('M').dt.to_timestamp()
    elif frequency == "Quarterly":
        expected_dates = pd.date_range(start=start_date, end=end_date, freq='QS')
        actual_dates = df[date_col].dt.to_period('Q').dt.to_timestamp()
    elif frequency == "Daily":
        # For daily, just check business days roughly
        expected_dates = pd.date_range(start=start_date, end=end_date, freq='B')
        actual_dates = df[date_col]
    else:
        expected_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
        actual_dates = df[date_col].dt.to_period('M').dt.to_timestamp()

    # Find missing dates
    actual_set = set(actual_dates.unique())
    expected_set = set(expected_dates)
    missing_dates = sorted(expected_set - actual_set)

    # Check crisis period coverage
    crisis_expected = [d for d in expected_dates if CRISIS_START <= d <= CRISIS_END]
    crisis_actual = [d for d in actual_set if CRISIS_START <= d <= CRISIS_END]
    crisis_missing = [d for d in missing_dates if CRISIS_START <= d <= CRISIS_END]

    return {
        "start_date": start_date,
        "end_date": end_date,
        "expected_count": len(expected_dates),
        "actual_count": len(df),
        "missing_count": len(missing_dates),
        "missing_dates": missing_dates,
        "coverage_pct": (len(actual_set) / len(expected_set)) * 100 if expected_set else 100,
        "crisis_expected": len(crisis_expected),
        "crisis_actual": len(crisis_actual),
        "crisis_missing": crisis_missing,
        "crisis_coverage_pct": (len(crisis_actual) / len(crisis_expected)) * 100 if crisis_expected else 100,
    }


def render(selected_categories):
    """Render the Phase 1 Data Quality page."""
    st.title("Phase 1: Data Quality Analysis")

    if not selected_categories:
        st.error("No categories selected! Please select at least one category from the sidebar.")
        return

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Null Values", "Date Coverage", "Source Details"])

    # =========================================================================
    # TAB 1: NULL VALUES IN COLUMNS
    # =========================================================================
    with tab1:
        st.subheader("Missing Values (Nulls in Columns)")
        st.caption("Shows columns where data exists but values are null/empty")

        all_missing = []

        for name, info in RESERVE_DATA_SOURCES.items():
            if info["category"] not in selected_categories:
                continue

            df = load_source_data(name)
            if df is not None and info["value_cols"]:
                missing_df = compute_missing_analysis(df, info["date_col"], info["value_cols"])
                if missing_df is not None:
                    missing_df["Source"] = name
                    missing_df["Category"] = info["category"]
                    all_missing.append(missing_df)

        if all_missing:
            combined = pd.concat(all_missing, ignore_index=True)

            # Summary metrics
            col1, col2, col3 = st.columns(3)
            cols_with_missing = len(combined[combined["Missing"] > 0])
            crisis_missing = len(combined[combined["Crisis Period Missing"] > 0])

            col1.metric("Columns Analyzed", len(combined))
            col2.metric("With Null Values", cols_with_missing)
            col3.metric("Crisis Period Gaps", crisis_missing)

            # Show only columns with missing values
            missing_only = combined[combined["Missing"] > 0].sort_values("Missing %", ascending=False)
            if len(missing_only) > 0:
                st.dataframe(missing_only, hide_index=True, use_container_width=True)

                # Bar chart
                fig = px.bar(
                    missing_only,
                    x="Column",
                    y="Missing %",
                    color="Category",
                    hover_data=["Source", "Total Rows", "Missing"],
                    title="Null Value Percentage by Column"
                )
                fig.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No null values found in any columns!")

    # =========================================================================
    # TAB 2: DATE COVERAGE (MISSING ROWS/MONTHS)
    # =========================================================================
    with tab2:
        st.subheader("Date Coverage (Missing Months/Rows)")
        st.caption("Shows gaps in the time series - entire months that are missing")

        coverage_data = []

        for name, info in RESERVE_DATA_SOURCES.items():
            if info["category"] not in selected_categories:
                continue

            df = load_source_data(name)
            if df is not None:
                coverage = compute_coverage_analysis(df, info["date_col"], info["frequency"])
                if coverage:
                    coverage_data.append({
                        "Source": name,
                        "Category": info["category"],
                        "Frequency": info["frequency"],
                        "Start": coverage["start_date"].strftime("%Y-%m"),
                        "End": coverage["end_date"].strftime("%Y-%m"),
                        "Expected": coverage["expected_count"],
                        "Actual": coverage["actual_count"],
                        "Missing Rows": coverage["missing_count"],
                        "Coverage %": round(coverage["coverage_pct"], 1),
                        "Crisis Missing": len(coverage["crisis_missing"]),
                        "Crisis Coverage %": round(coverage["crisis_coverage_pct"], 1),
                        "_missing_dates": coverage["missing_dates"],
                        "_crisis_missing": coverage["crisis_missing"],
                    })

        if coverage_data:
            coverage_df = pd.DataFrame(coverage_data)

            # Summary metrics
            col1, col2, col3 = st.columns(3)
            sources_with_gaps = len(coverage_df[coverage_df["Missing Rows"] > 0])
            crisis_gaps = len(coverage_df[coverage_df["Crisis Missing"] > 0])

            col1.metric("Sources Analyzed", len(coverage_df))
            col2.metric("With Missing Months", sources_with_gaps)
            col3.metric("Crisis Period Gaps", crisis_gaps)

            # Display table (without internal columns)
            display_cols = ["Source", "Category", "Frequency", "Start", "End",
                          "Expected", "Actual", "Missing Rows", "Coverage %",
                          "Crisis Missing", "Crisis Coverage %"]
            st.dataframe(coverage_df[display_cols], hide_index=True, use_container_width=True)

            # Show sources with gaps
            gaps_df = coverage_df[coverage_df["Missing Rows"] > 0]
            if len(gaps_df) > 0:
                st.markdown("---")
                st.subheader("Sources with Missing Months")

                for _, row in gaps_df.iterrows():
                    with st.expander(f"**{row['Source']}** - {row['Missing Rows']} missing"):
                        missing_dates = row["_missing_dates"]
                        if missing_dates:
                            # Format nicely
                            date_strs = [d.strftime("%Y-%m") for d in missing_dates[:20]]
                            st.write("Missing dates:", ", ".join(date_strs))
                            if len(missing_dates) > 20:
                                st.write(f"... and {len(missing_dates) - 20} more")

                        crisis_missing = row["_crisis_missing"]
                        if crisis_missing:
                            st.warning(f"**Crisis period missing:** {[d.strftime('%Y-%m') for d in crisis_missing]}")

            # Timeline visualization
            st.markdown("---")
            st.subheader("Coverage Timeline")

            fig = go.Figure()

            for i, row in coverage_df.iterrows():
                start = pd.to_datetime(row["Start"])
                end = pd.to_datetime(row["End"])

                color = "#2ecc71" if row["Missing Rows"] == 0 else "#e74c3c" if row["Crisis Missing"] > 0 else "#f39c12"

                fig.add_trace(go.Scatter(
                    x=[start, end],
                    y=[row["Source"], row["Source"]],
                    mode='lines',
                    line=dict(color=color, width=12),
                    name=row["Source"],
                    showlegend=False,
                    hovertemplate=f"<b>{row['Source']}</b><br>Coverage: {row['Coverage %']}%<br>{row['Start']} to {row['End']}<extra></extra>"
                ))

            # Add crisis period shading
            fig.add_vrect(x0=CRISIS_START, x1=CRISIS_END,
                         fillcolor="rgba(255,0,0,0.1)", layer="below", line_width=0,
                         annotation_text="Crisis Period", annotation_position="top left")

            fig.update_layout(
                title="Data Coverage by Source (Green=Complete, Orange=Gaps, Red=Crisis Gaps)",
                xaxis_title="Date",
                yaxis_title="",
                height=max(400, len(coverage_df) * 30),
                xaxis=dict(range=['2005-01-01', '2026-01-01'])
            )
            st.plotly_chart(fig, use_container_width=True)

    # =========================================================================
    # TAB 3: SOURCE DETAILS
    # =========================================================================
    with tab3:
        st.subheader("Detailed Source Analysis")

        source_names = [name for name, info in RESERVE_DATA_SOURCES.items()
                       if info["category"] in selected_categories]

        selected_source = st.selectbox("Select a data source:", source_names)

        if selected_source:
            info = RESERVE_DATA_SOURCES[selected_source]
            df = load_source_data(selected_source)

            if df is not None:
                st.markdown(f"**File:** `{info['file']}`")
                st.markdown(f"**Frequency:** {info['frequency']}")
                st.markdown(f"**Description:** {info['description']}")

                # Coverage info
                coverage = compute_coverage_analysis(df, info["date_col"], info["frequency"])
                if coverage:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Date Range", f"{coverage['start_date'].strftime('%Y-%m')} to {coverage['end_date'].strftime('%Y-%m')}")
                    col2.metric("Total Rows", coverage["actual_count"])
                    col3.metric("Missing Months", coverage["missing_count"])
                    col4.metric("Crisis Coverage", f"{coverage['crisis_coverage_pct']:.0f}%")

                st.markdown("---")

                # Column-level null analysis
                if info["value_cols"]:
                    st.markdown("**Column-level Analysis:**")
                    col_data = []
                    for col in info["value_cols"]:
                        if col in df.columns:
                            nulls = df[col].isna().sum()
                            col_data.append({
                                "Column": col,
                                "Non-null": len(df) - nulls,
                                "Null": nulls,
                                "Null %": round((nulls / len(df)) * 100, 1)
                            })
                    if col_data:
                        st.dataframe(pd.DataFrame(col_data), hide_index=True, use_container_width=True)

                # Show missing dates if any
                if coverage and coverage["missing_dates"]:
                    st.markdown("---")
                    st.warning(f"**Missing dates ({len(coverage['missing_dates'])}):**")
                    date_strs = [d.strftime("%Y-%m") for d in coverage["missing_dates"]]
                    st.code(", ".join(date_strs))
