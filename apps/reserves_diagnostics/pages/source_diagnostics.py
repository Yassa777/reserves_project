"""Source Diagnostics page."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from ..config import RESERVE_DATA_SOURCES, CRISIS_START, CRISIS_END, DEFAULT_DATE
from ..data_loader import load_source_data
from ..utils import get_coverage_info


def render(selected_categories):
    """Render the Source Diagnostics page."""
    st.title("Individual Source Diagnostics")

    # Source selector
    available_sources = [name for name, info in RESERVE_DATA_SOURCES.items()
                         if info["category"] in selected_categories]

    selected_source = st.selectbox("Select Data Source:", available_sources)

    if selected_source:
        info = RESERVE_DATA_SOURCES[selected_source]
        df = load_source_data(selected_source)

        if df is not None:
            st.markdown(f"**Description:** {info['description']}")
            st.markdown(f"**Relevance:** {info['relevance']}")
            st.markdown(f"**Units:** {info['units']}")

            st.markdown("---")

            # Quick stats
            col1, col2, col3, col4 = st.columns(4)
            coverage = get_coverage_info(df, info["date_col"])

            col1.metric("Records", f"{coverage['records']:,}")
            col2.metric("Start Date", coverage["start"].strftime("%Y-%m-%d"))
            col3.metric("End Date", coverage["end"].strftime("%Y-%m-%d"))
            col4.metric("Frequency", info["frequency"])

            st.markdown("---")

            # Data preview
            st.subheader("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)

            # Column info
            st.subheader("Column Information")
            col_info = pd.DataFrame({
                "Column": df.columns,
                "Type": df.dtypes.astype(str),
                "Non-Null": df.notna().sum(),
                "Null": df.isna().sum(),
                "Unique": df.nunique()
            })
            st.dataframe(col_info, hide_index=True, use_container_width=True)

            # Time series plot
            if info["value_cols"]:
                st.subheader("Time Series Visualization")

                plot_cols = [col for col in info["value_cols"] if col in df.columns]
                if plot_cols and info["date_col"] in df.columns:
                    selected_col = st.selectbox("Select column to plot:", plot_cols)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df[info["date_col"]],
                        y=df[selected_col],
                        mode='lines',
                        name=selected_col,
                        line=dict(color='#3498db')
                    ))

                    # Add crisis period shading
                    fig.add_vrect(x0=CRISIS_START, x1=CRISIS_END,
                                  fillcolor="rgba(255,0,0,0.1)", layer="below", line_width=0)

                    # Add default marker
                    fig.add_vline(x=DEFAULT_DATE, line_dash="dash", line_color="red",
                                  annotation_text="Default")

                    fig.update_layout(
                        title=f"{selected_col} Over Time",
                        xaxis_title="Date",
                        yaxis_title=info["units"],
                        height=450
                    )

                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Could not load data for {selected_source}")
