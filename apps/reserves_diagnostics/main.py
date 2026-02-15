#!/usr/bin/env python3
"""
Reserves Data Dictionary & Diagnostics for Reserve Level Forecasting

Run with: streamlit run main.py
          streamlit run main.py -- 4  (shows phases 1-4)
          streamlit run main.py -- 9  (shows all phases)
"""

import streamlit as st
import sys

from .config import RESERVE_DATA_SOURCES
from .data_loader import load_variable_quality_summary
from .pages import overview
from .pages import diagnostics_qa
from .pages import source_diagnostics
from .pages import phase1_data_quality
from .pages import phase2_stationarity
from .pages import phase3_temporal
from .pages import phase4_volatility
from .pages import phase5_breaks
from .pages import phase6_relationships
from .pages import phase7_cointegration
from .pages import phase8_svar
from .pages import phase9_multibreaks
from .pages import forecast_comparison
from .pages import merged_panel
from .pages import export_data

# Parse command-line argument for number of phases to show
MAX_PHASES = 2  # default: show only phases 1-2
if len(sys.argv) > 1:
    try:
        MAX_PHASES = min(int(sys.argv[1]), 9)
    except ValueError:
        pass

# Page configuration
st.set_page_config(
    page_title="Reserves Data Dictionary & Diagnostics",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .diagnostic-pass { color: #28a745; font-weight: bold; }
    .diagnostic-warn { color: #ffc107; font-weight: bold; }
    .diagnostic-fail { color: #dc3545; font-weight: bold; }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 5px 0;
        border-left: 4px solid #3498db;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.title("Reserves Diagnostics")
st.sidebar.markdown("**Data Dictionary & Quality Checks**")
st.sidebar.markdown("---")

# Build pages list based on MAX_PHASES setting
ALL_PHASE_PAGES = [
    ("Phase 1: Data Quality", phase1_data_quality),
    ("Phase 2: Stationarity", phase2_stationarity),
    ("Phase 3: Temporal", phase3_temporal),
    ("Phase 4: Volatility", phase4_volatility),
    ("Phase 5: Breaks", phase5_breaks),
    ("Phase 6: Relationships", phase6_relationships),
    ("Phase 7: Cointegration", phase7_cointegration),
    ("Phase 8: SVAR", phase8_svar),
    ("Phase 9: Multi-Breaks", phase9_multibreaks),
]
visible_phases = ALL_PHASE_PAGES[:MAX_PHASES]

# Page mapping
PAGE_MAP = {
    "Overview": overview,
    "Diagnostics QA": diagnostics_qa,
    "Source Diagnostics": source_diagnostics,
    "Forecast Comparison": forecast_comparison,
}
for name, module in visible_phases:
    PAGE_MAP[name] = module
PAGE_MAP["Merged Panel"] = merged_panel
PAGE_MAP["Export Data"] = export_data

# Debug: show available pages
st.sidebar.caption(f"Pages loaded: {len(PAGE_MAP)}")

page = st.sidebar.radio("Navigation", list(PAGE_MAP.keys()), label_visibility="collapsed")

st.sidebar.markdown("---")

# Category filter
categories = list(set(info["category"] for info in RESERVE_DATA_SOURCES.values()))
selected_categories = st.sidebar.multiselect(
    "Filter by Category:",
    categories,
    default=categories
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Quick Stats**")
st.sidebar.metric("Data Sources", len(RESERVE_DATA_SOURCES))
quality_df_sidebar = load_variable_quality_summary()
if quality_df_sidebar is not None:
    st.sidebar.metric(
        "Usable Variables",
        int(quality_df_sidebar["is_usable"].sum()) if "is_usable" in quality_df_sidebar.columns else "N/A",
    )

# ============================================================================
# RENDER SELECTED PAGE
# ============================================================================

try:
    PAGE_MAP[page].render(selected_categories)
except Exception as e:
    st.error(f"Error rendering page: {e}")
    import traceback
    st.code(traceback.format_exc())

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "SL-FSI Reserves Diagnostics | Pre-Merge Data Quality for Reserve Forecasting"
    "</div>",
    unsafe_allow_html=True
)
