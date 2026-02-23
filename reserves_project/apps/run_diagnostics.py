#!/usr/bin/env python3
"""
Reserves Data Dictionary & Diagnostics for Reserve Level Forecasting

Run with: streamlit run run_diagnostics.py
          streamlit run run_diagnostics.py -- 4  (shows phases 1-4)
          streamlit run run_diagnostics.py -- 9  (shows all phases)
"""

import streamlit as st
import sys

from reserves_project.apps.reserves_diagnostics.config import RESERVE_DATA_SOURCES
from reserves_project.apps.reserves_diagnostics.data_loader import load_variable_quality_summary
from reserves_project.apps.reserves_diagnostics.pages import overview
from reserves_project.apps.reserves_diagnostics.pages import diagnostics_qa
from reserves_project.apps.reserves_diagnostics.pages import source_diagnostics
from reserves_project.apps.reserves_diagnostics.pages import phase1_data_quality
from reserves_project.apps.reserves_diagnostics.pages import phase2_stationarity
from reserves_project.apps.reserves_diagnostics.pages import phase3_temporal
from reserves_project.apps.reserves_diagnostics.pages import phase4_volatility
from reserves_project.apps.reserves_diagnostics.pages import phase5_breaks
from reserves_project.apps.reserves_diagnostics.pages import phase6_relationships
from reserves_project.apps.reserves_diagnostics.pages import phase7_cointegration
from reserves_project.apps.reserves_diagnostics.pages import phase8_svar
from reserves_project.apps.reserves_diagnostics.pages import phase9_multibreaks
from reserves_project.apps.reserves_diagnostics.pages import forecast_comparison
from reserves_project.apps.reserves_diagnostics.pages import merged_panel
from reserves_project.apps.reserves_diagnostics.pages import export_data

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

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.title("Reserves Diagnostics")
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
st.sidebar.metric("Data Sources", len(RESERVE_DATA_SOURCES))

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
st.caption("SL-FSI Reserves Diagnostics")
