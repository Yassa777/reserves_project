#!/usr/bin/env python3
"""
Interactive Data Dictionary for Reserve Adequacy Analysis

A Streamlit application providing an interactive exploration of all data sources,
column definitions, and derived metrics used in the SL-FSI Reserve Adequacy module.

Run with: streamlit run app_data_dictionary.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="SL-FSI Data Dictionary",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

from reserves_project.config.paths import DATA_DIR

# Paths
EXTERNAL_DIR = DATA_DIR / "external"
PROCESSED_DIR = DATA_DIR / "processed"
MANUAL_DIR = DATA_DIR / "manual_extraction"

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .formula-box {
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 15px;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 10px 15px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #cce5ff;
        border-left: 4px solid #004085;
        padding: 10px 15px;
        margin: 10px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA DEFINITIONS
# ============================================================================

DATA_SOURCES = {
    "Reserve Assets": {
        "file": "reserve_assets_monthly_cbsl.csv",
        "source": "CBSL Historical Data Series",
        "frequency": "Monthly",
        "range": "Nov 2013 - Dec 2025",
        "description": "Official reserve asset components including FX, IMF position, SDRs, gold",
        "url": "https://www.cbsl.gov.lk/en/statistics/statistical-tables/external-sector"
    },
    "Central Govt Debt": {
        "file": "central_govt_debt_quarterly.csv",
        "source": "CBSL SDDS",
        "frequency": "Quarterly",
        "range": "2000 - Q3 2025",
        "description": "Central government debt by type (domestic/foreign, short/long term)",
        "url": "https://www.cbsl.gov.lk/en/statistics/sdds-sri-lanka"
    },
    "External Debt (USD)": {
        "file": "external_debt_usd_quarterly.csv",
        "source": "CBSL Table 2.12",
        "frequency": "Quarterly",
        "range": "Q4 2012 - Q3 2025",
        "description": "External debt in USD by sector and maturity",
        "url": "https://www.cbsl.gov.lk/en/statistics/statistical-tables/external-sector"
    },
    "IIP": {
        "file": "iip_quarterly_2025.csv",
        "source": "CBSL IIP Release",
        "frequency": "Quarterly",
        "range": "Q4 2012 - Q3 2025",
        "description": "International Investment Position - portfolio liabilities",
        "url": "https://www.cbsl.gov.lk/en/statistics/statistical-tables/external-sector"
    },
    "Monetary Aggregates": {
        "file": "monetary_aggregates_monthly.csv",
        "source": "CBSL Table 4.02",
        "frequency": "Monthly",
        "range": "Dec 1995 - Sep 2025",
        "description": "M0 and M2 money supply",
        "url": "https://www.cbsl.gov.lk/en/statistics/statistical-tables/monetary-sector"
    },
    "Monthly Imports": {
        "file": "monthly_imports_usd.csv",
        "source": "CBSL Table 2.04",
        "frequency": "Monthly",
        "range": "Jan 2007 - Nov 2025",
        "description": "Merchandise imports (CIF basis)",
        "url": "https://www.cbsl.gov.lk/en/statistics/statistical-tables/external-sector"
    },
    "Monthly Exports": {
        "file": "monthly_exports_usd.csv",
        "source": "CBSL Table 2.02",
        "frequency": "Monthly",
        "range": "Jan 2007 - Nov 2025",
        "description": "Merchandise exports (FOB basis)",
        "url": "https://www.cbsl.gov.lk/en/statistics/statistical-tables/external-sector"
    },
    "Tourism Earnings": {
        "file": "tourism_earnings_monthly.csv",
        "source": "CBSL Table 2.14.1",
        "frequency": "Monthly",
        "range": "Jan 2009 - Nov 2025",
        "description": "Tourism receipts (travel services credit)",
        "url": "https://www.cbsl.gov.lk/en/statistics/statistical-tables/external-sector"
    },
    "Remittances": {
        "file": "remittances_monthly.csv",
        "source": "CBSL Table 2.14.2",
        "frequency": "Monthly",
        "range": "Jan 2009 - Nov 2025",
        "description": "Workers' remittances from abroad",
        "url": "https://www.cbsl.gov.lk/en/statistics/statistical-tables/external-sector"
    },
    "CSE Flows": {
        "file": "cse_flows_monthly.csv",
        "source": "CBSL Table 2.14.3",
        "frequency": "Monthly",
        "range": "Jan 2012 - Dec 2025",
        "description": "Colombo Stock Exchange foreign portfolio flows",
        "url": "https://www.cbsl.gov.lk/en/statistics/statistical-tables/external-sector"
    },
    "NFA (Money Supply)": {
        "file": "money_supply_monthly_clean.csv",
        "source": "CBSL eResearch",
        "frequency": "Monthly",
        "range": "Jan 2005 - Dec 2017",
        "description": "Net Foreign Assets from monetary aggregates",
        "url": "https://www.cbsl.lk/eresearch/"
    },
    "NFA + REER": {
        "file": "D14_reer_nfa.csv",
        "source": "CBSL eResearch",
        "frequency": "Monthly",
        "range": "Jan 2010 - Jul 2025",
        "description": "Net Foreign Assets with Real Effective Exchange Rate",
        "url": "https://www.cbsl.lk/eresearch/"
    }
}

RESERVE_COLUMNS = [
    {"column": "date", "definition": "Month-end date", "unit": "Date", "notes": "Format: YYYY-MM-DD"},
    {"column": "gross_reserves_usd_m", "definition": "Total official reserve assets", "unit": "USD millions", "notes": "Sum of all components"},
    {"column": "fx_reserves_usd_m", "definition": "Foreign currency reserves", "unit": "USD millions", "notes": "Primarily USD, EUR, GBP"},
    {"column": "imf_position_usd_m", "definition": "Reserve position in IMF", "unit": "USD millions", "notes": "Sri Lanka's quota contribution"},
    {"column": "sdrs_usd_m", "definition": "Special Drawing Rights", "unit": "USD millions", "notes": "IMF-allocated SDRs"},
    {"column": "gold_usd_m", "definition": "Monetary gold holdings", "unit": "USD millions", "notes": "Valued at market prices"},
    {"column": "other_reserves_usd_m", "definition": "Other reserve assets", "unit": "USD millions", "notes": "Residual category"},
]

NFA_COLUMNS = [
    {"column": "nfa_monetary_auth_lkr_b", "definition": "NFA of Monetary Authorities", "unit": "LKR billion", "notes": "CBSL's net foreign position"},
    {"column": "nfa_comm_banks_total_lkr_b", "definition": "NFA of Commercial Banks (Total)", "unit": "LKR billion", "notes": "All commercial banks"},
    {"column": "nfa_comm_banks_dbu_lkr_b", "definition": "NFA of Commercial Banks (DBUs)", "unit": "LKR billion", "notes": "Domestic Banking Units"},
    {"column": "nfa_comm_banks_obu_lkr_b", "definition": "NFA of Commercial Banks (OBUs)", "unit": "LKR billion", "notes": "Offshore Banking Units"},
]

BENCHMARKS = {
    "Import Cover": {
        "formula": "Gross Reserves (USD) / Monthly Imports (USD)",
        "thresholds": [
            ("Comfortable", "≥ 6 months", "#28a745"),
            ("IMF Minimum", "≥ 3 months", "#ffc107"),
            ("Warning", "< 2 months", "#fd7e14"),
            ("Critical", "< 1 month", "#dc3545"),
        ],
        "interpretation": "Measures how many months of imports can be financed by current reserves."
    },
    "Greenspan-Guidotti": {
        "formula": "Gross Reserves (USD) / Short-term External Debt (USD)",
        "thresholds": [
            ("Adequate", "≥ 1.0", "#28a745"),
            ("Near-breach", "1.0 - 1.5", "#ffc107"),
            ("Breach", "< 1.0", "#dc3545"),
        ],
        "interpretation": "Indicates ability to service all short-term debt without external financing."
    },
    "IMF ARA Metric": {
        "formula": "5% × Exports + 5% × M2 (USD) + 30% × ST Debt + 15% × Portfolio Liabilities",
        "thresholds": [
            ("Comfortable", "≥ 150%", "#28a745"),
            ("Adequate", "100-150%", "#ffc107"),
            ("Breach", "< 100%", "#dc3545"),
        ],
        "interpretation": "IMF's comprehensive reserve adequacy assessment using weighted formula."
    }
}

CRISIS_EVENTS = [
    ("2019-04-21", "Easter Sunday bombings", "Pre-crisis baseline"),
    ("2020-03-01", "COVID-19 pandemic begins", "External shock"),
    ("2021-03-01", "PBOC swap activated", "Net reserves adjustment"),
    ("2021-07-01", "Food emergency declared", "Early warning"),
    ("2021-09-01", "Economic emergency declared", "Early warning"),
    ("2022-04-12", "Sovereign default announced", "Primary event"),
    ("2022-07-05", "Wickremesinghe becomes president", "Political transition"),
    ("2023-03-20", "IMF EFF approved", "Recovery marker"),
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data
def load_data(filename):
    """Load a CSV file from external directory."""
    filepath = EXTERNAL_DIR / filename
    if filepath.exists():
        return pd.read_csv(filepath, parse_dates=['date'] if 'date' in pd.read_csv(filepath, nrows=1).columns else None)

    # Try processed directory
    filepath = PROCESSED_DIR / filename
    if filepath.exists():
        return pd.read_csv(filepath, parse_dates=['date'] if 'date' in pd.read_csv(filepath, nrows=1).columns else None)

    return None


def get_data_coverage():
    """Calculate actual data coverage from files."""
    coverage = []
    for name, info in DATA_SOURCES.items():
        df = load_data(info["file"])
        if df is not None and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            coverage.append({
                "Dataset": name,
                "Start": df['date'].min(),
                "End": df['date'].max(),
                "Records": len(df),
                "Frequency": info["frequency"]
            })
        else:
            coverage.append({
                "Dataset": name,
                "Start": pd.NaT,
                "End": pd.NaT,
                "Records": 0,
                "Frequency": info["frequency"]
            })
    return pd.DataFrame(coverage)


def create_coverage_timeline(coverage_df):
    """Create a Gantt-style timeline of data coverage."""
    fig = go.Figure()

    colors = px.colors.qualitative.Set2

    for i, row in coverage_df.iterrows():
        if pd.notna(row['Start']) and pd.notna(row['End']):
            fig.add_trace(go.Scatter(
                x=[row['Start'], row['End']],
                y=[row['Dataset'], row['Dataset']],
                mode='lines',
                line=dict(color=colors[i % len(colors)], width=20),
                name=row['Dataset'],
                hovertemplate=f"<b>{row['Dataset']}</b><br>" +
                              f"Start: {row['Start'].strftime('%Y-%m')}<br>" +
                              f"End: {row['End'].strftime('%Y-%m')}<br>" +
                              f"Records: {row['Records']:,}<extra></extra>"
            ))

    # Add crisis event markers using shapes (avoids timestamp arithmetic issue)
    for date, event, _ in CRISIS_EVENTS:
        date_str = str(date)  # Convert to string to avoid pandas timestamp issues
        fig.add_shape(
            type="line",
            x0=date_str, x1=date_str,
            y0=0, y1=1,
            yref="paper",
            line=dict(color="red", width=1, dash="dash"),
            opacity=0.5
        )
        # Add annotation separately
        fig.add_annotation(
            x=date_str,
            y=1.02,
            yref="paper",
            text=event[:15] + "..." if len(event) > 15 else event,
            showarrow=False,
            font=dict(size=8, color="red"),
            textangle=-45
        )

    fig.update_layout(
        title="Data Coverage Timeline",
        xaxis_title="Date",
        yaxis_title="Dataset",
        height=500,
        showlegend=False,
        xaxis=dict(
            range=['2005-01-01', '2026-01-01']
        ),
        margin=dict(t=80)  # Extra top margin for annotations
    )

    return fig


# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.title("📚 Data Dictionary")
st.sidebar.markdown("**Reserve Adequacy Analysis**")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigate to:",
    [
        "🏠 Overview",
        "📂 Data Sources",
        "📊 Column Definitions",
        "🎯 Benchmarks & Metrics",
        "📈 NFA Series",
        "⏱️ Data Coverage",
        "🔗 References"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Quick Stats**")
total_sources = len(DATA_SOURCES)
st.sidebar.metric("Data Sources", total_sources)

# Try to get actual record count
try:
    coverage = get_data_coverage()
    total_records = coverage['Records'].sum()
    st.sidebar.metric("Total Records", f"{total_records:,}")
except:
    pass

st.sidebar.markdown("---")
st.sidebar.markdown("*Last Updated: January 2026*")


# ============================================================================
# PAGE: OVERVIEW
# ============================================================================

if page == "🏠 Overview":
    st.title("📚 Data Dictionary: Reserve Adequacy Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Primary Source", "CBSL", help="Central Bank of Sri Lanka")
    with col2:
        st.metric("Data Range", "1995-2025", help="Varying coverage by series")
    with col3:
        st.metric("Key Event", "Apr 12, 2022", help="Sovereign default announced")

    st.markdown("---")

    # Research question
    st.subheader("🔬 Research Question")
    st.info("""
    **At what reserve level did Sri Lanka's 2022 sovereign default crisis become inevitable?**

    This analysis examines multiple reserve adequacy benchmarks including Import Cover,
    Greenspan-Guidotti Ratio, and IMF ARA Metric to identify ex-ante warning thresholds.
    """)

    # Crisis timeline
    st.subheader("📅 Key Event Timeline")

    timeline_df = pd.DataFrame(CRISIS_EVENTS, columns=["Date", "Event", "Category"])
    timeline_df['Date'] = pd.to_datetime(timeline_df['Date'])

    fig = px.scatter(
        timeline_df,
        x="Date",
        y=[1]*len(timeline_df),
        text="Event",
        color="Category",
        size=[20]*len(timeline_df),
        height=200
    )
    fig.update_traces(textposition='top center')
    fig.update_layout(
        yaxis=dict(visible=False),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Quick navigation cards
    st.subheader("🚀 Quick Navigation")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **📂 Data Sources**
        - Reserve Assets
        - External Debt
        - Trade Data
        - Portfolio Flows
        """)

    with col2:
        st.markdown("""
        **🎯 Key Metrics**
        - Import Cover
        - Greenspan-Guidotti
        - IMF ARA Metric
        - Net Usable Reserves
        """)

    with col3:
        st.markdown("""
        **📈 New: NFA Series**
        - Monetary Authority NFA
        - Commercial Bank NFA
        - Crisis Signal Power
        """)


# ============================================================================
# PAGE: DATA SOURCES
# ============================================================================

elif page == "📂 Data Sources":
    st.title("📂 Data Source Files")

    # Filter options
    col1, col2 = st.columns([2, 1])
    with col1:
        search = st.text_input("🔍 Search datasets", placeholder="e.g., reserves, debt, imports...")
    with col2:
        freq_filter = st.multiselect("Filter by frequency", ["Monthly", "Quarterly"], default=["Monthly", "Quarterly"])

    # Display sources
    for name, info in DATA_SOURCES.items():
        if search.lower() not in name.lower() and search.lower() not in info["description"].lower():
            if search:
                continue
        if info["frequency"] not in freq_filter:
            continue

        with st.expander(f"**{name}** — {info['file']}", expanded=False):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"**Description:** {info['description']}")
                st.markdown(f"**Source:** {info['source']}")
                st.markdown(f"**Frequency:** {info['frequency']}")
                st.markdown(f"**Data Range:** {info['range']}")
                st.markdown(f"[🔗 CBSL Source]({info['url']})")

            with col2:
                # Try to load and show preview
                df = load_data(info["file"])
                if df is not None:
                    st.metric("Records", f"{len(df):,}")
                    st.metric("Columns", len(df.columns))

            # Data preview
            if df is not None:
                st.markdown("**Data Preview:**")
                st.dataframe(df.head(5), use_container_width=True)

                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=info["file"],
                    mime="text/csv"
                )
            else:
                st.warning("File not found or unable to load.")


# ============================================================================
# PAGE: COLUMN DEFINITIONS
# ============================================================================

elif page == "📊 Column Definitions":
    st.title("📊 Column Definitions")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Reserve Assets",
        "External Debt",
        "Trade & Flows",
        "Monetary"
    ])

    with tab1:
        st.subheader("Reserve Assets Columns")
        st.markdown(f"**Source:** `reserve_assets_monthly_cbsl.csv`")

        df = pd.DataFrame(RESERVE_COLUMNS)
        st.dataframe(
            df,
            column_config={
                "column": st.column_config.TextColumn("Column Name", width="medium"),
                "definition": st.column_config.TextColumn("Definition", width="large"),
                "unit": st.column_config.TextColumn("Unit", width="small"),
                "notes": st.column_config.TextColumn("Notes", width="medium"),
            },
            hide_index=True,
            use_container_width=True
        )

        st.markdown("**Formula:**")
        st.code("""
gross_reserves_usd_m = fx_reserves_usd_m
                     + imf_position_usd_m
                     + sdrs_usd_m
                     + gold_usd_m
                     + other_reserves_usd_m
        """, language="python")

    with tab2:
        st.subheader("External Debt Columns")

        debt_columns = [
            {"column": "govt_total_usd_m", "definition": "Total government external debt", "unit": "USD millions"},
            {"column": "govt_short_term_usd_m", "definition": "Government short-term external debt", "unit": "USD millions"},
            {"column": "govt_long_term_usd_m", "definition": "Government long-term external debt", "unit": "USD millions"},
            {"column": "central_bank_usd_m", "definition": "Central bank external debt", "unit": "USD millions"},
            {"column": "total_short_term_usd_m", "definition": "Combined short-term external debt", "unit": "USD millions"},
        ]

        st.dataframe(pd.DataFrame(debt_columns), hide_index=True, use_container_width=True)

        st.info("**Key for Greenspan-Guidotti:** `govt_short_term_usd_m` is the critical denominator.")

    with tab3:
        st.subheader("Trade & External Flows")

        trade_columns = [
            {"column": "imports_usd_m", "definition": "Monthly merchandise imports", "unit": "USD millions", "file": "monthly_imports_usd.csv"},
            {"column": "exports_usd_m", "definition": "Monthly merchandise exports", "unit": "USD millions", "file": "monthly_exports_usd.csv"},
            {"column": "tourism_earnings_usd_m", "definition": "Monthly tourism receipts", "unit": "USD millions", "file": "tourism_earnings_monthly.csv"},
            {"column": "remittances_usd_m", "definition": "Monthly remittance inflows", "unit": "USD millions", "file": "remittances_monthly.csv"},
            {"column": "cse_net_usd_m", "definition": "Net CSE portfolio flows", "unit": "USD millions", "file": "cse_flows_monthly.csv"},
        ]

        st.dataframe(pd.DataFrame(trade_columns), hide_index=True, use_container_width=True)

    with tab4:
        st.subheader("Monetary Aggregates")

        monetary_columns = [
            {"column": "reserve_money_m0_lkr_m", "definition": "Reserve money (M0)", "unit": "LKR millions"},
            {"column": "broad_money_m2_lkr_m", "definition": "Broad money (M2)", "unit": "LKR millions"},
            {"column": "money_multiplier_m1", "definition": "M1 money multiplier", "unit": "Ratio"},
        ]

        st.dataframe(pd.DataFrame(monetary_columns), hide_index=True, use_container_width=True)

        st.markdown("**M2 Definition:**")
        st.code("M2 = Currency in circulation + Demand deposits + Time deposits + Savings deposits")


# ============================================================================
# PAGE: BENCHMARKS & METRICS
# ============================================================================

elif page == "🎯 Benchmarks & Metrics":
    st.title("🎯 Reserve Adequacy Benchmarks")

    for name, info in BENCHMARKS.items():
        with st.expander(f"**{name}**", expanded=True):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"*{info['interpretation']}*")
                st.markdown("**Formula:**")
                st.code(info['formula'])

            with col2:
                st.markdown("**Thresholds:**")
                for level, value, color in info['thresholds']:
                    st.markdown(
                        f"<span style='color:{color}; font-weight:bold;'>●</span> {level}: {value}",
                        unsafe_allow_html=True
                    )

    st.markdown("---")

    # Net Usable Reserves
    st.subheader("📉 Net Usable Reserves (PBOC Adjustment)")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Formula:**
        ```
        Net Usable Reserves = Gross Reserves - PBOC Swap Amount
        ```

        **PBOC Swap Details:**
        - Amount: **USD 1,500 million**
        - Start Date: March 1, 2021
        - Status: Encumbered (not freely usable)
        """)

    with col2:
        st.warning("""
        The China-Sri Lanka bilateral currency swap inflates gross reserves
        without providing equivalent liquidity. For true reserve adequacy
        assessment, this amount should be deducted.
        """)

    # IMF ARA breakdown
    st.subheader("📐 IMF ARA Metric Components")

    ara_components = pd.DataFrame([
        {"Component": "Annual Exports", "Weight": "5%", "Source": "monthly_exports_usd.csv", "Transformation": "Sum quarterly × 4"},
        {"Component": "Broad Money M2", "Weight": "5%", "Source": "monetary_aggregates_monthly.csv", "Transformation": "Convert LKR to USD"},
        {"Component": "Short-term Debt", "Weight": "30%", "Source": "external_debt_usd_quarterly.csv", "Transformation": "Direct use"},
        {"Component": "Portfolio Liabilities", "Weight": "15%", "Source": "iip_quarterly_2025.csv", "Transformation": "Sum equity + debt"},
    ])

    st.dataframe(ara_components, hide_index=True, use_container_width=True)

    st.code("""
# ARA Calculation
ara_total = (0.05 * annual_exports) +
            (0.05 * m2_usd) +
            (0.30 * short_term_debt) +
            (0.15 * portfolio_liabilities)

ara_ratio = actual_reserves / ara_total
    """, language="python")


# ============================================================================
# PAGE: NFA SERIES
# ============================================================================

elif page == "📈 NFA Series":
    st.title("📈 Net Foreign Assets (NFA) Series")

    st.info("""
    While Gross Reserves show only the asset side, **Net Foreign Assets**
    reveal the true picture by accounting for foreign liabilities. NFA = Foreign Assets - Foreign Liabilities.
    """)

    # NFA column definitions
    st.subheader("📋 NFA Column Definitions")
    st.dataframe(pd.DataFrame(NFA_COLUMNS), hide_index=True, use_container_width=True)

    # Try to load and visualize NFA data
    st.subheader("📊 NFA Crisis Timeline")

    nfa_df = load_data("D14_reer_nfa.csv")

    if nfa_df is not None:
        nfa_df.columns = ['date', 'nfa_monetary_auth', 'nfa_commercial_banks', 'reer']
        nfa_df['date'] = pd.to_datetime(nfa_df['date'])

        # Clean up parentheses notation for negatives
        for col in ['nfa_monetary_auth', 'nfa_commercial_banks']:
            if nfa_df[col].dtype == 'object':
                nfa_df[col] = nfa_df[col].astype(str).str.replace(r'\(([^)]+)\)', r'-\1', regex=True)
                nfa_df[col] = nfa_df[col].str.replace(',', '').astype(float)

        # Filter to crisis period
        crisis_nfa = nfa_df[(nfa_df['date'] >= '2020-01-01') & (nfa_df['date'] <= '2024-12-31')]

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(
                x=crisis_nfa['date'],
                y=crisis_nfa['nfa_monetary_auth'],
                name="Monetary Authority NFA",
                line=dict(color='#1f77b4', width=2)
            ),
            secondary_y=False
        )

        fig.add_trace(
            go.Scatter(
                x=crisis_nfa['date'],
                y=crisis_nfa['nfa_commercial_banks'],
                name="Commercial Banks NFA",
                line=dict(color='#ff7f0e', width=2)
            ),
            secondary_y=False
        )

        # Add default line
        fig.add_vline(x=pd.Timestamp('2022-04-12'), line_dash="dash", line_color="red",
                      annotation_text="Default", annotation_position="top")

        # Add zero line
        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)

        fig.update_layout(
            title="Net Foreign Assets During Crisis Period",
            xaxis_title="Date",
            yaxis_title="NFA (LKR millions)",
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Key observations
        st.subheader("🔑 Key Observations")

        col1, col2 = st.columns(2)

        with col1:
            st.error("""
            **Monetary Authority NFA:**
            - Jan 2021: +418B LKR (positive)
            - Aug 2021: **Turned negative** (8 months before default)
            - Apr 2022: -1,462B LKR (peak negative)
            """)

        with col2:
            st.warning("""
            **Commercial Banks NFA:**
            - Consistently negative throughout 2021
            - Peak negative: -788B LKR (Jun 2021)
            - Recovery started mid-2022
            """)

        # Data preview
        st.subheader("📄 Data Preview")
        st.dataframe(crisis_nfa.dropna().tail(20), use_container_width=True)

    else:
        st.warning("NFA data file not found. Please ensure D14_reer_nfa.csv exists in data/processed/")


# ============================================================================
# PAGE: DATA COVERAGE
# ============================================================================

elif page == "⏱️ Data Coverage":
    st.title("⏱️ Data Coverage Analysis")

    # Load coverage data
    coverage_df = get_data_coverage()

    # Timeline visualization
    st.subheader("📅 Coverage Timeline")
    fig = create_coverage_timeline(coverage_df)
    st.plotly_chart(fig, use_container_width=True)

    # Coverage table
    st.subheader("📋 Coverage Summary")

    display_df = coverage_df.copy()
    display_df['Start'] = display_df['Start'].dt.strftime('%Y-%m')
    display_df['End'] = display_df['End'].dt.strftime('%Y-%m')
    display_df['Records'] = display_df['Records'].apply(lambda x: f"{x:,}")

    st.dataframe(display_df, hide_index=True, use_container_width=True)

    # Data gaps
    st.subheader("⚠️ Known Data Gaps")

    st.markdown("""
    | Gap | Impact | Mitigation |
    |-----|--------|------------|
    | Gross Reserves pre-Nov 2013 | Cannot backtest 2008-2009 crisis | Use IIP quarterly data or CBSL Annual Reports |
    | NFA detailed breakdown pre-2021 | Limited crisis buildup visibility | Use aggregate NFA from money supply |
    | Quarterly vs Monthly debt | Requires interpolation | Use quarterly-only analysis |
    """)

    # Crisis period coverage check
    st.subheader("✅ Crisis Period Coverage (2020-2024)")

    crisis_check = []
    for name, info in DATA_SOURCES.items():
        df = load_data(info["file"])
        if df is not None and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            crisis_data = df[(df['date'] >= '2020-01-01') & (df['date'] <= '2024-12-31')]
            coverage = "✅ Complete" if len(crisis_data) > 10 else "⚠️ Partial"
        else:
            coverage = "❌ Missing"
        crisis_check.append({"Dataset": name, "Crisis Coverage": coverage})

    st.dataframe(pd.DataFrame(crisis_check), hide_index=True, use_container_width=True)


# ============================================================================
# PAGE: REFERENCES
# ============================================================================

elif page == "🔗 References":
    st.title("🔗 References & URLs")

    st.subheader("📊 Primary Data Sources")

    sources = [
        ("CBSL Statistical Tables", "https://www.cbsl.gov.lk/en/statistics/statistical-tables"),
        ("CBSL External Sector Statistics", "https://www.cbsl.gov.lk/en/statistics/statistical-tables/external-sector"),
        ("CBSL Monetary Sector Statistics", "https://www.cbsl.gov.lk/en/statistics/statistical-tables/monetary-sector"),
        ("CBSL SDDS (Debt Data)", "https://www.cbsl.gov.lk/en/statistics/sdds-sri-lanka"),
        ("CBSL eResearch Data Library", "https://www.cbsl.lk/eresearch/"),
        ("CBSL Reserve Position Clarification", "https://www.cbsl.gov.lk/en/reserve-position"),
    ]

    for name, url in sources:
        st.markdown(f"- [{name}]({url})")

    st.subheader("📚 CBSL Table References")

    tables = [
        ("Table 2.02", "Monthly Exports", "Trade"),
        ("Table 2.04", "Monthly Imports", "Trade"),
        ("Table 2.12", "Outstanding External Debt", "External Debt"),
        ("Table 2.14.1", "Tourism Earnings", "External Flows"),
        ("Table 2.14.2", "Workers' Remittances", "External Flows"),
        ("Table 2.14.3", "CSE Portfolio Flows", "External Flows"),
        ("Table 2.15.2", "Reserve Data Template (Historical)", "SDDS Reserves"),
        ("Table 4.02", "Monetary Aggregates", "Money Supply"),
        ("Table 4.11", "Reserve Money, Multipliers, Velocity", "Money Supply"),
    ]

    st.dataframe(
        pd.DataFrame(tables, columns=["Table", "Description", "Category"]),
        hide_index=True,
        use_container_width=True
    )

    st.subheader("📖 Methodology References")

    st.markdown("""
    | Metric | Reference |
    |--------|-----------|
    | Greenspan-Guidotti Rule | Greenspan, A. (1999). "Currency reserves and debt." IMF/World Bank remarks |
    | IMF ARA Metric | IMF (2011). "Assessing Reserve Adequacy." IMF Policy Paper |
    | Import Cover | IMF traditional metric; 3-month minimum guideline |
    """)

    st.subheader("💻 Project Scripts")

    scripts = [
        ("parse_imports_iip.py", "Parse CBSL imports, exports, and IIP data"),
        ("parse_cbsl_tables.py", "Parse CBSL monetary and external data"),
        ("parse_cbsl_html_v2.py", "Parse CBSL HTML exports (NFA, money supply)"),
        ("analyze_benchmarks.py", "Calculate threshold breaches and lead times"),
        ("app_reserve_adequacy.py", "Streamlit dashboard for visualization"),
        ("app_data_dictionary.py", "This interactive data dictionary"),
    ]

    st.dataframe(
        pd.DataFrame(scripts, columns=["Script", "Purpose"]),
        hide_index=True,
        use_container_width=True
    )


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "SL-FSI Reserve Adequacy Data Dictionary | Last Updated: January 2026"
    "</div>",
    unsafe_allow_html=True
)
