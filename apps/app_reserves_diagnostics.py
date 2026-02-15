#!/usr/bin/env python3
"""
Reserves Data Dictionary & Diagnostics for Reserve Level Forecasting

A Streamlit application providing:
1. Pre-merge diagnostics for each data source (Phase 1 tests)
2. Data quality metrics and visualizations
3. Merged reserves forecasting panel creation
4. Coverage analysis and gap identification

Run with: streamlit run app_reserves_diagnostics.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import acf, pacf
from pathlib import Path
from datetime import datetime
from scipy import stats
import json
import sys
import warnings
warnings.filterwarnings('ignore')

# Parse command-line argument for number of phases to show
# Usage: streamlit run app_reserves_diagnostics.py -- 4  (shows phases 1-4)
#        streamlit run app_reserves_diagnostics.py -- 6  (shows all phases)
MAX_PHASES = 2  # default: show only phases 1-2
if len(sys.argv) > 1:
    try:
        MAX_PHASES = min(int(sys.argv[1]), 6)  # Cap at 6 phases
    except ValueError:
        pass  # Keep default if not a valid integer

# Page configuration
st.set_page_config(
    page_title="Reserves Data Dictionary & Diagnostics",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths - anchor to project root (2 levels up from apps/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EXTERNAL_DIR = DATA_DIR / "external"
MERGED_DIR = DATA_DIR / "merged"
PROCESSED_DIR = DATA_DIR / "processed"

# Crisis dates
DEFAULT_DATE = datetime(2022, 4, 12)
CRISIS_START = datetime(2020, 1, 1)
CRISIS_END = datetime(2024, 12, 31)
PBOC_SWAP_DATE = datetime(2021, 3, 1)
PBOC_SWAP_USD_M = 1500

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
# DATA SOURCE DEFINITIONS - All Reserve-Related Data
# ============================================================================

RESERVE_DATA_SOURCES = {
    # Core Reserve Data
    "Gross Reserves (Compiled)": {
        "file": "D12_reserves_compiled.csv",
        "directory": "external",
        "frequency": "Monthly",
        "date_col": "date",
        "value_cols": ["gross_reserves_usd_m", "fx_reserves_usd_m", "imf_position_usd_m",
                       "sdrs_usd_m", "gold_usd_m", "import_cover_months"],
        "units": "USD millions",
        "category": "Core Reserves",
        "description": "Gross official reserve assets from CBSL + World Bank interpolation",
        "relevance": "TARGET VARIABLE for forecasting"
    },
    "Reserve Assets (CBSL Official)": {
        "file": "reserve_assets_monthly_cbsl.csv",
        "directory": "external",
        "frequency": "Monthly",
        "date_col": "date",
        "value_cols": ["gross_reserves_usd_m", "fx_reserves_usd_m", "imf_position_usd_m",
                       "sdrs_usd_m", "gold_usd_m", "other_reserves_usd_m"],
        "units": "USD millions",
        "category": "Core Reserves",
        "description": "Official CBSL reserve asset breakdown (Nov 2013+)",
        "relevance": "Higher-quality official data for recent period"
    },
    # Trade Flows
    "Merchandise Exports": {
        "file": "monthly_exports_usd.csv",
        "directory": "external",
        "frequency": "Monthly",
        "date_col": "date",
        "value_cols": ["exports_usd_m"],
        "units": "USD millions",
        "category": "Trade Flows",
        "description": "Monthly merchandise exports (FOB)",
        "relevance": "Primary FX inflow - drives reserve accumulation"
    },
    "Merchandise Imports": {
        "file": "monthly_imports_usd.csv",
        "directory": "external",
        "frequency": "Monthly",
        "date_col": "date",
        "value_cols": ["imports_usd_m"],
        "units": "USD millions",
        "category": "Trade Flows",
        "description": "Monthly merchandise imports (CIF)",
        "relevance": "Primary FX outflow - import cover denominator"
    },
    # External Revenue
    "Tourism Earnings": {
        "file": "tourism_earnings_monthly.csv",
        "directory": "external",
        "frequency": "Monthly",
        "date_col": "date",
        "value_cols": ["tourism_earnings_usd_m"],
        "units": "USD millions",
        "category": "External Revenue",
        "description": "Foreign exchange from tourism",
        "relevance": "Significant FX inflow - highly cyclical"
    },
    "Workers' Remittances": {
        "file": "remittances_monthly.csv",
        "directory": "external",
        "frequency": "Monthly",
        "date_col": "date",
        "value_cols": ["remittances_usd_m"],
        "units": "USD millions",
        "category": "External Revenue",
        "description": "Inward remittances from workers abroad",
        "relevance": "Stable FX inflow - buffer during crises"
    },
    # Capital Flows
    "CSE Portfolio Flows": {
        "file": "cse_flows_monthly.csv",
        "directory": "external",
        "frequency": "Monthly",
        "date_col": "date",
        "value_cols": ["cse_inflows_usd_m", "cse_outflows_usd_m", "cse_net_usd_m"],
        "units": "USD millions",
        "category": "Capital Flows",
        "description": "Foreign portfolio flows through Colombo Stock Exchange",
        "relevance": "Volatile capital flows - sudden stop risk"
    },
    "External Debt (USD)": {
        "file": "external_debt_usd_quarterly.csv",
        "directory": "external",
        "frequency": "Quarterly",
        "date_col": "date",
        "value_cols": ["govt_total_usd_m", "govt_short_term_usd_m", "govt_long_term_usd_m",
                       "central_bank_usd_m", "total_short_term_usd_m"],
        "units": "USD millions",
        "category": "Capital Flows",
        "description": "External debt by sector and maturity",
        "relevance": "Short-term debt creates refinancing pressure"
    },
    "Int'l Investment Position": {
        "file": "iip_quarterly_2025.csv",
        "directory": "external",
        "frequency": "Quarterly",
        "date_col": "date",
        "value_cols": ["portfolio_equity", "portfolio_debt", "reserve_assets", "total_liabilities"],
        "units": "USD billions",
        "category": "Capital Flows",
        "description": "External balance sheet position",
        "relevance": "Portfolio liabilities for IMF ARA metric"
    },
    # Monetary
    "Monetary Aggregates (M0, M2)": {
        "file": "monetary_aggregates_monthly.csv",
        "directory": "external",
        "frequency": "Monthly",
        "date_col": "date",
        "value_cols": ["reserve_money_m0_lkr_m", "broad_money_m2_lkr_m"],
        "units": "LKR millions",
        "category": "Monetary",
        "description": "Reserve money and broad money supply",
        "relevance": "M2 component of IMF ARA metric"
    },
    "Reserve Money Velocity": {
        "file": "reserve_money_velocity_monthly.csv",
        "directory": "external",
        "frequency": "Monthly",
        "date_col": "date",
        "value_cols": ["reserve_money_total_lkr_m", "money_multiplier_m1"],
        "units": "LKR millions / Ratio",
        "category": "Monetary",
        "description": "Reserve money and money multiplier",
        "relevance": "Credit expansion capacity indicator"
    },
    # Exchange Rate & Competitiveness
    "Exchange Rate (USD/LKR)": {
        "file": "slfsi_monthly_panel.csv",
        "directory": "merged",
        "frequency": "Monthly",
        "date_col": "date",
        "value_cols": ["usd_lkr"],
        "units": "LKR per USD",
        "category": "Exchange Rate",
        "description": "USD/LKR spot exchange rate",
        "relevance": "FX conversion + depreciation pressure"
    },
    "REER & Net Foreign Assets": {
        "file": "D14_reer_nfa.csv",
        "directory": "processed",
        "frequency": "Monthly",
        "date_col": "date",
        "value_cols": None,  # Will detect dynamically
        "units": "Index / LKR millions",
        "category": "Exchange Rate",
        "description": "Real Effective Exchange Rate and NFA",
        "relevance": "Competitiveness and external position"
    },
    # Debt
    "Central Government Debt": {
        "file": "central_govt_debt_quarterly.csv",
        "directory": "external",
        "frequency": "Quarterly",
        "date_col": "date",
        "value_cols": ["total_debt_lkr_m", "domestic_debt_lkr_m", "foreign_debt_lkr_m",
                       "total_short_term_lkr_m"],
        "units": "LKR millions",
        "category": "Debt",
        "description": "Central government debt by type",
        "relevance": "Greenspan-Guidotti ratio denominator"
    },
    # Interest Rates
    "Policy Rates": {
        "file": "D10_policy_rates_daily.csv",
        "directory": "external",
        "frequency": "Daily",
        "date_col": "date",
        "value_cols": ["sdfr", "slfr", "policy_ceiling"],
        "units": "Percentage",
        "category": "Interest Rates",
        "description": "CBSL policy rates",
        "relevance": "Monetary policy stance affecting capital flows"
    },
    "AWCMR": {
        "file": "awcmr_monthly_cbsl.csv",
        "directory": "external",
        "frequency": "Monthly",
        "date_col": "date",
        "value_cols": ["awcmr_monthly"],
        "units": "Percentage",
        "category": "Interest Rates",
        "description": "Average Weighted Call Money Rate",
        "relevance": "Interbank liquidity indicator"
    },
    # Prices
    "Gold Price": {
        "file": "D6_gold_usd.csv",
        "directory": "external",
        "frequency": "Daily",
        "date_col": "date",
        "value_cols": ["gold_usd_oz"],
        "units": "USD per ounce",
        "category": "Prices",
        "description": "Gold price (London fixing)",
        "relevance": "Affects reserve valuation (gold component)"
    },
    "NCPI Inflation": {
        "file": "D13_inflation_monthly_compiled.csv",
        "directory": "external",
        "frequency": "Monthly",
        "date_col": "date",
        "value_cols": ["ncpi_yoy_pct"],
        "units": "Percentage (YoY)",
        "category": "Prices",
        "description": "National Consumer Price Index inflation",
        "relevance": "Real reserve adequacy calculations"
    },
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data
def load_source_data(source_name):
    """Load data from a specific source."""
    if source_name not in RESERVE_DATA_SOURCES:
        return None

    info = RESERVE_DATA_SOURCES[source_name]

    if info["directory"] == "external":
        filepath = EXTERNAL_DIR / info["file"]
    elif info["directory"] == "merged":
        filepath = MERGED_DIR / info["file"]
    else:
        filepath = PROCESSED_DIR / info["file"]

    if not filepath.exists():
        return None

    try:
        df = pd.read_csv(filepath)
        if info["date_col"] in df.columns:
            df[info["date_col"]] = pd.to_datetime(df[info["date_col"]])
        return df
    except Exception as e:
        st.error(f"Error loading {source_name}: {e}")
        return None


def compute_missing_analysis(df, date_col, value_cols):
    """Compute missing value statistics."""
    if df is None or value_cols is None:
        return None

    results = []
    for col in value_cols:
        if col not in df.columns:
            continue

        total = len(df)
        missing = df[col].isna().sum()
        missing_pct = (missing / total) * 100 if total > 0 else 0

        # Check for gaps in crisis period
        if date_col in df.columns:
            crisis_df = df[(df[date_col] >= CRISIS_START) & (df[date_col] <= CRISIS_END)]
            crisis_missing = crisis_df[col].isna().sum() if len(crisis_df) > 0 else 0
            crisis_total = len(crisis_df)
        else:
            crisis_missing = 0
            crisis_total = 0

        results.append({
            "Column": col,
            "Total Rows": total,
            "Missing": missing,
            "Missing %": round(missing_pct, 1),
            "Crisis Period Missing": crisis_missing,
            "Crisis Period Total": crisis_total
        })

    return pd.DataFrame(results)


def compute_outlier_analysis(df, value_cols, method="iqr", threshold=1.5):
    """Detect outliers using IQR or Z-score method."""
    if df is None or value_cols is None:
        return None

    results = []
    for col in value_cols:
        if col not in df.columns:
            continue

        series = df[col].dropna()
        if len(series) < 10:
            continue

        if method == "iqr":
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            outliers = series[(series < lower) | (series > upper)]
        else:  # z-score
            z_scores = np.abs(stats.zscore(series))
            outliers = series[z_scores > threshold]

        results.append({
            "Column": col,
            "N": len(series),
            "Outliers": len(outliers),
            "Outlier %": round((len(outliers) / len(series)) * 100, 2),
            "Min": round(series.min(), 2),
            "Max": round(series.max(), 2),
            "Mean": round(series.mean(), 2),
            "Std": round(series.std(), 2)
        })

    return pd.DataFrame(results)


def compute_distribution_stats(df, value_cols):
    """Compute distribution statistics including normality tests."""
    if df is None or value_cols is None:
        return None

    results = []
    for col in value_cols:
        if col not in df.columns:
            continue

        series = df[col].dropna()
        if len(series) < 20:
            continue

        # Normality test (Jarque-Bera)
        try:
            jb_stat, jb_pval = stats.jarque_bera(series)
        except:
            jb_stat, jb_pval = np.nan, np.nan

        results.append({
            "Column": col,
            "N": len(series),
            "Mean": round(series.mean(), 2),
            "Median": round(series.median(), 2),
            "Std": round(series.std(), 2),
            "Skewness": round(series.skew(), 3),
            "Kurtosis": round(series.kurtosis(), 3),
            "JB Stat": round(jb_stat, 2) if not np.isnan(jb_stat) else "N/A",
            "JB p-value": round(jb_pval, 4) if not np.isnan(jb_pval) else "N/A",
            "Normal?": "Yes" if jb_pval > 0.05 else "No" if not np.isnan(jb_pval) else "N/A"
        })

    return pd.DataFrame(results)


def get_coverage_info(df, date_col):
    """Get date coverage information."""
    if df is None or date_col not in df.columns:
        return None

    df[date_col] = pd.to_datetime(df[date_col])

    return {
        "start": df[date_col].min(),
        "end": df[date_col].max(),
        "records": len(df),
        "date_range_days": (df[date_col].max() - df[date_col].min()).days
    }


def normalize_monthly_index(df, date_col="date"):
    """Normalize source timestamps to month-start for consistent merges."""
    if df is None or date_col not in df.columns:
        return None
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col]).dt.to_period("M").dt.to_timestamp(how="start")
    return out


@st.cache_data
def load_diagnostic_results():
    """Load diagnostics JSON output if available."""
    path = DATA_DIR / "diagnostics" / "diagnostic_results.json"
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data
def load_variable_quality_summary():
    """Load variable coverage/quality summary if available."""
    path = DATA_DIR / "diagnostics" / "variable_quality_summary.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_data
def create_merged_reserves_panel():
    """Create a merged panel of all reserve-related variables for forecasting."""

    # Start with the compiled reserves as base
    reserves = load_source_data("Gross Reserves (Compiled)")
    if reserves is None:
        return None

    reserves = normalize_monthly_index(reserves, "date")
    panel = reserves[['date', 'gross_reserves_usd_m', 'import_cover_months']].copy()
    panel = panel.set_index('date')

    # Add detailed reserve components from CBSL official (where available)
    cbsl_reserves = load_source_data("Reserve Assets (CBSL Official)")
    if cbsl_reserves is not None:
        cbsl_reserves = normalize_monthly_index(cbsl_reserves, "date")
        cbsl_reserves = cbsl_reserves.set_index('date')
        for col in ['fx_reserves_usd_m', 'imf_position_usd_m', 'sdrs_usd_m', 'gold_usd_m']:
            if col in cbsl_reserves.columns:
                panel[col] = cbsl_reserves[col]

    # Add exports
    exports = load_source_data("Merchandise Exports")
    if exports is not None:
        exports = normalize_monthly_index(exports, "date")
        exports = exports.set_index('date')
        panel['exports_usd_m'] = exports['exports_usd_m']

    # Add imports
    imports = load_source_data("Merchandise Imports")
    if imports is not None:
        imports = normalize_monthly_index(imports, "date")
        imports = imports.set_index('date')
        panel['imports_usd_m'] = imports['imports_usd_m']

    # Add tourism
    tourism = load_source_data("Tourism Earnings")
    if tourism is not None:
        tourism = normalize_monthly_index(tourism, "date")
        tourism = tourism.set_index('date')
        panel['tourism_usd_m'] = tourism['tourism_earnings_usd_m']

    # Add remittances
    remittances = load_source_data("Workers' Remittances")
    if remittances is not None:
        remittances = normalize_monthly_index(remittances, "date")
        remittances = remittances.set_index('date')
        panel['remittances_usd_m'] = remittances['remittances_usd_m']

    # Add CSE flows
    cse = load_source_data("CSE Portfolio Flows")
    if cse is not None:
        cse = normalize_monthly_index(cse, "date")
        cse = cse.set_index('date')
        panel['cse_net_usd_m'] = cse['cse_net_usd_m']

    # Add monetary aggregates
    monetary = load_source_data("Monetary Aggregates (M0, M2)")
    if monetary is not None:
        monetary = normalize_monthly_index(monetary, "date")
        monetary = monetary.set_index('date')
        panel['m0_lkr_m'] = monetary['reserve_money_m0_lkr_m']
        panel['m2_lkr_m'] = monetary['broad_money_m2_lkr_m']

    # Add exchange rate from panel
    fx_panel = load_source_data("Exchange Rate (USD/LKR)")
    if fx_panel is not None:
        fx_panel = normalize_monthly_index(fx_panel, "date")
        fx_panel = fx_panel.set_index('date')
        panel['usd_lkr'] = fx_panel['usd_lkr']

    # Add inflation
    inflation = load_source_data("NCPI Inflation")
    if inflation is not None:
        inflation = normalize_monthly_index(inflation, "date")
        inflation = inflation.set_index('date')
        panel['inflation_yoy_pct'] = inflation['ncpi_yoy_pct']

    # Quarterly data - will be forward-filled
    ext_debt = load_source_data("External Debt (USD)")
    if ext_debt is not None:
        ext_debt = normalize_monthly_index(ext_debt, "date")
        ext_debt = ext_debt.set_index('date')
        for col in ['govt_short_term_usd_m', 'total_short_term_usd_m']:
            if col in ext_debt.columns:
                panel[col] = ext_debt[col]

    iip = load_source_data("Int'l Investment Position")
    if iip is not None:
        iip = normalize_monthly_index(iip, "date")
        iip = iip.set_index('date')
        if 'portfolio_equity' in iip.columns and 'portfolio_debt' in iip.columns:
            panel['portfolio_liabilities_usd_b'] = iip['portfolio_equity'].fillna(0) + iip['portfolio_debt'].fillna(0)

    govt_debt = load_source_data("Central Government Debt")
    if govt_debt is not None:
        govt_debt = normalize_monthly_index(govt_debt, "date")
        govt_debt = govt_debt.set_index('date')
        panel['total_short_term_debt_lkr_m'] = govt_debt['total_short_term_lkr_m']

    # Reset index
    panel = panel.reset_index()

    # Forward-fill quarterly data
    quarterly_cols = ['govt_short_term_usd_m', 'total_short_term_usd_m',
                      'portfolio_liabilities_usd_b', 'total_short_term_debt_lkr_m']
    for col in quarterly_cols:
        if col in panel.columns:
            panel[col] = panel[col].ffill()

    # Calculate derived metrics
    # Trade balance
    if 'exports_usd_m' in panel.columns and 'imports_usd_m' in panel.columns:
        panel['trade_balance_usd_m'] = panel['exports_usd_m'] - panel['imports_usd_m']

    # Current account proxy (exports + remittances + tourism - imports)
    ca_cols = ['exports_usd_m', 'remittances_usd_m', 'tourism_usd_m']
    if all(col in panel.columns for col in ca_cols) and 'imports_usd_m' in panel.columns:
        panel['ca_proxy_usd_m'] = (panel['exports_usd_m'].fillna(0) +
                                   panel['remittances_usd_m'].fillna(0) +
                                   panel['tourism_usd_m'].fillna(0) -
                                   panel['imports_usd_m'].fillna(0))

    # Net usable reserves (post PBOC swap)
    panel['net_usable_reserves_usd_m'] = panel['gross_reserves_usd_m'].copy()
    panel.loc[panel['date'] >= PBOC_SWAP_DATE, 'net_usable_reserves_usd_m'] -= PBOC_SWAP_USD_M

    # Greenspan-Guidotti ratio (if we have short-term debt in USD)
    if 'total_short_term_usd_m' in panel.columns:
        panel['gg_ratio'] = panel['gross_reserves_usd_m'] / panel['total_short_term_usd_m']

    # M2 in USD (for ARA)
    if 'm2_lkr_m' in panel.columns and 'usd_lkr' in panel.columns:
        panel['m2_usd_m'] = panel['m2_lkr_m'] / panel['usd_lkr']

    # Reserve changes
    panel['reserve_change_usd_m'] = panel['gross_reserves_usd_m'].diff()
    panel['reserve_change_pct'] = panel['gross_reserves_usd_m'].pct_change() * 100

    # Sort by date
    panel = panel.sort_values('date').reset_index(drop=True)

    return panel


# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.title("🔬 Reserves Diagnostics")
st.sidebar.markdown("**Data Dictionary & Quality Checks**")
st.sidebar.markdown("---")

# Build pages list based on MAX_PHASES setting
ALL_PHASE_PAGES = [
    "Phase 1: Data Quality",
    "Phase 2: Stationarity",
    "Phase 3: Temporal",
    "Phase 4: Volatility",
    "Phase 5: Breaks",
    "Phase 6: Relationships",
]
visible_phases = ALL_PHASE_PAGES[:MAX_PHASES]

pages = ["Overview", "Diagnostics QA", "Source Diagnostics"] + visible_phases + ["Merged Panel", "Export Data"]

page = st.sidebar.radio("", pages)

st.sidebar.markdown("---")

# Category filter
categories = list(set(info["category"] for info in RESERVE_DATA_SOURCES.values()))
selected_categories = st.sidebar.multiselect(
    "",
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
# PAGE: OVERVIEW
# ============================================================================

if page == "Overview":
    st.title("🔬 Reserves Data Dictionary & Diagnostics")

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
         "When": "BEFORE merge", "Status": "✅ Available"},
        {"Phase": "2. Distributions", "Tests": "Normality, skewness, kurtosis",
         "When": "BEFORE merge", "Status": "✅ Available"},
        {"Phase": "3. Coverage", "Tests": "Date ranges, gaps, frequency alignment",
         "When": "BEFORE merge", "Status": "✅ Available"},
        {"Phase": "4. Stationarity", "Tests": "ADF, KPSS, Phillips-Perron",
         "When": "AFTER merge", "Status": "🔜 Next step"},
        {"Phase": "5. Relationships", "Tests": "Cross-correlation, Granger causality",
         "When": "AFTER merge", "Status": "🔜 Next step"},
    ])

    st.dataframe(phases, hide_index=True, use_container_width=True)

    st.markdown("---")

    # Data source summary table
    st.subheader("📂 Reserve-Related Data Sources")

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
            "Status": "✅" if df is not None else "❌"
        })

    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)


# ============================================================================
# PAGE: DIAGNOSTICS QA
# ============================================================================

elif page == "Diagnostics QA":
    st.title("✅ Diagnostics Quality Assurance")
    st.markdown("*Check test validity, coverage, and skipped variables before modeling*")

    diag_json = load_diagnostic_results()
    quality_df = load_variable_quality_summary()

    if diag_json is None or quality_df is None:
        st.warning("Run `python scripts/run_diagnostics.py` first to generate QA artifacts.")
    else:
        meta = diag_json.get("metadata", {})
        skipped = pd.DataFrame(meta.get("variables_skipped", []))
        tested = meta.get("variables_tested", [])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Variables Requested", len(meta.get("variables_requested", [])))
        col2.metric("Variables Tested", len(tested))
        col3.metric("Variables Skipped", len(skipped))
        col4.metric("Run Timestamp", str(meta.get("timestamp", "N/A"))[:19])

        st.markdown("---")
        st.subheader("📋 Variable Coverage & Readiness")
        st.dataframe(quality_df, hide_index=True, use_container_width=True)

        usable = quality_df[quality_df["is_usable"] == True] if "is_usable" in quality_df.columns else pd.DataFrame()
        if not usable.empty:
            st.subheader("📊 Coverage of Usable Variables")
            fig = go.Figure(
                go.Bar(
                    x=usable["variable"],
                    y=usable["coverage_pct"],
                    marker_color="#2ecc71",
                    text=usable["coverage_pct"].astype(str) + "%",
                    textposition="auto",
                )
            )
            fig.update_layout(
                title="Non-null Coverage by Variable",
                xaxis_title="Variable",
                yaxis_title="Coverage %",
                xaxis_tickangle=-45,
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

        if not skipped.empty:
            st.subheader("⚠️ Skipped Variables")
            st.dataframe(skipped, hide_index=True, use_container_width=True)

        st.info(
            "Granger causality is now run on first-differenced series to reduce spurious inference from non-stationary levels."
        )


# ============================================================================
# PAGE: SOURCE DIAGNOSTICS
# ============================================================================

elif page == "Source Diagnostics":
    st.title("🔍 Individual Source Diagnostics")

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
            st.subheader("📄 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)

            # Column info
            st.subheader("📊 Column Information")
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
                st.subheader("📈 Time Series Visualization")

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


# ============================================================================
# PAGE: MISSING VALUES
# ============================================================================

elif page == "Phase 1: Data Quality":
    st.title("📊 Missing Value Analysis")
    st.markdown("*Phase 1 Diagnostic: Identify gaps before merging*")

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

        # Summary
        st.subheader("📋 Missing Value Summary")

        # Overall stats
        col1, col2, col3 = st.columns(3)

        total_cols = len(combined)
        cols_with_missing = len(combined[combined["Missing"] > 0])
        crisis_missing = len(combined[combined["Crisis Period Missing"] > 0])

        col1.metric("Total Columns Analyzed", total_cols)
        col2.metric("Columns with Missing Data", cols_with_missing)
        col3.metric("Columns with Crisis Gaps", crisis_missing)

        st.markdown("---")

        # Detailed table
        st.subheader("📝 Detailed Missing Value Report")

        # Sort by missing percentage
        display_df = combined.sort_values("Missing %", ascending=False)

        # Color code based on missing percentage
        def highlight_missing(val):
            if isinstance(val, (int, float)):
                if val > 10:
                    return 'background-color: #f8d7da'
                elif val > 0:
                    return 'background-color: #fff3cd'
            return ''

        st.dataframe(display_df, hide_index=True, use_container_width=True)

        # Visualization
        st.subheader("📊 Missing Values by Source")

        fig = px.bar(
            combined[combined["Missing"] > 0],
            x="Column",
            y="Missing %",
            color="Category",
            hover_data=["Source", "Total Rows", "Missing"],
            title="Missing Value Percentage by Column"
        )
        fig.update_layout(height=500, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        # Crisis period focus
        st.subheader("⚠️ Crisis Period Gaps (2020-2024)")
        crisis_gaps = combined[combined["Crisis Period Missing"] > 0]
        if len(crisis_gaps) > 0:
            st.warning(f"**{len(crisis_gaps)} columns** have missing data during the crisis period!")
            st.dataframe(crisis_gaps[["Source", "Column", "Crisis Period Missing", "Crisis Period Total"]],
                         hide_index=True, use_container_width=True)
        else:
            st.success("✅ All columns have complete data for the crisis period!")
    else:
        st.warning("No data sources available for analysis")


# ============================================================================
# PAGE: OUTLIER DETECTION
# ============================================================================

elif page == "🎯 Outlier Detection":
    st.title("🎯 Outlier Detection")
    st.markdown("*Phase 1 Diagnostic: Identify anomalous values*")

    # Method selection
    col1, col2 = st.columns(2)
    method = col1.selectbox("Detection Method:", ["IQR", "Z-Score"])
    threshold = col2.slider("Threshold:", 1.0, 5.0, 1.5 if method == "IQR" else 3.0, 0.5)

    all_outliers = []

    for name, info in RESERVE_DATA_SOURCES.items():
        if info["category"] not in selected_categories:
            continue

        df = load_source_data(name)
        if df is not None and info["value_cols"]:
            outlier_df = compute_outlier_analysis(df, info["value_cols"],
                                                   method=method.lower().replace("-", ""),
                                                   threshold=threshold)
            if outlier_df is not None:
                outlier_df["Source"] = name
                outlier_df["Category"] = info["category"]
                all_outliers.append(outlier_df)

    if all_outliers:
        combined = pd.concat(all_outliers, ignore_index=True)

        st.subheader("📋 Outlier Summary")

        col1, col2, col3 = st.columns(3)
        total_cols = len(combined)
        cols_with_outliers = len(combined[combined["Outliers"] > 0])
        total_outliers = combined["Outliers"].sum()

        col1.metric("Columns Analyzed", total_cols)
        col2.metric("Columns with Outliers", cols_with_outliers)
        col3.metric("Total Outliers Found", int(total_outliers))

        st.markdown("---")

        st.subheader("📝 Detailed Outlier Report")
        display_df = combined.sort_values("Outlier %", ascending=False)
        st.dataframe(display_df, hide_index=True, use_container_width=True)

        # Visualization
        st.subheader("📊 Outlier Distribution")

        fig = px.scatter(
            combined[combined["Outliers"] > 0],
            x="Column",
            y="Outlier %",
            size="Outliers",
            color="Category",
            hover_data=["Source", "Min", "Max", "Mean", "Std"],
            title="Outlier Percentage by Column"
        )
        fig.update_layout(height=500, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        # Box plots for selected source
        st.subheader("📦 Box Plot Analysis")
        selected_source = st.selectbox("Select source for box plots:",
                                        list(RESERVE_DATA_SOURCES.keys()))

        info = RESERVE_DATA_SOURCES[selected_source]
        df = load_source_data(selected_source)

        if df is not None and info["value_cols"]:
            plot_cols = [col for col in info["value_cols"] if col in df.columns]
            if plot_cols:
                fig = go.Figure()
                for col in plot_cols:
                    fig.add_trace(go.Box(y=df[col].dropna(), name=col))

                fig.update_layout(
                    title=f"Box Plots for {selected_source}",
                    yaxis_title=info["units"],
                    height=450
                )
                st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PAGE: DISTRIBUTIONS
# ============================================================================

elif page == "📈 Distributions":
    st.title("📈 Distribution Analysis")
    st.markdown("*Phase 2 Diagnostic: Normality and statistical properties*")

    all_distributions = []

    for name, info in RESERVE_DATA_SOURCES.items():
        if info["category"] not in selected_categories:
            continue

        df = load_source_data(name)
        if df is not None and info["value_cols"]:
            dist_df = compute_distribution_stats(df, info["value_cols"])
            if dist_df is not None:
                dist_df["Source"] = name
                dist_df["Category"] = info["category"]
                all_distributions.append(dist_df)

    if all_distributions:
        combined = pd.concat(all_distributions, ignore_index=True)

        st.subheader("📋 Distribution Summary")

        col1, col2, col3 = st.columns(3)
        total_cols = len(combined)
        normal_cols = len(combined[combined["Normal?"] == "Yes"])
        non_normal = len(combined[combined["Normal?"] == "No"])

        col1.metric("Columns Analyzed", total_cols)
        col2.metric("Normally Distributed", normal_cols)
        col3.metric("Non-Normal", non_normal)

        st.info("""
        **Interpretation:**
        - Non-normal distributions may require **log transformation** or **robust methods**
        - High skewness (|skew| > 1) indicates asymmetry
        - High kurtosis (|kurt| > 3) indicates heavy tails (outlier-prone)
        """)

        st.markdown("---")

        st.subheader("📝 Distribution Statistics")
        st.dataframe(combined, hide_index=True, use_container_width=True)

        # Histogram for selected column
        st.subheader("📊 Histogram Analysis")

        selected_source = st.selectbox("Select source:", list(RESERVE_DATA_SOURCES.keys()))
        info = RESERVE_DATA_SOURCES[selected_source]
        df = load_source_data(selected_source)

        if df is not None and info["value_cols"]:
            plot_cols = [col for col in info["value_cols"] if col in df.columns]
            if plot_cols:
                selected_col = st.selectbox("Select column:", plot_cols)

                col1, col2 = st.columns(2)

                with col1:
                    fig = px.histogram(df, x=selected_col, nbins=30,
                                       title=f"Distribution of {selected_col}")
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # Q-Q plot
                    series = df[selected_col].dropna()
                    if len(series) > 10:
                        theoretical_q = stats.norm.ppf(np.linspace(0.01, 0.99, len(series)))
                        sample_q = np.sort(series)

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=theoretical_q,
                            y=sample_q,
                            mode='markers',
                            name='Data'
                        ))
                        fig.add_trace(go.Scatter(
                            x=[theoretical_q.min(), theoretical_q.max()],
                            y=[theoretical_q.min(), theoretical_q.max()],
                            mode='lines',
                            name='Normal Line',
                            line=dict(color='red', dash='dash')
                        ))
                        fig.update_layout(
                            title=f"Q-Q Plot for {selected_col}",
                            xaxis_title="Theoretical Quantiles",
                            yaxis_title="Sample Quantiles"
                        )
                        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PAGE: COVERAGE TIMELINE
# ============================================================================

elif page == "⏱️ Coverage Timeline":
    st.title("⏱️ Data Coverage Timeline")
    st.markdown("*Visualize date ranges and identify gaps*")

    # Build coverage data
    coverage_data = []

    for name, info in RESERVE_DATA_SOURCES.items():
        if info["category"] not in selected_categories:
            continue

        df = load_source_data(name)
        coverage = get_coverage_info(df, info["date_col"]) if df is not None else None

        if coverage:
            coverage_data.append({
                "Source": name,
                "Category": info["category"],
                "Frequency": info["frequency"],
                "Start": coverage["start"],
                "End": coverage["end"],
                "Records": coverage["records"]
            })

    if coverage_data:
        coverage_df = pd.DataFrame(coverage_data)

        # Gantt-style timeline
        st.subheader("📅 Coverage Timeline")

        fig = go.Figure()

        colors = px.colors.qualitative.Set2
        categories = coverage_df["Category"].unique()
        category_colors = {cat: colors[i % len(colors)] for i, cat in enumerate(categories)}

        for i, row in coverage_df.iterrows():
            fig.add_trace(go.Scatter(
                x=[row["Start"], row["End"]],
                y=[row["Source"], row["Source"]],
                mode='lines',
                line=dict(color=category_colors[row["Category"]], width=15),
                name=row["Category"],
                showlegend=i == coverage_df[coverage_df["Category"] == row["Category"]].index[0],
                hovertemplate=f"<b>{row['Source']}</b><br>" +
                              f"Start: {row['Start'].strftime('%Y-%m')}<br>" +
                              f"End: {row['End'].strftime('%Y-%m')}<br>" +
                              f"Records: {row['Records']:,}<extra></extra>"
            ))

        # Add crisis period
        fig.add_vrect(x0=CRISIS_START, x1=CRISIS_END,
                      fillcolor="rgba(255,0,0,0.1)", layer="below", line_width=0)

        # Add default line
        fig.add_vline(x=DEFAULT_DATE, line_dash="dash", line_color="red",
                      annotation_text="Default (Apr 2022)")

        fig.update_layout(
            title="Data Coverage by Source",
            xaxis_title="Date",
            yaxis_title="Data Source",
            height=600,
            xaxis=dict(range=['2005-01-01', '2026-06-01']),
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Coverage table
        st.subheader("📋 Coverage Details")

        display_df = coverage_df.copy()
        display_df["Start"] = display_df["Start"].dt.strftime("%Y-%m-%d")
        display_df["End"] = display_df["End"].dt.strftime("%Y-%m-%d")
        display_df["Records"] = display_df["Records"].apply(lambda x: f"{x:,}")

        st.dataframe(display_df.sort_values("Category"), hide_index=True, use_container_width=True)

        # Gap analysis
        st.subheader("⚠️ Coverage Gaps for Key Periods")

        crisis_check = []
        for name, info in RESERVE_DATA_SOURCES.items():
            df = load_source_data(name)
            if df is not None and info["date_col"] in df.columns:
                df[info["date_col"]] = pd.to_datetime(df[info["date_col"]])

                # Check crisis period
                crisis_df = df[(df[info["date_col"]] >= CRISIS_START) &
                               (df[info["date_col"]] <= CRISIS_END)]

                # Check pre-crisis (for baseline comparison)
                pre_crisis = df[(df[info["date_col"]] >= datetime(2015, 1, 1)) &
                                (df[info["date_col"]] < CRISIS_START)]

                crisis_check.append({
                    "Source": name,
                    "Category": info["category"],
                    "Pre-Crisis (2015-2019)": "✅" if len(pre_crisis) > 0 else "❌",
                    "Crisis (2020-2024)": "✅" if len(crisis_df) > 0 else "❌",
                    "Crisis Records": len(crisis_df)
                })

        st.dataframe(pd.DataFrame(crisis_check), hide_index=True, use_container_width=True)


# ============================================================================
# PAGE: MERGED PANEL
# ============================================================================

elif page == "Phase 2: Stationarity":
    st.title("📉 Phase 2: Stationarity & Integration Order")

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
        st.subheader("📋 Integration Order Summary")
        st.dataframe(int_df, hide_index=True, use_container_width=True)

        if diag_json is not None:
            za_rows = pd.DataFrame(diag_json.get("phase2_stationarity", {}).get("zivot_andrews", []))
            if not za_rows.empty and {"variable", "za_statistic", "break_date"}.issubset(za_rows.columns):
                st.subheader("🧭 Zivot-Andrews Break Candidates")
                st.dataframe(
                    za_rows[["variable", "za_statistic", "break_date", "stationary_5pct"]],
                    hide_index=True,
                    use_container_width=True,
                )

        # Interpretation
        st.subheader("🎯 Key Findings")
        st.info("""
        **Target Variable (`gross_reserves_usd_m`):**
        - ADF: p=0.549 → Cannot reject unit root
        - KPSS: p<0.01 → Reject stationarity
        - **Conclusion:** I(1) - First differencing required

        **Implication:** Use `reserve_change_usd_m` (first difference) or apply differencing before modeling.
        """)

        # ACF visualization for target
        st.subheader("📈 ACF of Reserves (showing persistence)")

        panel = create_merged_reserves_panel()
        if panel is not None and 'gross_reserves_usd_m' in panel.columns:
            from statsmodels.tsa.stattools import acf
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


elif page == "Phase 3: Temporal":
    st.title("🔄 Phase 3: Temporal Dependence Structure")
    st.markdown("*Autocorrelation, seasonality, and persistence analysis*")

    panel = create_merged_reserves_panel()

    if panel is not None:
        # Variable selector
        num_cols = [c for c in panel.columns if c != 'date' and panel[c].dtype in ['float64', 'int64']]
        selected_var = st.selectbox("Select variable:", num_cols, index=0)

        series = panel[selected_var].dropna()

        if len(series) > 30:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("ACF (Autocorrelation)")
                from statsmodels.tsa.stattools import acf
                acf_vals = acf(series, nlags=24)

                fig = go.Figure()
                fig.add_trace(go.Bar(x=list(range(len(acf_vals))), y=acf_vals, marker_color='#3498db'))
                conf = 1.96 / np.sqrt(len(series))
                fig.add_hline(y=conf, line_dash="dash", line_color="red")
                fig.add_hline(y=-conf, line_dash="dash", line_color="red")
                fig.update_layout(title=f"ACF of {selected_var}", xaxis_title="Lag", yaxis_title="ACF", height=350)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("PACF (Partial Autocorrelation)")
                from statsmodels.tsa.stattools import pacf
                pacf_vals = pacf(series, nlags=24)

                fig = go.Figure()
                fig.add_trace(go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals, marker_color='#e74c3c'))
                fig.add_hline(y=conf, line_dash="dash", line_color="red")
                fig.add_hline(y=-conf, line_dash="dash", line_color="red")
                fig.update_layout(title=f"PACF of {selected_var}", xaxis_title="Lag", yaxis_title="PACF", height=350)
                st.plotly_chart(fig, use_container_width=True)

            # STL decomposition
            st.subheader("📊 Seasonal Decomposition (STL)")

            if len(series) >= 24:
                from statsmodels.tsa.seasonal import STL
                try:
                    stl = STL(series, period=12, robust=True)
                    result = stl.fit()

                    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                        subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'))

                    fig.add_trace(go.Scatter(y=series.values, mode='lines', name='Original'), row=1, col=1)
                    fig.add_trace(go.Scatter(y=result.trend, mode='lines', name='Trend', line=dict(color='#e74c3c')), row=2, col=1)
                    fig.add_trace(go.Scatter(y=result.seasonal, mode='lines', name='Seasonal', line=dict(color='#2ecc71')), row=3, col=1)
                    fig.add_trace(go.Scatter(y=result.resid, mode='lines', name='Residual', line=dict(color='#9b59b6')), row=4, col=1)

                    fig.update_layout(height=700, showlegend=False, title=f"STL Decomposition of {selected_var}")
                    st.plotly_chart(fig, use_container_width=True)

                    # Strength metrics
                    var_resid = np.var(result.resid)
                    var_deseas = np.var(result.trend + result.resid)
                    var_detrend = np.var(result.seasonal + result.resid)

                    trend_strength = max(0, 1 - var_resid / var_deseas) if var_deseas > 0 else 0
                    seasonal_strength = max(0, 1 - var_resid / var_detrend) if var_detrend > 0 else 0

                    col1, col2 = st.columns(2)
                    col1.metric("Trend Strength", f"{trend_strength:.2%}")
                    col2.metric("Seasonal Strength", f"{seasonal_strength:.2%}")
                except Exception as e:
                    st.warning(f"Could not perform STL decomposition: {e}")
        else:
            st.warning("Insufficient data for temporal analysis")


elif page == "Phase 4: Volatility":
    st.title("📈 Phase 4: Volatility & Heteroskedasticity")
    st.markdown("*ARCH effects and volatility regime analysis*")

    # Load ARCH results
    arch_path = DATA_DIR / "diagnostics" / "arch_summary.csv"
    if arch_path.exists():
        arch_df = pd.read_csv(arch_path)

        st.subheader("📋 ARCH-LM Test Results")
        st.dataframe(arch_df, hide_index=True, use_container_width=True)

        # Visualize ARCH effects
        st.subheader("📊 ARCH Effects by Variable")

        fig = go.Figure()
        colors = ['#e74c3c' if x else '#2ecc71' for x in arch_df['has_arch_effects']]

        fig.add_trace(go.Bar(
            x=arch_df['variable'],
            y=arch_df['arch_lm_stat'],
            marker_color=colors,
            text=arch_df['arch_lm_pvalue'].apply(lambda x: f'p={x:.3f}'),
            textposition='auto'
        ))

        fig.add_hline(y=21.03, line_dash="dash", line_color="orange",
                      annotation_text="5% critical value (χ² 12df)")

        fig.update_layout(
            title="ARCH-LM Test Statistics (Red = Significant ARCH effects)",
            xaxis_title="Variable",
            yaxis_title="ARCH-LM Statistic",
            height=450,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)

    # Rolling volatility
    st.subheader("📈 Rolling Volatility Analysis")

    panel = create_merged_reserves_panel()
    if panel is not None:
        fig = go.Figure()

        # Calculate rolling volatility for reserves
        for var, color in [('gross_reserves_usd_m', '#3498db'), ('reserve_change_usd_m', '#e74c3c')]:
            if var in panel.columns:
                rolling_vol = panel.set_index('date')[var].rolling(12).std()
                fig.add_trace(go.Scatter(
                    x=rolling_vol.index,
                    y=rolling_vol.values,
                    mode='lines',
                    name=f'{var} (12m rolling std)',
                    line=dict(color=color)
                ))

        fig.add_vrect(x0=CRISIS_START, x1=CRISIS_END,
                      fillcolor="rgba(255,0,0,0.1)", layer="below", line_width=0)
        fig.add_vline(x=DEFAULT_DATE, line_dash="dash", line_color="red",
                      annotation_text="Default")

        fig.update_layout(
            title="Rolling 12-Month Volatility",
            xaxis_title="Date",
            yaxis_title="Standard Deviation",
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)

        # Volatility regime comparison
        st.subheader("📊 Volatility Regime Comparison")

        if 'gross_reserves_usd_m' in panel.columns:
            pre_crisis = panel[(panel['date'] >= '2015-01-01') & (panel['date'] < CRISIS_START)]['gross_reserves_usd_m'].std()
            crisis = panel[(panel['date'] >= CRISIS_START) & (panel['date'] <= '2022-12-31')]['gross_reserves_usd_m'].std()
            post_crisis = panel[panel['date'] >= '2023-01-01']['gross_reserves_usd_m'].std()

            col1, col2, col3 = st.columns(3)
            col1.metric("Pre-Crisis Vol (2015-2019)", f"{pre_crisis:.0f}")
            col2.metric("Crisis Vol (2020-2022)", f"{crisis:.0f}", f"{crisis/pre_crisis:.1f}×")
            col3.metric("Post-Crisis Vol (2023+)", f"{post_crisis:.0f}", f"{post_crisis/pre_crisis:.1f}×")


elif page == "Phase 5: Breaks":
    st.title("🔀 Phase 5: Structural Break Detection")
    st.markdown("*Chow test and CUSUM analysis for regime changes*")

    # Load Chow test results
    chow_path = DATA_DIR / "diagnostics" / "chow_test_summary.csv"
    if chow_path.exists():
        chow_df = pd.read_csv(chow_path)

        st.subheader("📋 Chow Test Results (Break Date: April 2022)")
        st.dataframe(chow_df, hide_index=True, use_container_width=True)

        # Visualize
        st.subheader("📊 Structural Break Detection")

        fig = go.Figure()
        colors = ['#e74c3c' if x else '#2ecc71' for x in chow_df['break_confirmed']]

        fig.add_trace(go.Bar(
            x=chow_df['variable'],
            y=chow_df['f_statistic'],
            marker_color=colors,
            text=chow_df['p_value'].apply(lambda x: f'p={x:.3f}'),
            textposition='auto'
        ))

        fig.add_hline(y=3.0, line_dash="dash", line_color="orange",
                      annotation_text="Approximate 5% critical value")

        fig.update_layout(
            title="Chow Test F-Statistics (Red = Significant Break)",
            xaxis_title="Variable",
            yaxis_title="F-Statistic",
            height=450,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)

        # Key finding
        st.subheader("⚠️ Surprising Finding")
        st.warning("""
        **The Chow test does NOT detect a structural break in reserves at April 2022.**

        This suggests:
        1. The crisis affected the **level** but not the **autoregressive dynamics**
        2. The reserve decline was a **gradual process** rather than an abrupt regime shift
        3. The PBOC swap may have masked the break in gross reserves

        However, **exports** and **trade balance** DO show clear structural breaks.
        """)

    # Time series with break
    st.subheader("📈 Reserve Levels with Potential Break Points")

    panel = create_merged_reserves_panel()
    if panel is not None and 'gross_reserves_usd_m' in panel.columns:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=panel['date'],
            y=panel['gross_reserves_usd_m'],
            mode='lines',
            name='Gross Reserves',
            line=dict(color='#3498db', width=2)
        ))

        # Add break lines
        fig.add_vline(x=DEFAULT_DATE, line_dash="dash", line_color="red",
                      annotation_text="Default (Apr 2022)")
        fig.add_vline(x=PBOC_DATE, line_dash="dash", line_color="orange",
                      annotation_text="PBOC Swap (Mar 2021)")

        fig.update_layout(
            title="Reserve Levels with Key Dates",
            xaxis_title="Date",
            yaxis_title="USD Millions",
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)


elif page == "Phase 6: Relationships":
    st.title("🔗 Phase 6: Relationship Analysis")
    st.markdown("*Cross-correlation and Granger causality tests*")

    # Load Granger results
    gc_path = DATA_DIR / "diagnostics" / "granger_causality_summary.csv"
    if gc_path.exists():
        gc_df = pd.read_csv(gc_path)

        st.subheader("📋 Granger Causality Test Results")
        st.markdown("*H₀: X does not Granger-cause gross_reserves_usd_m*")
        st.dataframe(gc_df, hide_index=True, use_container_width=True)

        # Visualize
        fig = go.Figure()
        colors = ['#e74c3c' if x else '#95a5a6' for x in gc_df['granger_causes']]

        fig.add_trace(go.Bar(
            x=gc_df['test'],
            y=gc_df['best_p_value'],
            marker_color=colors,
            text=gc_df['best_p_value'].apply(lambda x: f'p={x:.3f}'),
            textposition='auto'
        ))

        fig.add_hline(y=0.05, line_dash="dash", line_color="red",
                      annotation_text="5% significance")

        fig.update_layout(
            title="Granger Causality p-values (Lower = More Evidence of Causality)",
            xaxis_title="Test",
            yaxis_title="p-value",
            height=400,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info("""
        **Key Finding:** Granger tests are evaluated on first-differenced series.

        This suggests:
        - Inference is less sensitive to spurious level relationships
        - Contemporaneous dynamics may dominate lagged effects
        - Consider VAR/SVAR designs with simultaneous channels
        """)

    # Cross-correlation visualization
    st.subheader("📈 Cross-Correlation Analysis")

    panel = create_merged_reserves_panel()
    if panel is not None:
        target = 'gross_reserves_usd_m'
        predictors = ['exports_usd_m', 'imports_usd_m', 'remittances_usd_m', 'cse_net_usd_m']
        predictors = [p for p in predictors if p in panel.columns]

        selected_pred = st.selectbox("Select predictor:", predictors)

        if selected_pred and target in panel.columns:
            common = panel[[target, selected_pred]].dropna()

            if len(common) > 30:
                # Compute CCF manually
                max_lag = 12
                ccf_vals = []
                for lag in range(-max_lag, max_lag + 1):
                    if lag < 0:
                        corr = common[target].iloc[-lag:].corr(common[selected_pred].iloc[:lag])
                    elif lag > 0:
                        corr = common[target].iloc[:-lag].corr(common[selected_pred].iloc[lag:])
                    else:
                        corr = common[target].corr(common[selected_pred])
                    ccf_vals.append(corr if not np.isnan(corr) else 0)

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=list(range(-max_lag, max_lag + 1)),
                    y=ccf_vals,
                    marker_color='#3498db'
                ))

                conf = 1.96 / np.sqrt(len(common))
                fig.add_hline(y=conf, line_dash="dash", line_color="red")
                fig.add_hline(y=-conf, line_dash="dash", line_color="red")

                fig.update_layout(
                    title=f"Cross-Correlation: {selected_pred} vs {target}",
                    xaxis_title="Lag (negative = predictor leads)",
                    yaxis_title="Correlation",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

                max_idx = np.argmax(np.abs(ccf_vals))
                max_lag_val = list(range(-max_lag, max_lag + 1))[max_idx]
                st.metric("Max Correlation", f"{ccf_vals[max_idx]:.3f}", f"at lag {max_lag_val}")


elif page == "Merged Panel":
    st.title("🔗 Merged Reserves Forecasting Panel")
    st.markdown("*Combined monthly panel for reserve level forecasting*")

    with st.spinner("Creating merged panel..."):
        panel = create_merged_reserves_panel()

    if panel is not None:
        st.success(f"✅ Created panel with **{len(panel)} rows** and **{len(panel.columns)} columns**")

        # Panel overview
        st.subheader("📋 Panel Overview")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Date Range", f"{panel['date'].min().strftime('%Y-%m')} to {panel['date'].max().strftime('%Y-%m')}")
        col2.metric("Total Rows", f"{len(panel):,}")
        col3.metric("Total Columns", len(panel.columns))
        col4.metric("Target Variable", "gross_reserves_usd_m")

        st.markdown("---")

        # Column categories
        st.subheader("📊 Panel Variables by Category")

        var_categories = {
            "Target": ["gross_reserves_usd_m", "net_usable_reserves_usd_m", "reserve_change_usd_m", "reserve_change_pct"],
            "Reserve Components": ["fx_reserves_usd_m", "imf_position_usd_m", "sdrs_usd_m", "gold_usd_m"],
            "Trade Flows": ["exports_usd_m", "imports_usd_m", "trade_balance_usd_m"],
            "External Revenue": ["tourism_usd_m", "remittances_usd_m"],
            "Capital Flows": ["cse_net_usd_m", "govt_short_term_usd_m", "total_short_term_usd_m", "portfolio_liabilities_usd_b"],
            "Monetary": ["m0_lkr_m", "m2_lkr_m", "m2_usd_m"],
            "Exchange Rate": ["usd_lkr"],
            "Prices": ["inflation_yoy_pct"],
            "Derived Metrics": ["import_cover_months", "gg_ratio", "ca_proxy_usd_m"]
        }

        for cat, vars in var_categories.items():
            available = [v for v in vars if v in panel.columns]
            if available:
                with st.expander(f"**{cat}** ({len(available)} variables)"):
                    for var in available:
                        non_null = panel[var].notna().sum()
                        pct = (non_null / len(panel)) * 100
                        st.markdown(f"- `{var}`: {non_null:,} values ({pct:.1f}% coverage)")

        st.markdown("---")

        # Data preview
        st.subheader("📄 Data Preview")
        st.dataframe(panel.tail(20), use_container_width=True)

        # Correlation matrix
        st.subheader("📈 Correlation Matrix (Key Variables)")

        key_vars = ['gross_reserves_usd_m', 'exports_usd_m', 'imports_usd_m',
                    'remittances_usd_m', 'tourism_usd_m', 'cse_net_usd_m', 'usd_lkr']
        key_vars = [v for v in key_vars if v in panel.columns]

        if len(key_vars) > 2:
            corr_df = panel[key_vars].corr()

            fig = px.imshow(
                corr_df,
                x=key_vars,
                y=key_vars,
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1,
                title="Correlation Matrix"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

        # Time series plot
        st.subheader("📈 Key Variables Over Time")

        plot_vars = st.multiselect(
            "Select variables to plot:",
            [col for col in panel.columns if col != 'date'],
            default=['gross_reserves_usd_m']
        )

        if plot_vars:
            fig = go.Figure()
            for var in plot_vars:
                fig.add_trace(go.Scatter(
                    x=panel['date'],
                    y=panel[var],
                    mode='lines',
                    name=var
                ))

            fig.add_vrect(x0=CRISIS_START, x1=CRISIS_END,
                          fillcolor="rgba(255,0,0,0.1)", layer="below", line_width=0)
            fig.add_vline(x=DEFAULT_DATE, line_dash="dash", line_color="red")

            fig.update_layout(
                title="Selected Variables Over Time",
                xaxis_title="Date",
                yaxis_title="Value",
                height=500,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Could not create merged panel. Check that core reserve data is available.")


# ============================================================================
# PAGE: EXPORT DATA
# ============================================================================

elif page == "Export Data":
    st.title("💾 Export Data")
    st.markdown("*Download merged panel and diagnostic reports*")

    # Merged panel export
    st.subheader("📦 Merged Reserves Forecasting Panel")

    with st.spinner("Creating merged panel..."):
        panel = create_merged_reserves_panel()

    if panel is not None:
        st.success(f"Panel ready: {len(panel)} rows × {len(panel.columns)} columns")

        # Preview
        st.dataframe(panel.head(), use_container_width=True)

        # Download button
        csv = panel.to_csv(index=False)
        st.download_button(
            label="📥 Download Merged Panel (CSV)",
            data=csv,
            file_name="reserves_forecasting_panel.csv",
            mime="text/csv"
        )

        # Save to data/merged
        if st.button("💾 Save to data/merged/reserves_forecasting_panel.csv"):
            output_path = MERGED_DIR / "reserves_forecasting_panel.csv"
            panel.to_csv(output_path, index=False)
            st.success(f"Saved to {output_path}")

    st.markdown("---")

    # Diagnostic reports
    st.subheader("📋 Diagnostic Reports")

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
                label="📥 Download Missing Value Report",
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
                label="📥 Download Distribution Report",
                data=csv_dist,
                file_name="distribution_report.csv",
                mime="text/csv"
            )


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
