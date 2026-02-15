"""Data loading functions with Streamlit caching."""

import streamlit as st
import pandas as pd
import json

from .config import (
    DATA_DIR, EXTERNAL_DIR, MERGED_DIR, PROCESSED_DIR,
    RESERVE_DATA_SOURCES, PBOC_SWAP_DATE, PBOC_SWAP_USD_M
)


def normalize_monthly_index(df, date_col="date"):
    """Normalize source timestamps to month-start for consistent merges."""
    if df is None or date_col not in df.columns:
        return None
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col]).dt.to_period("M").dt.to_timestamp(how="start")
    return out


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
    if 'exports_usd_m' in panel.columns and 'imports_usd_m' in panel.columns:
        panel['trade_balance_usd_m'] = panel['exports_usd_m'] - panel['imports_usd_m']

    ca_cols = ['exports_usd_m', 'remittances_usd_m', 'tourism_usd_m']
    if all(col in panel.columns for col in ca_cols) and 'imports_usd_m' in panel.columns:
        panel['ca_proxy_usd_m'] = (panel['exports_usd_m'].fillna(0) +
                                   panel['remittances_usd_m'].fillna(0) +
                                   panel['tourism_usd_m'].fillna(0) -
                                   panel['imports_usd_m'].fillna(0))

    # Net usable reserves (post PBOC swap)
    panel['net_usable_reserves_usd_m'] = panel['gross_reserves_usd_m'].copy()
    panel.loc[panel['date'] >= PBOC_SWAP_DATE, 'net_usable_reserves_usd_m'] -= PBOC_SWAP_USD_M

    # Greenspan-Guidotti ratio
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
