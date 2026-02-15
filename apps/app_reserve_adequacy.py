#!/usr/bin/env python3
"""
Reserve Adequacy Dashboard

Streamlit app to visualize reserve adequacy metrics including:
- Greenspan-Guidotti ratio (Reserves / Short-term Debt)
- Import Cover (Months of imports)
- Net Usable Reserves
- Data coverage gaps

Created: January 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Key dates as datetime objects for plotly
DEFAULT_DATE = datetime(2022, 4, 12)
CRISIS_START = datetime(2020, 1, 1)
CRISIS_END = datetime(2024, 12, 31)

# Page config
st.set_page_config(
    page_title="Reserve Adequacy Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
DATA_DIR = Path("data")
EXTERNAL_DIR = DATA_DIR / "external"
MERGED_DIR = DATA_DIR / "merged"
PBOC_SWAP_USD_M = 1500  # Encumbered swap amount used for net reserves adjustments

SCENARIO_SHOCKS = {
    "Baseline": {
        "exports": 1.0,
        "m2": 1.0,
        "debt": 1.0,
        "portfolio": 1.0,
        "reserves": 1.0,
        "imports": 1.0,
        "remittances": 1.0,
        "tourism": 1.0,
        "cse": 1.0,
    },
    "Downside": {
        "exports": 0.9,      # weaker exports growth
        "m2": 0.98,          # slower monetary expansion
        "debt": 1.05,        # higher ST debt growth
        "portfolio": 1.05,   # higher portfolio liabilities
        "reserves": 0.97,    # softer reserve accumulation (fallback only)
        "imports": 1.05,     # higher imports
        "remittances": 0.92, # weaker Gulf employment / hawala channel
        "tourism": 0.85,     # external demand shock to tourism
        "cse": 0.8,          # portfolio outflows
    },
    "Upside": {
        "exports": 1.05,     # stronger exports growth
        "m2": 1.02,          # modestly faster money growth
        "debt": 0.95,        # slower ST debt accumulation
        "portfolio": 0.95,   # slower portfolio liabilities growth
        "reserves": 1.02,    # faster reserve build-up (fallback only)
        "imports": 0.98,     # softer imports
        "remittances": 1.05, # stronger remittance corridor
        "tourism": 1.10,     # tourism boom
        "cse": 1.1,          # portfolio inflows
    }
}


@st.cache_data
def load_reserve_data():
    """Load and merge reserve data from multiple sources."""
    
    # Load main reserve assets (CBSL historical)
    reserves_path = EXTERNAL_DIR / "reserve_assets_monthly_cbsl.csv"
    if reserves_path.exists():
        reserves = pd.read_csv(reserves_path, parse_dates=['date'])
    else:
        # Fallback to compiled
        reserves = pd.read_csv(EXTERNAL_DIR / "D12_reserves_compiled.csv", parse_dates=['date'])
    
    return reserves


@st.cache_data
def load_debt_data():
    """Load central government debt data."""
    debt_path = EXTERNAL_DIR / "central_govt_debt_quarterly.csv"
    if debt_path.exists():
        debt = pd.read_csv(debt_path, parse_dates=['date'])
        return debt
    return None


@st.cache_data  
def load_fx_data():
    """Load FX rates from monthly panel."""
    panel_path = MERGED_DIR / "slfsi_monthly_panel.csv"
    if panel_path.exists():
        panel = pd.read_csv(panel_path, parse_dates=['date'])
        fx = panel[['date', 'usd_lkr']].dropna()
        return fx
    return None


@st.cache_data
def load_iip_data():
    """Load IIP data (portfolio liabilities)."""
    iip_path = EXTERNAL_DIR / "iip_quarterly_clean.csv"
    if iip_path.exists():
        return pd.read_csv(iip_path, parse_dates=['date'])
    return None


@st.cache_data
def load_money_supply_data():
    """Load money supply (M2) data - NEW from table4.02."""
    # Try new comprehensive M2 data first
    m2_path = EXTERNAL_DIR / "monetary_aggregates_monthly.csv"
    if m2_path.exists():
        return pd.read_csv(m2_path, parse_dates=['date'])
    # Fallback to old parsed data
    m2_path = EXTERNAL_DIR / "money_supply_monthly_clean.csv"
    if m2_path.exists():
        return pd.read_csv(m2_path, parse_dates=['date'])
    return None


@st.cache_data
def load_external_debt_usd():
    """Load external debt in USD (from table2.12)."""
    path = EXTERNAL_DIR / "external_debt_usd_quarterly.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=['date'])
    return None


@st.cache_data
def load_tourism_data():
    """Load tourism earnings (USD millions)."""
    path = EXTERNAL_DIR / "tourism_earnings_monthly.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=['date'])
    return None


@st.cache_data
def load_remittances_data():
    """Load workers' remittances (USD millions)."""
    path = EXTERNAL_DIR / "remittances_monthly.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=['date'])
    return None


@st.cache_data
def load_cse_flows():
    """Load CSE inflows/outflows (USD millions)."""
    path = EXTERNAL_DIR / "cse_flows_monthly.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=['date'])
    return None


@st.cache_data
def load_reserve_money_velocity():
    """Load reserve money and multiplier data."""
    path = EXTERNAL_DIR / "reserve_money_velocity_monthly.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=['date'])
    return None


@st.cache_data
def load_monthly_imports():
    """Load monthly imports in USD (from table2.04)."""
    path = EXTERNAL_DIR / "monthly_imports_usd.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=['date'])
    return None


@st.cache_data
def load_monthly_exports():
    """Load monthly exports in USD (from table2.02)."""
    path = EXTERNAL_DIR / "monthly_exports_usd.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=['date'])
    return None


@st.cache_data
def load_iip_2025():
    """Load updated IIP data with portfolio liabilities (Q4 2012 - Q3 2025)."""
    path = EXTERNAL_DIR / "iip_quarterly_2025.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=['date'])
        # Calculate total portfolio liabilities
        if 'portfolio_equity' in df.columns and 'portfolio_debt' in df.columns:
            df['portfolio_liabilities'] = df['portfolio_equity'].fillna(0) + df['portfolio_debt'].fillna(0)
        return df
    return None


def calculate_imf_ara(reserves, m2, short_term_debt, portfolio_liab, fx, exports=None):
    """
    Calculate IMF Assessing Reserve Adequacy (ARA) metric.
    
    Formula: ARA = 5% × Exports + 5% × M2 + 30% × Short-term Debt + 15% × Portfolio Liabilities
    
    All components now available for full ARA calculation.
    """
    if reserves is None or m2 is None or short_term_debt is None:
        return None
    
    # Get M2 column name
    m2_col = 'broad_money_m2_lkr_m' if 'broad_money_m2_lkr_m' in m2.columns else 'broad_money_m2'
    if m2_col not in m2.columns:
        return None
    
    # Resample M2 to quarterly (end of quarter)
    m2_q = m2.set_index('date').resample('Q').last().reset_index()
    m2_q = m2_q[['date', m2_col]].rename(columns={m2_col: 'm2_lkr_m'})
    
    # Merge with FX to convert M2 to USD
    if fx is not None:
        fx_q = fx.set_index('date').resample('Q').last().reset_index()
        m2_q = m2_q.merge(fx_q[['date', 'usd_lkr']], on='date', how='left')
        m2_q['m2_usd_m'] = m2_q['m2_lkr_m'] / m2_q['usd_lkr']
    else:
        return None
    
    # Use short-term debt in USD directly if available
    if 'govt_short_term_usd_m' in short_term_debt.columns:
        debt_col = 'govt_short_term_usd_m'
    elif 'total_short_term_usd_m' in short_term_debt.columns:
        debt_col = 'total_short_term_usd_m'
    else:
        return None
    
    # Start with reserves
    result = reserves[['date', 'gross_reserves_usd_m']].copy()
    result = result.set_index('date').resample('Q').last().reset_index()
    
    # Merge M2
    result = result.merge(m2_q[['date', 'm2_usd_m']], on='date', how='left')
    
    # Merge short-term debt
    result = result.merge(short_term_debt[['date', debt_col]], on='date', how='left')
    result = result.rename(columns={debt_col: 'short_term_debt_usd_m'})
    
    # Merge portfolio liabilities if available
    # IIP dates use 1st of month (e.g. 2025-09-01) — normalize to end-of-quarter
    if portfolio_liab is not None and 'portfolio_liabilities' in portfolio_liab.columns:
        pl = portfolio_liab[['date', 'portfolio_liabilities']].copy()
        pl['date'] = pl['date'] + pd.offsets.QuarterEnd(0)
        result = result.merge(pl, on='date', how='left')
    else:
        result['portfolio_liabilities'] = np.nan

    # Forward-fill M2, debt, and portfolio for incomplete trailing quarters
    for col in ['m2_usd_m', 'short_term_debt_usd_m', 'portfolio_liabilities']:
        if col in result.columns:
            result[col] = result[col].ffill()

    # Merge exports if available (rolling 4-quarter sum for proper annualization)
    if exports is not None and 'exports_usd_m' in exports.columns:
        exports_q = exports.set_index('date').resample('Q').sum().reset_index()
        exports_q = exports_q[['date', 'exports_usd_m']].rename(columns={'exports_usd_m': 'quarterly_exports_usd_m'})
        exports_q['annual_exports_usd_m'] = exports_q['quarterly_exports_usd_m'].rolling(4, min_periods=1).sum()
        result = result.merge(exports_q[['date', 'annual_exports_usd_m']], on='date', how='left')
    else:
        result['annual_exports_usd_m'] = np.nan
    
    # Calculate ARA components
    result['ara_exports_component'] = 0.05 * result['annual_exports_usd_m'].fillna(0)
    result['ara_m2_component'] = 0.05 * result['m2_usd_m']
    result['ara_debt_component'] = 0.30 * result['short_term_debt_usd_m']
    result['ara_portfolio_component'] = 0.15 * result['portfolio_liabilities'].fillna(0)
    
    # Full ARA if exports available, partial otherwise
    has_exports = result['annual_exports_usd_m'].notna().any()
    
    result['ara_total'] = (
        result['ara_exports_component'].fillna(0) +
        result['ara_m2_component'].fillna(0) +
        result['ara_debt_component'].fillna(0) +
        result['ara_portfolio_component'].fillna(0)
    )
    
    # ARA ratio (reserves / ARA requirement)
    result['ara_ratio'] = result['gross_reserves_usd_m'] / result['ara_total']
    result['ara_adequate'] = result['ara_ratio'] >= 1.0
    result['is_full_ara'] = has_exports

    # Net reserves and net ARA ratio (excluding PBOC swap)
    result['net_reserves_usd_m'] = result['gross_reserves_usd_m'].copy()
    result.loc[result['date'] >= pd.Timestamp('2021-03-01'), 'net_reserves_usd_m'] -= PBOC_SWAP_USD_M
    result['net_ara_ratio'] = result['net_reserves_usd_m'] / result['ara_total']

    return result.dropna(subset=['gross_reserves_usd_m', 'ara_total'])


def calculate_greenspan_guidotti(reserves, debt, fx):
    """
    Calculate Greenspan-Guidotti ratio.
    
    Formula: Reserves (USD) / Short-term External Debt (USD)
    Threshold: >= 1.0 (reserves should cover short-term debt)
    
    Steps:
    1. Get quarterly short-term debt in LKR
    2. Get end-of-quarter FX rates (USD/LKR)
    3. Convert debt to USD
    4. Get reserves for same dates
    5. Calculate ratio
    """
    if debt is None or fx is None:
        return None
    
    # Filter debt to rows with short-term debt data
    debt_valid = debt[debt['total_short_term_lkr_m'].notna()].copy()
    
    # Merge with FX rates (find closest month for each quarter)
    # For quarterly dates, get the FX rate from that month
    debt_valid['month'] = debt_valid['date'].dt.to_period('M').dt.to_timestamp()
    fx['month'] = fx['date'].dt.to_period('M').dt.to_timestamp()
    
    merged = pd.merge(
        debt_valid,
        fx[['month', 'usd_lkr']],
        on='month',
        how='left'
    )
    
    # Convert short-term debt to USD
    # total_short_term_lkr_m is in LKR millions
    # usd_lkr is LKR per 1 USD
    # So: USD millions = LKR millions / usd_lkr
    merged['short_term_debt_usd_m'] = merged['total_short_term_lkr_m'] / merged['usd_lkr']
    
    # Merge with reserves
    reserves_monthly = reserves.copy()
    reserves_monthly['month'] = reserves_monthly['date'].dt.to_period('M').dt.to_timestamp()
    
    result = pd.merge(
        merged,
        reserves_monthly[['month', 'gross_reserves_usd_m']],
        on='month',
        how='left'
    )
    
    # Calculate Greenspan-Guidotti ratio
    result['gg_ratio'] = result['gross_reserves_usd_m'] / result['short_term_debt_usd_m']
    
    # Add threshold breach indicator
    result['gg_breach'] = result['gg_ratio'] < 1.0
    
    return result[['date', 'gross_reserves_usd_m', 'total_short_term_lkr_m', 
                   'usd_lkr', 'short_term_debt_usd_m', 'gg_ratio', 'gg_breach']].dropna()


def create_data_coverage_chart(data_sources):
    """Create a timeline showing data coverage gaps."""
    
    fig = go.Figure()
    
    colors = {
        'available': '#2ecc71',  # Green
        'gap': '#e74c3c',        # Red
        'partial': '#f39c12'     # Orange
    }
    
    y_positions = list(range(len(data_sources)))
    
    for i, (name, info) in enumerate(data_sources.items()):
        start = info['start']
        end = info['end']
        status = info.get('status', 'available')
        
        fig.add_trace(go.Scatter(
            x=[start, end],
            y=[i, i],
            mode='lines',
            line=dict(color=colors[status], width=20),
            name=name,
            hovertemplate=f"{name}<br>%{{x}}<extra></extra>"
        ))
        
        # Add label
        fig.add_annotation(
            x=start,
            y=i,
            text=name,
            xanchor='right',
            xshift=-10,
            showarrow=False,
            font=dict(size=12)
        )
    
    # Add crisis period marker
    fig.add_vrect(
        x0=CRISIS_START, x1=CRISIS_END,
        fillcolor="rgba(255,0,0,0.1)",
        layer="below",
        line_width=0,
    )
    
    # Add default date marker as a shape (more compatible)
    fig.add_shape(
        type="line",
        x0=DEFAULT_DATE, x1=DEFAULT_DATE,
        y0=0, y1=1,
        yref="paper",
        line=dict(color="red", width=2, dash="dash")
    )
    
    fig.update_layout(
        title="Data Coverage Timeline",
        xaxis_title="Date",
        yaxis=dict(
            tickvals=y_positions,
            ticktext=[""] * len(y_positions),  # Labels added via annotations
            showgrid=False
        ),
        height=400,
        showlegend=False,
        margin=dict(l=200)
    )
    
    return fig


def to_quarterly(df, value_col, how="last"):
    """Convert a dataframe with a date column to quarterly frequency."""
    if df is None or value_col not in df.columns:
        return None
    tmp = df[['date', value_col]].dropna().copy()
    if len(tmp) == 0:
        return None
    if how == "sum":
        tmp = tmp.set_index('date').resample('Q').sum().reset_index()
    else:
        tmp = tmp.set_index('date').resample('Q').last().reset_index()
    return tmp


def forecast_with_growth(series_q, value_col, horizon, lookback=8, scenario_multiplier=1.0,
                         start_date=None, max_qoq_growth=None):
    """
    Simple growth-based forecast using median QoQ growth over the lookback window.

    max_qoq_growth: if set, caps the adjusted growth rate (e.g. 0.05 = 5% per quarter).
                    Useful for preventing extrapolation of post-recovery spikes.
    """
    if series_q is None or value_col not in series_q.columns or horizon <= 0:
        return None
    s = series_q.dropna(subset=[value_col]).sort_values('date')
    if len(s) < 2:
        return None
    growth = s[value_col].pct_change().dropna()
    base_growth = growth.tail(lookback).median() if len(growth) > 0 else 0.0
    adj_growth = base_growth * scenario_multiplier
    if max_qoq_growth is not None:
        adj_growth = max(min(adj_growth, max_qoq_growth), -max_qoq_growth)
    last_date = start_date if start_date is not None else s['date'].iloc[-1]
    last_value = s[s['date'] <= last_date][value_col].iloc[-1]
    forecasts = []
    for step in range(1, horizon + 1):
        forecast_date = last_date + pd.offsets.QuarterEnd(step)
        last_value = last_value * (1 + adj_growth)
        forecasts.append({'date': forecast_date, value_col: last_value})
    return pd.DataFrame(forecasts)


def prepare_quarterly_imports(monthly_imports):
    """Aggregate monthly imports to quarterly sums."""
    if monthly_imports is None or 'imports_usd_m' not in monthly_imports.columns:
        return None
    imp = monthly_imports[['date', 'imports_usd_m']].dropna().copy()
    if len(imp) == 0:
        return None
    return imp.set_index('date').resample('Q').sum().reset_index()


def prepare_quarterly_bop_components(monthly_exports, monthly_imports, remittances, tourism, cse_flows):
    """
    Aggregate monthly BoP flow data to quarterly sums for the reserve accumulation model.

    Returns a single quarterly DataFrame with columns:
      date, exports_usd_m, imports_usd_m, remittances_usd_m,
      tourism_earnings_usd_m, cse_net_usd_m
    """
    source_map = {
        'exports_usd_m': monthly_exports,
        'imports_usd_m': monthly_imports,
        'remittances_usd_m': remittances,
        'tourism_earnings_usd_m': tourism,
        'cse_net_usd_m': cse_flows,
    }
    quarterly_frames = []
    for col, src in source_map.items():
        if src is not None and col in src.columns:
            tmp = src[['date', col]].dropna().copy()
            if len(tmp) == 0:
                continue
            q = tmp.set_index('date').resample('Q').sum().reset_index()
            quarterly_frames.append(q)

    if not quarterly_frames:
        return None

    result = quarterly_frames[0]
    for qf in quarterly_frames[1:]:
        result = result.merge(qf, on='date', how='outer')

    return result.sort_values('date').reset_index(drop=True)


def calibrate_bop_residual(reserves_q, bop_q, lookback=8):
    """
    Compute the BoP residual (FDI + debt disbursements − debt service + other)
    by comparing actual reserve changes to observable BoP flows.

    Returns the median quarterly residual over the lookback window.
    """
    if reserves_q is None or bop_q is None:
        return 0.0

    merged = reserves_q.merge(bop_q, on='date', how='inner').sort_values('date')
    if len(merged) < 2:
        return 0.0

    merged['delta_reserves'] = merged['gross_reserves_usd_m'].diff()

    observable = pd.Series(0.0, index=merged.index)
    for col, sign in [
        ('exports_usd_m', 1),
        ('remittances_usd_m', 1),
        ('tourism_earnings_usd_m', 1),
        ('imports_usd_m', -1),
        ('cse_net_usd_m', 1),
    ]:
        if col in merged.columns:
            observable = observable + sign * merged[col].fillna(0)

    merged['residual'] = merged['delta_reserves'] - observable

    recent = merged.dropna(subset=['residual']).tail(lookback)
    if len(recent) == 0:
        return 0.0

    # TODO(human): Enhance residual calibration — consider decomposing into
    # known debt service (from external_debt_usd_quarterly short-term debt changes)
    # and a stochastic remainder, or weighting recent quarters more heavily.
    return recent['residual'].median()


def forecast_reserves_bop(reserves_q, bop_q, horizon, lookback=8,
                          scenario_shocks=None, start_date=None):
    """
    Project reserve levels using a structural BoP flow accumulation model.

    Each quarter:
      delta = exports + remittances + tourism − imports + cse_net + residual
      reserves[t] = reserves[t−1] + delta

    Returns (forecast_df, component_df):
      - forecast_df: date + gross_reserves_usd_m
      - component_df: full BoP decomposition for charting
    """
    if reserves_q is None or bop_q is None:
        return None, None

    shocks = scenario_shocks or {}

    if start_date is None:
        start_date = reserves_q.dropna(subset=['gross_reserves_usd_m'])['date'].max()

    last_reserves = reserves_q[reserves_q['date'] <= start_date]['gross_reserves_usd_m'].iloc[-1]

    # Project each BoP component independently
    component_specs = [
        ('exports_usd_m', 'exports'),
        ('imports_usd_m', 'imports'),
        ('remittances_usd_m', 'remittances'),
        ('tourism_earnings_usd_m', 'tourism'),
        ('cse_net_usd_m', 'cse'),
    ]

    future_dates = [start_date + pd.offsets.QuarterEnd(i) for i in range(1, horizon + 1)]
    result = pd.DataFrame({'date': future_dates})

    # Cap at 5% QoQ (~22% annualized) to prevent post-recovery spike extrapolation
    BOP_MAX_QOQ = 0.05

    for col, shock_key in component_specs:
        src = bop_q[['date', col]].dropna() if col in bop_q.columns else None
        fcast = forecast_with_growth(
            src, col, horizon, lookback=lookback,
            scenario_multiplier=shocks.get(shock_key, 1.0),
            max_qoq_growth=BOP_MAX_QOQ,
            start_date=start_date,
        )
        if fcast is not None and col in fcast.columns:
            result = result.merge(fcast[['date', col]], on='date', how='left')
        else:
            result[col] = 0.0

    # Calibrate residual from historical data
    residual_q = calibrate_bop_residual(reserves_q, bop_q, lookback=lookback)
    result['residual_usd_m'] = residual_q

    # BoP identity: delta = inflows − outflows + residual
    result['delta_reserves'] = (
        result['exports_usd_m'].fillna(0)
        + result['remittances_usd_m'].fillna(0)
        + result['tourism_earnings_usd_m'].fillna(0)
        - result['imports_usd_m'].fillna(0)
        + result['cse_net_usd_m'].fillna(0)
        + result['residual_usd_m']
    )

    # Accumulate reserve levels
    levels = []
    current = last_reserves
    for delta in result['delta_reserves']:
        current = current + delta
        levels.append(current)
    result['gross_reserves_usd_m'] = levels

    forecast_df = result[['date', 'gross_reserves_usd_m']].copy()
    component_df = result.copy()

    return forecast_df, component_df


def enrich_with_import_cover(df, imports_q):
    """Add import cover and net import cover metrics using quarterly imports."""
    df = df.copy()
    if 'imports_usd_m' in df.columns:
        # Imports already present (e.g. from forecast merge); use directly
        monthly_imports = (df['imports_usd_m'] / 3).replace({0: np.nan})
    elif imports_q is not None:
        merged = df.merge(imports_q[['date', 'imports_usd_m']], on='date', how='left')
        monthly_imports = (merged['imports_usd_m'] / 3).replace({0: np.nan})
    else:
        monthly_imports = pd.Series([np.nan] * len(df))
    df['import_cover'] = df['gross_reserves_usd_m'] / monthly_imports
    df['net_reserves'] = df['gross_reserves_usd_m'].copy()
    df.loc[df['date'] >= pd.Timestamp('2021-03-01'), 'net_reserves'] -= PBOC_SWAP_USD_M
    df['net_import_cover'] = df['net_reserves'] / monthly_imports
    return df


def build_forecast_scenarios(ara_history, reserves, monthly_imports, horizon=8, lookback=8,
                             monthly_exports=None, remittances=None, tourism=None, cse_flows=None):
    """
    Build forecast panels for each scenario.

    Reserve levels are projected using a structural BoP flow accumulation model
    when sufficient BoP data is available, falling back to simple growth-based
    extrapolation otherwise.

    Returns (history_panel, scenario_panels_dict, bop_decompositions_dict).
    """
    if ara_history is None or len(ara_history) == 0 or horizon <= 0:
        return None, {}, {}

    reserves_q = to_quarterly(reserves, 'gross_reserves_usd_m')
    imports_q = prepare_quarterly_imports(monthly_imports)
    last_date = ara_history['date'].max()
    future_dates = [last_date + pd.offsets.QuarterEnd(i) for i in range(1, horizon + 1)]

    # Prepare quarterly BoP components for structural reserve projection
    bop_q = prepare_quarterly_bop_components(
        monthly_exports, monthly_imports, remittances, tourism, cse_flows
    )
    use_bop = bop_q is not None and len(bop_q) >= 4

    # ── History panel ────────────────────────────────────────────────
    history_panel = ara_history.copy()
    history_panel['required_100'] = history_panel['ara_total']
    history_panel['required_150'] = history_panel['ara_total'] * 1.5
    history_panel['gap_to_100'] = history_panel['gross_reserves_usd_m'] - history_panel['required_100']
    history_panel['gap_to_150'] = history_panel['gross_reserves_usd_m'] - history_panel['required_150']
    if 'short_term_debt_usd_m' in history_panel.columns:
        history_panel['gg_ratio'] = history_panel['gross_reserves_usd_m'] / history_panel['short_term_debt_usd_m']

    # Net ARA columns (from calculate_imf_ara)
    if 'net_reserves_usd_m' in history_panel.columns:
        history_panel['net_ara_ratio'] = history_panel['net_reserves_usd_m'] / history_panel['ara_total']
        history_panel['net_gap_to_100'] = history_panel['net_reserves_usd_m'] - history_panel['required_100']
        history_panel['net_gap_to_150'] = history_panel['net_reserves_usd_m'] - history_panel['required_150']

    history_panel['period'] = "History"
    history_panel['scenario'] = "History"
    history_panel = enrich_with_import_cover(history_panel, imports_q)

    # ── Scenario panels ─────────────────────────────────────────────
    scenario_panels = {}
    bop_decompositions = {}

    for scenario_name, shocks in SCENARIO_SHOCKS.items():
        future = pd.DataFrame({'date': future_dates})

        # ARA requirement components (always growth-based)
        future_exp = forecast_with_growth(
            ara_history[['date', 'annual_exports_usd_m']] if 'annual_exports_usd_m' in ara_history.columns else None,
            'annual_exports_usd_m', horizon, lookback=lookback,
            scenario_multiplier=shocks.get('exports', 1.0), start_date=last_date,
        )
        future_m2 = forecast_with_growth(
            ara_history[['date', 'm2_usd_m']] if 'm2_usd_m' in ara_history.columns else None,
            'm2_usd_m', horizon, lookback=lookback,
            scenario_multiplier=shocks.get('m2', 1.0), start_date=last_date,
        )
        future_debt = forecast_with_growth(
            ara_history[['date', 'short_term_debt_usd_m']] if 'short_term_debt_usd_m' in ara_history.columns else None,
            'short_term_debt_usd_m', horizon, lookback=lookback,
            scenario_multiplier=shocks.get('debt', 1.0), start_date=last_date,
        )
        future_portfolio = forecast_with_growth(
            ara_history[['date', 'portfolio_liabilities']] if 'portfolio_liabilities' in ara_history.columns else None,
            'portfolio_liabilities', horizon, lookback=lookback,
            scenario_multiplier=shocks.get('portfolio', 1.0), start_date=last_date,
        )

        # Reserve level projection: BoP model or growth fallback
        if use_bop:
            future_reserves, bop_components = forecast_reserves_bop(
                reserves_q, bop_q, horizon, lookback=lookback,
                scenario_shocks=shocks, start_date=last_date,
            )
            if bop_components is not None:
                bop_decompositions[scenario_name] = bop_components
        else:
            future_reserves = forecast_with_growth(
                reserves_q, 'gross_reserves_usd_m', horizon,
                lookback=lookback, scenario_multiplier=shocks.get('reserves', 1.0),
                start_date=last_date,
            )

        # Imports projection (for import cover)
        future_imports = forecast_with_growth(
            imports_q, 'imports_usd_m', horizon, lookback=lookback,
            scenario_multiplier=shocks.get('imports', 1.0), start_date=last_date,
        )

        # Merge ARA requirement components + reserves + imports into future df
        for col, fcast in [
            ('annual_exports_usd_m', future_exp),
            ('m2_usd_m', future_m2),
            ('short_term_debt_usd_m', future_debt),
            ('portfolio_liabilities', future_portfolio),
            ('gross_reserves_usd_m', future_reserves),
            ('imports_usd_m', future_imports),
        ]:
            if fcast is not None and col in fcast.columns:
                future = future.merge(fcast[['date', col]], on='date', how='left')
            else:
                future[col] = np.nan

        # ARA components and ratios
        future['ara_exports_component'] = 0.05 * future['annual_exports_usd_m']
        future['ara_m2_component'] = 0.05 * future['m2_usd_m']
        future['ara_debt_component'] = 0.30 * future['short_term_debt_usd_m']
        future['ara_portfolio_component'] = 0.15 * future['portfolio_liabilities']
        future['ara_total'] = (
            future['ara_exports_component'].fillna(0)
            + future['ara_m2_component'].fillna(0)
            + future['ara_debt_component'].fillna(0)
            + future['ara_portfolio_component'].fillna(0)
        )
        future['ara_ratio'] = future['gross_reserves_usd_m'] / future['ara_total']
        future['required_100'] = future['ara_total']
        future['required_150'] = future['ara_total'] * 1.5
        future['gap_to_100'] = future['gross_reserves_usd_m'] - future['required_100']
        future['gap_to_150'] = future['gross_reserves_usd_m'] - future['required_150']

        # Net ARA (all forecast dates are post-2021, so always subtract swap)
        future['net_reserves_usd_m'] = future['gross_reserves_usd_m'] - PBOC_SWAP_USD_M
        future['net_ara_ratio'] = future['net_reserves_usd_m'] / future['ara_total']
        future['net_gap_to_100'] = future['net_reserves_usd_m'] - future['required_100']
        future['net_gap_to_150'] = future['net_reserves_usd_m'] - future['required_150']

        future['period'] = "Forecast"
        future['scenario'] = scenario_name
        if 'short_term_debt_usd_m' in future.columns:
            future['gg_ratio'] = future['gross_reserves_usd_m'] / future['short_term_debt_usd_m']

        # Import cover and net reserves (for import cover metric)
        future = enrich_with_import_cover(future, future_imports if future_imports is not None else imports_q)

        scenario_panels[scenario_name] = future

    return history_panel, scenario_panels, bop_decompositions


def main():
    st.title("📊 Reserve Adequacy Dashboard")
    st.markdown("*Analyzing Sri Lanka's reserve adequacy using multiple benchmarks*")
    
    # Sidebar
    st.sidebar.header("Data Sources")
    
    # Load all data
    with st.spinner("Loading data..."):
        reserves = load_reserve_data()
        debt = load_debt_data()
        fx = load_fx_data()
        iip = load_iip_data()
        m2 = load_money_supply_data()
        # New data sources
        ext_debt_usd = load_external_debt_usd()
        tourism = load_tourism_data()
        remittances = load_remittances_data()
        # Latest additions
        monthly_imports = load_monthly_imports()
        monthly_exports = load_monthly_exports()
        iip_2025 = load_iip_2025()
        cse_flows = load_cse_flows()
        reserve_money = load_reserve_money_velocity()
    
    reserves_base = reserves.copy() if reserves is not None else None
    ara_data = calculate_imf_ara(reserves_base, m2, ext_debt_usd, iip_2025, fx, monthly_exports)
    
    # Show data loading status in sidebar
    st.sidebar.subheader("Core Data")
    if reserves is not None:
        st.sidebar.success(f"✓ Reserves: {len(reserves)} records")
    else:
        st.sidebar.error("✗ Reserves")
    
    if ext_debt_usd is not None:
        st.sidebar.success(f"✓ Ext Debt USD: {len(ext_debt_usd)} records")
    else:
        st.sidebar.warning("⚠ Ext Debt USD")
    
    if fx is not None:
        st.sidebar.success(f"✓ FX Rates: {len(fx)} records")
    else:
        st.sidebar.error("✗ FX Rates")
    
    st.sidebar.subheader("Monetary")
    if m2 is not None:
        st.sidebar.success(f"✓ M2: {len(m2)} records")
    else:
        st.sidebar.warning("⚠ M2")
    
    if reserve_money is not None:
        st.sidebar.success(f"✓ Reserve Money: {len(reserve_money)} records")
    else:
        st.sidebar.warning("⚠ Reserve Money")
    
    st.sidebar.subheader("External Flows")
    if tourism is not None:
        st.sidebar.success(f"✓ Tourism: {len(tourism)} records")
    else:
        st.sidebar.warning("⚠ Tourism")
    
    if remittances is not None:
        st.sidebar.success(f"✓ Remittances: {len(remittances)} records")
    else:
        st.sidebar.warning("⚠ Remittances")
    
    if cse_flows is not None:
        st.sidebar.success(f"✓ CSE Flows: {len(cse_flows)} records")
    else:
        st.sidebar.warning("⚠ CSE Flows")
    
    st.sidebar.subheader("Benchmarking")
    if monthly_imports is not None:
        st.sidebar.success(f"✓ Imports: {len(monthly_imports)} records")
    else:
        st.sidebar.warning("⚠ Imports")
    
    if monthly_exports is not None:
        st.sidebar.success(f"✓ Exports: {len(monthly_exports)} records")
    else:
        st.sidebar.warning("⚠ Exports")
    
    if iip_2025 is not None:
        st.sidebar.success(f"✓ IIP 2025: {len(iip_2025)} records")
    else:
        st.sidebar.warning("⚠ IIP 2025")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📈 Greenspan-Guidotti",
        "📊 Reserve Metrics",
        "🎯 IMF ARA",
        "🧭 Forecasts",
        "💰 External Flows",
        "🏦 Monetary",
        "🔍 Data Coverage",
        "📋 Data Tables"
    ])
    
    # ===================
    # TAB 1: Greenspan-Guidotti
    # ===================
    with tab1:
        st.header("Greenspan-Guidotti Ratio")
        
        st.markdown("""
        **Formula**: `Reserves (USD) / Short-term External Debt (USD)`
        
        **Threshold**: Ratio ≥ 1.0 means reserves can cover all short-term debt
        
        **Interpretation**: 
        - Below 1.0 = Vulnerable to sudden stop / rollover crisis
        - Above 1.0 = Adequate to meet short-term obligations
        """)
        
        # Calculate GG ratio
        gg_data = calculate_greenspan_guidotti(reserves_base, debt, fx)
        
        if gg_data is not None and len(gg_data) > 0:
            # Show metrics
            col1, col2, col3, col4 = st.columns(4)
            
            latest = gg_data.iloc[-1]
            min_gg = gg_data['gg_ratio'].min()
            min_date = gg_data.loc[gg_data['gg_ratio'].idxmin(), 'date']
            breach_count = gg_data['gg_breach'].sum()
            
            col1.metric("Latest GG Ratio", f"{latest['gg_ratio']:.2f}", 
                       delta="Adequate" if latest['gg_ratio'] >= 1 else "BREACH")
            col2.metric("Minimum GG Ratio", f"{min_gg:.2f}", 
                       delta=f"{min_date.strftime('%b %Y')}")
            col3.metric("Breach Quarters", f"{breach_count}", 
                       delta=f"of {len(gg_data)} total")
            col4.metric("Latest Reserves", f"${latest['gross_reserves_usd_m']:,.0f}M")
            
            # GG Ratio chart
            fig = go.Figure()
            
            # Add ratio line
            fig.add_trace(go.Scatter(
                x=gg_data['date'],
                y=gg_data['gg_ratio'],
                mode='lines+markers',
                name='GG Ratio',
                line=dict(color='#3498db', width=2),
                hovertemplate="Date: %{x}<br>GG Ratio: %{y:.2f}<extra></extra>"
            ))
            
            # Add threshold line
            fig.add_hline(y=1.0, line_dash="dash", line_color="red",
                         annotation_text="Threshold (1.0)", annotation_position="right")
            
            # Color breach periods
            breach_periods = gg_data[gg_data['gg_breach']]
            if len(breach_periods) > 0:
                fig.add_trace(go.Scatter(
                    x=breach_periods['date'],
                    y=breach_periods['gg_ratio'],
                    mode='markers',
                    name='Breach',
                    marker=dict(color='red', size=10, symbol='x'),
                    hovertemplate="BREACH<br>Date: %{x}<br>GG Ratio: %{y:.2f}<extra></extra>"
                ))
            
            # Add default marker
            fig.add_shape(
                type="line", x0=DEFAULT_DATE, x1=DEFAULT_DATE,
                y0=0, y1=1, yref="paper",
                line=dict(color="gray", width=2, dash="dash")
            )
            fig.add_annotation(x=DEFAULT_DATE, y=1, yref="paper", text="Default",
                              showarrow=False, yshift=10)
            
            fig.update_layout(
                title="Greenspan-Guidotti Ratio Over Time",
                xaxis_title="Date",
                yaxis_title="GG Ratio (Reserves / ST Debt)",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, width="stretch")
            
            # Component breakdown
            st.subheader("Component Breakdown")
            
            fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                subplot_titles=("Reserves (USD M)", "Short-term Debt (USD M)"),
                                vertical_spacing=0.1)
            
            fig2.add_trace(go.Scatter(
                x=gg_data['date'], y=gg_data['gross_reserves_usd_m'],
                mode='lines', name='Reserves', fill='tozeroy',
                line=dict(color='#2ecc71')
            ), row=1, col=1)
            
            fig2.add_trace(go.Scatter(
                x=gg_data['date'], y=gg_data['short_term_debt_usd_m'],
                mode='lines', name='ST Debt', fill='tozeroy',
                line=dict(color='#e74c3c')
            ), row=2, col=1)
            
            fig2.update_layout(height=600, showlegend=True)
            st.plotly_chart(fig2, width="stretch")
            
        else:
            st.warning("Unable to calculate Greenspan-Guidotti ratio. Check data availability.")
    
    # ===================
    # TAB 2: Reserve Metrics
    # ===================
    with tab2:
        st.header("Reserve Metrics Overview")
        
        if reserves is not None:
            # Import cover calculation - use actual imports if available
            if monthly_imports is not None and len(monthly_imports) > 0:
                # Merge reserves with imports
                reserves_ic = reserves.merge(
                    monthly_imports[['date', 'imports_usd_m']], 
                    on='date', 
                    how='left'
                )
                # Forward fill missing imports
                reserves_ic['imports_usd_m'] = reserves_ic['imports_usd_m'].ffill().bfill()
                reserves_ic['import_cover'] = reserves_ic['gross_reserves_usd_m'] / reserves_ic['imports_usd_m']
                st.info("Using actual monthly import data for import cover calculation")
            else:
                # Fallback to estimate
                MONTHLY_IMPORTS = 1500  # USD millions estimate
                reserves_ic = reserves.copy()
                reserves_ic['imports_usd_m'] = MONTHLY_IMPORTS
                reserves_ic['import_cover'] = reserves_ic['gross_reserves_usd_m'] / MONTHLY_IMPORTS
                st.warning("Using estimated monthly imports ($1.5B)")
            
            # Net usable reserves (post PBOC swap in 2021)
            reserves_ic['net_reserves'] = reserves_ic['gross_reserves_usd_m'].copy()
            reserves_ic.loc[reserves_ic['date'] >= '2021-03-01', 'net_reserves'] -= PBOC_SWAP_USD_M
            reserves_ic['net_import_cover'] = reserves_ic['net_reserves'] / reserves_ic['imports_usd_m']
            
            # Use reserves_ic for display
            reserves = reserves_ic
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            
            latest = reserves.iloc[-1]
            min_res = reserves['gross_reserves_usd_m'].min()
            min_date = reserves.loc[reserves['gross_reserves_usd_m'].idxmin(), 'date']
            
            col1.metric("Latest Reserves", f"${latest['gross_reserves_usd_m']:,.0f}M")
            col2.metric("Import Cover", f"{latest['import_cover']:.1f} months")
            col3.metric("Minimum Reserves", f"${min_res:,.0f}M", 
                       delta=f"{min_date.strftime('%b %Y')}")
            
            # Reserve components chart
            if 'fx_reserves_usd_m' in reserves.columns:
                st.subheader("Reserve Composition")
                
                fig = go.Figure()
                
                components = ['fx_reserves_usd_m', 'gold_usd_m', 'sdrs_usd_m', 'imf_position_usd_m']
                colors = ['#3498db', '#f1c40f', '#9b59b6', '#1abc9c']
                names = ['FX Reserves', 'Gold', 'SDRs', 'IMF Position']
                
                for comp, color, name in zip(components, colors, names):
                    if comp in reserves.columns:
                        fig.add_trace(go.Scatter(
                            x=reserves['date'],
                            y=reserves[comp],
                            mode='lines',
                            name=name,
                            stackgroup='one',
                            line=dict(color=color)
                        ))
                
                fig.update_layout(
                    title="Reserve Composition Over Time",
                    xaxis_title="Date",
                    yaxis_title="USD Millions",
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, width="stretch")
            
            # Import cover chart
            st.subheader("Import Cover")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=reserves['date'],
                y=reserves['import_cover'],
                mode='lines',
                name='Gross Import Cover',
                line=dict(color='#3498db', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=reserves['date'],
                y=reserves['net_import_cover'],
                mode='lines',
                name='Net Import Cover (excl. PBOC)',
                line=dict(color='#e74c3c', width=2, dash='dot')
            ))
            
            # Add threshold lines
            fig.add_hline(y=3, line_dash="dash", line_color="orange",
                         annotation_text="3 months (IMF minimum)", annotation_position="right")
            fig.add_hline(y=6, line_dash="dash", line_color="green",
                         annotation_text="6 months (comfortable)", annotation_position="right")
            
            fig.add_shape(
                type="line", x0=DEFAULT_DATE, x1=DEFAULT_DATE,
                y0=0, y1=1, yref="paper",
                line=dict(color="gray", width=2, dash="dash")
            )
            fig.add_annotation(x=DEFAULT_DATE, y=1, yref="paper", text="Default",
                              showarrow=False, yshift=10)
            
            fig.update_layout(
                title="Import Cover Months",
                xaxis_title="Date",
                yaxis_title="Months of Imports",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, width="stretch")
    
    # ===================
    # TAB 3: IMF ARA Metric
    # ===================
    with tab3:
        st.header("IMF ARA Metric")
        st.markdown("""
        **Assessing Reserve Adequacy (ARA)** is the IMF's framework for evaluating reserve adequacy.
        
        **Full Formula**: `ARA = 5% × Exports + 5% × M2 + 30% × Short-term Debt + 15% × Portfolio Liabilities`
        
        **Threshold**: Reserves should be 100-150% of ARA for adequate coverage.
        """)
        
        if ara_data is not None and len(ara_data) > 0:
            # Show whether full or partial ARA
            if ara_data['is_full_ara'].iloc[0]:
                st.success("✅ **Full IMF ARA** - All 4 components available (Exports, M2, Short-term Debt, Portfolio Liabilities)")
            else:
                st.warning("⚠️ **Partial IMF ARA** - Missing exports data")
            # ARA Ratio Chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=ara_data['date'],
                y=ara_data['ara_ratio'] * 100,
                mode='lines+markers',
                name='Gross ARA Ratio',
                line=dict(color='#3498db', width=2),
                marker=dict(size=6)
            ))

            # Net ARA Ratio (excl. PBOC swap)
            if 'net_ara_ratio' in ara_data.columns:
                fig.add_trace(go.Scatter(
                    x=ara_data['date'],
                    y=ara_data['net_ara_ratio'] * 100,
                    mode='lines+markers',
                    name='Net ARA Ratio (excl. PBOC swap)',
                    line=dict(color='#e74c3c', width=2, dash='dot'),
                    marker=dict(size=4)
                ))

            # Threshold lines
            fig.add_hline(y=100, line_dash="dash", line_color="red",
                         annotation_text="100% (Minimum)", annotation_position="right")
            fig.add_hline(y=150, line_dash="dash", line_color="green",
                         annotation_text="150% (Comfortable)", annotation_position="right")

            # Crisis period
            fig.add_vrect(x0=CRISIS_START, x1=CRISIS_END,
                         fillcolor="rgba(255,0,0,0.1)", layer="below", line_width=0)

            fig.add_shape(type="line", x0=DEFAULT_DATE, x1=DEFAULT_DATE,
                         y0=0, y1=1, yref="paper",
                         line=dict(color="gray", width=2, dash="dash"))

            fig.update_layout(
                title="IMF ARA Ratio (Gross vs Net of PBOC Swap)",
                xaxis_title="Date",
                yaxis_title="ARA Ratio (%)",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, width="stretch")
            
            # Component breakdown
            st.subheader("ARA Components Breakdown")
            
            fig2 = go.Figure()
            
            # Add exports component if available
            if 'ara_exports_component' in ara_data.columns:
                fig2.add_trace(go.Bar(
                    x=ara_data['date'],
                    y=ara_data['ara_exports_component'],
                    name='5% × Exports',
                    marker_color='#f39c12'
                ))
            
            fig2.add_trace(go.Bar(
                x=ara_data['date'],
                y=ara_data['ara_m2_component'],
                name='5% × M2',
                marker_color='#9b59b6'
            ))
            
            fig2.add_trace(go.Bar(
                x=ara_data['date'],
                y=ara_data['ara_debt_component'],
                name='30% × Short-term Debt',
                marker_color='#e74c3c'
            ))
            
            fig2.add_trace(go.Bar(
                x=ara_data['date'],
                y=ara_data['ara_portfolio_component'],
                name='15% × Portfolio Liabilities',
                marker_color='#3498db'
            ))
            
            fig2.add_trace(go.Scatter(
                x=ara_data['date'],
                y=ara_data['gross_reserves_usd_m'],
                mode='lines',
                name='Actual Reserves',
                line=dict(color='#2ecc71', width=3)
            ))
            
            fig2.update_layout(
                title="ARA Components vs Actual Reserves (Full IMF ARA)",
                xaxis_title="Date",
                yaxis_title="USD Millions",
                barmode='stack',
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig2, width="stretch")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            latest = ara_data.iloc[-1]
            min_ara = ara_data['ara_ratio'].min()
            min_date = ara_data.loc[ara_data['ara_ratio'].idxmin(), 'date']

            col1.metric("Gross ARA Ratio", f"{latest['ara_ratio']*100:.0f}%",
                       delta="Adequate" if latest['ara_ratio'] >= 1.0 else "Below threshold")
            col2.metric("Net ARA Ratio (excl. PBOC)", f"{latest['net_ara_ratio']*100:.0f}%",
                       delta="Adequate" if latest['net_ara_ratio'] >= 1.0 else "Below threshold")
            col3.metric("Minimum ARA Ratio", f"{min_ara*100:.0f}%",
                       delta=f"{min_date.strftime('%b %Y')}")
            col4.metric("ARA Requirement", f"${latest['ara_total']:,.0f}M")
        else:
            st.warning("Unable to calculate IMF ARA. Check that M2, short-term debt (USD), and reserves data are available.")
    
    # ===================
    # TAB 4: Forecasts
    # ===================
    with tab4:
        st.header("Forecasts & Scenarios")
        st.markdown("""
        Project the required reserve path using IMF ARA components and stress it under three scenarios:
        **Baseline**, **Downside**, and **Upside**. Forecasts use median quarterly growth over a recent
        lookback window and apply scenario multipliers to remain transparent and reproducible.
        """)
        
        if ara_data is None or len(ara_data) == 0:
            st.warning("IMF ARA inputs are missing. Please ensure reserves, M2, short-term debt (USD), FX, and portfolio liabilities are loaded.")
        else:
            col_a, col_b = st.columns(2)
            horizon = col_a.slider("Forecast horizon (quarters)", min_value=4, max_value=12, value=8, step=1)
            lookback = col_b.slider("Lookback for median QoQ growth", min_value=4, max_value=12, value=8, step=1)
            
            history_panel, scenario_panels, bop_decompositions = build_forecast_scenarios(
                ara_history=ara_data,
                reserves=reserves_base,
                monthly_imports=monthly_imports,
                horizon=horizon,
                lookback=lookback,
                monthly_exports=monthly_exports,
                remittances=remittances,
                tourism=tourism,
                cse_flows=cse_flows,
            )
            
            if not scenario_panels:
                st.warning("Unable to build forecasts. Check that ARA components and reserves cover at least two quarters.")
            else:
                if bop_decompositions:
                    st.success("Reserve projection: **Structural BoP Flow Accumulation Model** "
                               "(exports + remittances + tourism − imports + CSE flows + calibrated residual)")
                else:
                    st.info("Reserve projection: **Median QoQ growth extrapolation** (BoP data insufficient for structural model)")

                scenario_choice = st.selectbox("Select scenario to visualize", list(scenario_panels.keys()))
                selected = scenario_panels[scenario_choice]
                combined = pd.concat([history_panel, selected]).sort_values('date')
                hist = combined[combined['period'] == "History"]
                fut = combined[combined['period'] == "Forecast"]
                
                # Cross-scenario summary at forecast horizon
                summary_rows = []
                for name, df_s in scenario_panels.items():
                    df_s_valid = df_s.dropna(subset=['ara_ratio'])
                    if len(df_s_valid) == 0:
                        continue
                    tail = df_s_valid.iloc[-1]
                    row = {
                        'Scenario': name,
                        'End Date': tail['date'].strftime('%Y-Q%q'),
                        'Gross ARA (%)': round(tail['ara_ratio'] * 100, 1) if not np.isnan(tail['ara_ratio']) else np.nan,
                        'Net ARA (%)': round(tail.get('net_ara_ratio', np.nan) * 100, 1) if 'net_ara_ratio' in tail and not np.isnan(tail.get('net_ara_ratio', np.nan)) else np.nan,
                        'Gap vs 100% (USD m)': round(tail['gap_to_100'], 1),
                        'Net Gap vs 100%': round(tail['net_gap_to_100'], 1) if 'net_gap_to_100' in tail else np.nan,
                        'Import Cover (mo)': round(tail['import_cover'], 2) if 'import_cover' in tail else np.nan,
                        'GG Ratio': round(tail.get('gg_ratio', np.nan), 2) if 'gg_ratio' in tail else np.nan,
                    }
                    summary_rows.append(row)
                if summary_rows:
                    st.subheader("Scenario Summary at Horizon End")
                    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, width="stretch")
                
                # ARA ratio chart
                st.subheader("ARA Ratio: History vs Forecast")
                fig_ratio = go.Figure()
                fig_ratio.add_trace(go.Scatter(
                    x=hist['date'], y=hist['ara_ratio'] * 100,
                    mode='lines', name='History', line=dict(color='#3498db', width=2)
                ))
                fig_ratio.add_trace(go.Scatter(
                    x=fut['date'], y=fut['ara_ratio'] * 100,
                    mode='lines+markers', name=f'Gross Forecast - {scenario_choice}',
                    line=dict(color='#e67e22', width=2, dash='dash'),
                    marker=dict(size=6)
                ))
                # Net ARA ratio lines
                if 'net_ara_ratio' in hist.columns:
                    fig_ratio.add_trace(go.Scatter(
                        x=hist['date'], y=hist['net_ara_ratio'] * 100,
                        mode='lines', name='Net ARA (History, excl. PBOC)',
                        line=dict(color='#e74c3c', width=1.5, dash='dot')
                    ))
                if 'net_ara_ratio' in fut.columns:
                    fig_ratio.add_trace(go.Scatter(
                        x=fut['date'], y=fut['net_ara_ratio'] * 100,
                        mode='lines+markers', name=f'Net ARA Forecast - {scenario_choice}',
                        line=dict(color='#c0392b', width=1.5, dash='dashdot'),
                        marker=dict(size=4)
                    ))
                fig_ratio.add_hline(y=100, line_dash="dash", line_color="red",
                                    annotation_text="100% minimum", annotation_position="right")
                fig_ratio.add_hline(y=150, line_dash="dash", line_color="green",
                                    annotation_text="150% comfortable", annotation_position="right")
                fig_ratio.update_layout(
                    xaxis_title="Date",
                    yaxis_title="ARA Ratio (%)",
                    height=450,
                    hovermode='x unified'
                )
                st.plotly_chart(fig_ratio, width="stretch")
                
                # Reserves vs required band
                st.subheader("Reserves vs Required (100-150% of ARA)")
                fig_band = go.Figure()
                fig_band.add_trace(go.Scatter(
                    x=hist['date'], y=hist['gross_reserves_usd_m'],
                    mode='lines', name='Actual Reserves (History)',
                    line=dict(color='#2ecc71', width=2)
                ))
                fig_band.add_trace(go.Scatter(
                    x=fut['date'], y=fut['gross_reserves_usd_m'],
                    mode='lines+markers', name=f'Reserves Forecast - {scenario_choice}',
                    line=dict(color='#16a085', width=2, dash='dash'),
                    marker=dict(size=6)
                ))
                fig_band.add_trace(go.Scatter(
                    x=combined['date'], y=combined['required_100'],
                    mode='lines', name='Required (100% ARA)',
                    line=dict(color='#e74c3c', width=2, dash='dot')
                ))
                fig_band.add_trace(go.Scatter(
                    x=combined['date'], y=combined['required_150'],
                    mode='lines', name='Required (150% ARA)',
                    line=dict(color='#9b59b6', width=2, dash='dot')
                ))
                # Net reserves lines (excl. PBOC swap)
                if 'net_reserves_usd_m' in hist.columns:
                    fig_band.add_trace(go.Scatter(
                        x=hist['date'], y=hist['net_reserves_usd_m'],
                        mode='lines', name='Net Reserves (History, excl. PBOC)',
                        line=dict(color='#e74c3c', width=2, dash='dot')
                    ))
                if 'net_reserves_usd_m' in fut.columns:
                    fig_band.add_trace(go.Scatter(
                        x=fut['date'], y=fut['net_reserves_usd_m'],
                        mode='lines+markers', name=f'Net Reserves Forecast - {scenario_choice}',
                        line=dict(color='#c0392b', width=2, dash='dashdot'),
                        marker=dict(size=5)
                    ))
                fig_band.update_layout(
                    xaxis_title="Date",
                    yaxis_title="USD Millions",
                    height=500,
                    hovermode='x unified'
                )
                st.plotly_chart(fig_band, width="stretch")
                
                # Robustness chart: Import cover and GG
                st.subheader("Robustness: Import Cover and GG Ratio")
                fig_robust = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                           subplot_titles=("Import Cover (months)", "Greenspan-Guidotti Ratio"))
                if 'import_cover' in combined.columns:
                    fig_robust.add_trace(go.Scatter(
                        x=hist['date'], y=hist['import_cover'],
                        mode='lines', name='Import Cover (History)',
                        line=dict(color='#2980b9', width=2)
                    ), row=1, col=1)
                    fig_robust.add_trace(go.Scatter(
                        x=fut['date'], y=fut['import_cover'],
                        mode='lines+markers', name=f'Import Cover Forecast - {scenario_choice}',
                        line=dict(color='#8e44ad', width=2, dash='dash'),
                        marker=dict(size=6)
                    ), row=1, col=1)
                    fig_robust.add_hline(y=3, line_dash="dash", line_color="orange", row=1, col=1,
                                         annotation_text="3 months", annotation_position="right")
                    fig_robust.add_hline(y=6, line_dash="dash", line_color="green", row=1, col=1,
                                         annotation_text="6 months", annotation_position="right")
                
                if 'gg_ratio' in combined.columns:
                    fig_robust.add_trace(go.Scatter(
                        x=hist['date'], y=hist['gg_ratio'],
                        mode='lines', name='GG Ratio (History)',
                        line=dict(color='#27ae60', width=2)
                    ), row=2, col=1)
                    fig_robust.add_trace(go.Scatter(
                        x=fut['date'], y=fut['gg_ratio'],
                        mode='lines+markers', name=f'GG Ratio Forecast - {scenario_choice}',
                        line=dict(color='#c0392b', width=2, dash='dash'),
                        marker=dict(size=6)
                    ), row=2, col=1)
                    fig_robust.add_hline(y=1.0, line_dash="dash", line_color="red", row=2, col=1,
                                         annotation_text="1.0 threshold", annotation_position="right")
                
                fig_robust.update_layout(height=600, hovermode='x unified', showlegend=True)
                st.plotly_chart(fig_robust, width="stretch")
                
                # Data table and download
                st.subheader(f"Forecast Data - {scenario_choice}")
                st.dataframe(selected, hide_index=True, width="stretch")
                csv = selected.to_csv(index=False)
                st.download_button(f"Download {scenario_choice} CSV", csv, f"forecast_{scenario_choice.lower()}.csv", "text/csv")

                # BoP Decomposition Chart
                if bop_decompositions and scenario_choice in bop_decompositions:
                    st.subheader("BoP Component Decomposition of Reserve Changes")
                    st.markdown("""
                    *Structural drivers of projected quarterly reserve changes.
                    Residual captures FDI, debt disbursements/service, IMF tranches, and other unmodeled flows.*
                    """)

                    bop_df = bop_decompositions[scenario_choice]

                    fig_bop = go.Figure()

                    # Inflow components (positive bars)
                    for col, name, color in [
                        ('exports_usd_m', 'Exports', '#2ecc71'),
                        ('remittances_usd_m', 'Remittances', '#27ae60'),
                        ('tourism_earnings_usd_m', 'Tourism', '#1abc9c'),
                        ('cse_net_usd_m', 'CSE Net Flows', '#3498db'),
                        ('residual_usd_m', 'Residual (FDI, debt svc, other)', '#95a5a6'),
                    ]:
                        if col in bop_df.columns:
                            fig_bop.add_trace(go.Bar(
                                x=bop_df['date'], y=bop_df[col],
                                name=name, marker_color=color,
                            ))

                    # Imports as negative bar
                    if 'imports_usd_m' in bop_df.columns:
                        fig_bop.add_trace(go.Bar(
                            x=bop_df['date'], y=-bop_df['imports_usd_m'],
                            name='Imports (outflow)', marker_color='#e74c3c',
                        ))

                    # Net delta line
                    if 'delta_reserves' in bop_df.columns:
                        fig_bop.add_trace(go.Scatter(
                            x=bop_df['date'], y=bop_df['delta_reserves'],
                            mode='lines+markers', name='Net \u0394Reserves',
                            line=dict(color='black', width=3),
                            marker=dict(size=8),
                        ))

                    fig_bop.update_layout(
                        barmode='relative',
                        xaxis_title="Date",
                        yaxis_title="USD Millions (quarterly)",
                        height=500,
                        hovermode='x unified',
                    )
                    st.plotly_chart(fig_bop, width="stretch")

    # ===================
    # TAB 5: External Flows
    # ===================
    with tab5:
        st.header("External Flows")
        st.markdown("*Tourism earnings, remittances, and portfolio flows*")
        
        col1, col2 = st.columns(2)
        
        # Tourism Earnings
        with col1:
            st.subheader("Tourism Earnings")
            if tourism is not None and len(tourism) > 0:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=tourism['date'],
                    y=tourism['tourism_earnings_usd_m'],
                    mode='lines',
                    name='Tourism',
                    fill='tozeroy',
                    line=dict(color='#3498db')
                ))
                fig.add_shape(type="line", x0=DEFAULT_DATE, x1=DEFAULT_DATE,
                             y0=0, y1=1, yref="paper",
                             line=dict(color="red", width=1, dash="dash"))
                fig.update_layout(
                    xaxis_title="Date", yaxis_title="USD Millions",
                    height=350, margin=dict(t=30)
                )
                st.plotly_chart(fig, width="stretch")
                
                # Stats
                recent = tourism[tourism['date'] >= '2023-01-01']
                if len(recent) > 0:
                    st.metric("Avg Monthly (2023+)", f"${recent['tourism_earnings_usd_m'].mean():,.0f}M")
            else:
                st.warning("Tourism data not available")
        
        # Remittances
        with col2:
            st.subheader("Workers' Remittances")
            if remittances is not None and len(remittances) > 0:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=remittances['date'],
                    y=remittances['remittances_usd_m'],
                    mode='lines',
                    name='Remittances',
                    fill='tozeroy',
                    line=dict(color='#2ecc71')
                ))
                fig.add_shape(type="line", x0=DEFAULT_DATE, x1=DEFAULT_DATE,
                             y0=0, y1=1, yref="paper",
                             line=dict(color="red", width=1, dash="dash"))
                fig.update_layout(
                    xaxis_title="Date", yaxis_title="USD Millions",
                    height=350, margin=dict(t=30)
                )
                st.plotly_chart(fig, width="stretch")
                
                recent = remittances[remittances['date'] >= '2023-01-01']
                if len(recent) > 0:
                    st.metric("Avg Monthly (2023+)", f"${recent['remittances_usd_m'].mean():,.0f}M")
            else:
                st.warning("Remittances data not available")
        
        # CSE Flows
        st.subheader("CSE Portfolio Flows")
        if cse_flows is not None and len(cse_flows) > 0:
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=cse_flows['date'],
                y=cse_flows['cse_net_usd_m'],
                name='Net Flows',
                marker_color=np.where(cse_flows['cse_net_usd_m'] >= 0, '#2ecc71', '#e74c3c')
            ))
            
            fig.add_shape(type="line", x0=DEFAULT_DATE, x1=DEFAULT_DATE,
                         y0=0, y1=1, yref="paper",
                         line=dict(color="gray", width=2, dash="dash"))
            
            fig.update_layout(
                title="Monthly CSE Net Portfolio Flows",
                xaxis_title="Date", yaxis_title="USD Millions (Net)",
                height=400
            )
            st.plotly_chart(fig, width="stretch")
            
            # Calculate cumulative
            cse_flows_sorted = cse_flows.sort_values('date')
            cse_flows_sorted['cumulative'] = cse_flows_sorted['cse_net_usd_m'].cumsum()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Net Inflows", f"${cse_flows_sorted['cumulative'].iloc[-1]:,.0f}M")
            col2.metric("2022 Outflows", f"${cse_flows[(cse_flows['date'].dt.year == 2022)]['cse_net_usd_m'].sum():,.0f}M")
            col3.metric("2024 Flows", f"${cse_flows[(cse_flows['date'].dt.year == 2024)]['cse_net_usd_m'].sum():,.0f}M")
        else:
            st.warning("CSE flows data not available")
    
    # ===================
    # TAB 6: Monetary
    # ===================
    with tab6:
        st.header("Monetary Indicators")
        st.markdown("*Broad money, reserve money, and multipliers*")
        
        # M2 Chart
        st.subheader("Broad Money Supply (M2)")
        if m2 is not None and len(m2) > 0:
            # Check which column we have
            m2_col = 'broad_money_m2_lkr_m' if 'broad_money_m2_lkr_m' in m2.columns else 'broad_money_m2'
            
            if m2_col in m2.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=m2['date'],
                    y=m2[m2_col] / 1e6,  # Convert to trillions
                    mode='lines',
                    name='M2',
                    line=dict(color='#9b59b6', width=2)
                ))
                fig.add_shape(type="line", x0=DEFAULT_DATE, x1=DEFAULT_DATE,
                             y0=0, y1=1, yref="paper",
                             line=dict(color="red", width=1, dash="dash"))
                fig.update_layout(
                    xaxis_title="Date", yaxis_title="LKR Trillions",
                    height=400
                )
                st.plotly_chart(fig, width="stretch")
                
                col1, col2 = st.columns(2)
                latest = m2.iloc[-1]
                col1.metric("Latest M2", f"LKR {latest[m2_col]/1e6:,.1f}T")
                col2.metric("Date", latest['date'].strftime('%Y-%m'))
        else:
            st.warning("M2 data not available")
        
        # Reserve Money
        st.subheader("Reserve Money & Multiplier")
        if reserve_money is not None and len(reserve_money) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=reserve_money['date'],
                    y=reserve_money['reserve_money_total_lkr_m'] / 1e6,
                    mode='lines',
                    name='Reserve Money',
                    line=dict(color='#e67e22', width=2)
                ))
                fig.update_layout(
                    title="Reserve Money (M0)",
                    xaxis_title="Date", yaxis_title="LKR Trillions",
                    height=350
                )
                st.plotly_chart(fig, width="stretch")
            
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=reserve_money['date'],
                    y=reserve_money['money_multiplier_m1'],
                    mode='lines',
                    name='M1 Multiplier',
                    line=dict(color='#1abc9c', width=2)
                ))
                fig.update_layout(
                    title="Money Multiplier (M1)",
                    xaxis_title="Date", yaxis_title="Multiplier",
                    height=350
                )
                st.plotly_chart(fig, width="stretch")
        else:
            st.warning("Reserve money data not available")
        
        # External Debt in USD
        st.subheader("External Debt (USD)")
        if ext_debt_usd is not None and len(ext_debt_usd) > 0:
            fig = go.Figure()
            
            if 'govt_total_usd_m' in ext_debt_usd.columns:
                fig.add_trace(go.Scatter(
                    x=ext_debt_usd['date'],
                    y=ext_debt_usd['govt_total_usd_m'] / 1000,
                    mode='lines',
                    name='Govt Total',
                    line=dict(color='#3498db', width=2)
                ))
            
            if 'govt_short_term_usd_m' in ext_debt_usd.columns:
                fig.add_trace(go.Scatter(
                    x=ext_debt_usd['date'],
                    y=ext_debt_usd['govt_short_term_usd_m'] / 1000,
                    mode='lines',
                    name='Govt Short-term',
                    line=dict(color='#e74c3c', width=2)
                ))
            
            fig.add_shape(type="line", x0=DEFAULT_DATE, x1=DEFAULT_DATE,
                         y0=0, y1=1, yref="paper",
                         line=dict(color="gray", width=2, dash="dash"))
            
            fig.update_layout(
                title="External Debt by Category",
                xaxis_title="Date", yaxis_title="USD Billions",
                height=400
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.warning("External debt USD data not available")
    
    # ===================
    # TAB 7: Data Coverage
    # ===================
    with tab7:
        st.header("Data Coverage Analysis")
        
        st.markdown("""
        This section shows the coverage of each data source and highlights gaps 
        that affect our ability to calculate reserve adequacy metrics.
        """)
        
        # Build data sources info
        data_sources = {}
        
        if reserves is not None:
            data_sources['Reserves (CBSL)'] = {
                'start': reserves['date'].min(),
                'end': reserves['date'].max(),
                'status': 'available',
                'records': len(reserves)
            }
        
        if debt is not None:
            data_sources['Short-term Debt'] = {
                'start': debt['date'].min(),
                'end': debt['date'].max(),
                'status': 'available',
                'records': len(debt)
            }
        
        if fx is not None:
            data_sources['USD/LKR FX Rate'] = {
                'start': fx['date'].min(),
                'end': fx['date'].max(),
                'status': 'available',
                'records': len(fx)
            }
        
        if iip is not None:
            pl_valid = iip[iip['portfolio_liabilities'].notna()]
            if len(pl_valid) > 0:
                data_sources['Portfolio Liabilities'] = {
                    'start': pl_valid['date'].min(),
                    'end': pl_valid['date'].max(),
                    'status': 'partial',  # Ends before crisis
                    'records': len(pl_valid)
                }
        
        if m2 is not None:
            # Handle both column naming conventions
            m2_col = 'broad_money_m2_lkr_m' if 'broad_money_m2_lkr_m' in m2.columns else 'broad_money_m2'
            if m2_col in m2.columns:
                m2_valid = m2[m2[m2_col].notna()]
                if len(m2_valid) > 0:
                    data_sources['Broad Money M2'] = {
                        'start': m2_valid['date'].min(),
                        'end': m2_valid['date'].max(),
                        'status': 'available',  # Now we have full data
                        'records': len(m2_valid)
                    }
        
        # Coverage table
        st.subheader("Data Source Summary")
        
        coverage_df = pd.DataFrame([
            {
                'Source': name,
                'Start Date': info['start'].strftime('%Y-%m-%d'),
                'End Date': info['end'].strftime('%Y-%m-%d'),
                'Records': info['records'],
                'Status': '✅ Complete' if info['status'] == 'available' else '⚠️ Partial'
            }
            for name, info in data_sources.items()
        ])
        
        st.dataframe(coverage_df, width="stretch", hide_index=True)
        
        # Coverage timeline chart
        st.subheader("Coverage Timeline")
        
        fig = create_data_coverage_chart(data_sources)
        st.plotly_chart(fig, width="stretch")
        
        # Gap analysis
        st.subheader("Data Gaps for Reserve Adequacy Metrics")
        
        gaps = []
        
        # Check GG ratio gaps
        gg_data = calculate_greenspan_guidotti(reserves_base, debt, fx)
        if gg_data is not None:
            gaps.append({
                'Metric': 'Greenspan-Guidotti Ratio',
                'Status': '✅ Calculable',
                'Coverage': f"{gg_data['date'].min().strftime('%Y-Q%q')} to {gg_data['date'].max().strftime('%Y-Q%q')}",
                'Gap': 'None for 2014-2025'
            })
        else:
            gaps.append({
                'Metric': 'Greenspan-Guidotti Ratio',
                'Status': '❌ Missing data',
                'Coverage': 'N/A',
                'Gap': 'Missing FX rates or debt data'
            })
        
        # Check IMF ARA gaps
        if m2 is not None and iip is not None:
            m2_col = 'broad_money_m2_lkr_m' if 'broad_money_m2_lkr_m' in m2.columns else 'broad_money_m2'
            if m2_col in m2.columns:
                m2_end = m2[m2[m2_col].notna()]['date'].max()
                pl_end = iip[iip['portfolio_liabilities'].notna()]['date'].max()
                gaps.append({
                    'Metric': 'IMF ARA Metric',
                    'Status': '✅ M2 Complete' if m2_end.year >= 2025 else '⚠️ Partial',
                    'Coverage': f"M2 to {m2_end.strftime('%Y-%m')}, Portfolio to {pl_end.strftime('%Y-%m')}",
                    'Gap': 'Portfolio liabilities end before crisis' if pl_end.year < 2020 else 'None'
                })
        else:
            gaps.append({
                'Metric': 'IMF ARA Metric',
                'Status': '❌ Missing data',
                'Coverage': 'N/A',
                'Gap': 'Missing M2 or Portfolio Liabilities'
            })
        
        # Import cover
        gaps.append({
            'Metric': 'Import Cover',
            'Status': '✅ Calculable',
            'Coverage': f"{reserves['date'].min().strftime('%Y-%m')} to {reserves['date'].max().strftime('%Y-%m')}",
            'Gap': 'Uses estimated monthly imports ($1.5B)'
        })
        
        st.dataframe(pd.DataFrame(gaps), width="stretch", hide_index=True)
        
        # Recommendations
        st.subheader("📋 Recommendations to Fill Gaps")
        
        st.markdown("""
        To enable full IMF ARA calculations for the crisis period (2020-2024):
        
        1. **Broad Money M2 (2014-2025)**
           - Source: CBSL Monthly Bulletin, Statistical Appendix Table 2.1
           - URL: https://www.cbsl.gov.lk/en/publications/statistical-tables
        
        2. **Portfolio Investment Liabilities (2013-2025)**
           - Source: CBSL International Investment Position releases
           - URL: https://www.cbsl.gov.lk/en/statistics/statistical-tables/external-sector
           - Alternative: IMF International Financial Statistics (IFS)
        
        3. **Export Data (for full ARA)**
           - Source: CBSL External Sector Statistics
           - Already partially available in monthly panel
        """)
    
    # ===================
    # TAB 8: Data Tables
    # ===================
    with tab8:
        st.header("Raw Data Tables")
        
        data_option = st.selectbox(
            "Select data to view:",
            ["Greenspan-Guidotti Calculation", "Reserve Assets", "External Debt (USD)",
             "Short-term Debt (LKR)", "Money Supply (M2)", "Tourism Earnings", 
             "Remittances", "CSE Flows", "Reserve Money", "IIP Data"]
        )
        
        if data_option == "Greenspan-Guidotti Calculation":
            gg_data = calculate_greenspan_guidotti(reserves_base, debt, fx)
            if gg_data is not None:
                st.dataframe(gg_data.sort_values('date', ascending=False), 
                            width="stretch", hide_index=True)
                csv = gg_data.to_csv(index=False)
                st.download_button("Download CSV", csv, "greenspan_guidotti.csv", "text/csv")
        
        elif data_option == "Reserve Assets":
            if reserves is not None:
                st.dataframe(reserves.sort_values('date', ascending=False), 
                            width="stretch", hide_index=True)
        
        elif data_option == "External Debt (USD)":
            if ext_debt_usd is not None:
                st.dataframe(ext_debt_usd.sort_values('date', ascending=False),
                            width="stretch", hide_index=True)
                csv = ext_debt_usd.to_csv(index=False)
                st.download_button("Download CSV", csv, "external_debt_usd.csv", "text/csv")
        
        elif data_option == "Short-term Debt (LKR)":
            if debt is not None:
                st.dataframe(debt.sort_values('date', ascending=False),
                            width="stretch", hide_index=True)
        
        elif data_option == "Money Supply (M2)":
            if m2 is not None:
                st.dataframe(m2.sort_values('date', ascending=False),
                            width="stretch", hide_index=True)
        
        elif data_option == "Tourism Earnings":
            if tourism is not None:
                st.dataframe(tourism.sort_values('date', ascending=False),
                            width="stretch", hide_index=True)
        
        elif data_option == "Remittances":
            if remittances is not None:
                st.dataframe(remittances.sort_values('date', ascending=False),
                            width="stretch", hide_index=True)
        
        elif data_option == "CSE Flows":
            if cse_flows is not None:
                st.dataframe(cse_flows.sort_values('date', ascending=False),
                            width="stretch", hide_index=True)
        
        elif data_option == "Reserve Money":
            if reserve_money is not None:
                st.dataframe(reserve_money.sort_values('date', ascending=False),
                            width="stretch", hide_index=True)
        
        elif data_option == "IIP Data":
            if iip is not None:
                st.dataframe(iip.sort_values('date', ascending=False),
                            width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
