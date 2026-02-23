"""Configuration constants and data source definitions."""

from datetime import datetime

from reserves_project.config.paths import PROJECT_ROOT, DATA_DIR

# Paths - anchor to project root
EXTERNAL_DIR = DATA_DIR / "external"
MERGED_DIR = DATA_DIR / "merged"
PROCESSED_DIR = DATA_DIR / "processed"

# Crisis dates
DEFAULT_DATE = datetime(2022, 4, 12)
CRISIS_START = datetime(2020, 1, 1)
CRISIS_END = datetime(2024, 12, 31)
PBOC_SWAP_DATE = datetime(2021, 3, 1)
PBOC_SWAP_USD_M = 1500

# Data source definitions
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
        "value_cols": None,
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
