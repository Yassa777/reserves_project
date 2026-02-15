"""Shared configuration for reserves diagnostics."""

from pathlib import Path
import pandas as pd

# Project root is 3 levels up from this file: config.py -> diagnostics_phases -> scripts -> reserves_project -> SL-FSI
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MERGED_DIR = DATA_DIR / "merged"
OUTPUT_DIR = DATA_DIR / "diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CRISIS_START = pd.Timestamp("2020-01-01")
CRISIS_END = pd.Timestamp("2024-12-31")
DEFAULT_DATE = pd.Timestamp("2022-04-12")
PBOC_DATE = pd.Timestamp("2021-03-01")

KEY_VARIABLES = [
    "gross_reserves_usd_m",
    "net_usable_reserves_usd_m",
    "exports_usd_m",
    "imports_usd_m",
    "remittances_usd_m",
    "tourism_usd_m",
    "cse_net_usd_m",
    "usd_lkr",
    "m2_usd_m",
    "trade_balance_usd_m",
    "reserve_change_usd_m",
]

TARGET_VARIABLE = "gross_reserves_usd_m"
PHASE6_PREDICTORS = [
    "exports_usd_m",
    "imports_usd_m",
    "remittances_usd_m",
    "usd_lkr",
    "cse_net_usd_m",
]

PHASE7_COINTEGRATION_VARS = [
    "gross_reserves_usd_m",
    "exports_usd_m",
    "imports_usd_m",
    "remittances_usd_m",
    "usd_lkr",
]

PHASE8_SVAR_VARS = [
    "usd_lkr",
    "imports_usd_m",
    "exports_usd_m",
    "gross_reserves_usd_m",
]

PHASE9_BREAK_VARS = [
    "gross_reserves_usd_m",
    "net_usable_reserves_usd_m",
    "usd_lkr",
    "exports_usd_m",
    "imports_usd_m",
    "remittances_usd_m",
    "trade_balance_usd_m",
    "reserve_change_usd_m",
]

MIN_OBS_CORE = 20
MIN_OBS_BREAKS = 50
MIN_OBS_RELATIONSHIP = 50
MIN_OBS_COINTEGRATION = 80
MIN_OBS_SVAR = 80
