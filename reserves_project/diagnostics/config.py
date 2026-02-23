"""Shared configuration for reserves diagnostics."""

import pandas as pd

from reserves_project.config.paths import PROJECT_ROOT, DATA_DIR
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
