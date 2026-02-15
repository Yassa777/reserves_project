"""Configuration for forecasting dataset preparation."""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DIAG_DIR = DATA_DIR / "diagnostics"
OUTPUT_DIR = DATA_DIR / "forecast_prep"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_VAR = "gross_reserves_usd_m"

BASELINE_ARIMA_EXOG_VARS = [
    "exports_usd_m",
    "imports_usd_m",
    "remittances_usd_m",
    "usd_lkr",
]

BASELINE_VECM_SYSTEM_VARS = [
    "gross_reserves_usd_m",
    "exports_usd_m",
    "imports_usd_m",
    "remittances_usd_m",
    "usd_lkr",
]

BASELINE_MS_VAR_SYSTEM_VARS = [
    "gross_reserves_usd_m",
    "usd_lkr",
    "exports_usd_m",
    "imports_usd_m",
]

EXPANDED_ARIMA_EXOG_VARS = [
    "exports_usd_m",
    "imports_usd_m",
    "remittances_usd_m",
    "usd_lkr",
    "trade_balance_usd_m",
    "m2_usd_m",
    "tourism_usd_m",
    "cse_net_usd_m",
]

EXPANDED_VECM_SYSTEM_VARS = [
    "gross_reserves_usd_m",
    "exports_usd_m",
    "imports_usd_m",
    "remittances_usd_m",
    "usd_lkr",
    "trade_balance_usd_m",
    "m2_usd_m",
]

EXPANDED_MS_VAR_SYSTEM_VARS = [
    "gross_reserves_usd_m",
    "usd_lkr",
    "exports_usd_m",
    "imports_usd_m",
    "trade_balance_usd_m",
    "m2_usd_m",
]

VARSETS = {
    "baseline": {
        "arima_exog": BASELINE_ARIMA_EXOG_VARS,
        "vecm_system": BASELINE_VECM_SYSTEM_VARS,
        "ms_var_system": BASELINE_MS_VAR_SYSTEM_VARS,
    },
    "expanded": {
        "arima_exog": EXPANDED_ARIMA_EXOG_VARS,
        "vecm_system": EXPANDED_VECM_SYSTEM_VARS,
        "ms_var_system": EXPANDED_MS_VAR_SYSTEM_VARS,
    },
}

DEFAULT_VARSET = "baseline"

MISSING_STRATEGY = {
    "method": "ffill_limit",
    "limit": 3,
    "drop_remaining": True,
}

# Backwards-compatible aliases (baseline)
ARIMA_EXOG_VARS = BASELINE_ARIMA_EXOG_VARS
VECM_SYSTEM_VARS = BASELINE_VECM_SYSTEM_VARS
MS_VAR_SYSTEM_VARS = BASELINE_MS_VAR_SYSTEM_VARS

TRAIN_END = pd.Timestamp("2019-12-01")
VALID_END = pd.Timestamp("2022-12-01")

MIN_OBS_VECM = 100
MIN_OBS_MS = 80


def get_varset(name: str | None) -> dict:
    if not name:
        return VARSETS[DEFAULT_VARSET]
    return VARSETS.get(name, VARSETS[DEFAULT_VARSET])


def get_varset_name(name: str | None) -> str:
    if not name:
        return DEFAULT_VARSET
    return name if name in VARSETS else DEFAULT_VARSET


def get_output_dir(varset: str | None = None) -> Path:
    suffix = ""
    varset_name = get_varset_name(varset)
    if varset_name != "baseline":
        suffix = f"_{varset_name}"
    out_dir = DATA_DIR / f"forecast_prep{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
