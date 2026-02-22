"""
Variable Set Configuration for Academic Reserves Forecasting Pipeline.

This module defines theoretically-motivated variable sets for systematic comparison:
1. Parsimonious Core - Minimal economically-motivated set
2. BoP-Focused - Balance of Payments drivers
3. Monetary Policy - Policy intervention channel
4. PCA-Reduced - Data-driven dimensionality reduction
5. Kitchen Sink (Full) - All available variables (benchmark for overfitting)

Reference: Specification 01 - Variable Sets Definition
"""

from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

from .paths import PROJECT_ROOT, DATA_DIR

# =============================================================================
# Path Configuration
# =============================================================================

SOURCE_DATA_PATH = DATA_DIR / "merged" / "reserves_forecasting_panel.csv"
SUPPLEMENTARY_DATA_PATH = DATA_DIR / "merged" / "slfsi_monthly_panel.csv"
HISTORICAL_FX_PATH = DATA_DIR / "external" / "historical_fx.csv"
OUTPUT_DIR = DATA_DIR / "forecast_prep_academic"

# =============================================================================
# Core Configuration
# =============================================================================

TARGET_VAR = "gross_reserves_usd_m"

# Train/Validation/Test Split
TRAIN_END = pd.Timestamp("2019-12-01")
VALID_END = pd.Timestamp("2022-12-01")

# Minimum observation thresholds
MIN_OBS_ARIMA = 60
MIN_OBS_VECM = 100
MIN_OBS_VAR = 80
MIN_OBS_PCA = 80

# Missing data strategy
MISSING_STRATEGY = {
    "method": "ffill_limit",
    "limit": 3,
    "drop_remaining": True,
}

# =============================================================================
# Variable Set Definitions
# =============================================================================

VARSET_PARSIMONIOUS: Dict[str, Any] = {
    "name": "parsimonious",
    "target": TARGET_VAR,
    "arima_exog": ["trade_balance_usd_m", "usd_lkr"],
    "vecm_system": ["gross_reserves_usd_m", "trade_balance_usd_m", "usd_lkr"],
    "var_system": ["gross_reserves_usd_m", "trade_balance_usd_m", "usd_lkr"],
    "description": "Minimal set: net trade + exchange rate",
    "economic_rationale": (
        "Trade balance captures primary current account driver. "
        "Exchange rate reflects intervention + valuation effects. "
        "Only 3 variables enables robust estimation with limited data."
    ),
}

VARSET_BOP: Dict[str, Any] = {
    "name": "bop",
    "target": TARGET_VAR,
    "arima_exog": ["exports_usd_m", "imports_usd_m", "remittances_usd_m", "tourism_usd_m"],
    "vecm_system": [
        "gross_reserves_usd_m",
        "exports_usd_m",
        "imports_usd_m",
        "remittances_usd_m",
        "tourism_usd_m",
    ],
    "var_system": [
        "gross_reserves_usd_m",
        "exports_usd_m",
        "imports_usd_m",
        "remittances_usd_m",
    ],
    "description": "Current account flow decomposition",
    "economic_rationale": (
        "Disaggregated flows allow modeling differential dynamics. "
        "Tourism and remittances are major Sri Lankan inflows. "
        "Excludes exchange rate to avoid endogeneity concerns in BoP identity."
    ),
}

VARSET_MONETARY: Dict[str, Any] = {
    "name": "monetary",
    "target": TARGET_VAR,
    "arima_exog": ["usd_lkr", "m2_usd_m"],
    "vecm_system": ["gross_reserves_usd_m", "usd_lkr", "m2_usd_m"],
    "var_system": ["gross_reserves_usd_m", "usd_lkr", "m2_usd_m"],
    "description": "Monetary policy and exchange rate intervention",
    "economic_rationale": (
        "CBSL intervenes via USD sales/purchases affecting reserves. "
        "M2 growth signals monetary stance (sterilization capacity). "
        "Exchange rate reflects intervention pressure."
    ),
}

VARSET_PCA: Dict[str, Any] = {
    "name": "pca",
    "target": TARGET_VAR,
    "source_vars": [
        "exports_usd_m",
        "imports_usd_m",
        "remittances_usd_m",
        "tourism_usd_m",
        "usd_lkr",
        "m2_usd_m",
        "cse_net_usd_m",
        "trade_balance_usd_m",
    ],
    "n_components": 3,
    "arima_exog": ["PC1", "PC2", "PC3"],
    "vecm_system": ["gross_reserves_usd_m", "PC1", "PC2", "PC3"],
    "var_system": ["gross_reserves_usd_m", "PC1", "PC2", "PC3"],
    "description": "Principal components of all macro variables",
    "economic_rationale": (
        "Data-driven dimensionality reduction captures common factors. "
        "Standardize all source variables before PCA. "
        "Use training sample only for PCA fit to avoid look-ahead bias."
    ),
}

VARSET_FULL: Dict[str, Any] = {
    "name": "full",
    "target": TARGET_VAR,
    "arima_exog": [
        "exports_usd_m",
        "imports_usd_m",
        "remittances_usd_m",
        "tourism_usd_m",
        "usd_lkr",
        "m2_usd_m",
        "cse_net_usd_m",
        "trade_balance_usd_m",
    ],
    "vecm_system": [
        "gross_reserves_usd_m",
        "exports_usd_m",
        "imports_usd_m",
        "remittances_usd_m",
        "usd_lkr",
        "m2_usd_m",
        "trade_balance_usd_m",
    ],
    "var_system": [
        "gross_reserves_usd_m",
        "exports_usd_m",
        "imports_usd_m",
        "remittances_usd_m",
        "usd_lkr",
        "m2_usd_m",
    ],
    "description": "All available variables (overfitting benchmark)",
    "economic_rationale": (
        "Kitchen sink model to benchmark overfitting. "
        "Should perform worse than parsimonious sets out-of-sample. "
        "Useful for assessing value of variable selection."
    ),
}

# Registry of all variable sets
VARIABLE_SETS: Dict[str, Dict[str, Any]] = {
    "parsimonious": VARSET_PARSIMONIOUS,
    "bop": VARSET_BOP,
    "monetary": VARSET_MONETARY,
    "pca": VARSET_PCA,
    "full": VARSET_FULL,
}

# Default ordering for execution
VARSET_ORDER: List[str] = ["parsimonious", "bop", "monetary", "pca", "full"]


__all__ = [
    # Paths
    "PROJECT_ROOT",
    "DATA_DIR",
    "SOURCE_DATA_PATH",
    "SUPPLEMENTARY_DATA_PATH",
    "HISTORICAL_FX_PATH",
    "OUTPUT_DIR",
    # Core config
    "TARGET_VAR",
    "TRAIN_END",
    "VALID_END",
    "MIN_OBS_ARIMA",
    "MIN_OBS_VECM",
    "MIN_OBS_VAR",
    "MIN_OBS_PCA",
    "MISSING_STRATEGY",
    # Variable sets
    "VARSET_PARSIMONIOUS",
    "VARSET_BOP",
    "VARSET_MONETARY",
    "VARSET_PCA",
    "VARSET_FULL",
    "VARIABLE_SETS",
    "VARSET_ORDER",
    # Functions
    "get_varset",
    "get_all_varsets",
    "get_output_dir",
    "get_all_required_vars",
]


def get_varset(name: str) -> Dict[str, Any]:
    """Get a variable set configuration by name.

    Parameters
    ----------
    name : str
        Name of the variable set (parsimonious, bop, monetary, pca, full)

    Returns
    -------
    dict
        Variable set configuration dictionary

    Raises
    ------
    KeyError
        If variable set name is not found
    """
    if name not in VARIABLE_SETS:
        available = list(VARIABLE_SETS.keys())
        raise KeyError(f"Variable set '{name}' not found. Available: {available}")
    return VARIABLE_SETS[name]


def get_all_varsets() -> Dict[str, Dict[str, Any]]:
    """Return all variable set configurations."""
    return VARIABLE_SETS


def get_output_dir(varset_name: str) -> Path:
    """Get output directory for a specific variable set.

    Parameters
    ----------
    varset_name : str
        Name of the variable set

    Returns
    -------
    Path
        Output directory path for the variable set
    """
    out_dir = OUTPUT_DIR / f"varset_{varset_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def get_all_required_vars(varset: Dict[str, Any]) -> List[str]:
    """Extract all unique variables required by a variable set.

    Parameters
    ----------
    varset : dict
        Variable set configuration

    Returns
    -------
    list
        List of unique variable names
    """
    all_vars = set()

    # Add target
    all_vars.add(varset["target"])

    # Add ARIMA exogenous variables
    all_vars.update(varset.get("arima_exog", []))

    # Add VECM system variables
    all_vars.update(varset.get("vecm_system", []))

    # Add VAR system variables
    all_vars.update(varset.get("var_system", []))

    # Add PCA source variables if present
    all_vars.update(varset.get("source_vars", []))

    # Remove generated PC variables from requirements
    all_vars = {v for v in all_vars if not v.startswith("PC")}

    return sorted(list(all_vars))
