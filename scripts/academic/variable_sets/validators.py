"""
Data Validators for Academic Reserves Forecasting Pipeline.

This module provides validation utilities to check data availability
and quality for each variable set before dataset preparation.

Reference: Specification 01 - Variable Sets Definition
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

from .config import (
    TARGET_VAR,
    TRAIN_END,
    VALID_END,
    MIN_OBS_ARIMA,
    MIN_OBS_VECM,
    MIN_OBS_VAR,
    MIN_OBS_PCA,
    MISSING_STRATEGY,
)


def validate_variable_set(
    df: pd.DataFrame,
    varset_config: Dict[str, Any],
    min_obs: int = 100
) -> Dict[str, Any]:
    """
    Check if variable set has sufficient observations.

    Parameters
    ----------
    df : pd.DataFrame
        Source data with DatetimeIndex
    varset_config : dict
        Variable set configuration from config.py
    min_obs : int
        Minimum required observations for validity

    Returns
    -------
    dict
        Validation results with:
        - valid: bool indicating if set meets requirements
        - n_obs: int number of complete observations
        - missing_vars: list of variables not in source data
        - date_range: tuple of (start_date, end_date)
        - train_obs: int observations in training period
        - valid_obs: int observations in validation period
        - test_obs: int observations in test period
    """
    # Collect all required variables (exclude PC* columns which are generated)
    all_vars = set()
    all_vars.add(varset_config.get("target", TARGET_VAR))

    for key in ["arima_exog", "vecm_system", "var_system", "source_vars"]:
        vars_list = varset_config.get(key, [])
        # Filter out PC columns
        all_vars.update(v for v in vars_list if not v.startswith("PC"))

    all_vars = list(all_vars)

    # Check for missing variables
    missing_vars = [v for v in all_vars if v not in df.columns]
    if missing_vars:
        return {
            "valid": False,
            "n_obs": 0,
            "missing_vars": missing_vars,
            "date_range": (None, None),
            "train_obs": 0,
            "valid_obs": 0,
            "test_obs": 0,
            "error": f"Missing variables: {missing_vars}"
        }

    # Get complete cases
    subset = df[all_vars].dropna()
    n_obs = len(subset)

    if n_obs == 0:
        return {
            "valid": False,
            "n_obs": 0,
            "missing_vars": [],
            "date_range": (None, None),
            "train_obs": 0,
            "valid_obs": 0,
            "test_obs": 0,
            "error": "No complete observations after removing missing values"
        }

    # Compute split counts
    train_obs = len(subset[subset.index <= TRAIN_END])
    valid_obs = len(subset[(subset.index > TRAIN_END) & (subset.index <= VALID_END)])
    test_obs = len(subset[subset.index > VALID_END])

    return {
        "valid": n_obs >= min_obs,
        "n_obs": n_obs,
        "missing_vars": [],
        "date_range": (subset.index.min(), subset.index.max()),
        "train_obs": train_obs,
        "valid_obs": valid_obs,
        "test_obs": test_obs,
    }


def validate_all_varsets(
    df: pd.DataFrame,
    varsets: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Validate all variable sets and return summary table.

    Parameters
    ----------
    df : pd.DataFrame
        Source data
    varsets : dict
        Dictionary of variable set configurations

    Returns
    -------
    pd.DataFrame
        Summary table with validation results for each variable set
    """
    results = []

    for name, config in varsets.items():
        # Determine appropriate minimum observation threshold
        if name == "pca":
            min_obs = MIN_OBS_PCA
        else:
            min_obs = MIN_OBS_VECM

        validation = validate_variable_set(df, config, min_obs=min_obs)

        results.append({
            "variable_set": name,
            "valid": validation["valid"],
            "total_obs": validation["n_obs"],
            "train_obs": validation["train_obs"],
            "valid_obs": validation["valid_obs"],
            "test_obs": validation["test_obs"],
            "start_date": validation["date_range"][0],
            "end_date": validation["date_range"][1],
            "missing_vars": ", ".join(validation.get("missing_vars", [])),
            "n_arima_exog": len(config.get("arima_exog", [])),
            "n_vecm_vars": len(config.get("vecm_system", [])),
        })

    return pd.DataFrame(results)


def apply_missing_strategy(
    df: pd.DataFrame,
    columns: List[str],
    strategy: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Apply missing data handling strategy to specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    columns : list
        Columns to apply strategy to
    strategy : dict, optional
        Missing strategy configuration. Defaults to MISSING_STRATEGY.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame
    dict
        Statistics about missing data handling (original_missing, filled, dropped)
    """
    if strategy is None:
        strategy = MISSING_STRATEGY

    result = df.copy()
    available_cols = [c for c in columns if c in result.columns]

    # Count original missing values
    original_missing = result[available_cols].isna().sum().sum()

    method = strategy.get("method", "ffill_limit")
    limit = strategy.get("limit", 3)
    drop_remaining = strategy.get("drop_remaining", True)

    if method == "ffill_limit":
        # Forward fill with limit
        result[available_cols] = result[available_cols].ffill(limit=limit)

    elif method == "ffill":
        # Forward fill without limit
        result[available_cols] = result[available_cols].ffill()

    elif method == "interpolate":
        # Linear interpolation
        result[available_cols] = result[available_cols].interpolate(
            method="linear", limit=limit
        )

    # Count remaining missing after fill
    remaining_missing = result[available_cols].isna().sum().sum()
    filled_count = original_missing - remaining_missing

    # Drop remaining missing if configured
    dropped_rows = 0
    if drop_remaining:
        original_len = len(result)
        result = result.dropna(subset=available_cols)
        dropped_rows = original_len - len(result)

    stats = {
        "original_missing": int(original_missing),
        "filled": int(filled_count),
        "dropped_rows": int(dropped_rows),
        "final_obs": len(result),
    }

    return result, stats


def check_data_quality(
    df: pd.DataFrame,
    columns: List[str]
) -> Dict[str, Any]:
    """
    Perform comprehensive data quality checks.

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    columns : list
        Columns to check

    Returns
    -------
    dict
        Quality metrics including:
        - missing_pct: percentage of missing values per column
        - constant_cols: columns with zero variance
        - infinite_values: count of inf values
        - outlier_count: number of values > 3 std from mean
    """
    available_cols = [c for c in columns if c in df.columns]
    subset = df[available_cols]

    # Missing percentage
    missing_pct = (subset.isna().sum() / len(subset) * 100).to_dict()

    # Constant columns (zero variance)
    constant_cols = [c for c in available_cols if subset[c].std() == 0]

    # Infinite values
    numeric_subset = subset.select_dtypes(include=[np.number])
    inf_count = np.isinf(numeric_subset).sum().sum()

    # Outliers (>3 std from mean)
    outlier_counts = {}
    for col in numeric_subset.columns:
        mean = numeric_subset[col].mean()
        std = numeric_subset[col].std()
        if std > 0:
            outliers = ((numeric_subset[col] - mean).abs() > 3 * std).sum()
            outlier_counts[col] = int(outliers)

    return {
        "missing_pct": missing_pct,
        "constant_cols": constant_cols,
        "infinite_values": int(inf_count),
        "outlier_counts": outlier_counts,
        "total_outliers": sum(outlier_counts.values()),
    }


def validate_date_index(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate the DatetimeIndex of the source data.

    Parameters
    ----------
    df : pd.DataFrame
        Input data

    Returns
    -------
    dict
        Index validation results
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        return {
            "valid": False,
            "error": "Index is not DatetimeIndex",
        }

    # Check for duplicates
    has_duplicates = df.index.duplicated().any()

    # Check frequency
    inferred_freq = pd.infer_freq(df.index)

    # Check for gaps (assuming monthly)
    date_range = pd.date_range(df.index.min(), df.index.max(), freq="MS")
    missing_dates = date_range.difference(df.index)

    return {
        "valid": True,
        "start_date": df.index.min(),
        "end_date": df.index.max(),
        "n_observations": len(df),
        "has_duplicates": has_duplicates,
        "inferred_frequency": inferred_freq,
        "n_missing_months": len(missing_dates),
        "missing_months": [str(d.date()) for d in missing_dates[:10]],  # First 10
    }


def check_train_valid_test_split(
    df: pd.DataFrame,
    train_end: pd.Timestamp = TRAIN_END,
    valid_end: pd.Timestamp = VALID_END,
) -> Dict[str, Any]:
    """
    Verify train/validation/test split has sufficient observations.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with DatetimeIndex
    train_end : pd.Timestamp
        End of training period
    valid_end : pd.Timestamp
        End of validation period

    Returns
    -------
    dict
        Split statistics and warnings
    """
    train_df = df[df.index <= train_end]
    valid_df = df[(df.index > train_end) & (df.index <= valid_end)]
    test_df = df[df.index > valid_end]

    warnings = []

    if len(train_df) < MIN_OBS_VECM:
        warnings.append(f"Training set has only {len(train_df)} obs (need {MIN_OBS_VECM})")

    if len(valid_df) < 12:
        warnings.append(f"Validation set has only {len(valid_df)} obs (need >= 12)")

    if len(test_df) < 6:
        warnings.append(f"Test set has only {len(test_df)} obs (need >= 6)")

    return {
        "train_obs": len(train_df),
        "train_range": (train_df.index.min(), train_df.index.max()) if len(train_df) > 0 else (None, None),
        "valid_obs": len(valid_df),
        "valid_range": (valid_df.index.min(), valid_df.index.max()) if len(valid_df) > 0 else (None, None),
        "test_obs": len(test_df),
        "test_range": (test_df.index.min(), test_df.index.max()) if len(test_df) > 0 else (None, None),
        "warnings": warnings,
        "valid_split": len(warnings) == 0,
    }
