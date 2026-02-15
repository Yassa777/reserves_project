#!/usr/bin/env python3
"""
Prepare Variable Sets for Academic Reserves Forecasting Pipeline.

This script generates datasets for each theoretically-motivated variable set:
1. Parsimonious - Minimal economically-motivated set (3 vars)
2. BoP - Balance of Payments drivers (5 vars)
3. Monetary - Policy intervention channel (3 vars)
4. PCA - Data-driven dimensionality reduction (3 PCs)
5. Full - All available variables (benchmark for overfitting)

Reference: Specification 01 - Variable Sets Definition

Usage:
    python prepare_variable_sets.py [--varset NAME] [--verbose]

Options:
    --varset NAME    Process only specified variable set (default: all)
    --verbose        Enable verbose output
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from variable_sets.config import (
    TARGET_VAR,
    TRAIN_END,
    VALID_END,
    MISSING_STRATEGY,
    VARIABLE_SETS,
    VARSET_ORDER,
    SOURCE_DATA_PATH,
    SUPPLEMENTARY_DATA_PATH,
    HISTORICAL_FX_PATH,
    OUTPUT_DIR,
    DATA_DIR,
    get_varset,
    get_output_dir,
    get_all_required_vars,
)

from variable_sets.pca_builder import (
    build_pca_factors,
    interpret_loadings,
    generate_scree_data,
)

from variable_sets.validators import (
    validate_variable_set,
    validate_all_varsets,
    apply_missing_strategy,
    check_data_quality,
    validate_date_index,
    check_train_valid_test_split,
)


def _normalize_to_month_start(index: pd.Index) -> pd.DatetimeIndex:
    """Normalize DatetimeIndex to month start dates."""
    dt_index = pd.to_datetime(index)
    return dt_index.to_period("M").to_timestamp(how="start")


def load_source_data(verbose: bool = False) -> pd.DataFrame:
    """Load and prepare source data for processing.

    This function handles known data gaps by:
    1. Merging USD/LKR exchange rate from historical_fx.csv if missing
    2. Merging USD/LKR from slfsi_monthly_panel.csv as fallback
    3. Computing m2_usd_m from m2_lkr_m / usd_lkr if missing
    """
    if verbose:
        print(f"Loading source data from: {SOURCE_DATA_PATH}")

    if not SOURCE_DATA_PATH.exists():
        raise FileNotFoundError(f"Source data not found: {SOURCE_DATA_PATH}")

    df = pd.read_csv(SOURCE_DATA_PATH, parse_dates=["date"], index_col="date")
    df.index = _normalize_to_month_start(df.index)
    df = df.sort_index()

    if verbose:
        print(f"  Loaded {len(df)} observations, {len(df.columns)} columns")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")

    # Repair missing usd_lkr data
    if "usd_lkr" in df.columns and df["usd_lkr"].notna().sum() == 0:
        if verbose:
            print("  Repairing missing usd_lkr data...")

        # Try historical_fx first
        if HISTORICAL_FX_PATH.exists():
            fx = pd.read_csv(HISTORICAL_FX_PATH, parse_dates=["date"], index_col="date")
            fx.index = _normalize_to_month_start(fx.index)
            if "usd_lkr" in fx.columns:
                df["usd_lkr"] = df["usd_lkr"].combine_first(fx["usd_lkr"])
                if verbose:
                    print(f"    Merged {fx['usd_lkr'].notna().sum()} obs from historical_fx.csv")

        # Fall back to slfsi_monthly_panel
        if SUPPLEMENTARY_DATA_PATH.exists() and df["usd_lkr"].notna().sum() < len(df) * 0.5:
            supp = pd.read_csv(SUPPLEMENTARY_DATA_PATH, parse_dates=["date"], index_col="date")
            supp.index = _normalize_to_month_start(supp.index)
            if "usd_lkr" in supp.columns:
                df["usd_lkr"] = df["usd_lkr"].combine_first(supp["usd_lkr"])
                if verbose:
                    print(f"    Supplemented from slfsi_monthly_panel.csv")

        if verbose:
            print(f"    usd_lkr now has {df['usd_lkr'].notna().sum()} observations")

    # Compute m2_usd_m if missing and inputs available
    if "m2_usd_m" in df.columns and df["m2_usd_m"].notna().sum() == 0:
        if {"m2_lkr_m", "usd_lkr"}.issubset(df.columns):
            if verbose:
                print("  Computing m2_usd_m from m2_lkr_m / usd_lkr...")
            safe_rate = df["usd_lkr"].replace(0, np.nan)
            df["m2_usd_m"] = df["m2_lkr_m"] / safe_rate
            if verbose:
                print(f"    m2_usd_m now has {df['m2_usd_m'].notna().sum()} observations")

    return df


def prepare_arima_dataset(
    df: pd.DataFrame,
    varset: Dict[str, Any],
    verbose: bool = False
) -> pd.DataFrame:
    """Prepare dataset for ARIMA modeling (target + exogenous variables)."""
    target = varset["target"]
    exog_vars = varset.get("arima_exog", [])

    # Filter out PC columns if not PCA set (they will be added later)
    if varset["name"] != "pca":
        columns = [target] + [v for v in exog_vars if not v.startswith("PC")]
    else:
        # For PCA, we just need target - PCs will be joined
        columns = [target]

    available_cols = [c for c in columns if c in df.columns]
    result = df[available_cols].copy()

    if verbose:
        print(f"  ARIMA dataset: {len(available_cols)} columns")

    return result


def prepare_vecm_dataset(
    df: pd.DataFrame,
    varset: Dict[str, Any],
    verbose: bool = False
) -> pd.DataFrame:
    """Prepare dataset for VECM modeling (system of variables in levels)."""
    system_vars = varset.get("vecm_system", [])

    # Filter out PC columns if not PCA set
    if varset["name"] != "pca":
        columns = [v for v in system_vars if not v.startswith("PC")]
    else:
        columns = [varset["target"]]

    available_cols = [c for c in columns if c in df.columns]
    result = df[available_cols].copy()

    if verbose:
        print(f"  VECM dataset: {len(available_cols)} columns")

    return result


def prepare_var_dataset(
    df: pd.DataFrame,
    varset: Dict[str, Any],
    verbose: bool = False
) -> pd.DataFrame:
    """Prepare dataset for VAR modeling."""
    system_vars = varset.get("var_system", [])

    # Filter out PC columns if not PCA set
    if varset["name"] != "pca":
        columns = [v for v in system_vars if not v.startswith("PC")]
    else:
        columns = [varset["target"]]

    available_cols = [c for c in columns if c in df.columns]
    result = df[available_cols].copy()

    if verbose:
        print(f"  VAR dataset: {len(available_cols)} columns")

    return result


def process_pca_varset(
    df: pd.DataFrame,
    varset: Dict[str, Any],
    output_dir: Path,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Process PCA variable set with factor extraction.

    Returns metadata including PCA loadings and variance explained.
    """
    if verbose:
        print("\n  Building PCA factors...")

    source_vars = varset["source_vars"]
    n_components = varset["n_components"]

    # Build PCA factors
    factors_df, loadings, variance_explained, pca_metadata = build_pca_factors(
        df=df,
        source_vars=source_vars,
        n_components=n_components,
        train_end=TRAIN_END,
    )

    if verbose:
        print(f"    Extracted {n_components} components from {len(source_vars)} variables")
        print(f"    Cumulative variance explained: {sum(variance_explained)*100:.1f}%")

    # Save PCA loadings
    loadings_path = output_dir / "pca_loadings.csv"
    loadings.to_csv(loadings_path)

    # Save variance explained
    var_explained_df = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(n_components)],
        "variance_explained_pct": variance_explained * 100,
        "cumulative_pct": np.cumsum(variance_explained) * 100,
    })
    var_explained_path = output_dir / "pca_variance_explained.csv"
    var_explained_df.to_csv(var_explained_path, index=False)

    # Generate interpretation table
    interpretation_df = interpret_loadings(loadings, variance_explained)
    interpretation_path = output_dir / "pca_interpretation.csv"
    interpretation_df.to_csv(interpretation_path, index=False)

    # Generate scree data
    scree_data = generate_scree_data(df, source_vars, TRAIN_END)
    scree_path = output_dir / "pca_scree_data.csv"
    scree_data.to_csv(scree_path, index=False)

    if verbose:
        print("\n    PCA Loadings (top 3 per component):")
        for col in loadings.columns:
            top_loadings = loadings[col].abs().nlargest(3)
            loading_strs = [f"{idx}: {loadings.loc[idx, col]:.3f}" for idx in top_loadings.index]
            print(f"      {col}: {', '.join(loading_strs)}")

    # Merge factors with target
    target_df = df[[varset["target"]]].copy()
    combined = target_df.join(factors_df, how="inner")

    # Apply missing strategy
    combined, missing_stats = apply_missing_strategy(
        combined,
        combined.columns.tolist(),
        MISSING_STRATEGY
    )

    return {
        "factors_df": factors_df,
        "combined_df": combined,
        "loadings": loadings,
        "variance_explained": variance_explained.tolist(),
        "pca_metadata": pca_metadata,
        "interpretation": interpretation_df.to_dict("records"),
        "missing_stats": missing_stats,
    }


def process_standard_varset(
    df: pd.DataFrame,
    varset: Dict[str, Any],
    verbose: bool = False
) -> Dict[str, pd.DataFrame]:
    """Process a standard (non-PCA) variable set."""
    arima_df = prepare_arima_dataset(df, varset, verbose)
    vecm_df = prepare_vecm_dataset(df, varset, verbose)
    var_df = prepare_var_dataset(df, varset, verbose)

    # Apply missing strategy to each
    all_cols = list(set(arima_df.columns) | set(vecm_df.columns) | set(var_df.columns))
    combined = df[all_cols].copy()
    combined, missing_stats = apply_missing_strategy(combined, all_cols, MISSING_STRATEGY)

    # Re-extract after missing handling
    arima_df = combined[[c for c in arima_df.columns if c in combined.columns]]
    vecm_df = combined[[c for c in vecm_df.columns if c in combined.columns]]
    var_df = combined[[c for c in var_df.columns if c in combined.columns]]

    return {
        "arima_df": arima_df,
        "vecm_df": vecm_df,
        "var_df": var_df,
        "missing_stats": missing_stats,
    }


def save_varset_datasets(
    datasets: Dict[str, pd.DataFrame],
    output_dir: Path,
    varset: Dict[str, Any],
    is_pca: bool = False,
    verbose: bool = False
) -> None:
    """Save all datasets for a variable set."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_pca:
        # For PCA, save combined dataset with PCs
        combined_df = datasets["combined_df"]

        # ARIMA dataset: target + PCs
        arima_cols = [varset["target"]] + varset["arima_exog"]
        arima_df = combined_df[[c for c in arima_cols if c in combined_df.columns]]
        arima_df.to_csv(output_dir / "arima_dataset.csv")

        # VECM dataset: target + PCs
        vecm_cols = varset["vecm_system"]
        vecm_df = combined_df[[c for c in vecm_cols if c in combined_df.columns]]
        vecm_df.to_csv(output_dir / "vecm_levels.csv")

        # VAR dataset: target + PCs
        var_cols = varset["var_system"]
        var_df = combined_df[[c for c in var_cols if c in combined_df.columns]]
        var_df.to_csv(output_dir / "var_system.csv")

        if verbose:
            print(f"  Saved PCA datasets to: {output_dir}")

    else:
        # Standard variable sets
        datasets["arima_df"].to_csv(output_dir / "arima_dataset.csv")
        datasets["vecm_df"].to_csv(output_dir / "vecm_levels.csv")
        datasets["var_df"].to_csv(output_dir / "var_system.csv")

        if verbose:
            print(f"  Saved datasets to: {output_dir}")


def generate_metadata(
    varset: Dict[str, Any],
    datasets: Dict[str, Any],
    validation: Dict[str, Any],
    output_dir: Path,
    is_pca: bool = False
) -> Dict[str, Any]:
    """Generate and save metadata JSON for a variable set."""

    if is_pca:
        combined_df = datasets["combined_df"]
        n_obs = len(combined_df)
        date_range = (str(combined_df.index.min().date()), str(combined_df.index.max().date()))
        train_obs = len(combined_df[combined_df.index <= TRAIN_END])
        valid_obs = len(combined_df[(combined_df.index > TRAIN_END) & (combined_df.index <= VALID_END)])
        test_obs = len(combined_df[combined_df.index > VALID_END])
    else:
        arima_df = datasets["arima_df"]
        n_obs = len(arima_df)
        date_range = (str(arima_df.index.min().date()), str(arima_df.index.max().date()))
        train_obs = len(arima_df[arima_df.index <= TRAIN_END])
        valid_obs = len(arima_df[(arima_df.index > TRAIN_END) & (arima_df.index <= VALID_END)])
        test_obs = len(arima_df[arima_df.index > VALID_END])

    metadata = {
        "variable_set": varset["name"],
        "description": varset["description"],
        "economic_rationale": varset.get("economic_rationale", ""),
        "target": varset["target"],
        "arima_exog": varset.get("arima_exog", []),
        "vecm_system": varset.get("vecm_system", []),
        "var_system": varset.get("var_system", []),
        "n_observations": n_obs,
        "date_range": date_range,
        "train_obs": train_obs,
        "valid_obs": valid_obs,
        "test_obs": test_obs,
        "train_end": str(TRAIN_END.date()),
        "valid_end": str(VALID_END.date()),
        "missing_strategy": MISSING_STRATEGY,
        "missing_stats": datasets.get("missing_stats", {}),
        "created_at": datetime.now().isoformat(),
    }

    if is_pca:
        metadata["pca"] = {
            "source_vars": varset["source_vars"],
            "n_components": varset["n_components"],
            "variance_explained": datasets["variance_explained"],
            "cumulative_variance_explained": sum(datasets["variance_explained"]),
            "interpretation": datasets["interpretation"],
        }

    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return metadata


def generate_summary_table(all_metadata: Dict[str, Dict]) -> pd.DataFrame:
    """Generate summary table of all variable sets."""
    rows = []
    for name, meta in all_metadata.items():
        rows.append({
            "variable_set": name,
            "n_arima_exog": len(meta.get("arima_exog", [])),
            "n_vecm_vars": len(meta.get("vecm_system", [])),
            "n_var_vars": len(meta.get("var_system", [])),
            "total_obs": meta["n_observations"],
            "train_obs": meta["train_obs"],
            "valid_obs": meta["valid_obs"],
            "test_obs": meta["test_obs"],
            "date_start": meta["date_range"][0],
            "date_end": meta["date_range"][1],
        })

    return pd.DataFrame(rows)


def process_varset(
    df: pd.DataFrame,
    varset_name: str,
    verbose: bool = False
) -> Optional[Dict[str, Any]]:
    """Process a single variable set and return metadata."""
    varset = get_varset(varset_name)
    output_dir = get_output_dir(varset_name)

    if verbose:
        print(f"\nProcessing variable set: {varset_name}")
        print(f"  Description: {varset['description']}")

    # Validate variable set
    validation = validate_variable_set(df, varset)

    if not validation["valid"]:
        print(f"  WARNING: Variable set {varset_name} validation failed")
        if validation.get("missing_vars"):
            print(f"    Missing variables: {validation['missing_vars']}")
        if validation.get("error"):
            print(f"    Error: {validation['error']}")
        return None

    if verbose:
        print(f"  Validation: {validation['n_obs']} observations available")

    # Process based on type
    if varset_name == "pca":
        pca_result = process_pca_varset(df, varset, output_dir, verbose)
        save_varset_datasets(pca_result, output_dir, varset, is_pca=True, verbose=verbose)
        metadata = generate_metadata(varset, pca_result, validation, output_dir, is_pca=True)
    else:
        datasets = process_standard_varset(df, varset, verbose)
        save_varset_datasets(datasets, output_dir, varset, is_pca=False, verbose=verbose)
        metadata = generate_metadata(varset, datasets, validation, output_dir, is_pca=False)

    if verbose:
        print(f"  Final observations: {metadata['n_observations']}")
        print(f"    Train: {metadata['train_obs']}, Valid: {metadata['valid_obs']}, Test: {metadata['test_obs']}")

    return metadata


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Prepare variable sets for academic reserves forecasting"
    )
    parser.add_argument(
        "--varset",
        type=str,
        default=None,
        help="Process only specified variable set (default: all)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Academic Variable Set Preparation")
    print("=" * 70)
    print(f"Target variable: {TARGET_VAR}")
    print(f"Training period ends: {TRAIN_END.date()}")
    print(f"Validation period ends: {VALID_END.date()}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)

    # Load source data
    try:
        df = load_source_data(verbose=args.verbose)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Validate date index
    date_validation = validate_date_index(df)
    if not date_validation["valid"]:
        print(f"ERROR: Invalid date index - {date_validation.get('error')}")
        sys.exit(1)

    if args.verbose:
        print(f"\nDate index validation:")
        print(f"  Observations: {date_validation['n_observations']}")
        print(f"  Frequency: {date_validation['inferred_frequency']}")
        if date_validation["n_missing_months"] > 0:
            print(f"  Missing months: {date_validation['n_missing_months']}")

    # Check train/valid/test split
    split_check = check_train_valid_test_split(df)
    if args.verbose:
        print(f"\nTrain/Valid/Test split:")
        print(f"  Train: {split_check['train_obs']} obs ({split_check['train_range'][0]} to {split_check['train_range'][1]})")
        print(f"  Valid: {split_check['valid_obs']} obs ({split_check['valid_range'][0]} to {split_check['valid_range'][1]})")
        print(f"  Test:  {split_check['test_obs']} obs ({split_check['test_range'][0]} to {split_check['test_range'][1]})")
        if split_check["warnings"]:
            for warning in split_check["warnings"]:
                print(f"  WARNING: {warning}")

    # Validate all variable sets
    print("\nValidating variable sets...")
    validation_df = validate_all_varsets(df, VARIABLE_SETS)
    print(validation_df.to_string(index=False))

    # Determine which variable sets to process
    if args.varset:
        if args.varset not in VARIABLE_SETS:
            print(f"ERROR: Unknown variable set '{args.varset}'")
            print(f"Available: {list(VARIABLE_SETS.keys())}")
            sys.exit(1)
        varsets_to_process = [args.varset]
    else:
        varsets_to_process = VARSET_ORDER

    # Process variable sets
    print("\n" + "=" * 70)
    print("Processing Variable Sets")
    print("=" * 70)

    all_metadata = {}
    for varset_name in varsets_to_process:
        metadata = process_varset(df, varset_name, verbose=args.verbose)
        if metadata:
            all_metadata[varset_name] = metadata

    # Generate and save summary table
    if all_metadata:
        summary_df = generate_summary_table(all_metadata)
        summary_path = OUTPUT_DIR / "variable_set_summary.csv"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_path, index=False)

        print("\n" + "=" * 70)
        print("Variable Set Summary")
        print("=" * 70)
        print(summary_df.to_string(index=False))

        # Print PCA results if processed
        if "pca" in all_metadata and "pca" in all_metadata["pca"]:
            pca_info = all_metadata["pca"]["pca"]
            print("\n" + "-" * 70)
            print("PCA Results")
            print("-" * 70)
            print(f"Components: {pca_info['n_components']}")
            print(f"Cumulative variance explained: {pca_info['cumulative_variance_explained']*100:.1f}%")
            print("\nVariance by component:")
            for i, var_exp in enumerate(pca_info["variance_explained"]):
                print(f"  PC{i+1}: {var_exp*100:.1f}%")
            print("\nInterpretation:")
            for interp in pca_info["interpretation"]:
                print(f"  {interp['component']}: {interp['interpretation']} ({interp['variance_explained_pct']:.1f}%)")

    print("\n" + "=" * 70)
    print("Execution Complete")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Variable sets processed: {len(all_metadata)}")

    return all_metadata


if __name__ == "__main__":
    main()
