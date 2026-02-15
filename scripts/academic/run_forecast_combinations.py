#!/usr/bin/env python3
"""
Run Forecast Combinations

This script loads forecasts from all available models and applies
various combination methods. Implements:
1. Static combinations (estimated on training data)
2. Rolling combinations (re-estimated periodically)
3. Performance evaluation vs individual models

Outputs saved to data/forecast_results_academic/combinations/

References:
- Bates & Granger (1969): The Combination of Forecasts
- Granger & Ramanathan (1984): Improved Methods of Combining Forecasts
- Timmermann (2006): Forecast Combinations (Handbook of Economic Forecasting)
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from reserves_project.scripts.academic.models.forecast_combiner import (
    ForecastCombiner,
    rolling_combination_backtest,
    compare_all_methods
)
from reserves_project.scripts.academic.models.combination_analysis import (
    compute_forecast_metrics,
    compute_combination_diagnostics,
    analyze_weight_stability,
    create_summary_report
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = PROJECT_ROOT / "data"
FORECAST_RESULTS_DIR = DATA_DIR / "forecast_results"
OUTPUT_DIR = DATA_DIR / "forecast_results_academic" / "combinations"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Combination methods to evaluate
COMBINATION_METHODS = [
    "equal",       # Simple average
    "mse",         # Inverse MSE weights
    "gr_none",     # Granger-Ramanathan unconstrained
    "gr_sum",      # GR with sum-to-one constraint
    "gr_convex",   # GR with convex constraint
    "trimmed",     # Trimmed mean (robust)
    "median"       # Median (robust)
]

# Model files to load
MODEL_FILES = {
    "ARIMA": "arima_forecast.csv",
    "VECM": "vecm_forecast.csv",
    "MS_VAR": "ms_var_forecast.csv",
    "MS_VECM": "ms_vecm_forecast.csv",
    "Naive": "naive_forecast.csv"
}


def load_model_forecasts() -> Tuple[Dict[str, pd.DataFrame], pd.Series]:
    """
    Load all model forecasts and align to common dates.

    Returns
    -------
    forecasts : dict
        {model_name: DataFrame with DatetimeIndex and 'forecast' column}
    actuals : pd.Series
        Actual values with DatetimeIndex
    """
    logger.info("Loading model forecasts...")

    forecasts = {}
    actuals = None

    for model_name, filename in MODEL_FILES.items():
        filepath = FORECAST_RESULTS_DIR / filename

        if not filepath.exists():
            logger.warning(f"Forecast file not found: {filepath}")
            continue

        df = pd.read_csv(filepath, parse_dates=['date'])
        df.set_index('date', inplace=True)
        df.index = pd.to_datetime(df.index)

        forecasts[model_name] = df[['forecast', 'split']].copy()

        # Extract actuals (should be same across all files)
        if actuals is None:
            actuals = df['actual'].copy()

        logger.info(f"  Loaded {model_name}: {len(df)} observations")

    if len(forecasts) == 0:
        raise ValueError("No forecast files found!")

    logger.info(f"Loaded {len(forecasts)} models")

    return forecasts, actuals


def align_forecasts(
    forecasts: Dict[str, pd.DataFrame],
    actuals: pd.Series
) -> Tuple[Dict[str, np.ndarray], np.ndarray, pd.DatetimeIndex]:
    """
    Align all forecasts to common dates.

    Returns
    -------
    aligned_forecasts : dict
        {model_name: forecast_array}
    aligned_actuals : np.ndarray
        Actual values
    common_dates : pd.DatetimeIndex
        Common date index
    """
    # Find common dates
    common_dates = actuals.index.copy()
    for name, df in forecasts.items():
        common_dates = common_dates.intersection(df.index)

    common_dates = common_dates.sort_values()
    logger.info(f"Common dates: {len(common_dates)} observations")
    logger.info(f"  Date range: {common_dates[0]} to {common_dates[-1]}")

    # Align data
    aligned_forecasts = {}
    for name, df in forecasts.items():
        aligned_forecasts[name] = df.loc[common_dates, 'forecast'].values

    aligned_actuals = actuals.loc[common_dates].values

    return aligned_forecasts, aligned_actuals, common_dates


def get_split_indices(
    forecasts: Dict[str, pd.DataFrame],
    common_dates: pd.DatetimeIndex
) -> Tuple[int, int, int]:
    """
    Get validation and test split indices.

    For forecast combinations:
    - Validation period is used for ESTIMATING combination weights
    - Test period is used for OUT-OF-SAMPLE evaluation

    Returns
    -------
    val_start_idx : int
        Start of validation period (index 0 if all data is validation/test)
    val_end_idx : int
        End of validation period (= start of test)
    test_end_idx : int
        End of test period (= len of data)
    """
    # Use first model's split information
    first_model = list(forecasts.keys())[0]
    split_info = forecasts[first_model].loc[common_dates, 'split']

    # Find where validation starts
    val_mask = split_info == 'validation'
    test_mask = split_info == 'test'

    if val_mask.any():
        val_start_idx = val_mask.argmax()
    else:
        val_start_idx = 0

    if test_mask.any():
        val_end_idx = test_mask.argmax()
    else:
        val_end_idx = len(common_dates)

    test_end_idx = len(common_dates)

    n_val = val_end_idx - val_start_idx
    n_test = test_end_idx - val_end_idx

    logger.info(f"Validation period: index {val_start_idx} to {val_end_idx - 1} ({n_val} obs)")
    logger.info(f"Test period: index {val_end_idx} to {test_end_idx - 1} ({n_test} obs)")

    return val_start_idx, val_end_idx, test_end_idx


def run_static_combinations(
    forecasts: Dict[str, np.ndarray],
    actuals: np.ndarray,
    train_end_idx: int
) -> Tuple[Dict[str, np.ndarray], Dict[str, pd.DataFrame]]:
    """
    Run all static combination methods.

    Returns
    -------
    combined_forecasts : dict
        {method: combined_forecast_array}
    weight_tables : dict
        {method: weight DataFrame}
    """
    logger.info("Running static combinations...")

    combined_forecasts = {}
    weight_tables = {}

    for method in COMBINATION_METHODS:
        logger.info(f"  Fitting {method}...")

        combiner = ForecastCombiner(combination_method=method)
        combiner.fit(forecasts, actuals, train_end_idx)
        combined = combiner.combine(forecasts)

        combined_forecasts[method] = combined
        weight_tables[method] = combiner.get_weights_table()

        if combiner.weights is not None:
            weight_str = ", ".join([
                f"{m}: {w:.3f}" for m, w in combiner.weights.items()
            ])
            logger.info(f"    Weights: {weight_str}")

    return combined_forecasts, weight_tables


def run_rolling_combinations(
    forecasts: Dict[str, pd.DataFrame],
    actuals: pd.Series,
    train_end: str,
    refit_interval: int = 12
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run rolling combination backtest.

    Returns
    -------
    results : pd.DataFrame
        Rolling backtest results
    weight_history : dict
        Weight evolution over time
    """
    logger.info(f"Running rolling combinations (refit every {refit_interval} months)...")

    results, weight_history = rolling_combination_backtest(
        models_forecasts=forecasts,
        actuals=actuals,
        methods=COMBINATION_METHODS,
        train_end=train_end,
        refit_interval=refit_interval
    )

    logger.info(f"  Generated {len(results)} observations")

    return results, weight_history


def compute_individual_metrics(
    forecasts: Dict[str, np.ndarray],
    actuals: np.ndarray,
    start_idx: int,
    end_idx: int
) -> pd.DataFrame:
    """
    Compute metrics for individual models.
    """
    results = []

    for name, fc in forecasts.items():
        eval_fc = fc[start_idx:end_idx]
        eval_actuals = actuals[start_idx:end_idx]

        metrics = compute_forecast_metrics(eval_fc, eval_actuals)
        metrics['model'] = name
        results.append(metrics)

    return pd.DataFrame(results)


def save_results(
    combined_forecasts: Dict[str, np.ndarray],
    weight_tables: Dict[str, pd.DataFrame],
    rolling_results: pd.DataFrame,
    weight_history: Dict,
    diagnostics_val: pd.DataFrame,
    diagnostics_test: pd.DataFrame,
    individual_metrics_val: pd.DataFrame,
    individual_metrics_test: pd.DataFrame,
    common_dates: pd.DatetimeIndex
):
    """
    Save all results to output directory.
    """
    logger.info("Saving results...")

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Save combined forecasts
    combined_df = pd.DataFrame(index=common_dates)
    for method, fc in combined_forecasts.items():
        combined_df[f'combined_{method}'] = fc
    combined_df.to_csv(OUTPUT_DIR / "combination_forecasts.csv")
    logger.info(f"  Saved combination_forecasts.csv")

    # 2. Save weight tables
    all_weights = []
    for method, df in weight_tables.items():
        df = df.copy()
        df['combination_method'] = method
        all_weights.append(df)

    weights_df = pd.concat(all_weights, ignore_index=True)
    weights_df.to_csv(OUTPUT_DIR / "combination_weights.csv", index=False)
    logger.info(f"  Saved combination_weights.csv")

    # 3. Save rolling backtest results
    rolling_results.to_csv(OUTPUT_DIR / "combination_rolling_backtest.csv")
    logger.info(f"  Saved combination_rolling_backtest.csv")

    # 4. Save weight history (for DMA prep)
    with open(OUTPUT_DIR / "weight_history.json", 'w') as f:
        # Convert numpy types for JSON serialization
        serializable_history = {}
        for method, history in weight_history.items():
            serializable_history[method] = []
            for entry in history:
                serializable_entry = {
                    'date': entry['date'],
                    'intercept': float(entry['intercept']) if entry.get('intercept') else 0.0
                }
                if entry.get('weights'):
                    serializable_entry['weights'] = {
                        k: float(v) for k, v in entry['weights'].items()
                    }
                serializable_history[method].append(serializable_entry)

        json.dump(serializable_history, f, indent=2)
    logger.info(f"  Saved weight_history.json")

    # 5. Save diagnostics
    diagnostics_val.to_csv(OUTPUT_DIR / "diagnostics_validation.csv", index=False)
    diagnostics_test.to_csv(OUTPUT_DIR / "diagnostics_test.csv", index=False)
    logger.info(f"  Saved diagnostics files")

    # 6. Save individual model metrics for comparison
    individual_metrics_val.to_csv(OUTPUT_DIR / "individual_metrics_validation.csv", index=False)
    individual_metrics_test.to_csv(OUTPUT_DIR / "individual_metrics_test.csv", index=False)
    logger.info(f"  Saved individual metrics files")

    # 7. Create and save summary JSON
    summary = create_combination_summary(
        diagnostics_val, diagnostics_test,
        individual_metrics_val, individual_metrics_test,
        weight_tables
    )

    with open(OUTPUT_DIR / "combination_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  Saved combination_summary.json")


def create_combination_summary(
    diagnostics_val: pd.DataFrame,
    diagnostics_test: pd.DataFrame,
    individual_val: pd.DataFrame,
    individual_test: pd.DataFrame,
    weight_tables: Dict[str, pd.DataFrame]
) -> Dict:
    """
    Create comprehensive summary dictionary.
    """
    # Best individual models
    best_individual_val = individual_val.loc[individual_val['rmse'].idxmin()]
    best_individual_test = individual_test.loc[individual_test['rmse'].idxmin()]

    # Best combination methods
    best_combo_val = diagnostics_val.loc[diagnostics_val['rmse'].idxmin()]
    best_combo_test = diagnostics_test.loc[diagnostics_test['rmse'].idxmin()]

    # Weight summaries
    weight_summary = {}
    for method, df in weight_tables.items():
        if 'weight' in df.columns and 'model' in df.columns:
            weight_summary[method] = {
                row['model']: float(row['weight'])
                for _, row in df.iterrows()
            }
            if 'intercept' in df.columns:
                weight_summary[method]['_intercept'] = float(df['intercept'].iloc[0])

    return {
        "timestamp": datetime.now().isoformat(),
        "n_models": len(individual_val),
        "n_combination_methods": len(COMBINATION_METHODS),
        "validation_period": {
            "best_individual_model": best_individual_val['model'],
            "best_individual_rmse": float(best_individual_val['rmse']),
            "best_combination_method": best_combo_val['method'],
            "best_combination_rmse": float(best_combo_val['rmse']),
            "improvement_pct": float(
                (best_individual_val['rmse'] - best_combo_val['rmse']) /
                best_individual_val['rmse'] * 100
            )
        },
        "test_period": {
            "best_individual_model": best_individual_test['model'],
            "best_individual_rmse": float(best_individual_test['rmse']),
            "best_combination_method": best_combo_test['method'],
            "best_combination_rmse": float(best_combo_test['rmse']),
            "improvement_pct": float(
                (best_individual_test['rmse'] - best_combo_test['rmse']) /
                best_individual_test['rmse'] * 100
            )
        },
        "combination_weights": weight_summary,
        "all_methods_validation": {
            row['method']: {
                'rmse': float(row['rmse']),
                'mae': float(row['mae']),
                'efficiency': float(row['efficiency']) if pd.notna(row['efficiency']) else None
            }
            for _, row in diagnostics_val.iterrows()
        },
        "all_methods_test": {
            row['method']: {
                'rmse': float(row['rmse']),
                'mae': float(row['mae']),
                'efficiency': float(row['efficiency']) if pd.notna(row['efficiency']) else None
            }
            for _, row in diagnostics_test.iterrows()
        }
    }


def main():
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("FORECAST COMBINATION ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")

    # Step 1: Load forecasts
    forecasts_df, actuals = load_model_forecasts()

    # Step 2: Align to common dates
    forecasts, actuals_aligned, common_dates = align_forecasts(forecasts_df, actuals)

    # Step 3: Get split indices
    # val_start_idx = start of validation (weight estimation period)
    # val_end_idx = end of validation = start of test (out-of-sample evaluation)
    # test_end_idx = end of test period
    val_start_idx, val_end_idx, test_end_idx = get_split_indices(forecasts_df, common_dates)

    # For combination weight estimation, we use the VALIDATION period
    # The validation period acts as our "training" data for combinations
    # since the individual model forecasts are already produced
    weight_estimation_end_idx = val_end_idx  # Use entire validation for weight estimation

    logger.info(f"Weight estimation period: index 0 to {weight_estimation_end_idx - 1}")

    # Step 4: Run static combinations
    # Weights are estimated on validation period, applied to test period
    combined_forecasts, weight_tables = run_static_combinations(
        forecasts, actuals_aligned, weight_estimation_end_idx
    )

    # Step 5: Run rolling combinations
    # Start rolling from end of validation period
    if val_end_idx > 0:
        train_end_date = str(common_dates[val_end_idx - 1].date())
    else:
        train_end_date = str(common_dates[0].date())

    rolling_results, weight_history = run_rolling_combinations(
        forecasts_df, actuals,
        train_end=train_end_date,
        refit_interval=12
    )

    # Step 6: Compute diagnostics
    logger.info("Computing diagnostics...")

    # Validation period diagnostics (in-sample for weight estimation)
    diagnostics_val = compute_combination_diagnostics(
        combined_forecasts=combined_forecasts,
        individual_forecasts=forecasts,
        actuals=actuals_aligned,
        start_idx=val_start_idx,
        end_idx=val_end_idx
    )

    # Test period diagnostics (true out-of-sample)
    diagnostics_test = compute_combination_diagnostics(
        combined_forecasts=combined_forecasts,
        individual_forecasts=forecasts,
        actuals=actuals_aligned,
        start_idx=val_end_idx,
        end_idx=test_end_idx
    )

    # Individual model metrics
    individual_val = compute_individual_metrics(
        forecasts, actuals_aligned, val_start_idx, val_end_idx
    )
    individual_test = compute_individual_metrics(
        forecasts, actuals_aligned, val_end_idx, test_end_idx
    )

    # Step 7: Print summary
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION PERIOD RESULTS")
    logger.info("=" * 60)

    logger.info("\nIndividual Models:")
    for _, row in individual_val.iterrows():
        logger.info(f"  {row['model']}: RMSE={row['rmse']:.2f}, MAE={row['mae']:.2f}")

    logger.info("\nCombination Methods:")
    for _, row in diagnostics_val.iterrows():
        eff_str = f", Efficiency={row['efficiency']:.3f}" if pd.notna(row['efficiency']) else ""
        logger.info(f"  {row['method']}: RMSE={row['rmse']:.2f}, MAE={row['mae']:.2f}{eff_str}")

    logger.info("\n" + "=" * 60)
    logger.info("TEST PERIOD RESULTS")
    logger.info("=" * 60)

    logger.info("\nIndividual Models:")
    for _, row in individual_test.iterrows():
        logger.info(f"  {row['model']}: RMSE={row['rmse']:.2f}, MAE={row['mae']:.2f}")

    logger.info("\nCombination Methods:")
    for _, row in diagnostics_test.iterrows():
        eff_str = f", Efficiency={row['efficiency']:.3f}" if pd.notna(row['efficiency']) else ""
        logger.info(f"  {row['method']}: RMSE={row['rmse']:.2f}, MAE={row['mae']:.2f}{eff_str}")

    # Step 8: Save results
    save_results(
        combined_forecasts=combined_forecasts,
        weight_tables=weight_tables,
        rolling_results=rolling_results,
        weight_history=weight_history,
        diagnostics_val=diagnostics_val,
        diagnostics_test=diagnostics_test,
        individual_metrics_val=individual_val,
        individual_metrics_test=individual_test,
        common_dates=common_dates
    )

    logger.info("\n" + "=" * 60)
    logger.info("EXECUTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output directory: {OUTPUT_DIR}")

    return {
        'combined_forecasts': combined_forecasts,
        'weight_tables': weight_tables,
        'diagnostics_val': diagnostics_val,
        'diagnostics_test': diagnostics_test
    }


if __name__ == "__main__":
    results = main()
