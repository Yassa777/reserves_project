#!/usr/bin/env python3
"""
Run Dynamic Model Averaging (DMA) and Dynamic Model Selection (DMS).

This script implements Phase 3 of the academic forecasting pipeline:
1. Collect forecasts from all available models (baseline + academic)
2. Align forecasts to common dates
3. Grid search for optimal forgetting factor (alpha)
4. Generate DMA and DMS combined forecasts
5. Compute performance metrics and comparisons
6. Generate publication-quality visualizations
7. Save all outputs

Usage:
    python run_dma_dms.py [--alphas 0.90,0.95,0.99,1.00] [--skip-plots]

Author: Academic Pipeline
Date: 2026-02-10
Specification: 09_DMA_DMS_SPEC.md

References:
- Raftery, A.E., Karny, M., & Ettler, P. (2010). Online Prediction Under
  Model Uncertainty via Dynamic Model Averaging. Technometrics.
- Koop, G. & Korobilis, D. (2012). Forecasting Inflation Using Dynamic
  Model Averaging. International Economic Review.
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Add project to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from reserves_project.scripts.academic.models.dma import (
    DynamicModelAveraging,
    DMAResults,
    run_dma_grid_search,
    rolling_dma_backtest,
    compute_dma_metrics,
)
from reserves_project.scripts.academic.models.dma_visualization import (
    plot_dma_weights_stacked,
    plot_dms_selection_path,
    plot_weight_evolution_by_model,
    plot_alpha_sensitivity,
    plot_dma_vs_individual,
    plot_selection_frequency,
    plot_weight_heatmap,
    plot_performance_comparison,
    create_dma_report_figures,
)

# =============================================================================
# Configuration
# =============================================================================

# Target variable
TARGET_VAR = "gross_reserves_usd_m"

# Train/Validation/Test Split
TRAIN_END = "2019-12-01"
VALID_END = "2022-12-01"

# Forgetting factor grid
ALPHA_GRID = [0.90, 0.95, 0.99, 1.00]

# Output directories
DATA_DIR = PROJECT_ROOT / "data"
BASELINE_RESULTS_DIR = DATA_DIR / "forecast_results"
ACADEMIC_RESULTS_DIR = DATA_DIR / "forecast_results_academic"
OUTPUT_DIR = ACADEMIC_RESULTS_DIR / "dma"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Crisis periods for visualization
CRISIS_PERIODS = [
    ("2018-01-01", "2019-06-01"),  # Currency crisis
    ("2021-07-01", "2022-09-01"),  # Debt default crisis
]

# Warmup periods
WARMUP_PERIODS = 12


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_baseline_forecasts() -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Load forecasts from baseline models (ARIMA, VECM, MS-VAR, MS-VECM, Naive).

    Returns
    -------
    df : pd.DataFrame
        Long-format dataframe with all forecasts
    forecasts : dict
        {model_name: forecast_array} aligned to common dates
    """
    backtests_path = BASELINE_RESULTS_DIR / "rolling_backtests.csv"

    if not backtests_path.exists():
        raise FileNotFoundError(f"Baseline backtests not found: {backtests_path}")

    df = pd.read_csv(backtests_path, parse_dates=['date'])

    # Get unique models
    models = df['model'].unique()
    print(f"  Found {len(models)} baseline models: {list(models)}")

    # Pivot to wide format
    df_wide = df.pivot(index='date', columns='model', values='forecast')
    df_wide['actual'] = df.groupby('date')['actual'].first()

    # Create forecast dictionary
    forecasts = {}
    for model in models:
        if model in df_wide.columns:
            forecasts[model] = df_wide[model].values

    return df_wide, forecasts


def load_bvar_forecasts() -> Dict[str, np.ndarray]:
    """
    Load forecasts from BVAR models (multiple variable sets).

    Returns
    -------
    forecasts : dict
        {model_name: forecast_array}
    """
    forecasts = {}
    bvar_dir = ACADEMIC_RESULTS_DIR / "bvar"

    if not bvar_dir.exists():
        print("  BVAR directory not found, skipping")
        return forecasts

    # Look for rolling backtest files
    for varset in ['parsimonious', 'bop', 'monetary', 'pca', 'full']:
        backtest_file = bvar_dir / f"bvar_rolling_backtest_{varset}.csv"

        if not backtest_file.exists():
            continue

        df = pd.read_csv(backtest_file)

        # Filter for horizon=1 (one-step-ahead)
        if 'horizon' in df.columns:
            df_h1 = df[df['horizon'] == 1].copy()
        else:
            df_h1 = df.copy()

        if df_h1.empty:
            continue

        # Convert date columns
        date_col = 'forecast_date' if 'forecast_date' in df_h1.columns else 'date'
        if date_col in df_h1.columns:
            df_h1['date'] = pd.to_datetime(df_h1[date_col])

        model_name = f"BVAR_{varset[:3]}"  # e.g., BVAR_par, BVAR_bop

        # Get forecast column
        fc_col = 'forecast_point' if 'forecast_point' in df_h1.columns else 'forecast'
        if fc_col in df_h1.columns:
            forecasts[model_name] = df_h1.set_index('date')[fc_col]
            print(f"    Loaded {model_name}: {len(df_h1)} obs")

    return forecasts


def load_combination_forecasts() -> Dict[str, np.ndarray]:
    """
    Load forecasts from combination methods.

    Returns
    -------
    forecasts : dict
        {model_name: forecast_array}
    """
    forecasts = {}
    comb_dir = ACADEMIC_RESULTS_DIR / "combinations"

    if not comb_dir.exists():
        print("  Combinations directory not found, skipping")
        return forecasts

    # Load combination forecasts
    comb_file = comb_dir / "combination_forecasts.csv"

    if not comb_file.exists():
        return forecasts

    df = pd.read_csv(comb_file, parse_dates=['date'], index_col='date')

    # Map column names to cleaner model names
    name_mapping = {
        'combined_equal': 'EqualWeight',
        'combined_mse': 'MSE-Weight',
        'combined_gr_convex': 'GR-Convex',
        'combined_trimmed': 'TrimmedMean',
        'combined_median': 'Median',
    }

    for col, name in name_mapping.items():
        if col in df.columns:
            forecasts[name] = df[col]
            print(f"    Loaded {name}: {len(df)} obs")

    return forecasts


def load_tvar_forecasts() -> Dict[str, np.ndarray]:
    """
    Load forecasts from TVAR models.

    Returns
    -------
    forecasts : dict
        {model_name: forecast_array}
    """
    forecasts = {}
    tvar_dir = ACADEMIC_RESULTS_DIR / "tvar"

    if not tvar_dir.exists():
        print("  TVAR directory not found, skipping")
        return forecasts

    for varset in ['parsimonious', 'bop', 'monetary']:
        backtest_file = tvar_dir / f"tvar_rolling_backtest_{varset}.csv"

        if not backtest_file.exists():
            continue

        df = pd.read_csv(backtest_file)

        # Filter for horizon=1
        if 'horizon' in df.columns:
            df_h1 = df[df['horizon'] == 1].copy()
        else:
            df_h1 = df.copy()

        if df_h1.empty:
            continue

        date_col = 'forecast_date' if 'forecast_date' in df_h1.columns else 'date'
        if date_col in df_h1.columns:
            df_h1['date'] = pd.to_datetime(df_h1[date_col])

        model_name = f"TVAR_{varset[:3]}"
        forecasts[model_name] = df_h1.set_index('date')['forecast']
        print(f"    Loaded {model_name}: {len(df_h1)} obs")

    return forecasts


def load_favar_forecasts() -> Dict[str, np.ndarray]:
    """
    Load forecasts from FAVAR model.

    Returns
    -------
    forecasts : dict
        {model_name: forecast_array}
    """
    forecasts = {}
    favar_dir = ACADEMIC_RESULTS_DIR / "favar"

    if not favar_dir.exists():
        print("  FAVAR directory not found, skipping")
        return forecasts

    # Load h=1 backtest
    backtest_file = favar_dir / "favar_rolling_backtest_h1.csv"

    if not backtest_file.exists():
        return forecasts

    df = pd.read_csv(backtest_file, parse_dates=['date'], index_col='date')

    if 'forecast' in df.columns:
        forecasts['FAVAR'] = df['forecast']
        print(f"    Loaded FAVAR: {len(df)} obs")

    return forecasts


def load_midas_forecasts() -> Dict[str, np.ndarray]:
    """
    Load forecasts from MIDAS models.

    Returns
    -------
    forecasts : dict
        {model_name: forecast_array}
    """
    forecasts = {}
    midas_dir = ACADEMIC_RESULTS_DIR / "midas"

    if not midas_dir.exists():
        print("  MIDAS directory not found, skipping")
        return forecasts

    backtest_file = midas_dir / "midas_rolling_backtest.csv"

    if not backtest_file.exists():
        return forecasts

    df = pd.read_csv(backtest_file)

    # Filter for horizon=1 and specific model
    if 'horizon' in df.columns:
        df_h1 = df[df['horizon'] == 1].copy()
    else:
        df_h1 = df.copy()

    # Get unique models
    if 'model' in df_h1.columns:
        for model in df_h1['model'].unique():
            df_model = df_h1[df_h1['model'] == model].copy()
            df_model['date'] = pd.to_datetime(df_model['origin'])

            # Simplify model name
            if 'exp_almon' in model:
                name = 'MIDAS_almon'
            elif 'beta' in model:
                name = 'MIDAS_beta'
            else:
                name = model.replace('MIDAS_AR_', 'MIDAS_')

            forecasts[name] = df_model.set_index('date')['forecast']
            print(f"    Loaded {name}: {len(df_model)} obs")

    return forecasts


def align_forecasts(
    base_df: pd.DataFrame,
    additional_forecasts: Dict[str, pd.Series]
) -> Tuple[pd.DatetimeIndex, np.ndarray, Dict[str, np.ndarray]]:
    """
    Align all forecasts to common dates.

    Parameters
    ----------
    base_df : pd.DataFrame
        Base dataframe with dates as index and 'actual' column
    additional_forecasts : dict
        {model_name: pd.Series} with date index

    Returns
    -------
    dates : pd.DatetimeIndex
        Common dates
    actuals : np.ndarray
        Actual values
    forecasts : dict
        {model_name: np.ndarray} aligned to dates
    """
    dates = base_df.index
    actuals = base_df['actual'].values

    forecasts = {}

    # Add baseline model forecasts from base_df
    for col in base_df.columns:
        if col != 'actual':
            forecasts[col] = base_df[col].values

    # Add additional forecasts, aligned to dates
    for name, series in additional_forecasts.items():
        if isinstance(series, pd.Series):
            # Reindex to base dates
            aligned = series.reindex(dates)
            forecasts[name] = aligned.values
        else:
            # Assume already aligned
            if len(series) == len(dates):
                forecasts[name] = series

    # Report coverage
    print("\n  Forecast coverage by model:")
    for name, fc in forecasts.items():
        valid = ~np.isnan(fc)
        print(f"    {name}: {valid.sum()}/{len(fc)} ({valid.mean()*100:.1f}%)")

    return dates, actuals, forecasts


def load_all_forecasts() -> Tuple[pd.DatetimeIndex, np.ndarray, Dict[str, np.ndarray]]:
    """
    Load and align all available forecasts.

    Returns
    -------
    dates : pd.DatetimeIndex
        Common date index
    actuals : np.ndarray
        Actual values
    model_forecasts : dict
        {model_name: forecast_array}
    """
    print("\nLoading forecasts...")

    # 1. Load baseline forecasts
    print("\n  Loading baseline models...")
    base_df, base_forecasts = load_baseline_forecasts()

    # 2. Load academic model forecasts
    additional = {}

    print("\n  Loading BVAR models...")
    additional.update(load_bvar_forecasts())

    print("\n  Loading combination forecasts...")
    additional.update(load_combination_forecasts())

    print("\n  Loading TVAR models...")
    additional.update(load_tvar_forecasts())

    print("\n  Loading FAVAR...")
    additional.update(load_favar_forecasts())

    print("\n  Loading MIDAS models...")
    additional.update(load_midas_forecasts())

    # 3. Align all forecasts to common dates
    print("\n  Aligning forecasts to common dates...")
    dates, actuals, forecasts = align_forecasts(base_df, additional)

    print(f"\n  Total: {len(forecasts)} models, {len(dates)} time periods")
    print(f"  Date range: {dates.min().strftime('%Y-%m')} to {dates.max().strftime('%Y-%m')}")

    return dates, actuals, forecasts


# =============================================================================
# Main DMA/DMS Functions
# =============================================================================

def run_dma_analysis(
    dates: pd.DatetimeIndex,
    actuals: np.ndarray,
    model_forecasts: Dict[str, np.ndarray],
    alphas: List[float] = ALPHA_GRID
) -> Dict:
    """
    Run complete DMA/DMS analysis.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Date index
    actuals : np.ndarray
        Actual values
    model_forecasts : dict
        Model forecasts
    alphas : list
        Forgetting factor grid

    Returns
    -------
    results : dict
        Complete DMA/DMS results
    """
    print("\n" + "="*70)
    print("Running DMA/DMS Analysis")
    print("="*70)

    # Create split masks
    train_mask = dates <= pd.Timestamp(TRAIN_END)
    valid_mask = (dates > pd.Timestamp(TRAIN_END)) & (dates <= pd.Timestamp(VALID_END))
    test_mask = dates > pd.Timestamp(VALID_END)

    print(f"\n  Train: {train_mask.sum()} obs (ends {TRAIN_END})")
    print(f"  Validation: {valid_mask.sum()} obs (ends {VALID_END})")
    print(f"  Test: {test_mask.sum()} obs")

    # 1. Grid search for optimal alpha
    print("\n  Running alpha grid search...")
    best_alpha, alpha_results = run_dma_grid_search(
        model_forecasts=model_forecasts,
        actuals=actuals,
        alphas=alphas,
        validation_mask=valid_mask,
        warmup_periods=WARMUP_PERIODS
    )
    print(f"\n  Alpha grid search results:")
    print(alpha_results.to_string(index=False))
    print(f"\n  Optimal alpha: {best_alpha}")

    # 2. Run DMA with optimal alpha
    print(f"\n  Running DMA with alpha={best_alpha}...")
    dma = DynamicModelAveraging(
        alpha=best_alpha,
        method="dma",
        warmup_periods=WARMUP_PERIODS
    )
    dma_forecasts, dma_weights = dma.fit_predict(
        model_forecasts=model_forecasts,
        actuals=actuals,
        dates=dates
    )
    dma_results = dma.results_

    # 3. Run DMS with optimal alpha
    print(f"\n  Running DMS with alpha={best_alpha}...")
    dms = DynamicModelAveraging(
        alpha=best_alpha,
        method="dms",
        warmup_periods=WARMUP_PERIODS
    )
    dms_forecasts, dms_weights = dms.fit_predict(
        model_forecasts=model_forecasts,
        actuals=actuals,
        dates=dates
    )
    dms_results = dms.results_

    # 4. Compute performance metrics
    print("\n  Computing performance metrics...")

    # Validation period metrics
    valid_metrics_dma = compute_dma_metrics(
        dma_forecasts, actuals, model_forecasts, mask=valid_mask
    )
    valid_metrics_dma['split'] = 'validation'

    # Test period metrics
    test_metrics_dma = compute_dma_metrics(
        dma_forecasts, actuals, model_forecasts, mask=test_mask
    )
    test_metrics_dma['split'] = 'test'

    # Add DMS metrics
    valid_metrics_dms = compute_dma_metrics(
        dms_forecasts, actuals, model_forecasts, mask=valid_mask
    )
    valid_metrics_dms.loc[valid_metrics_dms['model'] == 'DMA', 'model'] = 'DMS'
    valid_metrics_dms['split'] = 'validation'

    test_metrics_dms = compute_dma_metrics(
        dms_forecasts, actuals, model_forecasts, mask=test_mask
    )
    test_metrics_dms.loc[test_metrics_dms['model'] == 'DMA', 'model'] = 'DMS'
    test_metrics_dms['split'] = 'test'

    # Combine all metrics
    all_metrics = pd.concat([
        valid_metrics_dma,
        test_metrics_dma,
        valid_metrics_dms[valid_metrics_dms['model'] == 'DMS'],
        test_metrics_dms[test_metrics_dms['model'] == 'DMS']
    ], ignore_index=True)

    # 5. Weight summaries
    print("\n  Computing weight summaries...")
    weight_summary = dma_results.get_weight_summary()

    # Selection frequency by split
    valid_sel_freq = dms_results.get_selection_frequency(valid_mask)
    test_sel_freq = dms_results.get_selection_frequency(test_mask)

    print("\n  DMS Selection Frequency (Validation):")
    for m, f in valid_sel_freq.items():
        print(f"    {m}: {f*100:.1f}%")

    print("\n  DMS Selection Frequency (Test):")
    for m, f in test_sel_freq.items():
        print(f"    {m}: {f*100:.1f}%")

    # 6. Rolling backtest
    print("\n  Running rolling backtest...")
    backtest_results, backtest_summary = rolling_dma_backtest(
        model_forecasts=model_forecasts,
        actuals=actuals,
        dates=dates,
        train_end=TRAIN_END,
        valid_end=VALID_END,
        alpha=best_alpha
    )

    print("\n  Backtest Summary:")
    print(backtest_summary.to_string(index=False))

    return {
        'dates': dates,
        'actuals': actuals,
        'model_forecasts': model_forecasts,
        'dma_results': dma_results,
        'dms_results': dms_results,
        'dma_forecasts': dma_forecasts,
        'dms_forecasts': dms_forecasts,
        'alpha_search': alpha_results,
        'best_alpha': best_alpha,
        'all_metrics': all_metrics,
        'weight_summary': weight_summary,
        'valid_sel_freq': valid_sel_freq,
        'test_sel_freq': test_sel_freq,
        'backtest_results': backtest_results,
        'backtest_summary': backtest_summary,
        'train_mask': train_mask,
        'valid_mask': valid_mask,
        'test_mask': test_mask,
    }


def save_results(results: Dict, output_dir: Path, skip_plots: bool = False):
    """
    Save all DMA/DMS results.

    Parameters
    ----------
    results : dict
        Results from run_dma_analysis
    output_dir : Path
        Output directory
    skip_plots : bool
        Whether to skip generating plots
    """
    print("\n" + "="*70)
    print("Saving Results")
    print("="*70)

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    dates = results['dates']
    dma_results = results['dma_results']
    dms_results = results['dms_results']

    # 1. Save forecasts
    print("\n  Saving forecasts...")

    # DMA forecasts
    dma_fc_df = pd.DataFrame({
        'date': dates,
        'actual': results['actuals'],
        'dma_forecast': results['dma_forecasts']
    })
    dma_fc_df.to_csv(output_dir / "dma_forecasts.csv", index=False)

    # DMS forecasts
    dms_fc_df = pd.DataFrame({
        'date': dates,
        'actual': results['actuals'],
        'dms_forecast': results['dms_forecasts']
    })
    dms_fc_df.to_csv(output_dir / "dms_forecasts.csv", index=False)

    # 2. Save weights
    print("  Saving weights...")
    weights_df = dma_results.get_weight_df()
    weights_df.to_csv(output_dir / "dma_weights.csv")

    # 3. Save alpha selection results
    print("  Saving alpha selection...")
    results['alpha_search'].to_csv(output_dir / "alpha_grid_search.csv", index=False)

    alpha_selection = {
        'optimal_alpha': float(results['best_alpha']),
        'grid_searched': ALPHA_GRID,
        'timestamp': datetime.now().isoformat(),
    }
    with open(output_dir / "dma_alpha_selection.json", 'w') as f:
        json.dump(alpha_selection, f, indent=2)

    # 4. Save weight summary
    print("  Saving weight summary...")
    results['weight_summary'].to_csv(output_dir / "dma_weight_summary.csv", index=False)

    weight_summary_dict = results['weight_summary'].set_index('model').to_dict('index')
    with open(output_dir / "dma_weight_summary.json", 'w') as f:
        json.dump(weight_summary_dict, f, indent=2, default=float)

    # 5. Save selection frequency
    print("  Saving selection frequency...")
    sel_freq = {
        'validation': results['valid_sel_freq'].to_dict(),
        'test': results['test_sel_freq'].to_dict(),
    }
    with open(output_dir / "dms_selection_frequency.json", 'w') as f:
        json.dump(sel_freq, f, indent=2, default=float)

    # 6. Save performance metrics
    print("  Saving performance metrics...")
    results['all_metrics'].to_csv(output_dir / "dma_performance_metrics.csv", index=False)

    # 7. Save backtest results
    print("  Saving backtest results...")
    results['backtest_results'].to_csv(output_dir / "dma_rolling_backtest.csv")
    results['backtest_summary'].to_csv(output_dir / "dma_backtest_summary.csv", index=False)

    # 8. Generate and save figures
    if not skip_plots:
        print("\n  Generating figures...")

        try:
            # Weight evolution (stacked)
            fig_path = figures_dir / "dma_weight_evolution.png"
            plot_dma_weights_stacked(
                dma_results.weights_history,
                dma_results.model_names,
                dates,
                crisis_periods=CRISIS_PERIODS,
                save_path=str(fig_path)
            )
            import matplotlib.pyplot as plt
            plt.close()
            print(f"    Saved: {fig_path.name}")

            # DMS selection path
            fig_path = figures_dir / "dms_selection_path.png"
            plot_dms_selection_path(
                dms_results.weights_history,
                dms_results.model_names,
                dates,
                save_path=str(fig_path)
            )
            plt.close()
            print(f"    Saved: {fig_path.name}")

            # Weight by model
            fig_path = figures_dir / "dma_weight_by_model.png"
            plot_weight_evolution_by_model(
                dma_results.weights_history,
                dma_results.model_names,
                dates,
                save_path=str(fig_path)
            )
            plt.close()
            print(f"    Saved: {fig_path.name}")

            # Alpha sensitivity
            fig_path = figures_dir / "alpha_sensitivity.png"
            plot_alpha_sensitivity(
                results['alpha_search'],
                save_path=str(fig_path)
            )
            plt.close()
            print(f"    Saved: {fig_path.name}")

            # DMA vs individual
            fig_path = figures_dir / "dma_vs_individual.png"
            plot_dma_vs_individual(
                results['dma_forecasts'],
                results['model_forecasts'],
                results['actuals'],
                dates,
                save_path=str(fig_path)
            )
            plt.close()
            print(f"    Saved: {fig_path.name}")

            # Selection frequency
            fig_path = figures_dir / "dms_selection_frequency.png"
            overall_freq = dms_results.get_selection_frequency()
            plot_selection_frequency(
                overall_freq,
                save_path=str(fig_path)
            )
            plt.close()
            print(f"    Saved: {fig_path.name}")

            # Weight heatmap
            fig_path = figures_dir / "dma_weight_heatmap.png"
            plot_weight_heatmap(
                dma_results.weights_history,
                dma_results.model_names,
                dates,
                save_path=str(fig_path)
            )
            plt.close()
            print(f"    Saved: {fig_path.name}")

            # Performance comparison (validation)
            fig_path = figures_dir / "performance_comparison_validation.png"
            valid_metrics = results['all_metrics'][
                results['all_metrics']['split'] == 'validation'
            ]
            plot_performance_comparison(
                valid_metrics,
                metric='rmse',
                title='Validation Period: RMSE Comparison',
                save_path=str(fig_path)
            )
            plt.close()
            print(f"    Saved: {fig_path.name}")

            # Performance comparison (test)
            fig_path = figures_dir / "performance_comparison_test.png"
            test_metrics = results['all_metrics'][
                results['all_metrics']['split'] == 'test'
            ]
            plot_performance_comparison(
                test_metrics,
                metric='rmse',
                title='Test Period: RMSE Comparison',
                save_path=str(fig_path)
            )
            plt.close()
            print(f"    Saved: {fig_path.name}")

        except Exception as e:
            print(f"  Warning: Error generating plots: {e}")

    print(f"\n  All results saved to: {output_dir}")


def print_summary(results: Dict):
    """Print summary of DMA/DMS results."""
    print("\n" + "="*70)
    print("DMA/DMS Summary")
    print("="*70)

    print(f"\n  Models in pool: {len(results['dma_results'].model_names)}")
    for m in results['dma_results'].model_names:
        print(f"    - {m}")

    print(f"\n  Optimal forgetting factor: {results['best_alpha']}")

    print("\n  Weight Summary (mean weight by model):")
    weight_summary = results['weight_summary'].sort_values('mean_weight', ascending=False)
    for _, row in weight_summary.iterrows():
        print(f"    {row['model']}: {row['mean_weight']:.3f} (std: {row['std_weight']:.3f})")

    # Performance comparison
    print("\n  Performance Comparison:")
    print("\n  Validation Period:")
    valid_metrics = results['all_metrics'][
        results['all_metrics']['split'] == 'validation'
    ].sort_values('rmse')
    for _, row in valid_metrics.head(10).iterrows():
        print(f"    {row['model']}: RMSE={row['rmse']:.1f}, MAE={row['mae']:.1f}")

    print("\n  Test Period:")
    test_metrics = results['all_metrics'][
        results['all_metrics']['split'] == 'test'
    ].sort_values('rmse')
    for _, row in test_metrics.head(10).iterrows():
        print(f"    {row['model']}: RMSE={row['rmse']:.1f}, MAE={row['mae']:.1f}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run Dynamic Model Averaging (DMA) and Dynamic Model Selection (DMS)"
    )
    parser.add_argument(
        '--alphas',
        type=str,
        default=','.join(map(str, ALPHA_GRID)),
        help="Comma-separated list of alpha values to search"
    )
    parser.add_argument(
        '--skip-plots',
        action='store_true',
        help="Skip generating plots"
    )
    args = parser.parse_args()

    # Parse alphas
    alphas = [float(a) for a in args.alphas.split(',')]

    print("="*70)
    print("Dynamic Model Averaging / Dynamic Model Selection")
    print("Specification 09: DMA/DMS")
    print("="*70)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Alpha grid: {alphas}")
    print(f"Output: {OUTPUT_DIR}")

    # Load all forecasts
    dates, actuals, model_forecasts = load_all_forecasts()

    # Filter models with sufficient coverage
    min_coverage = 0.5
    valid_models = {}
    for name, fc in model_forecasts.items():
        coverage = (~np.isnan(fc)).mean()
        if coverage >= min_coverage:
            valid_models[name] = fc
        else:
            print(f"  Excluding {name}: {coverage*100:.1f}% coverage < {min_coverage*100:.0f}%")

    print(f"\n  Using {len(valid_models)} models with >{min_coverage*100:.0f}% coverage")

    # Run DMA/DMS analysis
    results = run_dma_analysis(
        dates=dates,
        actuals=actuals,
        model_forecasts=valid_models,
        alphas=alphas
    )

    # Save results
    save_results(results, OUTPUT_DIR, skip_plots=args.skip_plots)

    # Print summary
    print_summary(results)

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)


if __name__ == "__main__":
    main()
