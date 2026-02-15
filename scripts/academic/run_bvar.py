#!/usr/bin/env python3
"""
Run Bayesian VAR with Minnesota Prior for Reserves Forecasting.

This script executes the complete BVAR pipeline:
1. Load prepared variable sets from Phase 1
2. Grid search for optimal hyperparameters (lambda1, lambda3, n_lags)
3. Fit final BVAR models with selected hyperparameters
4. Generate point and density forecasts
5. Run rolling backtests with 12-month refit interval
6. Save all outputs and diagnostics

Usage:
    python run_bvar.py [--varsets parsimonious,bop] [--skip-grid-search]

Author: Academic Pipeline
Date: 2026-02-10
Specification: 03_BVAR_SPEC.md
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

# Add project to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from reserves_project.scripts.academic.models.bvar import (
    BayesianVAR,
    grid_search_hyperparameters,
)
from reserves_project.scripts.academic.models.bvar_diagnostics import (
    create_diagnostic_report,
    diagnose_convergence,
)
from reserves_project.scripts.academic.variable_sets.config import (
    TARGET_VAR,
    TRAIN_END,
    VALID_END,
    VARIABLE_SETS,
    VARSET_ORDER,
    OUTPUT_DIR as VARSET_OUTPUT_DIR,
)

# =============================================================================
# Configuration
# =============================================================================

# Output directory for BVAR results
OUTPUT_DIR = PROJECT_ROOT / "data" / "forecast_results_academic" / "bvar"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Hyperparameter grid
LAMBDA1_GRID = [0.05, 0.1, 0.2, 0.5]
LAMBDA3_GRID = [1.0, 2.0]
N_LAGS_GRID = [1, 2, 3, 4]

# Gibbs sampler settings
N_DRAWS = 5000
N_BURN = 1000

# Rolling backtest settings
REFIT_INTERVAL = 12  # Refit every 12 months
FORECAST_HORIZONS = [1, 3, 6, 12]

# Random seed for reproducibility
RANDOM_STATE = 42

# =============================================================================
# Data Loading
# =============================================================================

def load_varset_data(varset_name: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load prepared VAR system data for a variable set.

    Parameters
    ----------
    varset_name : str
        Name of variable set (parsimonious, bop, monetary, pca, full)

    Returns
    -------
    df : pd.DataFrame
        VAR system data with DatetimeIndex
    var_names : list of str
        Variable names in order
    """
    data_path = VARSET_OUTPUT_DIR / f"varset_{varset_name}" / "var_system.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"VAR system data not found: {data_path}")

    df = pd.read_csv(data_path, parse_dates=['date'], index_col='date')
    var_names = df.columns.tolist()

    print(f"  Loaded {varset_name}: {len(df)} observations, {len(var_names)} variables")
    return df, var_names


def prepare_train_valid_test(
    df: pd.DataFrame,
    train_end: pd.Timestamp = TRAIN_END,
    valid_end: pd.Timestamp = VALID_END
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/validation/test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with DatetimeIndex
    train_end : pd.Timestamp
        End of training period
    valid_end : pd.Timestamp
        End of validation period

    Returns
    -------
    train_df, valid_df, test_df : tuple of DataFrames
    """
    train_df = df[df.index <= train_end]
    valid_df = df[(df.index > train_end) & (df.index <= valid_end)]
    test_df = df[df.index > valid_end]

    return train_df, valid_df, test_df


# =============================================================================
# Hyperparameter Selection
# =============================================================================

def run_hyperparameter_search(
    Y_train: np.ndarray,
    varset_name: str,
    target_idx: int = 0,
    verbose: bool = True
) -> Dict:
    """
    Run grid search for optimal BVAR hyperparameters.

    Uses time-series cross-validation on training data.

    Parameters
    ----------
    Y_train : np.ndarray
        Training data
    varset_name : str
        Variable set name for logging
    target_idx : int
        Index of target variable
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Best hyperparameters and search results
    """
    print(f"\n  Hyperparameter search for {varset_name}...")

    results = grid_search_hyperparameters(
        Y=Y_train,
        target_idx=target_idx,
        lambda1_grid=LAMBDA1_GRID,
        lambda3_grid=LAMBDA3_GRID,
        n_lags_grid=N_LAGS_GRID,
        n_folds=5,
        n_draws=1000,  # Reduced for CV speed
        n_burn=200,
        random_state=RANDOM_STATE,
        verbose=verbose,
    )

    print(f"\n  Best params: {results['best_params']}, CV RMSE: {results['best_rmse']:.4f}")
    return results


# =============================================================================
# Model Fitting
# =============================================================================

def fit_bvar_model(
    Y: np.ndarray,
    var_names: List[str],
    hyperparams: Dict,
    n_draws: int = N_DRAWS,
    n_burn: int = N_BURN,
    random_state: int = RANDOM_STATE
) -> BayesianVAR:
    """
    Fit BVAR model with specified hyperparameters.

    Parameters
    ----------
    Y : np.ndarray
        Data matrix (T x k)
    var_names : list of str
        Variable names
    hyperparams : dict
        Hyperparameters (n_lags, lambda1, lambda3)
    n_draws : int
        Number of posterior draws
    n_burn : int
        Burn-in period
    random_state : int
        Random seed

    Returns
    -------
    BayesianVAR
        Fitted model
    """
    model = BayesianVAR(
        n_lags=hyperparams['n_lags'],
        lambda1=hyperparams['lambda1'],
        lambda3=hyperparams['lambda3'],
        n_draws=n_draws,
        n_burn=n_burn,
        random_state=random_state,
    )
    model.fit(Y, var_names=var_names)
    return model


# =============================================================================
# Forecasting
# =============================================================================

def generate_forecasts(
    model: BayesianVAR,
    df_train: pd.DataFrame,
    h: int = 12,
    target_idx: int = 0
) -> Dict:
    """
    Generate point and density forecasts.

    Parameters
    ----------
    model : BayesianVAR
        Fitted model
    df_train : pd.DataFrame
        Training data (for date indexing)
    h : int
        Forecast horizon
    target_idx : int
        Target variable index

    Returns
    -------
    dict
        Point and density forecasts with dates
    """
    # Point forecast
    point_fcst = model.forecast_point(h=h)

    # Density forecast (includes percentiles)
    density_fcst = model.forecast(h=h, return_draws=False, include_shock=True)

    # Create forecast dates
    last_date = df_train.index[-1]
    forecast_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=h,
        freq='MS'
    )

    return {
        'dates': forecast_dates,
        'point': point_fcst[:, target_idx],
        'mean': density_fcst['mean'][:, target_idx],
        'median': density_fcst['median'][:, target_idx],
        'lower_2.5': density_fcst['lower_2.5'][:, target_idx],
        'lower_5': density_fcst['lower_5'][:, target_idx],
        'lower_10': density_fcst['lower_10'][:, target_idx],
        'lower_25': density_fcst['lower_25'][:, target_idx],
        'upper_75': density_fcst['upper_75'][:, target_idx],
        'upper_90': density_fcst['upper_90'][:, target_idx],
        'upper_95': density_fcst['upper_95'][:, target_idx],
        'upper_97.5': density_fcst['upper_97.5'][:, target_idx],
        'std': density_fcst['std'][:, target_idx],
    }


# =============================================================================
# Rolling Backtest
# =============================================================================

def run_rolling_backtest(
    df: pd.DataFrame,
    hyperparams: Dict,
    var_names: List[str],
    target_idx: int = 0,
    train_end: pd.Timestamp = TRAIN_END,
    valid_end: pd.Timestamp = VALID_END,
    refit_interval: int = REFIT_INTERVAL,
    horizons: List[int] = None,
    n_draws: int = 2000,
    n_burn: int = 500,
    random_state: int = RANDOM_STATE,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run rolling backtest with periodic refitting.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset
    hyperparams : dict
        BVAR hyperparameters
    var_names : list of str
        Variable names
    target_idx : int
        Target variable index
    train_end : pd.Timestamp
        Initial training end
    valid_end : pd.Timestamp
        Validation end (for split annotation)
    refit_interval : int
        Months between refits
    horizons : list of int
        Forecast horizons to evaluate
    n_draws : int
        Posterior draws (reduced for speed)
    n_burn : int
        Burn-in draws
    random_state : int
        Random seed
    verbose : bool
        Print progress

    Returns
    -------
    pd.DataFrame
        Backtest results with forecasts and errors
    """
    if horizons is None:
        horizons = FORECAST_HORIZONS

    results = []
    target_name = var_names[target_idx]

    # Get all forecast dates (validation + test period)
    all_dates = df[df.index > train_end].index

    # Track when to refit
    last_fit_date = None
    model = None

    for i, forecast_origin in enumerate(all_dates):
        # Determine split
        split = 'validation' if forecast_origin <= valid_end else 'test'

        # Check if we need to refit
        need_refit = (
            model is None or
            last_fit_date is None or
            (forecast_origin - last_fit_date).days > refit_interval * 30
        )

        if need_refit:
            if verbose:
                print(f"  Refitting at {forecast_origin.strftime('%Y-%m-%d')}...")

            # Training data up to current origin
            train_data = df[df.index < forecast_origin]
            Y_train = train_data.values

            if len(Y_train) < hyperparams['n_lags'] + 10:
                continue

            try:
                model = fit_bvar_model(
                    Y=Y_train,
                    var_names=var_names,
                    hyperparams=hyperparams,
                    n_draws=n_draws,
                    n_burn=n_burn,
                    random_state=random_state,
                )
                last_fit_date = forecast_origin
            except Exception as e:
                warnings.warn(f"Fit failed at {forecast_origin}: {e}")
                continue
        else:
            # Update model with new data (simple approach: just reuse)
            # For true expanding window, would need to refit
            train_data = df[df.index < forecast_origin]
            Y_train = train_data.values

            try:
                model = fit_bvar_model(
                    Y=Y_train,
                    var_names=var_names,
                    hyperparams=hyperparams,
                    n_draws=n_draws,
                    n_burn=n_burn,
                    random_state=random_state,
                )
            except Exception as e:
                warnings.warn(f"Fit failed at {forecast_origin}: {e}")
                continue

        # Generate forecasts for each horizon
        max_h = max(horizons)
        try:
            point_forecasts = model.forecast_point(h=max_h)
            density_forecasts = model.forecast(h=max_h, return_draws=False)
        except Exception as e:
            warnings.warn(f"Forecast failed at {forecast_origin}: {e}")
            continue

        # Evaluate at each horizon
        for h in horizons:
            forecast_date = forecast_origin + pd.DateOffset(months=h)

            # Get actual value if available
            if forecast_date in df.index:
                actual = df.loc[forecast_date, target_name]
            else:
                actual = np.nan

            forecast_point = point_forecasts[h-1, target_idx]
            forecast_mean = density_forecasts['mean'][h-1, target_idx]
            forecast_lower = density_forecasts['lower_10'][h-1, target_idx]
            forecast_upper = density_forecasts['upper_90'][h-1, target_idx]

            results.append({
                'forecast_origin': forecast_origin,
                'forecast_date': forecast_date,
                'horizon': h,
                'split': split,
                'actual': actual,
                'forecast_point': forecast_point,
                'forecast_mean': forecast_mean,
                'forecast_lower_10': forecast_lower,
                'forecast_upper_90': forecast_upper,
                'error': actual - forecast_point if not np.isnan(actual) else np.nan,
                'abs_error': abs(actual - forecast_point) if not np.isnan(actual) else np.nan,
                'sq_error': (actual - forecast_point) ** 2 if not np.isnan(actual) else np.nan,
                'in_interval': (forecast_lower <= actual <= forecast_upper) if not np.isnan(actual) else np.nan,
            })

    return pd.DataFrame(results)


def compute_backtest_metrics(backtest_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary metrics from backtest results.

    Parameters
    ----------
    backtest_df : pd.DataFrame
        Raw backtest results

    Returns
    -------
    pd.DataFrame
        Summary metrics by split and horizon
    """
    metrics = []

    for split in backtest_df['split'].unique():
        for horizon in backtest_df['horizon'].unique():
            subset = backtest_df[
                (backtest_df['split'] == split) &
                (backtest_df['horizon'] == horizon) &
                (backtest_df['actual'].notna())
            ]

            if len(subset) == 0:
                continue

            errors = subset['error'].values
            abs_errors = subset['abs_error'].values
            sq_errors = subset['sq_error'].values
            actuals = subset['actual'].values

            # Scale for MASE (using naive forecast error)
            naive_errors = np.abs(np.diff(actuals))
            scale = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0

            metrics.append({
                'split': split,
                'horizon': horizon,
                'n_forecasts': len(subset),
                'mae': np.mean(abs_errors),
                'rmse': np.sqrt(np.mean(sq_errors)),
                'mase': np.mean(abs_errors) / scale if scale > 0 else np.nan,
                'bias': np.mean(errors),
                'coverage_80': np.mean(subset['in_interval'].astype(float)),
            })

    return pd.DataFrame(metrics)


# =============================================================================
# Output Saving
# =============================================================================

def save_forecasts(
    forecasts: Dict,
    varset_name: str,
    output_dir: Path = OUTPUT_DIR
) -> None:
    """Save point and density forecasts to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Point forecasts
    point_df = pd.DataFrame({
        'date': forecasts['dates'],
        'forecast': forecasts['point'],
        'mean': forecasts['mean'],
        'median': forecasts['median'],
    })
    point_df.to_csv(output_dir / f"bvar_forecasts_{varset_name}.csv", index=False)

    # Density forecasts (percentiles)
    density_df = pd.DataFrame({
        'date': forecasts['dates'],
        'mean': forecasts['mean'],
        'lower_2.5': forecasts['lower_2.5'],
        'lower_5': forecasts['lower_5'],
        'lower_10': forecasts['lower_10'],
        'lower_25': forecasts['lower_25'],
        'upper_75': forecasts['upper_75'],
        'upper_90': forecasts['upper_90'],
        'upper_95': forecasts['upper_95'],
        'upper_97.5': forecasts['upper_97.5'],
        'std': forecasts['std'],
    })
    density_df.to_csv(output_dir / f"bvar_density_{varset_name}.csv", index=False)


def save_hyperparams(
    hyperparams: Dict,
    search_results: Dict,
    varset_name: str,
    output_dir: Path = OUTPUT_DIR
) -> None:
    """Save hyperparameter selection results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'best_params': hyperparams,
        'best_cv_rmse': search_results['best_rmse'],
        'all_results': search_results['all_results'],
        'timestamp': datetime.now().isoformat(),
    }

    with open(output_dir / f"bvar_hyperparams_{varset_name}.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)


def save_posterior_summary(
    model: BayesianVAR,
    diagnostics: Dict,
    varset_name: str,
    var_names: List[str],
    output_dir: Path = OUTPUT_DIR
) -> None:
    """Save posterior summary and diagnostics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = model.get_posterior_summary()

    # Convert numpy arrays to lists for JSON
    results = {
        'hyperparameters': model.get_hyperparameters(),
        'coefficient_means': summary['coef_mean'].tolist(),
        'coefficient_stds': summary['coef_std'].tolist(),
        'sigma_mean': summary['sigma_mean'].tolist(),
        'variable_names': var_names,
        'convergence': {
            'converged': diagnostics['convergence']['converged'],
            'rhat_max': diagnostics['convergence']['summary']['rhat_max'],
            'rhat_mean': diagnostics['convergence']['summary']['rhat_mean'],
            'ess_min': diagnostics['convergence']['summary']['ess_min'],
            'ess_mean': diagnostics['convergence']['summary']['ess_mean'],
        },
        'timestamp': datetime.now().isoformat(),
    }

    with open(output_dir / f"bvar_posterior_summary_{varset_name}.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)


def save_backtest_results(
    backtest_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    varset_name: str,
    output_dir: Path = OUTPUT_DIR
) -> None:
    """Save rolling backtest results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    backtest_df.to_csv(output_dir / f"bvar_rolling_backtest_{varset_name}.csv", index=False)
    metrics_df.to_csv(output_dir / f"bvar_backtest_metrics_{varset_name}.csv", index=False)


# =============================================================================
# Visualization
# =============================================================================

def create_figures(
    model: BayesianVAR,
    forecasts: Dict,
    df: pd.DataFrame,
    varset_name: str,
    var_names: List[str],
    target_idx: int = 0,
    figures_dir: Path = FIGURES_DIR
) -> None:
    """
    Create diagnostic and forecast figures.

    Parameters
    ----------
    model : BayesianVAR
        Fitted model
    forecasts : Dict
        Forecast results
    df : pd.DataFrame
        Original data
    varset_name : str
        Variable set name
    var_names : list of str
        Variable names
    target_idx : int
        Target variable index
    figures_dir : Path
        Output directory for figures
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        warnings.warn("matplotlib not available, skipping figures")
        return

    figures_dir.mkdir(parents=True, exist_ok=True)
    target_name = var_names[target_idx]

    # 1. Fan Chart
    fig, ax = plt.subplots(figsize=(12, 6))

    # Historical data
    ax.plot(df.index, df[target_name], 'b-', label='Historical', linewidth=1.5)

    # Forecast with fan
    forecast_dates = forecasts['dates']
    ax.fill_between(
        forecast_dates,
        forecasts['lower_2.5'],
        forecasts['upper_97.5'],
        alpha=0.2, color='red', label='95% CI'
    )
    ax.fill_between(
        forecast_dates,
        forecasts['lower_10'],
        forecasts['upper_90'],
        alpha=0.3, color='red', label='80% CI'
    )
    ax.plot(forecast_dates, forecasts['mean'], 'r-', linewidth=2, label='Forecast Mean')

    ax.axvline(df.index[-1], color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel('Date')
    ax.set_ylabel(target_name)
    ax.set_title(f'BVAR Fan Chart - {varset_name}')
    ax.legend(loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(figures_dir / f"bvar_fan_chart_{varset_name}.png", dpi=150)
    plt.close()

    # 2. Coefficient Posterior (own first lag)
    fig, axes = plt.subplots(1, min(4, len(var_names)), figsize=(14, 4))
    if len(var_names) == 1:
        axes = [axes]

    for i, (ax, vname) in enumerate(zip(axes, var_names[:4])):
        chain = model.coef_posterior[:, 1 + i, i]  # Own first lag
        ax.hist(chain, bins=50, density=True, alpha=0.7, color='steelblue')
        ax.axvline(np.mean(chain), color='red', linestyle='--', label=f'Mean: {np.mean(chain):.3f}')
        ax.axvline(1.0, color='green', linestyle=':', label='Prior Mean: 1.0')
        ax.set_xlabel(f'Own Lag-1 ({vname})')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)

    plt.suptitle(f'Posterior Distributions - {varset_name}', y=1.02)
    plt.tight_layout()
    plt.savefig(figures_dir / f"bvar_coefficient_posterior_{varset_name}.png", dpi=150)
    plt.close()

    # 3. Trace Plot for target variable's own lag
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    chain = model.coef_posterior[:, 1 + target_idx, target_idx]

    # Trace
    axes[0].plot(chain, alpha=0.7, linewidth=0.5)
    axes[0].axhline(np.mean(chain), color='red', linestyle='--')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Value')
    axes[0].set_title(f'Trace Plot: Own Lag-1 for {target_name}')

    # Running Mean
    running_mean = np.cumsum(chain) / np.arange(1, len(chain) + 1)
    axes[1].plot(running_mean, color='blue', linewidth=1)
    axes[1].axhline(np.mean(chain), color='red', linestyle='--')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Running Mean')
    axes[1].set_title('Convergence Check')

    plt.tight_layout()
    plt.savefig(figures_dir / f"bvar_trace_plot_{varset_name}.png", dpi=150)
    plt.close()

    print(f"  Figures saved to {figures_dir}")


# =============================================================================
# Main Pipeline
# =============================================================================

def run_bvar_pipeline(
    varset_names: Optional[List[str]] = None,
    skip_grid_search: bool = False,
    default_hyperparams: Optional[Dict] = None,
    verbose: bool = True
) -> Dict:
    """
    Run complete BVAR pipeline for specified variable sets.

    Parameters
    ----------
    varset_names : list of str, optional
        Variable sets to process. Default: all sets.
    skip_grid_search : bool, default=False
        Skip hyperparameter search and use defaults
    default_hyperparams : dict, optional
        Default hyperparameters if skipping search
    verbose : bool, default=True
        Print progress

    Returns
    -------
    dict
        All results for each variable set
    """
    if varset_names is None:
        varset_names = VARSET_ORDER

    if default_hyperparams is None:
        default_hyperparams = {
            'n_lags': 2,
            'lambda1': 0.2,
            'lambda3': 1.0,
        }

    results = {}
    all_metrics = []

    print("=" * 70)
    print("BVAR with Minnesota Prior - Academic Pipeline")
    print("=" * 70)
    print(f"Variable sets: {varset_names}")
    print(f"Gibbs draws: {N_DRAWS} (burn-in: {N_BURN})")
    print(f"Training end: {TRAIN_END}, Validation end: {VALID_END}")
    print("=" * 70)

    for varset_name in varset_names:
        print(f"\n{'='*70}")
        print(f"Processing: {varset_name}")
        print("=" * 70)

        try:
            # 1. Load data
            df, var_names = load_varset_data(varset_name)
            target_idx = var_names.index(TARGET_VAR) if TARGET_VAR in var_names else 0

            # 2. Split data
            train_df, valid_df, test_df = prepare_train_valid_test(df)
            Y_train = train_df.values

            print(f"  Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")

            # 3. Hyperparameter search
            if skip_grid_search:
                hyperparams = default_hyperparams.copy()
                search_results = {'best_rmse': np.nan, 'all_results': []}
                print(f"  Using default hyperparams: {hyperparams}")
            else:
                search_results = run_hyperparameter_search(
                    Y_train=Y_train,
                    varset_name=varset_name,
                    target_idx=target_idx,
                    verbose=verbose,
                )
                hyperparams = search_results['best_params']

            # 4. Fit final model
            print(f"\n  Fitting final model with {N_DRAWS} draws...")
            model = fit_bvar_model(
                Y=train_df.values,
                var_names=var_names,
                hyperparams=hyperparams,
                n_draws=N_DRAWS,
                n_burn=N_BURN,
                random_state=RANDOM_STATE,
            )

            # 5. Convergence diagnostics
            print("  Running convergence diagnostics...")
            diagnostics = create_diagnostic_report(
                model,
                var_names=var_names,
                Y_actual=train_df.values,
            )

            if diagnostics['overall_converged']:
                print("  Convergence: PASSED")
            else:
                print("  Convergence: WARNING - some issues detected")
                for warning in diagnostics['convergence']['warnings'][:3]:
                    print(f"    - {warning}")

            # 6. Generate forecasts
            print("  Generating forecasts...")
            forecasts = generate_forecasts(
                model=model,
                df_train=train_df,
                h=12,
                target_idx=target_idx,
            )

            # 7. Rolling backtest
            print("  Running rolling backtest...")
            backtest_df = run_rolling_backtest(
                df=df,
                hyperparams=hyperparams,
                var_names=var_names,
                target_idx=target_idx,
                train_end=TRAIN_END,
                valid_end=VALID_END,
                refit_interval=REFIT_INTERVAL,
                n_draws=2000,
                n_burn=500,
                verbose=verbose,
            )

            metrics_df = compute_backtest_metrics(backtest_df)
            metrics_df['varset'] = varset_name
            all_metrics.append(metrics_df)

            # 8. Save outputs
            print("  Saving outputs...")
            save_forecasts(forecasts, varset_name)
            save_hyperparams(hyperparams, search_results, varset_name)
            save_posterior_summary(model, diagnostics, varset_name, var_names)
            save_backtest_results(backtest_df, metrics_df, varset_name)

            # 9. Create figures
            print("  Creating figures...")
            create_figures(
                model=model,
                forecasts=forecasts,
                df=df,
                varset_name=varset_name,
                var_names=var_names,
                target_idx=target_idx,
            )

            # Store results
            results[varset_name] = {
                'hyperparams': hyperparams,
                'cv_rmse': search_results['best_rmse'],
                'converged': diagnostics['overall_converged'],
                'rhat_max': diagnostics['convergence']['summary']['rhat_max'],
                'metrics': metrics_df.to_dict('records'),
            }

            print(f"\n  Completed: {varset_name}")

        except Exception as e:
            print(f"\n  ERROR processing {varset_name}: {e}")
            import traceback
            traceback.print_exc()
            results[varset_name] = {'error': str(e)}

    # Save combined metrics
    if all_metrics:
        combined_metrics = pd.concat(all_metrics, ignore_index=True)
        combined_metrics.to_csv(OUTPUT_DIR / "bvar_combined_metrics.csv", index=False)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for varset_name, res in results.items():
        if 'error' in res:
            print(f"{varset_name}: ERROR - {res['error']}")
        else:
            print(f"{varset_name}:")
            print(f"  Hyperparams: n_lags={res['hyperparams']['n_lags']}, "
                  f"lambda1={res['hyperparams']['lambda1']}, "
                  f"lambda3={res['hyperparams']['lambda3']}")
            print(f"  CV RMSE: {res['cv_rmse']:.4f}" if not np.isnan(res['cv_rmse']) else "  CV RMSE: N/A")
            print(f"  Converged: {res['converged']}, R-hat max: {res['rhat_max']:.4f}")

    print(f"\nOutputs saved to: {OUTPUT_DIR}")

    return results


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command-line interface entry point."""
    parser = argparse.ArgumentParser(
        description="Run BVAR with Minnesota Prior for reserves forecasting"
    )
    parser.add_argument(
        "--varsets",
        type=str,
        default=None,
        help="Comma-separated list of variable sets (default: all)"
    )
    parser.add_argument(
        "--skip-grid-search",
        action="store_true",
        help="Skip hyperparameter grid search"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )

    args = parser.parse_args()

    varset_names = None
    if args.varsets:
        varset_names = [v.strip() for v in args.varsets.split(',')]

    run_bvar_pipeline(
        varset_names=varset_names,
        skip_grid_search=args.skip_grid_search,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
