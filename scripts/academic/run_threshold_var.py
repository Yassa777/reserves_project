#!/usr/bin/env python3
"""
Threshold VAR Analysis for Sri Lankan Reserves Forecasting.

Main execution script for Specification 06: Threshold VAR.
Estimates TVAR models with exchange rate depreciation as threshold variable.

Usage:
    python run_threshold_var.py [--varset NAME] [--verbose] [--n-bootstrap N]

Options:
    --varset NAME       Process only specified variable set (default: all)
    --verbose           Enable verbose output
    --n-bootstrap N     Number of bootstrap replications for linearity test (default: 500)
    --skip-bootstrap    Skip bootstrap test (faster execution)
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from reserves_project.scripts.academic.models.tvar import (
    ThresholdVAR,
    compute_threshold_variable,
    load_threshold_variable_from_fx,
)
from reserves_project.scripts.academic.models.tvar_tests import (
    linearity_test,
    bootstrap_linearity_test,
    threshold_confidence_interval,
    compare_tvar_msvar,
    regime_persistence_test,
)

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = PROJECT_ROOT / "data"
ACADEMIC_DATA_DIR = DATA_DIR / "forecast_prep_academic"
OUTPUT_DIR = DATA_DIR / "forecast_results_academic" / "tvar"
FIGURES_DIR = OUTPUT_DIR / "figures"
HISTORICAL_FX_PATH = DATA_DIR / "external" / "historical_fx.csv"

# TVAR Configuration
TVAR_CONFIG = {
    "n_lags": 2,
    "delay": 1,  # Use t-1 threshold variable
    "trim": 0.15,  # Trim 15% from each end
    "min_obs_per_regime": 24,  # Minimum 2 years per regime
}

# Variable sets to process
VARSET_ORDER = ["parsimonious", "bop", "monetary"]

# Rolling backtest configuration
BACKTEST_CONFIG = {
    "initial_window": 84,  # 7 years
    "step_size": 3,  # Quarterly re-estimation
    "forecast_horizon": 12,
    "min_obs_per_regime_backtest": 18,  # Relaxed for shorter windows
}


# =============================================================================
# Utility Functions
# =============================================================================

def ensure_directories():
    """Create output directories."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")


def serialize_for_json(obj):
    """Convert numpy types to JSON-serializable types."""
    if isinstance(obj, dict):
        # Convert keys to strings if they're timestamps
        return {str(k) if isinstance(k, pd.Timestamp) else serialize_for_json(k): serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_json(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.int_)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float_)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif pd.isna(obj):
        return None
    elif isinstance(obj, pd.Timestamp):
        return str(obj.date())
    elif isinstance(obj, (pd.DatetimeIndex, pd.Index)):
        return [str(x) for x in obj]
    else:
        return obj


def load_varset_data(varset_name: str) -> pd.DataFrame:
    """Load VAR system data for a variable set."""
    varset_dir = ACADEMIC_DATA_DIR / f"varset_{varset_name}"
    var_path = varset_dir / "var_system.csv"

    if not var_path.exists():
        raise FileNotFoundError(f"VAR system not found: {var_path}")

    df = pd.read_csv(var_path, parse_dates=["date"], index_col="date")
    df.index = df.index.to_period("M").to_timestamp()
    df = df.sort_index()

    return df


def load_threshold_variable() -> pd.Series:
    """
    Load and compute threshold variable (exchange rate % change).

    Tries multiple sources:
    1. Historical FX file
    2. VAR system data (if usd_lkr is present)
    """
    # Try historical FX first
    if HISTORICAL_FX_PATH.exists():
        try:
            z = load_threshold_variable_from_fx(HISTORICAL_FX_PATH)
            print(f"  Loaded threshold variable from: {HISTORICAL_FX_PATH}")
            print(f"    {len(z)} observations, {z.index.min().date()} to {z.index.max().date()}")
            return z
        except Exception as e:
            print(f"  Warning: Could not load from historical_fx.csv: {e}")

    # Try parsimonious varset (contains usd_lkr)
    try:
        df = load_varset_data("parsimonious")
        if "usd_lkr" in df.columns:
            z = compute_threshold_variable(df, "usd_lkr", "pct_change")
            print(f"  Computed threshold variable from parsimonious varset")
            print(f"    {len(z)} observations")
            return z
    except Exception as e:
        print(f"  Warning: Could not compute from parsimonious varset: {e}")

    raise ValueError("Could not load threshold variable from any source")


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_regime_indicators(
    tvar: ThresholdVAR,
    varset_name: str,
    target_var: str = "gross_reserves_usd_m",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot series with regime shading.

    Three panels:
    1. Target variable with regime shading
    2. Threshold variable with threshold line
    3. Regime indicator
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    dates = tvar.Y.index
    regime_series = tvar.get_regime_series()

    # Panel 1: Target variable with regime shading
    ax1 = axes[0]
    if target_var in tvar.Y.columns:
        ax1.plot(dates, tvar.Y[target_var], 'b-', linewidth=1.5, label=target_var)
    else:
        # Plot first variable
        first_var = tvar.Y.columns[0]
        ax1.plot(dates, tvar.Y[first_var], 'b-', linewidth=1.5, label=first_var)

    # Add regime shading
    ylim = ax1.get_ylim()
    for i, (start, end) in enumerate(zip(dates[:-1], dates[1:])):
        if tvar.regime_indicators.loc[start] == 1:
            ax1.axvspan(start, end, alpha=0.3, color='red')

    ax1.set_ylabel('Reserves (USD m)')
    ax1.set_title(f'TVAR Regime Indicators - {varset_name.capitalize()} Variable Set')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Threshold variable
    ax2 = axes[1]
    ax2.plot(dates, tvar.z, 'k-', linewidth=1, label=tvar.threshold_var_name)
    ax2.axhline(tvar.threshold, color='red', linestyle='--', linewidth=2,
                label=f'Threshold = {tvar.threshold:.2f}%')
    ax2.fill_between(dates, tvar.z, tvar.threshold,
                     where=tvar.z > tvar.threshold,
                     alpha=0.3, color='red', label='Crisis regime')
    ax2.set_ylabel('Exchange Rate % Change')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Regime indicator
    ax3 = axes[2]
    ax3.fill_between(dates, 0, tvar.regime_indicators,
                     step='mid', alpha=0.5, color='red')
    ax3.set_ylabel('Regime')
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['Stable', 'Crisis'])
    ax3.set_xlabel('Date')
    ax3.grid(True, alpha=0.3)

    # Format x-axis
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax3.xaxis.set_major_locator(mdates.YearLocator(2))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_threshold_selection(
    tvar: ThresholdVAR,
    varset_name: str,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot SSR as function of threshold value."""
    if tvar.grid_search_results is None:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    ssr_values = tvar.grid_search_results["ssr_values"]
    thresholds = [x[0] for x in ssr_values]
    ssrs = [x[1] for x in ssr_values]

    # Filter out infinite values for plotting
    valid = [s < np.inf for s in ssrs]
    thresholds_plot = [t for t, v in zip(thresholds, valid) if v]
    ssrs_plot = [s for s, v in zip(ssrs, valid) if v]

    ax.plot(thresholds_plot, ssrs_plot, 'b-', linewidth=1.5)
    ax.axvline(tvar.threshold, color='red', linestyle='--', linewidth=2,
               label=f'Optimal: {tvar.threshold:.2f}%')
    ax.scatter([tvar.threshold], [tvar.grid_search_results["best_ssr"]],
               color='red', s=100, zorder=5)

    ax.set_xlabel('Threshold Value (%)')
    ax.set_ylabel('Sum of Squared Residuals')
    ax.set_title(f'Threshold Selection - {varset_name.capitalize()}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_forecast_scenarios(
    tvar: ThresholdVAR,
    varset_name: str,
    target_var: str = "gross_reserves_usd_m",
    h: int = 12,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot forecast scenarios for stable vs crisis regimes."""
    scenarios = tvar.forecast_by_scenario(h=h)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Historical data
    dates = tvar.Y.index
    target_idx = list(tvar.Y.columns).index(target_var) if target_var in tvar.Y.columns else 0
    ax.plot(dates, tvar.Y.iloc[:, target_idx], 'b-', linewidth=1.5, label='Historical')

    # Forecast dates
    last_date = dates[-1]
    forecast_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=h,
        freq="MS"
    )

    # Scenario forecasts
    ax.plot(forecast_dates, scenarios["regime1_scenario"][:, target_idx],
            'g--', linewidth=2, label='Stable Regime Scenario')
    ax.plot(forecast_dates, scenarios["regime2_scenario"][:, target_idx],
            'r--', linewidth=2, label='Crisis Regime Scenario')

    # Shade forecast period
    ax.axvspan(forecast_dates[0], forecast_dates[-1], alpha=0.1, color='gray')

    ax.set_xlabel('Date')
    ax.set_ylabel('Reserves (USD m)')
    ax.set_title(f'TVAR Forecast Scenarios - {varset_name.capitalize()}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Format dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# =============================================================================
# Rolling Backtest
# =============================================================================

def run_rolling_backtest(
    Y: pd.DataFrame,
    z: pd.Series,
    config: Dict[str, Any],
    verbose: bool = False
) -> pd.DataFrame:
    """
    Run rolling window backtest for TVAR.

    Parameters
    ----------
    Y : pd.DataFrame
        VAR system data
    z : pd.Series
        Threshold variable
    config : dict
        Backtest configuration
    verbose : bool
        Print progress

    Returns
    -------
    pd.DataFrame
        Backtest results with columns: date, horizon, actual, forecast, error
    """
    results = []

    initial_window = config["initial_window"]
    step_size = config["step_size"]
    h = config["forecast_horizon"]
    min_obs = config.get("min_obs_per_regime_backtest", 18)

    # Align data
    common_idx = Y.index.intersection(z.index)
    Y = Y.loc[common_idx]
    z = z.loc[common_idx]

    T = len(Y)
    n_windows = (T - initial_window - h) // step_size + 1

    if n_windows <= 0:
        print("  Warning: Insufficient data for rolling backtest")
        return pd.DataFrame()

    target_col = Y.columns[0]  # Assume first column is target

    for w in range(n_windows):
        start_idx = 0
        end_idx = initial_window + w * step_size

        if end_idx + h > T:
            break

        # Training data
        Y_train = Y.iloc[start_idx:end_idx]
        z_train = z.iloc[start_idx:end_idx]

        # Test data
        Y_test = Y.iloc[end_idx:end_idx + h]

        try:
            # Fit TVAR
            tvar = ThresholdVAR(
                n_lags=TVAR_CONFIG["n_lags"],
                delay=TVAR_CONFIG["delay"],
                trim=TVAR_CONFIG["trim"],
                min_obs_per_regime=min_obs
            )
            tvar.fit(Y_train, z_train)

            # Generate forecasts (use last regime)
            forecasts = tvar.forecast(h=h)

            # Store results
            for horizon in range(h):
                results.append({
                    "forecast_origin": Y_train.index[-1],
                    "forecast_date": Y_test.index[horizon],
                    "horizon": horizon + 1,
                    "actual": Y_test[target_col].iloc[horizon],
                    "forecast": forecasts[horizon, 0],
                    "error": Y_test[target_col].iloc[horizon] - forecasts[horizon, 0],
                    "threshold": tvar.threshold,
                    "current_regime": int(tvar.regime_indicators.iloc[-1]),
                })

        except Exception as e:
            if verbose:
                print(f"    Window {w+1}: Failed - {e}")
            continue

    return pd.DataFrame(results)


def compute_backtest_metrics(backtest_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute forecast accuracy metrics from backtest results."""
    if len(backtest_df) == 0:
        return {}

    metrics = {}

    # Overall metrics
    errors = backtest_df["error"]
    metrics["overall"] = {
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mae": float(np.mean(np.abs(errors))),
        "mape": float(np.mean(np.abs(errors / backtest_df["actual"]) * 100)),
        "n_forecasts": len(backtest_df),
    }

    # By horizon
    metrics["by_horizon"] = {}
    for h in backtest_df["horizon"].unique():
        h_df = backtest_df[backtest_df["horizon"] == h]
        h_errors = h_df["error"]
        metrics["by_horizon"][int(h)] = {
            "rmse": float(np.sqrt(np.mean(h_errors ** 2))),
            "mae": float(np.mean(np.abs(h_errors))),
            "n_forecasts": len(h_df),
        }

    # By regime at forecast origin
    metrics["by_regime"] = {}
    for regime in [0, 1]:
        r_df = backtest_df[backtest_df["current_regime"] == regime]
        if len(r_df) > 0:
            r_errors = r_df["error"]
            regime_name = "stable" if regime == 0 else "crisis"
            metrics["by_regime"][regime_name] = {
                "rmse": float(np.sqrt(np.mean(r_errors ** 2))),
                "mae": float(np.mean(np.abs(r_errors))),
                "n_forecasts": len(r_df),
            }

    return metrics


# =============================================================================
# Main Processing
# =============================================================================

def process_varset(
    varset_name: str,
    z: pd.Series,
    verbose: bool = False,
    n_bootstrap: int = 500,
    skip_bootstrap: bool = False
) -> Dict[str, Any]:
    """Process a single variable set."""
    print(f"\n{'='*60}")
    print(f"Processing: {varset_name.upper()}")
    print(f"{'='*60}")

    results = {
        "varset": varset_name,
        "timestamp": datetime.now().isoformat(),
    }

    # Load data
    try:
        Y = load_varset_data(varset_name)
        print(f"  Loaded {len(Y)} observations, {Y.shape[1]} variables")
        print(f"  Variables: {Y.columns.tolist()}")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        results["error"] = str(e)
        return results

    # Fit TVAR
    print("\n  Fitting Threshold VAR...")
    try:
        tvar = ThresholdVAR(**TVAR_CONFIG)
        tvar.fit(Y, z, threshold_var_name="usd_lkr_pct_change")

        results["tvar"] = tvar.summary()
        print(f"    Optimal threshold: {tvar.threshold:.2f}%")
        print(f"    Regime 1 (Stable): {tvar.regime_stats['regime1']['n_obs']} obs ({tvar.regime_stats['regime1']['pct']:.1f}%)")
        print(f"    Regime 2 (Crisis): {tvar.regime_stats['regime2']['n_obs']} obs ({tvar.regime_stats['regime2']['pct']:.1f}%)")
        print(f"    Regime transitions: {tvar.regime_stats['n_transitions']}")

    except ValueError as e:
        print(f"  ERROR fitting TVAR: {e}")
        results["error"] = str(e)
        return results

    # Threshold confidence interval
    print("\n  Computing threshold confidence interval...")
    ci_result = threshold_confidence_interval(
        Y, z,
        n_lags=TVAR_CONFIG["n_lags"],
        delay=TVAR_CONFIG["delay"],
        trim=TVAR_CONFIG["trim"],
        min_obs_per_regime=TVAR_CONFIG["min_obs_per_regime"]
    )
    results["threshold_ci"] = ci_result
    print(f"    95% CI: [{ci_result['lower']:.2f}%, {ci_result['upper']:.2f}%]")

    # Linearity test
    print("\n  Testing for nonlinearity...")
    lin_test = linearity_test(
        Y, z,
        n_lags=TVAR_CONFIG["n_lags"],
        delay=TVAR_CONFIG["delay"],
        trim=TVAR_CONFIG["trim"],
        min_obs_per_regime=TVAR_CONFIG["min_obs_per_regime"]
    )
    results["linearity_test"] = lin_test
    print(f"    F-statistic: {lin_test['f_statistic']:.2f}")
    print(f"    SSR reduction: {lin_test['ssr_reduction_pct']:.1f}%")

    # Note if TVAR doesn't improve over linear VAR
    if lin_test['ssr_reduction_pct'] < 0:
        print("    NOTE: TVAR does not improve over linear VAR (negative SSR reduction)")
        print("    This suggests sharp regime switching may not be supported by the data")
        results["tvar_improves"] = False
    else:
        results["tvar_improves"] = True

    # Bootstrap linearity test
    if not skip_bootstrap:
        print(f"\n  Bootstrap linearity test ({n_bootstrap} replications)...")
        boot_test = bootstrap_linearity_test(
            Y, z,
            n_lags=TVAR_CONFIG["n_lags"],
            delay=TVAR_CONFIG["delay"],
            trim=TVAR_CONFIG["trim"],
            min_obs_per_regime=TVAR_CONFIG["min_obs_per_regime"],
            n_bootstrap=n_bootstrap,
            random_state=42
        )
        results["bootstrap_test"] = boot_test
        print(f"    Bootstrap p-value: {boot_test['bootstrap_p_value']:.4f}")
        print(f"    Reject linearity at 5%: {boot_test['reject_linearity']}")
    else:
        print("  Skipping bootstrap test")
        results["bootstrap_test"] = {"skipped": True}

    # Regime persistence
    print("\n  Analyzing regime persistence...")
    persistence = regime_persistence_test(tvar.regime_indicators)
    results["regime_persistence"] = persistence
    if persistence["p_00"] is not None:
        print(f"    P(stay stable | stable): {persistence['p_00']:.2f}")
        print(f"    P(stay crisis | crisis): {persistence['p_11']:.2f}")
        print(f"    Persistence index: {persistence['persistence_index']:.2f}")

    # Generate plots
    print("\n  Generating plots...")
    try:
        # Regime plot
        fig = plot_regime_indicators(
            tvar, varset_name,
            save_path=FIGURES_DIR / f"tvar_regimes_{varset_name}.png"
        )
        plt.close(fig)

        # Threshold selection
        fig = plot_threshold_selection(
            tvar, varset_name,
            save_path=FIGURES_DIR / f"tvar_threshold_selection_{varset_name}.png"
        )
        if fig:
            plt.close(fig)

        # Forecast scenarios
        fig = plot_forecast_scenarios(
            tvar, varset_name,
            save_path=FIGURES_DIR / f"tvar_forecast_scenarios_{varset_name}.png"
        )
        plt.close(fig)

        print("    Saved regime, threshold selection, and forecast scenario plots")

    except Exception as e:
        print(f"    Warning: Plot generation failed - {e}")

    # Save regime indicators
    regime_df = tvar.get_regime_series()
    regime_path = OUTPUT_DIR / f"tvar_regime_indicators_{varset_name}.csv"
    regime_df.to_csv(regime_path)
    print(f"    Saved regime indicators to: {regime_path.name}")

    # Generate forecasts
    print("\n  Generating forecasts...")
    forecast_df = tvar.forecast_df(h=12)
    forecast_path = OUTPUT_DIR / f"tvar_forecasts_{varset_name}.csv"
    forecast_df.to_csv(forecast_path)
    results["forecasts"] = forecast_df.to_dict()
    print(f"    Saved 12-month forecasts to: {forecast_path.name}")

    # Scenario forecasts
    scenarios = tvar.forecast_by_scenario(h=12)
    results["forecast_scenarios"] = {
        "regime1": scenarios["regime1_scenario"].tolist(),
        "regime2": scenarios["regime2_scenario"].tolist(),
    }

    # Rolling backtest
    print("\n  Running rolling backtest...")
    backtest_df = run_rolling_backtest(Y, z, BACKTEST_CONFIG, verbose=verbose)
    if len(backtest_df) > 0:
        backtest_path = OUTPUT_DIR / f"tvar_rolling_backtest_{varset_name}.csv"
        backtest_df.to_csv(backtest_path, index=False)

        backtest_metrics = compute_backtest_metrics(backtest_df)
        results["backtest_metrics"] = backtest_metrics
        print(f"    Backtest RMSE: {backtest_metrics['overall']['rmse']:.2f}")
        print(f"    Backtest MAE: {backtest_metrics['overall']['mae']:.2f}")
        print(f"    Saved backtest results to: {backtest_path.name}")
    else:
        print("    Warning: No backtest results generated")

    # Store regime periods
    results["regime_periods"] = tvar.get_regime_periods()

    return results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Threshold VAR Analysis")
    parser.add_argument(
        "--varset", type=str, default=None,
        help="Process only specified variable set"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=500,
        help="Number of bootstrap replications"
    )
    parser.add_argument(
        "--skip-bootstrap", action="store_true",
        help="Skip bootstrap linearity test"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("THRESHOLD VAR ANALYSIS")
    print("Specification 06: Regime-Switching Based on Observable Threshold")
    print("=" * 70)
    print(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Threshold variable: Exchange rate % change (USD/LKR)")
    print(f"Configuration: {TVAR_CONFIG}")

    # Setup
    ensure_directories()

    # Load threshold variable
    print("\nLoading threshold variable...")
    try:
        z = load_threshold_variable()
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        sys.exit(1)

    # Determine which variable sets to process
    if args.varset:
        if args.varset not in VARSET_ORDER:
            print(f"ERROR: Unknown variable set '{args.varset}'")
            print(f"Available: {VARSET_ORDER}")
            sys.exit(1)
        varsets_to_process = [args.varset]
    else:
        varsets_to_process = VARSET_ORDER

    # Process each variable set
    all_results = {}
    for varset_name in varsets_to_process:
        results = process_varset(
            varset_name, z,
            verbose=args.verbose,
            n_bootstrap=args.n_bootstrap,
            skip_bootstrap=args.skip_bootstrap
        )
        all_results[varset_name] = results

    # Save consolidated results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    # Save individual results
    for varset_name, results in all_results.items():
        result_path = OUTPUT_DIR / f"tvar_threshold_{varset_name}.json"
        with open(result_path, 'w') as f:
            json.dump(serialize_for_json(results), f, indent=2)
        print(f"  Saved: {result_path.name}")

    # Create summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    summary_rows = []
    for varset_name, results in all_results.items():
        if "error" in results:
            continue

        tvar_summary = results.get("tvar", {})
        ci = results.get("threshold_ci", {})
        lin_test = results.get("linearity_test", {})
        boot_test = results.get("bootstrap_test", {})
        backtest = results.get("backtest_metrics", {})

        row = {
            "variable_set": varset_name,
            "threshold": tvar_summary.get("threshold"),
            "ci_lower": ci.get("lower"),
            "ci_upper": ci.get("upper"),
            "regime1_pct": tvar_summary.get("regime_stats", {}).get("regime1", {}).get("pct"),
            "regime2_pct": tvar_summary.get("regime_stats", {}).get("regime2", {}).get("pct"),
            "f_statistic": lin_test.get("f_statistic"),
            "ssr_reduction_pct": lin_test.get("ssr_reduction_pct"),
            "bootstrap_p_value": boot_test.get("bootstrap_p_value"),
            "reject_linearity": boot_test.get("reject_linearity"),
            "backtest_rmse": backtest.get("overall", {}).get("rmse"),
            "backtest_mae": backtest.get("overall", {}).get("mae"),
        }
        summary_rows.append(row)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = OUTPUT_DIR / "tvar_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to: {summary_path}")

        print("\nThreshold Estimates:")
        print("-" * 60)
        for _, row in summary_df.iterrows():
            print(f"  {row['variable_set']:12s}: {row['threshold']:.2f}% "
                  f"[{row['ci_lower']:.2f}, {row['ci_upper']:.2f}]")

        print("\nLinearity Test Results:")
        print("-" * 60)
        for _, row in summary_df.iterrows():
            reject = "Yes" if row.get('reject_linearity') else "No"
            p_val = row.get('bootstrap_p_value', np.nan)
            p_str = f"{p_val:.4f}" if not pd.isna(p_val) else "N/A"
            print(f"  {row['variable_set']:12s}: F={row['f_statistic']:.2f}, "
                  f"p={p_str}, Reject H0: {reject}")

        print("\nBacktest Performance:")
        print("-" * 60)
        for _, row in summary_df.iterrows():
            rmse = row.get('backtest_rmse', np.nan)
            mae = row.get('backtest_mae', np.nan)
            if not pd.isna(rmse):
                print(f"  {row['variable_set']:12s}: RMSE={rmse:.2f}, MAE={mae:.2f}")

    print("\n" + "=" * 70)
    print("EXECUTION COMPLETE")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Figures: {len(list(FIGURES_DIR.glob('*.png')))}")

    return all_results


if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    results = main()
