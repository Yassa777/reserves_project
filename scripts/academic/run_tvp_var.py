"""
TVP-VAR Execution Script for Academic Reserves Forecasting Pipeline.

Runs Time-Varying Parameter VAR for specified variable sets and generates:
- Time-varying coefficient paths
- Convergence diagnostics
- Comparison with structural break dates
- Rolling backtests
- Forecast density outputs

Reference: Specification 04 - TVP-VAR

Usage:
    python run_tvp_var.py
    python run_tvp_var.py --varsets parsimonious bop monetary
    python run_tvp_var.py --fast  # Reduced draws for testing
"""

import os
import sys
import json
import warnings
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from reserves_project.scripts.academic.models.tvp_var import TVP_VAR, fit_tvp_var_from_dataframe
from reserves_project.scripts.academic.models.tvp_var_diagnostics import (
    plot_convergence_diagnostics,
    plot_coefficient_paths,
    plot_tvp_vs_breaks,
    plot_volatility_paths,
    plot_forecast_fan,
    rolling_backtest,
    create_comparison_summary
)
from reserves_project.scripts.academic.variable_sets.config import (
    VARIABLE_SETS,
    TARGET_VAR,
    TRAIN_END,
    VALID_END,
)

# =============================================================================
# Configuration
# =============================================================================

# Paths
DATA_DIR = PROJECT_ROOT / "data" / "forecast_prep_academic"
BREAKS_DIR = PROJECT_ROOT / "data" / "structural_breaks"
OUTPUT_DIR = PROJECT_ROOT / "data" / "forecast_results_academic" / "tvp_var"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Model configuration
TVP_CONFIG = {
    "n_lags": 1,
    "stochastic_vol": True,
    "n_draws": 5000,
    "n_burn": 2000,
    "random_state": 42
}

# Fast mode for testing (uses Kalman filter only)
TVP_CONFIG_FAST = {
    "n_lags": 1,
    "stochastic_vol": False,
    "n_draws": 500,
    "n_burn": 0,
    "random_state": 42,
    "fast_mode": True
}

# Rolling backtest config (uses fast mode for speed)
BACKTEST_CONFIG = {
    "train_start": "2007-01-01",
    "test_start": "2021-01-01",  # Shorter test window
    "test_end": "2024-12-01",
    "h": 3,  # 3-month ahead forecast
    "step": 6,  # Semi-annual rolling window for speed
    "n_draws": 100,
    "n_burn": 0,
    "stochastic_vol": False,
    "fast_mode": True  # Use Kalman filter only for speed
}

# Variable sets to run (can be overridden by command line)
DEFAULT_VARSETS = ["parsimonious", "bop", "monetary"]


# =============================================================================
# Utility Functions
# =============================================================================

def ensure_directories():
    """Create output directories."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")


def load_data(varset_name: str) -> pd.DataFrame:
    """Load VAR system data for a variable set."""
    var_dir = DATA_DIR / f"varset_{varset_name}"
    data_path = var_dir / "var_system.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    df = pd.read_csv(data_path, parse_dates=['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    print(f"  Loaded {len(df)} observations: {df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')}")
    return df


def load_break_dates() -> dict:
    """Load structural break dates from Bai-Perron analysis."""
    breaks_path = BREAKS_DIR / "bai_perron_results.json"

    if not breaks_path.exists():
        print(f"  Warning: Break dates not found at {breaks_path}")
        return {}

    with open(breaks_path, 'r') as f:
        results = json.load(f)

    return results


def serialize_for_json(obj):
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_json(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.int_)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.floating)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.strftime('%Y-%m-%d')
    elif pd.isna(obj):
        return None
    else:
        return obj


# =============================================================================
# Main Execution Functions
# =============================================================================

def fit_tvp_var_for_varset(
    varset_name: str,
    config: dict,
    verbose: bool = True
) -> dict:
    """
    Fit TVP-VAR for a single variable set.

    Returns dict with model, coefficients, volatility, diagnostics.
    """
    print(f"\n{'='*70}")
    print(f"FITTING TVP-VAR: {varset_name.upper()}")
    print(f"{'='*70}")

    # Get variable set config
    varset = VARIABLE_SETS[varset_name]
    system_vars = varset["var_system"]

    print(f"  System variables: {system_vars}")

    # Load data
    df = load_data(varset_name)

    # Filter to training + validation period for initial fit
    df_fit = df.copy()

    # Fit model
    print(f"\n  Fitting TVP-VAR...")
    model = fit_tvp_var_from_dataframe(
        df=df_fit,
        target_var=TARGET_VAR,
        system_vars=system_vars,
        **config,
        verbose=verbose
    )

    # Extract results
    results = {
        "varset": varset_name,
        "config": config,
        "system_vars": system_vars,
        "n_obs": model.T,
        "model": model
    }

    # Get coefficients for target variable
    coefs = model.get_time_varying_coefficients(var_idx=0)
    results["coefficients"] = coefs

    # Get volatility if stochastic vol enabled
    if config.get("stochastic_vol", False):
        vol = model.get_volatility_path()
        results["volatility"] = vol

    # Convergence diagnostics
    diag = model.get_convergence_diagnostics()
    results["diagnostics"] = diag

    print(f"\n  Convergence diagnostics:")
    print(f"    Mean ESS: {diag['mean_ess']:.1f}")
    print(f"    Min ESS: {diag['min_ess']:.1f}")
    print(f"    Mean autocorr (lag 1): {diag['mean_autocorr_lag1']:.3f}")

    # Generate forecasts
    forecast = model.forecast(h=12)
    results["forecast"] = forecast

    return results


def generate_plots(
    results: dict,
    break_dates: list,
    segment_stats: list
):
    """Generate all diagnostic plots for a variable set."""
    varset_name = results["varset"]
    model = results["model"]
    coefs = results["coefficients"]
    system_vars = results["system_vars"]

    print(f"\n  Generating plots for {varset_name}...")

    # 1. Convergence diagnostics
    try:
        fig = plot_convergence_diagnostics(
            model,
            n_params=min(6, model.n_coefs * model.k),
            save_path=FIGURES_DIR / f"tvp_convergence_{varset_name}.png"
        )
        plt.close(fig)
        print(f"    Saved: tvp_convergence_{varset_name}.png")
    except Exception as e:
        print(f"    Error in convergence plot: {e}")

    # 2. Coefficient paths
    try:
        fig = plot_coefficient_paths(
            coefs,
            var_name=TARGET_VAR,
            break_dates=break_dates,
            save_path=FIGURES_DIR / f"tvp_coefficient_paths_{varset_name}.png"
        )
        plt.close(fig)
        print(f"    Saved: tvp_coefficient_paths_{varset_name}.png")
    except Exception as e:
        print(f"    Error in coefficient paths plot: {e}")

    # 3. TVP vs structural breaks comparison
    if break_dates and len(coefs['coef_names']) > 1:
        try:
            # Find exchange rate coefficient index
            target_idx = 1  # First lag coefficient
            for i, name in enumerate(coefs['coef_names']):
                if 'usd_lkr' in name.lower() or 'exchange' in name.lower():
                    target_idx = i
                    break

            break_labels = [f"Break {i+1}" for i in range(len(break_dates))]

            fig = plot_tvp_vs_breaks(
                coefs,
                break_dates=break_dates,
                break_labels=break_labels,
                target_coef_idx=target_idx,
                save_path=FIGURES_DIR / f"tvp_vs_breaks_{varset_name}.png"
            )
            plt.close(fig)
            print(f"    Saved: tvp_vs_breaks_{varset_name}.png")
        except Exception as e:
            print(f"    Error in TVP vs breaks plot: {e}")

    # 4. Volatility paths (if available)
    if "volatility" in results and results["volatility"] is not None:
        try:
            fig = plot_volatility_paths(
                results["volatility"],
                var_names=system_vars,
                break_dates=break_dates,
                save_path=FIGURES_DIR / f"tvp_volatility_{varset_name}.png"
            )
            plt.close(fig)
            print(f"    Saved: tvp_volatility_{varset_name}.png")
        except Exception as e:
            print(f"    Error in volatility plot: {e}")

    # 5. Forecast fan chart
    try:
        # Get historical data
        df = load_data(varset_name)
        history = df[TARGET_VAR].iloc[-36:]  # Last 3 years

        # Generate forecast dates
        last_date = df.index[-1]
        forecast_dates = pd.date_range(
            last_date + pd.DateOffset(months=1),
            periods=12,
            freq='MS'
        )

        fig = plot_forecast_fan(
            results["forecast"],
            history=history,
            forecast_dates=forecast_dates,
            var_name=TARGET_VAR,
            save_path=FIGURES_DIR / f"tvp_forecast_{varset_name}.png"
        )
        plt.close(fig)
        print(f"    Saved: tvp_forecast_{varset_name}.png")
    except Exception as e:
        print(f"    Error in forecast plot: {e}")


def run_rolling_backtest(
    varset_name: str,
    config: dict
) -> dict:
    """Run rolling window backtest for a variable set."""
    print(f"\n  Running rolling backtest for {varset_name}...")

    varset = VARIABLE_SETS[varset_name]
    system_vars = varset["var_system"]

    # Load full data
    df = load_data(varset_name)

    # Run backtest
    backtest_results = rolling_backtest(
        df=df,
        model_class=TVP_VAR,
        system_vars=system_vars,
        target_var=TARGET_VAR,
        **BACKTEST_CONFIG,
        verbose=True
    )

    if "error" in backtest_results:
        print(f"    Backtest error: {backtest_results['error']}")
        return backtest_results

    metrics = backtest_results["metrics"]
    print(f"\n  Backtest results ({BACKTEST_CONFIG['h']}-month ahead):")
    print(f"    MAE: {metrics['mae']:.2f}")
    print(f"    RMSE: {metrics['rmse']:.2f}")
    print(f"    MAPE: {metrics['mape']:.2f}%")
    print(f"    Direction Accuracy: {metrics['directional_accuracy']:.1f}%")
    print(f"    80% Coverage: {metrics['coverage_80']:.1f}%")
    print(f"    N forecasts: {backtest_results['n_forecasts']}")

    return backtest_results


def save_results(
    results: dict,
    backtest_results: dict,
    break_dates: list,
    segment_stats: list
):
    """Save all results to files."""
    varset_name = results["varset"]
    coefs = results["coefficients"]

    print(f"\n  Saving results for {varset_name}...")

    # 1. Time-varying coefficients
    coef_df = pd.DataFrame(
        coefs['mean'],
        index=coefs['dates'],
        columns=coefs['coef_names']
    )
    coef_df.to_csv(OUTPUT_DIR / f"tvp_coefficients_{varset_name}.csv")
    print(f"    Saved: tvp_coefficients_{varset_name}.csv")

    # 2. Coefficient credible intervals
    ci_data = {
        'date': coefs['dates'],
    }
    for i, name in enumerate(coefs['coef_names']):
        ci_data[f'{name}_mean'] = coefs['mean'][:, i]
        ci_data[f'{name}_lower_16'] = coefs['lower_16'][:, i]
        ci_data[f'{name}_upper_84'] = coefs['upper_84'][:, i]
        ci_data[f'{name}_lower_5'] = coefs['lower_5'][:, i]
        ci_data[f'{name}_upper_95'] = coefs['upper_95'][:, i]

    ci_df = pd.DataFrame(ci_data)
    ci_df.to_csv(OUTPUT_DIR / f"tvp_coefficients_ci_{varset_name}.csv", index=False)
    print(f"    Saved: tvp_coefficients_ci_{varset_name}.csv")

    # 3. Volatility (if available)
    if "volatility" in results and results["volatility"] is not None:
        vol = results["volatility"]
        vol_df = pd.DataFrame(
            vol['mean'],
            index=vol['dates'],
            columns=results["system_vars"]
        )
        vol_df.to_csv(OUTPUT_DIR / f"tvp_volatility_{varset_name}.csv")
        print(f"    Saved: tvp_volatility_{varset_name}.csv")

    # 4. Forecasts
    forecast = results["forecast"]
    forecast_df = pd.DataFrame({
        'horizon': range(1, forecast['mean'].shape[0] + 1),
        'mean': forecast['mean'][:, 0],
        'median': forecast['median'][:, 0],
        'lower_10': forecast['lower_10'][:, 0],
        'upper_90': forecast['upper_90'][:, 0],
        'lower_5': forecast['lower_5'][:, 0],
        'upper_95': forecast['upper_95'][:, 0]
    })
    forecast_df.to_csv(OUTPUT_DIR / f"tvp_forecast_{varset_name}.csv", index=False)
    print(f"    Saved: tvp_forecast_{varset_name}.csv")

    # 5. Rolling backtest results
    if "forecasts" in backtest_results:
        backtest_results["forecasts"].to_csv(
            OUTPUT_DIR / f"tvp_rolling_backtest_{varset_name}.csv"
        )
        print(f"    Saved: tvp_rolling_backtest_{varset_name}.csv")

    # 6. Diagnostics and summary
    summary = {
        "varset": varset_name,
        "execution_time": datetime.now().isoformat(),
        "config": serialize_for_json(results["config"]),
        "system_vars": results["system_vars"],
        "n_obs": results["n_obs"],
        "convergence": serialize_for_json(results["diagnostics"]),
        "break_dates": break_dates,
        "backtest_metrics": serialize_for_json(backtest_results.get("metrics", {})),
        "backtest_n_forecasts": backtest_results.get("n_forecasts", 0)
    }

    with open(OUTPUT_DIR / f"tvp_summary_{varset_name}.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"    Saved: tvp_summary_{varset_name}.json")

    # 7. Comparison with segments (if segment stats available)
    if segment_stats:
        try:
            comparison_df = create_comparison_summary(
                {"coefs": coefs},
                break_dates,
                segment_stats,
                save_path=OUTPUT_DIR / f"tvp_segment_comparison_{varset_name}.csv"
            )
            print(f"    Saved: tvp_segment_comparison_{varset_name}.csv")
        except Exception as e:
            print(f"    Error creating comparison: {e}")


def create_combined_summary(all_results: dict, all_backtests: dict):
    """Create summary across all variable sets."""
    print(f"\n{'='*70}")
    print("CREATING COMBINED SUMMARY")
    print(f"{'='*70}")

    rows = []
    for varset_name, results in all_results.items():
        backtest = all_backtests.get(varset_name, {})
        metrics = backtest.get("metrics", {})
        diag = results.get("diagnostics", {})

        rows.append({
            "varset": varset_name,
            "n_vars": len(results["system_vars"]),
            "n_obs": results["n_obs"],
            "mean_ess": diag.get("mean_ess", np.nan),
            "min_ess": diag.get("min_ess", np.nan),
            "autocorr_lag1": diag.get("mean_autocorr_lag1", np.nan),
            "backtest_mae": metrics.get("mae", np.nan),
            "backtest_rmse": metrics.get("rmse", np.nan),
            "backtest_mape": metrics.get("mape", np.nan),
            "backtest_dir_acc": metrics.get("directional_accuracy", np.nan),
            "coverage_80": metrics.get("coverage_80", np.nan),
            "n_forecasts": backtest.get("n_forecasts", 0)
        })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUTPUT_DIR / "tvp_var_summary.csv", index=False)
    print(f"\nSummary saved to: {OUTPUT_DIR / 'tvp_var_summary.csv'}")

    print("\n" + summary_df.to_string())

    return summary_df


# =============================================================================
# Main Entry Point
# =============================================================================

def main(varsets: list = None, fast: bool = False, skip_backtest: bool = False):
    """Main execution function."""
    print("=" * 70)
    print("TVP-VAR ESTIMATION")
    print("Specification 04: Time-Varying Parameter VAR")
    print("=" * 70)
    print(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Setup
    ensure_directories()

    # Choose configuration
    config = TVP_CONFIG_FAST if fast else TVP_CONFIG
    if fast:
        print("\nRunning in FAST mode (reduced draws)")

    # Choose variable sets
    if varsets is None:
        varsets = DEFAULT_VARSETS
    print(f"\nVariable sets: {varsets}")

    # Load structural break information
    break_info = load_break_dates()
    reserves_breaks = break_info.get("gross_reserves_usd_m", {})
    break_dates = reserves_breaks.get("break_dates", [])
    segment_stats = reserves_breaks.get("segment_stats", [])

    print(f"\nStructural breaks for reserves: {break_dates}")

    # Run for each variable set
    all_results = {}
    all_backtests = {}

    for varset_name in varsets:
        if varset_name not in VARIABLE_SETS:
            print(f"\nWarning: Variable set '{varset_name}' not found, skipping")
            continue

        try:
            # Fit model
            results = fit_tvp_var_for_varset(varset_name, config)
            all_results[varset_name] = results

            # Generate plots
            generate_plots(results, break_dates, segment_stats)

            # Run backtest (optional)
            if skip_backtest:
                print(f"\n  Skipping rolling backtest")
                backtest_results = {}
            else:
                backtest_results = run_rolling_backtest(varset_name, config)
            all_backtests[varset_name] = backtest_results

            # Save results
            save_results(results, backtest_results, break_dates, segment_stats)

        except Exception as e:
            print(f"\nError processing {varset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Create combined summary
    if all_results:
        summary_df = create_combined_summary(all_results, all_backtests)

    # Final summary
    print("\n" + "=" * 70)
    print("EXECUTION COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"\nVariable sets processed: {list(all_results.keys())}")

    return {
        "results": all_results,
        "backtests": all_backtests
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TVP-VAR estimation")
    parser.add_argument(
        "--varsets",
        nargs="+",
        default=None,
        help="Variable sets to run (default: parsimonious, bop, monetary)"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use reduced MCMC draws for faster execution"
    )
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Skip rolling backtest (much faster)"
    )

    args = parser.parse_args()

    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    main(varsets=args.varsets, fast=args.fast, skip_backtest=args.skip_backtest)
