#!/usr/bin/env python3
"""
Run MIDAS Models for Reserves Forecasting.

This script implements Mixed Data Sampling (MIDAS) regression to exploit
daily exchange rate data for predicting monthly reserves.

Features:
1. Loads and prepares daily USD/LKR exchange rate data
2. Fits MIDAS models with different weighting schemes (Almon, Beta)
3. Compares with monthly-only baseline models
4. Performs rolling backtests
5. Generates visualizations

Reference: Specification 07 - MIDAS

Usage:
    python run_midas.py [--varset NAME] [--verbose]
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "models"))

from models.midas import MIDAS, MIDAS_AR, UMIDAS, prepare_hf_exchange_rate, midas_information_gain
from models.midas_weights import exp_almon_weights, beta_weights, step_weights, uniform_weights

# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
ACADEMIC_PREP_DIR = DATA_DIR / "forecast_prep_academic"
OUTPUT_DIR = DATA_DIR / "forecast_results_academic" / "midas"
FIGURES_DIR = OUTPUT_DIR / "figures"

# MIDAS Configuration
N_HF_LAGS = 22  # Trading days per month
N_LF_LAGS = 3   # Monthly lags of HF data to include
WEIGHT_TYPES = ["exp_almon", "beta"]

# Backtesting configuration
BACKTEST_START = pd.Timestamp("2022-01-01")
FORECAST_HORIZONS = [1, 3, 6]


def load_daily_exchange_rate(verbose: bool = False) -> pd.Series:
    """
    Load daily USD/LKR exchange rate data from raw file.

    The raw file is an HTML table disguised as .xls.
    """
    fx_file = RAW_DATA_DIR / "USD Spot Rate.xls"

    if not fx_file.exists():
        raise FileNotFoundError(f"Daily FX data not found: {fx_file}")

    if verbose:
        print(f"Loading daily FX data from: {fx_file}")

    # Read HTML table
    dfs = pd.read_html(str(fx_file))
    df = dfs[1]  # Second table contains the data

    # Extract date columns (from column 4 onwards)
    date_cols = [c for c in df.columns if '20' in str(c)]

    # Get the USD Spot Rate row (row 1)
    rate_row = df.iloc[1, 4:].values

    # Create series
    dates = pd.to_datetime(date_cols)
    rates = pd.to_numeric(rate_row, errors='coerce')
    fx_series = pd.Series(rates, index=dates, name='usd_lkr').dropna()

    if verbose:
        print(f"  Loaded {len(fx_series)} daily observations")
        print(f"  Date range: {fx_series.index.min().date()} to {fx_series.index.max().date()}")

    return fx_series


def load_monthly_data(varset: str = "parsimonious", verbose: bool = False) -> pd.DataFrame:
    """Load monthly reserves data from academic prep directory."""
    data_path = ACADEMIC_PREP_DIR / f"varset_{varset}" / "arima_dataset.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Monthly data not found: {data_path}")

    if verbose:
        print(f"Loading monthly data from: {data_path}")

    df = pd.read_csv(data_path, parse_dates=["date"], index_col="date")

    if verbose:
        print(f"  Loaded {len(df)} monthly observations")
        print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
        print(f"  Variables: {list(df.columns)}")

    return df


def prepare_midas_data(
    monthly_df: pd.DataFrame,
    daily_fx: pd.Series,
    target_col: str = "gross_reserves_usd_m",
    verbose: bool = False
) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Prepare data for MIDAS estimation.

    Returns:
        Y_monthly: Target variable (monthly)
        X_daily: High-frequency regressor (daily FX returns)
        X_monthly: Additional monthly regressors
    """
    # Determine overlap period
    fx_start = daily_fx.index.min()
    fx_end = daily_fx.index.max()
    monthly_start = monthly_df.index.min()
    monthly_end = monthly_df.index.max()

    overlap_start = max(fx_start, monthly_start)
    overlap_end = min(fx_end, monthly_end)

    if verbose:
        print(f"\nData alignment:")
        print(f"  Monthly data: {monthly_start.date()} to {monthly_end.date()}")
        print(f"  Daily FX: {fx_start.date()} to {fx_end.date()}")
        print(f"  Overlap: {overlap_start.date()} to {overlap_end.date()}")

    # Filter monthly data to overlap period
    monthly_overlap = monthly_df[
        (monthly_df.index >= overlap_start) &
        (monthly_df.index <= overlap_end)
    ].copy()

    # Extend daily FX backward for lag computation
    # Need N_LF_LAGS months of history before first monthly observation
    fx_need_start = overlap_start - pd.DateOffset(months=N_LF_LAGS + 1)
    daily_fx_aligned = daily_fx[daily_fx.index >= fx_need_start].copy()

    # Compute log returns for stationarity
    daily_returns = np.log(daily_fx_aligned).diff().dropna()
    daily_returns.name = 'usd_lkr_return'

    # Fill weekends/holidays
    daily_returns = daily_returns.resample('D').ffill()

    # Extract target and exogenous monthly variables
    Y_monthly = monthly_overlap[target_col].copy()

    # Get additional monthly regressors (exclude target and usd_lkr since we use daily)
    exog_cols = [c for c in monthly_overlap.columns
                 if c not in [target_col, 'usd_lkr']]

    if len(exog_cols) > 0:
        X_monthly = monthly_overlap[exog_cols].copy()
    else:
        X_monthly = None

    if verbose:
        print(f"  Monthly observations for MIDAS: {len(Y_monthly)}")
        print(f"  Daily returns observations: {len(daily_returns)}")
        if X_monthly is not None:
            print(f"  Additional monthly regressors: {list(X_monthly.columns)}")

    return Y_monthly, daily_returns, X_monthly


def fit_midas_models(
    Y_monthly: pd.Series,
    X_daily: pd.Series,
    X_monthly: Optional[pd.DataFrame] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Fit MIDAS models with different weight functions.
    """
    results = {}

    for weight_type in WEIGHT_TYPES:
        if verbose:
            print(f"\n  Fitting MIDAS with {weight_type} weights...")

        try:
            model = MIDAS(
                weight_type=weight_type,
                n_hf_lags=N_HF_LAGS,
                n_lf_lags=N_LF_LAGS
            )
            model.fit(Y_monthly, X_daily, X_monthly)

            results[weight_type] = {
                "model": model,
                "summary": model.summary(),
                "success": True
            }

            if verbose:
                summary = model.summary()
                print(f"    R-squared: {summary['r2']:.4f}")
                print(f"    RMSE: {summary['rmse']:.2f}")
                print(f"    Theta: {summary['theta']}")

        except Exception as e:
            if verbose:
                print(f"    Error: {e}")
            results[weight_type] = {"success": False, "error": str(e)}

    # Also fit MIDAS-AR
    if verbose:
        print(f"\n  Fitting MIDAS-AR (with 1 AR lag)...")

    try:
        midas_ar = MIDAS_AR(
            n_ar_lags=1,
            weight_type="exp_almon",
            n_hf_lags=N_HF_LAGS,
            n_lf_lags=N_LF_LAGS
        )
        midas_ar.fit(Y_monthly, X_daily, X_monthly)

        results["midas_ar"] = {
            "model": midas_ar,
            "summary": midas_ar.summary(),
            "success": True
        }

        if verbose:
            summary = midas_ar.summary()
            print(f"    R-squared: {summary['r2']:.4f}")
            print(f"    AR coef: {summary['ar_coefs']}")

    except Exception as e:
        if verbose:
            print(f"    Error: {e}")
        results["midas_ar"] = {"success": False, "error": str(e)}

    return results


def fit_monthly_baseline(
    Y_monthly: pd.Series,
    X_monthly: Optional[pd.DataFrame] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Fit baseline monthly-only model (AR + monthly regressors).
    """
    if verbose:
        print("\n  Fitting monthly baseline (AR model)...")

    # Create AR lags
    ar_lags = 3
    X = pd.DataFrame(index=Y_monthly.index)
    for lag in range(1, ar_lags + 1):
        X[f"Y_lag{lag}"] = Y_monthly.shift(lag)

    if X_monthly is not None:
        X = pd.concat([X, X_monthly], axis=1)

    # Add intercept
    X = X.dropna()
    Y = Y_monthly.loc[X.index]

    # OLS
    X_mat = np.column_stack([np.ones(len(X)), X.values])
    Y_vec = Y.values

    beta, _, _, _ = np.linalg.lstsq(X_mat, Y_vec, rcond=None)
    fitted = X_mat @ beta
    residuals = Y_vec - fitted

    ssr = np.sum(residuals ** 2)
    sst = np.sum((Y_vec - Y_vec.mean()) ** 2)
    r2 = 1 - ssr / sst
    rmse = np.sqrt(ssr / len(Y_vec))

    if verbose:
        print(f"    R-squared: {r2:.4f}")
        print(f"    RMSE: {rmse:.2f}")

    return {
        "beta": beta,
        "r2": r2,
        "rmse": rmse,
        "n_obs": len(Y_vec),
        "fitted": fitted,
        "residuals": residuals
    }


def rolling_backtest(
    Y_monthly: pd.Series,
    X_daily: pd.Series,
    X_monthly: Optional[pd.DataFrame] = None,
    backtest_start: pd.Timestamp = BACKTEST_START,
    horizons: list = [1, 3, 6],
    verbose: bool = False
) -> pd.DataFrame:
    """
    Perform rolling backtest of MIDAS models.

    Includes:
    - Pure MIDAS (without AR terms) - typically worse
    - MIDAS-AR (with AR(1) term) - should match/beat baseline
    - AR baseline for comparison
    """
    if verbose:
        print(f"\nRunning rolling backtest from {backtest_start.date()}...")

    results = []

    # Get backtest dates
    backtest_dates = Y_monthly[Y_monthly.index >= backtest_start].index

    if len(backtest_dates) < 2:
        if verbose:
            print("  Insufficient data for backtest")
        return pd.DataFrame()

    n_origins = len(backtest_dates[:-max(horizons)])
    if verbose:
        print(f"  Number of forecast origins: {n_origins}")

    for i, origin in enumerate(backtest_dates[:-max(horizons)]):
        # Training data up to origin
        train_mask = Y_monthly.index < origin
        Y_train = Y_monthly[train_mask]
        X_daily_train = X_daily[X_daily.index < origin]

        if X_monthly is not None:
            X_monthly_train = X_monthly[train_mask]
        else:
            X_monthly_train = None

        if len(Y_train) < 24:  # Need at least 2 years for training
            continue

        # Fit MIDAS-AR models (most relevant comparison)
        for weight_type in ["exp_almon", "beta"]:
            try:
                model = MIDAS_AR(
                    n_ar_lags=1,
                    weight_type=weight_type,
                    n_hf_lags=N_HF_LAGS,
                    n_lf_lags=N_LF_LAGS
                )
                model.fit(Y_train, X_daily_train, X_monthly_train)

                # Forecasts for each horizon
                for h in horizons:
                    forecast_date = origin + pd.DateOffset(months=h)

                    if forecast_date in Y_monthly.index:
                        actual = Y_monthly.loc[forecast_date]
                        forecasts = model.forecast(h=h)
                        forecast = forecasts[-1]

                        results.append({
                            "origin": origin,
                            "horizon": h,
                            "model": f"MIDAS_AR_{weight_type}",
                            "forecast": forecast,
                            "actual": actual,
                            "error": forecast - actual
                        })

            except Exception:
                continue

        # Fit AR baseline for comparison
        try:
            ar_lags = 3
            X_ar = pd.DataFrame(index=Y_train.index)
            for lag in range(1, ar_lags + 1):
                X_ar[f"Y_lag{lag}"] = Y_train.shift(lag)

            if X_monthly_train is not None:
                X_ar = pd.concat([X_ar, X_monthly_train], axis=1)

            X_ar = X_ar.dropna()
            Y_ar = Y_train.loc[X_ar.index]

            X_mat = np.column_stack([np.ones(len(X_ar)), X_ar.values])
            beta_ar, _, _, _ = np.linalg.lstsq(X_mat, Y_ar.values, rcond=None)

            for h in horizons:
                forecast_date = origin + pd.DateOffset(months=h)

                if forecast_date in Y_monthly.index:
                    actual = Y_monthly.loc[forecast_date]

                    # Simple AR forecast (using last known values)
                    Y_last = Y_monthly.loc[:origin].iloc[-(ar_lags):]
                    if X_monthly is not None:
                        x_exog = X_monthly.loc[:origin].iloc[-1].values
                        x_pred = np.concatenate([[1], Y_last.values[::-1], x_exog])
                    else:
                        x_pred = np.concatenate([[1], Y_last.values[::-1]])

                    # Multi-step forecast via iteration
                    forecast = np.dot(x_pred[:len(beta_ar)], beta_ar)

                    results.append({
                        "origin": origin,
                        "horizon": h,
                        "model": "AR_baseline",
                        "forecast": forecast,
                        "actual": actual,
                        "error": forecast - actual
                    })

        except Exception:
            continue

    if len(results) == 0:
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    # Compute error metrics by model and horizon
    if verbose:
        print("\n  Backtest Results:")
        summary = results_df.groupby(["model", "horizon"]).agg({
            "error": [
                ("MAE", lambda x: np.abs(x).mean()),
                ("RMSE", lambda x: np.sqrt((x**2).mean())),
                ("N", "count")
            ]
        })
        print(summary.to_string())

    return results_df


def plot_weight_functions(
    midas_results: Dict[str, Any],
    save_path: Optional[Path] = None,
    verbose: bool = False
) -> None:
    """Plot estimated MIDAS weight functions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    lags = np.arange(1, N_HF_LAGS + 1)

    for idx, weight_type in enumerate(WEIGHT_TYPES):
        ax = axes[idx]

        if weight_type in midas_results and midas_results[weight_type]["success"]:
            model = midas_results[weight_type]["model"]
            weights = model.weights
            theta = model.theta

            ax.bar(lags, weights, alpha=0.7, color='steelblue')
            ax.set_xlabel("Lag (trading days)")
            ax.set_ylabel("Weight")

            if weight_type == "exp_almon":
                title = f"Exponential Almon Weights\n(θ₁={theta[0]:.3f}, θ₂={theta[1]:.4f})"
            else:
                title = f"Beta Weights\n(α={theta[0]:.2f}, β={theta[1]:.2f})"

            ax.set_title(title)
            ax.grid(True, alpha=0.3)

        else:
            ax.text(0.5, 0.5, "Model not fitted",
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{weight_type} Weights")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if verbose:
            print(f"  Saved weight plot: {save_path}")

    plt.close()


def plot_comparison(
    Y_monthly: pd.Series,
    midas_results: Dict[str, Any],
    baseline_results: Dict[str, Any],
    save_path: Optional[Path] = None,
    verbose: bool = False
) -> None:
    """Plot MIDAS vs baseline comparison."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Get fitted values
    dates = Y_monthly.index

    # Plot 1: Actual vs Fitted
    ax1 = axes[0]
    ax1.plot(dates, Y_monthly.values, 'k-', label='Actual', linewidth=1.5)

    for weight_type in WEIGHT_TYPES:
        if weight_type in midas_results and midas_results[weight_type]["success"]:
            model = midas_results[weight_type]["model"]
            # Get valid dates
            valid_dates = dates[model.valid_mask]
            ax1.plot(valid_dates, model.fitted, '--',
                    label=f'MIDAS ({weight_type})', alpha=0.8)

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Gross Reserves (USD millions)")
    ax1.set_title("Reserves: Actual vs MIDAS Fitted Values")
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Model comparison bar chart
    ax2 = axes[1]

    models = []
    rmses = []
    r2s = []

    # Baseline
    models.append("AR Baseline")
    rmses.append(baseline_results["rmse"])
    r2s.append(baseline_results["r2"])

    # MIDAS models
    for weight_type in WEIGHT_TYPES:
        if weight_type in midas_results and midas_results[weight_type]["success"]:
            summary = midas_results[weight_type]["summary"]
            models.append(f"MIDAS\n({weight_type})")
            rmses.append(summary["rmse"])
            r2s.append(summary["r2"])

    if "midas_ar" in midas_results and midas_results["midas_ar"]["success"]:
        summary = midas_results["midas_ar"]["summary"]
        models.append("MIDAS-AR")
        rmses.append(summary["rmse"])
        r2s.append(summary["r2"])

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax2.bar(x - width/2, rmses, width, label='RMSE', color='coral')
    ax2_twin = ax2.twinx()
    bars2 = ax2_twin.bar(x + width/2, r2s, width, label='R²', color='steelblue')

    ax2.set_xlabel("Model")
    ax2.set_ylabel("RMSE", color='coral')
    ax2_twin.set_ylabel("R²", color='steelblue')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.set_title("Model Performance Comparison")

    # Add legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if verbose:
            print(f"  Saved comparison plot: {save_path}")

    plt.close()


def save_results(
    midas_results: Dict[str, Any],
    baseline_results: Dict[str, Any],
    backtest_df: pd.DataFrame,
    output_dir: Path,
    verbose: bool = False
) -> None:
    """Save all results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model summaries
    summaries = []
    for model_name, result in midas_results.items():
        if result.get("success"):
            summary = result["summary"].copy()
            summary["model"] = model_name
            summaries.append(summary)

    # Add baseline
    baseline_summary = {
        "model": "ar_baseline",
        "r2": baseline_results["r2"],
        "rmse": baseline_results["rmse"],
        "n_obs": baseline_results["n_obs"]
    }
    summaries.append(baseline_summary)

    summaries_df = pd.DataFrame(summaries)
    summaries_df.to_csv(output_dir / "midas_model_summaries.csv", index=False)

    # Save weight estimates
    weights_data = []
    for weight_type in WEIGHT_TYPES:
        if weight_type in midas_results and midas_results[weight_type]["success"]:
            model = midas_results[weight_type]["model"]
            for lag, weight in enumerate(model.weights, 1):
                weights_data.append({
                    "weight_type": weight_type,
                    "lag": lag,
                    "weight": weight
                })

    if weights_data:
        weights_df = pd.DataFrame(weights_data)
        weights_df.to_csv(output_dir / "midas_weights.csv", index=False)

    # Save coefficients
    coefs_data = {}
    for model_name, result in midas_results.items():
        if result.get("success"):
            summary = result["summary"]
            coefs_data[model_name] = {
                "theta": summary.get("theta"),
                "intercept": summary.get("intercept"),
                "hf_coefs": summary.get("hf_coefs"),
                "ar_coefs": summary.get("ar_coefs")
            }

    with open(output_dir / "midas_coefficients.json", "w") as f:
        json.dump(coefs_data, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))

    # Save backtest results
    if len(backtest_df) > 0:
        backtest_df.to_csv(output_dir / "midas_rolling_backtest.csv", index=False)

    if verbose:
        print(f"\n  Results saved to: {output_dir}")


def compute_information_gain(
    midas_results: Dict[str, Any],
    baseline_results: Dict[str, Any]
) -> Dict[str, float]:
    """Compute information gain from using daily data."""
    gains = {}
    baseline_rmse = baseline_results["rmse"]

    for weight_type in WEIGHT_TYPES:
        if weight_type in midas_results and midas_results[weight_type]["success"]:
            midas_rmse = midas_results[weight_type]["summary"]["rmse"]
            gains[weight_type] = midas_information_gain(midas_rmse, baseline_rmse)

    if "midas_ar" in midas_results and midas_results["midas_ar"]["success"]:
        midas_rmse = midas_results["midas_ar"]["summary"]["rmse"]
        gains["midas_ar"] = midas_information_gain(midas_rmse, baseline_rmse)

    return gains


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Run MIDAS models for reserves forecasting")
    parser.add_argument("--varset", type=str, default="parsimonious",
                       help="Variable set to use (default: parsimonious)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--no-backtest", action="store_true",
                       help="Skip rolling backtest")
    args = parser.parse_args()

    print("=" * 70)
    print("MIDAS Model Estimation")
    print("Mixed Data Sampling for Reserves Forecasting")
    print("=" * 70)
    print(f"Variable set: {args.varset}")
    print(f"HF lags per month: {N_HF_LAGS}")
    print(f"LF lag months: {N_LF_LAGS}")
    print("=" * 70)

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n[1] Loading Data")
    print("-" * 40)

    try:
        daily_fx = load_daily_exchange_rate(verbose=args.verbose)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("\nDaily exchange rate data not available.")
        print("MIDAS requires high-frequency data to be valuable.")
        sys.exit(1)

    try:
        monthly_df = load_monthly_data(varset=args.varset, verbose=args.verbose)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Prepare data for MIDAS
    Y_monthly, X_daily, X_monthly = prepare_midas_data(
        monthly_df, daily_fx, verbose=args.verbose
    )

    # Check if we have enough data
    if len(Y_monthly) < 24:
        print(f"\nWARNING: Only {len(Y_monthly)} monthly observations in overlap period.")
        print("MIDAS results may be unreliable with limited data.")

    # Fit MIDAS models
    print("\n[2] Fitting MIDAS Models")
    print("-" * 40)
    midas_results = fit_midas_models(Y_monthly, X_daily, X_monthly, verbose=args.verbose)

    # Fit baseline
    print("\n[3] Fitting Baseline Model")
    print("-" * 40)
    baseline_results = fit_monthly_baseline(Y_monthly, X_monthly, verbose=args.verbose)

    # Compute information gain
    info_gains = compute_information_gain(midas_results, baseline_results)

    print("\n[4] Information Gain from Daily Data")
    print("-" * 40)
    for model, gain in info_gains.items():
        direction = "improvement" if gain > 0 else "deterioration"
        print(f"  {model}: {gain:.2f}% {direction}")

    # Rolling backtest
    if not args.no_backtest:
        print("\n[5] Rolling Backtest")
        print("-" * 40)
        backtest_df = rolling_backtest(
            Y_monthly, X_daily, X_monthly,
            backtest_start=BACKTEST_START,
            horizons=FORECAST_HORIZONS,
            verbose=args.verbose
        )
    else:
        backtest_df = pd.DataFrame()
        print("\n[5] Rolling Backtest (skipped)")

    # Generate plots
    print("\n[6] Generating Visualizations")
    print("-" * 40)

    plot_weight_functions(
        midas_results,
        save_path=FIGURES_DIR / "midas_weight_functions.png",
        verbose=args.verbose
    )

    plot_comparison(
        Y_monthly, midas_results, baseline_results,
        save_path=FIGURES_DIR / "midas_vs_monthly.png",
        verbose=args.verbose
    )

    # Save results
    print("\n[7] Saving Results")
    print("-" * 40)
    save_results(
        midas_results, baseline_results, backtest_df,
        OUTPUT_DIR, verbose=args.verbose
    )

    # Summary
    print("\n" + "=" * 70)
    print("MIDAS Estimation Complete")
    print("=" * 70)

    print("\nModel Summary:")
    print("-" * 50)
    print(f"{'Model':<25} {'R²':>10} {'RMSE':>12}")
    print("-" * 50)
    print(f"{'AR Baseline':<25} {baseline_results['r2']:>10.4f} {baseline_results['rmse']:>12.2f}")

    for model_name, result in midas_results.items():
        if result.get("success"):
            summary = result["summary"]
            print(f"{model_name:<25} {summary['r2']:>10.4f} {summary['rmse']:>12.2f}")

    print("-" * 50)

    print(f"\nOutput directory: {OUTPUT_DIR}")

    # Return results for programmatic use
    return {
        "midas_results": {k: v["summary"] for k, v in midas_results.items() if v.get("success")},
        "baseline": baseline_results,
        "information_gains": info_gains,
        "backtest": backtest_df if len(backtest_df) > 0 else None
    }


if __name__ == "__main__":
    main()
