#!/usr/bin/env python3
"""
Run Factor-Augmented VAR (FAVAR) for Reserves Forecasting.

This script implements Spec 05: FAVAR using pre-extracted PCA factors.
Following Stock & Watson (2002) and Bernanke et al. (2005):
1. Uses PCA factors from Phase 1 (varset_pca)
2. Estimates VAR on [reserves, PC1, PC2, PC3]
3. Computes IRFs and FEVD
4. Performs rolling backtest

Usage:
    python run_factor_var.py [--n-factors N] [--n-lags N] [--verbose]

Output:
    data/forecast_results_academic/favar/
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models.favar import (
    FAVAR,
    rolling_favar_forecast,
    compute_forecast_metrics,
)
from models.factor_selection import (
    bai_ng_criteria,
    kaiser_criterion,
    elbow_detection,
    select_n_factors,
)

# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PCA_DATA_DIR = DATA_DIR / "forecast_prep_academic" / "varset_pca"
OUTPUT_DIR = DATA_DIR / "forecast_results_academic" / "favar"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Model parameters
DEFAULT_N_FACTORS = 3
DEFAULT_N_LAGS = 2
FORECAST_HORIZONS = [1, 3, 6, 12]

# Time split
TRAIN_END = pd.Timestamp("2019-12-01")
VALID_END = pd.Timestamp("2022-12-01")


def load_pca_data(verbose: bool = False) -> Dict[str, Any]:
    """
    Load pre-extracted PCA factors and loadings from Phase 1.

    Returns
    -------
    dict
        Dictionary with keys: var_system, loadings, variance_explained, metadata
    """
    if verbose:
        print(f"Loading PCA data from: {PCA_DATA_DIR}")

    # Load VAR system data (reserves + PCs)
    var_system_path = PCA_DATA_DIR / "var_system.csv"
    if not var_system_path.exists():
        raise FileNotFoundError(f"VAR system data not found: {var_system_path}")

    var_system = pd.read_csv(var_system_path, parse_dates=["date"], index_col="date")

    # Load factor loadings
    loadings_path = PCA_DATA_DIR / "pca_loadings.csv"
    loadings = pd.read_csv(loadings_path, index_col=0)

    # Load variance explained
    var_explained_path = PCA_DATA_DIR / "pca_variance_explained.csv"
    var_explained = pd.read_csv(var_explained_path)

    # Load metadata
    metadata_path = PCA_DATA_DIR / "metadata.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    if verbose:
        print(f"  Loaded {len(var_system)} observations")
        print(f"  Variables: {list(var_system.columns)}")
        print(f"  Date range: {var_system.index.min()} to {var_system.index.max()}")
        print(f"  Factor loadings: {loadings.shape}")

    return {
        "var_system": var_system,
        "loadings": loadings,
        "variance_explained": var_explained["variance_explained_pct"].values / 100,
        "metadata": metadata,
    }


def run_factor_selection(
    loadings: pd.DataFrame,
    var_system: pd.DataFrame,
    variance_explained: np.ndarray,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run factor selection analysis using the pre-extracted PCA results.

    Since the source data has significant missing values that prevent
    direct Bai-Ng analysis, we use the already-extracted PCA results
    from Phase 1 to validate factor selection.
    """
    if verbose:
        print("\nFactor selection based on pre-extracted PCA results...")

    # Get number of factors from loadings
    n_factors = loadings.shape[1]

    # Kaiser rule: eigenvalue > 1 on standardized data
    # Since PCA was done on standardized data, variance_explained * n_vars gives eigenvalues
    n_vars = len(loadings)
    eigenvalues = variance_explained * n_vars

    n_kaiser = int(np.sum(eigenvalues > 1))

    # Cumulative variance thresholds
    cumvar = np.cumsum(variance_explained)
    n_70pct = int(np.argmax(cumvar >= 0.70) + 1) if np.any(cumvar >= 0.70) else n_factors
    n_80pct = int(np.argmax(cumvar >= 0.80) + 1) if np.any(cumvar >= 0.80) else n_factors

    # Elbow detection on eigenvalues
    if len(eigenvalues) > 2:
        first_diffs = -np.diff(eigenvalues)
        second_diffs = np.diff(first_diffs)
        n_elbow = int(np.argmax(second_diffs) + 2) if len(second_diffs) > 0 else n_factors
        n_elbow = max(1, min(n_elbow, n_factors))
    else:
        first_diffs = []
        second_diffs = []
        n_elbow = n_factors

    results = {
        "n_factors_used": n_factors,
        "eigenvalues": eigenvalues.tolist(),
        "variance_explained": variance_explained.tolist(),
        "cumulative_variance": cumvar.tolist(),
        "recommendations": {
            "Kaiser": n_kaiser,
            "Elbow": n_elbow,
            "70pct_variance": n_70pct,
            "80pct_variance": n_80pct,
        },
        "note": "Factor selection based on pre-extracted PCA from Phase 1",
    }

    if verbose:
        print(f"  Number of factors used: {n_factors}")
        print(f"  Eigenvalues: {[f'{e:.2f}' for e in eigenvalues]}")
        print(f"  Cumulative variance: {cumvar[-1]*100:.1f}%")
        print(f"  Kaiser rule suggests: {n_kaiser} factors")
        print(f"  Elbow suggests: {n_elbow} factors")
        print(f"  70% variance threshold: {n_70pct} factors")
        print(f"  80% variance threshold: {n_80pct} factors")

    return results


def fit_favar_model(
    var_system: pd.DataFrame,
    loadings: pd.DataFrame,
    variance_explained: np.ndarray,
    n_factors: int = 3,
    n_lags: int = 2,
    verbose: bool = False
) -> FAVAR:
    """
    Fit FAVAR model on the full training sample.
    """
    if verbose:
        print(f"\nFitting FAVAR with {n_factors} factors, {n_lags} lags...")

    # Extract reserves and factors
    Y = var_system["gross_reserves_usd_m"]
    factor_cols = [c for c in var_system.columns if c.startswith("PC")]
    factors = var_system[factor_cols]

    # Fit model
    model = FAVAR(n_factors=n_factors, n_lags=n_lags)
    model.fit(
        Y=Y,
        factors=factors,
        loadings=loadings,
        variance_explained=variance_explained,
        train_end=TRAIN_END,
    )

    if verbose:
        stats = model.get_summary_stats()
        print(f"  Observations: {stats['n_obs']}")
        print(f"  AIC: {stats['aic']:.2f}")
        print(f"  BIC: {stats['bic']:.2f}")
        print(f"  Stable: {stats['is_stable']}")
        print(f"  Max eigenvalue modulus: {stats['max_eigenvalue_modulus']:.4f}")

    return model


def run_rolling_backtest(
    var_system: pd.DataFrame,
    loadings: pd.DataFrame,
    variance_explained: np.ndarray,
    n_factors: int = 3,
    n_lags: int = 2,
    horizons: list = None,
    verbose: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Run rolling backtest for multiple forecast horizons.
    """
    if horizons is None:
        horizons = FORECAST_HORIZONS

    if verbose:
        print(f"\nRunning rolling backtest for horizons: {horizons}...")

    Y = var_system["gross_reserves_usd_m"]
    factor_cols = [c for c in var_system.columns if c.startswith("PC")]
    factors = var_system[factor_cols]

    results = {}
    metrics = {}

    for h in horizons:
        if verbose:
            print(f"  Horizon h={h}...")

        backtest = rolling_favar_forecast(
            Y=Y,
            factors=factors,
            loadings=loadings,
            variance_explained=variance_explained,
            train_end=TRAIN_END,
            test_end=var_system.index.max(),
            n_factors=n_factors,
            n_lags=n_lags,
            h=h,
            expanding=True,
        )

        results[f"h{h}"] = backtest
        metrics[f"h{h}"] = compute_forecast_metrics(backtest)

        if verbose:
            m = metrics[f"h{h}"]
            print(f"    RMSE: {m['RMSE']:.2f}, MAE: {m['MAE']:.2f}, MAPE: {m['MAPE']:.2f}%")

    return {"results": results, "metrics": metrics}


def compute_irf_fevd(
    model: FAVAR,
    periods: int = 24,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Compute impulse response functions and forecast error variance decomposition.
    """
    if verbose:
        print(f"\nComputing IRF and FEVD for {periods} periods...")

    irf = model.impulse_response(periods=periods, orthogonalized=True)
    fevd = model.variance_decomposition(periods=periods)

    if verbose:
        # Print FEVD for reserves at different horizons
        reserves_fevd = fevd["reserves"]
        print("\n  FEVD for reserves (% variance explained by each shock):")
        for h in [1, 6, 12, 24]:
            if h <= len(reserves_fevd):
                row = reserves_fevd.iloc[h - 1] * 100
                print(f"    h={h:2d}: " + ", ".join([f"{c}={row[c]:.1f}%" for c in row.index]))

    return {"irf": irf, "fevd": fevd}


def create_interpretation_table(
    model: FAVAR,
    loadings: pd.DataFrame,
    variance_explained: np.ndarray
) -> pd.DataFrame:
    """
    Create factor interpretation table for the paper.
    """
    interpretation = model.get_factor_interpretation(top_n=4)

    rows = []
    for i, (factor, top_vars) in enumerate(interpretation.items()):
        var_exp = variance_explained[i] * 100 if i < len(variance_explained) else 0

        # Format top loadings
        loadings_str = ", ".join([
            f"{var}({sign}{abs(loading):.2f})"
            for var, sign, loading in top_vars
        ])

        rows.append({
            "Factor": factor,
            "Variance_Explained_Pct": var_exp,
            "Top_Loadings": loadings_str,
            "Interpretation": _interpret_factor(top_vars),
        })

    return pd.DataFrame(rows)


def _interpret_factor(top_vars: list) -> str:
    """
    Generate economic interpretation based on top loadings.
    """
    var_names = [v[0] for v in top_vars]

    # Simple heuristics for interpretation
    if any("trade" in v.lower() for v in var_names):
        if any("balance" in v.lower() for v in var_names):
            return "Trade balance dynamics"
        return "Trade activity scale"

    if any("m2" in v.lower() for v in var_names) or any("usd_lkr" in v.lower() for v in var_names):
        return "Monetary/Exchange rate conditions"

    if any("remittance" in v.lower() for v in var_names) or any("tourism" in v.lower() for v in var_names):
        return "Service account inflows"

    if any("cse" in v.lower() for v in var_names):
        return "Capital flows/Portfolio investment"

    return "Mixed macro factor"


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_scree(factor_selection: Dict[str, Any], output_path: Path):
    """Plot scree plot with factor selection criteria."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Eigenvalues with Kaiser rule
    eigenvalues = factor_selection.get("eigenvalues", [])
    var_explained = factor_selection.get("variance_explained", [])

    ax1 = axes[0]
    if eigenvalues:
        x = range(1, len(eigenvalues) + 1)
        ax1.bar(x, eigenvalues, alpha=0.7, label="Eigenvalue", color="steelblue")
        ax1.axhline(y=1, color="r", linestyle="--", label="Kaiser threshold", linewidth=2)
        ax1.set_xlabel("Component")
        ax1.set_ylabel("Eigenvalue")
        ax1.set_title("Scree Plot with Kaiser Rule")
        ax1.set_xticks(list(x))
        ax1.legend()

    # Right: Cumulative variance explained
    ax2 = axes[1]
    cumvar = factor_selection.get("cumulative_variance", [])

    if cumvar:
        x = range(1, len(cumvar) + 1)
        ax2.bar(x, [v * 100 for v in var_explained], alpha=0.7, label="Individual", color="steelblue")
        ax2.plot(x, [v * 100 for v in cumvar], "ro-", label="Cumulative", linewidth=2, markersize=8)
        ax2.axhline(y=70, color="g", linestyle="--", alpha=0.7, label="70% threshold")
        ax2.axhline(y=80, color="orange", linestyle="--", alpha=0.7, label="80% threshold")
        ax2.set_xlabel("Number of Components")
        ax2.set_ylabel("Variance Explained (%)")
        ax2.set_title("Variance Explained by Components")
        ax2.set_xticks(list(x))
        ax2.legend(loc="center right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_factor_paths(var_system: pd.DataFrame, output_path: Path):
    """Plot factor time series."""
    factor_cols = [c for c in var_system.columns if c.startswith("PC")]

    fig, axes = plt.subplots(len(factor_cols), 1, figsize=(12, 3 * len(factor_cols)), sharex=True)
    if len(factor_cols) == 1:
        axes = [axes]

    for i, col in enumerate(factor_cols):
        ax = axes[i]
        ax.plot(var_system.index, var_system[col], "b-", linewidth=0.8)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(x=TRAIN_END, color="r", linestyle="--", alpha=0.7, label="Train/Test split")
        ax.set_ylabel(col)
        ax.set_title(f"Factor {col}")
        if i == 0:
            ax.legend()

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_loadings_heatmap(loadings: pd.DataFrame, output_path: Path):
    """Plot factor loadings heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        loadings,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        ax=ax,
        cbar_kws={"label": "Loading"}
    )

    ax.set_title("Factor Loadings Matrix")
    ax.set_xlabel("Factor")
    ax.set_ylabel("Variable")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_irf_reserves(irf: Dict[str, pd.DataFrame], output_path: Path):
    """Plot impulse responses of reserves to all shocks."""
    reserves_irf = irf.get("reserves")
    if reserves_irf is None:
        return

    shock_cols = reserves_irf.columns
    n_shocks = len(shock_cols)

    fig, axes = plt.subplots(1, n_shocks, figsize=(4 * n_shocks, 4), sharey=True)
    if n_shocks == 1:
        axes = [axes]

    for i, col in enumerate(shock_cols):
        ax = axes[i]
        response = reserves_irf[col]
        ax.plot(response.index, response.values, "b-", linewidth=1.5)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.fill_between(response.index, 0, response.values, alpha=0.3)
        ax.set_xlabel("Periods")
        ax.set_title(col.replace("shock_", "Shock: "))
        if i == 0:
            ax.set_ylabel("Reserves Response")

    plt.suptitle("Impulse Response: Reserves to Shocks", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_fevd_reserves(fevd: Dict[str, pd.DataFrame], output_path: Path):
    """Plot forecast error variance decomposition for reserves."""
    reserves_fevd = fevd.get("reserves")
    if reserves_fevd is None:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Stacked area plot
    reserves_fevd_pct = reserves_fevd * 100
    reserves_fevd_pct.plot.area(ax=ax, alpha=0.7, stacked=True)

    ax.set_xlabel("Forecast Horizon")
    ax.set_ylabel("Variance Share (%)")
    ax.set_title("Forecast Error Variance Decomposition: Reserves")
    ax.legend(title="Shock", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_xlim(0, len(reserves_fevd) - 1)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_backtest_results(backtest: Dict[str, pd.DataFrame], output_path: Path):
    """Plot rolling backtest actual vs forecast."""
    n_horizons = len(backtest)
    fig, axes = plt.subplots(n_horizons, 1, figsize=(12, 4 * n_horizons), sharex=True)
    if n_horizons == 1:
        axes = [axes]

    for i, (horizon, results) in enumerate(backtest.items()):
        ax = axes[i]

        ax.plot(results["date"], results["actual"], "b-", label="Actual", linewidth=1.5)
        ax.plot(results["date"], results["forecast"], "r--", label="Forecast", linewidth=1.5)

        ax.set_ylabel("Reserves (USD M)")
        ax.set_title(f"Rolling Backtest: {horizon}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run FAVAR model for reserves forecasting"
    )
    parser.add_argument(
        "--n-factors", type=int, default=DEFAULT_N_FACTORS,
        help=f"Number of factors (default: {DEFAULT_N_FACTORS})"
    )
    parser.add_argument(
        "--n-lags", type=int, default=DEFAULT_N_LAGS,
        help=f"VAR lag order (default: {DEFAULT_N_LAGS})"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose output"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("FAVAR Model for Reserves Forecasting")
    print("=" * 70)
    print(f"N Factors: {args.n_factors}")
    print(f"N Lags: {args.n_lags}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load pre-extracted PCA data
    pca_data = load_pca_data(verbose=args.verbose)
    var_system = pca_data["var_system"]
    loadings = pca_data["loadings"]
    variance_explained = pca_data["variance_explained"]
    metadata = pca_data["metadata"]

    # Run factor selection analysis
    print("\n" + "-" * 70)
    print("Factor Selection Analysis")
    print("-" * 70)
    factor_selection = run_factor_selection(
        loadings=loadings,
        var_system=var_system,
        variance_explained=variance_explained,
        verbose=args.verbose
    )

    # Save factor selection results
    factor_selection_path = OUTPUT_DIR / "favar_factor_selection.json"
    # Convert numpy arrays to lists for JSON serialization
    factor_selection_json = {
        k: (v.tolist() if isinstance(v, np.ndarray) else
            ({kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in v.items()} if isinstance(v, dict) else v))
        for k, v in factor_selection.items()
    }
    with open(factor_selection_path, "w") as f:
        json.dump(factor_selection_json, f, indent=2, default=str)

    # Plot scree
    if "eigenvalues" in factor_selection:
        plot_scree(factor_selection, FIGURES_DIR / "scree_plot.png")
        if args.verbose:
            print("  Saved scree plot")

    # Fit FAVAR model
    print("\n" + "-" * 70)
    print("FAVAR Model Estimation")
    print("-" * 70)
    model = fit_favar_model(
        var_system=var_system,
        loadings=loadings,
        variance_explained=variance_explained,
        n_factors=args.n_factors,
        n_lags=args.n_lags,
        verbose=args.verbose,
    )

    # Save factors
    factors_path = OUTPUT_DIR / "favar_factors.csv"
    model.factors.to_csv(factors_path)

    # Save loadings
    loadings_path = OUTPUT_DIR / "favar_loadings.csv"
    loadings.to_csv(loadings_path)

    # Create interpretation table
    interpretation_df = create_interpretation_table(model, loadings, variance_explained)
    interpretation_path = OUTPUT_DIR / "favar_interpretation.csv"
    interpretation_df.to_csv(interpretation_path, index=False)

    if args.verbose:
        print("\nFactor Interpretation:")
        print(interpretation_df.to_string(index=False))

    # Save as JSON too
    interpretation_json = interpretation_df.to_dict(orient="records")
    with open(OUTPUT_DIR / "favar_interpretation.json", "w") as f:
        json.dump(interpretation_json, f, indent=2)

    # Compute IRF and FEVD
    print("\n" + "-" * 70)
    print("Impulse Response and Variance Decomposition")
    print("-" * 70)
    irf_fevd = compute_irf_fevd(model, periods=24, verbose=args.verbose)

    # Save IRF for reserves
    reserves_irf = irf_fevd["irf"]["reserves"]
    irf_path = OUTPUT_DIR / "favar_irf.csv"
    reserves_irf.to_csv(irf_path)

    # Save FEVD for reserves
    reserves_fevd = irf_fevd["fevd"]["reserves"]
    fevd_path = OUTPUT_DIR / "favar_fevd.csv"
    reserves_fevd.to_csv(fevd_path)

    # Run rolling backtest
    print("\n" + "-" * 70)
    print("Rolling Backtest")
    print("-" * 70)
    backtest = run_rolling_backtest(
        var_system=var_system,
        loadings=loadings,
        variance_explained=variance_explained,
        n_factors=args.n_factors,
        n_lags=args.n_lags,
        horizons=FORECAST_HORIZONS,
        verbose=args.verbose,
    )

    # Save backtest results
    for horizon, results in backtest["results"].items():
        results.to_csv(OUTPUT_DIR / f"favar_rolling_backtest_{horizon}.csv", index=False)

    # Save combined backtest metrics
    metrics_df = pd.DataFrame(backtest["metrics"]).T
    metrics_df.index.name = "horizon"
    metrics_path = OUTPUT_DIR / "favar_rolling_backtest.csv"
    metrics_df.to_csv(metrics_path)

    if args.verbose:
        print("\nBacktest Metrics:")
        print(metrics_df.to_string())

    # Generate forecasts
    print("\n" + "-" * 70)
    print("Generating Forecasts")
    print("-" * 70)
    forecasts = model.forecast_with_intervals(h=12, alpha=0.1)

    forecasts_df = pd.DataFrame({
        "forecast": forecasts["mean"]["reserves"],
        "lower_90": forecasts["lower"]["reserves"],
        "upper_90": forecasts["upper"]["reserves"],
    })
    forecasts_path = OUTPUT_DIR / "favar_forecasts.csv"
    forecasts_df.to_csv(forecasts_path)

    if args.verbose:
        print("\n12-month ahead forecasts:")
        print(forecasts_df.to_string())

    # Generate all figures
    print("\n" + "-" * 70)
    print("Generating Figures")
    print("-" * 70)

    plot_factor_paths(var_system, FIGURES_DIR / "factor_paths.png")
    plot_loadings_heatmap(loadings, FIGURES_DIR / "loadings_heatmap.png")
    plot_irf_reserves(irf_fevd["irf"], FIGURES_DIR / "irf_reserves.png")
    plot_fevd_reserves(irf_fevd["fevd"], FIGURES_DIR / "fevd_reserves.png")
    plot_backtest_results(backtest["results"], FIGURES_DIR / "backtest_results.png")

    if args.verbose:
        print("  Saved all figures to:", FIGURES_DIR)

    # Save model summary
    summary = {
        "model": "FAVAR",
        "n_factors": args.n_factors,
        "n_lags": model.var_results.k_ar,
        "train_end": str(TRAIN_END.date()),
        "valid_end": str(VALID_END.date()),
        "model_stats": model.get_summary_stats(),
        "factor_selection": factor_selection.get("recommendations", {}),
        "backtest_metrics": backtest["metrics"],
        "created_at": datetime.now().isoformat(),
    }

    summary_path = OUTPUT_DIR / "favar_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 70)
    print("FAVAR Execution Complete")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"\nFiles created:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        if f.is_file():
            print(f"  - {f.name}")
    print(f"\nFigures created:")
    for f in sorted(FIGURES_DIR.glob("*.png")):
        print(f"  - {f.name}")

    print("\n" + "-" * 70)
    print("Rolling Backtest Summary")
    print("-" * 70)
    print(metrics_df.to_string())

    return summary


if __name__ == "__main__":
    main()
