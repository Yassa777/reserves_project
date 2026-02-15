"""
Statistical Tests for Forecast Evaluation
==========================================

Main execution script for Spec 10.

This script:
1. Collects all available forecasts from baseline and academic models
2. Runs Diebold-Mariano tests for pairwise comparison
3. Runs Model Confidence Set procedure
4. Evaluates density forecasts (if available)
5. Runs encompassing tests
6. Generates tables and figures for the paper

Usage:
    python run_statistical_tests.py

Output:
    data/statistical_tests/
        dm_test_matrix.csv
        dm_pvalues_matrix.csv
        mcs_results.json
        mcs_summary.csv
        encompassing_tests.csv
        density_evaluation.csv (if applicable)
        figures/
            dm_heatmap.png
            mcs_elimination.png

Author: Academic Forecasting Pipeline
Date: 2026-02-10
"""

import os
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from reserves_project.scripts.academic.tests.diebold_mariano import (
    dm_test_matrix,
    dm_test_vs_benchmark,
    format_dm_table_for_paper,
    dm_test_hln,
)
from reserves_project.scripts.academic.tests.model_confidence_set import (
    model_confidence_set,
    mcs_summary_table,
    mcs_with_pvalues,
)
from reserves_project.scripts.academic.tests.density_evaluation import (
    evaluate_density_forecasts,
    density_evaluation_summary,
    pit_histogram_test,
)
from reserves_project.scripts.academic.tests.encompassing import (
    pairwise_encompassing_matrix,
    format_encompassing_table,
    encompassing_summary,
    optimal_combination_weights,
)

# =============================================================================
# Configuration
# =============================================================================

SIGNIFICANCE_LEVEL = 0.10  # For MCS
DM_LOSS_FUNCTION = "squared"  # MSE loss
BOOTSTRAP_REPS = 1000
FORECAST_HORIZON = 1  # For HAC standard errors

# Paths
BASE_DATA_DIR = PROJECT_ROOT / "data"
BASELINE_RESULTS_DIR = BASE_DATA_DIR / "forecast_results"
ACADEMIC_RESULTS_DIR = BASE_DATA_DIR / "forecast_results_academic"
OUTPUT_DIR = BASE_DATA_DIR / "statistical_tests"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Ensure output directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_baseline_forecasts() -> dict:
    """Load forecasts from baseline models."""
    forecasts = {}
    actuals = None

    # Standard forecast files
    forecast_files = {
        "Naive": "naive_forecast.csv",
        "ARIMA": "arima_forecast.csv",
        "VECM": "vecm_forecast.csv",
        "MS-VAR": "ms_var_forecast.csv",
        "MS-VECM": "ms_vecm_forecast.csv",
    }

    for model_name, filename in forecast_files.items():
        filepath = BASELINE_RESULTS_DIR / filename
        if filepath.exists():
            df = pd.read_csv(filepath, parse_dates=["date"])
            df = df.set_index("date")

            # Get test set only
            if "split" in df.columns:
                test_df = df[df["split"] == "test"]
            else:
                test_df = df

            if actuals is None and "actual" in test_df.columns:
                actuals = test_df["actual"].values

            if "forecast" in test_df.columns:
                forecasts[model_name] = test_df["forecast"].values

            print(f"Loaded {model_name}: {len(forecasts.get(model_name, []))} observations")

    return forecasts, actuals


def load_bvar_forecasts() -> tuple:
    """Load BVAR forecasts from academic results."""
    forecasts = {}
    density_forecasts = {}

    bvar_dir = ACADEMIC_RESULTS_DIR / "bvar"
    if not bvar_dir.exists():
        return forecasts, density_forecasts

    # Load different BVAR specifications
    specs = ["parsimonious", "bop", "monetary", "pca", "full"]

    for spec in specs:
        # Rolling backtest results (test set forecasts)
        backtest_file = bvar_dir / f"bvar_rolling_backtest_{spec}.csv"
        if backtest_file.exists():
            df = pd.read_csv(backtest_file)

            # Get h=1 forecasts only
            if "horizon" in df.columns:
                h1 = df[df["horizon"] == 1]
            else:
                h1 = df

            # Filter to test set
            if "split" in h1.columns:
                h1 = h1[h1["split"] == "test"]

            if "forecast_point" in h1.columns:
                forecasts[f"BVAR_{spec}"] = h1["forecast_point"].values
            elif "forecast_mean" in h1.columns:
                forecasts[f"BVAR_{spec}"] = h1["forecast_mean"].values

            # Load density forecasts if available
            density_file = bvar_dir / f"bvar_density_{spec}.csv"
            if density_file.exists():
                density_df = pd.read_csv(density_file)
                if "mean" in density_df.columns and "std" in density_df.columns:
                    density_forecasts[f"BVAR_{spec}"] = (
                        density_df["mean"].values,
                        density_df["std"].values
                    )

    return forecasts, density_forecasts


def load_dma_forecasts() -> tuple:
    """Load DMA/DMS forecasts."""
    forecasts = {}

    dma_dir = ACADEMIC_RESULTS_DIR / "dma"
    if not dma_dir.exists():
        return forecasts

    # Main DMA forecasts file
    dma_file = dma_dir / "dma_forecasts.csv"
    if dma_file.exists():
        df = pd.read_csv(dma_file, parse_dates=["date"])

        if "dma_forecast" in df.columns:
            valid = ~df["dma_forecast"].isna()
            forecasts["DMA"] = df.loc[valid, "dma_forecast"].values

    # DMA rolling backtest (for test set alignment)
    backtest_file = dma_dir / "dma_rolling_backtest.csv"
    if backtest_file.exists():
        df = pd.read_csv(backtest_file, parse_dates=["date"])

        # Get test set DMA forecasts
        if "dma_forecast" in df.columns:
            valid = ~df["dma_forecast"].isna()
            if valid.sum() > 0:
                forecasts["DMA"] = df.loc[valid, "dma_forecast"].values

        if "dms_forecast" in df.columns:
            valid = ~df["dms_forecast"].isna()
            if valid.sum() > 0:
                forecasts["DMS"] = df.loc[valid, "dms_forecast"].values

    return forecasts


def load_combination_forecasts() -> dict:
    """Load forecast combination results."""
    forecasts = {}

    comb_dir = ACADEMIC_RESULTS_DIR / "combinations"
    if not comb_dir.exists():
        return forecasts

    comb_file = comb_dir / "combination_forecasts.csv"
    if comb_file.exists():
        df = pd.read_csv(comb_file, parse_dates=["date"])

        # Get test set dates (2023 onwards)
        df = df[df["date"] >= "2023-01-01"]

        combinations = [
            ("combined_equal", "EqualWeight"),
            ("combined_mse", "MSE-Weight"),
            ("combined_gr_convex", "GR-Convex"),
            ("combined_trimmed", "TrimmedMean"),
            ("combined_median", "Median"),
        ]

        for col, name in combinations:
            if col in df.columns:
                forecasts[name] = df[col].values

    return forecasts


def load_all_forecasts() -> tuple:
    """Load all available forecasts and align them."""
    print("\n" + "="*60)
    print("Loading All Available Forecasts")
    print("="*60)

    # Load baseline forecasts (these have actuals)
    baseline_forecasts, actuals = load_baseline_forecasts()

    # Load academic model forecasts
    bvar_forecasts, density_forecasts = load_bvar_forecasts()
    dma_forecasts = load_dma_forecasts()
    combination_forecasts = load_combination_forecasts()

    # Combine all forecasts
    all_forecasts = {**baseline_forecasts}

    # Add BVAR (only parsimonious for main comparison to avoid clutter)
    if "BVAR_parsimonious" in bvar_forecasts:
        all_forecasts["BVAR"] = bvar_forecasts["BVAR_parsimonious"]

    # Add DMA
    all_forecasts.update(dma_forecasts)

    # Add best combination methods
    for name in ["EqualWeight", "MSE-Weight", "GR-Convex"]:
        if name in combination_forecasts:
            all_forecasts[name] = combination_forecasts[name]

    # Align lengths - find minimum length across all forecasts
    min_len = min(len(f) for f in all_forecasts.values() if len(f) > 0)
    min_len = min(min_len, len(actuals) if actuals is not None else min_len)

    print(f"\nAligning to {min_len} observations (test set)")

    # Truncate all to same length (from end to get most recent)
    aligned_forecasts = {}
    for name, fc in all_forecasts.items():
        if len(fc) >= min_len:
            aligned_forecasts[name] = fc[-min_len:]
        else:
            print(f"  Skipping {name}: only {len(fc)} observations")

    aligned_actuals = actuals[-min_len:] if actuals is not None else None

    print(f"\nModels included: {list(aligned_forecasts.keys())}")
    print(f"Test set size: {min_len}")

    return aligned_forecasts, aligned_actuals, density_forecasts


# =============================================================================
# Main Statistical Tests
# =============================================================================

def run_dm_tests(forecasts: dict, actuals: np.ndarray) -> tuple:
    """Run Diebold-Mariano tests for all model pairs."""
    print("\n" + "="*60)
    print("Diebold-Mariano Tests (HLN correction)")
    print("="*60)

    # Full pairwise matrix
    dm_stats, p_values = dm_test_matrix(
        actuals, forecasts,
        loss_fn=DM_LOSS_FUNCTION,
        h=FORECAST_HORIZON,
        use_hln=True
    )

    # Format for paper
    formatted_dm = format_dm_table_for_paper(dm_stats, p_values)

    print("\nDM Statistics Matrix (with significance):")
    print(formatted_dm.to_string())

    # Test vs Naive benchmark
    if "Naive" in forecasts:
        print("\n\nDM Tests vs Naive Benchmark:")
        vs_naive = dm_test_vs_benchmark(
            actuals, forecasts, "Naive",
            loss_fn=DM_LOSS_FUNCTION, h=FORECAST_HORIZON
        )
        print(vs_naive.to_string(index=False))

    # Test vs DMA if available
    if "DMA" in forecasts:
        print("\n\nDM Tests vs DMA:")
        vs_dma = dm_test_vs_benchmark(
            actuals, forecasts, "DMA",
            loss_fn=DM_LOSS_FUNCTION, h=FORECAST_HORIZON
        )
        print(vs_dma.to_string(index=False))

    return dm_stats, p_values, formatted_dm


def run_mcs(forecasts: dict, actuals: np.ndarray) -> dict:
    """Run Model Confidence Set procedure."""
    print("\n" + "="*60)
    print(f"Model Confidence Set (alpha = {SIGNIFICANCE_LEVEL})")
    print("="*60)

    mcs_result = model_confidence_set(
        actuals, forecasts,
        loss_fn=DM_LOSS_FUNCTION,
        alpha=SIGNIFICANCE_LEVEL,
        bootstrap_reps=BOOTSTRAP_REPS,
        statistic="range",
        seed=42
    )

    if "error" in mcs_result:
        print(f"Error: {mcs_result['error']}")
        return mcs_result

    print(f"\nModels in {int((1-SIGNIFICANCE_LEVEL)*100)}% MCS:")
    for model in mcs_result["mcs"]:
        print(f"  - {model}")

    print(f"\nEliminated models (in order):")
    for i, model in enumerate(mcs_result["eliminated"]):
        print(f"  {i+1}. {model}")

    # Summary table
    summary = mcs_summary_table(mcs_result, forecasts, actuals)
    print("\nMCS Summary Table:")
    print(summary.to_string(index=False))

    # MCS p-values for all models
    mcs_pvalues = mcs_with_pvalues(
        actuals, forecasts,
        loss_fn=DM_LOSS_FUNCTION,
        bootstrap_reps=BOOTSTRAP_REPS,
        seed=42
    )
    print("\nMCS p-values:")
    print(mcs_pvalues.to_string(index=False))

    return {
        "mcs_result": mcs_result,
        "summary": summary,
        "pvalues": mcs_pvalues,
    }


def run_encompassing_tests(forecasts: dict, actuals: np.ndarray) -> dict:
    """Run forecast encompassing tests."""
    print("\n" + "="*60)
    print("Forecast Encompassing Tests")
    print("="*60)

    # Pairwise encompassing matrix
    lambda_matrix, pval_matrix = pairwise_encompassing_matrix(
        actuals, forecasts, hac_lags=FORECAST_HORIZON
    )

    formatted = format_encompassing_table(lambda_matrix, pval_matrix)
    print("\nEncompassing Test Results (lambda2 with significance):")
    print(formatted.to_string())

    # Summary
    summary = encompassing_summary(actuals, forecasts, hac_lags=FORECAST_HORIZON)
    print("\nEncompassing Summary:")
    print(summary[["Model_1", "Model_2", "Interpretation"]].to_string(index=False))

    # Optimal combination weights
    opt_weights = optimal_combination_weights(actuals, forecasts, constrained=True)
    if "weights" in opt_weights:
        print("\nOptimal Combination Weights (constrained):")
        for model, weight in sorted(opt_weights["weights"].items(), key=lambda x: -x[1]):
            if weight > 0.01:
                print(f"  {model}: {weight:.3f}")

    return {
        "lambda_matrix": lambda_matrix,
        "pval_matrix": pval_matrix,
        "formatted": formatted,
        "summary": summary,
        "optimal_weights": opt_weights,
    }


def run_density_evaluation(actuals: np.ndarray, density_forecasts: dict) -> dict:
    """Run density forecast evaluation if available."""
    print("\n" + "="*60)
    print("Density Forecast Evaluation")
    print("="*60)

    if not density_forecasts:
        print("No density forecasts available.")
        return {}

    # Filter to aligned actuals
    results = {}

    for model_name, (means, stds) in density_forecasts.items():
        # Align lengths
        n = min(len(actuals), len(means), len(stds))
        if n < 10:
            continue

        eval_result = evaluate_density_forecasts(
            actuals[-n:], means[-n:], stds[-n:], model_name
        )
        results[model_name] = eval_result

        print(f"\n{model_name}:")
        print(f"  Mean CRPS: {eval_result['crps']['mean']:.4f}")
        print(f"  Mean Log Score: {eval_result['log_score']['mean']:.4f}")
        print(f"  Coverage 90%: {eval_result['coverage']['90_pct']:.2%}")
        print(f"  PIT uniform: {not eval_result['pit']['chi2_test'].get('reject_uniformity', True)}")

    return results


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_dm_heatmap(dm_stats: pd.DataFrame, p_values: pd.DataFrame, save_path: Path):
    """Create heatmap of DM test statistics."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create mask for diagonal
    mask = np.eye(len(dm_stats), dtype=bool)

    # Create annotations with significance stars
    annot = np.empty_like(dm_stats.values, dtype=object)
    for i in range(len(dm_stats)):
        for j in range(len(dm_stats)):
            if i == j:
                annot[i, j] = ""
            else:
                stat = dm_stats.iloc[i, j]
                pval = p_values.iloc[i, j]
                if np.isnan(stat):
                    annot[i, j] = ""
                else:
                    stars = ""
                    if pval < 0.01:
                        stars = "***"
                    elif pval < 0.05:
                        stars = "**"
                    elif pval < 0.10:
                        stars = "*"
                    annot[i, j] = f"{stat:.2f}{stars}"

    # Plot heatmap
    sns.heatmap(
        dm_stats.values,
        annot=annot,
        fmt="",
        mask=mask,
        cmap="RdBu_r",
        center=0,
        xticklabels=dm_stats.columns,
        yticklabels=dm_stats.index,
        ax=ax,
        cbar_kws={"label": "DM Statistic"},
        vmin=-4,
        vmax=4,
    )

    ax.set_title("Diebold-Mariano Test Statistics\n(Positive = Column model better)", fontsize=12)
    ax.set_xlabel("Model (Column)")
    ax.set_ylabel("Model (Row)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved DM heatmap: {save_path}")


def plot_mcs_results(mcs_result: dict, forecasts: dict, actuals: np.ndarray, save_path: Path):
    """Create visualization of MCS results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: RMSE with MCS membership
    model_metrics = []
    for model, fc in forecasts.items():
        valid = ~np.isnan(actuals - fc)
        rmse = np.sqrt(np.mean((actuals[valid] - fc[valid])**2))
        in_mcs = model in mcs_result["mcs"]
        model_metrics.append({"Model": model, "RMSE": rmse, "In_MCS": in_mcs})

    df = pd.DataFrame(model_metrics).sort_values("RMSE")

    colors = ["#2ecc71" if x else "#e74c3c" for x in df["In_MCS"]]
    bars = axes[0].barh(df["Model"], df["RMSE"], color=colors)
    axes[0].set_xlabel("RMSE")
    axes[0].set_title(f"Model Performance\n(Green = In {int((1-SIGNIFICANCE_LEVEL)*100)}% MCS)")
    axes[0].invert_yaxis()

    # Right: Elimination order
    if mcs_result["eliminated"]:
        elim_order = list(range(1, len(mcs_result["eliminated"]) + 1))
        axes[1].barh(mcs_result["eliminated"], elim_order, color="#e74c3c")
        axes[1].set_xlabel("Elimination Order")
        axes[1].set_title("Order of Elimination from MCS")
        axes[1].invert_yaxis()
    else:
        axes[1].text(0.5, 0.5, "No models eliminated",
                    ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title("Order of Elimination from MCS")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved MCS plot: {save_path}")


def plot_forecast_comparison(forecasts: dict, actuals: np.ndarray, save_path: Path):
    """Plot forecast comparison over time."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot actuals
    ax.plot(actuals, color="black", linewidth=2, label="Actual", zorder=10)

    # Plot forecasts
    colors = plt.cm.tab10(np.linspace(0, 1, len(forecasts)))
    for (name, fc), color in zip(forecasts.items(), colors):
        ax.plot(fc, color=color, linewidth=1, alpha=0.7, label=name)

    ax.set_xlabel("Time")
    ax.set_ylabel("Reserves")
    ax.set_title("Forecast Comparison (Test Set)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved forecast comparison: {save_path}")


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main execution function."""
    print("="*60)
    print("STATISTICAL TESTS FOR FORECAST EVALUATION")
    print("Specification 10 - Academic Forecasting Pipeline")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Load all forecasts
    forecasts, actuals, density_forecasts = load_all_forecasts()

    if actuals is None or len(forecasts) < 2:
        print("\nError: Insufficient data for statistical tests")
        return

    # Run Diebold-Mariano tests
    dm_stats, p_values, formatted_dm = run_dm_tests(forecasts, actuals)

    # Save DM results
    dm_stats.to_csv(OUTPUT_DIR / "dm_test_matrix.csv")
    p_values.to_csv(OUTPUT_DIR / "dm_pvalues_matrix.csv")
    formatted_dm.to_csv(OUTPUT_DIR / "dm_formatted_table.csv")
    print(f"\nSaved DM results to {OUTPUT_DIR}")

    # Run MCS
    mcs_results = run_mcs(forecasts, actuals)
    if "error" not in mcs_results.get("mcs_result", {}):
        # Save MCS results
        mcs_result = mcs_results["mcs_result"]
        with open(OUTPUT_DIR / "mcs_results.json", "w") as f:
            json.dump({
                "mcs": mcs_result["mcs"],
                "eliminated": mcs_result["eliminated"],
                "p_values": [float(p) for p in mcs_result["p_values"]],
                "alpha": mcs_result["alpha"],
                "n_obs": mcs_result["n_obs"],
            }, f, indent=2)

        mcs_results["summary"].to_csv(OUTPUT_DIR / "mcs_summary.csv", index=False)
        mcs_results["pvalues"].to_csv(OUTPUT_DIR / "mcs_pvalues.csv", index=False)
        print(f"Saved MCS results to {OUTPUT_DIR}")

    # Run encompassing tests
    enc_results = run_encompassing_tests(forecasts, actuals)
    enc_results["summary"].to_csv(OUTPUT_DIR / "encompassing_tests.csv", index=False)
    enc_results["formatted"].to_csv(OUTPUT_DIR / "encompassing_matrix.csv")
    if "optimal_weights" in enc_results and "weights" in enc_results["optimal_weights"]:
        with open(OUTPUT_DIR / "optimal_weights.json", "w") as f:
            json.dump(enc_results["optimal_weights"], f, indent=2, default=float)
    print(f"Saved encompassing results to {OUTPUT_DIR}")

    # Run density evaluation if available
    if density_forecasts:
        density_results = run_density_evaluation(actuals, density_forecasts)
        if density_results:
            # Save density evaluation results
            density_summary = []
            for model, result in density_results.items():
                density_summary.append({
                    "Model": model,
                    "Mean_CRPS": result["crps"]["mean"],
                    "Mean_LogScore": result["log_score"]["mean"],
                    "Coverage_90": result["coverage"]["90_pct"],
                    "Coverage_95": result["coverage"]["95_pct"],
                    "PIT_Chi2_pvalue": result["pit"]["chi2_test"].get("p_value", np.nan),
                })
            pd.DataFrame(density_summary).to_csv(
                OUTPUT_DIR / "density_evaluation.csv", index=False
            )

    # Generate figures
    print("\n" + "="*60)
    print("Generating Figures")
    print("="*60)

    plot_dm_heatmap(dm_stats, p_values, FIGURES_DIR / "dm_heatmap.png")

    if "mcs_result" in mcs_results and "error" not in mcs_results["mcs_result"]:
        plot_mcs_results(
            mcs_results["mcs_result"], forecasts, actuals,
            FIGURES_DIR / "mcs_results.png"
        )

    plot_forecast_comparison(forecasts, actuals, FIGURES_DIR / "forecast_comparison.png")

    # Print summary for paper
    print("\n" + "="*60)
    print("SUMMARY FOR PAPER")
    print("="*60)

    print(f"\n1. Number of models compared: {len(forecasts)}")
    print(f"   Models: {', '.join(forecasts.keys())}")

    if "mcs_result" in mcs_results and "error" not in mcs_results["mcs_result"]:
        print(f"\n2. Models in {int((1-SIGNIFICANCE_LEVEL)*100)}% Model Confidence Set:")
        for m in mcs_results["mcs_result"]["mcs"]:
            print(f"   - {m}")

    print(f"\n3. Test set size: {len(actuals)} observations")

    # Best model by RMSE
    best_rmse = float("inf")
    best_model = None
    for name, fc in forecasts.items():
        valid = ~np.isnan(actuals - fc)
        rmse = np.sqrt(np.mean((actuals[valid] - fc[valid])**2))
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = name

    print(f"\n4. Best model by RMSE: {best_model} (RMSE = {best_rmse:.2f})")

    print("\n" + "="*60)
    print("Statistical tests complete!")
    print("="*60)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
