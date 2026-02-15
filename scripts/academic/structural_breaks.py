"""
Structural Break Analysis for Sri Lankan Reserves

Main execution script for Specification 02: Structural Breaks.
Runs Bai-Perron, Chow, and CUSUM tests on reserves and related series.

Usage:
    python structural_breaks.py
"""

import os
import sys
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from reserves_project.scripts.academic.break_detection import (
    bai_perron_with_dates,
    sequential_bai_perron,
    compute_confidence_intervals,
    multiple_chow_tests,
    qlr_test,
    cusum_test_with_dates,
    cusumsq_test_with_dates,
    combined_stability_test,
    plot_series_with_breaks,
    plot_bic_selection,
    plot_cusum,
    plot_cusumsq,
    plot_regime_comparison,
    plot_chow_test_results
)

# ============================================================================
# Configuration
# ============================================================================

BAI_PERRON_CONFIG = {
    "max_breaks": 5,
    "min_segment_length": 24,  # 2 years minimum between breaks
    "significance_level": 0.05,
    "trimming_fraction": 0.15,
    "selection_criterion": "bic"
}

KNOWN_BREAK_DATES = [
    {"date": "2009-06-01", "event": "Post-war recovery begins"},
    {"date": "2018-10-01", "event": "Currency crisis onset"},
    {"date": "2020-03-01", "event": "COVID-19 lockdown"},
    {"date": "2022-04-01", "event": "Sovereign default"},
]

VARIABLES_FOR_BREAK_TESTS = [
    "gross_reserves_usd_m",       # Primary target
    "usd_lkr",                    # Exchange rate
    "trade_balance_usd_m",        # Trade balance (if available)
]

# Paths
DATA_PATH = PROJECT_ROOT / "data" / "merged" / "reserves_forecasting_panel.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "structural_breaks"
FIGURES_DIR = OUTPUT_DIR / "figures"


# ============================================================================
# Utility Functions
# ============================================================================

def load_data() -> pd.DataFrame:
    """Load and prepare the reserves forecasting panel."""
    print(f"Loading data from: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    print(f"  Loaded {len(df)} observations from {df.index[0]} to {df.index[-1]}")
    print(f"  Available columns: {df.columns.tolist()}")

    return df


def ensure_directories():
    """Create output directories if they don't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Figures directory: {FIGURES_DIR}")


def serialize_results(obj):
    """Convert numpy types to JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: serialize_results(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_results(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif pd.isna(obj):
        return None
    else:
        return obj


# ============================================================================
# Bai-Perron Analysis
# ============================================================================

def run_bai_perron_analysis(df: pd.DataFrame) -> dict:
    """Run Bai-Perron break detection on all variables."""
    print("\n" + "="*70)
    print("BAI-PERRON STRUCTURAL BREAK DETECTION")
    print("="*70)

    results = {}

    for var in VARIABLES_FOR_BREAK_TESTS:
        if var not in df.columns:
            print(f"\n  {var}: NOT AVAILABLE - skipping")
            continue

        series = df[var].dropna()
        if len(series) < BAI_PERRON_CONFIG["min_segment_length"] * 2:
            print(f"\n  {var}: Insufficient data ({len(series)} obs) - skipping")
            continue

        print(f"\n  {var}:")
        print(f"    Observations: {len(series)}")
        print(f"    Period: {series.index[0].strftime('%Y-%m')} to {series.index[-1].strftime('%Y-%m')}")

        # Run Bai-Perron test
        bp_result = bai_perron_with_dates(
            series=series,
            max_breaks=BAI_PERRON_CONFIG["max_breaks"],
            min_segment_length=BAI_PERRON_CONFIG["min_segment_length"],
            significance_level=BAI_PERRON_CONFIG["significance_level"],
            trimming_fraction=BAI_PERRON_CONFIG["trimming_fraction"],
            selection_criterion=BAI_PERRON_CONFIG["selection_criterion"]
        )

        print(f"    Optimal breaks: {bp_result['n_breaks']}")
        if bp_result.get('break_dates'):
            print(f"    Break dates: {bp_result['break_dates']}")

        # Compute confidence intervals if breaks found (using asymptotic method for speed)
        if bp_result['n_breaks'] > 0:
            ci = compute_confidence_intervals(
                y=series.values,
                break_indices=bp_result['break_indices'],
                confidence_level=0.95,
                method='asymptotic',  # Use asymptotic for speed
                n_bootstrap=100  # Reduced for speed if bootstrap is used
            )
            bp_result['confidence_intervals'] = {
                str(k): v for k, v in ci.items()
            }

            # Convert CI indices to dates
            bp_result['confidence_intervals_dates'] = {}
            for idx, (lower, upper) in ci.items():
                bp_result['confidence_intervals_dates'][str(idx)] = {
                    'lower': series.index[lower].strftime('%Y-%m-%d'),
                    'upper': series.index[upper].strftime('%Y-%m-%d')
                }

        # Print BIC values
        print(f"    BIC values:")
        for n_bkps, bic in sorted(bp_result['bic_values'].items()):
            marker = " *" if n_bkps == bp_result['n_breaks'] else ""
            print(f"      {n_bkps} breaks: {bic:.2f}{marker}")

        # Print segment statistics
        if bp_result['segment_stats']:
            print(f"    Segment statistics:")
            for i, stats in enumerate(bp_result['segment_stats']):
                period = f"{stats.get('start_date', '?')} to {stats.get('end_date', '?')}"
                print(f"      Regime {i+1}: mean={stats['mean']:.2f}, std={stats['std']:.2f} ({period})")

        results[var] = bp_result

        # Generate plots
        if bp_result.get('break_dates'):
            # Series with breaks
            fig = plot_series_with_breaks(
                series=series,
                break_dates=bp_result['break_dates'],
                title=f"Structural Breaks in {var}",
                ylabel=var,
                save_path=FIGURES_DIR / f"{var}_with_breaks.png"
            )
            plt.close(fig)

            # BIC selection plot
            fig = plot_bic_selection(
                bic_values=bp_result['bic_values'],
                lwz_values=bp_result.get('lwz_values'),
                optimal_n=bp_result['n_breaks'],
                title=f"BIC Selection: {var}",
                save_path=FIGURES_DIR / f"{var}_bic_selection.png"
            )
            plt.close(fig)

            # Regime comparison
            fig = plot_regime_comparison(
                series=series,
                break_dates=bp_result['break_dates'],
                title=f"Regime Comparison: {var}",
                save_path=FIGURES_DIR / f"{var}_regime_comparison.png"
            )
            plt.close(fig)

    return results


# ============================================================================
# Chow Test Analysis
# ============================================================================

def run_chow_tests(df: pd.DataFrame) -> dict:
    """Run Chow tests for known break dates."""
    print("\n" + "="*70)
    print("CHOW TESTS FOR KNOWN BREAK DATES")
    print("="*70)

    break_dates = [bd["date"] for bd in KNOWN_BREAK_DATES]
    event_names = [bd["event"] for bd in KNOWN_BREAK_DATES]

    results = {}

    for var in VARIABLES_FOR_BREAK_TESTS:
        if var not in df.columns:
            continue

        series = df[var].dropna()
        if len(series) < 50:
            continue

        print(f"\n  {var}:")

        chow_result = multiple_chow_tests(
            series=series,
            break_dates=break_dates,
            event_names=event_names,
            significance_level=BAI_PERRON_CONFIG["significance_level"]
        )

        results[var] = chow_result

        # Print results
        for test in chow_result['tests']:
            if test.get('valid', False):
                sig = "***" if test['p_value'] < 0.01 else "**" if test['p_value'] < 0.05 else "*" if test['p_value'] < 0.10 else ""
                print(f"    {test['event_name']}: F={test['f_statistic']:.2f}, p={test['p_value']:.4f} {sig}")
            else:
                print(f"    {test['event_name']}: {test.get('error', 'Invalid')}")

        # Summary
        n_sig = chow_result['summary']['n_significant']
        print(f"    Summary: {n_sig}/{len(break_dates)} breaks significant at 5%")

        # Generate plot
        fig = plot_chow_test_results(
            chow_results=chow_result,
            series=series,
            save_path=FIGURES_DIR / f"{var}_chow_tests.png"
        )
        plt.close(fig)

    return results


# ============================================================================
# CUSUM Analysis
# ============================================================================

def run_cusum_tests(df: pd.DataFrame) -> dict:
    """Run CUSUM and CUSUMSQ stability tests."""
    print("\n" + "="*70)
    print("CUSUM STABILITY TESTS")
    print("="*70)

    results = {}

    for var in VARIABLES_FOR_BREAK_TESTS:
        if var not in df.columns:
            continue

        series = df[var].dropna()
        if len(series) < 50:
            continue

        print(f"\n  {var}:")

        # Run combined stability test
        stability_result = combined_stability_test(
            series=series,
            significance_level=BAI_PERRON_CONFIG["significance_level"]
        )

        results[var] = stability_result

        # Print results
        cusum = stability_result['cusum']
        cusumsq = stability_result['cusumsq']

        print(f"    CUSUM: {'STABLE' if cusum.get('stable') else 'UNSTABLE'}")
        if not cusum.get('stable') and cusum.get('crossing_date'):
            print(f"      First crossing: {cusum['crossing_date']}")

        print(f"    CUSUMSQ: {'STABLE' if cusumsq.get('stable') else 'UNSTABLE'}")
        if not cusumsq.get('stable') and cusumsq.get('crossing_date'):
            print(f"      First crossing: {cusumsq['crossing_date']}")

        print(f"    Overall: {stability_result['overall_interpretation']}")

        # Generate plots
        fig = plot_cusum(
            cusum_result=cusum,
            dates=series.index,
            title=f"CUSUM Test: {var}",
            save_path=FIGURES_DIR / f"{var}_cusum.png"
        )
        plt.close(fig)

        fig = plot_cusumsq(
            cusumsq_result=cusumsq,
            dates=series.index,
            title=f"CUSUMSQ Test: {var}",
            save_path=FIGURES_DIR / f"{var}_cusumsq.png"
        )
        plt.close(fig)

    return results


# ============================================================================
# Create Summary Table
# ============================================================================

def create_break_summary(
    bai_perron_results: dict,
    chow_results: dict,
    cusum_results: dict
) -> pd.DataFrame:
    """Create consolidated summary of all break tests."""
    print("\n" + "="*70)
    print("CREATING BREAK SUMMARY")
    print("="*70)

    rows = []

    for var in VARIABLES_FOR_BREAK_TESTS:
        row = {"variable": var}

        # Bai-Perron results
        if var in bai_perron_results:
            bp = bai_perron_results[var]
            row["bp_n_breaks"] = bp['n_breaks']
            row["bp_break_dates"] = ", ".join(bp.get('break_dates', []))
            row["bp_optimal_bic"] = bp.get('optimal_bic')
        else:
            row["bp_n_breaks"] = None
            row["bp_break_dates"] = ""
            row["bp_optimal_bic"] = None

        # Chow test results
        if var in chow_results:
            chow = chow_results[var]
            sig_breaks = [b['event'] for b in chow['summary'].get('significant_breaks', [])]
            row["chow_significant_events"] = ", ".join(sig_breaks)
            row["chow_n_significant"] = chow['summary']['n_significant']
        else:
            row["chow_significant_events"] = ""
            row["chow_n_significant"] = None

        # CUSUM results
        if var in cusum_results:
            cusum = cusum_results[var]
            row["cusum_stable"] = cusum['cusum'].get('stable')
            row["cusumsq_stable"] = cusum['cusumsq'].get('stable')
            row["overall_stable"] = cusum.get('overall_stable')
        else:
            row["cusum_stable"] = None
            row["cusumsq_stable"] = None
            row["overall_stable"] = None

        rows.append(row)

    df_summary = pd.DataFrame(rows)

    # Save to CSV
    summary_path = OUTPUT_DIR / "break_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")

    # Print summary
    print("\nSummary Table:")
    print(df_summary.to_string())

    return df_summary


# ============================================================================
# Create Break Dummies for Forecasting
# ============================================================================

def create_break_dummies(
    dates: pd.DatetimeIndex,
    break_dates: list,
    dummy_type: str = "level"
) -> pd.DataFrame:
    """
    Create dummy variables for structural breaks.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Full date index
    break_dates : list
        List of break date strings
    dummy_type : str
        "level" - permanent shift (1 after break)
        "impulse" - one-time shock (1 at break only)
        "trend" - trend change (counter after break)

    Returns
    -------
    pd.DataFrame
        DataFrame with break dummy columns
    """
    dummies = pd.DataFrame(index=dates)

    for i, break_date in enumerate(break_dates):
        break_dt = pd.Timestamp(break_date)
        col_name = f"break_{i+1}_{dummy_type}"

        if dummy_type == "level":
            dummies[col_name] = (dates >= break_dt).astype(int)
        elif dummy_type == "impulse":
            # Find nearest date in index
            nearest_idx = np.argmin(np.abs(dates - break_dt))
            dummies[col_name] = 0
            dummies.iloc[nearest_idx, -1] = 1
        elif dummy_type == "trend":
            days_since = (dates - break_dt).days
            dummies[col_name] = np.maximum(0, days_since / 30)  # months since break

    return dummies


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    print("="*70)
    print("STRUCTURAL BREAK ANALYSIS")
    print("Specification 02: Bai-Perron, Chow, and CUSUM Tests")
    print("="*70)
    print(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Setup
    ensure_directories()
    df = load_data()

    # Run analyses
    bai_perron_results = run_bai_perron_analysis(df)
    chow_results = run_chow_tests(df)
    cusum_results = run_cusum_tests(df)

    # Create summary
    summary_df = create_break_summary(bai_perron_results, chow_results, cusum_results)

    # Save detailed results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    # Bai-Perron results
    bp_path = OUTPUT_DIR / "bai_perron_results.json"
    with open(bp_path, 'w') as f:
        json.dump(serialize_results(bai_perron_results), f, indent=2)
    print(f"Bai-Perron results saved to: {bp_path}")

    # Chow test results
    chow_path = OUTPUT_DIR / "chow_test_results.json"
    with open(chow_path, 'w') as f:
        json.dump(serialize_results(chow_results), f, indent=2)
    print(f"Chow test results saved to: {chow_path}")

    # CUSUM results
    cusum_path = OUTPUT_DIR / "cusum_results.json"
    with open(cusum_path, 'w') as f:
        json.dump(serialize_results(cusum_results), f, indent=2)
    print(f"CUSUM results saved to: {cusum_path}")

    # Create break dummies for the primary variable
    if "gross_reserves_usd_m" in bai_perron_results:
        bp_breaks = bai_perron_results["gross_reserves_usd_m"].get("break_dates", [])
        if bp_breaks:
            dummies = create_break_dummies(df.index, bp_breaks, dummy_type="level")
            dummies_path = OUTPUT_DIR / "break_dummies.csv"
            dummies.to_csv(dummies_path)
            print(f"Break dummies saved to: {dummies_path}")

    # Print final summary
    print("\n" + "="*70)
    print("EXECUTION COMPLETE")
    print("="*70)
    print(f"\nFigures generated: {len(list(FIGURES_DIR.glob('*.png')))}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Return results for programmatic use
    return {
        "bai_perron": bai_perron_results,
        "chow": chow_results,
        "cusum": cusum_results,
        "summary": summary_df
    }


if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    results = main()
