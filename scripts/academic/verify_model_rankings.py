"""
Comprehensive Model Verification and Ranking Analysis

This script:
1. Loads all forecasts and aligns them to a common test window
2. Computes naive benchmark RMSE (random walk)
3. Computes normalized RMSE (RMSE / Naive RMSE) for all models
4. Splits test into subperiods: pre-crisis, crisis, recovery
5. Computes directional accuracy for all models
6. Generates plots for top models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_DIR / "forecast_results_academic"
OUTPUT_DIR = DATA_DIR / "model_verification"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Test period
TEST_START = pd.Timestamp("2023-01-01")
TEST_END = pd.Timestamp("2025-12-01")

# Subperiods for Sri Lanka crisis context
SUBPERIODS = {
    "Pre-crisis (2023-01 to 2023-06)": (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-06-01")),
    "Crisis (2023-07 to 2024-06)": (pd.Timestamp("2023-07-01"), pd.Timestamp("2024-06-01")),
    "Recovery (2024-07+)": (pd.Timestamp("2024-07-01"), pd.Timestamp("2025-12-31")),
}

def load_actuals():
    """Load actual reserves data."""
    # Use DMA backtest which has full actuals
    dma_path = RESULTS_DIR / "dma" / "dma_rolling_backtest.csv"
    df = pd.read_csv(dma_path, parse_dates=['date'])
    df = df.set_index('date')
    return df['actual'].dropna()

def load_all_forecasts():
    """Load forecasts from all models, aligned to common index."""
    forecasts = {}

    # 1. BVAR models (use h=1 rolling backtest)
    for varset in ['parsimonious', 'bop', 'monetary', 'pca', 'full']:
        path = RESULTS_DIR / "bvar" / f"bvar_rolling_backtest_{varset}.csv"
        if path.exists():
            df = pd.read_csv(path)
            # Parse date column
            if 'forecast_date' in df.columns:
                df['forecast_date'] = pd.to_datetime(df['forecast_date'])
                # Get h=1 forecasts
                h1 = df[df['horizon'] == 1].copy()
                h1 = h1.set_index('forecast_date')
                if 'forecast_point' in h1.columns:
                    forecasts[f'BVAR_{varset}'] = h1['forecast_point']
                elif 'forecast' in h1.columns:
                    forecasts[f'BVAR_{varset}'] = h1['forecast']

    # 2. Combination models
    comb_path = RESULTS_DIR / "combinations" / "combination_rolling_backtest.csv"
    if comb_path.exists():
        df = pd.read_csv(comb_path)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        for col in df.columns:
            if col.startswith('combined_') and col != 'combined_gr_none' and col != 'combined_gr_sum':
                clean_name = col.replace('combined_', '').replace('_', '-').title()
                if 'Mse' in clean_name:
                    clean_name = 'MSE-Weight'
                elif 'Equal' in clean_name:
                    clean_name = 'EqualWeight'
                elif 'Gr' in clean_name and 'Convex' in clean_name:
                    clean_name = 'GR-Convex'
                elif 'Trimmed' in clean_name:
                    clean_name = 'TrimmedMean'
                elif 'Median' in clean_name:
                    clean_name = 'Median'
                forecasts[clean_name] = df[col]

    # 3. DMA/DMS
    dma_path = RESULTS_DIR / "dma" / "dma_rolling_backtest.csv"
    if dma_path.exists():
        df = pd.read_csv(dma_path)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        if 'dma_forecast' in df.columns:
            forecasts['DMA'] = df['dma_forecast']
        if 'dms_forecast' in df.columns:
            forecasts['DMS'] = df['dms_forecast']

    # 4. TVP-VAR (different format - uses 'target' column as date)
    for varset in ['parsimonious', 'bop', 'monetary']:
        path = RESULTS_DIR / "tvp_var" / f"tvp_rolling_backtest_{varset}.csv"
        if path.exists():
            df = pd.read_csv(path)
            # TVP uses 'target' column for date and 'forecast_mean' for forecast
            if 'target' in df.columns and 'forecast_mean' in df.columns:
                df['target'] = pd.to_datetime(df['target'])
                df = df.set_index('target')
                forecasts[f'TVP_{varset}'] = df['forecast_mean']

    # 5. FAVAR
    favar_path = RESULTS_DIR / "favar" / "favar_rolling_backtest.csv"
    if favar_path.exists():
        df = pd.read_csv(favar_path)
        # Identify date column
        date_col = None
        for col in ['date', 'forecast_date', 'target']:
            if col in df.columns:
                date_col = col
                break
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
        # Get h=1 if horizon column exists
        if 'horizon' in df.columns:
            df = df[df['horizon'] == 1]
        if 'forecast' in df.columns:
            forecasts['FAVAR'] = df['forecast']
        elif 'forecast_mean' in df.columns:
            forecasts['FAVAR'] = df['forecast_mean']

    # 6. MIDAS
    midas_path = RESULTS_DIR / "midas" / "midas_rolling_backtest.csv"
    if midas_path.exists():
        df = pd.read_csv(midas_path)
        date_col = None
        for col in ['date', 'forecast_date', 'target']:
            if col in df.columns:
                date_col = col
                break
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
        if 'forecast' in df.columns:
            forecasts['MIDAS'] = df['forecast']
        elif 'forecast_mean' in df.columns:
            forecasts['MIDAS'] = df['forecast_mean']

    return forecasts

def compute_naive_forecast(actuals):
    """Random walk forecast: y_{t+1} = y_t"""
    return actuals.shift(1)

def compute_metrics(actuals, forecasts, period_name="Full Test"):
    """Compute RMSE, MAE, and directional accuracy."""
    # Ensure datetime index
    if not isinstance(actuals.index, pd.DatetimeIndex):
        actuals = actuals.copy()
        actuals.index = pd.to_datetime(actuals.index)
    if not isinstance(forecasts.index, pd.DatetimeIndex):
        forecasts = forecasts.copy()
        forecasts.index = pd.to_datetime(forecasts.index)

    # Align
    common_idx = actuals.index.intersection(forecasts.index)
    if len(common_idx) < 3:
        return None

    a = actuals.loc[common_idx]
    f = forecasts.loc[common_idx]

    # Remove NaN
    valid = ~(a.isna() | f.isna())
    a = a[valid]
    f = f[valid]

    if len(a) < 3:
        return None

    errors = f - a

    # Metrics
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(np.abs(errors))

    # Directional accuracy
    actual_change = a.diff()
    forecast_change = f.diff()

    # Compare signs (excluding first obs which has NaN diff)
    actual_dir = np.sign(actual_change.iloc[1:])
    forecast_dir = np.sign(forecast_change.iloc[1:])

    correct_dir = (actual_dir == forecast_dir).sum()
    total_dir = len(actual_dir)
    dir_accuracy = (correct_dir / total_dir * 100) if total_dir > 0 else np.nan

    return {
        'period': period_name,
        'n_obs': len(a),
        'rmse': rmse,
        'mae': mae,
        'dir_accuracy': dir_accuracy,
    }

def main():
    print("=" * 70)
    print("MODEL VERIFICATION AND RANKING ANALYSIS")
    print("=" * 70)

    # Load data
    print("\n1. Loading actual reserves data...")
    actuals_full = load_actuals()
    # Ensure datetime index
    if not isinstance(actuals_full.index, pd.DatetimeIndex):
        actuals_full.index = pd.to_datetime(actuals_full.index)
    print(f"   Total observations: {len(actuals_full)}")
    print(f"   Date range: {actuals_full.index.min()} to {actuals_full.index.max()}")

    # Filter to test period
    actuals = actuals_full[(actuals_full.index >= TEST_START) & (actuals_full.index <= TEST_END)]
    print(f"   Test period observations: {len(actuals)}")

    print("\n2. Loading all model forecasts...")
    forecasts = load_all_forecasts()
    print(f"   Models loaded: {len(forecasts)}")
    for name in sorted(forecasts.keys()):
        fc = forecasts[name]
        # Ensure index is datetime
        if not isinstance(fc.index, pd.DatetimeIndex):
            fc.index = pd.to_datetime(fc.index)
            forecasts[name] = fc
        valid = fc.dropna()
        test_valid = valid[(valid.index >= TEST_START) & (valid.index <= TEST_END)]
        print(f"      {name}: {len(test_valid)} test obs")

    # Compute naive benchmark
    print("\n3. Computing naive (random walk) benchmark...")
    naive_forecast = compute_naive_forecast(actuals_full)
    # Ensure datetime index
    if not isinstance(naive_forecast.index, pd.DatetimeIndex):
        naive_forecast.index = pd.to_datetime(naive_forecast.index)
    naive_test = naive_forecast[(naive_forecast.index >= TEST_START) & (naive_forecast.index <= TEST_END)]
    naive_metrics = compute_metrics(actuals, naive_test, "Full Test")
    print(f"   Naive RMSE: {naive_metrics['rmse']:.2f} USD million")
    print(f"   Naive MAE: {naive_metrics['mae']:.2f} USD million")
    print(f"   Naive Dir Accuracy: {naive_metrics['dir_accuracy']:.1f}%")

    # Compute metrics for all models
    print("\n4. Computing metrics for all models (test period)...")
    results = []

    # Add naive
    naive_metrics['model'] = 'Naive (RW)'
    naive_metrics['normalized_rmse'] = 1.0  # by definition
    results.append(naive_metrics)

    for name, fc in forecasts.items():
        fc_test = fc[(fc.index >= TEST_START) & (fc.index <= TEST_END)]
        metrics = compute_metrics(actuals, fc_test, "Full Test")
        if metrics:
            metrics['model'] = name
            metrics['normalized_rmse'] = metrics['rmse'] / naive_metrics['rmse']
            results.append(metrics)

    # Create results DataFrame
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('rmse')

    print("\n" + "=" * 70)
    print("FULL TEST PERIOD RESULTS (sorted by RMSE)")
    print("=" * 70)
    print(f"\nTest Period: {TEST_START.strftime('%Y-%m')} to {TEST_END.strftime('%Y-%m')}")
    print(f"Naive Benchmark RMSE: {naive_metrics['rmse']:.2f} USD million\n")

    # Format and print table
    print(f"{'Model':<20} {'N':>5} {'RMSE':>10} {'NRMSE':>8} {'MAE':>10} {'Dir Acc':>8}")
    print("-" * 65)
    for _, row in df_results.iterrows():
        print(f"{row['model']:<20} {row['n_obs']:>5} {row['rmse']:>10.2f} {row['normalized_rmse']:>8.3f} {row['mae']:>10.2f} {row['dir_accuracy']:>7.1f}%")

    # Save results
    df_results.to_csv(OUTPUT_DIR / "full_test_results.csv", index=False)

    # Subperiod analysis
    print("\n" + "=" * 70)
    print("SUBPERIOD RMSE ANALYSIS")
    print("=" * 70)

    subperiod_results = []
    for period_name, (start, end) in SUBPERIODS.items():
        actuals_sub = actuals[(actuals.index >= start) & (actuals.index <= end)]
        if len(actuals_sub) < 3:
            print(f"\nSkipping {period_name}: insufficient observations")
            continue

        print(f"\n{period_name} ({len(actuals_sub)} obs):")

        # Naive for this period
        naive_sub = naive_forecast[(naive_forecast.index >= start) & (naive_forecast.index <= end)]
        naive_sub_metrics = compute_metrics(actuals_sub, naive_sub, period_name)
        naive_rmse_sub = naive_sub_metrics['rmse'] if naive_sub_metrics else np.nan

        period_models = [{'period': period_name, 'model': 'Naive (RW)',
                         'rmse': naive_rmse_sub, 'n_obs': len(actuals_sub)}]

        for name, fc in forecasts.items():
            fc_sub = fc[(fc.index >= start) & (fc.index <= end)]
            metrics = compute_metrics(actuals_sub, fc_sub, period_name)
            if metrics:
                period_models.append({
                    'period': period_name,
                    'model': name,
                    'rmse': metrics['rmse'],
                    'n_obs': metrics['n_obs'],
                })

        # Sort and display top 5
        period_df = pd.DataFrame(period_models).sort_values('rmse')
        print(f"  {'Model':<20} {'RMSE':>10}")
        print("  " + "-" * 32)
        for _, row in period_df.head(6).iterrows():
            print(f"  {row['model']:<20} {row['rmse']:>10.2f}")

        subperiod_results.extend(period_models)

    # Save subperiod results
    pd.DataFrame(subperiod_results).to_csv(OUTPUT_DIR / "subperiod_results.csv", index=False)

    # Directional accuracy summary
    print("\n" + "=" * 70)
    print("DIRECTIONAL ACCURACY (% months predicting correct direction)")
    print("=" * 70)
    print(f"\n{'Model':<20} {'Dir Accuracy':>12}")
    print("-" * 35)
    df_sorted_dir = df_results.sort_values('dir_accuracy', ascending=False)
    for _, row in df_sorted_dir.iterrows():
        print(f"{row['model']:<20} {row['dir_accuracy']:>11.1f}%")

    # Generate plots for top 3 models
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    # Get top 3 models (excluding naive and models with crazy values)
    valid_models = df_results[(df_results['model'] != 'Naive (RW)') &
                              (df_results['normalized_rmse'] < 100)].head(3)['model'].tolist()
    print(f"Top 3 models: {valid_models}")

    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot actuals
    ax.plot(actuals.index, actuals.values, 'k-', linewidth=2.5, label='Actual', zorder=10)

    # Plot top 3 model forecasts
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, model_name in enumerate(valid_models):
        fc = forecasts[model_name]
        fc_test = fc[(fc.index >= TEST_START) & (fc.index <= TEST_END)]
        ax.plot(fc_test.index, fc_test.values,
                color=colors[i], linewidth=1.5, linestyle='--',
                label=model_name, alpha=0.8)

    # Highlight crisis periods
    crisis_periods = [
        (pd.Timestamp("2022-04-01"), pd.Timestamp("2022-12-01"), "Default", 0.15),
        (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-01"), "Recovery Begins", 0.10),
    ]

    for start, end, label, alpha in crisis_periods:
        if start >= TEST_START - pd.Timedelta(days=180):
            ax.axvspan(start, end, alpha=alpha, color='red', zorder=1)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Foreign Reserves (USD million)', fontsize=12)
    ax.set_title('Actual vs Forecast Reserves: Top 3 Models (Test Period)', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top3_models_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "top3_models_comparison.pdf", bbox_inches='tight')
    plt.close()

    print(f"   Saved: {OUTPUT_DIR / 'top3_models_comparison.png'}")

    # Second plot: Actual vs Naive vs Best Model
    fig2, ax2 = plt.subplots(figsize=(14, 8))

    # Plot actuals
    ax2.plot(actuals.index, actuals.values, 'k-', linewidth=2.5, label='Actual', zorder=10)

    # Plot naive
    ax2.plot(naive_test.index, naive_test.values, 'r--', linewidth=1.5, label='Naive (Random Walk)', alpha=0.8)

    # Plot best non-naive model
    if len(valid_models) > 0:
        best_model = valid_models[0]
        fc = forecasts[best_model]
        fc_test = fc[(fc.index >= TEST_START) & (fc.index <= TEST_END)]
        ax2.plot(fc_test.index, fc_test.values, 'b--', linewidth=1.5, label=f'{best_model} (Best)', alpha=0.8)

    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Foreign Reserves (USD million)', fontsize=12)
    ax2.set_title('Reserves Forecast: Actual vs Naive vs Best Model', fontsize=14)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "actual_vs_naive_vs_best.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {OUTPUT_DIR / 'actual_vs_naive_vs_best.png'}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    best_model = df_results.iloc[0]
    print(f"\nBest Model: {best_model['model']}")
    print(f"  Test RMSE: {best_model['rmse']:.2f} USD million")
    print(f"  Normalized RMSE: {best_model['normalized_rmse']:.3f}")
    print(f"  Improvement vs Naive: {(1 - best_model['normalized_rmse']) * 100:.1f}%")
    print(f"  Directional Accuracy: {best_model['dir_accuracy']:.1f}%")

    print(f"\nNaive (Random Walk) Benchmark:")
    print(f"  RMSE: {naive_metrics['rmse']:.2f} USD million")
    print(f"  Directional Accuracy: {naive_metrics['dir_accuracy']:.1f}%")

    # Variables used
    print(f"\nTarget Variable: gross_reserves_usd_m")
    print(f"Sample Period: {actuals_full.index.min().strftime('%Y-%m')} to {actuals_full.index.max().strftime('%Y-%m')}")
    print(f"Train/Valid Split: 2019-12-01")
    print(f"Valid/Test Split: 2022-12-01")

    return df_results

if __name__ == "__main__":
    results = main()
