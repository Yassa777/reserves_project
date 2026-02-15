"""
Comprehensive Model Comparison - ALL Available Models

Including:
- MS-VAR (Markov-Switching VAR)
- MS-VECM (Markov-Switching VECM)
- ARIMA
- VECM
- BVAR variants
- DMA/DMS
- Naive (Random Walk)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "model_verification"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Periods
TEST_START = pd.Timestamp("2023-01-01")
TEST_END = pd.Timestamp("2025-12-01")
RECOVERY_START = pd.Timestamp("2024-07-01")
RECOVERY_END = pd.Timestamp("2025-12-01")

def load_all_models():
    """Load ALL available model forecasts."""
    forecasts = {}
    actuals = None

    # 1. Earlier pipeline models (MS-VAR, MS-VECM, ARIMA, VECM, Naive)
    earlier_models = {
        'MS-VAR': 'data/forecast_results/ms_var_forecast.csv',
        'MS-VECM': 'data/forecast_results/ms_vecm_forecast.csv',
        'ARIMA': 'data/forecast_results/arima_forecast.csv',
        'VECM': 'data/forecast_results/vecm_forecast.csv',
    }

    for name, path in earlier_models.items():
        full_path = DATA_DIR.parent / path
        if full_path.exists():
            df = pd.read_csv(full_path, parse_dates=['date'])
            df = df.set_index('date')
            forecasts[name] = df['forecast']
            if actuals is None:
                actuals = df['actual']

    # 2. BVAR models from academic pipeline
    bvar_dir = DATA_DIR / "forecast_results_academic" / "bvar"
    for varset in ['parsimonious', 'bop', 'monetary', 'pca', 'full']:
        path = bvar_dir / f"bvar_rolling_backtest_{varset}.csv"
        if path.exists():
            df = pd.read_csv(path)
            if 'forecast_date' in df.columns:
                df['forecast_date'] = pd.to_datetime(df['forecast_date'])
                h1 = df[df['horizon'] == 1].set_index('forecast_date')
                if 'forecast_point' in h1.columns:
                    forecasts[f'BVAR_{varset}'] = h1['forecast_point']

    # 3. DMA/DMS
    dma_path = DATA_DIR / "forecast_results_academic" / "dma" / "dma_rolling_backtest.csv"
    if dma_path.exists():
        df = pd.read_csv(dma_path, parse_dates=['date'])
        df = df.set_index('date')
        if 'dma_forecast' in df.columns:
            forecasts['DMA'] = df['dma_forecast']
        if 'dms_forecast' in df.columns:
            forecasts['DMS'] = df['dms_forecast']
        if actuals is None:
            actuals = df['actual']

    # 4. Naive (compute from actuals)
    if actuals is not None:
        forecasts['Naive'] = actuals.shift(1)

    return actuals, forecasts

def compute_metrics(actuals, forecast, period_start, period_end):
    """Compute RMSE, MAE for a specific period."""
    # Filter to period
    a = actuals[(actuals.index >= period_start) & (actuals.index <= period_end)]
    f = forecast[(forecast.index >= period_start) & (forecast.index <= period_end)]

    # Align
    common = a.index.intersection(f.index)
    if len(common) < 3:
        return None

    a = a.loc[common].dropna()
    f = f.loc[common].dropna()
    common = a.index.intersection(f.index)

    if len(common) < 3:
        return None

    a = a.loc[common]
    f = f.loc[common]

    errors = f - a
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(np.abs(errors))

    # Directional accuracy
    actual_dir = np.sign(a.diff().iloc[1:])
    forecast_dir = np.sign(f.diff().iloc[1:])
    dir_acc = (actual_dir == forecast_dir).mean() * 100

    return {
        'n': len(a),
        'rmse': rmse,
        'mae': mae,
        'dir_acc': dir_acc,
    }

def main():
    print("="*80)
    print("COMPREHENSIVE MODEL COMPARISON - ALL AVAILABLE MODELS")
    print("="*80)

    # Load all models
    actuals, forecasts = load_all_models()
    print(f"\nLoaded {len(forecasts)} models:")
    for name in sorted(forecasts.keys()):
        fc = forecasts[name].dropna()
        print(f"  {name}: {len(fc)} obs")

    # Define periods
    periods = {
        'Full Test (2023-01+)': (TEST_START, TEST_END),
        'Post-Crisis (2024-07+)': (RECOVERY_START, RECOVERY_END),
    }

    for period_name, (start, end) in periods.items():
        print(f"\n{'='*80}")
        print(f"{period_name}")
        print(f"{'='*80}")

        # Get naive benchmark
        naive_metrics = compute_metrics(actuals, forecasts['Naive'], start, end)
        naive_rmse = naive_metrics['rmse'] if naive_metrics else np.nan

        print(f"\nNaive Benchmark RMSE: {naive_rmse:.2f} USD million\n")

        results = []
        for name, fc in forecasts.items():
            metrics = compute_metrics(actuals, fc, start, end)
            if metrics:
                metrics['model'] = name
                metrics['vs_naive'] = (metrics['rmse'] / naive_rmse - 1) * 100
                metrics['beats_naive'] = metrics['rmse'] < naive_rmse
                results.append(metrics)

        # Sort by RMSE
        results_df = pd.DataFrame(results).sort_values('rmse')

        # Display
        print(f"{'Model':<20} {'N':>5} {'RMSE':>10} {'MAE':>10} {'vs Naive':>12} {'Dir Acc':>10} {'Beats?':>8}")
        print("-"*80)

        for _, row in results_df.iterrows():
            beats = "✓ YES" if row['beats_naive'] else ""
            print(f"{row['model']:<20} {row['n']:>5} {row['rmse']:>10.2f} {row['mae']:>10.2f} "
                  f"{row['vs_naive']:>+11.1f}% {row['dir_acc']:>9.1f}% {beats:>8}")

        # Count winners
        winners = results_df[results_df['beats_naive'] == True]
        losers = results_df[results_df['beats_naive'] == False]
        print(f"\n*** {len(winners)} models beat naive, {len(losers)-1} models worse than naive ***")

        # Save results
        results_df.to_csv(OUTPUT_DIR / f"all_models_{period_name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'plus')}.csv", index=False)

    # Create comparison plot
    print(f"\n{'='*80}")
    print("GENERATING COMPARISON PLOT")
    print("='*80")

    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    # Panel 1: Full test period
    ax1 = axes[0]
    results_full = []
    for name, fc in forecasts.items():
        metrics = compute_metrics(actuals, fc, TEST_START, TEST_END)
        if metrics and metrics['rmse'] < 5000:  # Filter out broken models
            metrics['model'] = name
            results_full.append(metrics)

    df1 = pd.DataFrame(results_full).sort_values('rmse')
    naive_rmse_full = df1[df1['model'] == 'Naive']['rmse'].values[0]

    colors1 = ['green' if r < naive_rmse_full else 'salmon' for r in df1['rmse']]
    colors1[df1[df1['model'] == 'Naive'].index.tolist()[0] - df1.index[0]] = 'gray'

    ax1.barh(range(len(df1)), df1['rmse'], color=colors1, alpha=0.7)
    ax1.set_yticks(range(len(df1)))
    ax1.set_yticklabels(df1['model'])
    ax1.axvline(naive_rmse_full, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('RMSE (USD million)')
    ax1.set_title(f'Full Test Period (2023-01 to 2025-12) - Naive RMSE = {naive_rmse_full:.0f}')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x')

    # Panel 2: Post-crisis period
    ax2 = axes[1]
    results_recovery = []
    for name, fc in forecasts.items():
        metrics = compute_metrics(actuals, fc, RECOVERY_START, RECOVERY_END)
        if metrics and metrics['rmse'] < 5000:
            metrics['model'] = name
            results_recovery.append(metrics)

    df2 = pd.DataFrame(results_recovery).sort_values('rmse')
    naive_rmse_recovery = df2[df2['model'] == 'Naive']['rmse'].values[0]

    colors2 = ['green' if r < naive_rmse_recovery else 'salmon' for r in df2['rmse']]
    colors2[df2[df2['model'] == 'Naive'].index.tolist()[0] - df2.index[0]] = 'gray'

    ax2.barh(range(len(df2)), df2['rmse'], color=colors2, alpha=0.7)
    ax2.set_yticks(range(len(df2)))
    ax2.set_yticklabels(df2['model'])
    ax2.axvline(naive_rmse_recovery, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('RMSE (USD million)')
    ax2.set_title(f'Post-Crisis Period (2024-07+) - Naive RMSE = {naive_rmse_recovery:.0f}')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "all_models_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'all_models_comparison.png'}")

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY: MODEL CATEGORIES")
    print("{'='*80}")

    print("""
MODEL TYPES EVALUATED:

1. REGIME-SWITCHING MODELS:
   - MS-VAR: Markov-Switching VAR (2 regimes)
   - MS-VECM: Markov-Switching VECM (2 regimes)

2. CLASSICAL ECONOMETRIC:
   - ARIMA: Autoregressive Integrated Moving Average
   - VECM: Vector Error Correction Model

3. BAYESIAN:
   - BVAR_*: Bayesian VAR with Minnesota prior (5 variable sets)

4. MODEL AVERAGING:
   - DMA: Dynamic Model Averaging
   - DMS: Dynamic Model Selection

5. BENCHMARK:
   - Naive: Random Walk (y_{t+1} = y_t)

WHAT WE DIDN'T TRY (potential additions):
   - TVP-VAR: Time-Varying Parameter VAR (implemented but numerically unstable)
   - LSTM/Neural Networks: Deep learning approaches
   - Ensemble methods: Random Forest, XGBoost on features
   - State-space models: Kalman Filter variants
   - Mixed-frequency: MIDAS (implemented but no test data)
""")

if __name__ == "__main__":
    main()
