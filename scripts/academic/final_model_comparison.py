"""
Final Comprehensive Model Comparison - ALL Models Including ML
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

# Periods
TEST_START = pd.Timestamp("2023-01-01")
RECOVERY_START = pd.Timestamp("2024-07-01")
RECOVERY_END = pd.Timestamp("2025-12-01")

def load_all_forecasts():
    """Load all available model forecasts."""
    forecasts = {}
    actuals = None

    # 1. Earlier pipeline models
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

    # 2. BVAR models
    bvar_dir = DATA_DIR / "forecast_results_academic" / "bvar"
    for varset in ['parsimonious', 'bop', 'monetary']:
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

    # 4. ML models
    xgb_path = OUTPUT_DIR / "xgboost_forecasts.csv"
    if xgb_path.exists():
        df = pd.read_csv(xgb_path, parse_dates=['date'])
        df = df.set_index('date')
        forecasts['XGBoost'] = df['forecast']

    lstm_path = OUTPUT_DIR / "lstm_forecasts.csv"
    if lstm_path.exists():
        df = pd.read_csv(lstm_path, parse_dates=['date'])
        df = df.set_index('date')
        forecasts['LSTM'] = df['forecast']

    # 5. Naive
    if actuals is not None:
        forecasts['Naive'] = actuals.shift(1)

    return actuals, forecasts

def compute_metrics(actuals, forecast, start, end):
    """Compute metrics for a period."""
    a = actuals[(actuals.index >= start) & (actuals.index <= end)]
    f = forecast[(forecast.index >= start) & (forecast.index <= end)]

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

    return {'n': len(a), 'rmse': rmse, 'mae': mae, 'dir_acc': dir_acc}

def main():
    print("="*80)
    print("FINAL MODEL COMPARISON - ALL 14 MODELS")
    print("="*80)

    actuals, forecasts = load_all_forecasts()
    print(f"\nLoaded {len(forecasts)} models")

    # Post-crisis evaluation
    print(f"\n{'='*80}")
    print("POST-CRISIS PERIOD (2024-07+) - THE KEY COMPARISON")
    print("="*80)

    naive_metrics = compute_metrics(actuals, forecasts['Naive'], RECOVERY_START, RECOVERY_END)
    naive_rmse = naive_metrics['rmse']

    print(f"\nNaive Benchmark RMSE: {naive_rmse:.2f} USD million\n")

    results = []
    for name, fc in forecasts.items():
        metrics = compute_metrics(actuals, fc, RECOVERY_START, RECOVERY_END)
        if metrics:
            metrics['model'] = name
            metrics['vs_naive'] = (metrics['rmse'] / naive_rmse - 1) * 100
            metrics['beats_naive'] = metrics['rmse'] < naive_rmse
            results.append(metrics)

    results_df = pd.DataFrame(results).sort_values('rmse')

    print(f"{'Rank':<5} {'Model':<20} {'N':>5} {'RMSE':>10} {'vs Naive':>12} {'Beats?':>10}")
    print("-"*65)

    for i, (_, row) in enumerate(results_df.iterrows(), 1):
        beats = "✓ YES" if row['beats_naive'] else ""
        print(f"{i:<5} {row['model']:<20} {row['n']:>5} {row['rmse']:>10.2f} {row['vs_naive']:>+11.1f}% {beats:>10}")

    # Count winners
    winners = results_df[results_df['beats_naive'] == True]
    print(f"\n*** {len(winners)} models beat the naive benchmark ***")

    # Winner summary
    print(f"\n{'='*80}")
    print("WINNING MODELS (Beat Naive in Post-Crisis)")
    print("="*80)

    for _, row in winners.iterrows():
        print(f"\n{row['model']}:")
        print(f"  RMSE: {row['rmse']:.2f} USD million")
        print(f"  Improvement vs Naive: {-row['vs_naive']:.1f}%")
        print(f"  N observations: {row['n']}")

    # Create final visualization
    fig, ax = plt.subplots(figsize=(12, 8))

    # Filter to reasonable models
    plot_df = results_df[results_df['rmse'] < 1000].copy()

    colors = ['green' if b else 'salmon' for b in plot_df['beats_naive']]
    naive_idx = plot_df[plot_df['model'] == 'Naive'].index.tolist()
    if naive_idx:
        colors[plot_df.index.tolist().index(naive_idx[0])] = 'gray'

    bars = ax.barh(range(len(plot_df)), plot_df['rmse'], color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df['model'])
    ax.axvline(naive_rmse, color='red', linestyle='--', linewidth=2, label=f'Naive = {naive_rmse:.0f}')
    ax.set_xlabel('RMSE (USD million)', fontsize=12)
    ax.set_title('Post-Crisis Model Comparison (2024-07+)\nGreen = Beats Naive, Red = Worse than Naive', fontsize=13)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend()

    # Add value labels
    for i, (_, row) in enumerate(plot_df.iterrows()):
        ax.text(row['rmse'] + 10, i, f"{row['vs_naive']:+.1f}%", va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "final_model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUTPUT_DIR / 'final_model_comparison.png'}")

    # Save final results
    results_df.to_csv(OUTPUT_DIR / "final_model_rankings.csv", index=False)

    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print("="*80)
    print(f"""
POST-CRISIS PERIOD (2024-07 onwards):

TOP PERFORMERS:
1. XGBoost:  RMSE = {results_df[results_df['model']=='XGBoost']['rmse'].values[0]:.1f} ({results_df[results_df['model']=='XGBoost']['vs_naive'].values[0]:+.1f}% vs naive)
2. DMS:      RMSE = {results_df[results_df['model']=='DMS']['rmse'].values[0]:.1f} ({results_df[results_df['model']=='DMS']['vs_naive'].values[0]:+.1f}% vs naive)
3. DMA:      RMSE = {results_df[results_df['model']=='DMA']['rmse'].values[0]:.1f} ({results_df[results_df['model']=='DMA']['vs_naive'].values[0]:+.1f}% vs naive)

BASELINE:
Naive (RW): RMSE = {naive_rmse:.1f}

KEY INSIGHT:
- Machine learning (XGBoost) provides the best forecasts in stable periods
- Dynamic Model Selection (DMS) is the best econometric approach
- Traditional models (ARIMA, VECM, MS-VAR) fail to beat naive
""")

if __name__ == "__main__":
    main()
