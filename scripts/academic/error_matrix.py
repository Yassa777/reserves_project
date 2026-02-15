"""
Comprehensive Error Matrix for All Models
- RMSE, MAE, MAPE, sMAPE, R-squared, Theil-U
"""

import pandas as pd
import numpy as np
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

def compute_all_metrics(actuals, forecast, start, end):
    """Compute comprehensive error metrics."""
    # Filter to period
    a = actuals[(actuals.index >= start) & (actuals.index <= end)]
    f = forecast[(forecast.index >= start) & (forecast.index <= end)]

    # Align indices
    common = a.index.intersection(f.index)
    if len(common) < 3:
        return None

    a = a.loc[common].dropna()
    f = f.loc[common].dropna()
    common = a.index.intersection(f.index)

    if len(common) < 3:
        return None

    a = a.loc[common].values
    f = f.loc[common].values

    n = len(a)
    errors = f - a
    abs_errors = np.abs(errors)
    sq_errors = errors ** 2

    # Basic metrics
    rmse = np.sqrt(np.mean(sq_errors))
    mae = np.mean(abs_errors)
    me = np.mean(errors)  # Mean Error (bias)

    # MAPE - Mean Absolute Percentage Error
    # Avoid division by zero
    mape = np.mean(np.abs(errors / a)) * 100

    # sMAPE - Symmetric Mean Absolute Percentage Error
    # Formula: 100/n * sum(|F-A| / ((|A|+|F|)/2))
    smape = 100 * np.mean(2 * abs_errors / (np.abs(a) + np.abs(f)))

    # R-squared
    ss_res = np.sum(sq_errors)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    # Theil's U statistic (U1)
    # U = sqrt(sum((F-A)^2)) / (sqrt(sum(F^2)) + sqrt(sum(A^2)))
    theil_u1 = np.sqrt(np.sum(sq_errors)) / (np.sqrt(np.sum(f**2)) + np.sqrt(np.sum(a**2)))

    # Theil's U2 (relative to naive)
    # Compare to random walk forecast
    naive_errors = a[1:] - a[:-1]  # Naive forecast error = actual change
    model_errors = f[1:] - a[1:]   # Model forecast error
    theil_u2 = np.sqrt(np.mean(model_errors**2)) / np.sqrt(np.mean(naive_errors**2)) if len(naive_errors) > 0 else np.nan

    # Directional Accuracy
    actual_dir = np.sign(np.diff(a))
    forecast_dir = np.sign(np.diff(f))
    dir_accuracy = np.mean(actual_dir == forecast_dir) * 100 if len(actual_dir) > 0 else np.nan

    return {
        'n': n,
        'rmse': rmse,
        'mae': mae,
        'me': me,
        'mape': mape,
        'smape': smape,
        'r_squared': r_squared,
        'theil_u1': theil_u1,
        'theil_u2': theil_u2,
        'dir_accuracy': dir_accuracy,
    }

def main():
    print("="*100)
    print("COMPREHENSIVE ERROR MATRIX - ALL MODELS")
    print("="*100)

    actuals, forecasts = load_all_forecasts()
    print(f"\nLoaded {len(forecasts)} models")

    # Compute metrics for both periods
    periods = {
        'Full Test (2023-01+)': (TEST_START, pd.Timestamp("2025-12-01")),
        'Post-Crisis (2024-07+)': (RECOVERY_START, RECOVERY_END),
    }

    for period_name, (start, end) in periods.items():
        print(f"\n{'='*100}")
        print(f"{period_name}")
        print("="*100)

        results = []
        for name, fc in forecasts.items():
            metrics = compute_all_metrics(actuals, fc, start, end)
            if metrics:
                metrics['model'] = name
                results.append(metrics)

        if not results:
            print("No valid results for this period")
            continue

        results_df = pd.DataFrame(results)

        # Get naive metrics for comparison
        naive_row = results_df[results_df['model'] == 'Naive'].iloc[0]
        naive_rmse = naive_row['rmse']

        # Add relative metrics
        results_df['rmse_vs_naive'] = (results_df['rmse'] / naive_rmse - 1) * 100
        results_df['beats_naive'] = results_df['rmse'] < naive_rmse

        # Sort by RMSE
        results_df = results_df.sort_values('rmse')

        # Display main metrics table
        print(f"\n{'Model':<18} {'N':>4} {'RMSE':>10} {'MAE':>10} {'MAPE':>8} {'sMAPE':>8} {'R²':>8} {'Dir Acc':>8}")
        print("-"*90)

        for _, row in results_df.iterrows():
            r2_str = f"{row['r_squared']:.3f}" if not np.isnan(row['r_squared']) else "N/A"
            print(f"{row['model']:<18} {row['n']:>4} {row['rmse']:>10.2f} {row['mae']:>10.2f} "
                  f"{row['mape']:>7.1f}% {row['smape']:>7.1f}% {r2_str:>8} {row['dir_accuracy']:>7.1f}%")

        # Display Theil-U comparison
        print(f"\n{'Model':<18} {'Theil U1':>10} {'Theil U2':>10} {'vs Naive':>12} {'Beats?':>8}")
        print("-"*65)

        for _, row in results_df.iterrows():
            beats = "✓ YES" if row['beats_naive'] else ""
            u2_str = f"{row['theil_u2']:.3f}" if not np.isnan(row['theil_u2']) else "N/A"
            print(f"{row['model']:<18} {row['theil_u1']:>10.4f} {u2_str:>10} "
                  f"{row['rmse_vs_naive']:>+11.1f}% {beats:>8}")

        # Save to CSV
        filename = f"error_matrix_{period_name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'plus')}.csv"
        results_df.to_csv(OUTPUT_DIR / filename, index=False)
        print(f"\nSaved: {OUTPUT_DIR / filename}")

    # Summary interpretation
    print(f"\n{'='*100}")
    print("METRIC INTERPRETATION GUIDE")
    print("="*100)
    print("""
METRIC DEFINITIONS:

1. RMSE (Root Mean Square Error)
   - Standard error metric, penalizes large errors more
   - Lower is better
   - Units: USD million

2. MAE (Mean Absolute Error)
   - Average absolute error
   - Lower is better
   - Units: USD million

3. MAPE (Mean Absolute Percentage Error)
   - Error as percentage of actual value
   - Lower is better
   - Can be distorted by small actual values

4. sMAPE (Symmetric MAPE)
   - Bounded version of MAPE (0-200%)
   - Lower is better
   - More stable than MAPE

5. R² (R-squared / Coefficient of Determination)
   - Proportion of variance explained
   - 1.0 = perfect, 0 = same as mean, negative = worse than mean
   - Higher is better

6. Theil U1
   - Normalized forecast error (0-1 scale)
   - 0 = perfect, 1 = worst
   - Lower is better

7. Theil U2
   - Ratio of model RMSE to naive RMSE
   - < 1 means model beats naive
   - Lower is better

8. Directional Accuracy
   - % of times model correctly predicts direction of change
   - > 50% is better than random
   - Higher is better
""")

if __name__ == "__main__":
    main()
