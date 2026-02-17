"""
Ensemble and Stacking Methods for Reserves Forecasting
=======================================================
Combines multiple base model forecasts using:
1. Simple averaging
2. Inverse-RMSE weighted averaging
3. Trimmed mean (drop worst models)
4. Stacking with Ridge meta-learner
5. Stacking with XGBoost meta-learner
6. Dynamic model selection (best recent performer)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "reserves_project" / "academic_deliverables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Periods
TRAIN_END = pd.Timestamp("2019-12-01")
VALID_END = pd.Timestamp("2022-12-01")
TEST_START = pd.Timestamp("2023-01-01")
RECOVERY_START = pd.Timestamp("2024-07-01")
RECOVERY_END = pd.Timestamp("2025-12-01")

TARGET = 'gross_reserves_usd_m'

# ============================================================================
# Load Base Model Forecasts
# ============================================================================

def load_all_base_forecasts():
    """Load forecasts from all available models."""
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
            if 'forecast' in df.columns:
                forecasts[name] = df['forecast']
                if actuals is None and 'actual' in df.columns:
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
        if actuals is None and 'actual' in df.columns:
            actuals = df['actual']

    # 4. XGBoost
    xgb_path = DATA_DIR / "model_verification" / "xgboost_forecasts.csv"
    if xgb_path.exists():
        df = pd.read_csv(xgb_path, parse_dates=['date'])
        df = df.set_index('date')
        if 'forecast' in df.columns:
            forecasts['XGBoost'] = df['forecast']

    # 5. LSTM
    lstm_path = DATA_DIR / "model_verification" / "lstm_forecasts.csv"
    if lstm_path.exists():
        df = pd.read_csv(lstm_path, parse_dates=['date'])
        df = df.set_index('date')
        if 'forecast' in df.columns:
            forecasts['LSTM'] = df['forecast']

    # 6. Naive
    if actuals is not None:
        forecasts['Naive'] = actuals.shift(1)

    return actuals, forecasts

def create_forecast_matrix(actuals, forecasts, start, end):
    """Create aligned matrix of forecasts for ensemble methods."""
    # Get common dates
    common_dates = actuals.index[(actuals.index >= start) & (actuals.index <= end)]

    for name, fc in forecasts.items():
        fc_dates = fc.dropna().index
        common_dates = common_dates.intersection(fc_dates)

    if len(common_dates) < 3:
        return None, None, None

    # Build matrix
    X = pd.DataFrame(index=common_dates)
    for name, fc in forecasts.items():
        X[name] = fc.loc[common_dates]

    y = actuals.loc[common_dates]

    # Drop any rows with NaN
    valid_mask = X.notna().all(axis=1) & y.notna()
    X = X[valid_mask]
    y = y[valid_mask]

    return X, y, X.columns.tolist()

# ============================================================================
# Ensemble Methods
# ============================================================================

def simple_average(X):
    """Equal-weighted average of all forecasts."""
    return X.mean(axis=1)

def inverse_rmse_weighted(X, y, validation_window=12):
    """Weight by inverse validation RMSE."""
    weights = {}

    for col in X.columns:
        # Use first `validation_window` points for weight estimation
        if len(X) > validation_window:
            val_pred = X[col].iloc[:validation_window]
            val_actual = y.iloc[:validation_window]
            rmse = np.sqrt(mean_squared_error(val_actual, val_pred))
            weights[col] = 1 / (rmse + 1e-6)
        else:
            weights[col] = 1.0

    # Normalize weights
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    # Apply weights
    result = pd.Series(0.0, index=X.index)
    for col, w in weights.items():
        result += w * X[col]

    return result, weights

def trimmed_mean(X, y, drop_n=3, validation_window=12):
    """Drop worst N models, average the rest."""
    # Compute validation RMSE for each model
    rmses = {}
    for col in X.columns:
        if len(X) > validation_window:
            val_pred = X[col].iloc[:validation_window]
            val_actual = y.iloc[:validation_window]
            rmses[col] = np.sqrt(mean_squared_error(val_actual, val_pred))
        else:
            rmses[col] = np.inf

    # Sort by RMSE and drop worst
    sorted_models = sorted(rmses.items(), key=lambda x: x[1])
    keep_models = [m[0] for m in sorted_models[:-drop_n]] if len(sorted_models) > drop_n else [m[0] for m in sorted_models]

    return X[keep_models].mean(axis=1), keep_models

def stacking_ridge(X, y, alpha=1.0):
    """Stack with Ridge regression meta-learner."""
    # Use time-series split: first 50% for training meta-learner
    split_idx = len(X) // 2

    if split_idx < 5:
        return simple_average(X), None, "insufficient_data"

    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]

    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X_train, y_train)

    # Predict for full period (in production would only use test)
    predictions = pd.Series(model.predict(X), index=X.index)

    weights = dict(zip(X.columns, model.coef_))

    return predictions, weights, model

def stacking_elastic(X, y, alpha=1.0, l1_ratio=0.5):
    """Stack with ElasticNet meta-learner."""
    split_idx = len(X) // 2

    if split_idx < 5:
        return simple_average(X), None, "insufficient_data"

    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]

    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True, max_iter=5000)
    model.fit(X_train, y_train)

    predictions = pd.Series(model.predict(X), index=X.index)
    weights = dict(zip(X.columns, model.coef_))

    return predictions, weights, model

def stacking_xgboost(X, y):
    """Stack with XGBoost meta-learner."""
    if not HAS_XGB:
        return simple_average(X), None, "xgboost_not_available"

    split_idx = len(X) // 2

    if split_idx < 5:
        return simple_average(X), None, "insufficient_data"

    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
    X_valid, y_valid = X.iloc[split_idx:split_idx + split_idx//2], y.iloc[split_idx:split_idx + split_idx//2]

    if len(X_valid) < 3:
        X_valid, y_valid = X_train, y_train

    model = xgb.XGBRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
        early_stopping_rounds=5
    )

    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

    predictions = pd.Series(model.predict(X), index=X.index)
    weights = dict(zip(X.columns, model.feature_importances_))

    return predictions, weights, model

def dynamic_selection(X, y, lookback=6):
    """Select best model based on recent performance."""
    predictions = pd.Series(index=X.index, dtype=float)
    selected_models = []

    for i in range(len(X)):
        if i < lookback:
            # Not enough history, use simple average
            predictions.iloc[i] = X.iloc[i].mean()
            selected_models.append("Average")
        else:
            # Compute recent RMSE for each model
            recent_errors = {}
            for col in X.columns:
                errors = (X[col].iloc[i-lookback:i] - y.iloc[i-lookback:i]) ** 2
                recent_errors[col] = np.sqrt(errors.mean())

            # Select best model
            best_model = min(recent_errors, key=recent_errors.get)
            predictions.iloc[i] = X[best_model].iloc[i]
            selected_models.append(best_model)

    return predictions, selected_models

def best_n_average(X, y, n=3, validation_window=12):
    """Average top N models by validation RMSE."""
    rmses = {}
    for col in X.columns:
        if len(X) > validation_window:
            val_pred = X[col].iloc[:validation_window]
            val_actual = y.iloc[:validation_window]
            rmses[col] = np.sqrt(mean_squared_error(val_actual, val_pred))
        else:
            rmses[col] = np.inf

    sorted_models = sorted(rmses.items(), key=lambda x: x[1])
    best_models = [m[0] for m in sorted_models[:n]]

    return X[best_models].mean(axis=1), best_models

# ============================================================================
# Evaluation
# ============================================================================

def compute_metrics(actuals, forecast):
    """Compute all metrics."""
    common = actuals.index.intersection(forecast.index)
    a = actuals.loc[common].dropna()
    f = forecast.loc[common].dropna()
    common = a.index.intersection(f.index)

    if len(common) < 3:
        return None

    a = a.loc[common]
    f = f.loc[common]

    errors = f.values - a.values
    abs_errors = np.abs(errors)
    sq_errors = errors ** 2

    n = len(a)
    rmse = np.sqrt(np.mean(sq_errors))
    mae = np.mean(abs_errors)
    mape = np.mean(np.abs(errors / a.values)) * 100

    ss_res = np.sum(sq_errors)
    ss_tot = np.sum((a.values - np.mean(a.values)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    actual_dir = np.sign(np.diff(a.values))
    forecast_dir = np.sign(np.diff(f.values))
    dir_acc = np.mean(actual_dir == forecast_dir) * 100 if len(actual_dir) > 0 else np.nan

    return {
        'n': n,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r_squared': r_squared,
        'dir_acc': dir_acc,
    }

def diebold_mariano_test(errors1, errors2):
    """Diebold-Mariano test."""
    d = errors1**2 - errors2**2
    n = len(d)
    if n < 10:
        return np.nan, np.nan
    d_mean = np.mean(d)
    var_d = np.var(d) / n
    if var_d <= 0:
        return np.nan, np.nan
    dm_stat = d_mean / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    return dm_stat, p_value

# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("ENSEMBLE STACKING FOR RESERVES FORECASTING")
    print("=" * 80)

    # Load base forecasts
    print("\nLoading base model forecasts...")
    actuals, forecasts = load_all_base_forecasts()
    print(f"Loaded {len(forecasts)} base models: {list(forecasts.keys())}")

    # Create forecast matrix for post-crisis period
    print(f"\nCreating forecast matrix for post-crisis period ({RECOVERY_START} to {RECOVERY_END})...")
    X, y, model_names = create_forecast_matrix(actuals, forecasts, RECOVERY_START, RECOVERY_END)

    if X is None:
        print("ERROR: Could not create forecast matrix")
        return

    print(f"Matrix shape: {X.shape} (dates × models)")
    print(f"Models in matrix: {model_names}")

    # Get naive benchmark
    naive = forecasts['Naive']
    naive_metrics = compute_metrics(actuals, naive)
    naive_rmse = naive_metrics['rmse'] if naive_metrics else np.nan
    print(f"\nNaive benchmark RMSE: {naive_rmse:.2f}")

    # Run ensemble methods
    print("\n" + "=" * 80)
    print("RUNNING ENSEMBLE METHODS")
    print("=" * 80)

    results = []
    ensemble_forecasts = {}

    # 1. Simple Average
    print("\n1. Simple Average...")
    avg_pred = simple_average(X)
    ensemble_forecasts['SimpleAvg'] = avg_pred
    metrics = compute_metrics(y, avg_pred)
    if metrics:
        vs_naive = (metrics['rmse'] / naive_rmse - 1) * 100 if naive_rmse else np.nan
        results.append({'method': 'SimpleAvg', **metrics, 'vs_naive': vs_naive})
        print(f"   RMSE: {metrics['rmse']:.2f} ({vs_naive:+.1f}% vs naive)")

    # 2. Inverse-RMSE Weighted
    print("\n2. Inverse-RMSE Weighted...")
    weighted_pred, weights = inverse_rmse_weighted(X, y)
    ensemble_forecasts['InvRMSE'] = weighted_pred
    metrics = compute_metrics(y, weighted_pred)
    if metrics:
        vs_naive = (metrics['rmse'] / naive_rmse - 1) * 100 if naive_rmse else np.nan
        results.append({'method': 'InvRMSE_Weighted', **metrics, 'vs_naive': vs_naive})
        print(f"   RMSE: {metrics['rmse']:.2f} ({vs_naive:+.1f}% vs naive)")
        print(f"   Top weights: {sorted(weights.items(), key=lambda x: -x[1])[:3]}")

    # 3. Trimmed Mean (drop worst 3)
    print("\n3. Trimmed Mean (drop worst 3)...")
    trimmed_pred, kept_models = trimmed_mean(X, y, drop_n=3)
    ensemble_forecasts['Trimmed'] = trimmed_pred
    metrics = compute_metrics(y, trimmed_pred)
    if metrics:
        vs_naive = (metrics['rmse'] / naive_rmse - 1) * 100 if naive_rmse else np.nan
        results.append({'method': 'TrimmedMean', **metrics, 'vs_naive': vs_naive})
        print(f"   RMSE: {metrics['rmse']:.2f} ({vs_naive:+.1f}% vs naive)")
        print(f"   Kept models: {kept_models}")

    # 4. Best-3 Average
    print("\n4. Best-3 Average...")
    best3_pred, best3_models = best_n_average(X, y, n=3)
    ensemble_forecasts['Best3Avg'] = best3_pred
    metrics = compute_metrics(y, best3_pred)
    if metrics:
        vs_naive = (metrics['rmse'] / naive_rmse - 1) * 100 if naive_rmse else np.nan
        results.append({'method': 'Best3Avg', **metrics, 'vs_naive': vs_naive})
        print(f"   RMSE: {metrics['rmse']:.2f} ({vs_naive:+.1f}% vs naive)")
        print(f"   Best 3 models: {best3_models}")

    # 5. Stacking with Ridge
    print("\n5. Stacking with Ridge...")
    ridge_pred, ridge_weights, ridge_model = stacking_ridge(X, y)
    ensemble_forecasts['StackRidge'] = ridge_pred
    metrics = compute_metrics(y, ridge_pred)
    if metrics:
        vs_naive = (metrics['rmse'] / naive_rmse - 1) * 100 if naive_rmse else np.nan
        results.append({'method': 'Stack_Ridge', **metrics, 'vs_naive': vs_naive})
        print(f"   RMSE: {metrics['rmse']:.2f} ({vs_naive:+.1f}% vs naive)")
        if ridge_weights:
            print(f"   Coefficients: {sorted(ridge_weights.items(), key=lambda x: -abs(x[1]))[:3]}")

    # 6. Stacking with ElasticNet
    print("\n6. Stacking with ElasticNet...")
    elastic_pred, elastic_weights, elastic_model = stacking_elastic(X, y)
    ensemble_forecasts['StackElastic'] = elastic_pred
    metrics = compute_metrics(y, elastic_pred)
    if metrics:
        vs_naive = (metrics['rmse'] / naive_rmse - 1) * 100 if naive_rmse else np.nan
        results.append({'method': 'Stack_ElasticNet', **metrics, 'vs_naive': vs_naive})
        print(f"   RMSE: {metrics['rmse']:.2f} ({vs_naive:+.1f}% vs naive)")
        if elastic_weights:
            nonzero = {k: v for k, v in elastic_weights.items() if abs(v) > 0.01}
            print(f"   Non-zero coefficients: {nonzero}")

    # 7. Stacking with XGBoost
    print("\n7. Stacking with XGBoost...")
    xgb_pred, xgb_weights, xgb_model = stacking_xgboost(X, y)
    ensemble_forecasts['StackXGB'] = xgb_pred
    metrics = compute_metrics(y, xgb_pred)
    if metrics:
        vs_naive = (metrics['rmse'] / naive_rmse - 1) * 100 if naive_rmse else np.nan
        results.append({'method': 'Stack_XGBoost', **metrics, 'vs_naive': vs_naive})
        print(f"   RMSE: {metrics['rmse']:.2f} ({vs_naive:+.1f}% vs naive)")
        if xgb_weights:
            print(f"   Top importances: {sorted(xgb_weights.items(), key=lambda x: -x[1])[:3]}")

    # 8. Dynamic Selection
    print("\n8. Dynamic Model Selection...")
    dynamic_pred, selected_models = dynamic_selection(X, y, lookback=3)
    ensemble_forecasts['Dynamic'] = dynamic_pred
    metrics = compute_metrics(y, dynamic_pred)
    if metrics:
        vs_naive = (metrics['rmse'] / naive_rmse - 1) * 100 if naive_rmse else np.nan
        results.append({'method': 'DynamicSelection', **metrics, 'vs_naive': vs_naive})
        print(f"   RMSE: {metrics['rmse']:.2f} ({vs_naive:+.1f}% vs naive)")
        from collections import Counter
        model_counts = Counter(selected_models)
        print(f"   Selection frequency: {model_counts.most_common(3)}")

    # Add XGBoost standalone for comparison
    print("\n9. XGBoost (standalone, for comparison)...")
    if 'XGBoost' in forecasts:
        xgb_standalone = forecasts['XGBoost']
        metrics = compute_metrics(actuals, xgb_standalone)
        if metrics:
            vs_naive = (metrics['rmse'] / naive_rmse - 1) * 100 if naive_rmse else np.nan
            results.append({'method': 'XGBoost_Standalone', **metrics, 'vs_naive': vs_naive})
            print(f"   RMSE: {metrics['rmse']:.2f} ({vs_naive:+.1f}% vs naive)")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - Sorted by RMSE")
    print("=" * 80)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('rmse')

    print(f"\n{'Method':<25} {'N':>4} {'RMSE':>10} {'vs Naive':>12} {'MAPE':>8} {'Dir Acc':>8}")
    print("-" * 75)

    for _, row in results_df.iterrows():
        beats = "✓" if row['vs_naive'] < 0 else ""
        print(f"{row['method']:<25} {row['n']:>4} {row['rmse']:>10.2f} {row['vs_naive']:>+11.1f}% {row['mape']:>7.1f}% {row['dir_acc']:>7.1f}% {beats}")

    # Save results
    results_df.to_csv(OUTPUT_DIR / "ensemble_results.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'ensemble_results.csv'}")

    # Save ensemble forecasts
    ensemble_df = pd.DataFrame(ensemble_forecasts)
    ensemble_df['actual'] = y
    ensemble_df.to_csv(OUTPUT_DIR / "ensemble_forecasts.csv")
    print(f"Saved: {OUTPUT_DIR / 'ensemble_forecasts.csv'}")

    # Winners
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    winners = results_df[results_df['vs_naive'] < 0]
    if not winners.empty:
        print(f"\nMethods beating naive ({len(winners)}):")
        for _, row in winners.iterrows():
            print(f"  {row['method']}: {row['vs_naive']:+.1f}% vs naive")
    else:
        print("\nNo ensemble method beats naive in post-crisis period")

    best = results_df.iloc[0]
    print(f"\nBest method: {best['method']} (RMSE={best['rmse']:.2f}, {best['vs_naive']:+.1f}% vs naive)")

    return results_df

if __name__ == "__main__":
    main()
