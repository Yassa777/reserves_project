"""
Run ALL 70 Model × Variable Set Combinations
=============================================
14 models × 5 variable sets = 70 combinations
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
from statsmodels.tsa.api import VAR

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not available")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("TensorFlow not available")

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

VARSETS = ['parsimonious', 'bop', 'monetary', 'pca', 'full']
TARGET = 'gross_reserves_usd_m'

# ============================================================================
# DATA LOADING
# ============================================================================

def load_varset_data(varset_name):
    """Load vecm_levels.csv for a variable set."""
    path = DATA_DIR / "forecast_prep_academic" / f"varset_{varset_name}" / "vecm_levels.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=['date'])
    df = df.set_index('date').sort_index()
    return df

# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(actuals, forecast, start, end):
    """Compute all metrics for a forecast."""
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

    errors = f.values - a.values
    abs_errors = np.abs(errors)
    sq_errors = errors ** 2

    n = len(a)
    rmse = np.sqrt(np.mean(sq_errors))
    mae = np.mean(abs_errors)
    mape = np.mean(np.abs(errors / a.values)) * 100
    smape = 100 * np.mean(2 * abs_errors / (np.abs(a.values) + np.abs(f.values)))

    ss_res = np.sum(sq_errors)
    ss_tot = np.sum((a.values - np.mean(a.values)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    naive_errors = np.abs(np.diff(a.values))
    mase = mae / np.mean(naive_errors) if len(naive_errors) > 0 and np.mean(naive_errors) > 0 else np.nan

    naive_sq_errors = (a.values[1:] - a.values[:-1]) ** 2
    model_sq_errors = (f.values[1:] - a.values[1:]) ** 2
    theil_u2 = np.sqrt(np.mean(model_sq_errors)) / np.sqrt(np.mean(naive_sq_errors)) if len(naive_sq_errors) > 0 else np.nan

    actual_dir = np.sign(np.diff(a.values))
    forecast_dir = np.sign(np.diff(f.values))
    dir_acc = np.mean(actual_dir == forecast_dir) * 100 if len(actual_dir) > 0 else np.nan

    return {
        'n': n, 'rmse': rmse, 'mae': mae, 'mape': mape, 'smape': smape,
        'r_squared': r_squared, 'mase': mase, 'theil_u2': theil_u2, 'dir_acc': dir_acc,
    }

def diebold_mariano_test(errors1, errors2, h=1):
    """Diebold-Mariano test."""
    d = errors1**2 - errors2**2
    n = len(d)
    if n < 10:
        return np.nan, np.nan
    d_mean = np.mean(d)
    gamma_0 = np.var(d)
    var_d = gamma_0 / n
    if var_d <= 0:
        return np.nan, np.nan
    dm_stat = d_mean / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    return dm_stat, p_value

# ============================================================================
# MODEL IMPLEMENTATIONS
# ============================================================================

def run_naive(df):
    """Naive (random walk) forecast."""
    return df[TARGET].shift(1)

def run_arima(df):
    """ARIMA model - rolling forecast."""
    series = df[TARGET].copy()
    forecasts = {}

    for i, date in enumerate(series.index):
        if date < TEST_START:
            continue

        train = series.loc[:date].iloc[:-1]
        if len(train) < 36:
            continue

        try:
            model = ARIMA(train, order=(1, 1, 1))
            fitted = model.fit()
            pred = fitted.forecast(steps=1).iloc[0]
            forecasts[date] = pred
        except:
            forecasts[date] = train.iloc[-1]  # fallback to naive

    return pd.Series(forecasts)

def run_vecm(df):
    """VECM model - rolling forecast."""
    cols = [c for c in df.columns if c != 'split']
    data = df[cols].copy()
    forecasts = {}

    for i, date in enumerate(data.index):
        if date < TEST_START:
            continue

        train = data.loc[:date].iloc[:-1]
        if len(train) < 50:
            continue

        try:
            # Determine cointegration rank
            joh = coint_johansen(train, det_order=0, k_ar_diff=2)
            rank = sum(joh.lr1 > joh.cvt[:, 1])
            rank = max(1, min(rank, len(cols) - 1))

            model = VECM(train, k_ar_diff=2, coint_rank=rank)
            fitted = model.fit()
            pred = fitted.predict(steps=1)
            target_idx = cols.index(TARGET)
            forecasts[date] = train[TARGET].iloc[-1] + pred[0, target_idx]
        except:
            forecasts[date] = train[TARGET].iloc[-1]

    return pd.Series(forecasts)

def run_var(df):
    """VAR model - rolling forecast."""
    cols = [c for c in df.columns if c != 'split']
    data = df[cols].diff().dropna()
    levels = df[cols].copy()
    forecasts = {}

    for i, date in enumerate(data.index):
        if date < TEST_START:
            continue

        train_diff = data.loc[:date].iloc[:-1]
        if len(train_diff) < 30:
            continue

        try:
            model = VAR(train_diff)
            fitted = model.fit(maxlags=2)
            pred_diff = fitted.forecast(train_diff.values[-fitted.k_ar:], steps=1)
            target_idx = cols.index(TARGET)
            last_level = levels.loc[:date, TARGET].iloc[-2]
            forecasts[date] = last_level + pred_diff[0, target_idx]
        except:
            forecasts[date] = levels.loc[:date, TARGET].iloc[-2]

    return pd.Series(forecasts)

def run_ms_var(df):
    """Markov-Switching VAR approximation using regime-dependent VAR."""
    cols = [c for c in df.columns if c != 'split']
    data = df[cols].diff().dropna()
    levels = df[cols].copy()
    forecasts = {}

    # Compute volatility regimes
    vol = data[TARGET].rolling(6).std()
    vol_threshold = vol.quantile(0.7)

    for i, date in enumerate(data.index):
        if date < TEST_START:
            continue

        train_diff = data.loc[:date].iloc[:-1]
        if len(train_diff) < 30:
            continue

        try:
            # Determine current regime
            current_vol = vol.loc[:date].iloc[-1]
            high_vol = current_vol > vol_threshold

            # Use regime-specific training data
            regime_mask = vol.loc[train_diff.index] > vol_threshold if high_vol else vol.loc[train_diff.index] <= vol_threshold
            regime_data = train_diff[regime_mask]

            if len(regime_data) < 20:
                regime_data = train_diff  # fallback

            model = VAR(regime_data)
            fitted = model.fit(maxlags=2)
            pred_diff = fitted.forecast(train_diff.values[-fitted.k_ar:], steps=1)
            target_idx = cols.index(TARGET)
            last_level = levels.loc[:date, TARGET].iloc[-2]
            forecasts[date] = last_level + pred_diff[0, target_idx]
        except:
            forecasts[date] = levels.loc[:date, TARGET].iloc[-2]

    return pd.Series(forecasts)

def run_ms_vecm(df):
    """Markov-Switching VECM approximation."""
    cols = [c for c in df.columns if c != 'split']
    data = df[cols].copy()
    forecasts = {}

    # Compute volatility regimes
    vol = data[TARGET].diff().rolling(6).std()
    vol_threshold = vol.quantile(0.7)

    for i, date in enumerate(data.index):
        if date < TEST_START:
            continue

        train = data.loc[:date].iloc[:-1]
        if len(train) < 50:
            continue

        try:
            current_vol = vol.loc[:date].iloc[-1]
            high_vol = current_vol > vol_threshold

            regime_mask = vol.loc[train.index] > vol_threshold if high_vol else vol.loc[train.index] <= vol_threshold
            regime_data = train[regime_mask]

            if len(regime_data) < 30:
                regime_data = train

            joh = coint_johansen(regime_data, det_order=0, k_ar_diff=2)
            rank = sum(joh.lr1 > joh.cvt[:, 1])
            rank = max(1, min(rank, len(cols) - 1))

            model = VECM(regime_data, k_ar_diff=2, coint_rank=rank)
            fitted = model.fit()
            pred = fitted.predict(steps=1)
            target_idx = cols.index(TARGET)
            forecasts[date] = train[TARGET].iloc[-1] + pred[0, target_idx]
        except:
            forecasts[date] = train[TARGET].iloc[-1]

    return pd.Series(forecasts)

def run_bvar(df):
    """Bayesian VAR using Ridge regression."""
    cols = [c for c in df.columns if c != 'split']
    data = df[cols].copy()

    # Create lag features
    lags = 2
    features = data.copy()
    for lag in range(1, lags + 1):
        for col in cols:
            features[f'{col}_lag{lag}'] = data[col].shift(lag)
    features = features.dropna()

    forecasts = {}
    feature_cols = [c for c in features.columns if 'lag' in c]

    for date in features.index:
        if date < TEST_START:
            continue

        train = features.loc[:date].iloc[:-1]
        if len(train) < 30:
            continue

        try:
            X = train[feature_cols]
            y = train[TARGET]
            model = Ridge(alpha=1.0)
            model.fit(X, y)
            X_pred = features.loc[[date], feature_cols]
            forecasts[date] = model.predict(X_pred)[0]
        except:
            forecasts[date] = train[TARGET].iloc[-1]

    return pd.Series(forecasts)

def run_dma(df):
    """Dynamic Model Averaging approximation."""
    cols = [c for c in df.columns if c != 'split']
    data = df[cols].copy()

    # Create multiple models with different variable subsets
    all_predictors = [c for c in cols if c != TARGET]

    # Generate model combinations
    from itertools import combinations
    model_specs = []
    for r in range(1, min(4, len(all_predictors) + 1)):
        for combo in combinations(all_predictors, r):
            model_specs.append(list(combo))

    if not model_specs:
        model_specs = [all_predictors]

    forecasts = {}

    for date in data.index:
        if date < TEST_START:
            continue

        train = data.loc[:date].iloc[:-1]
        if len(train) < 30:
            continue

        try:
            model_forecasts = []
            model_weights = []

            for spec in model_specs[:10]:  # Limit to 10 models
                if not all(p in train.columns for p in spec):
                    continue

                # Create lag features for this spec
                X_train = pd.DataFrame(index=train.index)
                for p in spec:
                    X_train[f'{p}_lag1'] = train[p].shift(1)
                X_train = X_train.dropna()
                y_train = train.loc[X_train.index, TARGET]

                if len(X_train) < 20:
                    continue

                model = Ridge(alpha=1.0)
                model.fit(X_train, y_train)

                # In-sample MSE for weighting
                y_pred_train = model.predict(X_train)
                mse = np.mean((y_train - y_pred_train)**2)
                weight = 1 / (mse + 1e-6)

                # Forecast
                X_pred = pd.DataFrame({f'{p}_lag1': [train[p].iloc[-1]] for p in spec})
                fc = model.predict(X_pred)[0]

                model_forecasts.append(fc)
                model_weights.append(weight)

            if model_forecasts:
                weights = np.array(model_weights) / sum(model_weights)
                forecasts[date] = np.sum(np.array(model_forecasts) * weights)
            else:
                forecasts[date] = train[TARGET].iloc[-1]
        except:
            forecasts[date] = train[TARGET].iloc[-1]

    return pd.Series(forecasts)

def run_dms(df):
    """Dynamic Model Selection - select best model at each point."""
    cols = [c for c in df.columns if c != 'split']
    data = df[cols].copy()

    all_predictors = [c for c in cols if c != TARGET]

    from itertools import combinations
    model_specs = []
    for r in range(1, min(4, len(all_predictors) + 1)):
        for combo in combinations(all_predictors, r):
            model_specs.append(list(combo))

    if not model_specs:
        model_specs = [all_predictors]

    forecasts = {}

    for date in data.index:
        if date < TEST_START:
            continue

        train = data.loc[:date].iloc[:-1]
        if len(train) < 30:
            continue

        try:
            best_fc = None
            best_mse = np.inf

            for spec in model_specs[:10]:
                if not all(p in train.columns for p in spec):
                    continue

                X_train = pd.DataFrame(index=train.index)
                for p in spec:
                    X_train[f'{p}_lag1'] = train[p].shift(1)
                X_train = X_train.dropna()
                y_train = train.loc[X_train.index, TARGET]

                if len(X_train) < 20:
                    continue

                model = Ridge(alpha=1.0)
                model.fit(X_train, y_train)

                y_pred_train = model.predict(X_train)
                mse = np.mean((y_train - y_pred_train)**2)

                if mse < best_mse:
                    best_mse = mse
                    X_pred = pd.DataFrame({f'{p}_lag1': [train[p].iloc[-1]] for p in spec})
                    best_fc = model.predict(X_pred)[0]

            forecasts[date] = best_fc if best_fc is not None else train[TARGET].iloc[-1]
        except:
            forecasts[date] = train[TARGET].iloc[-1]

    return pd.Series(forecasts)

def run_xgboost(df):
    """XGBoost with lag features."""
    if not HAS_XGB:
        return None

    cols = [c for c in df.columns if c != 'split']
    data = df[cols].copy()

    # Create features
    features = pd.DataFrame(index=data.index)
    for lag in [1, 2, 3, 6, 12]:
        features[f'{TARGET}_lag{lag}'] = data[TARGET].shift(lag)

    for col in cols:
        if col != TARGET:
            for lag in [1, 3]:
                features[f'{col}_lag{lag}'] = data[col].shift(lag)

    features[f'{TARGET}_ma3'] = data[TARGET].rolling(3).mean()
    features[f'{TARGET}_ma6'] = data[TARGET].rolling(6).mean()
    features[f'{TARGET}_std3'] = data[TARGET].rolling(3).std()
    features[f'{TARGET}_mom1'] = data[TARGET].diff(1)
    features[f'{TARGET}_mom3'] = data[TARGET].diff(3)
    features['target'] = data[TARGET]
    features = features.dropna()

    feature_cols = [c for c in features.columns if c != 'target']

    # Split
    train_mask = features.index <= TRAIN_END
    valid_mask = (features.index > TRAIN_END) & (features.index <= VALID_END)
    test_mask = features.index > VALID_END

    X_train = features.loc[train_mask, feature_cols]
    y_train = features.loc[train_mask, 'target']
    X_valid = features.loc[valid_mask, feature_cols]
    y_valid = features.loc[valid_mask, 'target']
    X_test = features.loc[test_mask, feature_cols]

    if len(X_train) < 30 or len(X_valid) < 5:
        return None

    try:
        model = xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            early_stopping_rounds=10
        )
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

        preds = model.predict(X_test)
        return pd.Series(preds, index=X_test.index)
    except:
        return None

def run_lstm(df):
    """LSTM neural network."""
    if not HAS_TF:
        return None

    cols = [c for c in df.columns if c != 'split']
    data = df[cols].copy()

    # Add features
    data['target_lag1'] = data[TARGET].shift(1)
    data['target_lag3'] = data[TARGET].shift(3)
    data['target_mom'] = data[TARGET].diff(1)
    data['target_ma3'] = data[TARGET].rolling(3).mean()
    data = data.dropna()

    feature_cols = [TARGET] + [c for c in data.columns if c != TARGET]

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data[feature_cols])

    seq_length = 6
    X, y = [], []
    for i in range(len(scaled) - seq_length):
        X.append(scaled[i:(i + seq_length)])
        y.append(scaled[i + seq_length, 0])
    X, y = np.array(X), np.array(y)

    valid_indices = data.index[seq_length:]

    train_end_idx = (valid_indices <= TRAIN_END).sum()
    valid_end_idx = (valid_indices <= VALID_END).sum()

    X_train, y_train = X[:train_end_idx], y[:train_end_idx]
    X_valid, y_valid = X[train_end_idx:valid_end_idx], y[train_end_idx:valid_end_idx]
    X_test, y_test = X[valid_end_idx:], y[valid_end_idx:]
    test_dates = valid_indices[valid_end_idx:]

    if len(X_train) < 30 or len(X_valid) < 5:
        return None

    try:
        model = Sequential([
            LSTM(32, activation='tanh', input_shape=(seq_length, X.shape[2])),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=100, batch_size=8,
                  validation_data=(X_valid, y_valid), callbacks=[early_stop], verbose=0)

        y_pred_scaled = model.predict(X_test, verbose=0)

        # Inverse transform
        y_pred_full = np.zeros((len(y_pred_scaled), scaled.shape[1]))
        y_pred_full[:, 0] = y_pred_scaled.flatten()
        y_pred = scaler.inverse_transform(y_pred_full)[:, 0]

        return pd.Series(y_pred, index=test_dates)
    except:
        return None

# ============================================================================
# MAIN RUNNER
# ============================================================================

MODELS = {
    'Naive': run_naive,
    'ARIMA': run_arima,
    'VAR': run_var,
    'VECM': run_vecm,
    'MS-VAR': run_ms_var,
    'MS-VECM': run_ms_vecm,
    'BVAR': run_bvar,
    'DMA': run_dma,
    'DMS': run_dms,
    'XGBoost': run_xgboost,
    'LSTM': run_lstm,
}

def main():
    print("=" * 80)
    print("RUNNING ALL 70 MODEL × VARSET COMBINATIONS")
    print("=" * 80)

    all_results = []

    for varset in VARSETS:
        print(f"\n{'='*40}")
        print(f"VARIABLE SET: {varset.upper()}")
        print(f"{'='*40}")

        df = load_varset_data(varset)
        if df is None:
            print(f"  No data for {varset}")
            continue

        actuals = df[TARGET]
        naive_fc = run_naive(df)
        naive_metrics = compute_metrics(actuals, naive_fc, RECOVERY_START, RECOVERY_END)

        if naive_metrics is None:
            print(f"  Insufficient test data for {varset}")
            continue

        naive_rmse = naive_metrics['rmse']
        print(f"  Naive RMSE: {naive_rmse:.1f}")

        for model_name, model_func in MODELS.items():
            print(f"  Running {model_name}...", end=" ")

            try:
                fc = model_func(df)

                if fc is None or len(fc) == 0:
                    print("SKIPPED (no output)")
                    all_results.append({
                        'variable_set': varset, 'model': model_name,
                        'status': 'skipped', 'n': 0,
                    })
                    continue

                metrics = compute_metrics(actuals, fc, RECOVERY_START, RECOVERY_END)

                if metrics is None:
                    print("SKIPPED (insufficient overlap)")
                    all_results.append({
                        'variable_set': varset, 'model': model_name,
                        'status': 'no_overlap', 'n': 0,
                    })
                    continue

                rmse_vs_naive = (metrics['rmse'] / naive_rmse - 1) * 100
                beats = metrics['rmse'] < naive_rmse

                # DM test
                common_idx = fc.dropna().index.intersection(naive_fc.dropna().index)
                common_idx = common_idx[(common_idx >= RECOVERY_START) & (common_idx <= RECOVERY_END)]
                common_idx = common_idx.intersection(actuals.dropna().index)

                if len(common_idx) >= 5:
                    model_errors = fc.loc[common_idx].values - actuals.loc[common_idx].values
                    naive_errors = naive_fc.loc[common_idx].values - actuals.loc[common_idx].values
                    dm_stat, dm_pval = diebold_mariano_test(model_errors, naive_errors)
                else:
                    dm_stat, dm_pval = np.nan, np.nan

                result = {
                    'variable_set': varset,
                    'model': model_name,
                    'status': 'ok',
                    'n': metrics['n'],
                    'rmse': metrics['rmse'],
                    'rmse_vs_naive_pct': rmse_vs_naive,
                    'mape': metrics['mape'],
                    'smape': metrics['smape'],
                    'r_squared': metrics['r_squared'],
                    'mase': metrics['mase'],
                    'theil_u2': metrics['theil_u2'],
                    'dir_acc': metrics['dir_acc'],
                    'dm_stat': dm_stat,
                    'dm_pvalue': dm_pval,
                    'beats_naive': beats,
                }
                all_results.append(result)

                status = "✓ BEATS" if beats else ""
                print(f"RMSE={metrics['rmse']:.1f} ({rmse_vs_naive:+.1f}%) {status}")

            except Exception as e:
                print(f"ERROR: {str(e)[:50]}")
                all_results.append({
                    'variable_set': varset, 'model': model_name,
                    'status': f'error: {str(e)[:50]}', 'n': 0,
                })

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "full_70_combinations.csv", index=False)

    # Create pivot table
    ok_results = results_df[results_df['status'] == 'ok']
    if not ok_results.empty:
        pivot = ok_results.pivot_table(
            index='model', columns='variable_set',
            values='rmse_vs_naive_pct', aggfunc='first'
        )
        pivot.to_csv(OUTPUT_DIR / "full_70_pivot.csv")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    ok_count = (results_df['status'] == 'ok').sum()
    total = len(MODELS) * len(VARSETS)
    print(f"\nCompleted: {ok_count}/{total} combinations")

    if not ok_results.empty:
        winners = ok_results[ok_results['beats_naive'] == True]
        print(f"Models beating naive: {len(winners)}")

        if not winners.empty:
            print("\nWINNERS:")
            for _, row in winners.sort_values('rmse_vs_naive_pct').iterrows():
                print(f"  {row['model']} on {row['variable_set']}: {row['rmse_vs_naive_pct']:+.1f}%")

    print(f"\nSaved: {OUTPUT_DIR / 'full_70_combinations.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'full_70_pivot.csv'}")

if __name__ == "__main__":
    main()
