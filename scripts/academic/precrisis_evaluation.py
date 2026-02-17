"""
Pre-Crisis Model Evaluation
============================
Evaluates models on pre-crisis period (2019) to establish clean baseline.
No crisis effects, no COVID effects.

Splits:
- Train: up to 2016-12
- Validation: 2017-01 to 2018-12
- Test: 2019-01 to 2019-12

Models: XGBoost, MS-VAR, VAR, ARIMA, LSTM, BVAR, VECM, MS-VECM, Naive
Variable Sets: parsimonious, bop, monetary, pca, full
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
from statsmodels.tsa.api import VAR
from scipy import stats

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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "reserves_project" / "academic_deliverables"
FIGURES_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Pre-crisis splits
TRAIN_END = pd.Timestamp("2016-12-01")
VALID_START = pd.Timestamp("2017-01-01")
VALID_END = pd.Timestamp("2018-12-01")
TEST_START = pd.Timestamp("2019-01-01")
TEST_END = pd.Timestamp("2019-12-01")

TARGET = 'gross_reserves_usd_m'
VARSETS = ['parsimonious', 'bop', 'monetary', 'pca', 'full']

# ============================================================================
# Data Loading
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
# Metrics
# ============================================================================

def compute_metrics(actuals, forecast):
    """Compute all metrics."""
    common = actuals.index.intersection(forecast.index)
    if len(common) < 3:
        return None

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
        'n': n, 'rmse': rmse, 'mae': mae, 'mape': mape,
        'r_squared': r_squared, 'dir_acc': dir_acc,
    }

# ============================================================================
# Model Implementations (Pre-Crisis Versions)
# ============================================================================

def run_naive(df):
    """Naive (random walk) forecast."""
    actuals = df[TARGET]
    forecasts = {}
    for date in actuals.index:
        if date < TEST_START or date > TEST_END:
            continue
        prev_idx = actuals.index.get_loc(date) - 1
        if prev_idx >= 0:
            forecasts[date] = actuals.iloc[prev_idx]
    return pd.Series(forecasts)

def run_arima(df):
    """ARIMA model - rolling forecast."""
    series = df[TARGET].copy()
    forecasts = {}

    for date in series.index:
        if date < TEST_START or date > TEST_END:
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
            forecasts[date] = train.iloc[-1]

    return pd.Series(forecasts)

def run_var(df):
    """VAR model - rolling forecast."""
    cols = [c for c in df.columns if c not in ['split']]
    data = df[cols].diff().dropna()
    levels = df[cols].copy()
    forecasts = {}

    for date in data.index:
        if date < TEST_START or date > TEST_END:
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

def run_vecm(df):
    """VECM model - rolling forecast."""
    cols = [c for c in df.columns if c not in ['split']]
    data = df[cols].copy()
    forecasts = {}

    for date in data.index:
        if date < TEST_START or date > TEST_END:
            continue

        train = data.loc[:date].iloc[:-1]
        if len(train) < 50:
            continue

        try:
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

def run_ms_var(df):
    """Markov-Switching VAR approximation."""
    cols = [c for c in df.columns if c not in ['split']]
    data = df[cols].diff().dropna()
    levels = df[cols].copy()
    forecasts = {}

    vol = data[TARGET].rolling(6).std()
    vol_threshold = vol.loc[:TRAIN_END].quantile(0.7)

    for date in data.index:
        if date < TEST_START or date > TEST_END:
            continue

        train_diff = data.loc[:date].iloc[:-1]
        if len(train_diff) < 30:
            continue

        try:
            current_vol = vol.loc[:date].iloc[-1]
            high_vol = current_vol > vol_threshold

            regime_mask = vol.loc[train_diff.index] > vol_threshold if high_vol else vol.loc[train_diff.index] <= vol_threshold
            regime_data = train_diff[regime_mask]

            if len(regime_data) < 20:
                regime_data = train_diff

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
    cols = [c for c in df.columns if c not in ['split']]
    data = df[cols].copy()
    forecasts = {}

    vol = data[TARGET].diff().rolling(6).std()
    vol_threshold = vol.loc[:TRAIN_END].quantile(0.7)

    for date in data.index:
        if date < TEST_START or date > TEST_END:
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
    cols = [c for c in df.columns if c not in ['split']]
    data = df[cols].copy()

    lags = 2
    features = data.copy()
    for lag in range(1, lags + 1):
        for col in cols:
            features[f'{col}_lag{lag}'] = data[col].shift(lag)
    features = features.dropna()

    forecasts = {}
    feature_cols = [c for c in features.columns if 'lag' in c]

    for date in features.index:
        if date < TEST_START or date > TEST_END:
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

def run_xgboost(df):
    """XGBoost with lag features."""
    if not HAS_XGB:
        return None

    cols = [c for c in df.columns if c not in ['split']]
    data = df[cols].copy()

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

    # Pre-crisis splits
    train_mask = features.index <= TRAIN_END
    valid_mask = (features.index > TRAIN_END) & (features.index <= VALID_END)
    test_mask = (features.index >= TEST_START) & (features.index <= TEST_END)

    X_train = features.loc[train_mask, feature_cols]
    y_train = features.loc[train_mask, 'target']
    X_valid = features.loc[valid_mask, feature_cols]
    y_valid = features.loc[valid_mask, 'target']
    X_test = features.loc[test_mask, feature_cols]

    if len(X_train) < 30 or len(X_valid) < 5 or len(X_test) < 3:
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

    cols = [c for c in df.columns if c not in ['split']]
    data = df[cols].copy()

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
    test_start_idx = (valid_indices < TEST_START).sum()
    test_end_idx = (valid_indices <= TEST_END).sum()

    X_train, y_train = X[:train_end_idx], y[:train_end_idx]
    X_valid, y_valid = X[train_end_idx:valid_end_idx], y[train_end_idx:valid_end_idx]
    X_test = X[test_start_idx:test_end_idx]
    test_dates = valid_indices[test_start_idx:test_end_idx]

    if len(X_train) < 30 or len(X_valid) < 5 or len(X_test) < 3:
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

        y_pred_full = np.zeros((len(y_pred_scaled), scaled.shape[1]))
        y_pred_full[:, 0] = y_pred_scaled.flatten()
        y_pred = scaler.inverse_transform(y_pred_full)[:, 0]

        return pd.Series(y_pred, index=test_dates)
    except:
        return None

# ============================================================================
# Main
# ============================================================================

MODELS = {
    'Naive': run_naive,
    'ARIMA': run_arima,
    'VAR': run_var,
    'VECM': run_vecm,
    'MS-VAR': run_ms_var,
    'MS-VECM': run_ms_vecm,
    'BVAR': run_bvar,
    'XGBoost': run_xgboost,
    'LSTM': run_lstm,
}

def main():
    print("=" * 80)
    print("PRE-CRISIS MODEL EVALUATION")
    print("=" * 80)
    print(f"\nSplits:")
    print(f"  Train:      up to {TRAIN_END.strftime('%Y-%m')}")
    print(f"  Validation: {VALID_START.strftime('%Y-%m')} to {VALID_END.strftime('%Y-%m')}")
    print(f"  Test:       {TEST_START.strftime('%Y-%m')} to {TEST_END.strftime('%Y-%m')}")
    print(f"\nModels: {list(MODELS.keys())}")
    print(f"Variable Sets: {VARSETS}")

    all_results = []

    for varset in VARSETS:
        print(f"\n{'='*60}")
        print(f"VARIABLE SET: {varset.upper()}")
        print(f"{'='*60}")

        df = load_varset_data(varset)
        if df is None:
            print(f"  No data for {varset}")
            continue

        # Check data coverage
        if df.index.min() > TRAIN_END or df.index.max() < TEST_END:
            print(f"  Insufficient data coverage: {df.index.min()} to {df.index.max()}")
            continue

        actuals = df[TARGET]
        test_actuals = actuals[(actuals.index >= TEST_START) & (actuals.index <= TEST_END)]
        print(f"  Data: {df.index.min().strftime('%Y-%m')} to {df.index.max().strftime('%Y-%m')}")
        print(f"  Test period actuals: {len(test_actuals)} observations")

        # Get naive benchmark
        naive_fc = run_naive(df)
        naive_metrics = compute_metrics(test_actuals, naive_fc)
        if naive_metrics is None:
            print(f"  Could not compute naive benchmark")
            continue

        naive_rmse = naive_metrics['rmse']
        print(f"  Naive RMSE: {naive_rmse:.2f}")

        for model_name, model_func in MODELS.items():
            print(f"  Running {model_name}...", end=" ")

            try:
                fc = model_func(df)

                if fc is None or len(fc) == 0:
                    print("SKIPPED")
                    all_results.append({
                        'variable_set': varset, 'model': model_name,
                        'status': 'skipped', 'n': 0,
                    })
                    continue

                metrics = compute_metrics(test_actuals, fc)

                if metrics is None:
                    print("NO OVERLAP")
                    all_results.append({
                        'variable_set': varset, 'model': model_name,
                        'status': 'no_overlap', 'n': 0,
                    })
                    continue

                rmse_vs_naive = (metrics['rmse'] / naive_rmse - 1) * 100
                beats = metrics['rmse'] < naive_rmse

                result = {
                    'variable_set': varset,
                    'model': model_name,
                    'status': 'ok',
                    'n': metrics['n'],
                    'rmse': metrics['rmse'],
                    'rmse_vs_naive_pct': rmse_vs_naive,
                    'mape': metrics['mape'],
                    'r_squared': metrics['r_squared'],
                    'dir_acc': metrics['dir_acc'],
                    'beats_naive': beats,
                }
                all_results.append(result)

                status = "✓ BEATS" if beats else ""
                print(f"RMSE={metrics['rmse']:.1f} ({rmse_vs_naive:+.1f}%) {status}")

            except Exception as e:
                print(f"ERROR: {str(e)[:40]}")
                all_results.append({
                    'variable_set': varset, 'model': model_name,
                    'status': f'error', 'n': 0,
                })

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "precrisis_results.csv", index=False)

    # Create pivot
    ok_results = results_df[results_df['status'] == 'ok']
    if not ok_results.empty:
        pivot = ok_results.pivot_table(
            index='model', columns='variable_set',
            values='rmse_vs_naive_pct', aggfunc='first'
        )
        pivot.to_csv(OUTPUT_DIR / "precrisis_pivot.csv")

    # Summary
    print("\n" + "=" * 80)
    print("PRE-CRISIS RESULTS SUMMARY")
    print("=" * 80)

    if not ok_results.empty:
        print(f"\n{'Model':<12} ", end="")
        for vs in VARSETS:
            print(f"{vs[:6]:>10}", end="")
        print()
        print("-" * 65)

        for model in MODELS.keys():
            print(f"{model:<12} ", end="")
            for vs in VARSETS:
                row = ok_results[(ok_results['model'] == model) & (ok_results['variable_set'] == vs)]
                if len(row) > 0:
                    val = row['rmse_vs_naive_pct'].values[0]
                    print(f"{val:>+9.1f}%", end="")
                else:
                    print(f"{'N/A':>10}", end="")
            print()

        # Winners
        winners = ok_results[ok_results['beats_naive'] == True]
        print(f"\n*** {len(winners)} combinations beat naive ***")

        if not winners.empty:
            print("\nWINNERS (sorted by improvement):")
            for _, row in winners.sort_values('rmse_vs_naive_pct').iterrows():
                print(f"  {row['model']} on {row['variable_set']}: {row['rmse_vs_naive_pct']:+.1f}%")

    # Create figures
    create_precrisis_figures(results_df)

    print(f"\nSaved: {OUTPUT_DIR / 'precrisis_results.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'precrisis_pivot.csv'}")

    return results_df

def create_precrisis_figures(results_df):
    """Create publication-quality figures."""
    ok_results = results_df[results_df['status'] == 'ok'].copy()

    if ok_results.empty:
        return

    # Figure 1: Heatmap
    pivot = ok_results.pivot_table(
        index='model', columns='variable_set',
        values='rmse_vs_naive_pct', aggfunc='first'
    )

    # Reorder
    col_order = ['parsimonious', 'bop', 'monetary', 'pca', 'full']
    pivot = pivot[[c for c in col_order if c in pivot.columns]]

    pivot['avg'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('avg')
    pivot = pivot.drop('avg', axis=1)

    fig, ax = plt.subplots(figsize=(12, 8))

    data = pivot.values
    models = pivot.index.tolist()
    varsets = [c.upper()[:6] for c in pivot.columns]

    cmap = plt.cm.RdYlGn_r
    data_clipped = np.clip(data, -50, 100)

    im = ax.imshow(data_clipped, cmap=cmap, aspect='auto', vmin=-50, vmax=100)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('RMSE vs Naive (%)', fontsize=12)

    ax.set_xticks(range(len(varsets)))
    ax.set_xticklabels(varsets, fontsize=11)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=11)

    for i in range(len(models)):
        for j in range(len(varsets)):
            val = data[i, j]
            if np.isnan(val):
                text = 'N/A'
                color = 'gray'
            else:
                text = f'{val:+.1f}%'
                color = 'white' if abs(val) > 30 else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=9, fontweight='bold')

    ax.set_xlabel('Variable Set', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Pre-Crisis Model Performance: RMSE vs Naive (%)\nTest Period: 2019-01 to 2019-12 | Green = Better',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "precrisis_heatmap.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"\nSaved: {FIGURES_DIR / 'precrisis_heatmap.png'}")

    # Figure 2: Detailed table
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')

    ok_results = ok_results.sort_values(['variable_set', 'rmse'])
    headers = ['VarSet', 'Model', 'N', 'RMSE', 'vs Naive', 'MAPE', 'R²', 'Dir%', 'Beats']

    table_data = []
    for _, row in ok_results.iterrows():
        beats = '✓' if row['beats_naive'] else ''
        r2 = f"{row['r_squared']:.3f}" if row['r_squared'] > -10 else '<-10'
        table_data.append([
            row['variable_set'][:6].upper(),
            row['model'],
            int(row['n']),
            f"{row['rmse']:.1f}",
            f"{row['rmse_vs_naive_pct']:+.1f}%",
            f"{row['mape']:.1f}%",
            r2,
            f"{row['dir_acc']:.0f}%",
            beats,
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.4)

    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', weight='bold')

    for i, row_data in enumerate(table_data, 1):
        if row_data[8] == '✓':
            for j in range(len(headers)):
                table[(i, j)].set_facecolor('#C6EFCE')

    plt.title('Pre-Crisis Model Results (Test: 2019)\nTrain: ≤2016-12 | Valid: 2017-2018 | Test: 2019',
              fontsize=14, weight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "precrisis_detailed.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'precrisis_detailed.png'}")

if __name__ == "__main__":
    main()
