"""
Machine Learning Models for Reserves Forecasting
- XGBoost (Gradient Boosting)
- LSTM (Long Short-Term Memory Neural Network)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Check for required packages
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not installed. Run: pip install xgboost")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("TensorFlow not installed. Run: pip install tensorflow")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "model_verification"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Periods
TRAIN_END = pd.Timestamp("2019-12-01")
VALID_END = pd.Timestamp("2022-12-01")
TEST_START = pd.Timestamp("2023-01-01")
RECOVERY_START = pd.Timestamp("2024-07-01")

def load_data():
    """Load and prepare data for ML models."""
    # Load parsimonious variable set (has longest coverage)
    path = DATA_DIR / "forecast_prep_academic" / "varset_parsimonious" / "vecm_levels.csv"
    df = pd.read_csv(path, parse_dates=['date'])
    df = df.set_index('date')
    df = df.sort_index()

    print(f"Loaded data: {len(df)} observations")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Columns: {list(df.columns)}")

    return df

def create_lag_features(df, target_col, lags=[1, 2, 3, 6, 12]):
    """Create lag features for ML models."""
    features = pd.DataFrame(index=df.index)

    # Lag features for target
    for lag in lags:
        features[f'{target_col}_lag{lag}'] = df[target_col].shift(lag)

    # Lag features for other columns
    for col in df.columns:
        if col != target_col:
            for lag in [1, 3]:
                features[f'{col}_lag{lag}'] = df[col].shift(lag)

    # Rolling statistics
    features[f'{target_col}_ma3'] = df[target_col].rolling(3).mean()
    features[f'{target_col}_ma6'] = df[target_col].rolling(6).mean()
    features[f'{target_col}_std3'] = df[target_col].rolling(3).std()

    # Momentum features
    features[f'{target_col}_mom1'] = df[target_col].diff(1)
    features[f'{target_col}_mom3'] = df[target_col].diff(3)

    # Target (next month's value)
    features['target'] = df[target_col]

    return features.dropna()

def train_xgboost(df, target_col='gross_reserves_usd_m'):
    """Train XGBoost model with rolling window."""
    print("\n" + "="*70)
    print("XGBOOST MODEL")
    print("="*70)

    if not HAS_XGBOOST:
        print("XGBoost not available!")
        return None, None

    # Create features
    features_df = create_lag_features(df, target_col)

    # Split data
    train_mask = features_df.index <= TRAIN_END
    valid_mask = (features_df.index > TRAIN_END) & (features_df.index <= VALID_END)
    test_mask = features_df.index > VALID_END

    feature_cols = [c for c in features_df.columns if c != 'target']

    X_train = features_df.loc[train_mask, feature_cols]
    y_train = features_df.loc[train_mask, 'target']
    X_valid = features_df.loc[valid_mask, feature_cols]
    y_valid = features_df.loc[valid_mask, 'target']
    X_test = features_df.loc[test_mask, feature_cols]
    y_test = features_df.loc[test_mask, 'target']

    print(f"\nTrain: {len(X_train)}, Valid: {len(X_valid)}, Test: {len(X_test)}")

    # Train XGBoost
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=10,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False
    )

    # Rolling predictions for test set
    predictions = []
    actuals = []
    dates = []

    # Use expanding window for predictions
    for i, (idx, row) in enumerate(X_test.iterrows()):
        pred = model.predict(row.values.reshape(1, -1))[0]
        predictions.append(pred)
        actuals.append(y_test.loc[idx])
        dates.append(idx)

    results = pd.DataFrame({
        'date': dates,
        'actual': actuals,
        'forecast': predictions
    }).set_index('date')

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(results['actual'], results['forecast']))
    mae = mean_absolute_error(results['actual'], results['forecast'])

    print(f"\nXGBoost Test Results:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")

    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nTop 5 Features:")
    for _, row in importance.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")

    return results, model

def create_sequences(data, seq_length=12):
    """Create sequences for LSTM."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length, 0])  # Predict target (first column)
    return np.array(X), np.array(y)

def train_lstm(df, target_col='gross_reserves_usd_m'):
    """Train LSTM model with improved architecture."""
    print("\n" + "="*70)
    print("LSTM MODEL")
    print("="*70)

    if not HAS_TF:
        print("TensorFlow not available!")
        return None, None

    # Prepare data with additional engineered features
    data = df.copy()

    # Add lag and momentum features
    data['target_lag1'] = data[target_col].shift(1)
    data['target_lag3'] = data[target_col].shift(3)
    data['target_mom'] = data[target_col].diff(1)
    data['target_ma3'] = data[target_col].rolling(3).mean()

    # Drop NaN rows
    data = data.dropna()

    # Columns to use
    feature_cols = [target_col] + [c for c in data.columns if c != target_col]

    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[feature_cols])

    # Create sequences - shorter sequence for better generalization
    seq_length = 6
    X, y = create_sequences(scaled_data, seq_length)

    # Align indices
    valid_indices = data.index[seq_length:]

    # Split data
    train_end_idx = (valid_indices <= TRAIN_END).sum()
    valid_end_idx = (valid_indices <= VALID_END).sum()

    X_train, y_train = X[:train_end_idx], y[:train_end_idx]
    X_valid, y_valid = X[train_end_idx:valid_end_idx], y[train_end_idx:valid_end_idx]
    X_test, y_test = X[valid_end_idx:], y[valid_end_idx:]
    test_dates = valid_indices[valid_end_idx:]

    print(f"\nTrain: {len(X_train)}, Valid: {len(X_valid)}, Test: {len(X_test)}")
    print(f"Sequence length: {seq_length}")
    print(f"Features: {X.shape[2]}")

    # Build simpler LSTM model (less prone to overfitting)
    model = Sequential([
        LSTM(32, activation='tanh', input_shape=(seq_length, X.shape[2])),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    # Train with early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    print("\nTraining LSTM...")
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=8,
        validation_data=(X_valid, y_valid),
        callbacks=[early_stop],
        verbose=0
    )

    print(f"Training completed in {len(history.history['loss'])} epochs")

    # Predict
    y_pred_scaled = model.predict(X_test, verbose=0)

    # Inverse transform predictions
    y_pred_full = np.zeros((len(y_pred_scaled), scaled_data.shape[1]))
    y_pred_full[:, 0] = y_pred_scaled.flatten()
    y_pred = scaler.inverse_transform(y_pred_full)[:, 0]

    y_actual_full = np.zeros((len(y_test), scaled_data.shape[1]))
    y_actual_full[:, 0] = y_test
    y_actual = scaler.inverse_transform(y_actual_full)[:, 0]

    results = pd.DataFrame({
        'date': test_dates,
        'actual': y_actual,
        'forecast': y_pred
    }).set_index('date')

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(results['actual'], results['forecast']))
    mae = mean_absolute_error(results['actual'], results['forecast'])

    print(f"\nLSTM Test Results:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")

    return results, model

def evaluate_and_compare(xgb_results, lstm_results, df, target_col='gross_reserves_usd_m'):
    """Evaluate and compare all models."""
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)

    # Get actuals and naive forecast
    actuals = df[target_col]
    naive = actuals.shift(1)

    # Define evaluation periods
    periods = {
        'Full Test (2023-01+)': (TEST_START, pd.Timestamp("2025-12-01")),
        'Post-Crisis (2024-07+)': (RECOVERY_START, pd.Timestamp("2025-12-01")),
    }

    for period_name, (start, end) in periods.items():
        print(f"\n{period_name}:")
        print("-" * 50)

        # Filter to period
        period_actuals = actuals[(actuals.index >= start) & (actuals.index <= end)]
        period_naive = naive[(naive.index >= start) & (naive.index <= end)]

        # Naive metrics
        naive_common = period_actuals.index.intersection(period_naive.dropna().index)
        naive_rmse = np.sqrt(mean_squared_error(
            period_actuals.loc[naive_common],
            period_naive.loc[naive_common]
        ))

        results = [{'model': 'Naive', 'rmse': naive_rmse, 'n': len(naive_common)}]

        # XGBoost metrics
        if xgb_results is not None:
            xgb_period = xgb_results[(xgb_results.index >= start) & (xgb_results.index <= end)]
            if len(xgb_period) >= 3:
                xgb_rmse = np.sqrt(mean_squared_error(xgb_period['actual'], xgb_period['forecast']))
                results.append({'model': 'XGBoost', 'rmse': xgb_rmse, 'n': len(xgb_period)})

        # LSTM metrics
        if lstm_results is not None:
            lstm_period = lstm_results[(lstm_results.index >= start) & (lstm_results.index <= end)]
            if len(lstm_period) >= 3:
                lstm_rmse = np.sqrt(mean_squared_error(lstm_period['actual'], lstm_period['forecast']))
                results.append({'model': 'LSTM', 'rmse': lstm_rmse, 'n': len(lstm_period)})

        # Display
        results_df = pd.DataFrame(results).sort_values('rmse')

        print(f"{'Model':<15} {'N':>5} {'RMSE':>12} {'vs Naive':>12} {'Beats?':>8}")
        print("-" * 55)

        for _, row in results_df.iterrows():
            vs_naive = (row['rmse'] / naive_rmse - 1) * 100
            beats = "✓ YES" if row['rmse'] < naive_rmse else ""
            print(f"{row['model']:<15} {row['n']:>5} {row['rmse']:>12.2f} {vs_naive:>+11.1f}% {beats:>8}")

    return results_df

def plot_results(xgb_results, lstm_results, df, target_col='gross_reserves_usd_m'):
    """Plot model predictions."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    actuals = df[target_col]
    naive = actuals.shift(1)

    # Full test period
    ax1 = axes[0]
    test_actuals = actuals[actuals.index >= TEST_START]
    test_naive = naive[naive.index >= TEST_START]

    ax1.plot(test_actuals.index, test_actuals.values, 'k-', linewidth=2, label='Actual')
    ax1.plot(test_naive.index, test_naive.values, 'r--', linewidth=1.5, alpha=0.7, label='Naive')

    if xgb_results is not None:
        xgb_test = xgb_results[xgb_results.index >= TEST_START]
        ax1.plot(xgb_test.index, xgb_test['forecast'], 'b--', linewidth=1.5, alpha=0.8, label='XGBoost')

    if lstm_results is not None:
        lstm_test = lstm_results[lstm_results.index >= TEST_START]
        ax1.plot(lstm_test.index, lstm_test['forecast'], 'g--', linewidth=1.5, alpha=0.8, label='LSTM')

    ax1.set_ylabel('Reserves (USD million)')
    ax1.set_title('Full Test Period: Actual vs ML Model Forecasts')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Recovery period zoom
    ax2 = axes[1]
    recovery_actuals = actuals[actuals.index >= RECOVERY_START]
    recovery_naive = naive[naive.index >= RECOVERY_START]

    ax2.plot(recovery_actuals.index, recovery_actuals.values, 'k-', linewidth=2, label='Actual')
    ax2.plot(recovery_naive.index, recovery_naive.values, 'r--', linewidth=1.5, alpha=0.7, label='Naive')

    if xgb_results is not None:
        xgb_recovery = xgb_results[xgb_results.index >= RECOVERY_START]
        if len(xgb_recovery) > 0:
            ax2.plot(xgb_recovery.index, xgb_recovery['forecast'], 'b--', linewidth=1.5, alpha=0.8, label='XGBoost')

    if lstm_results is not None:
        lstm_recovery = lstm_results[lstm_results.index >= RECOVERY_START]
        if len(lstm_recovery) > 0:
            ax2.plot(lstm_recovery.index, lstm_recovery['forecast'], 'g--', linewidth=1.5, alpha=0.8, label='LSTM')

    ax2.set_ylabel('Reserves (USD million)')
    ax2.set_xlabel('Date')
    ax2.set_title('Post-Crisis Recovery Period (2024-07+)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ml_models_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUTPUT_DIR / 'ml_models_comparison.png'}")

def main():
    print("="*70)
    print("MACHINE LEARNING MODELS FOR RESERVES FORECASTING")
    print("="*70)

    # Load data
    df = load_data()

    # Train XGBoost
    xgb_results, xgb_model = train_xgboost(df)

    # Train LSTM
    lstm_results, lstm_model = train_lstm(df)

    # Compare models
    if xgb_results is not None or lstm_results is not None:
        evaluate_and_compare(xgb_results, lstm_results, df)
        plot_results(xgb_results, lstm_results, df)

        # Save results
        if xgb_results is not None:
            xgb_results.to_csv(OUTPUT_DIR / "xgboost_forecasts.csv")
        if lstm_results is not None:
            lstm_results.to_csv(OUTPUT_DIR / "lstm_forecasts.csv")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

if __name__ == "__main__":
    main()
