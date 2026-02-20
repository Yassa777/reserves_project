"""
Differenced Target Evaluation
==============================
Compare forecasting changes in reserves vs levels directly.

Models: XGBoost, BVAR
Variable Sets: parsimonious, bop, monetary, pca, full
Evaluation Periods:
  - Pre-Crisis: Train ≤2016-12, Valid 2017-2018, Test 2019
  - Post-Crisis: Full sample, Test 2024-07+

Comparison:
  1. Direct level forecasting (original approach)
  2. Differenced forecasting (predict Δreserves, cumulate to levels)

Metrics: RMSE, MAPE, Directional Accuracy
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not available")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "reserves_project" / "academic_deliverables"
FIGURES_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

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
# Feature Engineering
# ============================================================================

def create_level_features(df, target_col):
    """Create features for level forecasting (original approach)."""
    cols = [c for c in df.columns if c not in ['split']]
    data = df[cols].copy()

    features = pd.DataFrame(index=data.index)

    # Target lags - IMPORTANT: lag must be >= 1 to avoid leakage
    for lag in [1, 2, 3, 6, 12]:
        if lag < len(data):
            features[f'{target_col}_lag{lag}'] = data[target_col].shift(lag)

    # Predictor lags - also lagged to avoid leakage
    for col in cols:
        if col != target_col:
            for lag in [1, 3]:
                if lag < len(data):
                    features[f'{col}_lag{lag}'] = data[col].shift(lag)

    # Rolling statistics - use shift(1) to avoid leakage
    features[f'{target_col}_ma3'] = data[target_col].rolling(3).mean().shift(1)
    features[f'{target_col}_ma6'] = data[target_col].rolling(6).mean().shift(1)
    features[f'{target_col}_std3'] = data[target_col].rolling(3).std().shift(1)
    features[f'{target_col}_mom1'] = data[target_col].diff(1).shift(1)
    features[f'{target_col}_mom3'] = data[target_col].diff(3).shift(1)

    features['target'] = data[target_col]
    features = features.dropna()

    return features


def create_diff_features(df, target_col):
    """Create features for differenced forecasting."""
    cols = [c for c in df.columns if c not in ['split']]
    data = df[cols].copy()

    # Differenced target
    diff_target = data[target_col].diff()

    features = pd.DataFrame(index=data.index)

    # Lagged changes (for target) - lag >= 1 to avoid leakage
    for lag in [1, 2, 3, 6, 12]:
        if lag < len(data):
            features[f'd_{target_col}_lag{lag}'] = diff_target.shift(lag)

    # Level lags (important context)
    for lag in [1, 2]:
        if lag < len(data):
            features[f'{target_col}_level_lag{lag}'] = data[target_col].shift(lag)

    # Predictor changes and levels - lagged to avoid leakage
    for col in cols:
        if col != target_col:
            diff_col = data[col].diff()
            for lag in [1, 3]:
                if lag < len(data):
                    features[f'd_{col}_lag{lag}'] = diff_col.shift(lag)
                    features[f'{col}_lag{lag}'] = data[col].shift(lag)

    # Rolling statistics on changes - shift(1) to avoid leakage
    features[f'd_{target_col}_ma3'] = diff_target.rolling(3).mean().shift(1)
    features[f'd_{target_col}_ma6'] = diff_target.rolling(6).mean().shift(1)
    features[f'd_{target_col}_std3'] = diff_target.rolling(3).std().shift(1)

    # Momentum indicators - these are lagged changes, shift to avoid leakage
    features[f'{target_col}_mom3'] = data[target_col].diff(3).shift(1)
    features[f'{target_col}_mom6'] = data[target_col].diff(6).shift(1)

    # Target is the CHANGE
    features['target'] = diff_target
    features['level'] = data[target_col]  # Keep levels for back-conversion
    features = features.dropna()

    return features

# ============================================================================
# Model Training
# ============================================================================

def train_xgboost_level(features, train_end, valid_end, test_start, test_end):
    """Train XGBoost on levels, return test forecasts."""
    if not HAS_XGB:
        return None, None

    feature_cols = [c for c in features.columns if c not in ['target', 'level']]

    train_mask = features.index <= train_end
    valid_mask = (features.index > train_end) & (features.index <= valid_end)
    test_mask = (features.index >= test_start) & (features.index <= test_end)

    X_train = features.loc[train_mask, feature_cols]
    y_train = features.loc[train_mask, 'target']
    X_valid = features.loc[valid_mask, feature_cols]
    y_valid = features.loc[valid_mask, 'target']
    X_test = features.loc[test_mask, feature_cols]

    if len(X_train) < 20 or len(X_test) == 0:
        return None, None

    model = xgb.XGBRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0
    )

    if len(X_valid) > 0:
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
                  verbose=False)
    else:
        model.fit(X_train, y_train)

    forecasts = pd.Series(model.predict(X_test), index=X_test.index)
    actuals = features.loc[test_mask, 'target']

    return forecasts, actuals


def train_bvar_level(features, train_end, valid_end, test_start, test_end):
    """Train BVAR (Ridge) on levels using rolling forecasts."""
    feature_cols = [c for c in features.columns if c not in ['target', 'level']]

    forecasts = {}
    actuals = {}

    for date in features.index:
        if date < test_start or date > test_end:
            continue

        # Train on all data strictly before this date
        train_data = features[features.index < date]
        if len(train_data) < 30:
            continue

        X_train = train_data[feature_cols]
        y_train = train_data['target']

        model = Ridge(alpha=10.0)
        model.fit(X_train, y_train)

        # Predict for current date
        X_pred = features.loc[[date], feature_cols]
        forecasts[date] = model.predict(X_pred)[0]
        actuals[date] = features.loc[date, 'target']

    if len(forecasts) < 3:
        return None, None

    return pd.Series(forecasts), pd.Series(actuals)


def train_xgboost_diff(features, train_end, valid_end, test_start, test_end):
    """Train XGBoost on differences, return test forecasts."""
    if not HAS_XGB:
        return None, None, None

    feature_cols = [c for c in features.columns if c not in ['target', 'level']]

    train_mask = features.index <= train_end
    valid_mask = (features.index > train_end) & (features.index <= valid_end)
    test_mask = (features.index >= test_start) & (features.index <= test_end)

    X_train = features.loc[train_mask, feature_cols]
    y_train = features.loc[train_mask, 'target']
    X_valid = features.loc[valid_mask, feature_cols]
    y_valid = features.loc[valid_mask, 'target']
    X_test = features.loc[test_mask, feature_cols]

    if len(X_train) < 20 or len(X_test) == 0:
        return None, None, None

    model = xgb.XGBRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0
    )

    if len(X_valid) > 0:
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
                  verbose=False)
    else:
        model.fit(X_train, y_train)

    diff_forecasts = pd.Series(model.predict(X_test), index=X_test.index)
    diff_actuals = features.loc[test_mask, 'target']
    levels = features.loc[test_mask, 'level']

    return diff_forecasts, diff_actuals, levels


def train_bvar_diff(features, train_end, valid_end, test_start, test_end):
    """Train BVAR (Ridge) on differences using rolling forecasts."""
    feature_cols = [c for c in features.columns if c not in ['target', 'level']]

    diff_forecasts = {}
    diff_actuals = {}
    levels = {}

    for date in features.index:
        if date < test_start or date > test_end:
            continue

        # Train on all data strictly before this date
        train_data = features[features.index < date]
        if len(train_data) < 30:
            continue

        X_train = train_data[feature_cols]
        y_train = train_data['target']

        model = Ridge(alpha=10.0)
        model.fit(X_train, y_train)

        # Predict for current date
        X_pred = features.loc[[date], feature_cols]
        diff_forecasts[date] = model.predict(X_pred)[0]
        diff_actuals[date] = features.loc[date, 'target']
        levels[date] = features.loc[date, 'level']

    if len(diff_forecasts) < 3:
        return None, None, None

    return pd.Series(diff_forecasts), pd.Series(diff_actuals), pd.Series(levels)


def convert_diff_to_levels(diff_forecasts, features, test_start):
    """Convert differenced forecasts to levels by cumulation."""
    # Get the last known level before test period
    pre_test = features[features.index < test_start]
    if len(pre_test) == 0:
        return None

    last_level = pre_test['level'].iloc[-1]

    # Cumulate: level_t = level_{t-1} + diff_forecast_t
    level_forecasts = []
    current_level = last_level

    for date in diff_forecasts.index:
        current_level = current_level + diff_forecasts.loc[date]
        level_forecasts.append(current_level)

    return pd.Series(level_forecasts, index=diff_forecasts.index)

# ============================================================================
# Metrics
# ============================================================================

def compute_metrics(actuals, forecasts):
    """Compute RMSE, MAPE, and directional accuracy."""
    common = actuals.index.intersection(forecasts.index)
    if len(common) < 3:
        return None

    a = actuals.loc[common]
    f = forecasts.loc[common]

    rmse = np.sqrt(np.mean((f - a)**2))
    mape = np.mean(np.abs((f - a) / a)) * 100

    # Directional accuracy (on changes)
    actual_dir = np.sign(np.diff(a.values))
    forecast_dir = np.sign(np.diff(f.values))
    dir_acc = np.mean(actual_dir == forecast_dir) * 100 if len(actual_dir) > 0 else np.nan

    return {'rmse': rmse, 'mape': mape, 'dir_acc': dir_acc, 'n': len(common)}


def compute_diff_dir_accuracy(diff_actuals, diff_forecasts):
    """Compute directional accuracy directly on differences."""
    common = diff_actuals.index.intersection(diff_forecasts.index)
    if len(common) < 1:
        return np.nan

    a = diff_actuals.loc[common]
    f = diff_forecasts.loc[common]

    actual_dir = np.sign(a.values)
    forecast_dir = np.sign(f.values)

    return np.mean(actual_dir == forecast_dir) * 100

# ============================================================================
# Main Evaluation
# ============================================================================

def evaluate_period(period_name, train_end, valid_end, test_start, test_end):
    """Run evaluation for a specific period."""
    print(f"\n{'='*70}")
    print(f"Evaluating: {period_name}")
    print(f"Train ≤ {train_end.strftime('%Y-%m')}, Valid {valid_end.strftime('%Y-%m')}, Test {test_start.strftime('%Y-%m')} to {test_end.strftime('%Y-%m')}")
    print('='*70)

    results = []

    for varset in VARSETS:
        df = load_varset_data(varset)
        if df is None:
            print(f"  {varset}: data not found")
            continue

        # Create features
        level_features = create_level_features(df, TARGET)
        diff_features = create_diff_features(df, TARGET)

        # Get actual levels for naive benchmark
        test_mask = (df.index >= test_start) & (df.index <= test_end)
        actual_levels = df.loc[test_mask, TARGET]

        if len(actual_levels) < 3:
            print(f"  {varset}: insufficient test data")
            continue

        # Naive forecast (random walk)
        naive_fc = actual_levels.shift(1).dropna()
        naive_actual = actual_levels.loc[naive_fc.index]
        naive_metrics = compute_metrics(naive_actual, naive_fc)

        if naive_metrics is None:
            continue

        naive_rmse = naive_metrics['rmse']

        # --- XGBoost Level ---
        xgb_level_fc, xgb_level_actual = train_xgboost_level(
            level_features, train_end, valid_end, test_start, test_end
        )

        if xgb_level_fc is not None:
            xgb_level_metrics = compute_metrics(xgb_level_actual, xgb_level_fc)
            if xgb_level_metrics:
                results.append({
                    'period': period_name,
                    'varset': varset,
                    'model': 'XGBoost',
                    'target': 'Level',
                    'rmse': xgb_level_metrics['rmse'],
                    'vs_naive': (xgb_level_metrics['rmse'] / naive_rmse - 1) * 100,
                    'mape': xgb_level_metrics['mape'],
                    'dir_acc': xgb_level_metrics['dir_acc'],
                    'n': xgb_level_metrics['n'],
                })

        # --- XGBoost Diff ---
        xgb_diff_fc, xgb_diff_actual, xgb_levels = train_xgboost_diff(
            diff_features, train_end, valid_end, test_start, test_end
        )

        if xgb_diff_fc is not None:
            # Convert to levels
            xgb_diff_level_fc = convert_diff_to_levels(xgb_diff_fc, diff_features, test_start)
            if xgb_diff_level_fc is not None:
                xgb_diff_metrics = compute_metrics(xgb_levels, xgb_diff_level_fc)
                diff_dir_acc = compute_diff_dir_accuracy(xgb_diff_actual, xgb_diff_fc)

                if xgb_diff_metrics:
                    results.append({
                        'period': period_name,
                        'varset': varset,
                        'model': 'XGBoost',
                        'target': 'Diff→Level',
                        'rmse': xgb_diff_metrics['rmse'],
                        'vs_naive': (xgb_diff_metrics['rmse'] / naive_rmse - 1) * 100,
                        'mape': xgb_diff_metrics['mape'],
                        'dir_acc': diff_dir_acc,  # Use direct diff accuracy
                        'n': xgb_diff_metrics['n'],
                    })

        # --- BVAR Level ---
        bvar_level_fc, bvar_level_actual = train_bvar_level(
            level_features, train_end, valid_end, test_start, test_end
        )

        if bvar_level_fc is not None:
            bvar_level_metrics = compute_metrics(bvar_level_actual, bvar_level_fc)
            if bvar_level_metrics:
                results.append({
                    'period': period_name,
                    'varset': varset,
                    'model': 'BVAR',
                    'target': 'Level',
                    'rmse': bvar_level_metrics['rmse'],
                    'vs_naive': (bvar_level_metrics['rmse'] / naive_rmse - 1) * 100,
                    'mape': bvar_level_metrics['mape'],
                    'dir_acc': bvar_level_metrics['dir_acc'],
                    'n': bvar_level_metrics['n'],
                })

        # --- BVAR Diff ---
        bvar_diff_fc, bvar_diff_actual, bvar_levels = train_bvar_diff(
            diff_features, train_end, valid_end, test_start, test_end
        )

        if bvar_diff_fc is not None:
            # Convert to levels
            bvar_diff_level_fc = convert_diff_to_levels(bvar_diff_fc, diff_features, test_start)
            if bvar_diff_level_fc is not None:
                bvar_diff_metrics = compute_metrics(bvar_levels, bvar_diff_level_fc)
                diff_dir_acc = compute_diff_dir_accuracy(bvar_diff_actual, bvar_diff_fc)

                if bvar_diff_metrics:
                    results.append({
                        'period': period_name,
                        'varset': varset,
                        'model': 'BVAR',
                        'target': 'Diff→Level',
                        'rmse': bvar_diff_metrics['rmse'],
                        'vs_naive': (bvar_diff_metrics['rmse'] / naive_rmse - 1) * 100,
                        'mape': bvar_diff_metrics['mape'],
                        'dir_acc': diff_dir_acc,  # Use direct diff accuracy
                        'n': bvar_diff_metrics['n'],
                    })

        # Add naive baseline
        results.append({
            'period': period_name,
            'varset': varset,
            'model': 'Naive',
            'target': 'Level',
            'rmse': naive_rmse,
            'vs_naive': 0.0,
            'mape': naive_metrics['mape'],
            'dir_acc': naive_metrics['dir_acc'],
            'n': naive_metrics['n'],
        })

    return pd.DataFrame(results)


def create_comparison_figure(df, period_name, filename):
    """Create academic comparison figure."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Panel 1: RMSE vs Naive by model and target type
    ax1 = axes[0]

    pivot = df.pivot_table(
        values='vs_naive',
        index='varset',
        columns=['model', 'target'],
        aggfunc='mean'
    )

    x = np.arange(len(pivot.index))
    width = 0.15

    colors = {
        ('XGBoost', 'Level'): '#1f77b4',
        ('XGBoost', 'Diff→Level'): '#aec7e8',
        ('BVAR', 'Level'): '#ff7f0e',
        ('BVAR', 'Diff→Level'): '#ffbb78',
        ('Naive', 'Level'): '#7f7f7f',
    }

    for i, col in enumerate(pivot.columns):
        offset = (i - len(pivot.columns)/2 + 0.5) * width
        color = colors.get(col, '#333333')
        label = f"{col[0]} ({col[1]})"
        bars = ax1.bar(x + offset, pivot[col].values, width, label=label, color=color, alpha=0.85)

    ax1.axhline(0, color='red', linestyle='--', linewidth=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels([v.upper() for v in pivot.index], fontsize=11)
    ax1.set_ylabel('RMSE vs Naive (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Level Forecast Accuracy\n{period_name}', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # Panel 2: Directional Accuracy
    ax2 = axes[1]

    pivot_dir = df.pivot_table(
        values='dir_acc',
        index='varset',
        columns=['model', 'target'],
        aggfunc='mean'
    )

    for i, col in enumerate(pivot_dir.columns):
        offset = (i - len(pivot_dir.columns)/2 + 0.5) * width
        color = colors.get(col, '#333333')
        label = f"{col[0]} ({col[1]})"
        bars = ax2.bar(x + offset, pivot_dir[col].values, width, label=label, color=color, alpha=0.85)

    ax2.axhline(50, color='gray', linestyle='--', linewidth=2, label='Random (50%)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([v.upper() for v in pivot_dir.index], fontsize=11)
    ax2.set_ylabel('Directional Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Direction of Change Accuracy\n{period_name}', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {FIGURES_DIR / filename}")


def create_summary_table(df, period_name, filename):
    """Create detailed table figure."""
    fig, ax = plt.subplots(figsize=(16, max(8, len(df) * 0.4)))
    ax.axis('off')

    # Prepare table data
    table_df = df.sort_values(['varset', 'model', 'target'])

    headers = ['VarSet', 'Model', 'Target', 'N', 'RMSE', 'vs Naive', 'MAPE', 'Dir Acc', 'Beats']

    table_data = []
    for _, row in table_df.iterrows():
        beats = '✓' if row['vs_naive'] < 0 else ''
        table_data.append([
            row['varset'].upper(),
            row['model'],
            row['target'],
            int(row['n']),
            f"{row['rmse']:.1f}",
            f"{row['vs_naive']:+.1f}%",
            f"{row['mape']:.1f}%",
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
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', weight='bold')

    # Highlight winners and diff models
    for i, row_data in enumerate(table_data, 1):
        is_diff = 'Diff' in row_data[2]
        beats_naive = row_data[8] == '✓'

        if beats_naive and is_diff:
            for j in range(len(headers)):
                table[(i, j)].set_facecolor('#C6EFCE')  # Green - diff beats naive
        elif beats_naive:
            for j in range(len(headers)):
                table[(i, j)].set_facecolor('#DDEBF7')  # Light blue - level beats naive
        elif is_diff:
            for j in range(len(headers)):
                table[(i, j)].set_facecolor('#FFF2CC')  # Yellow - diff model

    plt.title(f'Level vs Differenced Forecasting - {period_name}\n'
              f'Green=Diff beats naive | Blue=Level beats naive | Yellow=Diff model',
              fontsize=13, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {FIGURES_DIR / filename}")


def create_improvement_summary(precrisis_df, postcrisis_df):
    """Create summary comparing level vs diff improvements."""

    def compute_improvement(df, model):
        level = df[(df['model'] == model) & (df['target'] == 'Level')]
        diff = df[(df['model'] == model) & (df['target'] == 'Diff→Level')]

        if len(level) == 0 or len(diff) == 0:
            return None

        merged = level.merge(diff, on='varset', suffixes=('_level', '_diff'))

        return {
            'rmse_improvement': (merged['vs_naive_level'] - merged['vs_naive_diff']).mean(),
            'dir_improvement': (merged['dir_acc_diff'] - merged['dir_acc_level']).mean(),
        }

    print("\n" + "="*70)
    print("LEVEL vs DIFFERENCED FORECASTING: SUMMARY")
    print("="*70)

    for period_name, df in [('Pre-Crisis', precrisis_df), ('Post-Crisis', postcrisis_df)]:
        print(f"\n### {period_name} ###")

        for model in ['XGBoost', 'BVAR']:
            imp = compute_improvement(df, model)
            if imp:
                print(f"\n{model}:")
                print(f"  RMSE improvement (Level→Diff): {imp['rmse_improvement']:+.1f}pp")
                print(f"  Dir Acc improvement:           {imp['dir_improvement']:+.1f}pp")


def main():
    # Pre-Crisis Evaluation
    precrisis_df = evaluate_period(
        "Pre-Crisis (2019)",
        train_end=pd.Timestamp("2016-12-01"),
        valid_end=pd.Timestamp("2018-12-01"),
        test_start=pd.Timestamp("2019-01-01"),
        test_end=pd.Timestamp("2019-12-01"),
    )

    # Post-Crisis Evaluation
    postcrisis_df = evaluate_period(
        "Post-Crisis (2024-07+)",
        train_end=pd.Timestamp("2022-06-01"),
        valid_end=pd.Timestamp("2024-06-01"),
        test_start=pd.Timestamp("2024-07-01"),
        test_end=pd.Timestamp("2025-12-01"),
    )

    # Save results
    precrisis_df.to_csv(OUTPUT_DIR / "diff_vs_level_precrisis.csv", index=False)
    postcrisis_df.to_csv(OUTPUT_DIR / "diff_vs_level_postcrisis.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'diff_vs_level_precrisis.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'diff_vs_level_postcrisis.csv'}")

    # Create figures
    create_comparison_figure(precrisis_df, "Pre-Crisis (2019)", "diff_vs_level_precrisis.png")
    create_comparison_figure(postcrisis_df, "Post-Crisis (2024-07+)", "diff_vs_level_postcrisis.png")

    create_summary_table(precrisis_df, "Pre-Crisis (2019)", "diff_vs_level_precrisis_table.png")
    create_summary_table(postcrisis_df, "Post-Crisis (2024-07+)", "diff_vs_level_postcrisis_table.png")

    # Print summary
    create_improvement_summary(precrisis_df, postcrisis_df)

    # Print tables
    print("\n" + "="*70)
    print("PRE-CRISIS RESULTS")
    print("="*70)
    print(precrisis_df.to_string(index=False))

    print("\n" + "="*70)
    print("POST-CRISIS RESULTS")
    print("="*70)
    print(postcrisis_df.to_string(index=False))


if __name__ == "__main__":
    main()
