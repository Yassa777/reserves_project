"""
Diagnose BVAR's paradox: Good RMSE but bad directional accuracy.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "reserves_project" / "academic_deliverables"
FIGURES_DIR = OUTPUT_DIR / "figures"

TRAIN_END = pd.Timestamp("2016-12-01")
TEST_START = pd.Timestamp("2019-01-01")
TEST_END = pd.Timestamp("2019-12-01")
TARGET = 'gross_reserves_usd_m'


def load_bop_data():
    path = DATA_DIR / "forecast_prep_academic" / "varset_bop" / "vecm_levels.csv"
    df = pd.read_csv(path, parse_dates=['date'])
    df = df.set_index('date').sort_index()
    return df


def run_bvar(df):
    """BVAR using Ridge regression - returns forecasts."""
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

        X = train[feature_cols]
        y = train[TARGET]
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        X_pred = features.loc[[date], feature_cols]
        forecasts[date] = model.predict(X_pred)[0]

    return pd.Series(forecasts)


def run_naive(df):
    """Naive forecast - previous value."""
    actuals = df[TARGET]
    forecasts = {}
    for date in actuals.index:
        if date < TEST_START or date > TEST_END:
            continue
        prev_idx = actuals.index.get_loc(date) - 1
        if prev_idx >= 0:
            forecasts[date] = actuals.iloc[prev_idx]
    return pd.Series(forecasts)


def main():
    df = load_bop_data()
    actuals = df[TARGET]

    bvar_fc = run_bvar(df)
    naive_fc = run_naive(df)

    # Align all series
    common = actuals.index.intersection(bvar_fc.index).intersection(naive_fc.index)
    common = common[(common >= TEST_START) & (common <= TEST_END)]

    actual = actuals.loc[common]
    bvar = bvar_fc.loc[common]
    naive = naive_fc.loc[common]

    print("=" * 70)
    print("BVAR vs Naive: Directional Accuracy Diagnosis")
    print("=" * 70)

    # Monthly changes
    actual_change = actual.diff().dropna()
    bvar_change = bvar.diff().dropna()
    naive_change = naive.diff().dropna()

    # Directions
    actual_dir = np.sign(actual_change)
    bvar_dir = np.sign(bvar_change)

    print("\n### Month-by-Month Analysis ###\n")
    print(f"{'Date':<12} {'Actual':>10} {'BVAR':>10} {'Naive':>10} {'Act Chg':>10} {'BVAR Chg':>10} {'Dir Match':>10}")
    print("-" * 82)

    for i, date in enumerate(actual.index):
        act_val = actual.loc[date]
        bvar_val = bvar.loc[date]
        naive_val = naive.loc[date]

        if i > 0:
            prev_date = actual.index[i-1]
            act_chg = actual.loc[date] - actual.loc[prev_date]
            bvar_chg = bvar.loc[date] - bvar.loc[prev_date]
            dir_match = "✓" if np.sign(act_chg) == np.sign(bvar_chg) else "✗"
            print(f"{date.strftime('%Y-%m'):<12} {act_val:>10.1f} {bvar_val:>10.1f} {naive_val:>10.1f} {act_chg:>+10.1f} {bvar_chg:>+10.1f} {dir_match:>10}")
        else:
            print(f"{date.strftime('%Y-%m'):<12} {act_val:>10.1f} {bvar_val:>10.1f} {naive_val:>10.1f} {'--':>10} {'--':>10} {'--':>10}")

    # Compute metrics
    bvar_rmse = np.sqrt(np.mean((bvar - actual)**2))
    naive_rmse = np.sqrt(np.mean((naive - actual)**2))
    bvar_vs_naive = (bvar_rmse / naive_rmse - 1) * 100

    bvar_dir_acc = np.mean(actual_dir.values == bvar_dir.values) * 100

    print("\n" + "=" * 70)
    print("### Summary Statistics ###")
    print("=" * 70)
    print(f"\nBVAR RMSE:        {bvar_rmse:.1f}")
    print(f"Naive RMSE:       {naive_rmse:.1f}")
    print(f"BVAR vs Naive:    {bvar_vs_naive:+.1f}%")
    print(f"BVAR Dir Acc:     {bvar_dir_acc:.1f}%")

    # The key insight
    print("\n" + "=" * 70)
    print("### Why This Happens ###")
    print("=" * 70)

    # Check forecast smoothness
    actual_volatility = actual_change.std()
    bvar_volatility = bvar_change.std()

    print(f"\nActual month-to-month volatility (std): {actual_volatility:.1f}")
    print(f"BVAR forecast volatility (std):         {bvar_volatility:.1f}")
    print(f"BVAR dampening ratio:                   {bvar_volatility/actual_volatility:.2f}x")

    # Check bias
    print(f"\nActual mean change:   {actual_change.mean():+.1f}")
    print(f"BVAR mean change:     {bvar_change.mean():+.1f}")

    # Count direction mismatches by type
    n_up_actual = (actual_dir > 0).sum()
    n_down_actual = (actual_dir < 0).sum()

    bvar_up_when_down = ((bvar_dir > 0) & (actual_dir < 0)).sum()
    bvar_down_when_up = ((bvar_dir < 0) & (actual_dir > 0)).sum()

    print(f"\nActual direction distribution:")
    print(f"  Up months:   {n_up_actual}")
    print(f"  Down months: {n_down_actual}")

    print(f"\nBVAR direction errors:")
    print(f"  Predicted UP when actual DOWN:   {bvar_up_when_down}")
    print(f"  Predicted DOWN when actual UP:   {bvar_down_when_up}")

    # Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Panel 1: Levels
    ax1 = axes[0]
    ax1.plot(actual.index, actual.values, 'b-o', label='Actual', linewidth=2, markersize=8)
    ax1.plot(bvar.index, bvar.values, 'r--s', label='BVAR Forecast', linewidth=2, markersize=8)
    ax1.plot(naive.index, naive.values, 'g:^', label='Naive Forecast', linewidth=1.5, markersize=6, alpha=0.7)
    ax1.set_ylabel('Reserves (USD M)', fontsize=12)
    ax1.set_title('BVAR Paradox: Good Level Forecasts, Bad Direction\nPre-Crisis Period (2019)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Month-over-month changes
    ax2 = axes[1]
    width = 10
    x = np.arange(len(actual_change))
    ax2.bar(x - width/2, actual_change.values, width, label='Actual Change', color='blue', alpha=0.7)
    ax2.bar(x + width/2, bvar_change.values, width, label='BVAR Predicted Change', color='red', alpha=0.7)
    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels([d.strftime('%Y-%m') for d in actual_change.index], rotation=45, ha='right')
    ax2.set_ylabel('Monthly Change (USD M)', fontsize=12)
    ax2.set_title(f'Month-over-Month Changes | BVAR Volatility = {bvar_volatility/actual_volatility:.0%} of Actual', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Panel 3: Direction match
    ax3 = axes[2]
    matches = (actual_dir.values == bvar_dir.values).astype(int)
    colors = ['green' if m else 'red' for m in matches]
    ax3.bar(range(len(matches)), matches, color=colors, alpha=0.8)
    ax3.axhline(0.5, color='gray', linestyle='--', label='Random Guess')
    ax3.set_xticks(range(len(actual_dir)))
    ax3.set_xticklabels([d.strftime('%Y-%m') for d in actual_dir.index], rotation=45, ha='right')
    ax3.set_ylabel('Direction Match', fontsize=12)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['Wrong', 'Correct'])
    ax3.set_title(f'Direction Accuracy = {bvar_dir_acc:.0f}% | Green = Correct, Red = Wrong', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "bvar_direction_diagnosis.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"\nSaved: {FIGURES_DIR / 'bvar_direction_diagnosis.png'}")

    # THE KEY EXPLANATION
    print("\n" + "=" * 70)
    print("### THE EXPLANATION ###")
    print("=" * 70)
    print("""
BVAR exhibits the "Smooth Forecast Paradox":

1. RIDGE REGULARIZATION DAMPENS FORECASTS
   - Ridge penalty shrinks coefficients toward zero
   - This makes forecasts less responsive to recent changes
   - Forecasts track the LEVEL well but miss SHORT-TERM DIRECTION

2. BVAR FORECASTS ARE SMOOTHER THAN REALITY
   - Actual reserves are volatile (mean-reverting with shocks)
   - BVAR forecasts a gradual trend
   - When reality zigzags, BVAR keeps going straight

3. RMSE REWARDS CLOSENESS TO LEVEL
   - Being close to the right neighborhood = low RMSE
   - RMSE doesn't care about direction

4. DIRECTION ACCURACY PUNISHES LAG
   - If actual goes UP then DOWN, but BVAR predicts UP, UP
   - BVAR is wrong on direction but close on level

This is analogous to a moving average smoothing:
- Good at tracking the level/trend
- Bad at capturing reversals
""")


if __name__ == "__main__":
    main()
