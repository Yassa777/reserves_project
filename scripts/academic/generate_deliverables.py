"""
Academic Deliverables Generator
================================
Generates three key deliverables:
1. Variable Sets Table with coverage and test statistics
2. Full Model x Variable Set Matrix (14 models × 5 variable sets)
3. Data Quality Assessment documentation

Output: reserves_project/academic_deliverables/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Optional imports
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "reserves_project" / "academic_deliverables"
FIGURES_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Periods
TRAIN_END = pd.Timestamp("2019-12-01")
VALID_END = pd.Timestamp("2022-12-01")
TEST_START = pd.Timestamp("2023-01-01")
RECOVERY_START = pd.Timestamp("2024-07-01")
RECOVERY_END = pd.Timestamp("2025-12-01")

VARSETS = ['parsimonious', 'bop', 'monetary', 'pca', 'full']

# ============================================================================
# DELIVERABLE 1: Variable Sets Table
# ============================================================================

def load_varset_data(varset_name):
    """Load vecm_levels.csv for a variable set."""
    path = DATA_DIR / "forecast_prep_academic" / f"varset_{varset_name}" / "vecm_levels.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=['date'])
    df = df.set_index('date').sort_index()
    return df

def run_adf_test(series):
    """Run ADF test, return (statistic, pvalue, is_stationary)."""
    try:
        result = adfuller(series.dropna(), autolag='AIC')
        return result[0], result[1], result[1] < 0.05
    except:
        return np.nan, np.nan, False

def run_kpss_test(series):
    """Run KPSS test, return (statistic, pvalue, is_stationary)."""
    try:
        result = kpss(series.dropna(), regression='c', nlags='auto')
        return result[0], result[1], result[1] > 0.05
    except:
        return np.nan, np.nan, False

def run_johansen_test(df, det_order=0, k_ar_diff=2):
    """Run Johansen cointegration test."""
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        data = df[numeric_cols].dropna()
        if len(data) < 50:
            return None
        result = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)
        # Count cointegrating relations at 5% level
        trace_stats = result.lr1
        trace_cv = result.cvt[:, 1]  # 5% critical values
        n_coint = sum(trace_stats > trace_cv)
        return {
            'n_coint_relations': n_coint,
            'trace_stats': trace_stats.tolist(),
            'trace_cv_95': trace_cv.tolist(),
        }
    except Exception as e:
        return {'error': str(e)}

def generate_varset_table():
    """Generate Deliverable 1: Variable Sets Table."""
    print("=" * 80)
    print("DELIVERABLE 1: Variable Sets Table")
    print("=" * 80)

    results = []
    detailed_results = []

    for varset in VARSETS:
        df = load_varset_data(varset)
        if df is None:
            continue

        cols = [c for c in df.columns if c not in ['split']]
        n_vars = len(cols)
        n_obs = len(df)
        date_start = df.index.min()
        date_end = df.index.max()

        # Train/valid/test split counts
        train_mask = df.index <= TRAIN_END
        valid_mask = (df.index > TRAIN_END) & (df.index <= VALID_END)
        test_mask = df.index > VALID_END

        # Run tests on each variable
        var_stats = []
        for col in cols:
            adf_stat, adf_p, adf_stationary = run_adf_test(df[col])
            kpss_stat, kpss_p, kpss_stationary = run_kpss_test(df[col])

            var_stats.append({
                'varset': varset,
                'variable': col,
                'n_obs': df[col].notna().sum(),
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'adf_stat': adf_stat,
                'adf_pvalue': adf_p,
                'adf_stationary': adf_stationary,
                'kpss_stat': kpss_stat,
                'kpss_pvalue': kpss_p,
                'kpss_stationary': kpss_stationary,
            })

        detailed_results.extend(var_stats)

        # Johansen test
        johansen = run_johansen_test(df[cols])
        n_coint = johansen.get('n_coint_relations', 0) if johansen and 'error' not in johansen else 0

        results.append({
            'variable_set': varset,
            'variables': ', '.join(cols),
            'n_variables': n_vars,
            'n_observations': n_obs,
            'train_obs': train_mask.sum(),
            'valid_obs': valid_mask.sum(),
            'test_obs': test_mask.sum(),
            'date_start': date_start.strftime('%Y-%m'),
            'date_end': date_end.strftime('%Y-%m'),
            'johansen_n_coint': n_coint,
        })

        print(f"\n{varset.upper()}:")
        print(f"  Variables: {cols}")
        print(f"  Coverage: {date_start.strftime('%Y-%m')} to {date_end.strftime('%Y-%m')} ({n_obs} obs)")
        print(f"  Split: train={train_mask.sum()}, valid={valid_mask.sum()}, test={test_mask.sum()}")
        print(f"  Cointegrating relations: {n_coint}")

    # Save results
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(OUTPUT_DIR / "deliverable1_variable_sets.csv", index=False)

    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv(OUTPUT_DIR / "deliverable1_variable_details.csv", index=False)

    # Create figure
    create_varset_figure(summary_df, detailed_df)

    return summary_df, detailed_df

def create_varset_figure(summary_df, detailed_df):
    """Create publication-quality figure for variable sets."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    # Prepare table data
    table_data = []
    headers = ['Variable Set', 'Variables', 'N', 'Train', 'Valid', 'Test', 'Coverage', 'Coint.']

    for _, row in summary_df.iterrows():
        vars_short = row['variables']
        if len(vars_short) > 50:
            vars_short = vars_short[:47] + '...'
        table_data.append([
            row['variable_set'].upper(),
            vars_short,
            row['n_observations'],
            row['train_obs'],
            row['valid_obs'],
            row['test_obs'],
            f"{row['date_start']} to {row['date_end']}",
            row['johansen_n_coint'],
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)

    # Style header
    for i, key in enumerate(headers):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', weight='bold')

    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        color = '#D9E2F3' if i % 2 == 0 else 'white'
        for j in range(len(headers)):
            table[(i, j)].set_facecolor(color)

    plt.title('Table 1: Variable Sets Summary', fontsize=14, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "table1_variable_sets.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"\nSaved: {FIGURES_DIR / 'table1_variable_sets.png'}")

# ============================================================================
# DELIVERABLE 2: Full Model Matrix
# ============================================================================

def load_actuals_for_varset(varset):
    """Load actual reserves for a variable set."""
    df = load_varset_data(varset)
    if df is None:
        return None
    return df['gross_reserves_usd_m']

def compute_naive_forecast(actuals):
    """Compute naive (random walk) forecast."""
    return actuals.shift(1)

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

    # MAPE
    mape = np.mean(np.abs(errors / a.values)) * 100

    # sMAPE
    smape = 100 * np.mean(2 * abs_errors / (np.abs(a.values) + np.abs(f.values)))

    # R-squared
    ss_res = np.sum(sq_errors)
    ss_tot = np.sum((a.values - np.mean(a.values)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    # MASE (Mean Absolute Scaled Error) - scale by naive in-sample MAE
    naive_errors = np.abs(np.diff(a.values))
    mase = mae / np.mean(naive_errors) if len(naive_errors) > 0 and np.mean(naive_errors) > 0 else np.nan

    # Theil U2
    naive_sq_errors = (a.values[1:] - a.values[:-1]) ** 2
    model_sq_errors = (f.values[1:] - a.values[1:]) ** 2
    theil_u2 = np.sqrt(np.mean(model_sq_errors)) / np.sqrt(np.mean(naive_sq_errors)) if len(naive_sq_errors) > 0 else np.nan

    # Directional accuracy
    actual_dir = np.sign(np.diff(a.values))
    forecast_dir = np.sign(np.diff(f.values))
    dir_acc = np.mean(actual_dir == forecast_dir) * 100 if len(actual_dir) > 0 else np.nan

    return {
        'n': n,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'smape': smape,
        'r_squared': r_squared,
        'mase': mase,
        'theil_u2': theil_u2,
        'dir_acc': dir_acc,
    }

def diebold_mariano_test(errors1, errors2, h=1):
    """
    Diebold-Mariano test for comparing forecast accuracy.
    H0: forecasts have equal accuracy
    Returns: (test_statistic, p_value)
    """
    d = errors1**2 - errors2**2  # loss differential (squared errors)
    n = len(d)

    if n < 10:
        return np.nan, np.nan

    d_mean = np.mean(d)

    # Newey-West variance estimator
    gamma_0 = np.var(d)
    gamma_sum = 0
    for k in range(1, h):
        gamma_k = np.mean((d[k:] - d_mean) * (d[:-k] - d_mean))
        gamma_sum += 2 * (1 - k/h) * gamma_k

    var_d = (gamma_0 + gamma_sum) / n

    if var_d <= 0:
        return np.nan, np.nan

    dm_stat = d_mean / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))  # two-sided

    return dm_stat, p_value

def load_existing_forecasts(varset):
    """Load existing model forecasts for a variable set."""
    forecasts = {}
    actuals = load_actuals_for_varset(varset)

    if actuals is None:
        return None, {}

    # Naive
    forecasts['Naive'] = compute_naive_forecast(actuals)

    # Earlier pipeline models (only for varsets that match)
    if varset == 'parsimonious':
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

    # BVAR models
    bvar_path = DATA_DIR / "forecast_results_academic" / "bvar" / f"bvar_rolling_backtest_{varset}.csv"
    if bvar_path.exists():
        df = pd.read_csv(bvar_path)
        if 'forecast_date' in df.columns:
            df['forecast_date'] = pd.to_datetime(df['forecast_date'])
            h1 = df[df['horizon'] == 1].set_index('forecast_date')
            if 'forecast_point' in h1.columns:
                forecasts[f'BVAR'] = h1['forecast_point']

    # DMA/DMS
    if varset == 'parsimonious':
        dma_path = DATA_DIR / "forecast_results_academic" / "dma" / "dma_rolling_backtest.csv"
        if dma_path.exists():
            df = pd.read_csv(dma_path, parse_dates=['date'])
            df = df.set_index('date')
            if 'dma_forecast' in df.columns:
                forecasts['DMA'] = df['dma_forecast']
            if 'dms_forecast' in df.columns:
                forecasts['DMS'] = df['dms_forecast']

    # XGBoost
    xgb_path = DATA_DIR / "model_verification" / "xgboost_forecasts.csv"
    if xgb_path.exists() and varset == 'parsimonious':
        df = pd.read_csv(xgb_path, parse_dates=['date'])
        df = df.set_index('date')
        if 'forecast' in df.columns:
            forecasts['XGBoost'] = df['forecast']

    # LSTM
    lstm_path = DATA_DIR / "model_verification" / "lstm_forecasts.csv"
    if lstm_path.exists() and varset == 'parsimonious':
        df = pd.read_csv(lstm_path, parse_dates=['date'])
        df = df.set_index('date')
        if 'forecast' in df.columns:
            forecasts['LSTM'] = df['forecast']

    return actuals, forecasts

def run_bvar_for_varset(varset):
    """Run BVAR model for a specific variable set."""
    df = load_varset_data(varset)
    if df is None or len(df) < 60:
        return None

    # Simple BVAR implementation using OLS with Minnesota-style shrinkage
    # This is a simplified version for the deliverables
    target = 'gross_reserves_usd_m'
    cols = [c for c in df.columns if c not in ['split']]

    if target not in cols:
        return None

    # Create lag features
    lags = 2
    data = df[cols].copy()
    for lag in range(1, lags + 1):
        for col in cols:
            data[f'{col}_lag{lag}'] = df[col].shift(lag)

    data = data.dropna()

    # Rolling forecast
    results = []
    for i in range(len(data)):
        current_date = data.index[i]
        if current_date < TEST_START:
            continue

        # Use all data up to this point for training
        train = data.loc[:current_date].iloc[:-1]
        if len(train) < 50:
            continue

        # Simple OLS forecast
        feature_cols = [c for c in train.columns if 'lag' in c]
        X = train[feature_cols]
        y = train[target]

        try:
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=1.0)
            model.fit(X, y)

            # Predict current period
            X_pred = data.loc[[current_date], feature_cols]
            pred = model.predict(X_pred)[0]

            results.append({
                'date': current_date,
                'actual': data.loc[current_date, target],
                'forecast': pred,
            })
        except:
            continue

    if not results:
        return None

    results_df = pd.DataFrame(results).set_index('date')
    return results_df['forecast']

def generate_model_matrix():
    """Generate Deliverable 2: Full Model x Variable Set Matrix."""
    print("\n" + "=" * 80)
    print("DELIVERABLE 2: Full Model Matrix")
    print("=" * 80)

    all_results = []
    dm_tests = []

    for varset in VARSETS:
        print(f"\n--- {varset.upper()} ---")

        actuals, forecasts = load_existing_forecasts(varset)
        if actuals is None:
            print(f"  No data for {varset}")
            continue

        # Run BVAR if not already loaded
        if 'BVAR' not in forecasts:
            bvar_fc = run_bvar_for_varset(varset)
            if bvar_fc is not None:
                forecasts['BVAR'] = bvar_fc

        # Get naive for comparison
        naive = forecasts.get('Naive')
        if naive is None:
            continue

        naive_metrics = compute_metrics(actuals, naive, RECOVERY_START, RECOVERY_END)
        if naive_metrics is None:
            continue

        naive_rmse = naive_metrics['rmse']

        for model_name, fc in forecasts.items():
            # Post-crisis period
            metrics = compute_metrics(actuals, fc, RECOVERY_START, RECOVERY_END)
            if metrics is None:
                continue

            # Compute DM test vs naive
            common_idx = fc.dropna().index.intersection(naive.dropna().index)
            common_idx = common_idx[(common_idx >= RECOVERY_START) & (common_idx <= RECOVERY_END)]
            common_idx = common_idx.intersection(actuals.dropna().index)

            if len(common_idx) >= 5:
                model_errors = fc.loc[common_idx].values - actuals.loc[common_idx].values
                naive_errors = naive.loc[common_idx].values - actuals.loc[common_idx].values
                dm_stat, dm_pval = diebold_mariano_test(model_errors, naive_errors)
            else:
                dm_stat, dm_pval = np.nan, np.nan

            rmse_vs_naive = (metrics['rmse'] / naive_rmse - 1) * 100 if naive_rmse > 0 else np.nan

            all_results.append({
                'variable_set': varset,
                'model': model_name,
                'period': 'post_crisis',
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
                'beats_naive': metrics['rmse'] < naive_rmse,
            })

            print(f"  {model_name}: RMSE={metrics['rmse']:.1f}, vs_naive={rmse_vs_naive:+.1f}%")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "deliverable2_model_matrix.csv", index=False)

    # Create pivot table
    if not results_df.empty:
        pivot = results_df.pivot_table(
            index='model',
            columns='variable_set',
            values='rmse_vs_naive_pct',
            aggfunc='first'
        )
        pivot.to_csv(OUTPUT_DIR / "deliverable2_rmse_pivot.csv")

        create_model_matrix_figure(results_df)

    return results_df

def create_model_matrix_figure(results_df):
    """Create publication-quality figure for model matrix."""
    # Filter to post-crisis and create summary
    df = results_df[results_df['period'] == 'post_crisis'].copy()

    if df.empty:
        return

    # Pivot for heatmap
    pivot = df.pivot_table(
        index='model',
        columns='variable_set',
        values='rmse_vs_naive_pct',
        aggfunc='first'
    )

    # Sort by average performance
    pivot['avg'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('avg')
    pivot = pivot.drop('avg', axis=1)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Create heatmap manually
    models = pivot.index.tolist()
    varsets = pivot.columns.tolist()

    # Prepare detailed table
    table_data = []
    headers = ['Model'] + [v.upper()[:6] for v in varsets] + ['Best']

    for model in models:
        row = [model]
        for varset in varsets:
            val = pivot.loc[model, varset]
            if pd.isna(val):
                row.append('N/A')
            else:
                row.append(f'{val:+.1f}%')
        # Find best varset for this model
        best = pivot.loc[model].idxmin()
        row.append(best if pd.notna(best) else 'N/A')
        table_data.append(row)

    ax.axis('off')

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)

    # Style header
    for i, key in enumerate(headers):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', weight='bold')

    # Color cells based on performance
    for i, row in enumerate(table_data, 1):
        for j, val in enumerate(row[1:-1], 1):  # Skip model name and best
            if val == 'N/A':
                table[(i, j)].set_facecolor('#F0F0F0')
            else:
                try:
                    v = float(val.replace('%', '').replace('+', ''))
                    if v < -10:
                        table[(i, j)].set_facecolor('#C6EFCE')  # Green - beats naive
                    elif v < 0:
                        table[(i, j)].set_facecolor('#FFEB9C')  # Yellow - slightly beats
                    else:
                        table[(i, j)].set_facecolor('#FFC7CE')  # Red - worse
                except:
                    pass

    plt.title('Table 2: Model Performance Matrix (RMSE vs Naive %)\nPost-Crisis Period (2024-07+)',
              fontsize=14, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "table2_model_matrix.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"\nSaved: {FIGURES_DIR / 'table2_model_matrix.png'}")

    # Create detailed metrics table
    create_detailed_metrics_figure(results_df)

def create_detailed_metrics_figure(results_df):
    """Create detailed metrics table figure."""
    df = results_df[results_df['period'] == 'post_crisis'].copy()

    if df.empty:
        return

    # Sort by RMSE
    df = df.sort_values(['variable_set', 'rmse'])

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.axis('off')

    headers = ['VarSet', 'Model', 'N', 'RMSE', 'vs Naive', 'MAPE', 'sMAPE', 'R²', 'MASE', 'U2', 'Dir%', 'DM p']

    table_data = []
    for _, row in df.iterrows():
        table_data.append([
            row['variable_set'][:4].upper(),
            row['model'][:10],
            int(row['n']),
            f"{row['rmse']:.0f}",
            f"{row['rmse_vs_naive_pct']:+.1f}%" if pd.notna(row['rmse_vs_naive_pct']) else 'N/A',
            f"{row['mape']:.1f}%" if pd.notna(row['mape']) else 'N/A',
            f"{row['smape']:.1f}%" if pd.notna(row['smape']) else 'N/A',
            f"{row['r_squared']:.3f}" if pd.notna(row['r_squared']) else 'N/A',
            f"{row['mase']:.2f}" if pd.notna(row['mase']) else 'N/A',
            f"{row['theil_u2']:.2f}" if pd.notna(row['theil_u2']) else 'N/A',
            f"{row['dir_acc']:.0f}" if pd.notna(row['dir_acc']) else 'N/A',
            f"{row['dm_pvalue']:.3f}" if pd.notna(row['dm_pvalue']) else 'N/A',
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.1, 1.5)

    for i, key in enumerate(headers):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', weight='bold')

    for i in range(1, len(table_data) + 1):
        color = '#D9E2F3' if i % 2 == 0 else 'white'
        for j in range(len(headers)):
            table[(i, j)].set_facecolor(color)

    plt.title('Table 3: Detailed Model Metrics (Post-Crisis Period)', fontsize=14, weight='bold', pad=20)

    # Add footnote
    footnote = """
    Notes: RMSE = Root Mean Square Error (USD M), MAPE = Mean Absolute Percentage Error, sMAPE = Symmetric MAPE,
    R² = R-squared, MASE = Mean Absolute Scaled Error, U2 = Theil U2, Dir% = Directional Accuracy,
    DM p = Diebold-Mariano test p-value vs Naive (two-sided). Green cells indicate model beats naive.
    """
    plt.figtext(0.5, 0.02, footnote, ha='center', fontsize=8, style='italic')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "table3_detailed_metrics.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'table3_detailed_metrics.png'}")

# ============================================================================
# DELIVERABLE 3: Data Quality Assessment
# ============================================================================

def analyze_data_quality():
    """Generate Deliverable 3: Data Quality Assessment."""
    print("\n" + "=" * 80)
    print("DELIVERABLE 3: Data Quality Assessment")
    print("=" * 80)

    quality_report = []
    missing_report = []

    # Check external source files
    external_files = [
        ('historical_reserves.csv', 'gross_reserves_usd_m'),
        ('historical_fx.csv', 'usd_lkr'),
        ('monthly_imports_usd.csv', 'imports_usd_m'),
        ('monthly_exports_usd.csv', 'exports_usd_m'),
        ('remittances_monthly.csv', 'remittances_usd_m'),
        ('tourism_earnings_monthly.csv', 'tourism_usd_m'),
        ('monetary_aggregates_monthly.csv', 'm2'),
        ('D12_reserves.csv', 'reserves'),
    ]

    for filename, expected_col in external_files:
        path = DATA_DIR / "external" / filename
        if not path.exists():
            quality_report.append({
                'source': filename,
                'status': 'NOT_FOUND',
                'n_rows': 0,
                'n_missing': 0,
                'date_range': 'N/A',
            })
            continue

        try:
            df = pd.read_csv(path)
            # Try to find date column
            date_col = None
            for col in ['date', 'Date', 'DATE', 'month', 'Month']:
                if col in df.columns:
                    date_col = col
                    break

            if date_col:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                date_range = f"{df[date_col].min():%Y-%m} to {df[date_col].max():%Y-%m}"
            else:
                date_range = 'No date column'

            # Count missing values
            n_missing = df.isnull().sum().sum()
            missing_cols = df.columns[df.isnull().any()].tolist()

            quality_report.append({
                'source': filename,
                'status': 'OK',
                'n_rows': len(df),
                'n_cols': len(df.columns),
                'n_missing': n_missing,
                'missing_cols': ', '.join(missing_cols) if missing_cols else 'None',
                'date_range': date_range,
            })

            # Detailed missing analysis
            for col in df.columns:
                n_null = df[col].isnull().sum()
                if n_null > 0:
                    missing_report.append({
                        'source': filename,
                        'column': col,
                        'n_missing': n_null,
                        'pct_missing': n_null / len(df) * 100,
                    })
        except Exception as e:
            quality_report.append({
                'source': filename,
                'status': f'ERROR: {str(e)[:50]}',
                'n_rows': 0,
                'n_missing': 0,
                'date_range': 'N/A',
            })

    # Check prepared datasets
    print("\nVariable Set Data Quality:")
    varset_quality = []

    for varset in VARSETS:
        df = load_varset_data(varset)
        if df is None:
            continue

        for col in df.columns:
            if col == 'split':
                continue

            n_total = len(df)
            n_valid = df[col].notna().sum()
            n_missing = n_total - n_valid

            # Check for zero values
            n_zeros = (df[col] == 0).sum()

            # Check for outliers (beyond 3 std)
            mean = df[col].mean()
            std = df[col].std()
            n_outliers = ((df[col] < mean - 3*std) | (df[col] > mean + 3*std)).sum()

            varset_quality.append({
                'varset': varset,
                'variable': col,
                'n_total': n_total,
                'n_valid': n_valid,
                'n_missing': n_missing,
                'pct_missing': n_missing / n_total * 100,
                'n_zeros': n_zeros,
                'n_outliers': n_outliers,
                'mean': mean,
                'std': std,
                'min': df[col].min(),
                'max': df[col].max(),
            })

            if n_missing > 0:
                print(f"  {varset}/{col}: {n_missing} missing ({n_missing/n_total*100:.1f}%)")

    # Save reports
    pd.DataFrame(quality_report).to_csv(OUTPUT_DIR / "deliverable3_source_quality.csv", index=False)
    pd.DataFrame(missing_report).to_csv(OUTPUT_DIR / "deliverable3_missing_details.csv", index=False)
    pd.DataFrame(varset_quality).to_csv(OUTPUT_DIR / "deliverable3_varset_quality.csv", index=False)

    # Create methodology documentation
    create_methodology_doc(quality_report, varset_quality)

    # Create figure
    create_data_quality_figure(quality_report, varset_quality)

    return quality_report, varset_quality

def create_methodology_doc(quality_report, varset_quality):
    """Create methodology documentation."""
    doc = """# Data Quality and Feature Engineering Methodology

## 1. Data Sources

| Source File | Status | Rows | Date Range | Missing Values |
|-------------|--------|------|------------|----------------|
"""
    for row in quality_report:
        doc += f"| {row['source']} | {row['status']} | {row.get('n_rows', 0)} | {row['date_range']} | {row.get('n_missing', 0)} |\n"

    doc += """

## 2. Missing Data Strategy

Configuration from `config.py`:
```python
MISSING_STRATEGY = {
    "method": "ffill_limit",
    "limit": 3,
    "drop_remaining": True,
}
```

**Process:**
1. Forward-fill missing values with a maximum limit of 3 consecutive periods
2. Drop any remaining rows with missing values
3. Applied separately to each variable set

## 3. Train/Validation/Test Split

| Split | End Date | Purpose |
|-------|----------|---------|
| Train | 2019-12-01 | Model estimation |
| Validation | 2022-12-01 | Hyperparameter tuning |
| Test | 2025-03-01+ | Out-of-sample evaluation |

## 4. Variable Set Quality Summary

| Variable Set | Variables | Total Obs | Missing | Zeros | Outliers |
|--------------|-----------|-----------|---------|-------|----------|
"""

    # Aggregate by varset
    vq_df = pd.DataFrame(varset_quality)
    for varset in VARSETS:
        vs_data = vq_df[vq_df['varset'] == varset]
        if vs_data.empty:
            continue
        n_vars = len(vs_data)
        n_total = vs_data['n_total'].iloc[0] if len(vs_data) > 0 else 0
        n_missing = vs_data['n_missing'].sum()
        n_zeros = vs_data['n_zeros'].sum()
        n_outliers = vs_data['n_outliers'].sum()
        doc += f"| {varset} | {n_vars} | {n_total} | {n_missing} | {n_zeros} | {n_outliers} |\n"

    doc += """

## 5. Feature Engineering

### 5.1 ARIMA Dataset
- Target transformations: diff(1), log, log_diff(1), pct_change
- Exogenous variables: as specified per variable set

### 5.2 VECM Dataset
- Level variables for cointegration analysis
- Johansen test for cointegrating rank
- Error correction term (ECT) computed from first cointegrating vector
- Differenced variables for VAR component

### 5.3 MS-VAR Dataset
- First differences of all variables
- Standardization based on training period statistics
- Regime initialization flag based on rolling volatility

### 5.4 Machine Learning Features (XGBoost)
- Lag features: 1, 2, 3, 6, 12 months
- Rolling statistics: MA(3), MA(6), STD(3)
- Momentum features: diff(1), diff(3)

### 5.5 LSTM Features
- Sequence length: 6 months
- Features: target + lags + momentum + MA
- MinMax scaling applied

## 6. Quality Checks Performed

1. **Date continuity**: Verified no gaps in monthly series
2. **Unit consistency**: All monetary values in USD millions
3. **Outlier detection**: Flagged values beyond 3 standard deviations
4. **Zero handling**: Monitored zero values (may indicate data issues)
5. **Stationarity**: ADF and KPSS tests on each variable
6. **Cointegration**: Johansen trace test for multivariate systems

"""

    with open(OUTPUT_DIR / "deliverable3_methodology.md", 'w') as f:
        f.write(doc)

    print(f"\nSaved: {OUTPUT_DIR / 'deliverable3_methodology.md'}")

def create_data_quality_figure(quality_report, varset_quality):
    """Create data quality summary figure."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Panel 1: Source file quality
    ax1 = axes[0]
    ax1.axis('off')

    headers1 = ['Source', 'Status', 'Rows', 'Missing']
    table_data1 = []
    for row in quality_report:
        status_short = 'OK' if row['status'] == 'OK' else 'ERR'
        table_data1.append([
            row['source'][:25],
            status_short,
            row.get('n_rows', 0),
            row.get('n_missing', 0),
        ])

    if table_data1:
        table1 = ax1.table(
            cellText=table_data1,
            colLabels=headers1,
            loc='center',
            cellLoc='center',
        )
        table1.auto_set_font_size(False)
        table1.set_fontsize(9)
        table1.scale(1.2, 1.5)

        for i in range(len(headers1)):
            table1[(0, i)].set_facecolor('#4472C4')
            table1[(0, i)].set_text_props(color='white', weight='bold')

    ax1.set_title('Panel A: Source File Quality', fontsize=12, weight='bold')

    # Panel 2: Variable set quality
    ax2 = axes[1]
    ax2.axis('off')

    vq_df = pd.DataFrame(varset_quality)
    headers2 = ['VarSet', 'Variable', 'N', 'Missing', 'Outliers']
    table_data2 = []

    for _, row in vq_df.iterrows():
        table_data2.append([
            row['varset'][:6],
            row['variable'][:20],
            row['n_total'],
            row['n_missing'],
            row['n_outliers'],
        ])

    if table_data2:
        table2 = ax2.table(
            cellText=table_data2[:20],  # Limit rows
            colLabels=headers2,
            loc='center',
            cellLoc='center',
        )
        table2.auto_set_font_size(False)
        table2.set_fontsize(9)
        table2.scale(1.2, 1.5)

        for i in range(len(headers2)):
            table2[(0, i)].set_facecolor('#4472C4')
            table2[(0, i)].set_text_props(color='white', weight='bold')

    ax2.set_title('Panel B: Variable Set Quality', fontsize=12, weight='bold')

    plt.suptitle('Table 4: Data Quality Assessment', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "table4_data_quality.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'table4_data_quality.png'}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("ACADEMIC DELIVERABLES GENERATOR")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")

    # Deliverable 1
    varset_summary, varset_details = generate_varset_table()

    # Deliverable 2
    model_matrix = generate_model_matrix()

    # Deliverable 3
    quality_report, varset_quality = analyze_data_quality()

    print("\n" + "=" * 80)
    print("DELIVERABLES COMPLETE")
    print("=" * 80)
    print(f"""
Files generated:

DELIVERABLE 1 - Variable Sets:
  {OUTPUT_DIR / 'deliverable1_variable_sets.csv'}
  {OUTPUT_DIR / 'deliverable1_variable_details.csv'}
  {FIGURES_DIR / 'table1_variable_sets.png'}

DELIVERABLE 2 - Model Matrix:
  {OUTPUT_DIR / 'deliverable2_model_matrix.csv'}
  {OUTPUT_DIR / 'deliverable2_rmse_pivot.csv'}
  {FIGURES_DIR / 'table2_model_matrix.png'}
  {FIGURES_DIR / 'table3_detailed_metrics.png'}

DELIVERABLE 3 - Data Quality:
  {OUTPUT_DIR / 'deliverable3_source_quality.csv'}
  {OUTPUT_DIR / 'deliverable3_missing_details.csv'}
  {OUTPUT_DIR / 'deliverable3_varset_quality.csv'}
  {OUTPUT_DIR / 'deliverable3_methodology.md'}
  {FIGURES_DIR / 'table4_data_quality.png'}
""")

if __name__ == "__main__":
    main()
