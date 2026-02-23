"""
MS-VAR Conditional Scenario Forecasting
========================================

Generates reserve forecasts for all 10 policy scenarios using MS-VAR.

Process:
1. Load data and fit MS-VAR
2. Generate exogenous paths for each scenario
3. Run conditional forecasts
4. Compare results in a summary table
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "reserves_project" / "scripts"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

# Import MS-VAR directly
import importlib.util
msvar_path = PROJECT_ROOT / "reserves_project" / "reserves_project" / "models" / "ms_switching_var.py"
spec = importlib.util.spec_from_file_location("ms_switching_var", msvar_path)
ms_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ms_module)
MarkovSwitchingVAR = ms_module.MarkovSwitchingVAR

from academic.scenarios.scenario_framework import (
    Scenario, ScenarioEngine, POLICY_SCENARIOS, create_scenario_fan_chart
)

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "scenario_analysis"
FIGURES_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "gross_reserves_usd_m"


def load_and_prepare_data(varset: str = "parsimonious"):
    """Load data and prepare for MS-VAR."""
    path = DATA_DIR / "forecast_prep_academic" / f"varset_{varset}" / "vecm_levels.csv"
    df = pd.read_csv(path, parse_dates=['date'])
    df = df.set_index('date').sort_index()

    # Get variable columns (exclude split if present)
    var_cols = [c for c in df.columns if c not in ['split']]

    return df, var_cols


def fit_msvar(df: pd.DataFrame, var_cols: list, ar_order: int = 2):
    """Fit MS-VAR model on the full dataset with volatility-based regime init."""
    # Use differences for stationarity
    diff_data = df[var_cols].diff().dropna()

    # Create initial regime assignments based on volatility
    # High volatility = crisis (regime 0), low volatility = recovery (regime 1)
    target_diff = diff_data[TARGET]
    rolling_vol = target_diff.rolling(6).std()
    vol_median = rolling_vol.median()

    # Binary regime assignment: 0 = high vol (crisis), 1 = low vol (recovery)
    init_states = (rolling_vol < vol_median).astype(int).values
    # Handle NaN from rolling
    init_states = np.nan_to_num(init_states, nan=0).astype(int)

    print(f"Initial regime distribution: {np.bincount(init_states)}")

    # Fit MS-VAR with initial states
    msvar = MarkovSwitchingVAR(n_regimes=2, ar_order=ar_order, max_iter=200)
    msvar.fit(diff_data.values, init_states=init_states)

    print(f"MS-VAR fitted with {msvar.n_regimes} regimes, AR({ar_order})")
    print(f"Transition matrix:\n{msvar.transition_}")
    print(f"Log-likelihood: {msvar.loglik_:.2f}")

    # Check regime differentiation
    coef_diff = np.abs(msvar.coefs_[0] - msvar.coefs_[1]).mean()
    print(f"Mean coefficient difference between regimes: {coef_diff:.4f}")

    return msvar, diff_data


def scenario_to_regime_probs(scenario: Scenario) -> np.ndarray:
    """
    Map policy scenario to regime probabilities.

    MS-VAR has 2 regimes:
    - Regime 0: Crisis/high-volatility
    - Regime 1: Recovery/low-volatility

    We map scenarios to regime probabilities based on their nature.
    """
    # Adverse scenarios push toward crisis regime
    # Upside scenarios push toward recovery regime

    scenario_regime_mapping = {
        "Baseline": [0.3, 0.7],  # Slight recovery bias (current state)
        "LKR Depreciation 10%": [0.5, 0.5],  # Mild stress
        "LKR Depreciation 20%": [0.7, 0.3],  # Significant stress
        "Export Shock (-15%)": [0.6, 0.4],  # Moderate stress
        "Remittance Decline (-20%)": [0.55, 0.45],  # Moderate stress
        "Tourism Recovery (+25%)": [0.2, 0.8],  # Strong recovery
        "Oil Price Shock": [0.6, 0.4],  # Moderate stress
        "IMF Tranche Delay": [0.65, 0.35],  # Confidence shock
        "Combined Adverse": [0.8, 0.2],  # Full stress test
        "Combined Upside": [0.1, 0.9],  # Best case
    }

    probs = scenario_regime_mapping.get(scenario.name, [0.5, 0.5])
    return np.array(probs)


def forecast_regime_conditional(
    msvar: MarkovSwitchingVAR,
    y_history: np.ndarray,
    steps: int,
    regime_probs: np.ndarray,
    lock_regime: bool = True,
) -> np.ndarray:
    """
    Custom forecast that can lock regime probabilities.

    If lock_regime=True, keeps regime_probs constant throughout
    (doesn't evolve via transition matrix).
    """
    p = msvar.ar_order
    history = y_history.copy()
    k = history.shape[1]
    r = msvar.n_regimes

    forecasts = []
    current_probs = regime_probs.copy()

    for h in range(steps):
        # Build lag features
        lags = []
        for lag in range(1, p + 1):
            lags.append(history[-lag])
        x = np.concatenate([np.array([1.0]), np.concatenate(lags)])

        # Compute regime-specific means
        regime_means = []
        for r_idx in range(r):
            coef = msvar.coefs_[r_idx]
            mu = x @ coef
            regime_means.append(mu)
        regime_means = np.vstack(regime_means)

        # Weighted average prediction
        y_pred = (current_probs[:, None] * regime_means).sum(axis=0)
        forecasts.append(y_pred)
        history = np.vstack([history, y_pred])

        # Update regime probs (or keep locked)
        if not lock_regime:
            current_probs = current_probs @ msvar.transition_

    return np.vstack(forecasts)


def forecast_scenario(
    msvar: MarkovSwitchingVAR,
    scenario: Scenario,
    diff_data: pd.DataFrame,
    level_data: pd.DataFrame,
    var_cols: list,
    regime_probs: np.ndarray = None,
) -> dict:
    """
    Generate forecast for a single scenario.

    MS-VAR scenarios are implemented via regime probabilities:
    - Adverse scenarios increase crisis regime probability
    - Upside scenarios increase recovery regime probability

    Regime probabilities are LOCKED throughout the forecast horizon
    to generate truly differentiated scenarios.

    Returns dict with forecast path and metrics.
    """
    horizon = scenario.horizon_months
    ar_order = msvar.ar_order

    # Get history for lag construction
    y_history = diff_data[var_cols].values[-ar_order:]

    # Get base values
    last_level = level_data[TARGET].iloc[-1]
    last_date = level_data.index[-1]

    # Map scenario to regime probabilities
    if regime_probs is None:
        regime_probs = scenario_to_regime_probs(scenario)

    # Run forecast with LOCKED regime probabilities
    diff_forecast = forecast_regime_conditional(
        msvar,
        y_history,
        steps=horizon,
        regime_probs=regime_probs,
        lock_regime=True  # Keep regime constant throughout
    )

    # Convert differenced forecast to levels
    target_idx = var_cols.index(TARGET)
    diff_target = diff_forecast[:, target_idx]
    level_forecast = last_level + np.cumsum(diff_target)

    # Generate forecast dates
    forecast_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=horizon,
        freq='MS'
    )

    # Compute summary metrics
    final_level = level_forecast[-1]
    total_change = final_level - last_level
    pct_change = (total_change / last_level) * 100
    avg_monthly_change = total_change / horizon

    return {
        'scenario': scenario.name,
        'description': scenario.description,
        'horizon': horizon,
        'start_level': last_level,
        'end_level': final_level,
        'total_change': total_change,
        'pct_change': pct_change,
        'avg_monthly_change': avg_monthly_change,
        'min_level': level_forecast.min(),
        'max_level': level_forecast.max(),
        'forecast_path': pd.Series(level_forecast, index=forecast_dates),
        'regime_probs_used': regime_probs.tolist(),
    }


def run_all_scenarios(varset: str = "parsimonious"):
    """Run forecasts for all 10 policy scenarios."""
    print("="*70)
    print("MS-VAR CONDITIONAL SCENARIO FORECASTING")
    print("="*70)

    # Load data
    level_data, var_cols = load_and_prepare_data(varset)
    print(f"\nVariable set: {varset}")
    print(f"Variables: {var_cols}")
    print(f"Last observation: {level_data.index[-1]}")
    print(f"Last reserves: ${level_data[TARGET].iloc[-1]:,.0f}M")

    # Fit MS-VAR
    print("\n### Fitting MS-VAR ###")
    msvar, diff_data = fit_msvar(level_data, var_cols)

    # Get final regime probabilities
    final_regime_probs = msvar.smoothed_probs_[-1] if msvar.smoothed_probs_ is not None else np.array([0.5, 0.5])
    print(f"\nCurrent regime probabilities: Regime 0 = {final_regime_probs[0]:.1%}, Regime 1 = {final_regime_probs[1]:.1%}")

    # Run all scenarios
    print("\n### Running Scenarios ###")
    results = []
    forecast_paths = {}

    for scenario_key, scenario in POLICY_SCENARIOS.items():
        result = forecast_scenario(
            msvar, scenario, diff_data, level_data, var_cols
        )
        results.append(result)
        forecast_paths[scenario.name] = result['forecast_path']
        print(f"  {scenario.name}: {result['start_level']:,.0f} → {result['end_level']:,.0f} ({result['pct_change']:+.1f}%)")

    # Create summary DataFrame
    summary_df = pd.DataFrame([
        {
            'Scenario': r['scenario'],
            'Description': r['description'],
            'Start (USD M)': r['start_level'],
            'End (USD M)': r['end_level'],
            'Change (USD M)': r['total_change'],
            'Change (%)': r['pct_change'],
            'Avg Monthly': r['avg_monthly_change'],
            'Min': r['min_level'],
            'Max': r['max_level'],
        }
        for r in results
    ])

    # Sort by end level
    summary_df = summary_df.sort_values('End (USD M)', ascending=False)

    return summary_df, forecast_paths, results


def create_results_table_figure(summary_df: pd.DataFrame, output_path: Path):
    """Create publication-quality results table."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')

    # Format numeric columns
    table_data = []
    for _, row in summary_df.iterrows():
        table_data.append([
            row['Scenario'],
            f"${row['Start (USD M)']:,.0f}",
            f"${row['End (USD M)']:,.0f}",
            f"${row['Change (USD M)']:+,.0f}",
            f"{row['Change (%)']:+.1f}%",
            f"${row['Avg Monthly']:+,.0f}",
        ])

    headers = ['Scenario', 'Start', 'End (12m)', 'Total Δ', '% Change', 'Avg Monthly Δ']

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)

    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(color='white', weight='bold')

    # Color rows by outcome
    for i, row in enumerate(table_data, 1):
        pct_change = float(row[4].replace('%', '').replace('+', ''))
        if pct_change > 5:
            color = '#C6EFCE'  # Green - upside
        elif pct_change < -5:
            color = '#FFC7CE'  # Red - downside
        else:
            color = '#FFEB9C'  # Yellow - neutral

        for j in range(len(headers)):
            table[(i, j)].set_facecolor(color)

    plt.title('MS-VAR Scenario Forecasts: 12-Month Reserve Projections\n'
              'Green = Upside (>+5%) | Yellow = Neutral | Red = Downside (<-5%)',
              fontsize=14, weight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_fan_chart(forecast_paths: dict, start_level: float, output_path: Path):
    """Create fan chart with all scenario paths."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Color mapping
    colors = {
        'Baseline': 'black',
        'Combined Upside': 'darkgreen',
        'Tourism Recovery (+25%)': 'green',
        'Combined Adverse': 'darkred',
        'Oil Price Shock': 'red',
        'Export Shock (-15%)': 'orange',
        'Remittance Decline (-20%)': 'orangered',
        'LKR Depreciation 10%': 'coral',
        'LKR Depreciation 20%': 'crimson',
        'IMF Tranche Delay': 'purple',
    }

    linewidths = {
        'Baseline': 3,
        'Combined Upside': 2.5,
        'Combined Adverse': 2.5,
    }

    # Plot each scenario
    for name, path in forecast_paths.items():
        color = colors.get(name, 'gray')
        lw = linewidths.get(name, 1.5)
        linestyle = '-' if name in ['Baseline', 'Combined Upside', 'Combined Adverse'] else '--'
        alpha = 1.0 if name in ['Baseline', 'Combined Upside', 'Combined Adverse'] else 0.7

        ax.plot(path.index, path.values, color=color, linewidth=lw,
                linestyle=linestyle, alpha=alpha, label=name)

    # Add start point
    first_date = list(forecast_paths.values())[0].index[0]
    ax.scatter([first_date - pd.DateOffset(months=1)], [start_level],
               color='black', s=100, zorder=5, marker='o', label='Current')

    # Formatting
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Gross Reserves (USD Million)', fontsize=12)
    ax.set_title('MS-VAR Scenario Fan Chart: 12-Month Reserve Projections',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add critical threshold line
    ax.axhline(3000, color='red', linestyle=':', alpha=0.5, linewidth=2)
    ax.text(path.index[-1], 3100, 'Critical (~3mo imports)', fontsize=9, color='red')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Main execution."""
    # Run scenarios
    summary_df, forecast_paths, results = run_all_scenarios("parsimonious")

    # Print results table
    print("\n" + "="*70)
    print("SCENARIO RESULTS SUMMARY")
    print("="*70)
    print(summary_df.to_string(index=False))

    # Save CSV
    csv_path = OUTPUT_DIR / "msvar_scenario_results.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Create figures
    create_results_table_figure(summary_df, FIGURES_DIR / "msvar_scenario_table.png")

    start_level = results[0]['start_level']
    create_fan_chart(forecast_paths, start_level, FIGURES_DIR / "msvar_scenario_fan_chart.png")

    # Save forecast paths
    paths_df = pd.DataFrame(forecast_paths)
    paths_df.to_csv(OUTPUT_DIR / "msvar_scenario_paths.csv")
    print(f"Saved: {OUTPUT_DIR / 'msvar_scenario_paths.csv'}")

    print("\n" + "="*70)
    print("SCENARIO ANALYSIS COMPLETE")
    print("="*70)

    return summary_df, forecast_paths


if __name__ == "__main__":
    summary_df, forecast_paths = main()
