"""
Generate Robustness Tables and Academic Output
===============================================

Main execution script for Phase 5 (Spec 11).

This script:
1. Loads all forecast results from Phases 1-4
2. Computes subsample robustness (pre-crisis, crisis, post-default, COVID)
3. Computes horizon robustness (h=1,3,6,12)
4. Computes variable set robustness (5 specifications)
5. Generates LaTeX tables (Tables 1-6, Appendix A1-A6)
6. Generates publication figures
7. Compiles paper statistics JSON

Usage:
    python generate_robustness_tables.py

Output:
    data/robustness/
        tables/          LaTeX .tex files
        figures/         PNG/PDF figures
        summary/         JSON summaries

Author: Academic Forecasting Pipeline
Date: 2026-02-11
"""

import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from reserves_project.config.paths import PROJECT_ROOT, DATA_DIR
from reserves_project.robustness.subsample_analysis import (
    SubsampleAnalyzer,
    compute_subsample_metrics,
    subsample_robustness_table,
    compute_period_stability,
    STANDARD_PERIODS,
)
from reserves_project.robustness.horizon_analysis import (
    HorizonAnalyzer,
    horizon_comparison_table,
    horizon_ranking_stability,
    compute_forecast_deterioration,
)
from reserves_project.robustness.variable_set_analysis import (
    VariableSetAnalyzer,
    VARIABLE_SETS,
    variable_set_comparison,
    ranking_consistency_test,
    compute_varset_sensitivity,
)
from reserves_project.robustness.latex_tables import (
    LaTeXTableGenerator,
    generate_all_tables,
)
from reserves_project.robustness.publication_figures import (
    FigureGenerator,
    generate_all_figures,
)
from reserves_project.robustness.paper_statistics import (
    compile_paper_statistics,
    save_statistics,
    generate_statistics_summary,
)
from reserves_project.utils.run_manifest import write_run_manifest


# =============================================================================
# Configuration
# =============================================================================

# Paths
FORECAST_RESULTS_DIR = DATA_DIR / "forecast_results_academic"
STATISTICAL_TESTS_DIR = DATA_DIR / "statistical_tests"
STATISTICAL_TESTS_UNIFIED_DIR = DATA_DIR / "statistical_tests_unified"
FORECAST_PREP_DIR = DATA_DIR / "forecast_prep_academic"
UNIFIED_RESULTS_DIR = DATA_DIR / "forecast_results_unified"
OUTPUT_DIR = DATA_DIR / "robustness"

# Create output directories
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
SUMMARY_DIR = OUTPUT_DIR / "summary"

for d in [TABLES_DIR, FIGURES_DIR, SUMMARY_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_statistical_test_results():
    """Load existing DM and MCS results from Phase 4."""
    results = {}

    def _load_from_dir(base_dir: Path) -> dict:
        data = {}
        if base_dir is None:
            return data

        mcs_path = base_dir / "mcs_results.json"
        if mcs_path.exists():
            with open(mcs_path) as f:
                data["mcs_results"] = json.load(f)

        mcs_summary_path = base_dir / "mcs_summary.csv"
        if mcs_summary_path.exists():
            data["mcs_summary"] = pd.read_csv(mcs_summary_path)

        dm_stats_path = base_dir / "dm_test_matrix.csv"
        if dm_stats_path.exists():
            data["dm_stats"] = pd.read_csv(dm_stats_path, index_col=0)

        dm_pvalues_path = base_dir / "dm_pvalues_matrix.csv"
        if dm_pvalues_path.exists():
            data["dm_pvalues"] = pd.read_csv(dm_pvalues_path, index_col=0)

        return data

    unified = _load_from_dir(STATISTICAL_TESTS_UNIFIED_DIR)
    legacy = _load_from_dir(STATISTICAL_TESTS_DIR)

    results.update(unified)
    for key, value in legacy.items():
        results.setdefault(key, value)

    return results


def load_combination_forecasts():
    """Load combination forecast results."""
    comb_dir = FORECAST_RESULTS_DIR / "combinations"

    results = {}

    # Rolling backtest
    backtest_path = comb_dir / "combination_rolling_backtest.csv"
    if backtest_path.exists():
        results['backtest'] = pd.read_csv(backtest_path, parse_dates=['date'])

    # Summary
    summary_path = comb_dir / "combination_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            results['summary'] = json.load(f)

    return results


def load_bvar_results():
    """Load BVAR results across variable sets."""
    bvar_dir = FORECAST_RESULTS_DIR / "bvar"
    results = {}

    varsets = ['parsimonious', 'bop', 'monetary', 'pca', 'full']

    for varset in varsets:
        backtest_path = bvar_dir / f"bvar_rolling_backtest_{varset}.csv"
        if backtest_path.exists():
            results[varset] = pd.read_csv(backtest_path)

    return results


def load_favar_results():
    """Load FAVAR results with multi-horizon backtests."""
    favar_dir = FORECAST_RESULTS_DIR / "favar"
    results = {}

    # Multi-horizon backtests
    for h in [1, 3, 6, 12]:
        backtest_path = favar_dir / f"favar_rolling_backtest_h{h}.csv"
        if backtest_path.exists():
            results[h] = pd.read_csv(backtest_path, parse_dates=['date'])

    # Summary backtest
    backtest_path = favar_dir / "favar_rolling_backtest.csv"
    if backtest_path.exists():
        results['summary'] = pd.read_csv(backtest_path)

    return results


def load_dma_results():
    """Load DMA/DMS results."""
    dma_dir = FORECAST_RESULTS_DIR / "dma"
    results = {}

    # Rolling backtest
    backtest_path = dma_dir / "dma_rolling_backtest.csv"
    if backtest_path.exists():
        results['backtest'] = pd.read_csv(backtest_path, parse_dates=['date'])

    # Weights
    weights_path = dma_dir / "dma_weights.csv"
    if weights_path.exists():
        results['weights'] = pd.read_csv(weights_path, parse_dates=['date'])

    return results


def load_baseline_forecasts():
    """Load baseline model forecasts for subsample analysis."""
    baseline_dir = DATA_DIR / "forecast_results"

    models = {
        'Naive': 'naive_forecast.csv',
        'ARIMA': 'arima_forecast.csv',
        'VECM': 'vecm_forecast.csv',
        'MS_VAR': 'ms_var_forecast.csv',
        'MS_VECM': 'ms_vecm_forecast.csv',
    }

    results = {}
    for model, filename in models.items():
        filepath = baseline_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath, parse_dates=['date'])
            results[model] = df

    return results


def load_unified_results():
    """Load unified rolling-origin summaries and forecasts."""
    summaries = {}
    forecasts = {}

    candidate_dirs = [
        UNIFIED_RESULTS_DIR,
        PROJECT_ROOT / "reserves_project" / "data" / "forecast_results_unified",
        Path.cwd() / "data" / "forecast_results_unified",
    ]

    unified_dir = None
    for candidate in candidate_dirs:
        if candidate.exists():
            unified_dir = candidate
            break

    if unified_dir is None:
        return summaries, forecasts

    for varset in ['parsimonious', 'bop', 'monetary', 'pca', 'full']:
        summary_path = unified_dir / f"rolling_origin_summary_{varset}.csv"
        if summary_path.exists():
            summaries[varset] = pd.read_csv(summary_path)

        forecast_path = unified_dir / f"rolling_origin_forecasts_{varset}.csv"
        if forecast_path.exists():
            forecasts[varset] = pd.read_csv(forecast_path, parse_dates=['forecast_date', 'forecast_origin'])

    return summaries, forecasts


def build_main_accuracy_from_unified(
    summaries: dict,
    mcs_summary: pd.DataFrame | None = None,
    varset: str = 'parsimonious',
    horizon: int = 1,
    split: str = 'test',
) -> pd.DataFrame | None:
    df = summaries.get(varset)
    if df is None or df.empty:
        return None

    df = df[(df['horizon'] == horizon) & (df['split'] == split)].copy()
    if df.empty:
        return None

    df = df.rename(columns={
        'model': 'Model',
        'rmse': 'RMSE',
        'mae': 'MAE',
        'mape': 'MAPE',
        'policy_loss': 'Policy_Loss',
    })

    df['Rank'] = df['RMSE'].rank(method='min').astype(int)
    df['In_MCS'] = False
    if mcs_summary is not None and not mcs_summary.empty:
        if 'Model' in mcs_summary.columns:
            in_mcs = mcs_summary.set_index('Model')['In_MCS'].to_dict()
            df['In_MCS'] = df['Model'].map(in_mcs).fillna(False)
    cols = ['Model', 'RMSE', 'MAE', 'MAPE', 'Policy_Loss', 'In_MCS', 'Rank']
    existing = [c for c in cols if c in df.columns]
    return df[existing]


def build_horizon_results_from_unified(
    summaries: dict,
    varset: str = 'parsimonious',
    split: str = 'test',
) -> pd.DataFrame | None:
    df = summaries.get(varset)
    if df is None or df.empty:
        return None

    df = df[df['split'] == split].copy()
    if df.empty:
        return None

    df = df.rename(columns={
        'model': 'Model',
        'horizon': 'Horizon',
        'rmse': 'RMSE',
    })
    return df[['Model', 'Horizon', 'RMSE']]


def build_varset_results_from_unified(
    summaries: dict,
    horizon: int = 1,
    split: str = 'test',
) -> pd.DataFrame | None:
    rows = []
    for varset, df in summaries.items():
        df = df[(df['horizon'] == horizon) & (df['split'] == split)].copy()
        if df.empty:
            continue

        varset_name = VARIABLE_SETS.get(varset, {}).get('name', varset.title())
        for _, row in df.iterrows():
            rows.append({
                'Variable_Set': varset_name,
                'Variable_Set_Key': varset,
                'Model': row['model'],
                'RMSE': row['rmse'],
            })

    if not rows:
        return None

    return pd.DataFrame(rows)


def build_subsample_results_from_unified(
    forecasts: dict,
    varset: str = 'parsimonious',
    horizon: int = 1,
) -> pd.DataFrame | None:
    df = forecasts.get(varset)
    if df is None or df.empty:
        return None

    df = df[df['horizon'] == horizon].copy()
    if df.empty:
        return None

    # Pivot to wide format for SubsampleAnalyzer
    wide = df.pivot_table(index='forecast_date', columns='model', values='forecast')
    actuals = df.groupby('forecast_date')['actual'].first()

    analyzer = SubsampleAnalyzer()
    results = analyzer.analyze(wide.reset_index().rename(columns={'forecast_date': 'date'}), actuals, date_col='date')
    return results


# =============================================================================
# Robustness Analysis Functions
# =============================================================================

def run_subsample_analysis(combination_backtest: pd.DataFrame, baseline_forecasts: dict):
    """Run subsample robustness analysis."""
    print("\n" + "="*60)
    print("SUBSAMPLE ROBUSTNESS ANALYSIS")
    print("="*60)

    # Prepare combined data
    if combination_backtest is not None and 'date' in combination_backtest.columns:
        df = combination_backtest.copy()
        df = df.set_index('date')

        # Ensure we have actuals
        if 'actual' not in df.columns:
            print("Warning: No actual values in combination backtest")
            return None

        actuals = df['actual']

        # Get forecast columns
        forecast_cols = [c for c in df.columns if c.startswith('combined_')]

        analyzer = SubsampleAnalyzer()
        results = analyzer.analyze(df[forecast_cols], actuals)

        print(f"\nAnalyzed {len(results['Model'].unique())} models across {len(results['Period'].unique())} periods")

        # Rankings
        rankings = analyzer.rank_by_period(results, metric='RMSE')
        print("\nModel rankings by period:")
        print(rankings)

        # Stability
        stability = compute_period_stability(results, metric='RMSE')
        print("\nModel stability:")
        print(stability.head(10))

        return results

    return None


def run_horizon_analysis(favar_results: dict, bvar_results: dict):
    """Run horizon robustness analysis."""
    print("\n" + "="*60)
    print("HORIZON ROBUSTNESS ANALYSIS")
    print("="*60)

    all_results = []

    # FAVAR horizons
    if favar_results:
        for h, df in favar_results.items():
            if isinstance(h, int) and 'actual' in df.columns:
                # Get valid forecasts
                fc_col = 'forecast' if 'forecast' in df.columns else 'forecast_point'
                if fc_col not in df.columns:
                    continue

                valid = ~(df['actual'].isna() | df[fc_col].isna())
                df_valid = df[valid]

                if len(df_valid) < 3:
                    continue

                actual = df_valid['actual'].values
                forecast = df_valid[fc_col].values
                errors = actual - forecast

                all_results.append({
                    'Model': 'FAVAR',
                    'Horizon': h,
                    'N_Obs': len(df_valid),
                    'RMSE': np.sqrt(np.mean(errors**2)),
                    'MAE': np.mean(np.abs(errors)),
                    'MAPE': np.mean(np.abs(errors / actual[actual != 0])) * 100,
                })

    # BVAR horizons (from parsimonious set)
    if 'parsimonious' in bvar_results:
        bvar_df = bvar_results['parsimonious']
        if 'horizon' in bvar_df.columns and 'split' in bvar_df.columns:
            test_df = bvar_df[bvar_df['split'] == 'test']

            for h in [1, 3, 6, 12]:
                h_df = test_df[test_df['horizon'] == h]

                fc_col = 'forecast_point' if 'forecast_point' in h_df.columns else 'forecast_mean'
                if fc_col not in h_df.columns or 'actual' not in h_df.columns:
                    continue

                valid = ~(h_df['actual'].isna() | h_df[fc_col].isna())
                h_valid = h_df[valid]

                if len(h_valid) < 3:
                    continue

                actual = h_valid['actual'].values
                forecast = h_valid[fc_col].values
                errors = actual - forecast

                all_results.append({
                    'Model': 'BVAR',
                    'Horizon': h,
                    'N_Obs': len(h_valid),
                    'RMSE': np.sqrt(np.mean(errors**2)),
                    'MAE': np.mean(np.abs(errors)),
                    'MAPE': np.mean(np.abs(errors / actual[actual != 0])) * 100 if (actual != 0).sum() > 0 else np.nan,
                })

    results_df = pd.DataFrame(all_results)

    if len(results_df) > 0:
        print(f"\nAnalyzed {len(results_df['Model'].unique())} models across {len(results_df['Horizon'].unique())} horizons")

        # Horizon comparison table
        horizon_table = horizon_comparison_table(results_df, metric='RMSE')
        print("\nRMSE by Horizon:")
        print(horizon_table)

        # Ranking stability
        stability = horizon_ranking_stability(results_df)
        print("\nRanking stability across horizons:")
        print(stability)

        # Deterioration
        deterioration = compute_forecast_deterioration(results_df)
        print("\nForecast deterioration:")
        print(deterioration)

        return results_df

    return None


def run_variable_set_analysis(bvar_results: dict):
    """Run variable set robustness analysis."""
    print("\n" + "="*60)
    print("VARIABLE SET ROBUSTNESS ANALYSIS")
    print("="*60)

    all_results = []

    for varset, df in bvar_results.items():
        if df is None or len(df) == 0:
            continue

        # Filter to test set and h=1
        if 'split' in df.columns:
            df = df[df['split'] == 'test']
        if 'horizon' in df.columns:
            df = df[df['horizon'] == 1]

        # Get forecast column
        fc_col = 'forecast_point' if 'forecast_point' in df.columns else 'forecast_mean'
        if fc_col not in df.columns or 'actual' not in df.columns:
            continue

        valid = ~(df['actual'].isna() | df[fc_col].isna())
        df_valid = df[valid]

        if len(df_valid) < 3:
            continue

        actual = df_valid['actual'].values
        forecast = df_valid[fc_col].values
        errors = actual - forecast

        all_results.append({
            'Model': 'BVAR',
            'Variable_Set': varset.title(),
            'Variable_Set_Key': varset,
            'N_Obs': len(df_valid),
            'RMSE': np.sqrt(np.mean(errors**2)),
            'MAE': np.mean(np.abs(errors)),
            'MAPE': np.mean(np.abs(errors / actual[actual != 0])) * 100 if (actual != 0).sum() > 0 else np.nan,
        })

    results_df = pd.DataFrame(all_results)

    if len(results_df) > 0:
        print(f"\nAnalyzed across {len(results_df['Variable_Set'].unique())} variable sets")
        print("\nRMSE by Variable Set:")
        print(results_df[['Variable_Set', 'RMSE', 'MAE', 'N_Obs']])

        # Sensitivity analysis
        sensitivity = compute_varset_sensitivity(
            results_df.pivot(index='Model', columns='Variable_Set', values='RMSE')
        )
        print("\nSensitivity to variable set:")
        print(sensitivity)

        return results_df

    return None


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main execution function."""
    print("="*70)
    print("PHASE 5 (SPEC 11): ROBUSTNESS TABLES AND ACADEMIC OUTPUT")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Load all data
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)

    # Statistical test results
    stat_results = load_statistical_test_results()
    print(f"Loaded statistical test results: {list(stat_results.keys())}")

    # Combination forecasts
    comb_results = load_combination_forecasts()
    print(f"Loaded combination results: {list(comb_results.keys())}")

    # BVAR results
    bvar_results = load_bvar_results()
    print(f"Loaded BVAR results for variable sets: {list(bvar_results.keys())}")

    # FAVAR results
    favar_results = load_favar_results()
    print(f"Loaded FAVAR results for horizons: {list(favar_results.keys())}")

    # DMA results
    dma_results = load_dma_results()
    print(f"Loaded DMA results: {list(dma_results.keys())}")

    # Unified rolling-origin results
    unified_summaries, unified_forecasts = load_unified_results()
    if unified_summaries:
        print(f"Loaded unified summaries: {list(unified_summaries.keys())}")
    if unified_forecasts:
        print(f"Loaded unified forecasts: {list(unified_forecasts.keys())}")

    # Baseline forecasts
    baseline_results = load_baseline_forecasts()
    print(f"Loaded baseline models: {list(baseline_results.keys())}")

    # Run robustness analyses
    subsample_results = None
    if unified_forecasts:
        subsample_results = build_subsample_results_from_unified(unified_forecasts)
    if subsample_results is None:
        subsample_results = run_subsample_analysis(
            comb_results.get('backtest'),
            baseline_results
        )

    horizon_results = None
    if unified_summaries:
        horizon_results = build_horizon_results_from_unified(unified_summaries)
    if horizon_results is None:
        horizon_results = run_horizon_analysis(favar_results, bvar_results)

    varset_results = None
    if unified_summaries:
        varset_results = build_varset_results_from_unified(unified_summaries)
    if varset_results is None:
        varset_results = run_variable_set_analysis(bvar_results)

    # Generate LaTeX tables
    print("\n" + "="*60)
    print("GENERATING LATEX TABLES")
    print("="*60)

    table_generator = LaTeXTableGenerator(TABLES_DIR)

    # Table 1: Main accuracy
    main_accuracy_df = None
    if unified_summaries:
        main_accuracy_df = build_main_accuracy_from_unified(
            unified_summaries,
            stat_results.get('mcs_summary'),
        )
    if main_accuracy_df is not None:
        latex = table_generator.generate_table1_accuracy(
            main_accuracy_df,
            pd.DataFrame()
        )
        path = table_generator.save_table(latex, 'table1_accuracy.tex')
        print(f"Generated (unified): {path}")
    elif 'mcs_summary' in stat_results and 'dm_stats' in stat_results:
        latex = table_generator.generate_table1_accuracy(
            stat_results['mcs_summary'],
            stat_results['dm_stats']
        )
        path = table_generator.save_table(latex, 'table1_accuracy.tex')
        print(f"Generated: {path}")

    # Table 2: DM tests
    if 'dm_stats' in stat_results and 'dm_pvalues' in stat_results:
        latex = table_generator.generate_table2_dm(
            stat_results['dm_stats'],
            stat_results['dm_pvalues']
        )
        path = table_generator.save_table(latex, 'table2_dm_tests.tex')
        print(f"Generated: {path}")

    # Table 3: MCS
    if 'mcs_results' in stat_results and 'mcs_summary' in stat_results:
        latex = table_generator.generate_table3_mcs(
            stat_results['mcs_results'],
            stat_results['mcs_summary']
        )
        path = table_generator.save_table(latex, 'table3_mcs.tex')
        print(f"Generated: {path}")

    # Table 4: Subsample robustness
    if subsample_results is not None and len(subsample_results) > 0:
        latex = table_generator.generate_table4_subsample(subsample_results)
        path = table_generator.save_table(latex, 'table4_subsample.tex')
        print(f"Generated: {path}")

    # Table 5: Horizon robustness
    if horizon_results is not None and len(horizon_results) > 0:
        latex = table_generator.generate_table5_horizon(horizon_results)
        path = table_generator.save_table(latex, 'table5_horizon.tex')
        print(f"Generated: {path}")

    # Table 6: Variable set robustness
    if varset_results is not None and len(varset_results) > 0:
        varset_pivot = varset_results.pivot(index='Model', columns='Variable_Set', values='RMSE')
        latex = table_generator.generate_table6_varset(varset_pivot)
        path = table_generator.save_table(latex, 'table6_varset.tex')
        print(f"Generated: {path}")

    # Generate figures
    print("\n" + "="*60)
    print("GENERATING PUBLICATION FIGURES")
    print("="*60)

    fig_generator = FigureGenerator(FIGURES_DIR)

    # Forecast comparison
    if 'backtest' in comb_results:
        backtest = comb_results['backtest']
        if 'actual' in backtest.columns:
            actual = backtest.set_index('date')['actual']
            forecast_cols = [c for c in backtest.columns if c.startswith('combined_')]
            forecasts = {c.replace('combined_', ''): backtest.set_index('date')[c] for c in forecast_cols}

            path = fig_generator.forecast_comparison(actual, forecasts, top_n=4)
            print(f"Generated: {path}")

    # DMA weight evolution
    if 'weights' in dma_results:
        path = fig_generator.dma_weight_evolution(dma_results['weights'], top_n=5)
        print(f"Generated: {path}")

    # Subsample bar chart
    if subsample_results is not None and len(subsample_results) > 0:
        path = fig_generator.subsample_bar_chart(subsample_results)
        print(f"Generated: {path}")

    # Horizon deterioration
    if horizon_results is not None and len(horizon_results) > 0:
        path = fig_generator.horizon_deterioration(horizon_results)
        print(f"Generated: {path}")

    # Compile paper statistics
    print("\n" + "="*60)
    print("COMPILING PAPER STATISTICS")
    print("="*60)

    stats = compile_paper_statistics(
        mcs_summary=stat_results.get('mcs_summary'),
        mcs_results=stat_results.get('mcs_results'),
        dm_stats=stat_results.get('dm_stats'),
        dm_pvalues=stat_results.get('dm_pvalues'),
        subsample_results=subsample_results,
        horizon_results=horizon_results,
        varset_results=varset_results,
        combination_summary=comb_results.get('summary'),
    )

    # Save statistics
    stats_path = SUMMARY_DIR / "paper_statistics.json"
    save_statistics(stats, stats_path)
    print(f"Saved statistics to: {stats_path}")

    # Generate summary
    summary = generate_statistics_summary(stats)
    summary_path = SUMMARY_DIR / "statistics_summary.md"
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"Saved summary to: {summary_path}")

    # Save CSV versions
    if subsample_results is not None:
        subsample_results.to_csv(SUMMARY_DIR / "subsample_results.csv", index=False)

    if horizon_results is not None:
        horizon_results.to_csv(SUMMARY_DIR / "horizon_results.csv", index=False)

    if varset_results is not None:
        varset_results.to_csv(SUMMARY_DIR / "varset_results.csv", index=False)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(summary)

    print("\n" + "="*60)
    print("PHASE 5 COMPLETE")
    print("="*60)
    print(f"Tables saved to: {TABLES_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"Statistics saved to: {SUMMARY_DIR}")

    config = {
        "output_dir": str(OUTPUT_DIR),
        "tables_dir": str(TABLES_DIR),
        "figures_dir": str(FIGURES_DIR),
        "summary_dir": str(SUMMARY_DIR),
        "uses_unified_results": bool(unified_summaries),
        "uses_unified_tests": STATISTICAL_TESTS_UNIFIED_DIR.exists(),
    }
    write_run_manifest(OUTPUT_DIR, config)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
