"""
Paper Statistics Compilation
=============================

Compiles all key numbers for inline text references in the academic paper.

Author: Academic Forecasting Pipeline
Date: 2026-02-11
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
from datetime import datetime


def compile_paper_statistics(
    mcs_summary: pd.DataFrame,
    mcs_results: Dict,
    dm_stats: pd.DataFrame,
    dm_pvalues: pd.DataFrame,
    subsample_results: pd.DataFrame,
    horizon_results: pd.DataFrame,
    varset_results: pd.DataFrame,
    combination_summary: Dict = None
) -> Dict[str, Any]:
    """
    Compile all key statistics for the paper.

    Parameters
    ----------
    mcs_summary : pd.DataFrame
        MCS summary with RMSE, MAE, rankings
    mcs_results : dict
        MCS results with membership lists
    dm_stats : pd.DataFrame
        DM test statistics matrix
    dm_pvalues : pd.DataFrame
        DM p-values matrix
    subsample_results : pd.DataFrame
        Subsample robustness results
    horizon_results : pd.DataFrame
        Horizon robustness results
    varset_results : pd.DataFrame
        Variable set robustness results
    combination_summary : dict, optional
        Forecast combination summary

    Returns
    -------
    dict
        All statistics organized by section
    """
    stats = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'version': '1.0',
        },
        'sample': {},
        'models': {},
        'accuracy': {},
        'statistical_tests': {},
        'robustness': {},
        'combinations': {},
    }

    # Sample information
    if mcs_summary is not None:
        stats['sample']['n_models'] = len(mcs_summary)
        stats['sample']['n_observations'] = mcs_results.get('n_obs', 0) if mcs_results else 0

    # Model rankings
    if mcs_summary is not None:
        stats['models']['all_models'] = mcs_summary['Model'].tolist()
        if 'Rank' in mcs_summary.columns:
            best_model = mcs_summary.loc[mcs_summary['Rank'] == 1, 'Model'].values
            stats['models']['best_by_rmse'] = best_model[0] if len(best_model) > 0 else None

    # Accuracy metrics
    if mcs_summary is not None and 'RMSE' in mcs_summary.columns:
        best_idx = mcs_summary['RMSE'].idxmin()
        worst_idx = mcs_summary['RMSE'].idxmax()

        stats['accuracy']['best_model'] = mcs_summary.loc[best_idx, 'Model']
        stats['accuracy']['best_rmse'] = float(mcs_summary.loc[best_idx, 'RMSE'])
        stats['accuracy']['best_mae'] = float(mcs_summary.loc[best_idx, 'MAE'])

        stats['accuracy']['worst_model'] = mcs_summary.loc[worst_idx, 'Model']
        stats['accuracy']['worst_rmse'] = float(mcs_summary.loc[worst_idx, 'RMSE'])

        # Improvement percentage
        improvement = (stats['accuracy']['worst_rmse'] - stats['accuracy']['best_rmse']) / stats['accuracy']['worst_rmse'] * 100
        stats['accuracy']['best_vs_worst_improvement_pct'] = float(improvement)

        # Naive benchmark comparison
        if 'Naive' in mcs_summary['Model'].values:
            naive_rmse = mcs_summary[mcs_summary['Model'] == 'Naive']['RMSE'].values[0]
            stats['accuracy']['naive_rmse'] = float(naive_rmse)
            stats['accuracy']['best_vs_naive_improvement_pct'] = float(
                (naive_rmse - stats['accuracy']['best_rmse']) / naive_rmse * 100
            )

    # MCS results
    if mcs_results is not None:
        stats['statistical_tests']['mcs_alpha'] = mcs_results.get('alpha', 0.10)
        stats['statistical_tests']['mcs_members'] = mcs_results.get('mcs', [])
        stats['statistical_tests']['mcs_size'] = len(mcs_results.get('mcs', []))
        stats['statistical_tests']['n_eliminated'] = len(mcs_results.get('eliminated', []))

    # DM tests
    if dm_stats is not None and dm_pvalues is not None:
        # Count significant pairwise differences
        n_pairs = dm_stats.shape[0] * (dm_stats.shape[0] - 1) / 2
        significant_10 = (dm_pvalues < 0.10).sum().sum() / 2  # divide by 2 for symmetric matrix
        significant_05 = (dm_pvalues < 0.05).sum().sum() / 2
        significant_01 = (dm_pvalues < 0.01).sum().sum() / 2

        stats['statistical_tests']['dm_n_pairs'] = int(n_pairs)
        stats['statistical_tests']['dm_significant_10pct'] = int(significant_10)
        stats['statistical_tests']['dm_significant_05pct'] = int(significant_05)
        stats['statistical_tests']['dm_significant_01pct'] = int(significant_01)

    # Subsample robustness
    if subsample_results is not None and len(subsample_results) > 0:
        # Find most consistent model
        if 'Model' in subsample_results.columns and 'RMSE' in subsample_results.columns:
            pivot = subsample_results.pivot(index='Model', columns='Period', values='RMSE')
            avg_rmse = pivot.mean(axis=1)
            std_rmse = pivot.std(axis=1)
            cv = std_rmse / avg_rmse

            most_consistent = cv.idxmin()
            least_consistent = cv.idxmax()

            stats['robustness']['subsample'] = {
                'most_consistent_model': most_consistent,
                'most_consistent_cv': float(cv[most_consistent]),
                'least_consistent_model': least_consistent,
                'least_consistent_cv': float(cv[least_consistent]),
                'n_periods': int(pivot.shape[1]),
            }

    # Horizon robustness
    if horizon_results is not None and len(horizon_results) > 0:
        # Best at each horizon
        horizon_best = {}
        for h in horizon_results['Horizon'].unique():
            h_data = horizon_results[horizon_results['Horizon'] == h]
            best_idx = h_data['RMSE'].idxmin()
            horizon_best[f'h{int(h)}'] = h_data.loc[best_idx, 'Model']

        stats['robustness']['horizon'] = {
            'best_by_horizon': horizon_best,
            'horizons_tested': sorted(horizon_results['Horizon'].unique().tolist()),
        }

        # Deterioration statistics
        if 1 in horizon_results['Horizon'].values and 12 in horizon_results['Horizon'].values:
            h1_best = horizon_results[horizon_results['Horizon'] == 1]['RMSE'].min()
            h12_best = horizon_results[horizon_results['Horizon'] == 12]['RMSE'].min()
            stats['robustness']['horizon']['deterioration_ratio'] = float(h12_best / h1_best)

    # Variable set robustness
    if varset_results is not None and len(varset_results) > 0:
        if isinstance(varset_results, pd.DataFrame):
            # Calculate sensitivity
            if 'Variable_Set' in varset_results.columns:
                pivot = varset_results.pivot(index='Model', columns='Variable_Set', values='RMSE')
            else:
                pivot = varset_results

            cv = pivot.std(axis=1) / pivot.mean(axis=1)
            most_robust = cv.idxmin()

            stats['robustness']['variable_set'] = {
                'most_robust_model': most_robust,
                'n_variable_sets': int(pivot.shape[1]),
                'variable_sets_tested': pivot.columns.tolist(),
            }

    # Combination forecasts
    if combination_summary is not None:
        stats['combinations'] = {
            'best_method_validation': combination_summary.get('validation_period', {}).get('best_combination_method'),
            'best_method_test': combination_summary.get('test_period', {}).get('best_combination_method'),
            'improvement_validation': combination_summary.get('validation_period', {}).get('improvement_pct'),
            'improvement_test': combination_summary.get('test_period', {}).get('improvement_pct'),
        }

    return stats


def format_inline_numbers(stats: Dict) -> Dict[str, str]:
    """
    Format statistics for inline citation in paper text.

    Returns dictionary with LaTeX-ready formatted strings.

    Parameters
    ----------
    stats : dict
        Statistics from compile_paper_statistics()

    Returns
    -------
    dict
        Formatted strings for inline use
    """
    formatted = {}

    # Sample size
    if 'sample' in stats:
        formatted['n_models'] = str(stats['sample'].get('n_models', 'N/A'))
        formatted['n_obs'] = str(stats['sample'].get('n_observations', 'N/A'))

    # Best model performance
    if 'accuracy' in stats:
        acc = stats['accuracy']
        if 'best_model' in acc:
            formatted['best_model'] = acc['best_model']
        if 'best_rmse' in acc:
            formatted['best_rmse'] = f"{acc['best_rmse']:.1f}"
        if 'best_vs_naive_improvement_pct' in acc:
            formatted['improvement_vs_naive'] = f"{acc['best_vs_naive_improvement_pct']:.1f}\\%"

    # MCS
    if 'statistical_tests' in stats:
        st = stats['statistical_tests']
        if 'mcs_size' in st:
            formatted['mcs_size'] = str(st['mcs_size'])
        if 'mcs_members' in st:
            formatted['mcs_members'] = ', '.join(st['mcs_members'])

    # Robustness
    if 'robustness' in stats:
        rob = stats['robustness']

        if 'subsample' in rob:
            formatted['most_consistent'] = rob['subsample'].get('most_consistent_model', 'N/A')
            cv = rob['subsample'].get('most_consistent_cv')
            if cv is not None:
                formatted['most_consistent_cv'] = f"{cv:.3f}"

        if 'horizon' in rob:
            dr = rob['horizon'].get('deterioration_ratio')
            if dr is not None:
                formatted['deterioration_ratio'] = f"{dr:.2f}"

    return formatted


def save_statistics(
    stats: Dict,
    output_path: Path,
    also_save_formatted: bool = True
) -> None:
    """
    Save statistics to JSON file.

    Parameters
    ----------
    stats : dict
        Statistics dictionary
    output_path : Path
        Output file path
    also_save_formatted : bool
        If True, also save formatted version
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save main stats
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)

    # Save formatted version
    if also_save_formatted:
        formatted = format_inline_numbers(stats)
        formatted_path = output_path.parent / f"{output_path.stem}_formatted.json"
        with open(formatted_path, 'w') as f:
            json.dump(formatted, f, indent=2)


def generate_statistics_summary(stats: Dict) -> str:
    """
    Generate a text summary of key statistics.

    Parameters
    ----------
    stats : dict
        Statistics dictionary

    Returns
    -------
    str
        Markdown-formatted summary
    """
    lines = ['# Key Statistics Summary', '']

    # Sample
    if 'sample' in stats:
        lines.append('## Sample')
        lines.append(f"- Number of models: {stats['sample'].get('n_models', 'N/A')}")
        lines.append(f"- Number of observations: {stats['sample'].get('n_observations', 'N/A')}")
        lines.append('')

    # Accuracy
    if 'accuracy' in stats:
        acc = stats['accuracy']
        lines.append('## Forecast Accuracy')
        lines.append(f"- Best model: {acc.get('best_model', 'N/A')}")
        lines.append(f"- Best RMSE: {acc.get('best_rmse', 'N/A'):.1f}")
        lines.append(f"- Improvement vs Naive: {acc.get('best_vs_naive_improvement_pct', 'N/A'):.1f}%")
        lines.append('')

    # Statistical tests
    if 'statistical_tests' in stats:
        st = stats['statistical_tests']
        lines.append('## Statistical Tests')
        lines.append(f"- MCS size (90%): {st.get('mcs_size', 'N/A')}")
        lines.append(f"- MCS members: {', '.join(st.get('mcs_members', []))}")
        lines.append(f"- DM significant pairs (10%): {st.get('dm_significant_10pct', 'N/A')}")
        lines.append('')

    # Robustness
    if 'robustness' in stats:
        rob = stats['robustness']
        lines.append('## Robustness Analysis')

        if 'subsample' in rob:
            lines.append(f"- Most consistent model: {rob['subsample'].get('most_consistent_model', 'N/A')}")

        if 'horizon' in rob:
            lines.append(f"- Deterioration ratio (h12/h1): {rob['horizon'].get('deterioration_ratio', 'N/A'):.2f}")

        lines.append('')

    return '\n'.join(lines)
