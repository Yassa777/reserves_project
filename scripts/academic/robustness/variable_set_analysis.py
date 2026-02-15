"""
Variable Set Robustness Analysis
=================================

Analyzes how model performance varies across the 5 variable sets:
- Parsimonious
- BOP (Balance of Payments)
- Monetary
- PCA (Principal Components)
- Full

Author: Academic Forecasting Pipeline
Date: 2026-02-11
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from scipy import stats


# Variable set configurations
VARIABLE_SETS = {
    'parsimonious': {
        'name': 'Parsimonious',
        'description': 'Minimal set with key drivers',
        'variables': ['exchange_rate', 'trade_balance'],
    },
    'bop': {
        'name': 'BOP',
        'description': 'Balance of payments focused',
        'variables': ['exports', 'imports', 'fdi', 'remittances'],
    },
    'monetary': {
        'name': 'Monetary',
        'description': 'Monetary policy focused',
        'variables': ['interest_rate', 'm2_growth'],
    },
    'pca': {
        'name': 'PCA',
        'description': 'Principal component factors',
        'variables': ['pc1', 'pc2', 'pc3'],
    },
    'full': {
        'name': 'Full',
        'description': 'Complete set of all variables',
        'variables': ['all'],
    },
}


class VariableSetAnalyzer:
    """Analyzes forecast robustness across variable sets."""

    def __init__(self, variable_sets: Dict = None):
        """
        Initialize analyzer.

        Parameters
        ----------
        variable_sets : dict, optional
            Variable set configurations
        """
        self.variable_sets = variable_sets or VARIABLE_SETS

    def load_variable_set_results(
        self,
        base_path: Path,
        model_name: str,
        file_pattern: str = '{model}_rolling_backtest_{varset}.csv'
    ) -> Dict[str, pd.DataFrame]:
        """
        Load forecast results for different variable sets.

        Parameters
        ----------
        base_path : Path
            Base directory
        model_name : str
            Model name (e.g., 'bvar')
        file_pattern : str
            File pattern with {model} and {varset} placeholders

        Returns
        -------
        dict
            Variable set name -> DataFrame
        """
        results = {}

        for varset_key, varset_info in self.variable_sets.items():
            filename = file_pattern.format(model=model_name, varset=varset_key)
            filepath = base_path / filename

            if filepath.exists():
                df = pd.read_csv(filepath)
                if 'forecast_origin' in df.columns:
                    df['date'] = pd.to_datetime(df['forecast_date'])
                elif 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])

                results[varset_key] = df

        return results

    def compare_variable_sets(
        self,
        varset_results: Dict[str, pd.DataFrame],
        actual_col: str = 'actual',
        forecast_col: str = 'forecast_point',
        horizon_filter: int = None
    ) -> pd.DataFrame:
        """
        Compare performance across variable sets.

        Parameters
        ----------
        varset_results : dict
            Variable set name -> forecast DataFrame
        actual_col : str
            Column name for actual values
        forecast_col : str
            Column name for forecast values
        horizon_filter : int, optional
            Only include forecasts at this horizon

        Returns
        -------
        pd.DataFrame
            Performance comparison
        """
        comparison = []

        for varset_key, df in varset_results.items():
            # Apply horizon filter
            if horizon_filter is not None and 'horizon' in df.columns:
                df = df[df['horizon'] == horizon_filter]

            # Filter to test set if available
            if 'split' in df.columns:
                df = df[df['split'] == 'test']

            # Handle different column names
            fc_col = forecast_col
            if forecast_col not in df.columns:
                if 'forecast_mean' in df.columns:
                    fc_col = 'forecast_mean'
                elif 'forecast' in df.columns:
                    fc_col = 'forecast'
                else:
                    continue

            if actual_col not in df.columns:
                continue

            # Remove invalid rows
            valid = ~(df[actual_col].isna() | df[fc_col].isna())
            df_valid = df[valid]

            if len(df_valid) < 3:
                continue

            actual = df_valid[actual_col].values
            forecast = df_valid[fc_col].values

            # Compute metrics
            errors = actual - forecast
            abs_errors = np.abs(errors)
            sq_errors = errors ** 2

            nonzero = actual != 0
            mape = np.mean(np.abs(errors[nonzero] / actual[nonzero])) * 100 if nonzero.sum() > 0 else np.nan

            varset_name = self.variable_sets[varset_key]['name']

            comparison.append({
                'Variable_Set': varset_name,
                'Variable_Set_Key': varset_key,
                'N_Obs': len(df_valid),
                'RMSE': np.sqrt(np.mean(sq_errors)),
                'MAE': np.mean(abs_errors),
                'MAPE': mape,
                'MSE': np.mean(sq_errors),
                'MedAE': np.median(abs_errors),
                'Bias': np.mean(errors),
            })

        return pd.DataFrame(comparison)


def variable_set_comparison(
    all_models_varsets: Dict[str, Dict[str, pd.DataFrame]],
    metric: str = 'RMSE'
) -> pd.DataFrame:
    """
    Compare all models across all variable sets.

    Parameters
    ----------
    all_models_varsets : dict
        Model name -> {varset -> DataFrame}
    metric : str
        Metric for comparison

    Returns
    -------
    pd.DataFrame
        Comparison table (models x variable sets)
    """
    analyzer = VariableSetAnalyzer()
    all_results = []

    for model_name, varset_data in all_models_varsets.items():
        model_comparison = analyzer.compare_variable_sets(varset_data)
        model_comparison['Model'] = model_name
        all_results.append(model_comparison)

    if not all_results:
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)

    # Pivot to get models x variable sets
    pivot = combined.pivot(index='Model', columns='Variable_Set', values=metric)

    return pivot


def ranking_consistency_test(
    varset_comparison: pd.DataFrame
) -> Dict:
    """
    Test for consistency of model rankings across variable sets.

    Uses Kendall's W (coefficient of concordance) to assess
    whether model rankings are consistent across variable sets.

    Parameters
    ----------
    varset_comparison : pd.DataFrame
        Comparison table from variable_set_comparison()

    Returns
    -------
    dict
        Test results including Kendall's W and p-value
    """
    # Convert values to ranks within each column
    rank_matrix = varset_comparison.rank(axis=0)

    n = rank_matrix.shape[0]  # number of models
    k = rank_matrix.shape[1]  # number of variable sets

    if n < 2 or k < 2:
        return {'error': 'Insufficient data'}

    # Compute row sums (total rank for each model)
    R = rank_matrix.sum(axis=1)

    # Mean rank sum
    R_bar = R.mean()

    # Sum of squared deviations
    S = np.sum((R - R_bar) ** 2)

    # Kendall's W
    W = (12 * S) / (k ** 2 * (n ** 3 - n))

    # Chi-square approximation for significance
    chi2 = k * (n - 1) * W
    df = n - 1
    p_value = 1 - stats.chi2.cdf(chi2, df)

    return {
        'kendall_w': W,
        'chi_square': chi2,
        'df': df,
        'p_value': p_value,
        'n_models': n,
        'n_variable_sets': k,
        'interpretation': interpret_kendall_w(W),
    }


def interpret_kendall_w(w: float) -> str:
    """Interpret Kendall's W coefficient."""
    if w >= 0.9:
        return 'Very strong agreement'
    elif w >= 0.7:
        return 'Strong agreement'
    elif w >= 0.5:
        return 'Moderate agreement'
    elif w >= 0.3:
        return 'Fair agreement'
    else:
        return 'Poor agreement'


def compute_varset_sensitivity(
    varset_comparison: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute sensitivity of each model to variable set choice.

    Parameters
    ----------
    varset_comparison : pd.DataFrame
        Comparison table from variable_set_comparison()

    Returns
    -------
    pd.DataFrame
        Sensitivity metrics for each model
    """
    sensitivity = []

    for model in varset_comparison.index:
        values = varset_comparison.loc[model].values
        valid_values = values[~np.isnan(values)]

        if len(valid_values) < 2:
            continue

        sensitivity.append({
            'Model': model,
            'Mean': np.mean(valid_values),
            'Std': np.std(valid_values),
            'CV': np.std(valid_values) / np.mean(valid_values) if np.mean(valid_values) != 0 else np.nan,
            'Min': np.min(valid_values),
            'Max': np.max(valid_values),
            'Range': np.max(valid_values) - np.min(valid_values),
            'Best_Worst_Ratio': np.max(valid_values) / np.min(valid_values) if np.min(valid_values) != 0 else np.nan,
            'N_VarSets': len(valid_values),
        })

    return pd.DataFrame(sensitivity).sort_values('CV')


def identify_best_varset_per_model(
    varset_comparison: pd.DataFrame
) -> pd.DataFrame:
    """
    Identify best variable set for each model.

    Parameters
    ----------
    varset_comparison : pd.DataFrame
        Comparison table (lower is better)

    Returns
    -------
    pd.DataFrame
        Best variable set for each model with metrics
    """
    results = []

    for model in varset_comparison.index:
        model_values = varset_comparison.loc[model]
        valid = model_values.dropna()

        if len(valid) == 0:
            continue

        best_varset = valid.idxmin()
        best_value = valid.min()
        worst_varset = valid.idxmax()
        worst_value = valid.max()

        results.append({
            'Model': model,
            'Best_VarSet': best_varset,
            'Best_Value': best_value,
            'Worst_VarSet': worst_varset,
            'Worst_Value': worst_value,
            'Improvement_Pct': (worst_value - best_value) / worst_value * 100 if worst_value != 0 else np.nan,
        })

    return pd.DataFrame(results).sort_values('Best_Value')
