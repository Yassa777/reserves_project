"""
Horizon Robustness Analysis
============================

Compares forecast performance across different horizons (h=1,3,6,12 months).

Author: Academic Forecasting Pipeline
Date: 2026-02-11
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from scipy import stats


class HorizonAnalyzer:
    """Analyzes forecast performance across different horizons."""

    def __init__(self, horizons: List[int] = None):
        """
        Initialize horizon analyzer.

        Parameters
        ----------
        horizons : list of int, optional
            Horizons to analyze. Defaults to [1, 3, 6, 12].
        """
        self.horizons = horizons or [1, 3, 6, 12]

    def load_horizon_forecasts(
        self,
        base_path: Path,
        model_name: str,
        file_pattern: str = '{model}_rolling_backtest_h{h}.csv'
    ) -> Dict[int, pd.DataFrame]:
        """
        Load forecast files for different horizons.

        Parameters
        ----------
        base_path : Path
            Base directory containing forecast files
        model_name : str
            Model name for file pattern
        file_pattern : str
            Pattern with {model} and {h} placeholders

        Returns
        -------
        dict
            Dictionary mapping horizon to forecast DataFrame
        """
        horizon_data = {}

        for h in self.horizons:
            filename = file_pattern.format(model=model_name, h=h)
            filepath = base_path / filename

            if filepath.exists():
                df = pd.read_csv(filepath, parse_dates=['date'])
                horizon_data[h] = df

        return horizon_data

    def analyze_by_horizon(
        self,
        horizon_forecasts: Dict[int, pd.DataFrame],
        actual_col: str = 'actual',
        forecast_col: str = 'forecast'
    ) -> pd.DataFrame:
        """
        Compute metrics for each horizon.

        Parameters
        ----------
        horizon_forecasts : dict
            Dictionary mapping horizon to forecast DataFrame
        actual_col : str
            Column name for actual values
        forecast_col : str
            Column name for forecast values

        Returns
        -------
        pd.DataFrame
            Metrics by horizon
        """
        results = []

        for h, df in horizon_forecasts.items():
            if actual_col not in df.columns or forecast_col not in df.columns:
                # Try alternative column names
                if 'forecast_point' in df.columns:
                    forecast_col = 'forecast_point'
                elif 'forecast_mean' in df.columns:
                    forecast_col = 'forecast_mean'

            if actual_col not in df.columns:
                continue

            # Filter to valid observations
            valid = ~(df[actual_col].isna() | df[forecast_col].isna())
            df_valid = df[valid]

            if len(df_valid) < 3:
                continue

            actual = df_valid[actual_col].values
            forecast = df_valid[forecast_col].values

            # Compute metrics
            errors = actual - forecast
            abs_errors = np.abs(errors)
            sq_errors = errors ** 2

            # MAPE with protection
            nonzero = actual != 0
            mape = np.mean(np.abs(errors[nonzero] / actual[nonzero])) * 100 if nonzero.sum() > 0 else np.nan

            # Theil U
            naive_errors = actual[1:] - actual[:-1]
            theil_u = np.sqrt(np.mean(sq_errors[1:])) / np.sqrt(np.mean(naive_errors**2)) if len(naive_errors) > 0 else np.nan

            results.append({
                'Horizon': h,
                'N_Obs': len(df_valid),
                'RMSE': np.sqrt(np.mean(sq_errors)),
                'MAE': np.mean(abs_errors),
                'MAPE': mape,
                'MSE': np.mean(sq_errors),
                'MedAE': np.median(abs_errors),
                'Theil_U': theil_u,
                'Bias': np.mean(errors),
            })

        return pd.DataFrame(results)

    def compare_horizons(
        self,
        all_model_horizons: Dict[str, Dict[int, pd.DataFrame]],
        actual_col: str = 'actual',
        forecast_col: str = 'forecast'
    ) -> pd.DataFrame:
        """
        Compare all models across all horizons.

        Parameters
        ----------
        all_model_horizons : dict
            Dictionary mapping model name to horizon forecasts dict
        actual_col : str
            Column name for actual values
        forecast_col : str
            Column name for forecast values

        Returns
        -------
        pd.DataFrame
            Complete comparison table
        """
        all_results = []

        for model_name, horizon_data in all_model_horizons.items():
            model_results = self.analyze_by_horizon(horizon_data, actual_col, forecast_col)
            model_results['Model'] = model_name
            all_results.append(model_results)

        if not all_results:
            return pd.DataFrame()

        combined = pd.concat(all_results, ignore_index=True)

        return combined


def horizon_comparison_table(
    horizon_results: pd.DataFrame,
    metric: str = 'RMSE',
    normalize_to_h1: bool = True
) -> pd.DataFrame:
    """
    Create formatted horizon comparison table.

    Parameters
    ----------
    horizon_results : pd.DataFrame
        Output from HorizonAnalyzer.compare_horizons()
    metric : str
        Metric to use for comparison
    normalize_to_h1 : bool
        If True, express values relative to h=1

    Returns
    -------
    pd.DataFrame
        Formatted comparison table (models x horizons)
    """
    # Pivot to get models x horizons
    pivot = horizon_results.pivot(index='Model', columns='Horizon', values=metric)

    if normalize_to_h1 and 1 in pivot.columns:
        # Normalize relative to h=1
        h1_values = pivot[1]
        for col in pivot.columns:
            pivot[col] = pivot[col] / h1_values

    # Sort by h=1 performance
    if 1 in pivot.columns:
        pivot = pivot.sort_values(1)

    return pivot


def horizon_ranking_stability(
    horizon_results: pd.DataFrame,
    metric: str = 'RMSE'
) -> pd.DataFrame:
    """
    Analyze how model rankings change across horizons.

    Parameters
    ----------
    horizon_results : pd.DataFrame
        Output from compare_horizons()
    metric : str
        Metric to analyze

    Returns
    -------
    pd.DataFrame
        Ranking stability analysis
    """
    # Compute ranks within each horizon
    rank_results = []

    for h in horizon_results['Horizon'].unique():
        h_data = horizon_results[horizon_results['Horizon'] == h].copy()
        h_data['Rank'] = h_data[metric].rank(method='min')

        for _, row in h_data.iterrows():
            rank_results.append({
                'Model': row['Model'],
                'Horizon': h,
                'Rank': int(row['Rank']),
                metric: row[metric],
            })

    rank_df = pd.DataFrame(rank_results)

    # Pivot to get rank table
    rank_pivot = rank_df.pivot(index='Model', columns='Horizon', values='Rank')

    # Compute stability metrics
    stability = []
    for model in rank_pivot.index:
        ranks = rank_pivot.loc[model].values
        valid_ranks = ranks[~np.isnan(ranks)]

        if len(valid_ranks) < 2:
            continue

        stability.append({
            'Model': model,
            'Mean_Rank': np.mean(valid_ranks),
            'Std_Rank': np.std(valid_ranks),
            'Min_Rank': np.min(valid_ranks),
            'Max_Rank': np.max(valid_ranks),
            'Rank_Range': np.max(valid_ranks) - np.min(valid_ranks),
            'N_Horizons': len(valid_ranks),
        })

    return pd.DataFrame(stability).sort_values('Mean_Rank')


def compute_forecast_deterioration(
    horizon_results: pd.DataFrame,
    metric: str = 'RMSE'
) -> pd.DataFrame:
    """
    Compute how forecast accuracy deteriorates with horizon.

    Parameters
    ----------
    horizon_results : pd.DataFrame
        Output from compare_horizons()
    metric : str
        Metric to analyze

    Returns
    -------
    pd.DataFrame
        Deterioration statistics for each model
    """
    deterioration = []

    for model in horizon_results['Model'].unique():
        model_data = horizon_results[horizon_results['Model'] == model].copy()
        model_data = model_data.sort_values('Horizon')

        if len(model_data) < 2:
            continue

        horizons = model_data['Horizon'].values
        values = model_data[metric].values

        # Fit linear trend
        if len(horizons) >= 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(horizons, values)

            # Compute deterioration rate (% increase per horizon step)
            h1_value = values[0] if horizons[0] == 1 else intercept + slope
            deterioration_rate = (slope / h1_value) * 100 if h1_value != 0 else np.nan

            deterioration.append({
                'Model': model,
                f'{metric}_h1': values[0],
                f'{metric}_h12': values[-1] if len(values) > 0 else np.nan,
                'Slope': slope,
                'R_squared': r_value ** 2,
                'Deterioration_Rate_Pct': deterioration_rate,
                'Ratio_h12_h1': values[-1] / values[0] if values[0] != 0 else np.nan,
            })

    return pd.DataFrame(deterioration).sort_values('Deterioration_Rate_Pct')


def test_horizon_differences(
    horizon_results: pd.DataFrame,
    model: str,
    metric: str = 'RMSE'
) -> Dict:
    """
    Statistical test for differences across horizons.

    Parameters
    ----------
    horizon_results : pd.DataFrame
        Output from compare_horizons()
    model : str
        Model name to test
    metric : str
        Metric to analyze

    Returns
    -------
    dict
        Test results
    """
    model_data = horizon_results[horizon_results['Model'] == model].copy()
    model_data = model_data.sort_values('Horizon')

    if len(model_data) < 2:
        return {'error': 'Insufficient data'}

    values = model_data[metric].values
    horizons = model_data['Horizon'].values

    # Trend test (linear regression significance)
    slope, intercept, r_value, p_value, std_err = stats.linregress(horizons, values)

    return {
        'model': model,
        'metric': metric,
        'n_horizons': len(horizons),
        'slope': slope,
        'r_squared': r_value ** 2,
        'p_value': p_value,
        'significant_trend': p_value < 0.05,
    }
