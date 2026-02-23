"""
Subsample Robustness Analysis
=============================

Analyzes model performance across different time periods:
- Pre-crisis (2012-2018)
- Crisis period (2019-2022)
- Post-default (2023-2025)
- COVID period (2020-2021)

Author: Academic Forecasting Pipeline
Date: 2026-02-11
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SubsamplePeriod:
    """Defines a subsample period for analysis."""
    name: str
    start_date: str
    end_date: str
    description: str


# Define standard subsample periods for Sri Lanka reserves analysis
STANDARD_PERIODS = {
    'pre_crisis': SubsamplePeriod(
        name='Pre-Crisis',
        start_date='2012-01-01',
        end_date='2018-12-31',
        description='Period before economic crisis'
    ),
    'crisis': SubsamplePeriod(
        name='Crisis',
        start_date='2019-01-01',
        end_date='2022-12-31',
        description='Economic crisis and external shocks'
    ),
    'post_default': SubsamplePeriod(
        name='Post-Default',
        start_date='2023-01-01',
        end_date='2025-12-31',
        description='Period after sovereign default'
    ),
    'covid': SubsamplePeriod(
        name='COVID',
        start_date='2020-01-01',
        end_date='2021-12-31',
        description='COVID-19 pandemic period'
    ),
    'full': SubsamplePeriod(
        name='Full Sample',
        start_date='2012-01-01',
        end_date='2025-12-31',
        description='Complete sample period'
    ),
}


class SubsampleAnalyzer:
    """Analyzes forecast performance across different time periods."""

    def __init__(self, periods: Optional[Dict[str, SubsamplePeriod]] = None):
        """
        Initialize subsample analyzer.

        Parameters
        ----------
        periods : dict, optional
            Dictionary of SubsamplePeriod objects. Uses STANDARD_PERIODS if None.
        """
        self.periods = periods or STANDARD_PERIODS

    def analyze(
        self,
        forecasts: pd.DataFrame,
        actuals: pd.Series,
        date_col: str = 'date'
    ) -> pd.DataFrame:
        """
        Analyze forecast performance across all subsample periods.

        Parameters
        ----------
        forecasts : pd.DataFrame
            DataFrame with date column and model forecast columns
        actuals : pd.Series
            Actual values with date index
        date_col : str
            Name of date column in forecasts DataFrame

        Returns
        -------
        pd.DataFrame
            Performance metrics for each model across all periods
        """
        results = []

        # Ensure date column is datetime
        if date_col in forecasts.columns:
            df = forecasts.copy()
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
        else:
            df = forecasts.copy()
            df.index = pd.to_datetime(df.index)

        # Align actuals
        if isinstance(actuals, pd.Series):
            actuals_aligned = actuals.copy()
            actuals_aligned.index = pd.to_datetime(actuals_aligned.index)
        else:
            actuals_aligned = pd.Series(actuals, index=df.index)

        # Get model columns (exclude non-forecast columns)
        exclude_cols = ['actual', 'split', 'horizon', 'error', 'abs_error', 'sq_error']
        model_cols = [c for c in df.columns if c not in exclude_cols and not c.startswith('weight_')]

        for period_key, period in self.periods.items():
            # Filter to period
            mask = (df.index >= period.start_date) & (df.index <= period.end_date)
            period_df = df[mask]
            period_actuals = actuals_aligned.reindex(period_df.index)

            if len(period_df) < 3:
                continue

            for model in model_cols:
                if model not in period_df.columns:
                    continue

                forecasts_model = period_df[model].values
                actual_vals = period_actuals.values

                # Remove NaN values
                valid = ~(np.isnan(forecasts_model) | np.isnan(actual_vals))
                if valid.sum() < 3:
                    continue

                fc = forecasts_model[valid]
                act = actual_vals[valid]

                # Compute metrics
                errors = act - fc
                metrics = compute_forecast_metrics(act, fc)
                metrics['Model'] = model
                metrics['Period'] = period.name
                metrics['Period_Key'] = period_key
                metrics['N_Obs'] = len(fc)
                metrics['Start'] = period_df.index[valid].min()
                metrics['End'] = period_df.index[valid].max()

                results.append(metrics)

        return pd.DataFrame(results)

    def rank_by_period(
        self,
        results: pd.DataFrame,
        metric: str = 'RMSE'
    ) -> pd.DataFrame:
        """
        Rank models by performance within each period.

        Parameters
        ----------
        results : pd.DataFrame
            Output from analyze()
        metric : str
            Metric to use for ranking (RMSE, MAE, MAPE)

        Returns
        -------
        pd.DataFrame
            Rankings table (models x periods)
        """
        rankings = []

        for period in results['Period'].unique():
            period_data = results[results['Period'] == period].copy()
            period_data['Rank'] = period_data[metric].rank(method='min')

            for _, row in period_data.iterrows():
                rankings.append({
                    'Model': row['Model'],
                    'Period': period,
                    'Rank': int(row['Rank']),
                    metric: row[metric],
                })

        rankings_df = pd.DataFrame(rankings)

        # Pivot to wide format
        pivot = rankings_df.pivot(index='Model', columns='Period', values='Rank')

        return pivot


def compute_forecast_metrics(
    actual: np.ndarray,
    forecast: np.ndarray
) -> Dict[str, float]:
    """
    Compute comprehensive forecast accuracy metrics.

    Parameters
    ----------
    actual : np.ndarray
        Actual values
    forecast : np.ndarray
        Forecast values

    Returns
    -------
    dict
        Dictionary of metrics
    """
    errors = actual - forecast
    abs_errors = np.abs(errors)
    sq_errors = errors ** 2

    # Avoid division by zero in MAPE
    nonzero_actual = actual != 0
    if nonzero_actual.sum() > 0:
        mape = np.mean(np.abs(errors[nonzero_actual] / actual[nonzero_actual])) * 100
    else:
        mape = np.nan

    # Theil's U statistic
    naive_errors = actual[1:] - actual[:-1]
    if len(naive_errors) > 0 and np.sqrt(np.mean(naive_errors**2)) > 0:
        theil_u = np.sqrt(np.mean(sq_errors[1:])) / np.sqrt(np.mean(naive_errors**2))
    else:
        theil_u = np.nan

    return {
        'RMSE': np.sqrt(np.mean(sq_errors)),
        'MAE': np.mean(abs_errors),
        'MAPE': mape,
        'MSE': np.mean(sq_errors),
        'MedAE': np.median(abs_errors),
        'Theil_U': theil_u,
        'Bias': np.mean(errors),
        'Std_Error': np.std(errors),
        'Min_Error': np.min(errors),
        'Max_Error': np.max(errors),
    }


def compute_subsample_metrics(
    backtest_df: pd.DataFrame,
    periods: Optional[Dict[str, SubsamplePeriod]] = None
) -> pd.DataFrame:
    """
    Convenience function to compute metrics from a backtest dataframe.

    Parameters
    ----------
    backtest_df : pd.DataFrame
        Backtest results with 'date', 'actual', and forecast columns
    periods : dict, optional
        Subsample periods to analyze

    Returns
    -------
    pd.DataFrame
        Metrics for each model/period combination
    """
    analyzer = SubsampleAnalyzer(periods)

    # Extract actuals
    if 'actual' in backtest_df.columns:
        actuals = backtest_df.set_index('date')['actual']
    else:
        raise ValueError("backtest_df must have 'actual' column")

    # Get forecast columns
    forecast_cols = [c for c in backtest_df.columns
                    if c not in ['date', 'actual', 'split', 'horizon', 'error']]

    forecasts = backtest_df[['date'] + forecast_cols].copy()

    return analyzer.analyze(forecasts, actuals)


def subsample_robustness_table(
    results: pd.DataFrame,
    metric: str = 'RMSE',
    format_decimals: int = 2
) -> pd.DataFrame:
    """
    Create a formatted robustness table for publication.

    Parameters
    ----------
    results : pd.DataFrame
        Output from SubsampleAnalyzer.analyze()
    metric : str
        Metric to display
    format_decimals : int
        Number of decimal places

    Returns
    -------
    pd.DataFrame
        Formatted table with models as rows, periods as columns
    """
    # Pivot to get models x periods
    pivot = results.pivot(index='Model', columns='Period', values=metric)

    # Add average ranking
    rank_cols = []
    for col in pivot.columns:
        rank_cols.append(pivot[col].rank(method='min'))
    rank_df = pd.concat(rank_cols, axis=1)
    pivot['Avg_Rank'] = rank_df.mean(axis=1)

    # Sort by average rank
    pivot = pivot.sort_values('Avg_Rank')

    # Format numbers
    format_str = f'{{:.{format_decimals}f}}'
    for col in pivot.columns:
        if col != 'Avg_Rank':
            pivot[col] = pivot[col].apply(lambda x: format_str.format(x) if pd.notna(x) else '-')
        else:
            pivot[col] = pivot[col].apply(lambda x: f'{x:.1f}')

    return pivot


def compute_period_stability(
    results: pd.DataFrame,
    metric: str = 'RMSE'
) -> pd.DataFrame:
    """
    Compute stability metrics for each model across periods.

    Parameters
    ----------
    results : pd.DataFrame
        Output from analyze()
    metric : str
        Metric to analyze

    Returns
    -------
    pd.DataFrame
        Stability metrics for each model
    """
    stability = []

    for model in results['Model'].unique():
        model_data = results[results['Model'] == model][metric].values

        if len(model_data) < 2:
            continue

        # Compute rankings in each period
        ranks = []
        for period in results['Period'].unique():
            period_data = results[results['Period'] == period]
            if model in period_data['Model'].values:
                rank = period_data[period_data['Model'] == model][metric].values[0]
                all_vals = period_data[metric].values
                ranks.append((np.sum(all_vals < rank) + 1))

        stability.append({
            'Model': model,
            f'Mean_{metric}': np.mean(model_data),
            f'Std_{metric}': np.std(model_data),
            f'CV_{metric}': np.std(model_data) / np.mean(model_data) if np.mean(model_data) != 0 else np.nan,
            'Mean_Rank': np.mean(ranks) if ranks else np.nan,
            'Std_Rank': np.std(ranks) if len(ranks) > 1 else np.nan,
            'N_Periods': len(model_data),
        })

    return pd.DataFrame(stability).sort_values('Mean_Rank')
