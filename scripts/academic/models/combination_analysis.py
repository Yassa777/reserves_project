"""
Combination Analysis and Evaluation Metrics

This module provides evaluation metrics for forecast combinations,
including relative performance measures and combination efficiency.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


def compute_forecast_metrics(
    forecast: np.ndarray,
    actual: np.ndarray,
    scale_for_mase: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute standard forecast accuracy metrics.

    Parameters
    ----------
    forecast : np.ndarray
        Forecast values
    actual : np.ndarray
        Actual values
    scale_for_mase : np.ndarray, optional
        In-sample errors for MASE scaling

    Returns
    -------
    metrics : dict
        Dictionary with MAE, RMSE, MAPE, sMAPE, (MASE if scale provided)
    """
    errors = forecast - actual
    valid_mask = ~np.isnan(errors)

    if valid_mask.sum() == 0:
        return {
            'mae': np.nan,
            'rmse': np.nan,
            'mape': np.nan,
            'smape': np.nan,
            'mase': np.nan
        }

    errors = errors[valid_mask]
    forecast_valid = forecast[valid_mask]
    actual_valid = actual[valid_mask]

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    # MAPE (handle zeros in actual)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape_vals = np.abs(errors / actual_valid) * 100
        mape_vals = mape_vals[np.isfinite(mape_vals)]
        mape = np.mean(mape_vals) if len(mape_vals) > 0 else np.nan

    # sMAPE
    with np.errstate(divide='ignore', invalid='ignore'):
        smape_vals = 200 * np.abs(errors) / (np.abs(actual_valid) + np.abs(forecast_valid))
        smape_vals = smape_vals[np.isfinite(smape_vals)]
        smape = np.mean(smape_vals) if len(smape_vals) > 0 else np.nan

    # MASE
    if scale_for_mase is not None and len(scale_for_mase) > 0:
        scale = np.mean(np.abs(scale_for_mase))
        mase = mae / scale if scale > 0 else np.nan
    else:
        mase = np.nan

    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'smape': smape,
        'mase': mase
    }


def relative_combination_value(
    combined_rmse: float,
    best_individual_rmse: float
) -> float:
    """
    Percentage improvement of combination over best individual model.

    Parameters
    ----------
    combined_rmse : float
        RMSE of the combined forecast
    best_individual_rmse : float
        RMSE of the best individual model

    Returns
    -------
    improvement : float
        Percentage improvement (positive = combination better)
    """
    if best_individual_rmse == 0 or np.isnan(best_individual_rmse):
        return np.nan

    return (best_individual_rmse - combined_rmse) / best_individual_rmse * 100


def combination_efficiency(
    combined_rmse: float,
    individual_rmses: List[float]
) -> float:
    """
    How close to oracle (best ex-post model) combination gets.

    efficiency = 1 - (combined - oracle) / (avg_individual - oracle)

    A value of 1 means the combination matches the oracle.
    A value of 0 means it performs like the average model.
    Negative values mean worse than average.

    Parameters
    ----------
    combined_rmse : float
        RMSE of the combined forecast
    individual_rmses : list
        RMSEs of individual models

    Returns
    -------
    efficiency : float
        Combination efficiency score
    """
    valid_rmses = [r for r in individual_rmses if not np.isnan(r)]

    if len(valid_rmses) == 0:
        return np.nan

    oracle = min(valid_rmses)
    avg_ind = np.mean(valid_rmses)

    if avg_ind == oracle:
        # All models have same RMSE
        if combined_rmse == oracle:
            return 1.0
        else:
            return 0.0

    return 1 - (combined_rmse - oracle) / (avg_ind - oracle)


def compute_combination_diagnostics(
    combined_forecasts: Dict[str, np.ndarray],
    individual_forecasts: Dict[str, np.ndarray],
    actuals: np.ndarray,
    start_idx: int,
    end_idx: Optional[int] = None
) -> pd.DataFrame:
    """
    Compute comprehensive diagnostics for all combination methods.

    Parameters
    ----------
    combined_forecasts : dict
        {method_name: combined_forecast_array}
    individual_forecasts : dict
        {model_name: forecast_array}
    actuals : np.ndarray
        Actual values
    start_idx : int
        Start index for evaluation
    end_idx : int, optional
        End index for evaluation

    Returns
    -------
    diagnostics : pd.DataFrame
        Diagnostic metrics for each combination method
    """
    if end_idx is None:
        end_idx = len(actuals)

    eval_actuals = actuals[start_idx:end_idx]

    # Compute individual model metrics
    individual_rmses = {}
    individual_maes = {}

    for name, fc in individual_forecasts.items():
        eval_fc = fc[start_idx:end_idx]
        metrics = compute_forecast_metrics(eval_fc, eval_actuals)
        individual_rmses[name] = metrics['rmse']
        individual_maes[name] = metrics['mae']

    best_individual_rmse = min(individual_rmses.values())
    best_individual_mae = min(individual_maes.values())
    best_model = min(individual_rmses, key=individual_rmses.get)

    # Compute combination metrics
    results = []

    for method, combined_fc in combined_forecasts.items():
        eval_fc = combined_fc[start_idx:end_idx]
        metrics = compute_forecast_metrics(eval_fc, eval_actuals)

        # Relative metrics
        rel_value = relative_combination_value(metrics['rmse'], best_individual_rmse)
        efficiency = combination_efficiency(
            metrics['rmse'],
            list(individual_rmses.values())
        )

        results.append({
            'method': method,
            'mae': metrics['mae'],
            'rmse': metrics['rmse'],
            'mape': metrics['mape'],
            'smape': metrics['smape'],
            'relative_value_pct': rel_value,
            'efficiency': efficiency,
            'vs_best_model': best_model,
            'best_individual_rmse': best_individual_rmse,
            'best_individual_mae': best_individual_mae
        })

    return pd.DataFrame(results)


def analyze_weight_stability(
    weight_history: List[Dict],
    model_names: List[str]
) -> pd.DataFrame:
    """
    Analyze stability of combination weights over time.

    Parameters
    ----------
    weight_history : list
        List of weight dictionaries with dates
    model_names : list
        Names of models

    Returns
    -------
    stability : pd.DataFrame
        Stability metrics for each model's weight
    """
    if len(weight_history) < 2:
        return pd.DataFrame({'note': ['Not enough weight updates for stability analysis']})

    # Extract weights over time
    weight_series = {name: [] for name in model_names}
    dates = []

    for entry in weight_history:
        dates.append(entry['date'])
        weights = entry['weights']
        for name in model_names:
            weight_series[name].append(weights.get(name, 0.0))

    # Compute stability metrics
    results = []
    for name in model_names:
        weights = np.array(weight_series[name])
        results.append({
            'model': name,
            'mean_weight': np.mean(weights),
            'std_weight': np.std(weights),
            'min_weight': np.min(weights),
            'max_weight': np.max(weights),
            'range': np.max(weights) - np.min(weights),
            'cv': np.std(weights) / np.mean(weights) if np.mean(weights) != 0 else np.nan
        })

    return pd.DataFrame(results)


def decompose_combination_variance(
    individual_forecasts: Dict[str, np.ndarray],
    weights: Dict[str, float],
    start_idx: int,
    end_idx: Optional[int] = None
) -> Dict[str, float]:
    """
    Decompose variance of combined forecast into model contributions.

    Parameters
    ----------
    individual_forecasts : dict
        {model_name: forecast_array}
    weights : dict
        {model_name: weight}
    start_idx : int
        Start index
    end_idx : int, optional
        End index

    Returns
    -------
    decomposition : dict
        Variance contribution from each model
    """
    model_names = list(individual_forecasts.keys())

    if end_idx is None:
        end_idx = len(individual_forecasts[model_names[0]])

    # Extract evaluation period
    forecasts_matrix = np.column_stack([
        individual_forecasts[m][start_idx:end_idx] for m in model_names
    ])

    weights_array = np.array([weights.get(m, 0.0) for m in model_names])

    # Compute weighted forecasts
    weighted_forecasts = forecasts_matrix * weights_array

    # Variance of each weighted component
    var_contributions = {}
    total_var = np.var(np.sum(weighted_forecasts, axis=1))

    for i, name in enumerate(model_names):
        # Marginal variance contribution
        var_contributions[name] = np.var(weighted_forecasts[:, i])

    return {
        'model_contributions': var_contributions,
        'total_variance': total_var
    }


def create_summary_report(
    diagnostics_val: pd.DataFrame,
    diagnostics_test: pd.DataFrame,
    weight_tables: Dict[str, pd.DataFrame]
) -> Dict:
    """
    Create a comprehensive summary report for combination analysis.

    Parameters
    ----------
    diagnostics_val : pd.DataFrame
        Validation period diagnostics
    diagnostics_test : pd.DataFrame
        Test period diagnostics
    weight_tables : dict
        {method: weight DataFrame}

    Returns
    -------
    report : dict
        Summary report
    """
    # Find best methods
    best_val_method = diagnostics_val.loc[diagnostics_val['rmse'].idxmin(), 'method']
    best_test_method = diagnostics_test.loc[diagnostics_test['rmse'].idxmin(), 'method']

    # Average performance
    avg_val_rmse = diagnostics_val['rmse'].mean()
    avg_test_rmse = diagnostics_test['rmse'].mean()

    # Compile weight summaries
    weight_summary = {}
    for method, df in weight_tables.items():
        if 'weight' in df.columns:
            weight_summary[method] = df[['model', 'weight']].to_dict('records')

    return {
        'best_validation_method': best_val_method,
        'best_test_method': best_test_method,
        'validation_metrics': diagnostics_val.to_dict('records'),
        'test_metrics': diagnostics_test.to_dict('records'),
        'average_validation_rmse': avg_val_rmse,
        'average_test_rmse': avg_test_rmse,
        'weight_summary': weight_summary,
        'efficiency_validation': diagnostics_val.set_index('method')['efficiency'].to_dict(),
        'efficiency_test': diagnostics_test.set_index('method')['efficiency'].to_dict()
    }
