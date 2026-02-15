"""
Forecast Combiner Framework

This module provides the ForecastCombiner class that wraps all combination
methods and provides a unified interface for fitting and combining forecasts.

Supports both static and rolling weight estimation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from datetime import datetime

from .combination_methods import (
    equal_weight_combination,
    mse_weight_combination,
    granger_ramanathan_combination,
    trimmed_mean_combination,
    median_combination,
    get_combination_weights
)


class ForecastCombiner:
    """
    Framework for combining forecasts from multiple models.

    Parameters
    ----------
    combination_method : str
        One of: "equal", "mse", "gr_none", "gr_sum", "gr_convex",
        "trimmed", "median"
    trim_pct : float, optional
        Trim percentage for trimmed mean (default 0.1)

    Attributes
    ----------
    weights : dict or None
        Estimated combination weights
    intercept : float
        Intercept term (for GR unconstrained)
    method : str
        Combination method name
    fitted : bool
        Whether the combiner has been fitted
    fit_timestamp : datetime or None
        When the combiner was last fitted

    Examples
    --------
    >>> combiner = ForecastCombiner(combination_method="gr_convex")
    >>> combiner.fit(forecasts, actuals, train_end_idx=60)
    >>> combined = combiner.combine(forecasts)
    """

    VALID_METHODS = [
        "equal", "mse", "gr_none", "gr_sum", "gr_convex",
        "trimmed", "median"
    ]

    def __init__(
        self,
        combination_method: str = "equal",
        trim_pct: float = 0.1
    ):
        if combination_method not in self.VALID_METHODS:
            raise ValueError(
                f"Unknown combination method: {combination_method}. "
                f"Valid methods: {self.VALID_METHODS}"
            )

        self.method = combination_method
        self.trim_pct = trim_pct
        self.weights: Optional[Dict[str, float]] = None
        self.intercept: float = 0.0
        self.fitted: bool = False
        self.fit_timestamp: Optional[datetime] = None
        self._model_names: Optional[List[str]] = None

    def fit(
        self,
        forecasts: Dict[str, np.ndarray],
        actuals: np.ndarray,
        train_end_idx: int
    ) -> 'ForecastCombiner':
        """
        Estimate combination weights using training data.

        Parameters
        ----------
        forecasts : dict
            {model_name: forecast_array}
        actuals : np.ndarray
            Actual target values
        train_end_idx : int
            End index of training period

        Returns
        -------
        self : ForecastCombiner
            Fitted combiner
        """
        self._model_names = list(forecasts.keys())

        if self.method == "equal":
            n = len(forecasts)
            self.weights = {m: 1.0 / n for m in forecasts.keys()}
            self.intercept = 0.0

        elif self.method == "mse":
            _, self.weights = mse_weight_combination(
                forecasts, actuals, train_end_idx
            )
            self.intercept = 0.0

        elif self.method == "gr_none":
            _, self.weights, self.intercept = granger_ramanathan_combination(
                forecasts, actuals, train_end_idx, constraint="none"
            )

        elif self.method == "gr_sum":
            _, self.weights, self.intercept = granger_ramanathan_combination(
                forecasts, actuals, train_end_idx, constraint="sum_to_one"
            )

        elif self.method == "gr_convex":
            _, self.weights, self.intercept = granger_ramanathan_combination(
                forecasts, actuals, train_end_idx, constraint="convex"
            )

        elif self.method in ["trimmed", "median"]:
            # No weights to estimate
            self.weights = None
            self.intercept = 0.0

        self.fitted = True
        self.fit_timestamp = datetime.now()

        return self

    def combine(self, forecasts: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Apply combination weights to forecasts.

        Parameters
        ----------
        forecasts : dict
            {model_name: forecast_array}

        Returns
        -------
        combined : np.ndarray
            Combined forecast
        """
        if self.method == "equal":
            return equal_weight_combination(forecasts)

        elif self.method == "trimmed":
            return trimmed_mean_combination(forecasts, trim_pct=self.trim_pct)

        elif self.method == "median":
            return median_combination(forecasts)

        else:
            # Weighted combination
            if self.weights is None:
                raise RuntimeError(
                    "Combiner not fitted. Call fit() before combine()."
                )

            model_names = list(forecasts.keys())
            combined = np.zeros(len(forecasts[model_names[0]]))

            for name in model_names:
                weight = self.weights.get(name, 0.0)
                combined += weight * forecasts[name]

            return combined + self.intercept

    def get_weights_table(self) -> pd.DataFrame:
        """
        Return weights as DataFrame for reporting.

        Returns
        -------
        weights_df : pd.DataFrame
            DataFrame with model names, weights, and intercept
        """
        if self.weights is None:
            return pd.DataFrame({
                "method": [self.method],
                "note": ["No explicit weights (trimmed/median)"]
            })

        return pd.DataFrame({
            "model": list(self.weights.keys()),
            "weight": list(self.weights.values()),
            "intercept": [self.intercept] * len(self.weights),
            "method": [self.method] * len(self.weights)
        })

    def get_weight_summary(self) -> Dict:
        """
        Return weight summary as dictionary.

        Returns
        -------
        summary : dict
            Summary with method, weights, intercept, fit_timestamp
        """
        return {
            "method": self.method,
            "weights": self.weights,
            "intercept": self.intercept,
            "fitted": self.fitted,
            "fit_timestamp": self.fit_timestamp.isoformat() if self.fit_timestamp else None,
            "model_names": self._model_names
        }


def rolling_combination_backtest(
    models_forecasts: Dict[str, pd.DataFrame],
    actuals: pd.Series,
    methods: List[str],
    train_end: str,
    refit_interval: int = 12
) -> Tuple[pd.DataFrame, Dict[str, List[Dict[str, float]]]]:
    """
    Rolling backtest for forecast combinations.

    Parameters
    ----------
    models_forecasts : dict
        {model_name: DataFrame with DatetimeIndex and 'forecast' column}
    actuals : pd.Series
        Actual values with DatetimeIndex
    methods : list
        List of combination methods to evaluate
    train_end : str
        End of initial training period (e.g., "2022-12-01")
    refit_interval : int
        Months between weight re-estimation (default 12)

    Returns
    -------
    results : pd.DataFrame
        Backtest results with actual and combined forecasts
    weight_history : dict
        {method: list of weight dicts over time}
    """
    # Find common dates across all models and actuals
    common_dates = actuals.index.copy()
    for name, fc_df in models_forecasts.items():
        common_dates = common_dates.intersection(fc_df.index)

    common_dates = common_dates.sort_values()

    # Align data to common dates
    actuals_aligned = actuals.loc[common_dates].values
    forecasts = {}
    for name, fc_df in models_forecasts.items():
        forecasts[name] = fc_df.loc[common_dates, 'forecast'].values

    # Find training end index
    train_end_dt = pd.Timestamp(train_end)
    train_end_idx = (common_dates <= train_end_dt).sum()

    # Initialize results
    results = pd.DataFrame(index=common_dates)
    results['actual'] = actuals_aligned

    # Track weight evolution
    weight_history: Dict[str, List[Dict[str, float]]] = {m: [] for m in methods}

    for method in methods:
        combiner = ForecastCombiner(combination_method=method)

        # Initial fit on training data
        combiner.fit(forecasts, actuals_aligned, train_end_idx)

        # Store initial weights
        if combiner.weights is not None:
            weight_history[method].append({
                'date': str(common_dates[train_end_idx - 1]),
                'weights': combiner.weights.copy(),
                'intercept': combiner.intercept
            })

        # Rolling combination
        combined = np.zeros(len(actuals_aligned))
        combined[:train_end_idx] = np.nan  # No forecasts in training period

        last_refit_t = train_end_idx

        for t in range(train_end_idx, len(actuals_aligned)):
            # Refit weights periodically
            if (t - last_refit_t) >= refit_interval and t > train_end_idx:
                combiner.fit(forecasts, actuals_aligned, t)
                last_refit_t = t

                # Store updated weights
                if combiner.weights is not None:
                    weight_history[method].append({
                        'date': str(common_dates[t - 1]),
                        'weights': combiner.weights.copy(),
                        'intercept': combiner.intercept
                    })

            # Apply combination at time t
            fc_t = {name: np.array([forecasts[name][t]]) for name in forecasts.keys()}
            combined[t] = combiner.combine(fc_t)[0]

        results[f'combined_{method}'] = combined

    return results, weight_history


def compare_all_methods(
    forecasts: Dict[str, np.ndarray],
    actuals: np.ndarray,
    train_end_idx: int,
    val_end_idx: Optional[int] = None
) -> pd.DataFrame:
    """
    Compare all combination methods on the same data.

    Parameters
    ----------
    forecasts : dict
        {model_name: forecast_array}
    actuals : np.ndarray
        Actual values
    train_end_idx : int
        End of training period
    val_end_idx : int, optional
        End of validation period (if None, uses all data after training)

    Returns
    -------
    comparison : pd.DataFrame
        Comparison of methods with MAE, RMSE, etc.
    """
    methods = ["equal", "mse", "gr_none", "gr_sum", "gr_convex", "trimmed", "median"]

    if val_end_idx is None:
        val_end_idx = len(actuals)

    results = []

    for method in methods:
        combiner = ForecastCombiner(combination_method=method)
        combiner.fit(forecasts, actuals, train_end_idx)
        combined = combiner.combine(forecasts)

        # Compute metrics on validation/test period
        errors = combined[train_end_idx:val_end_idx] - actuals[train_end_idx:val_end_idx]
        valid_errors = errors[~np.isnan(errors)]

        if len(valid_errors) > 0:
            mae = np.mean(np.abs(valid_errors))
            rmse = np.sqrt(np.mean(valid_errors ** 2))
        else:
            mae = np.nan
            rmse = np.nan

        results.append({
            'method': method,
            'mae': mae,
            'rmse': rmse,
            'n_obs': len(valid_errors)
        })

    return pd.DataFrame(results)
