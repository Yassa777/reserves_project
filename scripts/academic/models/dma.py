"""
Dynamic Model Averaging (DMA) and Dynamic Model Selection (DMS)

Implementation following Raftery et al. (2010) and Koop & Korobilis (2012).

DMA/DMS maintains time-varying model weights that adapt based on recent
predictive performance, using a forgetting factor mechanism.

References:
- Raftery, A.E., Karny, M., & Ettler, P. (2010). Online Prediction Under
  Model Uncertainty via Dynamic Model Averaging. Technometrics.
- Koop, G. & Korobilis, D. (2012). Forecasting Inflation Using Dynamic
  Model Averaging. International Economic Review.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import warnings


@dataclass
class DMAResults:
    """Container for DMA/DMS results."""

    # Core results
    combined_forecasts: np.ndarray
    weights_history: np.ndarray
    model_names: List[str]
    dates: Optional[pd.DatetimeIndex] = None

    # Configuration
    alpha: float = 0.99
    method: str = "dma"

    # Diagnostics
    predictive_likelihoods: Optional[np.ndarray] = None
    forecast_variances: Optional[Dict[str, np.ndarray]] = None

    def get_weight_df(self) -> pd.DataFrame:
        """Return weights as a DataFrame with dates and model names."""
        df = pd.DataFrame(
            self.weights_history,
            columns=self.model_names
        )
        if self.dates is not None:
            df.index = self.dates
        return df

    def get_weight_summary(self) -> pd.DataFrame:
        """Summary statistics for model weights."""
        return pd.DataFrame({
            'model': self.model_names,
            'mean_weight': np.mean(self.weights_history, axis=0),
            'std_weight': np.std(self.weights_history, axis=0),
            'median_weight': np.median(self.weights_history, axis=0),
            'max_weight': np.max(self.weights_history, axis=0),
            'min_weight': np.min(self.weights_history, axis=0),
        })

    def get_selection_frequency(self, split_mask: Optional[np.ndarray] = None) -> pd.Series:
        """
        For DMS: how often each model was selected (had highest weight).

        Parameters
        ----------
        split_mask : np.ndarray, optional
            Boolean mask to compute frequency for subset of periods
        """
        weights = self.weights_history
        if split_mask is not None:
            weights = weights[split_mask]

        selected = np.argmax(weights, axis=1)
        counts = np.bincount(selected, minlength=len(self.model_names))
        freq = pd.Series(
            counts / len(selected),
            index=self.model_names,
            name='selection_frequency'
        )
        return freq.sort_values(ascending=False)


class DynamicModelAveraging:
    """
    Dynamic Model Averaging and Selection.

    At each time t, maintains posterior model probability pi_{t|t-1,k} for
    each model k. These probabilities are updated based on predictive
    performance using a forgetting factor alpha.

    DMA: Weighted average of all model forecasts using posterior probabilities
    DMS: Select the model with highest posterior probability each period

    Parameters
    ----------
    alpha : float, default=0.99
        Forgetting factor in (0, 1]. Controls how fast old performance
        is discounted. alpha=1 means no forgetting (all history weighted equally).
        alpha=0.95 means 5% discount rate (recent performance emphasized).

    method : str, default="dma"
        "dma" for weighted average, "dms" for model selection

    variance_window : int, default=24
        Rolling window for estimating forecast error variances

    min_variance : float, default=1e-6
        Minimum variance to prevent numerical issues

    warmup_periods : int, default=12
        Number of periods to use equal weights before starting adaptation

    Attributes
    ----------
    model_names : list
        Names of models in the pool
    n_models : int
        Number of models
    results_ : DMAResults
        Fitted results after calling fit_predict
    """

    def __init__(
        self,
        alpha: float = 0.99,
        method: str = "dma",
        variance_window: int = 24,
        min_variance: float = 1e-6,
        warmup_periods: int = 12
    ):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if method not in ("dma", "dms"):
            raise ValueError("method must be 'dma' or 'dms'")

        self.alpha = alpha
        self.method = method
        self.variance_window = variance_window
        self.min_variance = min_variance
        self.warmup_periods = warmup_periods

        self.model_names = None
        self.n_models = None
        self.results_ = None

    def fit_predict(
        self,
        model_forecasts: Dict[str, np.ndarray],
        actuals: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
        forecast_variances: Optional[Dict[str, np.ndarray]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run DMA/DMS on a sequence of forecasts.

        Parameters
        ----------
        model_forecasts : dict
            {model_name: forecast_array} - one-step-ahead forecasts.
            All arrays must have same length as actuals.

        actuals : np.array
            Realized values (T,)

        dates : pd.DatetimeIndex, optional
            Date index for the forecasts

        forecast_variances : dict, optional
            {model_name: variance_array} - forecast error variances.
            If None, estimated from rolling window of past errors.

        Returns
        -------
        combined : np.array
            DMA or DMS combined forecasts (T,)
        weights : np.array
            Time-varying model weights (T, K)
        """
        self.model_names = list(model_forecasts.keys())
        self.n_models = len(self.model_names)
        T = len(actuals)

        # Validate inputs
        for m, fc in model_forecasts.items():
            if len(fc) != T:
                raise ValueError(f"Forecast length mismatch for {m}: {len(fc)} vs {T}")

        # Initialize weights (equal)
        weights = np.zeros((T, self.n_models))
        weights[0] = 1.0 / self.n_models

        # Estimate forecast variances if not provided
        if forecast_variances is None:
            forecast_variances = self._estimate_variances(model_forecasts, actuals)

        # Store predictive likelihoods for diagnostics
        pred_liks = np.zeros((T, self.n_models))

        # Combined forecasts
        combined = np.zeros(T)

        for t in range(T):
            # Get current model forecasts
            fc_t = np.array([model_forecasts[m][t] for m in self.model_names])

            if t < self.warmup_periods:
                # Warmup period: use equal weights
                weights[t] = 1.0 / self.n_models
                combined[t] = np.mean(fc_t)
            else:
                # Compute predictive likelihoods for t-1
                for k, m in enumerate(self.model_names):
                    fc_prev = model_forecasts[m][t-1]
                    var_prev = forecast_variances[m][t-1]
                    actual_prev = actuals[t-1]

                    # Handle NaN forecasts
                    if np.isnan(fc_prev) or np.isnan(actual_prev):
                        pred_liks[t-1, k] = 1.0 / self.n_models  # Neutral
                    else:
                        # Normal predictive density
                        pred_liks[t-1, k] = norm.pdf(
                            actual_prev,
                            loc=fc_prev,
                            scale=np.sqrt(max(var_prev, self.min_variance))
                        )

                # Ensure positive likelihoods
                pred_liks[t-1] = np.maximum(pred_liks[t-1], 1e-300)

                # Update weights with forgetting factor
                # pi_{t|t-1,k} propto pi_{t-1|t-1,k}^alpha * p(y_{t-1} | M_k)
                prior_weights = weights[t-1] ** self.alpha
                prior_weights /= prior_weights.sum()  # Normalize

                # Posterior weights
                posterior = prior_weights * pred_liks[t-1]
                if posterior.sum() > 0:
                    posterior /= posterior.sum()
                else:
                    # Fallback to equal weights if all likelihoods are zero
                    posterior = np.ones(self.n_models) / self.n_models

                weights[t] = posterior

                # Combined forecast
                if self.method == "dma":
                    # Weighted average
                    combined[t] = np.dot(posterior, fc_t)
                else:  # dms
                    # Select model with highest weight
                    best_model_idx = np.argmax(posterior)
                    combined[t] = fc_t[best_model_idx]

        # Store results
        self.results_ = DMAResults(
            combined_forecasts=combined,
            weights_history=weights,
            model_names=self.model_names,
            dates=dates,
            alpha=self.alpha,
            method=self.method,
            predictive_likelihoods=pred_liks,
            forecast_variances=forecast_variances
        )

        return combined, weights

    def _estimate_variances(
        self,
        model_forecasts: Dict[str, np.ndarray],
        actuals: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Estimate rolling forecast error variances.

        Uses an expanding window for the first variance_window periods,
        then a rolling window.
        """
        variances = {}

        for m in self.model_names:
            fc = model_forecasts[m]
            errors = actuals - fc
            var = np.zeros(len(actuals))

            for t in range(len(actuals)):
                if t < 2:
                    # Not enough data - use large variance
                    var[t] = np.nanvar(errors) if not np.all(np.isnan(errors)) else 1.0
                elif t < self.variance_window:
                    # Expanding window
                    var[t] = np.nanvar(errors[:t])
                else:
                    # Rolling window
                    var[t] = np.nanvar(errors[t-self.variance_window:t])

                # Ensure minimum variance
                if np.isnan(var[t]) or var[t] < self.min_variance:
                    var[t] = self.min_variance

            variances[m] = var

        return variances

    def get_weight_summary(self) -> Optional[pd.DataFrame]:
        """Summary statistics for model weights."""
        if self.results_ is None:
            return None
        return self.results_.get_weight_summary()

    def get_selection_frequency(
        self,
        split_mask: Optional[np.ndarray] = None
    ) -> Optional[pd.Series]:
        """For DMS: how often each model was selected."""
        if self.results_ is None:
            return None
        return self.results_.get_selection_frequency(split_mask)


def run_dma_grid_search(
    model_forecasts: Dict[str, np.ndarray],
    actuals: np.ndarray,
    alphas: List[float] = [0.90, 0.95, 0.99, 1.0],
    validation_mask: Optional[np.ndarray] = None,
    warmup_periods: int = 12
) -> Tuple[float, pd.DataFrame]:
    """
    Grid search over forgetting factor alpha.

    Returns best alpha based on validation RMSE.

    Parameters
    ----------
    model_forecasts : dict
        {model_name: forecast_array}
    actuals : np.array
        Realized values
    alphas : list
        Grid of alpha values to search
    validation_mask : np.array, optional
        Boolean mask for validation period. If None, uses all data
        after warmup.
    warmup_periods : int
        Number of warmup periods to exclude from evaluation

    Returns
    -------
    best_alpha : float
        Alpha value with lowest validation RMSE
    results_df : pd.DataFrame
        Performance for each alpha value
    """
    results = []

    for alpha in alphas:
        for method in ["dma", "dms"]:
            dma = DynamicModelAveraging(
                alpha=alpha,
                method=method,
                warmup_periods=warmup_periods
            )
            combined, _ = dma.fit_predict(model_forecasts, actuals)

            # Determine evaluation mask
            if validation_mask is not None:
                eval_mask = validation_mask
            else:
                eval_mask = np.arange(len(actuals)) >= warmup_periods

            # Compute metrics
            fc_eval = combined[eval_mask]
            act_eval = actuals[eval_mask]
            valid_idx = ~np.isnan(fc_eval) & ~np.isnan(act_eval)

            if valid_idx.sum() > 0:
                errors = fc_eval[valid_idx] - act_eval[valid_idx]
                rmse = np.sqrt(np.mean(errors ** 2))
                mae = np.mean(np.abs(errors))
            else:
                rmse = np.nan
                mae = np.nan

            results.append({
                'alpha': alpha,
                'method': method,
                'rmse': rmse,
                'mae': mae,
                'n_obs': valid_idx.sum()
            })

    results_df = pd.DataFrame(results)

    # Find best alpha for DMA
    dma_results = results_df[results_df['method'] == 'dma']
    best_idx = dma_results['rmse'].idxmin()
    best_alpha = results_df.loc[best_idx, 'alpha']

    return best_alpha, results_df


def rolling_dma_backtest(
    model_forecasts: Dict[str, np.ndarray],
    actuals: np.ndarray,
    dates: pd.DatetimeIndex,
    train_end: str,
    valid_end: str,
    alpha: float = 0.99,
    methods: List[str] = ["dma", "dms"]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rolling backtest for DMA and DMS.

    Parameters
    ----------
    model_forecasts : dict
        {model_name: forecast_array}
    actuals : np.array
        Realized values
    dates : pd.DatetimeIndex
        Date index
    train_end : str
        End of training period (e.g., "2019-12-01")
    valid_end : str
        End of validation period (e.g., "2022-12-01")
    alpha : float
        Forgetting factor
    methods : list
        Methods to evaluate ("dma", "dms")

    Returns
    -------
    results : pd.DataFrame
        Forecasts, actuals, and weights for each date
    summary : pd.DataFrame
        Performance metrics by split
    """
    train_mask = dates <= pd.Timestamp(train_end)
    valid_mask = (dates > pd.Timestamp(train_end)) & (dates <= pd.Timestamp(valid_end))
    test_mask = dates > pd.Timestamp(valid_end)

    results = pd.DataFrame(index=dates)
    results['actual'] = actuals

    all_weights = {}

    for method in methods:
        dma = DynamicModelAveraging(alpha=alpha, method=method)
        combined, weights = dma.fit_predict(model_forecasts, actuals, dates=dates)

        results[f'{method}_forecast'] = combined
        all_weights[method] = weights

        # Store weights for each model
        for k, m in enumerate(dma.model_names):
            results[f'weight_{method}_{m}'] = weights[:, k]

    # Also compute equal-weight combination for comparison
    model_arrays = np.array([model_forecasts[m] for m in model_forecasts.keys()])
    results['equal_weight_forecast'] = np.nanmean(model_arrays, axis=0)

    # Compute metrics by split
    summary = []
    for split_name, mask in [('train', train_mask),
                              ('validation', valid_mask),
                              ('test', test_mask)]:
        if mask.sum() == 0:
            continue

        for method in methods + ['equal_weight']:
            fc_col = f'{method}_forecast'
            if fc_col not in results.columns:
                continue

            fc = results.loc[mask, fc_col].values
            act = results.loc[mask, 'actual'].values

            valid_idx = ~np.isnan(fc) & ~np.isnan(act)
            if valid_idx.sum() == 0:
                continue

            errors = fc[valid_idx] - act[valid_idx]
            mae = np.mean(np.abs(errors))
            rmse = np.sqrt(np.mean(errors ** 2))
            mape = np.mean(np.abs(errors / act[valid_idx])) * 100

            summary.append({
                'method': method,
                'split': split_name,
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'n_obs': valid_idx.sum()
            })

    summary_df = pd.DataFrame(summary)

    return results, summary_df


class StateDependentDMA(DynamicModelAveraging):
    """
    DMA with weights that depend on state variables.

    Extends base DMA to allow the forgetting factor to vary with a state
    variable (e.g., volatility). During high-volatility periods, alpha
    can be reduced to allow faster adaptation.

    Parameters
    ----------
    base_alpha : float, default=0.99
        Base forgetting factor
    state_variable : np.array, optional
        Variable that affects weight dynamics (e.g., volatility)
    alpha_adjustment : float, default=0.05
        Amount to reduce alpha during high-state periods
    state_threshold : float, default=0.75
        Quantile threshold for "high state" (e.g., 75th percentile)
    """

    def __init__(
        self,
        base_alpha: float = 0.99,
        state_variable: Optional[np.ndarray] = None,
        alpha_adjustment: float = 0.05,
        state_threshold: float = 0.75,
        **kwargs
    ):
        super().__init__(alpha=base_alpha, **kwargs)
        self.base_alpha = base_alpha
        self.state_variable = state_variable
        self.alpha_adjustment = alpha_adjustment
        self.state_threshold = state_threshold

    def fit_predict(
        self,
        model_forecasts: Dict[str, np.ndarray],
        actuals: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
        forecast_variances: Optional[Dict[str, np.ndarray]] = None,
        state_variable: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run state-dependent DMA.

        If state_variable is provided, alpha is adjusted during high-state periods.
        """
        if state_variable is not None:
            self.state_variable = state_variable

        if self.state_variable is None:
            # Fall back to standard DMA
            return super().fit_predict(
                model_forecasts, actuals, dates, forecast_variances
            )

        # Compute state-dependent alpha sequence
        threshold = np.nanquantile(self.state_variable, self.state_threshold)
        high_state = self.state_variable > threshold

        self.model_names = list(model_forecasts.keys())
        self.n_models = len(self.model_names)
        T = len(actuals)

        # Initialize
        weights = np.zeros((T, self.n_models))
        weights[0] = 1.0 / self.n_models

        if forecast_variances is None:
            forecast_variances = self._estimate_variances(model_forecasts, actuals)

        pred_liks = np.zeros((T, self.n_models))
        combined = np.zeros(T)

        for t in range(T):
            fc_t = np.array([model_forecasts[m][t] for m in self.model_names])

            if t < self.warmup_periods:
                weights[t] = 1.0 / self.n_models
                combined[t] = np.mean(fc_t)
            else:
                # State-dependent alpha
                if high_state[t-1]:
                    alpha_t = max(self.base_alpha - self.alpha_adjustment, 0.8)
                else:
                    alpha_t = self.base_alpha

                # Compute predictive likelihoods
                for k, m in enumerate(self.model_names):
                    fc_prev = model_forecasts[m][t-1]
                    var_prev = forecast_variances[m][t-1]
                    actual_prev = actuals[t-1]

                    if np.isnan(fc_prev) or np.isnan(actual_prev):
                        pred_liks[t-1, k] = 1.0 / self.n_models
                    else:
                        pred_liks[t-1, k] = norm.pdf(
                            actual_prev,
                            loc=fc_prev,
                            scale=np.sqrt(max(var_prev, self.min_variance))
                        )

                pred_liks[t-1] = np.maximum(pred_liks[t-1], 1e-300)

                # Update with state-dependent alpha
                prior_weights = weights[t-1] ** alpha_t
                prior_weights /= prior_weights.sum()

                posterior = prior_weights * pred_liks[t-1]
                if posterior.sum() > 0:
                    posterior /= posterior.sum()
                else:
                    posterior = np.ones(self.n_models) / self.n_models

                weights[t] = posterior

                if self.method == "dma":
                    combined[t] = np.dot(posterior, fc_t)
                else:
                    best_model_idx = np.argmax(posterior)
                    combined[t] = fc_t[best_model_idx]

        self.results_ = DMAResults(
            combined_forecasts=combined,
            weights_history=weights,
            model_names=self.model_names,
            dates=dates,
            alpha=self.base_alpha,
            method=self.method,
            predictive_likelihoods=pred_liks,
            forecast_variances=forecast_variances
        )

        return combined, weights


def compute_dma_metrics(
    dma_forecasts: np.ndarray,
    actuals: np.ndarray,
    model_forecasts: Dict[str, np.ndarray],
    mask: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Compute forecast accuracy metrics comparing DMA to individual models.

    Parameters
    ----------
    dma_forecasts : np.array
        DMA combined forecasts
    actuals : np.array
        Realized values
    model_forecasts : dict
        Individual model forecasts
    mask : np.array, optional
        Boolean mask for evaluation period

    Returns
    -------
    metrics : pd.DataFrame
        RMSE, MAE, MAPE for each model and DMA
    """
    if mask is None:
        mask = np.ones(len(actuals), dtype=bool)

    results = []

    # DMA
    fc = dma_forecasts[mask]
    act = actuals[mask]
    valid = ~np.isnan(fc) & ~np.isnan(act)

    if valid.sum() > 0:
        errors = fc[valid] - act[valid]
        results.append({
            'model': 'DMA',
            'rmse': np.sqrt(np.mean(errors ** 2)),
            'mae': np.mean(np.abs(errors)),
            'mape': np.mean(np.abs(errors / act[valid])) * 100,
            'n_obs': valid.sum()
        })

    # Individual models
    for m, fc_m in model_forecasts.items():
        fc = fc_m[mask]
        valid = ~np.isnan(fc) & ~np.isnan(act)

        if valid.sum() > 0:
            errors = fc[valid] - act[valid]
            results.append({
                'model': m,
                'rmse': np.sqrt(np.mean(errors ** 2)),
                'mae': np.mean(np.abs(errors)),
                'mape': np.mean(np.abs(errors / act[valid])) * 100,
                'n_obs': valid.sum()
            })

    return pd.DataFrame(results).sort_values('rmse')
