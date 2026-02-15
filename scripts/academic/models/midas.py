"""
Mixed Data Sampling (MIDAS) Regression for Reserves Forecasting.

Handles frequency mismatch between monthly reserves and daily exchange rate data.

Classes:
    MIDAS: Core MIDAS regression with polynomial weighting
    MIDAS_AR: MIDAS with autoregressive low-frequency terms
    UMIDAS: Unrestricted MIDAS (direct OLS on blocked HF lags)

Reference:
    Ghysels, E., Santa-Clara, P., & Valkanov, R. (2006). Predicting Volatility:
    Getting the Most out of Return Data. Review of Financial Studies.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple, List
from scipy.optimize import minimize
import warnings

from .midas_weights import (
    compute_weights,
    weight_initial_params,
    exp_almon_weights,
    beta_weights,
    step_weights,
)


class MIDAS:
    """
    Mixed Data Sampling Regression.

    Aggregates high-frequency data (e.g., daily exchange rate) to predict
    low-frequency outcomes (e.g., monthly reserves) using polynomial weights.

    Parameters
    ----------
    weight_type : str
        Type of weighting scheme: 'exp_almon', 'beta', 'step', 'uniform'
    n_hf_lags : int
        Number of high-frequency lags per low-frequency period (default: 22 trading days)
    n_lf_lags : int
        Number of low-frequency lags to include (months of HF data)

    Attributes
    ----------
    theta : np.ndarray
        Optimized weight parameters
    beta : np.ndarray
        Regression coefficients (intercept + HF aggregates + LF regressors)
    weights : np.ndarray
        Final MIDAS weights after optimization
    fitted : np.ndarray
        In-sample fitted values
    residuals : np.ndarray
        In-sample residuals
    """

    def __init__(
        self,
        weight_type: str = "exp_almon",
        n_hf_lags: int = 22,
        n_lf_lags: int = 3
    ):
        self.weight_type = weight_type
        self.n_hf_lags = n_hf_lags
        self.n_lf_lags = n_lf_lags

        # Will be set after fit
        self.theta = None
        self.beta = None
        self.intercept = None
        self.hf_coefs = None
        self.lf_coefs = None
        self.weights = None
        self.fitted = None
        self.residuals = None
        self.valid_mask = None
        self.ssr = None
        self.r2 = None
        self.n_obs = None

        # Data storage for forecasting
        self.Y_lf = None
        self.X_hf = None
        self.X_lf = None

    def _get_weights(self, theta: np.ndarray) -> np.ndarray:
        """Compute weights given parameters."""
        return compute_weights(self.weight_type, self.n_hf_lags, theta)

    def _aggregate_hf_data(
        self,
        X_hf: pd.Series,
        dates_lf: pd.DatetimeIndex,
        weights: np.ndarray
    ) -> np.ndarray:
        """
        Aggregate high-frequency data to low-frequency using MIDAS weights.

        For each low-frequency date, aggregates the last n_hf_lags observations
        from the corresponding month using the weight vector.

        Parameters
        ----------
        X_hf : pd.Series
            High-frequency data with DatetimeIndex
        dates_lf : pd.DatetimeIndex
            Low-frequency dates (end of month)
        weights : np.ndarray
            MIDAS weights

        Returns
        -------
        X_agg : np.ndarray
            Shape (n_lf, n_lf_lags + 1) aggregated regressors
        """
        n_lf = len(dates_lf)
        X_agg = np.zeros((n_lf, self.n_lf_lags + 1))

        for t, date_lf in enumerate(dates_lf):
            for lag in range(self.n_lf_lags + 1):
                # Get the month for this lag
                lag_date = date_lf - pd.DateOffset(months=lag)
                month_start = lag_date.replace(day=1)
                month_end = (month_start + pd.DateOffset(months=1)) - pd.DateOffset(days=1)

                # Get HF data for this month
                mask = (X_hf.index >= month_start) & (X_hf.index <= month_end)
                hf_values = X_hf[mask].values

                if len(hf_values) >= self.n_hf_lags:
                    # Use last n_hf_lags observations (most recent within month)
                    hf_values = hf_values[-self.n_hf_lags:]
                    X_agg[t, lag] = np.dot(weights, hf_values)
                elif len(hf_values) > 0:
                    # Adjust weights for shorter months
                    adj_weights = weights[:len(hf_values)]
                    adj_weights = adj_weights / adj_weights.sum()
                    X_agg[t, lag] = np.dot(adj_weights, hf_values)
                else:
                    X_agg[t, lag] = np.nan

        return X_agg

    def _build_design_matrix(
        self,
        Y_lf: pd.Series,
        X_hf: pd.Series,
        X_lf: Optional[pd.DataFrame],
        weights: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build the design matrix for MIDAS regression.

        Returns
        -------
        X : np.ndarray
            Design matrix (intercept, HF aggregates, LF regressors)
        Y : np.ndarray
            Target vector
        valid : np.ndarray
            Boolean mask for valid observations
        """
        # Aggregate HF data
        X_agg = self._aggregate_hf_data(X_hf, Y_lf.index, weights)

        # Combine with LF regressors
        if X_lf is not None:
            X_combined = np.column_stack([X_agg, X_lf.values])
        else:
            X_combined = X_agg

        # Add intercept
        X = np.column_stack([np.ones(len(Y_lf)), X_combined])

        # Valid observations (no NaN)
        valid = ~np.isnan(X).any(axis=1) & ~np.isnan(Y_lf.values)

        return X, Y_lf.values, valid

    def fit(
        self,
        Y_lf: pd.Series,
        X_hf: pd.Series,
        X_lf: Optional[pd.DataFrame] = None,
        optimize: bool = True
    ):
        """
        Fit MIDAS regression via Non-Linear Least Squares.

        Parameters
        ----------
        Y_lf : pd.Series
            Low-frequency target (monthly reserves)
        X_hf : pd.Series
            High-frequency regressor (daily exchange rate)
        X_lf : pd.DataFrame, optional
            Additional low-frequency regressors
        optimize : bool
            Whether to optimize weight parameters (default: True)

        Returns
        -------
        self
        """
        # Store data
        self.Y_lf = Y_lf
        self.X_hf = X_hf
        self.X_lf = X_lf

        # Get initial parameters
        init_params, bounds = weight_initial_params(self.weight_type)

        if len(init_params) > 0 and optimize:
            # Optimize weight parameters via NLS
            def objective(theta):
                try:
                    weights = self._get_weights(theta)
                    X, Y, valid = self._build_design_matrix(Y_lf, X_hf, X_lf, weights)

                    if valid.sum() < 10:
                        return 1e10

                    X_valid = X[valid]
                    Y_valid = Y[valid]

                    # OLS given weights
                    beta, residuals, _, _ = np.linalg.lstsq(X_valid, Y_valid, rcond=None)
                    ssr = np.sum(residuals) if len(residuals) > 0 else np.sum((Y_valid - X_valid @ beta) ** 2)

                    return ssr

                except Exception:
                    return 1e10

            # Optimize
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = minimize(
                    objective,
                    init_params,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4}
                )

            self.theta = result.x

        else:
            # Use initial/default parameters
            self.theta = init_params if len(init_params) > 0 else np.array([])

        # Final estimation with optimized weights
        self.weights = self._get_weights(self.theta)
        X, Y, valid = self._build_design_matrix(Y_lf, X_hf, X_lf, self.weights)

        self.valid_mask = valid
        X_valid = X[valid]
        Y_valid = Y[valid]

        # OLS
        self.beta, _, _, _ = np.linalg.lstsq(X_valid, Y_valid, rcond=None)
        self.intercept = self.beta[0]
        self.hf_coefs = self.beta[1:self.n_lf_lags + 2]

        if X_lf is not None:
            self.lf_coefs = self.beta[self.n_lf_lags + 2:]

        # Fitted values and residuals
        self.fitted = X_valid @ self.beta
        self.residuals = Y_valid - self.fitted

        # Diagnostics
        self.n_obs = valid.sum()
        self.ssr = np.sum(self.residuals ** 2)
        sst = np.sum((Y_valid - Y_valid.mean()) ** 2)
        self.r2 = 1 - self.ssr / sst if sst > 0 else 0

        return self

    def predict(
        self,
        X_hf_new: pd.Series,
        X_lf_new: Optional[pd.DataFrame] = None,
        dates_new: Optional[pd.DatetimeIndex] = None
    ) -> np.ndarray:
        """
        Generate predictions for new data.

        Parameters
        ----------
        X_hf_new : pd.Series
            New high-frequency data
        X_lf_new : pd.DataFrame, optional
            New low-frequency regressors
        dates_new : pd.DatetimeIndex, optional
            Dates for predictions (default: inferred from X_hf_new)

        Returns
        -------
        predictions : np.ndarray
        """
        if self.beta is None:
            raise ValueError("Model must be fit before predicting")

        if dates_new is None:
            # Infer monthly dates from HF data
            dates_new = pd.DatetimeIndex(
                X_hf_new.resample('ME').last().index
            )

        # Aggregate HF data
        X_agg = self._aggregate_hf_data(X_hf_new, dates_new, self.weights)

        # Build design matrix
        if X_lf_new is not None:
            X_combined = np.column_stack([X_agg, X_lf_new.values])
        else:
            X_combined = X_agg

        X = np.column_stack([np.ones(len(dates_new)), X_combined])

        # Predict
        predictions = X @ self.beta

        return predictions

    def forecast(
        self,
        h: int = 1,
        X_hf_future: Optional[pd.Series] = None,
        X_lf_future: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """
        Generate h-step ahead forecasts.

        If future HF data is not available, uses last observed values.

        Parameters
        ----------
        h : int
            Forecast horizon (months)
        X_hf_future : pd.Series, optional
            Future high-frequency data (if available)
        X_lf_future : pd.DataFrame, optional
            Future low-frequency regressors

        Returns
        -------
        forecasts : np.ndarray
        """
        if self.beta is None:
            raise ValueError("Model must be fit before forecasting")

        forecasts = np.zeros(h)
        last_date = self.Y_lf.index[-1]

        for t in range(h):
            future_date = last_date + pd.DateOffset(months=t + 1)

            if X_hf_future is not None:
                # Use provided future HF data
                X_agg = self._aggregate_hf_data(
                    X_hf_future,
                    pd.DatetimeIndex([future_date]),
                    self.weights
                )[0]
            else:
                # Use last observed HF period (naive persistence)
                X_agg = self._aggregate_hf_data(
                    self.X_hf,
                    pd.DatetimeIndex([last_date]),
                    self.weights
                )[0]

            # Build regressor vector
            x_t = np.concatenate([[1], X_agg])

            if X_lf_future is not None and t < len(X_lf_future):
                x_t = np.concatenate([x_t, X_lf_future.iloc[t].values])
            elif self.X_lf is not None:
                x_t = np.concatenate([x_t, self.X_lf.iloc[-1].values])

            forecasts[t] = np.dot(x_t, self.beta)

        return forecasts

    def get_weight_plot_data(self) -> Dict[str, Any]:
        """Return data for plotting MIDAS weights."""
        return {
            "weights": self.weights,
            "lags": np.arange(1, self.n_hf_lags + 1),
            "theta": self.theta,
            "weight_type": self.weight_type
        }

    def summary(self) -> Dict[str, Any]:
        """Return model summary statistics."""
        return {
            "weight_type": self.weight_type,
            "n_hf_lags": self.n_hf_lags,
            "n_lf_lags": self.n_lf_lags,
            "theta": self.theta.tolist() if self.theta is not None else None,
            "intercept": float(self.intercept) if self.intercept is not None else None,
            "hf_coefs": self.hf_coefs.tolist() if self.hf_coefs is not None else None,
            "n_obs": int(self.n_obs) if self.n_obs is not None else None,
            "r2": float(self.r2) if self.r2 is not None else None,
            "ssr": float(self.ssr) if self.ssr is not None else None,
            "rmse": float(np.sqrt(self.ssr / self.n_obs)) if self.ssr is not None else None,
        }


class MIDAS_AR(MIDAS):
    """
    MIDAS with Autoregressive low-frequency lags.

    R_t = alpha + rho_1*R_{t-1} + ... + rho_p*R_{t-p} + beta*B(L)*X_t^{(d)} + eps_t

    Parameters
    ----------
    n_ar_lags : int
        Number of AR lags (default: 1)
    **midas_kwargs
        Additional arguments passed to MIDAS
    """

    def __init__(self, n_ar_lags: int = 1, **midas_kwargs):
        super().__init__(**midas_kwargs)
        self.n_ar_lags = n_ar_lags
        self.ar_coefs = None

    def fit(
        self,
        Y_lf: pd.Series,
        X_hf: pd.Series,
        X_lf: Optional[pd.DataFrame] = None,
        optimize: bool = True
    ):
        """
        Fit MIDAS-AR model.

        Adds lagged values of Y as additional low-frequency regressors.
        """
        # Create AR lags
        ar_lags = pd.concat(
            [Y_lf.shift(i).rename(f"Y_lag{i}") for i in range(1, self.n_ar_lags + 1)],
            axis=1
        )

        # Combine with existing LF regressors
        if X_lf is not None:
            X_lf_combined = pd.concat([X_lf, ar_lags], axis=1)
        else:
            X_lf_combined = ar_lags

        # Fit parent model
        super().fit(Y_lf, X_hf, X_lf_combined, optimize)

        # Extract AR coefficients
        n_hf_coefs = self.n_lf_lags + 2  # intercept + HF aggregates
        n_other_lf = 0 if X_lf is None else X_lf.shape[1]
        ar_start = n_hf_coefs + n_other_lf
        self.ar_coefs = self.beta[ar_start:ar_start + self.n_ar_lags]

        return self

    def summary(self) -> Dict[str, Any]:
        """Return model summary with AR coefficients."""
        base_summary = super().summary()
        base_summary["n_ar_lags"] = self.n_ar_lags
        base_summary["ar_coefs"] = self.ar_coefs.tolist() if self.ar_coefs is not None else None
        return base_summary


class UMIDAS:
    """
    Unrestricted MIDAS - direct OLS on blocked high-frequency lags.

    When n_hf_lags is small enough, can estimate each lag coefficient
    directly without polynomial restrictions.

    Parameters
    ----------
    n_hf_blocks : int
        Number of HF blocks (each block gets one coefficient)
    block_size : int
        Number of HF lags per block
    n_lf_lags : int
        Number of low-frequency lag periods
    """

    def __init__(
        self,
        n_hf_blocks: int = 4,
        block_size: int = 5,
        n_lf_lags: int = 1
    ):
        self.n_hf_blocks = n_hf_blocks
        self.block_size = block_size
        self.n_lf_lags = n_lf_lags
        self.n_hf_lags = n_hf_blocks * block_size

        self.beta = None
        self.block_coefs = None
        self.fitted = None
        self.residuals = None
        self.r2 = None

    def _aggregate_blocks(
        self,
        X_hf: pd.Series,
        dates_lf: pd.DatetimeIndex
    ) -> np.ndarray:
        """Aggregate HF data into blocks (simple averages within each block)."""
        n_lf = len(dates_lf)
        n_cols = (self.n_lf_lags + 1) * self.n_hf_blocks
        X_blocks = np.zeros((n_lf, n_cols))

        for t, date_lf in enumerate(dates_lf):
            for lag in range(self.n_lf_lags + 1):
                lag_date = date_lf - pd.DateOffset(months=lag)
                month_start = lag_date.replace(day=1)
                month_end = (month_start + pd.DateOffset(months=1)) - pd.DateOffset(days=1)

                mask = (X_hf.index >= month_start) & (X_hf.index <= month_end)
                hf_values = X_hf[mask].values

                if len(hf_values) >= self.n_hf_lags:
                    hf_values = hf_values[-self.n_hf_lags:]

                    # Average within each block
                    for b in range(self.n_hf_blocks):
                        start = b * self.block_size
                        end = min((b + 1) * self.block_size, len(hf_values))
                        col_idx = lag * self.n_hf_blocks + b
                        X_blocks[t, col_idx] = np.mean(hf_values[start:end])

                elif len(hf_values) > 0:
                    # Proportional blocks for shorter months
                    n_per_block = max(1, len(hf_values) // self.n_hf_blocks)
                    for b in range(self.n_hf_blocks):
                        start = b * n_per_block
                        end = min((b + 1) * n_per_block, len(hf_values))
                        col_idx = lag * self.n_hf_blocks + b
                        if start < len(hf_values):
                            X_blocks[t, col_idx] = np.mean(hf_values[start:end])
                        else:
                            X_blocks[t, col_idx] = np.nan
                else:
                    for b in range(self.n_hf_blocks):
                        col_idx = lag * self.n_hf_blocks + b
                        X_blocks[t, col_idx] = np.nan

        return X_blocks

    def fit(
        self,
        Y_lf: pd.Series,
        X_hf: pd.Series,
        X_lf: Optional[pd.DataFrame] = None
    ):
        """Fit U-MIDAS by OLS."""
        self.Y_lf = Y_lf
        self.X_hf = X_hf
        self.X_lf = X_lf

        # Build design matrix
        X_blocks = self._aggregate_blocks(X_hf, Y_lf.index)

        if X_lf is not None:
            X = np.column_stack([np.ones(len(Y_lf)), X_blocks, X_lf.values])
        else:
            X = np.column_stack([np.ones(len(Y_lf)), X_blocks])

        Y = Y_lf.values
        valid = ~np.isnan(X).any(axis=1) & ~np.isnan(Y)

        X_valid = X[valid]
        Y_valid = Y[valid]

        # OLS
        self.beta, _, _, _ = np.linalg.lstsq(X_valid, Y_valid, rcond=None)
        self.block_coefs = self.beta[1:1 + (self.n_lf_lags + 1) * self.n_hf_blocks]

        # Fitted values
        self.fitted = X_valid @ self.beta
        self.residuals = Y_valid - self.fitted

        # R-squared
        sst = np.sum((Y_valid - Y_valid.mean()) ** 2)
        ssr = np.sum(self.residuals ** 2)
        self.r2 = 1 - ssr / sst if sst > 0 else 0
        self.n_obs = valid.sum()

        return self

    def summary(self) -> Dict[str, Any]:
        """Return model summary."""
        return {
            "model": "U-MIDAS",
            "n_hf_blocks": self.n_hf_blocks,
            "block_size": self.block_size,
            "n_lf_lags": self.n_lf_lags,
            "n_obs": int(self.n_obs),
            "r2": float(self.r2),
            "block_coefs": self.block_coefs.tolist() if self.block_coefs is not None else None,
        }


def prepare_hf_exchange_rate(
    daily_df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    compute_returns: bool = True
) -> pd.Series:
    """
    Prepare high-frequency exchange rate data for MIDAS.

    Parameters
    ----------
    daily_df : pd.DataFrame
        DataFrame with 'usd_lkr' column and DatetimeIndex
    start_date : str, optional
        Start date for filtering
    end_date : str, optional
        End date for filtering
    compute_returns : bool
        If True, compute log returns for stationarity

    Returns
    -------
    pd.Series
        Prepared HF series (returns or levels)
    """
    if isinstance(daily_df, pd.Series):
        fx = daily_df.copy()
    else:
        fx = daily_df['usd_lkr'].copy()

    # Filter date range
    if start_date:
        fx = fx[fx.index >= start_date]
    if end_date:
        fx = fx[fx.index <= end_date]

    # Resample to daily and forward-fill weekends/holidays
    fx = fx.resample('D').ffill()

    if compute_returns:
        # Log returns
        fx_ret = np.log(fx).diff().dropna()
        fx_ret.name = 'usd_lkr_return'
        return fx_ret
    else:
        return fx


def align_midas_data(
    Y_monthly: pd.Series,
    X_daily: pd.Series,
    X_monthly_exog: Optional[pd.DataFrame] = None
) -> Tuple[pd.Series, pd.Series, Optional[pd.DataFrame]]:
    """
    Align monthly target with daily regressors for MIDAS.

    Ensures proper date handling:
    - Monthly data is normalized to end-of-month
    - Daily data covers required lag periods

    Parameters
    ----------
    Y_monthly : pd.Series
        Monthly target variable
    X_daily : pd.Series
        Daily regressor
    X_monthly_exog : pd.DataFrame, optional
        Additional monthly regressors

    Returns
    -------
    Y_aligned : pd.Series
    X_daily_aligned : pd.Series
    X_monthly_aligned : pd.DataFrame or None
    """
    # Normalize monthly to end-of-month
    Y_aligned = Y_monthly.copy()
    Y_aligned.index = Y_aligned.index.to_period('M').to_timestamp('ME')

    # Daily should cover all monthly periods plus lags
    first_month = Y_aligned.index.min() - pd.DateOffset(months=3)
    last_day = Y_aligned.index.max()

    X_daily_aligned = X_daily.loc[first_month:last_day]

    # Align monthly exog
    if X_monthly_exog is not None:
        X_monthly_aligned = X_monthly_exog.copy()
        X_monthly_aligned.index = X_monthly_aligned.index.to_period('M').to_timestamp('ME')
        X_monthly_aligned = X_monthly_aligned.reindex(Y_aligned.index)
    else:
        X_monthly_aligned = None

    return Y_aligned, X_daily_aligned, X_monthly_aligned


def midas_information_gain(midas_rmse: float, monthly_rmse: float) -> float:
    """
    Compute relative RMSE improvement from using high-frequency data.

    Parameters
    ----------
    midas_rmse : float
        RMSE of MIDAS model
    monthly_rmse : float
        RMSE of monthly-only model

    Returns
    -------
    improvement_pct : float
        Percentage improvement (positive = MIDAS better)
    """
    if monthly_rmse == 0:
        return 0.0
    return (monthly_rmse - midas_rmse) / monthly_rmse * 100
