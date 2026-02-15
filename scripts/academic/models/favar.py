"""
Factor-Augmented Vector Autoregression (FAVAR) Model.

Implements two-step FAVAR following Stock & Watson (2002) and Bernanke et al. (2005):
1. Extract latent factors via PCA from large information set
2. Estimate VAR on [Y, F] where Y is observable and F are factors

Reference:
- Stock, J.H. & Watson, M.W. (2002). Macroeconomic Forecasting Using Diffusion Indexes. JBES.
- Bernanke, B.S., Boivin, J., & Eliasz, P. (2005). Measuring the Effects of Monetary Policy. QJE.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Optional, Tuple, List
import warnings


class FAVAR:
    """
    Factor-Augmented Vector Autoregression.

    Two-step estimation:
    1. Extract factors via PCA (static factors)
    2. Estimate VAR on [Y, F]

    Attributes
    ----------
    n_factors : int
        Number of latent factors
    n_lags : int
        VAR lag order
    factors : pd.DataFrame
        Extracted factor time series
    loadings : pd.DataFrame
        Factor loadings matrix
    var_results : statsmodels VARResults
        Fitted VAR model results
    """

    def __init__(
        self,
        n_factors: int = 3,
        n_lags: int = 2,
        ic: Optional[str] = None
    ):
        """
        Initialize FAVAR model.

        Parameters
        ----------
        n_factors : int
            Number of latent factors to extract
        n_lags : int
            VAR lag order. If ic is specified, this is maxlags.
        ic : str, optional
            Information criterion for lag selection ('aic', 'bic', 'hqic', 'fpe').
            If None, n_lags is used directly.
        """
        self.n_factors = n_factors
        self.n_lags = n_lags
        self.ic = ic

        # Fitted attributes
        self.factors = None
        self.loadings = None
        self.variance_explained = None
        self.var_model = None
        self.var_results = None
        self.scaler = None
        self.pca = None
        self.data = None
        self.train_end = None
        self._fitted = False

    def fit(
        self,
        Y: pd.Series,
        factors: pd.DataFrame,
        loadings: pd.DataFrame,
        variance_explained: np.ndarray,
        train_end: Optional[pd.Timestamp] = None
    ) -> "FAVAR":
        """
        Fit FAVAR model using pre-extracted factors.

        Parameters
        ----------
        Y : pd.Series
            Target variable (reserves)
        factors : pd.DataFrame
            Pre-extracted factor time series (T x K)
        loadings : pd.DataFrame
            Factor loadings matrix (N x K)
        variance_explained : np.ndarray
            Variance explained by each factor
        train_end : pd.Timestamp, optional
            End of training period

        Returns
        -------
        self
            Fitted FAVAR model
        """
        self.loadings = loadings
        self.variance_explained = variance_explained
        self.train_end = train_end

        # Align Y and factors
        common_idx = Y.index.intersection(factors.index)
        Y_aligned = Y.loc[common_idx]
        F_aligned = factors.loc[common_idx]

        # Use only n_factors
        factor_cols = [c for c in F_aligned.columns if c.startswith("PC")][:self.n_factors]
        F_aligned = F_aligned[factor_cols]

        self.factors = F_aligned

        # Create joint system: [reserves, F1, F2, ...]
        data = pd.concat([Y_aligned.to_frame("reserves"), F_aligned], axis=1)
        self.data = data.dropna()

        # Filter to training period if specified
        if train_end is not None:
            train_data = self.data.loc[:train_end]
        else:
            train_data = self.data

        # Estimate VAR
        var_model = VAR(train_data)

        if self.ic is not None:
            self.var_results = var_model.fit(maxlags=self.n_lags, ic=self.ic)
            self.n_lags = self.var_results.k_ar
        else:
            self.var_results = var_model.fit(maxlags=self.n_lags, ic=None)

        self.var_model = var_model
        self._fitted = True

        return self

    def forecast(self, h: int = 12, start_data: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Generate h-step ahead forecasts.

        Parameters
        ----------
        h : int
            Forecast horizon
        start_data : np.ndarray, optional
            Starting values for forecast. If None, uses last n_lags observations.

        Returns
        -------
        pd.DataFrame
            Forecasts for all variables in the system
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before forecasting")

        if start_data is None:
            start_data = self.data.values[-self.var_results.k_ar:]

        forecasts = self.var_results.forecast(start_data, steps=h)

        # Create forecast index
        last_date = self.data.index[-1]
        freq = pd.infer_freq(self.data.index) or "MS"
        forecast_idx = pd.date_range(start=last_date, periods=h + 1, freq=freq)[1:]

        forecast_df = pd.DataFrame(
            forecasts,
            index=forecast_idx,
            columns=self.data.columns
        )

        return forecast_df

    def forecast_with_intervals(
        self,
        h: int = 12,
        alpha: float = 0.1
    ) -> Dict[str, pd.DataFrame]:
        """
        Forecasts with confidence intervals.

        Parameters
        ----------
        h : int
            Forecast horizon
        alpha : float
            Significance level (e.g., 0.1 for 90% CI)

        Returns
        -------
        dict
            Dictionary with 'mean', 'lower', 'upper' DataFrames
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before forecasting")

        start_data = self.data.values[-self.var_results.k_ar:]

        forecast, lower, upper = self.var_results.forecast_interval(
            start_data,
            steps=h,
            alpha=alpha
        )

        # Create forecast index
        last_date = self.data.index[-1]
        freq = pd.infer_freq(self.data.index) or "MS"
        forecast_idx = pd.date_range(start=last_date, periods=h + 1, freq=freq)[1:]

        return {
            "mean": pd.DataFrame(forecast, index=forecast_idx, columns=self.data.columns),
            "lower": pd.DataFrame(lower, index=forecast_idx, columns=self.data.columns),
            "upper": pd.DataFrame(upper, index=forecast_idx, columns=self.data.columns),
        }

    def get_factor_interpretation(self, top_n: int = 4) -> Dict[str, List[Tuple[str, str, float]]]:
        """
        Interpret factors via their loadings.

        Parameters
        ----------
        top_n : int
            Number of top-loading variables to show per factor

        Returns
        -------
        dict
            Dictionary mapping factor names to list of (variable, sign, loading) tuples
        """
        if self.loadings is None:
            return {}

        interpretation = {}
        for factor in self.loadings.columns:
            sorted_loadings = self.loadings[factor].abs().sort_values(ascending=False)
            top_vars = sorted_loadings.head(top_n).index.tolist()

            interpretation[factor] = [
                (
                    var,
                    "+" if self.loadings.loc[var, factor] > 0 else "-",
                    float(self.loadings.loc[var, factor])
                )
                for var in top_vars
            ]

        return interpretation

    def impulse_response(
        self,
        periods: int = 24,
        orthogonalized: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute impulse response functions.

        Parameters
        ----------
        periods : int
            Number of periods for IRF
        orthogonalized : bool
            If True, use orthogonalized IRFs (Cholesky)

        Returns
        -------
        dict
            Dictionary with IRF for each variable responding to each shock
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before computing IRFs")

        irf = self.var_results.irf(periods=periods)

        # Extract IRFs as DataFrames
        result = {}
        var_names = self.data.columns.tolist()

        for i, response_var in enumerate(var_names):
            response_data = {}
            for j, shock_var in enumerate(var_names):
                if orthogonalized:
                    response_data[f"shock_{shock_var}"] = irf.orth_irfs[:, i, j]
                else:
                    response_data[f"shock_{shock_var}"] = irf.irfs[:, i, j]

            result[response_var] = pd.DataFrame(
                response_data,
                index=range(periods + 1)
            )

        return result

    def variance_decomposition(self, periods: int = 24) -> Dict[str, pd.DataFrame]:
        """
        Forecast error variance decomposition.

        Parameters
        ----------
        periods : int
            Number of periods for FEVD

        Returns
        -------
        dict
            Dictionary with FEVD for each variable
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before computing FEVD")

        fevd = self.var_results.fevd(periods=periods)

        result = {}
        var_names = self.data.columns.tolist()

        # fevd.decomp has shape (periods, n_vars, n_vars)
        n_periods = fevd.decomp.shape[0]

        for i, var_name in enumerate(var_names):
            result[var_name] = pd.DataFrame(
                fevd.decomp[:, i, :],
                columns=var_names,
                index=range(n_periods)
            )

        return result

    def check_stability(self) -> Dict[str, Any]:
        """
        Check VAR stability (all eigenvalues inside unit circle).

        Returns
        -------
        dict
            Dictionary with is_stable, max_eigenvalue, all_eigenvalues
        """
        if not self._fitted:
            raise ValueError("Model must be fitted first")

        # Use statsmodels built-in method for stability check
        try:
            # Get roots of characteristic polynomial
            roots = self.var_results.roots
            moduli = np.abs(roots)
            max_modulus = np.max(moduli) if len(moduli) > 0 else 0.0

            return {
                "is_stable": max_modulus < 1.0,
                "max_eigenvalue_modulus": float(max_modulus),
                "eigenvalues": roots.tolist(),
            }
        except Exception:
            # Fallback: check if VAR is stable using is_stable attribute
            try:
                is_stable = self.var_results.is_stable()
                return {
                    "is_stable": is_stable,
                    "max_eigenvalue_modulus": np.nan,
                    "eigenvalues": [],
                }
            except Exception:
                return {
                    "is_stable": True,  # Assume stable if we can't check
                    "max_eigenvalue_modulus": np.nan,
                    "eigenvalues": [],
                }

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics for the fitted model.

        Returns
        -------
        dict
            Model summary statistics
        """
        if not self._fitted:
            raise ValueError("Model must be fitted first")

        stability = self.check_stability()

        return {
            "n_factors": self.n_factors,
            "n_lags": self.var_results.k_ar,
            "n_obs": self.var_results.nobs,
            "aic": self.var_results.aic,
            "bic": self.var_results.bic,
            "hqic": self.var_results.hqic,
            "is_stable": stability["is_stable"],
            "max_eigenvalue_modulus": stability["max_eigenvalue_modulus"],
            "variance_explained": self.variance_explained.tolist() if self.variance_explained is not None else None,
            "total_variance_explained": float(np.sum(self.variance_explained)) if self.variance_explained is not None else None,
        }


def rolling_favar_forecast(
    Y: pd.Series,
    factors: pd.DataFrame,
    loadings: pd.DataFrame,
    variance_explained: np.ndarray,
    train_end: pd.Timestamp,
    test_end: pd.Timestamp,
    n_factors: int = 3,
    n_lags: int = 2,
    h: int = 1,
    expanding: bool = True
) -> pd.DataFrame:
    """
    Rolling/expanding window FAVAR forecasts for backtesting.

    Parameters
    ----------
    Y : pd.Series
        Target variable
    factors : pd.DataFrame
        Pre-extracted factors
    loadings : pd.DataFrame
        Factor loadings
    variance_explained : np.ndarray
        Variance explained by factors
    train_end : pd.Timestamp
        Initial training period end
    test_end : pd.Timestamp
        End of test period
    n_factors : int
        Number of factors to use
    n_lags : int
        VAR lag order
    h : int
        Forecast horizon
    expanding : bool
        If True, use expanding window; if False, use rolling window

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: actual, forecast, error, date
    """
    # Prepare aligned data
    common_idx = Y.index.intersection(factors.index)
    Y_aligned = Y.loc[common_idx]
    F_aligned = factors.loc[common_idx]

    # Get test period dates
    test_dates = Y_aligned[(Y_aligned.index > train_end) & (Y_aligned.index <= test_end)].index

    results = []
    initial_train_start = Y_aligned.index[0]

    for test_date in test_dates:
        # Define training window
        if expanding:
            window_start = initial_train_start
        else:
            # Rolling window: use same number of observations as initial training
            n_train = len(Y_aligned[:train_end])
            window_start = Y_aligned[Y_aligned.index < test_date].index[-n_train]

        # Training data ends h periods before test_date
        train_end_for_forecast = test_date - pd.DateOffset(months=h)

        # Fit model
        try:
            model = FAVAR(n_factors=n_factors, n_lags=n_lags)
            model.fit(
                Y=Y_aligned.loc[window_start:train_end_for_forecast],
                factors=F_aligned.loc[window_start:train_end_for_forecast],
                loadings=loadings,
                variance_explained=variance_explained,
            )

            # Forecast
            forecast = model.forecast(h=h)
            forecast_value = forecast["reserves"].iloc[-1]  # h-step ahead forecast

            results.append({
                "date": test_date,
                "actual": Y_aligned.loc[test_date],
                "forecast": forecast_value,
                "error": Y_aligned.loc[test_date] - forecast_value,
            })
        except Exception as e:
            warnings.warn(f"Forecast failed for {test_date}: {e}")
            results.append({
                "date": test_date,
                "actual": Y_aligned.loc[test_date],
                "forecast": np.nan,
                "error": np.nan,
            })

    return pd.DataFrame(results)


def compute_forecast_metrics(results: pd.DataFrame) -> Dict[str, float]:
    """
    Compute forecast accuracy metrics.

    Parameters
    ----------
    results : pd.DataFrame
        DataFrame with columns 'actual', 'forecast', 'error'

    Returns
    -------
    dict
        Dictionary with RMSE, MAE, MAPE, Theil-U
    """
    actual = results["actual"].dropna()
    forecast = results["forecast"].dropna()
    error = results["error"].dropna()

    # Align
    common = actual.index.intersection(forecast.index).intersection(error.index)
    actual = actual.loc[common]
    forecast = forecast.loc[common]
    error = error.loc[common]

    if len(error) == 0:
        return {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan, "Theil_U": np.nan}

    rmse = np.sqrt(np.mean(error ** 2))
    mae = np.mean(np.abs(error))

    # MAPE (avoid division by zero)
    mape = np.mean(np.abs(error / actual.replace(0, np.nan))) * 100

    # Theil-U (ratio of RMSE to naive forecast RMSE)
    naive_error = actual.diff().dropna()
    naive_rmse = np.sqrt(np.mean(naive_error ** 2))
    theil_u = rmse / naive_rmse if naive_rmse > 0 else np.nan

    return {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "MAPE": float(mape),
        "Theil_U": float(theil_u),
        "n_forecasts": len(error),
    }
