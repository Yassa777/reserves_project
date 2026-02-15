"""
Threshold Vector Autoregression (TVAR) Model Implementation.

Implements a two-regime TVAR where regime switches are determined by an
observable threshold variable (e.g., exchange rate depreciation).

Reference: Specification 06 - Threshold VAR
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from scipy import stats
from statsmodels.tsa.api import VAR
import warnings


class ThresholdVAR:
    """
    Threshold Vector Autoregression with two regimes.

    Estimation via concentrated maximum likelihood:
    1. Grid search over threshold tau
    2. OLS estimation conditional on tau
    3. Select tau minimizing total SSR

    Model:
        Y_t = Phi_1(L) Y_{t-1} + eps_1_t   if z_{t-d} <= tau  (Regime 1: Stable)
        Y_t = Phi_2(L) Y_{t-1} + eps_2_t   if z_{t-d} > tau   (Regime 2: Crisis)

    Parameters
    ----------
    n_lags : int, default=2
        VAR lag order
    delay : int, default=1
        Threshold delay (d in z_{t-d})
    trim : float, default=0.15
        Trimming fraction (exclude extreme tau values from grid search)
    min_obs_per_regime : int, default=24
        Minimum observations required per regime (2 years monthly data)
    """

    def __init__(
        self,
        n_lags: int = 2,
        delay: int = 1,
        trim: float = 0.15,
        min_obs_per_regime: int = 24
    ):
        self.n_lags = n_lags
        self.delay = delay
        self.trim = trim
        self.min_obs_per_regime = min_obs_per_regime

        # Fitted attributes
        self.threshold: Optional[float] = None
        self.var_regime1: Optional[VAR] = None
        self.var_regime2: Optional[VAR] = None
        self.regime_indicators: Optional[pd.Series] = None
        self.Y: Optional[pd.DataFrame] = None
        self.z: Optional[pd.Series] = None
        self.threshold_var_name: str = "threshold"
        self.regime_stats: Optional[Dict[str, Any]] = None
        self.grid_search_results: Optional[Dict[str, Any]] = None
        self._fitted: bool = False

    def fit(
        self,
        Y: pd.DataFrame,
        z: pd.Series,
        threshold_var_name: str = "threshold"
    ) -> "ThresholdVAR":
        """
        Fit TVAR model.

        Parameters
        ----------
        Y : pd.DataFrame
            Multivariate system (T x k)
        z : pd.Series
            Threshold variable (e.g., exchange rate % change)
        threshold_var_name : str, default="threshold"
            Name of the threshold variable for labeling

        Returns
        -------
        self : ThresholdVAR
            Fitted model instance
        """
        # Align data on common index
        common_idx = Y.index.intersection(z.index)
        if len(common_idx) == 0:
            raise ValueError("No overlapping dates between Y and z")

        Y = Y.loc[common_idx].copy()
        z = z.loc[common_idx].copy()

        T = len(Y)
        k = Y.shape[1]

        # Create lagged threshold variable
        z_lagged = z.shift(self.delay).dropna()
        valid_idx = z_lagged.index
        Y = Y.loc[valid_idx]
        z_lagged = z_lagged.loc[valid_idx]

        # Grid search for threshold
        # Use a more robust approach: only include thresholds that guarantee
        # at least min_obs observations in each regime
        sorted_z = np.sort(z_lagged.values)
        n_grid = len(sorted_z)

        # Minimum observations per regime
        min_obs = max(
            self.min_obs_per_regime,
            (k * self.n_lags + 1) * 2  # At least 2x parameters
        )

        # Only consider thresholds where both regimes have >= min_obs
        # This automatically trims the appropriate amount
        valid_thresholds = []
        for i, tau in enumerate(sorted_z):
            n_below = i + 1
            n_above = n_grid - i - 1
            if n_below >= min_obs and n_above >= min_obs:
                valid_thresholds.append(tau)

        # Further trim by specified fraction if there are many valid points
        if len(valid_thresholds) > 20:
            trim_n = int(len(valid_thresholds) * self.trim)
            threshold_grid = np.array(valid_thresholds[trim_n:-trim_n] if trim_n > 0 else valid_thresholds)
        else:
            threshold_grid = np.array(valid_thresholds)

        if len(threshold_grid) == 0:
            raise ValueError(
                f"No valid thresholds found. "
                f"Need at least {min_obs} observations per regime."
            )

        best_ssr = np.inf
        best_threshold = None
        best_var1 = None
        best_var2 = None
        ssr_values = []

        for tau in threshold_grid:
            # Split data by regime
            regime1_mask = z_lagged <= tau
            regime2_mask = z_lagged > tau

            n1 = regime1_mask.sum()
            n2 = regime2_mask.sum()

            if n1 < min_obs or n2 < min_obs:
                ssr_values.append((tau, np.inf))
                continue

            # Estimate regime-specific VARs
            try:
                Y1 = Y[regime1_mask]
                Y2 = Y[regime2_mask]

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    var1 = VAR(Y1).fit(self.n_lags)
                    var2 = VAR(Y2).fit(self.n_lags)

                # Total SSR - use .values to ensure numpy array
                ssr1 = np.sum(var1.resid.values ** 2)
                ssr2 = np.sum(var2.resid.values ** 2)
                total_ssr = ssr1 + ssr2

                ssr_values.append((tau, total_ssr))

                if total_ssr < best_ssr:
                    best_ssr = total_ssr
                    best_threshold = tau
                    best_var1 = var1
                    best_var2 = var2

            except Exception:
                ssr_values.append((tau, np.inf))
                continue

        if best_threshold is None:
            raise ValueError(
                f"Could not find valid threshold. "
                f"Ensure at least {min_obs} observations per regime."
            )

        # Store results
        self.threshold = best_threshold
        self.var_regime1 = best_var1
        self.var_regime2 = best_var2
        self.regime_indicators = (z_lagged > best_threshold).astype(int)
        self.regime_indicators.name = "regime"
        self.Y = Y
        self.z = z_lagged
        self.threshold_var_name = threshold_var_name

        # Store grid search results
        self.grid_search_results = {
            "threshold_grid": threshold_grid,
            "ssr_values": ssr_values,
            "best_ssr": best_ssr,
        }

        # Compute regime statistics
        self._compute_regime_stats()
        self._fitted = True

        return self

    def _compute_regime_stats(self) -> None:
        """Compute regime-specific statistics."""
        regime1_mask = self.regime_indicators == 0
        regime2_mask = self.regime_indicators == 1

        self.regime_stats = {
            "regime1": {
                "name": "Stable",
                "condition": f"{self.threshold_var_name} <= {self.threshold:.4f}",
                "n_obs": int(regime1_mask.sum()),
                "pct": float(regime1_mask.mean() * 100),
                "mean_threshold_var": float(self.z[regime1_mask].mean()),
                "std_threshold_var": float(self.z[regime1_mask].std()),
                "start_date": str(self.z[regime1_mask].index.min().date()) if regime1_mask.any() else None,
                "end_date": str(self.z[regime1_mask].index.max().date()) if regime1_mask.any() else None,
            },
            "regime2": {
                "name": "Crisis",
                "condition": f"{self.threshold_var_name} > {self.threshold:.4f}",
                "n_obs": int(regime2_mask.sum()),
                "pct": float(regime2_mask.mean() * 100),
                "mean_threshold_var": float(self.z[regime2_mask].mean()),
                "std_threshold_var": float(self.z[regime2_mask].std()),
                "start_date": str(self.z[regime2_mask].index.min().date()) if regime2_mask.any() else None,
                "end_date": str(self.z[regime2_mask].index.max().date()) if regime2_mask.any() else None,
            },
            "threshold": float(self.threshold),
            "threshold_var": self.threshold_var_name,
            "total_obs": len(self.regime_indicators),
        }

        # Add regime transition statistics
        transitions = np.diff(self.regime_indicators.values)
        n_transitions = np.sum(transitions != 0)
        self.regime_stats["n_transitions"] = int(n_transitions)

        # Average regime duration
        if n_transitions > 0:
            avg_duration = len(self.regime_indicators) / (n_transitions + 1)
            self.regime_stats["avg_regime_duration"] = float(avg_duration)
        else:
            self.regime_stats["avg_regime_duration"] = float(len(self.regime_indicators))

    def forecast(
        self,
        h: int = 12,
        z_future: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate h-step ahead forecasts.

        Parameters
        ----------
        h : int, default=12
            Forecast horizon
        z_future : np.ndarray, optional
            Future values of threshold variable.
            If None, use last observed regime for all forecasts.

        Returns
        -------
        forecasts : np.ndarray (h x k)
            Point forecasts for each horizon
        """
        self._check_fitted()

        if z_future is None:
            # Use last regime for all forecasts
            last_z = self.z.iloc[-1]
            regime = 2 if last_z > self.threshold else 1
            var_model = self.var_regime2 if regime == 2 else self.var_regime1

            # Use that regime's VAR for all h steps
            last_obs = self.Y.values[-self.n_lags:]
            forecasts = var_model.forecast(last_obs, steps=h)

        else:
            # Regime can switch during forecast horizon
            forecasts = np.zeros((h, self.Y.shape[1]))
            current_obs = self.Y.values[-self.n_lags:]

            for t in range(h):
                z_t = z_future[t] if t < len(z_future) else z_future[-1]
                regime = 2 if z_t > self.threshold else 1
                var_model = self.var_regime2 if regime == 2 else self.var_regime1

                # One-step forecast
                fc_t = var_model.forecast(current_obs, steps=1)[0]
                forecasts[t] = fc_t

                # Update lags
                current_obs = np.vstack([current_obs[1:], fc_t])

        return forecasts

    def forecast_by_scenario(self, h: int = 12) -> Dict[str, np.ndarray]:
        """
        Generate forecasts for both regime scenarios.

        Useful for policy analysis: "what if we stay in stable regime"
        vs "what if we enter crisis regime".

        Parameters
        ----------
        h : int, default=12
            Forecast horizon

        Returns
        -------
        dict with keys:
            - "regime1_scenario": forecasts assuming stable regime
            - "regime2_scenario": forecasts assuming crisis regime
        """
        self._check_fitted()

        last_obs = self.Y.values[-self.n_lags:]

        # Scenario 1: Stay in regime 1 (stable)
        fc_regime1 = self.var_regime1.forecast(last_obs, steps=h)

        # Scenario 2: Stay in regime 2 (crisis)
        fc_regime2 = self.var_regime2.forecast(last_obs, steps=h)

        return {
            "regime1_scenario": fc_regime1,
            "regime2_scenario": fc_regime2,
        }

    def forecast_df(
        self,
        h: int = 12,
        z_future: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Generate forecasts as a DataFrame with proper dates.

        Parameters
        ----------
        h : int, default=12
            Forecast horizon
        z_future : np.ndarray, optional
            Future values of threshold variable

        Returns
        -------
        pd.DataFrame
            Forecasts with datetime index
        """
        self._check_fitted()

        forecasts = self.forecast(h=h, z_future=z_future)

        # Create forecast dates
        last_date = self.Y.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=h,
            freq="MS"
        )

        return pd.DataFrame(
            forecasts,
            index=forecast_dates,
            columns=self.Y.columns
        )

    def get_regime_series(self) -> pd.DataFrame:
        """
        Get regime indicators with dates.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: threshold_var, threshold, regime, regime_name
        """
        self._check_fitted()

        df = pd.DataFrame(index=self.z.index)
        df[self.threshold_var_name] = self.z
        df["threshold"] = self.threshold
        df["regime"] = self.regime_indicators
        df["regime_name"] = df["regime"].map({0: "Stable", 1: "Crisis"})

        return df

    def get_regime_periods(self) -> List[Dict[str, Any]]:
        """
        Get list of regime periods with start/end dates.

        Returns
        -------
        list of dicts
            Each dict has: regime, start_date, end_date, duration
        """
        self._check_fitted()

        periods = []
        current_regime = self.regime_indicators.iloc[0]
        start_date = self.regime_indicators.index[0]

        for i in range(1, len(self.regime_indicators)):
            if self.regime_indicators.iloc[i] != current_regime:
                # Regime change
                end_date = self.regime_indicators.index[i - 1]
                periods.append({
                    "regime": int(current_regime),
                    "regime_name": "Crisis" if current_regime == 1 else "Stable",
                    "start_date": str(start_date.date()),
                    "end_date": str(end_date.date()),
                    "duration_months": i - len(periods),
                })
                current_regime = self.regime_indicators.iloc[i]
                start_date = self.regime_indicators.index[i]

        # Add final period
        periods.append({
            "regime": int(current_regime),
            "regime_name": "Crisis" if current_regime == 1 else "Stable",
            "start_date": str(start_date.date()),
            "end_date": str(self.regime_indicators.index[-1].date()),
            "duration_months": len(self.regime_indicators) - sum(p["duration_months"] for p in periods),
        })

        return periods

    def summary(self) -> Dict[str, Any]:
        """
        Generate model summary.

        Returns
        -------
        dict
            Comprehensive model summary
        """
        self._check_fitted()

        return {
            "model": "Threshold VAR",
            "n_lags": self.n_lags,
            "delay": self.delay,
            "threshold": float(self.threshold),
            "threshold_variable": self.threshold_var_name,
            "n_variables": self.Y.shape[1],
            "variables": self.Y.columns.tolist(),
            "total_observations": len(self.Y),
            "regime_stats": self.regime_stats,
            "regime1_aic": float(self.var_regime1.aic),
            "regime1_bic": float(self.var_regime1.bic),
            "regime2_aic": float(self.var_regime2.aic),
            "regime2_bic": float(self.var_regime2.bic),
        }

    def _check_fitted(self) -> None:
        """Check if model is fitted."""
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")


def compute_threshold_variable(
    df: pd.DataFrame,
    var_name: str = "usd_lkr",
    method: str = "pct_change"
) -> pd.Series:
    """
    Compute threshold variable from exchange rate.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing exchange rate
    var_name : str, default="usd_lkr"
        Name of exchange rate column
    method : str, default="pct_change"
        Method to compute threshold variable:
        - "pct_change": Monthly percentage change
        - "log_diff": Log difference (approximately equal for small changes)
        - "diff": Simple first difference

    Returns
    -------
    pd.Series
        Threshold variable
    """
    if var_name not in df.columns:
        raise ValueError(f"Column '{var_name}' not found in DataFrame")

    series = df[var_name]

    if method == "pct_change":
        z = series.pct_change() * 100  # Percentage change
        z.name = f"{var_name}_pct_change"
    elif method == "log_diff":
        z = np.log(series).diff() * 100
        z.name = f"{var_name}_log_diff"
    elif method == "diff":
        z = series.diff()
        z.name = f"{var_name}_diff"
    else:
        raise ValueError(f"Unknown method: {method}")

    return z.dropna()


def load_threshold_variable_from_fx(
    fx_path: str,
    method: str = "pct_change"
) -> pd.Series:
    """
    Load and compute threshold variable from historical FX data.

    Parameters
    ----------
    fx_path : str
        Path to historical_fx.csv
    method : str, default="pct_change"
        Method to compute threshold variable

    Returns
    -------
    pd.Series
        Threshold variable with datetime index
    """
    import os
    from pathlib import Path

    fx_path = Path(fx_path)
    if not fx_path.exists():
        raise FileNotFoundError(f"FX data not found: {fx_path}")

    fx = pd.read_csv(fx_path, parse_dates=["date"], index_col="date")
    fx.index = fx.index.to_period("M").to_timestamp()
    fx = fx.sort_index()

    return compute_threshold_variable(fx, "usd_lkr", method)
