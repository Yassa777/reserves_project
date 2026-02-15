"""
Bayesian VAR with Minnesota Prior Implementation.

This module implements a Bayesian Vector Autoregression model with the Minnesota
(Litterman) prior for shrinkage-based multivariate forecasting.

Key features:
- Minnesota prior: own first lag ~ 1, others shrunk toward 0
- Gibbs sampling for full posterior inference
- Direct posterior computation for conjugate case
- Point and density (probabilistic) forecasting

References:
- Litterman, R. (1986). Forecasting with Bayesian VARs.
- Doan, T., Litterman, R., & Sims, C. (1984). Forecasting and Conditional Projection.
- Banbura, M., Giannone, D., & Reichlin, L. (2010). Large Bayesian VARs.

Author: Academic Pipeline
Date: 2026-02-10
"""

import numpy as np
from scipy import linalg
from scipy.stats import invwishart
from typing import Dict, Optional, Tuple, Union, List
import warnings


class BayesianVAR:
    """
    Bayesian VAR with Minnesota Prior.

    The Minnesota prior shrinks VAR coefficients toward a random walk:
    - Own first lag coefficients: prior mean = 1
    - All other coefficients: prior mean = 0
    - Tightness controlled by lambda1 (overall) and lambda3 (lag decay)

    Estimation via Gibbs sampling for full posterior inference.

    Parameters
    ----------
    n_lags : int, default=2
        Number of VAR lags (p)
    lambda1 : float, default=0.2
        Overall tightness parameter. Smaller = more shrinkage.
        Typical values: 0.05 (tight) to 0.5 (loose)
    lambda3 : float, default=1.0
        Lag decay parameter. Controls how fast prior tightens with lag.
        1.0 = linear decay, 2.0 = quadratic decay
    lambda_const : float, default=100.0
        Prior variance on constant term (diffuse)
    n_draws : int, default=5000
        Number of posterior draws from Gibbs sampler
    n_burn : int, default=1000
        Burn-in draws to discard
    random_state : int, optional
        Random seed for reproducibility

    Attributes
    ----------
    coef_mean : np.ndarray
        Posterior mean of coefficients, shape (n_coefs, k)
    coef_std : np.ndarray
        Posterior std of coefficients, shape (n_coefs, k)
    sigma_mean : np.ndarray
        Posterior mean of residual covariance, shape (k, k)
    coef_posterior : np.ndarray
        Full posterior draws of coefficients, shape (n_draws, n_coefs, k)
    sigma_posterior : np.ndarray
        Full posterior draws of covariance, shape (n_draws, k, k)
    k : int
        Number of variables in the VAR
    n_coefs : int
        Number of coefficients per equation (1 + k*p)
    """

    def __init__(
        self,
        n_lags: int = 2,
        lambda1: float = 0.2,
        lambda3: float = 1.0,
        lambda_const: float = 100.0,
        n_draws: int = 5000,
        n_burn: int = 1000,
        random_state: Optional[int] = None
    ):
        self.n_lags = n_lags
        self.lambda1 = lambda1
        self.lambda3 = lambda3
        self.lambda_const = lambda_const
        self.n_draws = n_draws
        self.n_burn = n_burn
        self.random_state = random_state

        # Will be set during fitting
        self.coef_mean = None
        self.coef_std = None
        self.sigma_mean = None
        self.coef_posterior = None
        self.sigma_posterior = None
        self.k = None
        self.n_coefs = None
        self._X = None
        self._Y = None
        self._var_scales = None  # For cross-equation scaling
        self._fitted = False

    def fit(self, Y: np.ndarray, var_names: Optional[List[str]] = None) -> "BayesianVAR":
        """
        Fit BVAR to multivariate time series data.

        Parameters
        ----------
        Y : np.ndarray, shape (T, k)
            Multivariate time series data. Each column is a variable.
        var_names : list of str, optional
            Variable names for interpretability

        Returns
        -------
        self : BayesianVAR
            Fitted model
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        T, k = Y.shape
        self.k = k
        self.var_names = var_names or [f"var_{i}" for i in range(k)]

        # Validate data
        if T <= self.n_lags:
            raise ValueError(f"Time series too short. Need T > {self.n_lags}, got T = {T}")

        # Estimate individual AR(1) residual variances for scaling
        self._var_scales = self._estimate_scales(Y)

        # Create lagged regressor matrices
        Y_lag, Y_obs = self._create_lags(Y)
        T_eff = Y_obs.shape[0]

        # Add constant term
        X = np.column_stack([np.ones(T_eff), Y_lag])
        self.n_coefs = X.shape[1]

        # Store for forecasting
        self._X = X
        self._Y = Y_obs
        self._Y_full = Y

        # Construct Minnesota prior
        prior_mean, prior_var = self._construct_minnesota_prior(k)

        # OLS estimates as starting point
        XtX = X.T @ X
        XtY = X.T @ Y_obs

        # Add small ridge for numerical stability
        XtX_reg = XtX + 1e-8 * np.eye(self.n_coefs)
        B_ols = linalg.solve(XtX_reg, XtY, assume_a='pos')
        residuals = Y_obs - X @ B_ols
        Sigma_ols = (residuals.T @ residuals) / (T_eff - self.n_coefs)

        # Posterior computation (natural conjugate form)
        # Posterior mean is weighted average of prior and OLS
        prior_precision = np.diag(1.0 / np.maximum(prior_var, 1e-10))
        post_precision = prior_precision + XtX
        post_var = linalg.inv(post_precision)
        post_mean = post_var @ (prior_precision @ prior_mean + XtY)

        # Store posterior summaries
        self.coef_mean = post_mean
        self.coef_std = np.sqrt(np.diag(post_var)).reshape(-1, 1).repeat(k, axis=1)
        self.sigma_mean = Sigma_ols

        # Run Gibbs sampler for full posterior
        self._run_gibbs_sampler(X, Y_obs, prior_mean, prior_var)

        self._fitted = True
        return self

    def _estimate_scales(self, Y: np.ndarray) -> np.ndarray:
        """
        Estimate AR(1) residual variances for each variable.
        Used for cross-equation scaling in Minnesota prior.
        """
        T, k = Y.shape
        scales = np.zeros(k)

        for i in range(k):
            y = Y[:, i]
            y_lag = y[:-1]
            y_current = y[1:]

            # Simple AR(1) regression
            beta = np.sum(y_lag * y_current) / np.sum(y_lag ** 2)
            resid = y_current - beta * y_lag
            scales[i] = np.var(resid, ddof=1)

        # Ensure positive scales
        scales = np.maximum(scales, 1e-10)
        return scales

    def _create_lags(self, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create lagged regressor matrix.

        Parameters
        ----------
        Y : np.ndarray, shape (T, k)
            Original time series

        Returns
        -------
        Y_lag : np.ndarray, shape (T-p, k*p)
            Lagged regressors
        Y_obs : np.ndarray, shape (T-p, k)
            Observed values (after losing first p observations)
        """
        T, k = Y.shape
        p = self.n_lags

        Y_lag = np.zeros((T - p, k * p))

        for lag in range(1, p + 1):
            start_col = (lag - 1) * k
            end_col = lag * k
            Y_lag[:, start_col:end_col] = Y[p - lag:T - lag, :]

        Y_obs = Y[p:, :]
        return Y_lag, Y_obs

    def _construct_minnesota_prior(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct Minnesota prior mean and variance.

        The prior specification:
        - Own first lag: mean = 1, variance = (lambda1)^2
        - Other lags/cross-equation: mean = 0, variance scaled by lag and variable variance
        - Constant: diffuse prior (large variance)

        Parameters
        ----------
        k : int
            Number of variables

        Returns
        -------
        prior_mean : np.ndarray, shape (n_coefs, k)
            Prior mean for each coefficient
        prior_var : np.ndarray, shape (n_coefs,)
            Prior variance for each coefficient (same across equations for simplicity)
        """
        n_coefs = self.n_coefs  # 1 + k*p

        # Prior mean: random walk (1 on own first lag, 0 elsewhere)
        prior_mean = np.zeros((n_coefs, k))

        # Set own first lag to 1 for each equation
        for i in range(k):
            # Index of own first lag in regressor matrix
            # Position: 1 (skip constant) + i (own variable in first lag block)
            prior_mean[1 + i, i] = 1.0

        # Prior variance
        prior_var = np.zeros(n_coefs)

        # Diffuse prior on constant
        prior_var[0] = self.lambda_const

        # Variance for lagged coefficients
        for lag in range(1, self.n_lags + 1):
            for j in range(k):
                idx = 1 + (lag - 1) * k + j

                # Minnesota variance formula:
                # For own lags: (lambda1 / lag^lambda3)^2
                # For cross-equation: scaled by relative variance
                # Here we use simplified version (same variance structure for all equations)
                prior_var[idx] = (self.lambda1 / (lag ** self.lambda3)) ** 2

        return prior_mean, prior_var

    def _run_gibbs_sampler(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        prior_mean: np.ndarray,
        prior_var: np.ndarray
    ) -> None:
        """
        Run Gibbs sampler for posterior inference.

        Samples from:
        1. B | Sigma, Y ~ Matrix Normal
        2. Sigma | B, Y ~ Inverse Wishart

        Parameters
        ----------
        X : np.ndarray, shape (T_eff, n_coefs)
            Regressor matrix
        Y : np.ndarray, shape (T_eff, k)
            Dependent variable matrix
        prior_mean : np.ndarray
            Prior mean for coefficients
        prior_var : np.ndarray
            Prior variance (diagonal)
        """
        T, k = Y.shape
        n_coefs = X.shape[1]

        # Storage for posterior draws
        B_draws = np.zeros((self.n_draws, n_coefs, k))
        Sigma_draws = np.zeros((self.n_draws, k, k))

        # Initialize at posterior mean
        B = self.coef_mean.copy()
        Sigma = self.sigma_mean.copy()

        # Prior precision matrix (diagonal)
        prior_precision_diag = 1.0 / np.maximum(prior_var, 1e-10)

        total_draws = self.n_burn + self.n_draws

        for draw in range(total_draws):
            # =============================================
            # Step 1: Draw B | Sigma, Y
            # =============================================
            try:
                Sigma_inv = linalg.inv(Sigma + 1e-8 * np.eye(k))
            except linalg.LinAlgError:
                Sigma_inv = np.eye(k)

            XtX = X.T @ X

            # For efficiency, we draw equation by equation
            # (exploiting diagonal prior structure)
            for i in range(k):
                # Posterior for i-th equation
                post_precision_i = np.diag(prior_precision_diag) * Sigma[i, i] + XtX
                try:
                    post_var_i = linalg.inv(post_precision_i)
                except linalg.LinAlgError:
                    post_var_i = np.eye(n_coefs) * 0.01

                post_mean_i = post_var_i @ (
                    np.diag(prior_precision_diag) * Sigma[i, i] @ prior_mean[:, i] +
                    X.T @ Y[:, i]
                )

                # Draw from multivariate normal
                try:
                    B[:, i] = np.random.multivariate_normal(post_mean_i, post_var_i)
                except (linalg.LinAlgError, ValueError):
                    # Fall back to diagonal approximation if covariance is singular
                    B[:, i] = post_mean_i + np.sqrt(np.diag(post_var_i)) * np.random.randn(n_coefs)

            # =============================================
            # Step 2: Draw Sigma | B, Y (Inverse Wishart)
            # =============================================
            residuals = Y - X @ B
            scale_matrix = residuals.T @ residuals

            # Add small regularization for numerical stability
            scale_matrix += 1e-6 * np.eye(k)

            # Degrees of freedom
            df = T - n_coefs + k + 1  # Ensure df > k - 1
            df = max(df, k + 1)

            try:
                Sigma = invwishart.rvs(df=df, scale=scale_matrix)
            except (ValueError, linalg.LinAlgError):
                # Keep previous Sigma if sampling fails
                pass

            # Ensure Sigma is symmetric
            Sigma = (Sigma + Sigma.T) / 2

            # =============================================
            # Store draws after burn-in
            # =============================================
            if draw >= self.n_burn:
                B_draws[draw - self.n_burn] = B
                Sigma_draws[draw - self.n_burn] = Sigma

        self.coef_posterior = B_draws
        self.sigma_posterior = Sigma_draws

        # Update posterior summaries from Gibbs draws
        self.coef_mean = np.mean(B_draws, axis=0)
        self.coef_std = np.std(B_draws, axis=0)
        self.sigma_mean = np.mean(Sigma_draws, axis=0)

    def forecast(
        self,
        h: int = 12,
        return_draws: bool = False,
        include_shock: bool = True
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Generate h-step ahead forecasts.

        Parameters
        ----------
        h : int, default=12
            Forecast horizon
        return_draws : bool, default=False
            If True, return all posterior draws
            If False, return summary statistics (mean and percentiles)
        include_shock : bool, default=True
            If True, include random shocks in simulation (density forecast)
            If False, generate deterministic forecasts from each draw

        Returns
        -------
        If return_draws is True:
            np.ndarray, shape (n_draws, h, k) - All forecast draws
        If return_draws is False:
            dict with keys:
                - 'mean': np.ndarray, shape (h, k) - Point forecasts
                - 'lower_10', 'upper_90': 80% prediction interval
                - 'lower_5', 'upper_95': 90% prediction interval
                - 'lower_2.5', 'upper_97.5': 95% prediction interval
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before forecasting")

        n_draws = self.coef_posterior.shape[0]
        k = self.k
        p = self.n_lags

        # Get last p observations for conditioning (in reverse order for lag structure)
        last_obs = self._Y_full[-p:][::-1].flatten()  # Shape: (k*p,)

        forecast_draws = np.zeros((n_draws, h, k))

        for d in range(n_draws):
            B = self.coef_posterior[d]  # Shape: (n_coefs, k)
            Sigma = self.sigma_posterior[d]  # Shape: (k, k)

            # Current lag values (will be updated iteratively)
            current_lags = last_obs.copy()

            for t in range(h):
                # Construct regressor: [1, lag1_var1, lag1_var2, ..., lag_p_var_k]
                x_t = np.concatenate([[1.0], current_lags[:k * p]])

                # Point forecast for this draw
                mean_t = x_t @ B

                if include_shock:
                    # Add random shock from error distribution
                    try:
                        shock = np.random.multivariate_normal(np.zeros(k), Sigma)
                    except (ValueError, linalg.LinAlgError):
                        shock = np.zeros(k)
                    forecast_t = mean_t + shock
                else:
                    forecast_t = mean_t

                forecast_draws[d, t] = forecast_t

                # Update lags: shift and insert new forecast
                current_lags = np.concatenate([forecast_t, current_lags[:-k]])

        if return_draws:
            return forecast_draws
        else:
            return {
                'mean': np.mean(forecast_draws, axis=0),
                'median': np.median(forecast_draws, axis=0),
                'lower_2.5': np.percentile(forecast_draws, 2.5, axis=0),
                'lower_5': np.percentile(forecast_draws, 5, axis=0),
                'lower_10': np.percentile(forecast_draws, 10, axis=0),
                'lower_25': np.percentile(forecast_draws, 25, axis=0),
                'upper_75': np.percentile(forecast_draws, 75, axis=0),
                'upper_90': np.percentile(forecast_draws, 90, axis=0),
                'upper_95': np.percentile(forecast_draws, 95, axis=0),
                'upper_97.5': np.percentile(forecast_draws, 97.5, axis=0),
                'std': np.std(forecast_draws, axis=0),
            }

    def forecast_point(self, h: int = 12) -> np.ndarray:
        """
        Generate point forecasts using posterior mean coefficients.

        This is faster than full density forecast as it doesn't
        require sampling.

        Parameters
        ----------
        h : int, default=12
            Forecast horizon

        Returns
        -------
        np.ndarray, shape (h, k)
            Point forecasts for each variable
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before forecasting")

        B = self.coef_mean
        k = self.k
        p = self.n_lags

        last_obs = self._Y_full[-p:][::-1].flatten()

        forecasts = np.zeros((h, k))
        current_lags = last_obs.copy()

        for t in range(h):
            x_t = np.concatenate([[1.0], current_lags[:k * p]])
            forecasts[t] = x_t @ B
            current_lags = np.concatenate([forecasts[t], current_lags[:-k]])

        return forecasts

    def get_posterior_summary(self) -> Dict[str, np.ndarray]:
        """
        Get summary statistics for posterior distributions.

        Returns
        -------
        dict
            Posterior summary with mean, std, and credible intervals
            for coefficients and covariance matrix
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted first")

        return {
            'coef_mean': self.coef_mean,
            'coef_std': self.coef_std,
            'coef_lower_2.5': np.percentile(self.coef_posterior, 2.5, axis=0),
            'coef_upper_97.5': np.percentile(self.coef_posterior, 97.5, axis=0),
            'sigma_mean': self.sigma_mean,
            'sigma_std': np.std(self.sigma_posterior, axis=0),
        }

    def get_hyperparameters(self) -> Dict[str, float]:
        """Return current hyperparameter settings."""
        return {
            'n_lags': self.n_lags,
            'lambda1': self.lambda1,
            'lambda3': self.lambda3,
            'lambda_const': self.lambda_const,
            'n_draws': self.n_draws,
            'n_burn': self.n_burn,
        }


def rolling_cv_rmse(
    Y: np.ndarray,
    lambda1: float,
    lambda3: float,
    n_lags: int,
    target_idx: int = 0,
    n_folds: int = 5,
    forecast_horizon: int = 12,
    n_draws: int = 1000,
    n_burn: int = 200,
    random_state: Optional[int] = None
) -> float:
    """
    Time-series cross-validation RMSE for hyperparameter selection.

    Uses expanding window with rolling evaluation.

    Parameters
    ----------
    Y : np.ndarray, shape (T, k)
        Multivariate time series
    lambda1 : float
        Overall tightness parameter
    lambda3 : float
        Lag decay parameter
    n_lags : int
        Number of VAR lags
    target_idx : int, default=0
        Index of target variable for RMSE calculation
    n_folds : int, default=5
        Number of CV folds
    forecast_horizon : int, default=12
        Forecast horizon for each fold
    n_draws : int, default=1000
        Posterior draws (reduced for CV speed)
    n_burn : int, default=200
        Burn-in draws
    random_state : int, optional
        Random seed

    Returns
    -------
    float
        Mean RMSE across CV folds
    """
    T = len(Y)
    min_train_size = max(60, n_lags + 10)  # Minimum training window

    # Ensure we have enough data for folds
    available_test = T - min_train_size - forecast_horizon
    if available_test < forecast_horizon * n_folds:
        n_folds = max(1, available_test // forecast_horizon)

    if n_folds < 1:
        return np.inf

    fold_size = available_test // n_folds
    rmse_list = []

    for fold in range(n_folds):
        train_end = min_train_size + fold * fold_size
        test_end = min(train_end + forecast_horizon, T)

        if test_end > T:
            break

        Y_train = Y[:train_end]
        Y_test = Y[train_end:test_end]

        if len(Y_test) == 0:
            continue

        try:
            bvar = BayesianVAR(
                n_lags=n_lags,
                lambda1=lambda1,
                lambda3=lambda3,
                n_draws=n_draws,
                n_burn=n_burn,
                random_state=random_state
            )
            bvar.fit(Y_train)
            forecasts = bvar.forecast_point(h=len(Y_test))

            # RMSE for target variable
            errors = forecasts[:, target_idx] - Y_test[:, target_idx]
            rmse = np.sqrt(np.mean(errors ** 2))
            rmse_list.append(rmse)

        except Exception as e:
            warnings.warn(f"CV fold {fold} failed: {e}")
            rmse_list.append(np.inf)

    if len(rmse_list) == 0:
        return np.inf

    return np.mean(rmse_list)


def grid_search_hyperparameters(
    Y: np.ndarray,
    target_idx: int = 0,
    lambda1_grid: List[float] = None,
    lambda3_grid: List[float] = None,
    n_lags_grid: List[int] = None,
    n_folds: int = 5,
    n_draws: int = 1000,
    n_burn: int = 200,
    random_state: Optional[int] = None,
    verbose: bool = True
) -> Dict:
    """
    Grid search for optimal BVAR hyperparameters.

    Parameters
    ----------
    Y : np.ndarray, shape (T, k)
        Multivariate time series
    target_idx : int, default=0
        Index of target variable
    lambda1_grid : list of float, optional
        Overall tightness values to try
    lambda3_grid : list of float, optional
        Lag decay values to try
    n_lags_grid : list of int, optional
        Lag orders to try
    n_folds : int, default=5
        Number of CV folds
    n_draws : int, default=1000
        Posterior draws for CV (reduced for speed)
    n_burn : int, default=200
        Burn-in draws
    random_state : int, optional
        Random seed
    verbose : bool, default=True
        Print progress

    Returns
    -------
    dict
        Best hyperparameters and full results grid
    """
    if lambda1_grid is None:
        lambda1_grid = [0.05, 0.1, 0.2, 0.5]
    if lambda3_grid is None:
        lambda3_grid = [1.0, 2.0]
    if n_lags_grid is None:
        n_lags_grid = [1, 2, 3, 4]

    results = []
    best_rmse = np.inf
    best_params = None

    total_combos = len(lambda1_grid) * len(lambda3_grid) * len(n_lags_grid)
    combo_idx = 0

    for n_lags in n_lags_grid:
        for lambda1 in lambda1_grid:
            for lambda3 in lambda3_grid:
                combo_idx += 1

                if verbose:
                    print(f"  [{combo_idx}/{total_combos}] n_lags={n_lags}, lambda1={lambda1}, lambda3={lambda3}", end="")

                rmse = rolling_cv_rmse(
                    Y=Y,
                    lambda1=lambda1,
                    lambda3=lambda3,
                    n_lags=n_lags,
                    target_idx=target_idx,
                    n_folds=n_folds,
                    n_draws=n_draws,
                    n_burn=n_burn,
                    random_state=random_state
                )

                results.append({
                    'n_lags': n_lags,
                    'lambda1': lambda1,
                    'lambda3': lambda3,
                    'cv_rmse': rmse,
                })

                if verbose:
                    print(f" -> RMSE: {rmse:.4f}")

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = {
                        'n_lags': n_lags,
                        'lambda1': lambda1,
                        'lambda3': lambda3,
                    }

    return {
        'best_params': best_params,
        'best_rmse': best_rmse,
        'all_results': results,
    }


__all__ = [
    'BayesianVAR',
    'rolling_cv_rmse',
    'grid_search_hyperparameters',
]
