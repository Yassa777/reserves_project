"""
Time-Varying Parameter VAR (TVP-VAR) Implementation.

Full Bayesian estimation via Gibbs sampling following Primiceri (2005).

Key Features:
- State-space representation with random walk parameters
- Carter-Kohn algorithm for state sampling
- Optional stochastic volatility
- Posterior sampling with credible intervals

Reference: Specification 04 - TVP-VAR
"""

import numpy as np
import pandas as pd
from scipy.stats import invwishart, multivariate_normal
from typing import Dict, List, Optional, Tuple, Any
import warnings


class TVP_VAR:
    """
    Time-Varying Parameter VAR with optional stochastic volatility.

    Estimation via Gibbs sampling following Primiceri (2005).

    Model specification:
        Observation: Y_t = X_t' * beta_t + epsilon_t,  epsilon_t ~ N(0, Sigma_t)
        State:       beta_t = beta_{t-1} + eta_t,      eta_t ~ N(0, Q)

    Parameters
    ----------
    n_lags : int
        Number of VAR lags (default: 1)
    stochastic_vol : bool
        If True, include stochastic volatility (default: True)
    n_draws : int
        Number of posterior draws (default: 5000)
    n_burn : int
        Burn-in period (default: 2000)
    random_state : int, optional
        Random seed for reproducibility

    Attributes
    ----------
    beta_posterior : np.ndarray
        Posterior draws of time-varying coefficients (n_draws, T, n_coefs * k)
    sigma_posterior : np.ndarray
        Posterior draws of variance (n_draws, T, k, k) if SV else (n_draws, k, k)
    Q_posterior : np.ndarray
        Posterior draws of state variance (n_draws, n_coefs * k, n_coefs * k)
    T : int
        Number of effective observations
    k : int
        Number of variables in VAR system
    n_coefs : int
        Number of coefficients per equation (including constant)
    """

    def __init__(
        self,
        n_lags: int = 1,
        stochastic_vol: bool = True,
        n_draws: int = 5000,
        n_burn: int = 2000,
        random_state: Optional[int] = None,
        fast_mode: bool = False
    ):
        self.n_lags = n_lags
        self.stochastic_vol = stochastic_vol
        self.n_draws = n_draws
        self.n_burn = n_burn
        self.random_state = random_state
        self.fast_mode = fast_mode  # Use Kalman filter only (no MCMC)

        # Posteriors (set after fitting)
        self.beta_posterior = None
        self.sigma_posterior = None
        self.Q_posterior = None

        # Model dimensions
        self.T = None
        self.k = None
        self.n_coefs = None
        self.Y = None
        self.X = None
        self.dates = None
        self.var_names = None

        # Diagnostics
        self.ols_beta = None
        self.ols_sigma = None
        self._fitted = False

    def fit(
        self,
        Y: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
        var_names: Optional[List[str]] = None,
        verbose: bool = True
    ) -> 'TVP_VAR':
        """
        Fit TVP-VAR via Gibbs sampling.

        Parameters
        ----------
        Y : np.ndarray, shape (T, k)
            Multivariate time series
        dates : pd.DatetimeIndex, optional
            Date index for time series
        var_names : list, optional
            Variable names
        verbose : bool
            Print progress updates

        Returns
        -------
        self
            Fitted model
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        T_full, k = Y.shape
        self.k = k
        self.var_names = var_names if var_names else [f"var_{i}" for i in range(k)]

        if verbose:
            print(f"Fitting TVP-VAR with {k} variables, {self.n_lags} lag(s)")
            print(f"  Observations: {T_full}")
            print(f"  Stochastic volatility: {self.stochastic_vol}")
            print(f"  Gibbs draws: {self.n_draws} (burn-in: {self.n_burn})")

        # Create lags
        Y_lag, Y_obs = self._create_lags(Y)
        T_eff = Y_obs.shape[0]
        self.T = T_eff

        # Store dates for effective sample
        if dates is not None:
            self.dates = dates[self.n_lags:]
        else:
            self.dates = pd.date_range(start='2007-01-01', periods=T_eff, freq='MS')

        # Add constant
        X = np.column_stack([np.ones(T_eff), Y_lag])
        n_coefs = X.shape[1]  # 1 + k * n_lags
        n_states = n_coefs * k
        self.n_coefs = n_coefs
        self.X = X
        self.Y = Y_obs

        if verbose:
            print(f"  Effective observations: {T_eff}")
            print(f"  Coefficients per equation: {n_coefs}")
            print(f"  Total state dimension: {n_states}")

        # Initialize with OLS
        beta_ols = np.linalg.lstsq(X, Y_obs, rcond=None)[0]
        residuals = Y_obs - X @ beta_ols
        Sigma_ols = residuals.T @ residuals / T_eff
        self.ols_beta = beta_ols
        self.ols_sigma = Sigma_ols

        # FAST MODE: Use Kalman smoother only (no MCMC)
        if self.fast_mode:
            if verbose:
                print("  Using FAST MODE (Kalman smoother, no MCMC)")
            return self._fit_kalman_only(Y_obs, X, beta_ols, Sigma_ols, n_states, k, n_coefs, T_eff, verbose)

        # Storage for posterior draws
        beta_draws = np.zeros((self.n_draws, T_eff, n_states))
        Q_draws = np.zeros((self.n_draws, n_states, n_states))

        if self.stochastic_vol:
            sigma_draws = np.zeros((self.n_draws, T_eff, k, k))
        else:
            sigma_draws = np.zeros((self.n_draws, k, k))

        # Initialize state paths
        beta_path = np.tile(beta_ols.flatten(), (T_eff, 1))  # (T, n_states)

        # Initialize Q with small variance (Minnesota-style prior)
        Q_prior_scale = 0.01
        Q = np.eye(n_states) * Q_prior_scale

        # Initialize volatility
        if self.stochastic_vol:
            Sigma_path = np.tile(Sigma_ols, (T_eff, 1, 1))
            log_vol = np.log(np.diag(Sigma_ols)) * np.ones((T_eff, 1))
            log_vol = np.tile(log_vol.flatten(), (T_eff, 1))[:, :k]
        else:
            Sigma = Sigma_ols.copy()
            Sigma_path = None

        # Gibbs sampling
        total_draws = self.n_burn + self.n_draws
        progress_interval = max(1, total_draws // 10)

        for draw in range(total_draws):
            if verbose and (draw + 1) % progress_interval == 0:
                pct = (draw + 1) / total_draws * 100
                print(f"    Progress: {draw + 1}/{total_draws} ({pct:.0f}%)")

            # Step 1: Sample beta_path | Y, Q, Sigma (Carter-Kohn)
            current_sigma = Sigma_path if self.stochastic_vol else Sigma
            beta_path = self._sample_states(
                Y_obs, X, beta_path, Q, current_sigma, k, n_coefs
            )

            # Step 2: Sample Q | beta_path
            Q = self._sample_Q(beta_path, n_states)

            # Step 3: Sample Sigma | Y, beta_path
            if self.stochastic_vol:
                Sigma_path, log_vol = self._sample_stochastic_vol(
                    Y_obs, X, beta_path, log_vol, k
                )
            else:
                Sigma = self._sample_sigma(Y_obs, X, beta_path, k)

            # Store after burn-in
            if draw >= self.n_burn:
                idx = draw - self.n_burn
                beta_draws[idx] = beta_path
                Q_draws[idx] = Q
                if self.stochastic_vol:
                    sigma_draws[idx] = Sigma_path
                else:
                    sigma_draws[idx] = Sigma

        self.beta_posterior = beta_draws
        self.Q_posterior = Q_draws
        self.sigma_posterior = sigma_draws
        self._fitted = True

        if verbose:
            print("  Fitting complete.")

        return self

    def _fit_kalman_only(
        self,
        Y_obs: np.ndarray,
        X: np.ndarray,
        beta_ols: np.ndarray,
        Sigma_ols: np.ndarray,
        n_states: int,
        k: int,
        n_coefs: int,
        T_eff: int,
        verbose: bool
    ) -> 'TVP_VAR':
        """
        Fast TVP estimation using Kalman filter/smoother only.

        No MCMC - provides point estimates and approximate uncertainty.
        Much faster than full Gibbs sampling.
        """
        # Set Q based on training sample variance (heuristic)
        Q = np.eye(n_states) * 0.001  # Small state drift

        # Forward filtering (Kalman filter)
        beta_filt = np.zeros((T_eff, n_states))
        P_filt = np.zeros((T_eff, n_states, n_states))

        beta_pred = beta_ols.flatten()
        P_pred = Q * 100  # Diffuse prior

        reg_eye = np.eye(n_states) * 1e-6
        reg_k = np.eye(k) * 1e-6

        for t in range(T_eff):
            # Observation matrix
            H_t = np.kron(np.eye(k), X[t:t+1])

            # Kalman gain
            S = H_t @ P_pred @ H_t.T + Sigma_ols + reg_k
            S = (S + S.T) / 2
            try:
                S_inv = np.linalg.inv(S)
            except:
                S_inv = np.linalg.pinv(S)

            K = P_pred @ H_t.T @ S_inv

            # Update
            beta_mat = beta_pred.reshape(n_coefs, k)
            y_pred = X[t] @ beta_mat
            innovation = Y_obs[t] - y_pred

            beta_filt[t] = beta_pred + K @ innovation
            P_filt[t] = (np.eye(n_states) - K @ H_t) @ P_pred
            P_filt[t] = (P_filt[t] + P_filt[t].T) / 2 + reg_eye

            # Predict
            if t < T_eff - 1:
                beta_pred = beta_filt[t]
                P_pred = P_filt[t] + Q

        # Backward smoothing (Rauch-Tung-Striebel)
        beta_smooth = np.zeros((T_eff, n_states))
        P_smooth = np.zeros((T_eff, n_states, n_states))

        beta_smooth[-1] = beta_filt[-1]
        P_smooth[-1] = P_filt[-1]

        for t in range(T_eff - 2, -1, -1):
            P_pred_next = P_filt[t] + Q
            P_pred_next = (P_pred_next + P_pred_next.T) / 2 + reg_eye

            try:
                J = P_filt[t] @ np.linalg.inv(P_pred_next)
            except:
                J = P_filt[t] @ np.linalg.pinv(P_pred_next)

            beta_smooth[t] = beta_filt[t] + J @ (beta_smooth[t+1] - beta_filt[t])
            P_smooth[t] = P_filt[t] + J @ (P_smooth[t+1] - P_pred_next) @ J.T
            P_smooth[t] = (P_smooth[t] + P_smooth[t].T) / 2 + reg_eye

        # Create "pseudo-posterior" by treating smoother output as mean
        # and using P_smooth for uncertainty
        # Store as if we had n_draws identical samples
        self.beta_posterior = np.tile(beta_smooth[np.newaxis, :, :], (self.n_draws, 1, 1))

        # Add small noise to simulate posterior draws
        for d in range(self.n_draws):
            for t in range(T_eff):
                try:
                    noise = np.random.multivariate_normal(np.zeros(n_states), P_smooth[t] * 0.1)
                    self.beta_posterior[d, t] += noise
                except:
                    pass

        self.Q_posterior = np.tile(Q[np.newaxis, :, :], (self.n_draws, 1, 1))
        self.sigma_posterior = np.tile(Sigma_ols[np.newaxis, :, :], (self.n_draws, 1, 1))

        self._fitted = True

        if verbose:
            print("  Kalman smoothing complete.")

        return self

    def _create_lags(self, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create lagged regressor matrix."""
        T, k = Y.shape
        Y_lag = np.zeros((T - self.n_lags, k * self.n_lags))

        for lag in range(1, self.n_lags + 1):
            start_col = (lag - 1) * k
            end_idx = -lag if lag < T else None
            Y_lag[:, start_col:start_col + k] = Y[self.n_lags - lag:end_idx]

        Y_obs = Y[self.n_lags:]
        return Y_lag, Y_obs

    def _sample_states(
        self,
        Y: np.ndarray,
        X: np.ndarray,
        beta_path: np.ndarray,
        Q: np.ndarray,
        Sigma: np.ndarray,
        k: int,
        n_coefs: int
    ) -> np.ndarray:
        """
        Carter-Kohn algorithm for sampling state paths.

        Uses forward filtering and backward sampling.
        """
        T = len(Y)
        n_states = n_coefs * k

        # Forward filtering storage
        beta_filt = np.zeros((T, n_states))
        P_filt = np.zeros((T, n_states, n_states))

        # Initialize with diffuse prior
        beta_pred = beta_path[0].copy()
        P_pred = Q * 100  # Diffuse initialization

        # Add small regularization for numerical stability
        reg_eye = np.eye(n_states) * 1e-6

        for t in range(T):
            # Get Sigma_t
            if Sigma.ndim == 3:
                Sigma_t = Sigma[t]
            else:
                Sigma_t = Sigma

            # Ensure Sigma_t is positive definite
            Sigma_t = (Sigma_t + Sigma_t.T) / 2
            Sigma_t += np.eye(k) * 1e-8

            # Observation matrix (maps states to observations)
            # For vectorized coefficients: y_t = (I_k kron x_t') * vec(beta_t)
            H_t = np.kron(np.eye(k), X[t:t+1])  # (k, n_states)

            # Kalman gain
            try:
                S = H_t @ P_pred @ H_t.T + Sigma_t
                S = (S + S.T) / 2 + np.eye(k) * 1e-8
                S_inv = np.linalg.inv(S)
                K = P_pred @ H_t.T @ S_inv
            except np.linalg.LinAlgError:
                # Fallback: use pseudo-inverse
                S = H_t @ P_pred @ H_t.T + Sigma_t
                K = P_pred @ H_t.T @ np.linalg.pinv(S)

            # Predicted observation
            beta_mat = beta_pred.reshape(n_coefs, k)
            y_pred = X[t] @ beta_mat

            # Update
            innovation = Y[t] - y_pred
            beta_filt[t] = beta_pred + K @ innovation
            P_filt[t] = (np.eye(n_states) - K @ H_t) @ P_pred
            P_filt[t] = (P_filt[t] + P_filt[t].T) / 2 + reg_eye

            # Predict next period
            if t < T - 1:
                beta_pred = beta_filt[t]
                P_pred = P_filt[t] + Q
                P_pred = (P_pred + P_pred.T) / 2

        # Backward sampling
        beta_sampled = np.zeros((T, n_states))

        # Sample last period
        try:
            P_last = (P_filt[-1] + P_filt[-1].T) / 2 + reg_eye
            beta_sampled[-1] = np.random.multivariate_normal(
                beta_filt[-1], P_last
            )
        except (np.linalg.LinAlgError, ValueError):
            beta_sampled[-1] = beta_filt[-1]

        for t in range(T - 2, -1, -1):
            # Smoothing step
            P_pred_next = P_filt[t] + Q
            P_pred_next = (P_pred_next + P_pred_next.T) / 2 + reg_eye

            try:
                P_pred_inv = np.linalg.inv(P_pred_next)
                J = P_filt[t] @ P_pred_inv
            except np.linalg.LinAlgError:
                J = P_filt[t] @ np.linalg.pinv(P_pred_next)

            beta_smooth = beta_filt[t] + J @ (beta_sampled[t + 1] - beta_filt[t])
            P_smooth = P_filt[t] - J @ Q @ J.T
            P_smooth = (P_smooth + P_smooth.T) / 2 + reg_eye

            # Sample with fallback for numerical issues
            try:
                # Use Cholesky decomposition for more stable sampling
                L = np.linalg.cholesky(P_smooth)
                z = np.random.standard_normal(n_states)
                beta_sampled[t] = beta_smooth + L @ z
            except np.linalg.LinAlgError:
                # If Cholesky fails, try eigenvalue fix
                try:
                    eigvals, eigvecs = np.linalg.eigh(P_smooth)
                    eigvals = np.maximum(eigvals, 1e-8)
                    P_fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    beta_sampled[t] = np.random.multivariate_normal(beta_smooth, P_fixed)
                except:
                    beta_sampled[t] = beta_smooth

        return beta_sampled

    def _sample_Q(self, beta_path: np.ndarray, n_states: int) -> np.ndarray:
        """Sample state variance Q from inverse-Wishart posterior."""
        T = beta_path.shape[0]

        # Compute state innovations
        innovations = np.diff(beta_path, axis=0)

        # Inverse-Wishart posterior
        # Prior: IW(nu_0, S_0) with nu_0 = n_states + 1, S_0 = I * 0.001
        nu_0 = n_states + 1
        S_0 = np.eye(n_states) * 0.001

        # Posterior parameters
        nu_post = nu_0 + (T - 1)
        S_post = S_0 + innovations.T @ innovations

        # Ensure positive definiteness
        S_post = (S_post + S_post.T) / 2
        S_post += np.eye(n_states) * 1e-8

        try:
            Q = invwishart.rvs(df=nu_post, scale=S_post)
        except (np.linalg.LinAlgError, ValueError):
            Q = S_post / nu_post  # Fallback to mode

        # Ensure symmetry
        Q = (Q + Q.T) / 2

        return Q

    def _sample_sigma(
        self,
        Y: np.ndarray,
        X: np.ndarray,
        beta_path: np.ndarray,
        k: int
    ) -> np.ndarray:
        """Sample constant observation variance Sigma."""
        T = len(Y)
        residuals = np.zeros((T, k))

        for t in range(T):
            beta_t = beta_path[t].reshape(-1, k)
            residuals[t] = Y[t] - X[t] @ beta_t

        # Inverse-Wishart posterior
        nu_0 = k + 1
        S_0 = np.eye(k) * 0.001
        nu_post = nu_0 + T
        S_post = S_0 + residuals.T @ residuals

        S_post = (S_post + S_post.T) / 2 + np.eye(k) * 1e-8

        try:
            Sigma = invwishart.rvs(df=nu_post, scale=S_post)
        except (np.linalg.LinAlgError, ValueError):
            Sigma = S_post / nu_post

        return (Sigma + Sigma.T) / 2

    def _sample_stochastic_vol(
        self,
        Y: np.ndarray,
        X: np.ndarray,
        beta_path: np.ndarray,
        log_vol: np.ndarray,
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample stochastic volatility paths using random walk Metropolis.

        Log-volatility follows: log(sigma_t) = log(sigma_{t-1}) + xi_t
        where xi_t ~ N(0, omega^2)
        """
        T = len(Y)
        Sigma_path = np.zeros((T, k, k))

        # Compute residuals
        residuals = np.zeros((T, k))
        for t in range(T):
            beta_t = beta_path[t].reshape(-1, k)
            residuals[t] = Y[t] - X[t] @ beta_t

        # Log-volatility innovation variance (fixed for simplicity)
        omega_sq = 0.1
        proposal_sd = 0.1

        for t in range(T):
            # Proposal for log-volatility
            if t == 0:
                log_vol_prop = log_vol[t] + np.random.normal(0, proposal_sd, k)
            else:
                # Random walk from previous period
                log_vol_prop = log_vol[t - 1] + np.random.normal(0, np.sqrt(omega_sq), k)

            # Construct diagonal variance matrices
            Sigma_prop = np.diag(np.exp(log_vol_prop))
            Sigma_curr = np.diag(np.exp(log_vol[t]))

            # Ensure positive definiteness
            Sigma_prop = np.maximum(Sigma_prop, np.eye(k) * 1e-8)
            Sigma_curr = np.maximum(Sigma_curr, np.eye(k) * 1e-8)

            # Log-likelihood ratio (Metropolis-Hastings)
            try:
                log_lik_prop = multivariate_normal.logpdf(
                    residuals[t], mean=np.zeros(k), cov=Sigma_prop
                )
                log_lik_curr = multivariate_normal.logpdf(
                    residuals[t], mean=np.zeros(k), cov=Sigma_curr
                )

                # Accept/reject
                log_alpha = log_lik_prop - log_lik_curr
                if np.log(np.random.random()) < log_alpha:
                    log_vol[t] = log_vol_prop
            except (ValueError, np.linalg.LinAlgError):
                # Keep current value on error
                pass

            Sigma_path[t] = np.diag(np.exp(log_vol[t]))

        return Sigma_path, log_vol

    def forecast(self, h: int = 12, use_final_params: bool = True) -> Dict[str, np.ndarray]:
        """
        Generate h-step ahead forecasts.

        Parameters
        ----------
        h : int
            Forecast horizon
        use_final_params : bool
            If True, use parameters from final period only.
            If False, allow parameter evolution during forecast.

        Returns
        -------
        dict
            'mean': Point forecast (h, k)
            'lower_10': 10th percentile (h, k)
            'upper_90': 90th percentile (h, k)
            'lower_5': 5th percentile (h, k)
            'upper_95': 95th percentile (h, k)
            'draws': All forecast draws (n_draws, h, k)
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before forecasting.")

        n_draws = self.beta_posterior.shape[0]
        k = self.k

        forecasts = np.zeros((n_draws, h, k))

        for d in range(n_draws):
            # Use final period parameters
            beta_T = self.beta_posterior[d, -1].reshape(-1, k)

            # Get variance
            if self.sigma_posterior.ndim == 4:
                Sigma = self.sigma_posterior[d, -1]
            else:
                Sigma = self.sigma_posterior[d]

            # Ensure positive definite
            Sigma = (Sigma + Sigma.T) / 2 + np.eye(k) * 1e-8

            # Get last observations for lags
            current_lags = self.Y[-self.n_lags:][::-1].flatten()

            for t in range(h):
                # Construct regressor (constant + lags)
                x_t = np.concatenate([[1], current_lags[:k * self.n_lags]])

                # Predicted mean
                mean_t = x_t @ beta_T

                # Add shock
                try:
                    shock = np.random.multivariate_normal(np.zeros(k), Sigma)
                except (ValueError, np.linalg.LinAlgError):
                    shock = np.zeros(k)

                forecasts[d, t] = mean_t + shock

                # Update lags
                if self.n_lags > 0:
                    current_lags = np.concatenate([
                        forecasts[d, t],
                        current_lags[:k * (self.n_lags - 1)]
                    ])

        return {
            'mean': np.mean(forecasts, axis=0),
            'median': np.median(forecasts, axis=0),
            'lower_10': np.percentile(forecasts, 10, axis=0),
            'upper_90': np.percentile(forecasts, 90, axis=0),
            'lower_5': np.percentile(forecasts, 5, axis=0),
            'upper_95': np.percentile(forecasts, 95, axis=0),
            'draws': forecasts
        }

    def get_time_varying_coefficients(
        self,
        var_idx: int = 0,
        equation_idx: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Extract time path of coefficients for a specific variable.

        Parameters
        ----------
        var_idx : int
            Index of dependent variable (equation) to analyze
        equation_idx : int, optional
            Deprecated, use var_idx

        Returns
        -------
        dict
            'mean': Posterior mean coefficients (T, n_coefs)
            'lower_16': 16th percentile (T, n_coefs)
            'upper_84': 84th percentile (T, n_coefs)
            'lower_5': 5th percentile (T, n_coefs)
            'upper_95': 95th percentile (T, n_coefs)
            'coef_names': Coefficient names
        """
        if not self._fitted:
            raise ValueError("Model must be fitted first.")

        if equation_idx is not None:
            var_idx = equation_idx

        k = self.k
        n_coefs = self.n_coefs

        # Reshape posterior: (n_draws, T, n_coefs, k)
        beta_reshaped = self.beta_posterior.reshape(
            self.n_draws, self.T, n_coefs, k
        )

        # Extract coefficients for target equation
        beta_target = beta_reshaped[:, :, :, var_idx]  # (n_draws, T, n_coefs)

        # Generate coefficient names
        coef_names = ['const']
        for lag in range(1, self.n_lags + 1):
            for v in range(k):
                vname = self.var_names[v] if self.var_names else f'var{v}'
                coef_names.append(f'{vname}_lag{lag}')

        return {
            'mean': np.mean(beta_target, axis=0),
            'median': np.median(beta_target, axis=0),
            'lower_16': np.percentile(beta_target, 16, axis=0),
            'upper_84': np.percentile(beta_target, 84, axis=0),
            'lower_5': np.percentile(beta_target, 5, axis=0),
            'upper_95': np.percentile(beta_target, 95, axis=0),
            'coef_names': coef_names,
            'dates': self.dates
        }

    def get_volatility_path(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract time path of volatilities (if stochastic vol enabled).

        Returns
        -------
        dict or None
            'mean': Posterior mean volatilities (T, k)
            'lower_16': 16th percentile (T, k)
            'upper_84': 84th percentile (T, k)
        """
        if not self.stochastic_vol:
            return None

        if self.sigma_posterior.ndim != 4:
            return None

        # Extract diagonal elements (variances)
        # sigma_posterior: (n_draws, T, k, k)
        T = self.sigma_posterior.shape[1]
        k = self.k

        variances = np.zeros((self.n_draws, T, k))
        for d in range(self.n_draws):
            for t in range(T):
                variances[d, t] = np.diag(self.sigma_posterior[d, t])

        # Convert to standard deviations
        stds = np.sqrt(variances)

        return {
            'mean': np.mean(stds, axis=0),
            'median': np.median(stds, axis=0),
            'lower_16': np.percentile(stds, 16, axis=0),
            'upper_84': np.percentile(stds, 84, axis=0),
            'lower_5': np.percentile(stds, 5, axis=0),
            'upper_95': np.percentile(stds, 95, axis=0),
            'dates': self.dates
        }

    def get_convergence_diagnostics(self) -> Dict[str, Any]:
        """
        Compute convergence diagnostics for MCMC chain.

        Returns
        -------
        dict
            'effective_sample_size': ESS for each parameter
            'acceptance_rate': (if applicable)
            'mean_autocorr': Mean autocorrelation at lag 1
        """
        if not self._fitted:
            raise ValueError("Model must be fitted first.")

        n_draws = self.n_draws
        n_states = self.beta_posterior.shape[2]

        # Use final period betas for diagnostics
        beta_final = self.beta_posterior[:, -1, :]  # (n_draws, n_states)

        # Effective sample size (simplified computation)
        ess = np.zeros(n_states)
        autocorr_lag1 = np.zeros(n_states)

        for i in range(n_states):
            chain = beta_final[:, i]
            # Compute autocorrelation at lag 1
            if len(chain) > 1:
                mean_chain = np.mean(chain)
                var_chain = np.var(chain)
                if var_chain > 0:
                    autocorr = np.corrcoef(chain[:-1], chain[1:])[0, 1]
                    autocorr_lag1[i] = autocorr
                    # Simple ESS approximation
                    ess[i] = n_draws * (1 - autocorr) / (1 + autocorr)
                else:
                    ess[i] = n_draws
            else:
                ess[i] = n_draws

        return {
            'effective_sample_size': ess,
            'mean_ess': np.mean(ess),
            'min_ess': np.min(ess),
            'mean_autocorr_lag1': np.mean(autocorr_lag1),
            'n_draws': n_draws,
            'n_states': n_states
        }

    def summary(self) -> pd.DataFrame:
        """
        Generate summary of posterior distributions.

        Returns
        -------
        pd.DataFrame
            Summary statistics for all parameters
        """
        if not self._fitted:
            raise ValueError("Model must be fitted first.")

        # Get coefficients for first equation
        coefs = self.get_time_varying_coefficients(var_idx=0)

        # Create summary at final period
        summary_data = []
        for i, name in enumerate(coefs['coef_names']):
            summary_data.append({
                'coefficient': name,
                'mean': coefs['mean'][-1, i],
                'median': coefs['median'][-1, i],
                'std': np.std(self.beta_posterior[:, -1, i]),
                'ci_lower_5': coefs['lower_5'][-1, i],
                'ci_upper_95': coefs['upper_95'][-1, i],
                'ci_lower_16': coefs['lower_16'][-1, i],
                'ci_upper_84': coefs['upper_84'][-1, i]
            })

        return pd.DataFrame(summary_data)


def fit_tvp_var_from_dataframe(
    df: pd.DataFrame,
    target_var: str,
    system_vars: List[str],
    n_lags: int = 1,
    stochastic_vol: bool = True,
    n_draws: int = 5000,
    n_burn: int = 2000,
    random_state: Optional[int] = None,
    fast_mode: bool = False,
    verbose: bool = True
) -> TVP_VAR:
    """
    Convenience function to fit TVP-VAR from a pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with date index and numeric columns
    target_var : str
        Name of target variable (should be first in system_vars)
    system_vars : list
        Variables to include in VAR system
    n_lags : int
        Number of lags
    stochastic_vol : bool
        Enable stochastic volatility
    n_draws : int
        Number of MCMC draws
    n_burn : int
        Burn-in period
    random_state : int, optional
        Random seed
    fast_mode : bool
        If True, use Kalman filter only (no MCMC, much faster)
    verbose : bool
        Print progress

    Returns
    -------
    TVP_VAR
        Fitted model
    """
    # Ensure target is first
    if system_vars[0] != target_var:
        system_vars = [target_var] + [v for v in system_vars if v != target_var]

    # Extract data
    Y = df[system_vars].values
    dates = df.index

    # Fit model
    model = TVP_VAR(
        n_lags=n_lags,
        stochastic_vol=stochastic_vol,
        n_draws=n_draws,
        n_burn=n_burn,
        random_state=random_state,
        fast_mode=fast_mode
    )

    model.fit(Y, dates=dates, var_names=system_vars, verbose=verbose)

    return model


__all__ = ['TVP_VAR', 'fit_tvp_var_from_dataframe']
