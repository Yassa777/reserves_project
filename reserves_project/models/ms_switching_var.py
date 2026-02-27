"""Markov-switching VAR with Gaussian regimes (EM estimation)."""

from __future__ import annotations

import numpy as np


def _stack_lags(y: np.ndarray, p: int) -> np.ndarray:
    t, k = y.shape
    cols = []
    for lag in range(1, p + 1):
        cols.append(y[p - lag : t - lag])
    return np.hstack(cols)


def _build_design(y: np.ndarray, p: int, exog: np.ndarray | None):
    t, k = y.shape
    y_target = y[p:]
    x_lags = _stack_lags(y, p)

    if exog is not None:
        exog = exog[p:]
        x = np.hstack([np.ones((t - p, 1)), x_lags, exog])
    else:
        x = np.hstack([np.ones((t - p, 1)), x_lags])

    return y_target, x


def _log_mvnpdf(resid: np.ndarray, sigma: np.ndarray) -> float:
    k = resid.shape[0]
    sign, logdet = np.linalg.slogdet(sigma)
    if sign <= 0:
        return -np.inf
    inv = np.linalg.inv(sigma)
    quad = resid.T @ inv @ resid
    return -0.5 * (k * np.log(2 * np.pi) + logdet + quad)


def _safe_cov(resid: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = weights[:, None]
    resid_w = resid * w
    denom = np.sum(weights)
    if denom <= 0:
        denom = 1.0
    sigma = resid_w.T @ resid_w / denom
    sigma += np.eye(resid.shape[1]) * 1e-6
    return sigma


class MarkovSwitchingVAR:
    def __init__(
        self,
        n_regimes: int = 2,
        ar_order: int = 1,
        max_iter: int = 50,
        tol: float = 1e-4,
    ):
        self.n_regimes = n_regimes
        self.ar_order = ar_order
        self.max_iter = max_iter
        self.tol = tol
        self.coefs_: list[np.ndarray] = []
        self.cov_: list[np.ndarray] = []
        self.transition_: np.ndarray | None = None
        self.loglik_: float | None = None
        self.loglik_path_: list[float] = []
        self.converged_: bool | None = None
        self.n_iter_: int = 0
        self.smoothed_probs_: np.ndarray | None = None
        self.init_states_summary_: dict | None = None

    def _initialize_params(
        self,
        y: np.ndarray,
        x: np.ndarray,
        init_states: np.ndarray | None,
        init_states_provided: bool = False,
    ):
        t, k = y.shape
        r = self.n_regimes
        if init_states is None:
            init_states = np.zeros(t, dtype=int)
        init_states = init_states[:t]

        counts = {str(i): int(np.sum(init_states == i)) for i in range(r)}
        self.init_states_summary_ = {
            "provided": bool(init_states_provided),
            "n_obs": int(t),
            "regime_counts": counts,
            "regime_shares": {k: (v / t if t > 0 else np.nan) for k, v in counts.items()},
        }

        probs = np.zeros((t, r))
        for i in range(r):
            probs[:, i] = (init_states == i).astype(float)
        if probs.sum() == 0:
            probs = np.full((t, r), 1.0 / r)

        self.coefs_ = []
        self.cov_ = []
        for i in range(r):
            w = probs[:, i]
            if w.sum() < 1e-6:
                w = np.ones(t)
            sqrt_w = np.sqrt(w)
            xw = x * sqrt_w[:, None]
            yw = y * sqrt_w[:, None]
            coef = np.linalg.lstsq(xw, yw, rcond=None)[0]
            resid = y - x @ coef
            self.coefs_.append(coef)
            self.cov_.append(_safe_cov(resid, w))

        trans = np.full((r, r), 1.0 / r)
        np.fill_diagonal(trans, 0.9)
        trans = trans / trans.sum(axis=1, keepdims=True)
        self.transition_ = trans

    def _e_step(self, y: np.ndarray, x: np.ndarray):
        t, k = y.shape
        r = self.n_regimes
        loglik = np.zeros((t, r))

        for t_idx in range(t):
            for r_idx in range(r):
                mean = x[t_idx] @ self.coefs_[r_idx]
                resid = y[t_idx] - mean
                loglik[t_idx, r_idx] = _log_mvnpdf(resid, self.cov_[r_idx])

        # Avoid underflow by scaling
        f = np.exp(loglik - loglik.max(axis=1, keepdims=True))

        alpha = np.zeros((t, r))
        scale = np.zeros(t)

        alpha[0] = f[0] / f[0].sum()
        scale[0] = f[0].sum()

        for t_idx in range(1, t):
            pred = alpha[t_idx - 1] @ self.transition_
            alpha[t_idx] = pred * f[t_idx]
            scale[t_idx] = alpha[t_idx].sum()
            if scale[t_idx] == 0:
                scale[t_idx] = 1e-12
            alpha[t_idx] /= scale[t_idx]

        beta = np.zeros((t, r))
        beta[-1] = 1.0
        for t_idx in range(t - 2, -1, -1):
            beta[t_idx] = (self.transition_ @ (f[t_idx + 1] * beta[t_idx + 1])) / scale[t_idx + 1]

        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)

        xi = np.zeros((t - 1, r, r))
        for t_idx in range(1, t):
            denom = scale[t_idx]
            if denom == 0:
                denom = 1e-12
            xi[t_idx - 1] = (
                alpha[t_idx - 1][:, None]
                * self.transition_
                * (f[t_idx] * beta[t_idx])[None, :]
            ) / denom

        ll = float(np.sum(np.log(scale)))
        return gamma, xi, ll

    def _m_step(self, y: np.ndarray, x: np.ndarray, gamma: np.ndarray, xi: np.ndarray):
        t, k = y.shape
        r = self.n_regimes

        new_coefs = []
        new_cov = []
        for r_idx in range(r):
            w = gamma[:, r_idx]
            sqrt_w = np.sqrt(w)
            xw = x * sqrt_w[:, None]
            yw = y * sqrt_w[:, None]
            coef = np.linalg.lstsq(xw, yw, rcond=None)[0]
            resid = y - x @ coef
            new_coefs.append(coef)
            new_cov.append(_safe_cov(resid, w))

        self.coefs_ = new_coefs
        self.cov_ = new_cov

        xi_sum = xi.sum(axis=0)
        gamma_sum = gamma[:-1].sum(axis=0)
        trans = xi_sum / gamma_sum[:, None]
        trans = np.nan_to_num(trans, nan=1.0 / r)
        trans = trans / trans.sum(axis=1, keepdims=True)
        self.transition_ = trans

    def fit(self, y: np.ndarray, exog: np.ndarray | None = None, init_states: np.ndarray | None = None):
        if self.max_iter < 1:
            raise ValueError("max_iter must be >= 1 for EM estimation.")

        y_target, x = _build_design(y, self.ar_order, exog)
        init_states_provided = init_states is not None
        if init_states is not None:
            init_states = init_states[self.ar_order :]
        self._initialize_params(y_target, x, init_states, init_states_provided=init_states_provided)

        self.loglik_path_ = []
        prev_ll = None
        converged = False
        n_iter = 0
        for iter_idx in range(1, self.max_iter + 1):
            gamma, xi, ll = self._e_step(y_target, x)
            self.loglik_path_.append(float(ll))
            self._m_step(y_target, x, gamma, xi)
            n_iter = iter_idx
            if prev_ll is not None and abs(ll - prev_ll) < self.tol:
                converged = True
                break
            prev_ll = ll

        self.loglik_ = float(self.loglik_path_[-1]) if self.loglik_path_ else None
        self.converged_ = converged
        self.n_iter_ = int(n_iter)
        self.smoothed_probs_ = gamma
        return self

    def expected_durations(self) -> np.ndarray:
        """Expected regime durations under the fitted transition matrix."""
        if self.transition_ is None:
            raise RuntimeError("Model not fitted; transition matrix unavailable.")
        diag = np.diag(self.transition_).astype(float)
        denom = np.clip(1.0 - diag, 1e-12, None)
        durations = 1.0 / denom
        durations[diag >= 1.0] = np.inf
        return durations

    def classification_certainty(self) -> dict:
        """Summary statistics of regime classification certainty."""
        if self.smoothed_probs_ is None:
            raise RuntimeError("Model not fitted; smoothed probabilities unavailable.")

        probs = np.asarray(self.smoothed_probs_, dtype=float)
        if probs.ndim != 2 or probs.shape[0] == 0:
            raise RuntimeError("Invalid smoothed probability matrix.")

        n_obs, n_regimes = probs.shape
        max_probs = probs.max(axis=1)
        assigned = probs.argmax(axis=1)
        entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)
        norm = np.log(n_regimes) if n_regimes > 1 else 1.0
        entropy_norm = entropy / norm if norm > 0 else np.zeros_like(entropy)

        counts = {str(i): int(np.sum(assigned == i)) for i in range(n_regimes)}
        shares = {k: (v / n_obs if n_obs > 0 else np.nan) for k, v in counts.items()}
        avg_probs = {str(i): float(np.mean(probs[:, i])) for i in range(n_regimes)}

        return {
            "n_obs": int(n_obs),
            "mean_max_probability": float(np.mean(max_probs)),
            "median_max_probability": float(np.median(max_probs)),
            "share_max_prob_ge_0_6": float(np.mean(max_probs >= 0.6)),
            "share_max_prob_ge_0_7": float(np.mean(max_probs >= 0.7)),
            "share_max_prob_ge_0_8": float(np.mean(max_probs >= 0.8)),
            "share_max_prob_ge_0_9": float(np.mean(max_probs >= 0.9)),
            "mean_entropy": float(np.mean(entropy)),
            "mean_normalized_entropy": float(np.mean(entropy_norm)),
            "regime_assignment_counts": counts,
            "regime_assignment_shares": shares,
            "average_regime_probabilities": avg_probs,
        }

    def forecast(
        self,
        y_history: np.ndarray,
        steps: int,
        exog_future: np.ndarray | None = None,
        regime_probs: np.ndarray | None = None,
        lock_regime: bool = False,
        regime_path: np.ndarray | None = None,
    ) -> np.ndarray:
        if exog_future is None:
            exog_future = np.zeros((steps, 0))
        if exog_future.ndim == 1:
            exog_future = exog_future[:, None]

        p = self.ar_order
        history = y_history.copy()
        k = history.shape[1]
        r = self.n_regimes

        if regime_path is not None and len(regime_path) < steps:
            raise ValueError("regime_path length must be >= forecast steps")

        if regime_probs is None:
            regime_probs = np.full(r, 1.0 / r)
        regime_probs = np.asarray(regime_probs, dtype=float)
        if regime_probs.ndim != 1 or regime_probs.shape[0] != r:
            regime_probs = np.full(r, 1.0 / r)

        forecasts = []
        for h in range(steps):
            if regime_path is not None:
                step_regime = int(regime_path[h])
                step_probs = np.zeros(r)
                step_probs[step_regime] = 1.0
            else:
                step_probs = regime_probs

            lags = []
            for lag in range(1, p + 1):
                lags.append(history[-lag])
            x = np.concatenate([np.array([1.0]), np.concatenate(lags), exog_future[h]])

            regime_means = []
            for r_idx in range(r):
                coef = self.coefs_[r_idx]
                mu = x @ coef
                regime_means.append(mu)
            regime_means = np.vstack(regime_means)

            y_pred = (step_probs[:, None] * regime_means).sum(axis=0)
            forecasts.append(y_pred)
            history = np.vstack([history, y_pred])

            if regime_path is None:
                if not lock_regime:
                    regime_probs = step_probs @ self.transition_
                else:
                    regime_probs = step_probs

        return np.vstack(forecasts)
