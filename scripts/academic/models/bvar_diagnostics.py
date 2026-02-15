"""
BVAR Diagnostics Module.

This module provides convergence diagnostics and model validation tools
for Bayesian VAR models estimated via Gibbs sampling.

Key diagnostics:
- Gelman-Rubin R-hat statistic (split-chain version)
- Effective sample size (ESS)
- Trace plot analysis
- Autocorrelation diagnostics
- Posterior predictive checks

References:
- Gelman, A., & Rubin, D. B. (1992). Inference from Iterative Simulation.
- Vehtari, A., et al. (2021). Rank-normalization and folding for R-hat.

Author: Academic Pipeline
Date: 2026-02-10
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings


def compute_rhat(chains: np.ndarray) -> float:
    """
    Compute the Gelman-Rubin R-hat statistic.

    Uses the split-chain approach: each chain is split in half,
    treating halves as separate chains.

    Parameters
    ----------
    chains : np.ndarray, shape (n_draws,) or (n_chains, n_draws)
        MCMC draws. If 1D, chain is split in half.

    Returns
    -------
    float
        R-hat statistic. Values close to 1.0 indicate convergence.
        Rule of thumb: R-hat < 1.1 is acceptable.
    """
    if chains.ndim == 1:
        # Split single chain in half
        n = len(chains)
        mid = n // 2
        chains = np.array([chains[:mid], chains[mid:2*mid]])

    n_chains, n_draws = chains.shape

    if n_draws < 4:
        return np.nan

    # Within-chain variance
    chain_means = np.mean(chains, axis=1)
    chain_vars = np.var(chains, axis=1, ddof=1)
    W = np.mean(chain_vars)

    # Between-chain variance
    overall_mean = np.mean(chain_means)
    B = n_draws * np.var(chain_means, ddof=1)

    # Marginal posterior variance estimate
    var_hat = ((n_draws - 1) / n_draws) * W + (1 / n_draws) * B

    # R-hat
    if W < 1e-10:
        return 1.0
    rhat = np.sqrt(var_hat / W)

    return rhat


def compute_ess(chain: np.ndarray, max_lag: int = None) -> float:
    """
    Compute effective sample size using autocorrelation method.

    ESS estimates the number of independent samples equivalent
    to the correlated MCMC chain.

    Parameters
    ----------
    chain : np.ndarray, shape (n_draws,)
        Single MCMC chain
    max_lag : int, optional
        Maximum lag for autocorrelation. Default is n_draws // 2.

    Returns
    -------
    float
        Effective sample size
    """
    n = len(chain)
    if n < 10:
        return float(n)

    if max_lag is None:
        max_lag = n // 2

    # Compute autocorrelations
    mean = np.mean(chain)
    var = np.var(chain, ddof=1)

    if var < 1e-10:
        return float(n)

    chain_centered = chain - mean
    autocorr = np.correlate(chain_centered, chain_centered, mode='full')
    autocorr = autocorr[n-1:] / (var * np.arange(n, 0, -1))

    # Sum autocorrelations until they become negative (Geyer's initial positive sequence)
    tau = 1.0
    for k in range(1, min(max_lag, n)):
        if k + 1 < len(autocorr) and autocorr[k] + autocorr[k+1] > 0:
            tau += 2 * autocorr[k]
        else:
            break

    ess = n / tau
    return max(1.0, ess)


def diagnose_convergence(
    coef_posterior: np.ndarray,
    sigma_posterior: np.ndarray,
    var_names: Optional[List[str]] = None,
    rhat_threshold: float = 1.1,
    ess_threshold: float = 100
) -> Dict:
    """
    Comprehensive convergence diagnostics for BVAR posterior.

    Parameters
    ----------
    coef_posterior : np.ndarray, shape (n_draws, n_coefs, k)
        Posterior draws for coefficients
    sigma_posterior : np.ndarray, shape (n_draws, k, k)
        Posterior draws for covariance matrix
    var_names : list of str, optional
        Variable names for reporting
    rhat_threshold : float, default=1.1
        Threshold for R-hat warning
    ess_threshold : float, default=100
        Minimum acceptable ESS

    Returns
    -------
    dict
        Diagnostic results with:
        - 'converged': bool, overall convergence assessment
        - 'rhat': dict of R-hat values for each parameter
        - 'ess': dict of ESS values
        - 'warnings': list of warning messages
        - 'summary': summary statistics
    """
    n_draws, n_coefs, k = coef_posterior.shape

    if var_names is None:
        var_names = [f"var_{i}" for i in range(k)]

    results = {
        'converged': True,
        'rhat': {},
        'ess': {},
        'warnings': [],
        'summary': {},
    }

    rhat_values = []
    ess_values = []

    # Diagnose coefficient chains
    for i in range(n_coefs):
        for j in range(k):
            chain = coef_posterior[:, i, j]
            param_name = f"B[{i},{j}]"

            # R-hat
            rhat = compute_rhat(chain)
            results['rhat'][param_name] = rhat
            rhat_values.append(rhat)

            if not np.isnan(rhat) and rhat > rhat_threshold:
                results['warnings'].append(
                    f"R-hat for {param_name} = {rhat:.3f} exceeds threshold {rhat_threshold}"
                )
                results['converged'] = False

            # ESS
            ess = compute_ess(chain)
            results['ess'][param_name] = ess
            ess_values.append(ess)

            if ess < ess_threshold:
                results['warnings'].append(
                    f"ESS for {param_name} = {ess:.1f} below threshold {ess_threshold}"
                )

    # Diagnose covariance chains
    for i in range(k):
        for j in range(i, k):  # Upper triangle only
            chain = sigma_posterior[:, i, j]
            param_name = f"Sigma[{i},{j}]"

            rhat = compute_rhat(chain)
            results['rhat'][param_name] = rhat
            rhat_values.append(rhat)

            if not np.isnan(rhat) and rhat > rhat_threshold:
                results['warnings'].append(
                    f"R-hat for {param_name} = {rhat:.3f} exceeds threshold"
                )
                results['converged'] = False

            ess = compute_ess(chain)
            results['ess'][param_name] = ess
            ess_values.append(ess)

    # Summary statistics
    rhat_values = [r for r in rhat_values if not np.isnan(r)]
    results['summary'] = {
        'n_parameters': len(rhat_values),
        'rhat_max': max(rhat_values) if rhat_values else np.nan,
        'rhat_mean': np.mean(rhat_values) if rhat_values else np.nan,
        'rhat_above_threshold': sum(r > rhat_threshold for r in rhat_values),
        'ess_min': min(ess_values) if ess_values else np.nan,
        'ess_mean': np.mean(ess_values) if ess_values else np.nan,
        'ess_below_threshold': sum(e < ess_threshold for e in ess_values),
    }

    return results


def trace_plot_data(
    chain: np.ndarray,
    param_name: str = "parameter",
    thin: int = 1
) -> Dict:
    """
    Prepare data for trace plots.

    Parameters
    ----------
    chain : np.ndarray, shape (n_draws,)
        MCMC chain
    param_name : str
        Parameter name for labeling
    thin : int, default=1
        Thinning factor for large chains

    Returns
    -------
    dict
        Data for plotting: iterations, values, running mean
    """
    thinned = chain[::thin]
    n = len(thinned)

    return {
        'param_name': param_name,
        'iterations': np.arange(n) * thin,
        'values': thinned,
        'running_mean': np.cumsum(thinned) / np.arange(1, n + 1),
        'mean': np.mean(thinned),
        'std': np.std(thinned),
    }


def autocorrelation_data(chain: np.ndarray, max_lag: int = 50) -> Dict:
    """
    Compute autocorrelation for diagnostic plotting.

    Parameters
    ----------
    chain : np.ndarray, shape (n_draws,)
        MCMC chain
    max_lag : int, default=50
        Maximum lag to compute

    Returns
    -------
    dict
        Autocorrelation data for each lag
    """
    n = len(chain)
    max_lag = min(max_lag, n - 1)

    mean = np.mean(chain)
    var = np.var(chain)

    if var < 1e-10:
        return {'lags': np.arange(max_lag + 1), 'acf': np.zeros(max_lag + 1)}

    chain_centered = chain - mean
    acf = np.zeros(max_lag + 1)

    for lag in range(max_lag + 1):
        if lag < n:
            acf[lag] = np.mean(chain_centered[:n-lag] * chain_centered[lag:]) / var

    return {
        'lags': np.arange(max_lag + 1),
        'acf': acf,
        'ci_upper': 1.96 / np.sqrt(n),
        'ci_lower': -1.96 / np.sqrt(n),
    }


def posterior_predictive_check(
    model,
    Y_actual: np.ndarray,
    n_reps: int = 500,
    target_idx: int = 0
) -> Dict:
    """
    Perform posterior predictive check.

    Generates replicated data from the posterior and compares
    summary statistics with actual data.

    Parameters
    ----------
    model : BayesianVAR
        Fitted BVAR model
    Y_actual : np.ndarray
        Actual observed data
    n_reps : int, default=500
        Number of replications
    target_idx : int, default=0
        Target variable index for detailed checks

    Returns
    -------
    dict
        Posterior predictive check results
    """
    n_draws = model.coef_posterior.shape[0]
    k = model.k
    T = len(Y_actual) - model.n_lags

    # Storage for replicated statistics
    rep_means = np.zeros((n_reps, k))
    rep_stds = np.zeros((n_reps, k))
    rep_autocorr = np.zeros(n_reps)

    # Draw indices
    draw_indices = np.random.choice(n_draws, size=n_reps, replace=True)

    for rep, draw_idx in enumerate(draw_indices):
        B = model.coef_posterior[draw_idx]
        Sigma = model.sigma_posterior[draw_idx]

        # Simulate replicated data
        Y_rep = np.zeros((T, k))
        Y_init = Y_actual[:model.n_lags]

        for t in range(T):
            if t < model.n_lags:
                lags = np.concatenate([
                    Y_init[model.n_lags - lag - 1:model.n_lags - lag].flatten()
                    for lag in range(model.n_lags)
                ])
            else:
                lags = np.concatenate([
                    Y_rep[t-lag-1:t-lag].flatten() if t-lag > 0 else Y_init[-1:].flatten()
                    for lag in range(model.n_lags)
                ])

            x_t = np.concatenate([[1.0], lags[:k * model.n_lags]])

            try:
                mean_t = x_t @ B
                shock = np.random.multivariate_normal(np.zeros(k), Sigma)
                Y_rep[t] = mean_t + shock
            except Exception:
                Y_rep[t] = Y_actual[model.n_lags + t] if model.n_lags + t < len(Y_actual) else 0

        rep_means[rep] = np.mean(Y_rep, axis=0)
        rep_stds[rep] = np.std(Y_rep, axis=0)

        # Lag-1 autocorrelation for target
        if len(Y_rep) > 1:
            rep_autocorr[rep] = np.corrcoef(Y_rep[:-1, target_idx], Y_rep[1:, target_idx])[0, 1]

    # Actual statistics
    Y_obs = Y_actual[model.n_lags:]
    actual_mean = np.mean(Y_obs, axis=0)
    actual_std = np.std(Y_obs, axis=0)
    actual_autocorr = np.corrcoef(Y_obs[:-1, target_idx], Y_obs[1:, target_idx])[0, 1]

    # Compute p-values (proportion of replications more extreme)
    pvalue_mean = np.mean(rep_means[:, target_idx] > actual_mean[target_idx])
    pvalue_std = np.mean(rep_stds[:, target_idx] > actual_std[target_idx])
    pvalue_autocorr = np.mean(rep_autocorr > actual_autocorr)

    return {
        'actual_mean': actual_mean,
        'actual_std': actual_std,
        'actual_autocorr': actual_autocorr,
        'rep_mean_dist': rep_means[:, target_idx],
        'rep_std_dist': rep_stds[:, target_idx],
        'rep_autocorr_dist': rep_autocorr,
        'pvalue_mean': pvalue_mean,
        'pvalue_std': pvalue_std,
        'pvalue_autocorr': pvalue_autocorr,
        'ppc_passed': (0.05 < pvalue_mean < 0.95) and (0.05 < pvalue_std < 0.95),
    }


def geweke_test(chain: np.ndarray, first_frac: float = 0.1, last_frac: float = 0.5) -> Dict:
    """
    Geweke convergence diagnostic.

    Compares means of first and last portions of chain using z-test.

    Parameters
    ----------
    chain : np.ndarray
        MCMC chain
    first_frac : float, default=0.1
        Fraction of chain for first window
    last_frac : float, default=0.5
        Fraction of chain for last window

    Returns
    -------
    dict
        z-score and p-value
    """
    n = len(chain)
    n_first = int(first_frac * n)
    n_last = int(last_frac * n)

    first_window = chain[:n_first]
    last_window = chain[-n_last:]

    mean_first = np.mean(first_window)
    mean_last = np.mean(last_window)

    # Spectral variance estimates (simplified)
    var_first = np.var(first_window, ddof=1) / n_first
    var_last = np.var(last_window, ddof=1) / n_last

    se = np.sqrt(var_first + var_last)

    if se < 1e-10:
        z_score = 0.0
    else:
        z_score = (mean_first - mean_last) / se

    # Two-sided p-value
    from scipy.stats import norm
    p_value = 2 * (1 - norm.cdf(abs(z_score)))

    return {
        'z_score': z_score,
        'p_value': p_value,
        'converged': abs(z_score) < 1.96,
    }


def create_diagnostic_report(
    model,
    var_names: Optional[List[str]] = None,
    Y_actual: Optional[np.ndarray] = None
) -> Dict:
    """
    Create comprehensive diagnostic report for BVAR model.

    Parameters
    ----------
    model : BayesianVAR
        Fitted BVAR model
    var_names : list of str, optional
        Variable names
    Y_actual : np.ndarray, optional
        Actual data for posterior predictive check

    Returns
    -------
    dict
        Comprehensive diagnostic report
    """
    report = {
        'hyperparameters': model.get_hyperparameters(),
        'convergence': diagnose_convergence(
            model.coef_posterior,
            model.sigma_posterior,
            var_names
        ),
    }

    # Geweke test for key parameters
    k = model.k
    geweke_results = {}
    for j in range(k):
        chain = model.coef_posterior[:, 1 + j, j]  # Own first lag
        geweke_results[f'own_lag1_var{j}'] = geweke_test(chain)
    report['geweke'] = geweke_results

    # Posterior predictive check if data provided
    if Y_actual is not None:
        try:
            report['ppc'] = posterior_predictive_check(model, Y_actual)
        except Exception as e:
            report['ppc'] = {'error': str(e)}

    # Overall assessment
    report['overall_converged'] = (
        report['convergence']['converged'] and
        all(g['converged'] for g in geweke_results.values())
    )

    return report


__all__ = [
    'compute_rhat',
    'compute_ess',
    'diagnose_convergence',
    'trace_plot_data',
    'autocorrelation_data',
    'posterior_predictive_check',
    'geweke_test',
    'create_diagnostic_report',
]
