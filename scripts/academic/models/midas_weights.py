"""
MIDAS Weight Functions for Mixed Data Sampling Regression.

Implements polynomial weighting schemes for aggregating high-frequency data:
1. Exponential Almon weights - flexible decay with 2 parameters
2. Beta weights - based on beta distribution density
3. Step function weights - piecewise constant blocks

Reference:
    Ghysels, E., Sinko, A., & Valkanov, R. (2007). MIDAS Regressions:
    Further Results and New Directions. Econometric Reviews.
"""

import numpy as np
from typing import Tuple, Optional


def exp_almon_weights(
    n_lags: int,
    theta1: float,
    theta2: float
) -> np.ndarray:
    """
    Exponential Almon polynomial weights.

    The weight for lag k is: w_k = exp(theta1*k + theta2*k^2) / sum(exp(...))

    Parameters
    ----------
    n_lags : int
        Number of high-frequency lags (e.g., 22 trading days per month)
    theta1 : float
        Linear decay parameter
    theta2 : float
        Quadratic decay parameter (typically negative for declining weights)

    Returns
    -------
    weights : np.ndarray
        Normalized weights summing to 1

    Notes
    -----
    - theta1 controls overall tilt (positive = increasing, negative = decreasing)
    - theta2 controls curvature (negative = convex decay from start)
    - When theta1 = theta2 = 0, gives uniform weights

    Examples
    --------
    >>> weights = exp_almon_weights(22, theta1=0.0, theta2=-0.01)
    >>> weights.sum()  # Should be 1.0
    """
    if n_lags < 1:
        raise ValueError("n_lags must be positive")

    # k goes from 1 to n_lags (day 1 is most recent, day n_lags is oldest)
    k = np.arange(1, n_lags + 1)

    # Compute raw weights
    log_weights = theta1 * k + theta2 * k ** 2

    # Numerical stability: subtract max before exp
    log_weights = log_weights - log_weights.max()
    raw_weights = np.exp(log_weights)

    # Normalize
    weights = raw_weights / raw_weights.sum()

    return weights


def beta_weights(
    n_lags: int,
    alpha: float,
    beta_param: float,
    eps: float = 1e-5
) -> np.ndarray:
    """
    Beta polynomial weights based on beta distribution density.

    The weight for lag k is proportional to: x^(alpha-1) * (1-x)^(beta-1)
    where x = k / (n_lags + 1) is normalized to (0, 1).

    Parameters
    ----------
    n_lags : int
        Number of high-frequency lags
    alpha : float
        Beta distribution alpha parameter (> 0)
    beta_param : float
        Beta distribution beta parameter (> 0)
    eps : float
        Small constant to avoid boundary issues

    Returns
    -------
    weights : np.ndarray
        Normalized weights summing to 1

    Notes
    -----
    - alpha = beta = 1: uniform weights
    - alpha = 1, beta > 1: declining weights (most weight on recent)
    - alpha > 1, beta = 1: increasing weights (most weight on old)
    - alpha > 1, beta > 1: hump-shaped weights

    Examples
    --------
    >>> weights = beta_weights(22, alpha=1.0, beta_param=5.0)
    >>> weights[0] > weights[-1]  # Recent lags should have higher weight
    True
    """
    if n_lags < 1:
        raise ValueError("n_lags must be positive")
    if alpha <= 0 or beta_param <= 0:
        raise ValueError("alpha and beta must be positive")

    # Normalized positions on (0, 1)
    k = np.linspace(eps, 1 - eps, n_lags)

    # Beta density (unnormalized)
    log_weights = (alpha - 1) * np.log(k) + (beta_param - 1) * np.log(1 - k)

    # Numerical stability
    log_weights = log_weights - log_weights.max()
    raw_weights = np.exp(log_weights)

    # Normalize
    weights = raw_weights / raw_weights.sum()

    return weights


def step_weights(
    n_lags: int,
    step_size: int = 5
) -> np.ndarray:
    """
    Step function weights (piecewise constant).

    Groups high-frequency observations into blocks of equal weight.
    This is useful when the number of lags is small enough for
    semi-parametric estimation.

    Parameters
    ----------
    n_lags : int
        Number of high-frequency lags
    step_size : int
        Number of lags per step/block

    Returns
    -------
    weights : np.ndarray
        Normalized weights summing to 1

    Notes
    -----
    - Useful for U-MIDAS when n_lags is small
    - Does not require optimization (no parameters)
    """
    if n_lags < 1:
        raise ValueError("n_lags must be positive")
    if step_size < 1:
        raise ValueError("step_size must be positive")

    n_steps = max(1, n_lags // step_size)
    weights = np.zeros(n_lags)

    for i in range(n_steps):
        start = i * step_size
        end = min((i + 1) * step_size, n_lags)
        weights[start:end] = 1.0 / n_steps

    # Handle remaining lags if n_lags is not divisible by step_size
    if n_lags % step_size != 0:
        remainder_start = n_steps * step_size
        weights[remainder_start:] = 1.0 / n_steps

    # Normalize
    weights = weights / weights.sum()

    return weights


def uniform_weights(n_lags: int) -> np.ndarray:
    """
    Uniform (equal) weights.

    Parameters
    ----------
    n_lags : int
        Number of high-frequency lags

    Returns
    -------
    weights : np.ndarray
        Equal weights summing to 1
    """
    if n_lags < 1:
        raise ValueError("n_lags must be positive")

    return np.ones(n_lags) / n_lags


def declining_weights(
    n_lags: int,
    decay: float = 0.9
) -> np.ndarray:
    """
    Geometric declining weights.

    Weight k is proportional to decay^k.

    Parameters
    ----------
    n_lags : int
        Number of high-frequency lags
    decay : float
        Decay rate (0 < decay < 1)

    Returns
    -------
    weights : np.ndarray
        Normalized weights summing to 1
    """
    if n_lags < 1:
        raise ValueError("n_lags must be positive")
    if not 0 < decay < 1:
        raise ValueError("decay must be between 0 and 1")

    k = np.arange(n_lags)
    raw_weights = decay ** k

    return raw_weights / raw_weights.sum()


def get_weight_function(weight_type: str):
    """
    Get weight function by name.

    Parameters
    ----------
    weight_type : str
        One of: 'exp_almon', 'beta', 'step', 'uniform', 'declining'

    Returns
    -------
    callable
        Weight function
    """
    weight_funcs = {
        'exp_almon': exp_almon_weights,
        'beta': beta_weights,
        'step': step_weights,
        'uniform': uniform_weights,
        'declining': declining_weights,
    }

    if weight_type not in weight_funcs:
        raise ValueError(
            f"Unknown weight type: {weight_type}. "
            f"Available: {list(weight_funcs.keys())}"
        )

    return weight_funcs[weight_type]


def weight_initial_params(weight_type: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get initial parameters and bounds for weight optimization.

    Parameters
    ----------
    weight_type : str
        Type of weight function

    Returns
    -------
    init_params : np.ndarray
        Initial parameter values
    bounds : list of tuples
        Parameter bounds for optimization
    """
    if weight_type == 'exp_almon':
        # theta1, theta2
        init_params = np.array([0.0, -0.01])
        bounds = [(-2.0, 2.0), (-0.5, 0.1)]

    elif weight_type == 'beta':
        # alpha, beta
        init_params = np.array([1.0, 5.0])
        bounds = [(0.1, 10.0), (0.1, 20.0)]

    elif weight_type in ['step', 'uniform', 'declining']:
        # No parameters to optimize
        init_params = np.array([])
        bounds = []

    else:
        raise ValueError(f"Unknown weight type: {weight_type}")

    return init_params, bounds


def compute_weights(
    weight_type: str,
    n_lags: int,
    params: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute weights given type and parameters.

    Parameters
    ----------
    weight_type : str
        Type of weight function
    n_lags : int
        Number of lags
    params : np.ndarray, optional
        Weight parameters (for parametric weight functions)

    Returns
    -------
    weights : np.ndarray
        Normalized weights
    """
    if weight_type == 'exp_almon':
        if params is None or len(params) < 2:
            params = np.array([0.0, -0.01])
        return exp_almon_weights(n_lags, params[0], params[1])

    elif weight_type == 'beta':
        if params is None or len(params) < 2:
            params = np.array([1.0, 5.0])
        return beta_weights(n_lags, params[0], params[1])

    elif weight_type == 'step':
        step_size = int(params[0]) if params is not None and len(params) > 0 else 5
        return step_weights(n_lags, step_size)

    elif weight_type == 'uniform':
        return uniform_weights(n_lags)

    elif weight_type == 'declining':
        decay = params[0] if params is not None and len(params) > 0 else 0.9
        return declining_weights(n_lags, decay)

    else:
        raise ValueError(f"Unknown weight type: {weight_type}")


def weight_gradient(
    weight_type: str,
    n_lags: int,
    params: np.ndarray
) -> np.ndarray:
    """
    Compute numerical gradient of weights with respect to parameters.

    Parameters
    ----------
    weight_type : str
        Type of weight function
    n_lags : int
        Number of lags
    params : np.ndarray
        Current parameter values

    Returns
    -------
    gradient : np.ndarray
        Shape (n_params, n_lags) gradient matrix
    """
    eps = 1e-6
    n_params = len(params)
    gradient = np.zeros((n_params, n_lags))

    w0 = compute_weights(weight_type, n_lags, params)

    for i in range(n_params):
        params_plus = params.copy()
        params_plus[i] += eps
        w_plus = compute_weights(weight_type, n_lags, params_plus)
        gradient[i] = (w_plus - w0) / eps

    return gradient
