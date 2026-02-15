"""
Factor Selection Methods for FAVAR.

Implements:
1. Bai-Ng (2002) Information Criteria for factor number selection
2. Kaiser Rule (eigenvalue > 1)
3. Scree plot elbow detection

Reference: Bai, J. & Ng, S. (2002). Determining the Number of Factors
           in Approximate Factor Models. Econometrica, 70(1), 191-221.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Tuple, Optional


def bai_ng_criteria(
    X: pd.DataFrame,
    max_factors: int = 10,
    train_end: Optional[pd.Timestamp] = None
) -> Dict[str, Any]:
    """
    Bai & Ng (2002) information criteria for selecting number of factors.

    Three criteria are computed:
    - IC1: Uses penalty (N+T)/(NT) * ln((NT)/(N+T))
    - IC2: Uses penalty (N+T)/(NT) * ln(min(N,T))
    - IC3: Uses penalty ln(min(N,T)) / min(N,T)

    Parameters
    ----------
    X : pd.DataFrame
        Panel of variables (T x N) for factor extraction
    max_factors : int
        Maximum number of factors to consider
    train_end : pd.Timestamp, optional
        If provided, use only training data for standardization

    Returns
    -------
    dict
        Dictionary containing:
        - IC1, IC2, IC3: Optimal number of factors by each criterion
        - ic1_values, ic2_values, ic3_values: Full series of IC values
        - variance_explained: Variance explained for each factor count
    """
    # Handle missing values
    X_clean = X.dropna()

    if train_end is not None:
        X_train = X_clean.loc[:train_end]
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_scaled = scaler.transform(X_clean)
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)

    T, N = X_scaled.shape
    max_factors = min(max_factors, min(T, N) - 1)

    ic1_values = []
    ic2_values = []
    ic3_values = []
    variance_explained = []

    for k in range(1, max_factors + 1):
        pca = PCA(n_components=k)
        factors = pca.fit_transform(X_scaled)
        loadings = pca.components_.T

        # Residual variance (V_k in Bai-Ng notation)
        X_hat = factors @ loadings.T
        V_k = np.mean((X_scaled - X_hat) ** 2)

        # Penalty terms from Bai-Ng (2002)
        # Note: Using C_NT = min(sqrt(N), sqrt(T)) approximation
        C_NT_sq = min(N, T)

        penalty1 = k * (N + T) / (N * T) * np.log((N * T) / (N + T))
        penalty2 = k * (N + T) / (N * T) * np.log(C_NT_sq)
        penalty3 = k * np.log(C_NT_sq) / C_NT_sq

        ic1_values.append(np.log(V_k) + penalty1)
        ic2_values.append(np.log(V_k) + penalty2)
        ic3_values.append(np.log(V_k) + penalty3)

        variance_explained.append(pca.explained_variance_ratio_.sum())

    return {
        "IC1": int(np.argmin(ic1_values) + 1),
        "IC2": int(np.argmin(ic2_values) + 1),
        "IC3": int(np.argmin(ic3_values) + 1),
        "ic1_values": ic1_values,
        "ic2_values": ic2_values,
        "ic3_values": ic3_values,
        "variance_explained": variance_explained,
        "n_obs": T,
        "n_vars": N,
    }


def kaiser_criterion(
    X: pd.DataFrame,
    max_factors: int = 10,
    train_end: Optional[pd.Timestamp] = None
) -> Dict[str, Any]:
    """
    Kaiser rule: retain factors with eigenvalue > 1 on standardized data.

    Parameters
    ----------
    X : pd.DataFrame
        Panel of variables (T x N)
    max_factors : int
        Maximum factors to compute
    train_end : pd.Timestamp, optional
        End of training period for standardization

    Returns
    -------
    dict
        n_factors: Number of factors by Kaiser rule
        eigenvalues: All computed eigenvalues
    """
    X_clean = X.dropna()

    if train_end is not None:
        X_train = X_clean.loc[:train_end]
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_scaled = scaler.transform(X_clean)
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)

    max_factors = min(max_factors, min(X_scaled.shape) - 1)

    pca = PCA(n_components=max_factors)
    pca.fit(X_scaled)

    eigenvalues = pca.explained_variance_

    # Kaiser rule: count eigenvalues > 1
    n_kaiser = int(np.sum(eigenvalues > 1))

    return {
        "n_factors": n_kaiser,
        "eigenvalues": eigenvalues.tolist(),
        "variance_explained": pca.explained_variance_ratio_.tolist(),
    }


def elbow_detection(
    X: pd.DataFrame,
    max_factors: int = 10,
    train_end: Optional[pd.Timestamp] = None
) -> Dict[str, Any]:
    """
    Detect elbow in scree plot for factor selection.

    Uses second derivative approach: find where the decline in
    eigenvalues starts to level off.

    Parameters
    ----------
    X : pd.DataFrame
        Panel of variables (T x N)
    max_factors : int
        Maximum factors to consider
    train_end : pd.Timestamp, optional
        End of training period

    Returns
    -------
    dict
        n_factors: Number of factors at elbow
        eigenvalues: All eigenvalues
        second_diffs: Second differences for elbow detection
    """
    X_clean = X.dropna()

    if train_end is not None:
        X_train = X_clean.loc[:train_end]
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_scaled = scaler.transform(X_clean)
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)

    max_factors = min(max_factors, min(X_scaled.shape) - 1)

    pca = PCA(n_components=max_factors)
    pca.fit(X_scaled)

    eigenvalues = pca.explained_variance_

    # First differences (rate of decline)
    first_diffs = -np.diff(eigenvalues)

    # Second differences (acceleration of decline)
    if len(first_diffs) > 1:
        second_diffs = np.diff(first_diffs)
        # Elbow is where second derivative is most negative
        # (steepest decline in rate of change)
        n_elbow = int(np.argmax(second_diffs) + 2)  # +2 for double differencing
    else:
        second_diffs = np.array([])
        n_elbow = 1

    # Bound to reasonable range
    n_elbow = max(1, min(n_elbow, max_factors))

    return {
        "n_factors": n_elbow,
        "eigenvalues": eigenvalues.tolist(),
        "first_diffs": first_diffs.tolist(),
        "second_diffs": second_diffs.tolist() if len(second_diffs) > 0 else [],
    }


def select_n_factors(
    X: pd.DataFrame,
    max_factors: int = 10,
    train_end: Optional[pd.Timestamp] = None
) -> Dict[str, Any]:
    """
    Comprehensive factor selection using multiple criteria.

    Combines:
    - Bai-Ng IC1, IC2, IC3 criteria
    - Kaiser rule (eigenvalue > 1)
    - Elbow detection

    Parameters
    ----------
    X : pd.DataFrame
        Panel of variables for factor extraction
    max_factors : int
        Maximum number of factors to consider
    train_end : pd.Timestamp, optional
        End of training period

    Returns
    -------
    dict
        Comprehensive results from all methods plus recommendation
    """
    bai_ng = bai_ng_criteria(X, max_factors, train_end)
    kaiser = kaiser_criterion(X, max_factors, train_end)
    elbow = elbow_detection(X, max_factors, train_end)

    # Collect all recommendations
    recommendations = [
        bai_ng["IC1"],
        bai_ng["IC2"],
        bai_ng["IC3"],
        kaiser["n_factors"],
        elbow["n_factors"],
    ]

    # Use median as consensus
    consensus = int(np.median(recommendations))

    return {
        "bai_ng": bai_ng,
        "kaiser": kaiser,
        "elbow": elbow,
        "recommendations": {
            "IC1": bai_ng["IC1"],
            "IC2": bai_ng["IC2"],
            "IC3": bai_ng["IC3"],
            "Kaiser": kaiser["n_factors"],
            "Elbow": elbow["n_factors"],
        },
        "consensus": consensus,
        "variance_explained_by_consensus": bai_ng["variance_explained"][consensus - 1],
    }
