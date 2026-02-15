"""
PCA Factor Builder for Academic Reserves Forecasting Pipeline.

This module implements principal component extraction from macro variables
using training data only to avoid look-ahead bias.

Reference: Specification 01 - Variable Sets Definition
Reference: Stock, J.H. & Watson, M.W. (2002). Forecasting Using Principal Components.
"""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def build_pca_factors(
    df: pd.DataFrame,
    source_vars: list,
    n_components: int,
    train_end: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, Dict[str, Any]]:
    """
    Extract PCA factors using training data only.

    Implements out-of-sample PCA by:
    1. Fitting StandardScaler on training period only
    2. Fitting PCA on standardized training data
    3. Transforming full dataset using training-fitted parameters

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with DatetimeIndex
    source_vars : list
        Variables to extract factors from
    n_components : int
        Number of principal components to extract
    train_end : pd.Timestamp
        End date for training period (inclusive)

    Returns
    -------
    factors_df : pd.DataFrame
        DataFrame with PC1, PC2, ... columns and same index as input
    loadings : pd.DataFrame
        Component loadings matrix (variables x components)
    variance_explained : np.ndarray
        Variance explained ratio by each component
    metadata : dict
        Additional PCA metadata including scaler and pca objects
    """
    # Validate inputs
    missing_vars = [v for v in source_vars if v not in df.columns]
    if missing_vars:
        raise ValueError(f"Missing source variables: {missing_vars}")

    # Extract source data (dropna to ensure complete cases for PCA)
    source_data = df[source_vars].copy()

    # Split into training and full data
    train_mask = source_data.index <= train_end
    train_data = source_data.loc[train_mask].dropna()

    if len(train_data) < n_components:
        raise ValueError(
            f"Insufficient training observations ({len(train_data)}) "
            f"for {n_components} components"
        )

    # Get complete cases for full dataset
    full_data = source_data.dropna()

    # Step 1: Fit scaler on training data only
    scaler = StandardScaler()
    scaler.fit(train_data)

    # Step 2: Fit PCA on standardized training data
    pca = PCA(n_components=n_components)
    scaled_train = scaler.transform(train_data)
    pca.fit(scaled_train)

    # Step 3: Transform full dataset using training-fitted parameters
    scaled_full = scaler.transform(full_data)
    factors = pca.transform(scaled_full)

    # Create output DataFrame with factor columns
    factor_cols = [f"PC{i+1}" for i in range(n_components)]
    factors_df = pd.DataFrame(
        factors,
        index=full_data.index,
        columns=factor_cols
    )

    # Create loadings DataFrame for interpretation
    loadings = pd.DataFrame(
        pca.components_.T,
        index=source_vars,
        columns=factor_cols
    )

    # Variance explained
    variance_explained = pca.explained_variance_ratio_

    # Compile metadata
    metadata = {
        "n_components": n_components,
        "source_vars": source_vars,
        "train_end": str(train_end.date()),
        "n_train_obs": len(train_data),
        "n_total_obs": len(full_data),
        "cumulative_variance_explained": float(np.cumsum(variance_explained)[-1]),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_std": scaler.scale_.tolist(),
    }

    return factors_df, loadings, variance_explained, metadata


def interpret_loadings(
    loadings: pd.DataFrame,
    variance_explained: np.ndarray,
    threshold: float = 0.3
) -> pd.DataFrame:
    """
    Generate interpretable summary of PCA loadings.

    Parameters
    ----------
    loadings : pd.DataFrame
        Component loadings matrix from build_pca_factors
    variance_explained : np.ndarray
        Variance explained by each component
    threshold : float
        Minimum absolute loading to include in interpretation

    Returns
    -------
    pd.DataFrame
        Summary table with component, variance explained, top loadings, interpretation
    """
    interpretations = []

    for i, col in enumerate(loadings.columns):
        component_loadings = loadings[col].sort_values(key=abs, ascending=False)

        # Get top loadings above threshold
        top_loadings = component_loadings[abs(component_loadings) >= threshold]
        top_loading_strs = [
            f"{var}({'+' if val > 0 else ''}{val:.2f})"
            for var, val in top_loadings.items()
        ]

        # Simple interpretation based on dominant loadings
        interpretation = _infer_interpretation(top_loadings)

        interpretations.append({
            "component": col,
            "variance_explained_pct": float(variance_explained[i] * 100),
            "top_loadings": ", ".join(top_loading_strs[:4]),  # Top 4
            "interpretation": interpretation
        })

    return pd.DataFrame(interpretations)


def _infer_interpretation(top_loadings: pd.Series) -> str:
    """
    Infer economic interpretation from loading patterns.

    Parameters
    ----------
    top_loadings : pd.Series
        Sorted loadings for a single component

    Returns
    -------
    str
        Economic interpretation string
    """
    vars_present = set(top_loadings.index)

    # Trade scale factor
    if {"exports_usd_m", "imports_usd_m"}.issubset(vars_present):
        if top_loadings.get("exports_usd_m", 0) * top_loadings.get("imports_usd_m", 0) > 0:
            return "Trade scale factor"

    # Trade balance direction
    if "trade_balance_usd_m" in vars_present:
        if abs(top_loadings.get("trade_balance_usd_m", 0)) > 0.4:
            return "Trade balance factor"

    # Monetary conditions
    if {"usd_lkr", "m2_usd_m"}.issubset(vars_present):
        return "Monetary conditions"

    # Exchange rate factor
    if "usd_lkr" in vars_present and abs(top_loadings.get("usd_lkr", 0)) > 0.5:
        return "Exchange rate factor"

    # Service inflows
    if {"tourism_usd_m", "remittances_usd_m"}.issubset(vars_present):
        return "Service inflows"

    # Capital flows
    if "cse_net_usd_m" in vars_present and abs(top_loadings.get("cse_net_usd_m", 0)) > 0.4:
        return "Capital flows"

    # Default
    return "Mixed factor"


def generate_scree_data(
    df: pd.DataFrame,
    source_vars: list,
    train_end: pd.Timestamp,
    max_components: int = None
) -> pd.DataFrame:
    """
    Generate scree plot data for component selection.

    Parameters
    ----------
    df : pd.DataFrame
        Source data with DatetimeIndex
    source_vars : list
        Variables to include in PCA
    train_end : pd.Timestamp
        End of training period
    max_components : int, optional
        Maximum components to compute. Defaults to min(n_vars, n_obs).

    Returns
    -------
    pd.DataFrame
        DataFrame with component number, eigenvalue, variance explained,
        cumulative variance explained
    """
    # Prepare data
    source_data = df[source_vars].copy()
    train_data = source_data.loc[source_data.index <= train_end].dropna()

    # Determine max components
    n_vars = len(source_vars)
    n_obs = len(train_data)
    if max_components is None:
        max_components = min(n_vars, n_obs - 1)

    # Fit PCA with all possible components
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_data)

    pca = PCA(n_components=max_components)
    pca.fit(scaled_train)

    # Build scree data
    scree_data = pd.DataFrame({
        "component": list(range(1, max_components + 1)),
        "eigenvalue": pca.explained_variance_,
        "variance_explained_pct": pca.explained_variance_ratio_ * 100,
        "cumulative_variance_pct": np.cumsum(pca.explained_variance_ratio_) * 100,
    })

    return scree_data


def kaiser_criterion(scree_data: pd.DataFrame) -> int:
    """
    Apply Kaiser criterion to determine number of components.

    Kaiser criterion: retain components with eigenvalue > 1
    (when using correlation matrix, which StandardScaler achieves).

    Parameters
    ----------
    scree_data : pd.DataFrame
        Output from generate_scree_data

    Returns
    -------
    int
        Recommended number of components
    """
    return int((scree_data["eigenvalue"] > 1).sum())


def elbow_criterion(scree_data: pd.DataFrame) -> int:
    """
    Apply elbow criterion to determine number of components.

    Finds the point of maximum curvature in the scree plot.

    Parameters
    ----------
    scree_data : pd.DataFrame
        Output from generate_scree_data

    Returns
    -------
    int
        Recommended number of components
    """
    eigenvalues = scree_data["eigenvalue"].values

    # Simple second derivative approach
    if len(eigenvalues) < 3:
        return 1

    # Compute second differences (approximation of curvature)
    second_diff = np.diff(eigenvalues, n=2)

    # Find point of maximum curvature (most negative second derivative)
    elbow_idx = np.argmin(second_diff) + 1  # +1 for offset due to differencing

    return int(elbow_idx + 1)  # +1 for 1-indexed components
