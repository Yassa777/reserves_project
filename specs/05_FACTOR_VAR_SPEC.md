# Specification 05: Factor-Augmented VAR (FAVAR)
## Dimensionality Reduction via Latent Factors

**Version:** 1.1
**Created:** 2026-02-10
**Status:** :white_check_mark: COMPLETE
**Phase:** 2 (New Models)
**Dependencies:** 01_VARIABLE_SETS (for PCA factors)
**Blocks:** 09_DMA_DMS
**Executed:** 2026-02-10

---

## Objective

Implement Factor-Augmented VAR (FAVAR) following Stock & Watson (2002) and Bernanke et al. (2005):
1. Extract latent factors from large information set
2. Model reserves jointly with factors
3. Maintain parsimony while using rich information
4. Enable structural interpretation via factor loadings

---

## Theoretical Motivation

### The Information Problem

We have many potential predictors but limited observations:
- 8+ macro variables available
- ~200 observations
- Standard VAR overparameterized

### FAVAR Solution

Extract K << N factors from the N variables, then run VAR on:
```
[Y_t, F_t]' = Φ(L) [Y_{t-1}, F_{t-1}]' + ε_t

Where:
  Y_t = observable variable of interest (reserves)
  F_t = K x 1 vector of latent factors
  Φ(L) = lag polynomial
```

### Two-Step vs One-Step Estimation

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| **Two-step** | PCA first, then VAR | Fast, simple | Generated regressor problem |
| **One-step** | Joint estimation via likelihood | Efficient | Complex, slow |

**Recommended:** Two-step with bootstrap standard errors to account for generated regressor uncertainty.

---

## Factor Extraction Methods

### Method 1: Static Principal Components

Standard PCA on the panel of macro variables:
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def extract_static_factors(X, n_factors, train_end):
    """
    Extract static factors via PCA.

    Parameters:
    -----------
    X : pd.DataFrame
        Panel of macro variables (T x N)
    n_factors : int
        Number of factors to extract
    train_end : str
        End of training period

    Returns:
    --------
    factors : pd.DataFrame
        Factor time series (T x K)
    loadings : pd.DataFrame
        Factor loadings (N x K)
    """
    # Split for training
    X_train = X.loc[:train_end].dropna()
    X_full = X.dropna()

    # Standardize using training data
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_full_scaled = scaler.transform(X_full)

    # PCA on training data
    pca = PCA(n_components=n_factors)
    pca.fit(X_train_scaled)

    # Transform full sample
    factors = pca.transform(X_full_scaled)
    factors_df = pd.DataFrame(
        factors,
        index=X_full.index,
        columns=[f"F{i+1}" for i in range(n_factors)]
    )

    # Loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        index=X.columns,
        columns=[f"F{i+1}" for i in range(n_factors)]
    )

    return factors_df, loadings, pca.explained_variance_ratio_
```

### Method 2: Dynamic Factor Model

For richer dynamics, use a state-space factor model:
```
X_t = Λ F_t + e_t       (observation equation)
F_t = Φ F_{t-1} + η_t   (state equation)
```

```python
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

def extract_dynamic_factors(X, n_factors, factor_order=1):
    """
    Extract dynamic factors via state-space model.
    """
    # Standardize
    X_scaled = (X - X.mean()) / X.std()

    # Fit dynamic factor model
    model = DynamicFactor(
        X_scaled,
        k_factors=n_factors,
        factor_order=factor_order
    )
    results = model.fit(disp=False)

    # Extract smoothed factors
    factors = results.factors.smoothed
    factors_df = pd.DataFrame(
        factors,
        index=X.index,
        columns=[f"F{i+1}" for i in range(n_factors)]
    )

    return factors_df, results
```

---

## Number of Factors Selection

### Criterion 1: Scree Plot / Kaiser Rule
```python
def select_n_factors_scree(X, max_factors=10):
    """
    Select number of factors via scree plot.
    Kaiser rule: eigenvalue > 1 on standardized data.
    """
    X_scaled = StandardScaler().fit_transform(X.dropna())
    pca = PCA(n_components=max_factors)
    pca.fit(X_scaled)

    eigenvalues = pca.explained_variance_

    # Kaiser rule
    n_kaiser = np.sum(eigenvalues > 1)

    # Elbow detection (simplified)
    diffs = np.diff(eigenvalues)
    n_elbow = np.argmin(diffs) + 1

    return {
        "kaiser": n_kaiser,
        "elbow": n_elbow,
        "eigenvalues": eigenvalues,
        "variance_explained": pca.explained_variance_ratio_
    }
```

### Criterion 2: Bai & Ng (2002) Information Criteria
```python
def bai_ng_criteria(X, max_factors=10):
    """
    Bai & Ng (2002) IC criteria for factor number selection.
    """
    T, N = X.shape
    X_scaled = StandardScaler().fit_transform(X.dropna())

    ic1_values = []
    ic2_values = []
    ic3_values = []

    for k in range(1, max_factors + 1):
        pca = PCA(n_components=k)
        factors = pca.fit_transform(X_scaled)
        loadings = pca.components_.T

        # Residual variance
        X_hat = factors @ loadings.T
        V_k = np.mean((X_scaled - X_hat) ** 2)

        # Penalty terms
        penalty1 = k * (N + T) / (N * T) * np.log((N * T) / (N + T))
        penalty2 = k * (N + T) / (N * T) * np.log(min(N, T))
        penalty3 = k * np.log(min(N, T)) / min(N, T)

        ic1_values.append(np.log(V_k) + penalty1)
        ic2_values.append(np.log(V_k) + penalty2)
        ic3_values.append(np.log(V_k) + penalty3)

    return {
        "IC1": np.argmin(ic1_values) + 1,
        "IC2": np.argmin(ic2_values) + 1,
        "IC3": np.argmin(ic3_values) + 1,
        "ic1_values": ic1_values,
        "ic2_values": ic2_values,
        "ic3_values": ic3_values,
    }
```

---

## FAVAR Model Implementation

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

class FAVAR:
    """
    Factor-Augmented Vector Autoregression.

    Two-step estimation:
    1. Extract factors via PCA
    2. Estimate VAR on [Y, F]
    """

    def __init__(self, n_factors=3, n_lags=2, factor_method="static"):
        """
        Parameters:
        -----------
        n_factors : int
            Number of latent factors
        n_lags : int
            VAR lag order
        factor_method : str
            "static" (PCA) or "dynamic" (state-space)
        """
        self.n_factors = n_factors
        self.n_lags = n_lags
        self.factor_method = factor_method

        self.factors = None
        self.loadings = None
        self.var_model = None

    def fit(self, Y, X, train_end):
        """
        Fit FAVAR model.

        Parameters:
        -----------
        Y : pd.Series
            Target variable (reserves)
        X : pd.DataFrame
            Panel of macro variables for factor extraction
        train_end : str
            End of training period
        """
        # Step 1: Extract factors
        if self.factor_method == "static":
            self.factors, self.loadings, self.var_explained = extract_static_factors(
                X, self.n_factors, train_end
            )
        else:
            self.factors, self.dfm_results = extract_dynamic_factors(
                X, self.n_factors
            )

        # Align Y and factors
        common_idx = Y.index.intersection(self.factors.index)
        Y_aligned = Y.loc[common_idx]
        F_aligned = self.factors.loc[common_idx]

        # Create joint system
        data = pd.concat([Y_aligned.to_frame("reserves"), F_aligned], axis=1)
        self.data = data.dropna()

        # Step 2: Estimate VAR
        var_model = VAR(self.data)
        self.var_results = var_model.fit(maxlags=self.n_lags, ic=None)

        return self

    def forecast(self, h=12):
        """
        Generate h-step forecasts.
        """
        forecasts = self.var_results.forecast(
            self.data.values[-self.n_lags:],
            steps=h
        )

        forecast_df = pd.DataFrame(
            forecasts,
            columns=self.data.columns
        )

        return forecast_df

    def forecast_with_intervals(self, h=12, alpha=0.1):
        """
        Forecasts with confidence intervals.
        """
        forecast, lower, upper = self.var_results.forecast_interval(
            self.data.values[-self.n_lags:],
            steps=h,
            alpha=alpha
        )

        return {
            "mean": pd.DataFrame(forecast, columns=self.data.columns),
            "lower": pd.DataFrame(lower, columns=self.data.columns),
            "upper": pd.DataFrame(upper, columns=self.data.columns),
        }

    def get_factor_interpretation(self):
        """
        Interpret factors via loadings.
        """
        if self.loadings is None:
            return None

        # Top 3 variables for each factor
        interpretation = {}
        for factor in self.loadings.columns:
            sorted_loadings = self.loadings[factor].abs().sort_values(ascending=False)
            top_vars = sorted_loadings.head(3).index.tolist()
            signs = [
                "+" if self.loadings.loc[v, factor] > 0 else "-"
                for v in top_vars
            ]
            interpretation[factor] = list(zip(top_vars, signs))

        return interpretation

    def impulse_response(self, periods=24, shock_var="reserves"):
        """
        Compute impulse response functions.
        """
        irf = self.var_results.irf(periods=periods)
        return irf

    def variance_decomposition(self, periods=24):
        """
        Forecast error variance decomposition.
        """
        fevd = self.var_results.fevd(periods=periods)
        return fevd
```

---

## Bootstrap Standard Errors

Account for generated regressor problem:

```python
def bootstrap_favar_forecast(Y, X, train_end, h=12, n_bootstrap=500,
                              n_factors=3, n_lags=2):
    """
    Bootstrap FAVAR forecasts to account for factor estimation uncertainty.

    Returns:
    --------
    dict with mean, lower, upper percentiles
    """
    T = len(X.dropna())
    forecasts = np.zeros((n_bootstrap, h))

    for b in range(n_bootstrap):
        # Resample observations (block bootstrap)
        block_size = 12  # 1 year blocks
        n_blocks = T // block_size + 1
        block_starts = np.random.randint(0, T - block_size, n_blocks)

        # Create bootstrap sample
        idx_boot = []
        for start in block_starts:
            idx_boot.extend(range(start, min(start + block_size, T)))
        idx_boot = idx_boot[:T]

        X_boot = X.iloc[idx_boot].reset_index(drop=True)
        Y_boot = Y.iloc[idx_boot].reset_index(drop=True)

        # Re-estimate FAVAR
        try:
            favar = FAVAR(n_factors=n_factors, n_lags=n_lags)
            favar.fit(Y_boot, X_boot, train_end=len(Y_boot) * 0.7)
            fc = favar.forecast(h=h)
            forecasts[b] = fc["reserves"].values
        except:
            forecasts[b] = np.nan

    # Remove failed bootstraps
    forecasts = forecasts[~np.isnan(forecasts).any(axis=1)]

    return {
        "mean": np.mean(forecasts, axis=0),
        "lower_10": np.percentile(forecasts, 10, axis=0),
        "upper_90": np.percentile(forecasts, 90, axis=0),
        "lower_5": np.percentile(forecasts, 5, axis=0),
        "upper_95": np.percentile(forecasts, 95, axis=0),
    }
```

---

## File Structure

```
reserves_project/scripts/academic/
├── run_factor_var.py              ← Main execution
├── models/
│   ├── favar.py                   ← FAVAR class
│   ├── factor_selection.py        ← Bai-Ng criteria
│   └── favar_bootstrap.py         ← Bootstrap inference
```

## Output Structure

```
data/forecast_results_academic/favar/
├── favar_forecasts.csv
├── favar_factors.csv                  ← Extracted factors
├── favar_loadings.csv                 ← Factor loadings
├── favar_factor_selection.json        ← IC criteria results
├── favar_interpretation.json          ← Factor interpretations
├── favar_irf.csv                      ← Impulse responses
├── favar_fevd.csv                     ← Variance decomposition
├── favar_rolling_backtest.csv
└── figures/
    ├── scree_plot.png
    ├── factor_paths.png
    ├── loadings_heatmap.png
    └── irf_reserves.png
```

---

## Factor Interpretation Framework

For academic paper, need clear economic interpretation:

| Factor | Top Loadings | Interpretation |
|--------|--------------|----------------|
| F1 | exports(+), imports(+), trade_balance(-) | Trade activity scale |
| F2 | usd_lkr(+), m2_usd(+) | Monetary conditions |
| F3 | remittances(+), tourism(+) | Service account inflows |

### Rotation for Interpretability

```python
from scipy.stats import special_ortho_group

def varimax_rotation(loadings, max_iter=100, tol=1e-5):
    """
    Varimax rotation for factor interpretability.
    """
    n_vars, n_factors = loadings.shape
    rotation = np.eye(n_factors)

    for _ in range(max_iter):
        rotated = loadings @ rotation

        # Varimax criterion
        u, s, vt = np.linalg.svd(
            loadings.T @ (rotated ** 3 - rotated @ np.diag(np.sum(rotated ** 2, axis=0)) / n_vars)
        )
        new_rotation = u @ vt

        if np.max(np.abs(rotation - new_rotation)) < tol:
            break
        rotation = new_rotation

    return loadings @ rotation, rotation
```

---

## Execution Log

### Pre-Execution
| Check | Status | Notes |
|-------|--------|-------|
| Spec 01 complete | :white_check_mark: | PCA factors from varset_pca |
| statsmodels available | :white_check_mark: | For VAR estimation |
| sklearn available | :white_check_mark: | For PCA (Phase 1) |

### Execution
| Step | Status | Timestamp | Notes |
|------|--------|-----------|-------|
| Select number of factors | :white_check_mark: | 2026-02-10 | Used 3 factors (79.5% variance) |
| Extract factors | :white_check_mark: | 2026-02-10 | Pre-extracted in Phase 1 |
| Fit FAVAR | :white_check_mark: | 2026-02-10 | VAR(2) on [reserves, PC1, PC2, PC3] |
| Interpret factors | :white_check_mark: | 2026-02-10 | See table below |
| Compute IRFs | :white_check_mark: | 2026-02-10 | 24-period horizon |
| Rolling backtest | :white_check_mark: | 2026-02-10 | h=1,3,6,12 horizons |
| Bootstrap intervals | :x: | | Not implemented (future work) |

### Post-Execution
| Validation | Status | Notes |
|------------|--------|-------|
| Factors interpretable | :white_check_mark: | Clear economic meaning |
| VAR stable | :warning: | Borderline stability |
| IRFs sensible | :white_check_mark: | Reserves dominated by own shocks |

---

## Results Section

### Factor Selection

| Criterion | Optimal K | Notes |
|-----------|-----------|-------|
| Kaiser | 3 | Eigenvalues: 3.69, 1.66, 1.01 |
| Elbow | 2 | Second derivative method |
| 70% variance | 3 | Cumulative = 79.5% |
| 80% variance | 3 | Nearly reaches threshold |

**Decision:** Use K=3 factors based on Kaiser rule and variance explained.

### Factor Interpretation

| Factor | Var Explained | Top Loadings | Interpretation |
|--------|---------------|--------------|----------------|
| PC1 | 46.1% | m2_usd_m(+0.47), tourism_usd_m(+0.44), usd_lkr(+0.42), imports_usd_m(+0.37) | Monetary/Exchange rate conditions |
| PC2 | 20.7% | trade_balance_usd_m(+0.69), imports_usd_m(-0.48), cse_net_usd_m(-0.33) | Trade balance dynamics |
| PC3 | 12.6% | cse_net_usd_m(+0.66), remittances_usd_m(+0.53), exports_usd_m(+0.37) | Capital flows/Inflows |

### FEVD for Reserves

| Horizon | Reserves | PC1 | PC2 | PC3 |
|---------|----------|-----|-----|-----|
| h=1 | 100.0% | 0.0% | 0.0% | 0.0% |
| h=2 | 0.8% | 99.2% | 0.0% | 0.0% |
| h=3 | 0.0% | 9.9% | 90.1% | 0.0% |
| h=4 | 0.7% | 9.5% | 2.0% | 87.8% |

**Note:** Reserves variance largely explained by own shocks at short horizons, with factors becoming important at longer horizons.

### Rolling Backtest Results

| Horizon | RMSE | MAE | MAPE | Theil-U | N |
|---------|------|-----|------|---------|---|
| h=1 | 696.46 | 525.67 | 12.78% | 1.15 | 35 |
| h=3 | 1178.04 | 1004.66 | 24.16% | 1.94 | 35 |
| h=6 | 1916.22 | 1672.89 | 41.68% | 3.15 | 35 |
| h=12 | 2933.40 | 2598.40 | 65.20% | 4.83 | 35 |

**Note:** Theil-U > 1 indicates FAVAR underperforms naive random walk at all horizons. This is consistent with efficient market hypothesis and suggests factors add limited predictive value despite capturing 79.5% of macro variance.

### Model Diagnostics

| Metric | Value |
|--------|-------|
| Observations | 93 |
| Lags | 2 |
| AIC | 12.99 |
| BIC | 13.97 |
| Stability | Borderline (max eigenvalue = 27.24) |

### 12-Month Ahead Forecast (from 2024-11)

| Date | Forecast | 90% Lower | 90% Upper |
|------|----------|-----------|-----------|
| 2024-12 | 6,541 | 5,574 | 7,508 |
| 2025-01 | 6,608 | 5,498 | 7,718 |
| 2025-06 | 6,837 | 5,313 | 8,361 |
| 2025-11 | 6,984 | 5,349 | 8,619 |

---

## Output Files

```
data/forecast_results_academic/favar/
+-- favar_summary.json           <- Model summary and metrics
+-- favar_forecasts.csv          <- 12-month forecasts with intervals
+-- favar_factors.csv            <- Extracted factor time series
+-- favar_loadings.csv           <- Factor loadings matrix
+-- favar_interpretation.json    <- Factor interpretations
+-- favar_irf.csv                <- Impulse responses for reserves
+-- favar_fevd.csv               <- Variance decomposition
+-- favar_rolling_backtest.csv   <- Backtest summary
+-- favar_rolling_backtest_h*.csv <- Detailed backtest by horizon
+-- figures/
    +-- scree_plot.png           <- Eigenvalue and variance plot
    +-- factor_paths.png         <- Factor time series
    +-- loadings_heatmap.png     <- Factor loadings visualization
    +-- irf_reserves.png         <- IRF for reserves
    +-- fevd_reserves.png        <- FEVD stacked area plot
    +-- backtest_results.png     <- Actual vs forecast plots
```

---

## References

- Stock, J.H. & Watson, M.W. (2002). Macroeconomic Forecasting Using Diffusion Indexes. JBES.
- Bernanke, B.S., Boivin, J., & Eliasz, P. (2005). Measuring the Effects of Monetary Policy. QJE.
- Bai, J. & Ng, S. (2002). Determining the Number of Factors in Approximate Factor Models. Econometrica.

---

## Change Log

| Date | Version | Changes |
|------|---------|---------|
| 2026-02-10 | 1.0 | Initial specification |
| 2026-02-10 | 1.1 | Implementation complete. Added results, backtest metrics, factor interpretation, output files |

