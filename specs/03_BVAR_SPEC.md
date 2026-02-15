# Specification 03: Bayesian VAR with Minnesota Prior
## Shrinkage-Based Multivariate Forecasting

**Version:** 1.0
**Created:** 2026-02-10
**Status:** :white_check_mark: COMPLETE
**Phase:** 2 (New Models)
**Dependencies:** 01_VARIABLE_SETS (variable set definitions)
**Blocks:** 09_DMA_DMS (provides forecasts for model averaging)

---

## Objective

Implement Bayesian Vector Autoregression (BVAR) with Minnesota prior for reserves forecasting:
1. Address overparameterization issues seen in standard VECM
2. Provide regularized coefficient estimates
3. Enable density (probabilistic) forecasting
4. Generate posterior distributions for uncertainty quantification

---

## Theoretical Motivation

### The Overparameterization Problem

A VAR(p) with k variables has `k + k²p` parameters. For our baseline system:
- k = 5 variables, p = 2 lags → 5 + 50 = 55 parameters
- With 200 observations → ~3.6 obs per parameter (problematic)

**Minnesota Prior Solution:**

Shrink coefficients toward a random walk prior:
- Own first lag ≈ 1 (random walk for each variable)
- Other coefficients ≈ 0 (limited cross-variable dynamics)
- Tighter prior = more shrinkage = fewer effective parameters

### Prior Specification

For coefficient `A_ij,l` (effect of lag-l of variable j on variable i):

```
A_ij,l ~ N(δ_ij, σ_ij,l²)

Where:
  δ_ij = 1 if i=j and l=1 (own first lag)
         0 otherwise

  σ_ij,l² = (λ₁/l^λ₃)² × (σ_i/σ_j)²   if i ≠ j (cross-equation)
          = (λ₁/l^λ₃)²                  if i = j (own lags)

Hyperparameters:
  λ₁ = overall tightness (0.1 = tight, 0.5 = loose)
  λ₃ = lag decay (1 = linear, 2 = quadratic)
```

---

## Implementation Details

### Package Options

| Package | Pros | Cons |
|---------|------|------|
| `bvarpy` | Purpose-built, Minnesota prior | Less maintained |
| `PyMC` | Flexible, full Bayesian | Complex setup |
| `statsmodels` + custom | Familiar API | Manual prior implementation |
| `arviz` + custom | Great diagnostics | Need to code model |

**Recommended:** Custom implementation using `numpy` + `scipy` for clarity and control.

### Core BVAR Implementation

```python
import numpy as np
from scipy import linalg
from scipy.stats import invwishart, matrix_normal

class BayesianVAR:
    """
    Bayesian VAR with Minnesota Prior.

    Estimation via Gibbs sampling or direct posterior computation
    (when using natural conjugate prior).
    """

    def __init__(self, n_lags=2, lambda1=0.2, lambda3=1.0, n_draws=5000, n_burn=1000):
        """
        Parameters:
        -----------
        n_lags : int
            Number of VAR lags
        lambda1 : float
            Overall tightness (smaller = more shrinkage)
        lambda3 : float
            Lag decay (1 = linear, 2 = quadratic)
        n_draws : int
            Number of posterior draws
        n_burn : int
            Burn-in draws to discard
        """
        self.n_lags = n_lags
        self.lambda1 = lambda1
        self.lambda3 = lambda3
        self.n_draws = n_draws
        self.n_burn = n_burn

        self.coef_posterior = None
        self.sigma_posterior = None

    def fit(self, Y):
        """
        Fit BVAR to data.

        Parameters:
        -----------
        Y : np.array, shape (T, k)
            Multivariate time series
        """
        T, k = Y.shape

        # Create lagged matrices
        Y_lag, Y_obs = self._create_lags(Y)
        T_eff = Y_obs.shape[0]

        # Add constant
        X = np.column_stack([np.ones(T_eff), Y_lag])
        n_coefs = X.shape[1]

        # Construct Minnesota prior
        prior_mean, prior_var = self._minnesota_prior(k, n_coefs)

        # OLS estimates (starting point)
        XtX = X.T @ X
        XtY = X.T @ Y_obs
        B_ols = np.linalg.solve(XtX, XtY)
        residuals = Y_obs - X @ B_ols
        Sigma_ols = (residuals.T @ residuals) / (T_eff - n_coefs)

        # Posterior with natural conjugate prior
        # Posterior mean: weighted average of prior and OLS
        prior_precision = np.diag(1 / prior_var)
        post_precision = prior_precision + XtX
        post_var = np.linalg.inv(post_precision)
        post_mean = post_var @ (prior_precision @ prior_mean + XtY)

        # Store posterior summaries
        self.coef_mean = post_mean
        self.coef_std = np.sqrt(np.diag(post_var)).reshape(-1, k)
        self.sigma_mean = Sigma_ols
        self.X = X
        self.Y = Y_obs
        self.k = k

        # Gibbs sampling for full posterior
        self._gibbs_sample(X, Y_obs, prior_mean, prior_var)

        return self

    def _create_lags(self, Y):
        """Create lagged regressor matrix."""
        T, k = Y.shape
        Y_lag = np.zeros((T - self.n_lags, k * self.n_lags))

        for lag in range(1, self.n_lags + 1):
            start_col = (lag - 1) * k
            Y_lag[:, start_col:start_col + k] = Y[self.n_lags - lag:-lag or None]

        Y_obs = Y[self.n_lags:]
        return Y_lag, Y_obs

    def _minnesota_prior(self, k, n_coefs):
        """
        Construct Minnesota prior mean and variance.
        """
        # Prior mean: random walk (1 on own first lag, 0 elsewhere)
        prior_mean = np.zeros((n_coefs, k))

        # Set own first lag to 1
        for i in range(k):
            prior_mean[1 + i, i] = 1.0  # +1 for constant

        # Prior variance
        prior_var = np.zeros(n_coefs)
        prior_var[0] = 100.0  # Diffuse prior on constant

        for lag in range(1, self.n_lags + 1):
            for j in range(k):
                idx = 1 + (lag - 1) * k + j
                # Minnesota variance formula
                prior_var[idx] = (self.lambda1 / (lag ** self.lambda3)) ** 2

        return prior_mean, prior_var

    def _gibbs_sample(self, X, Y, prior_mean, prior_var):
        """Gibbs sampling for posterior draws."""
        T, k = Y.shape
        n_coefs = X.shape[1]

        # Storage
        B_draws = np.zeros((self.n_draws, n_coefs, k))
        Sigma_draws = np.zeros((self.n_draws, k, k))

        # Initialize
        B = self.coef_mean.copy()
        Sigma = self.sigma_mean.copy()

        for draw in range(self.n_burn + self.n_draws):
            # Draw B | Sigma, Y
            prior_precision = np.diag(1 / prior_var)
            XtX = X.T @ X
            Sigma_inv = np.linalg.inv(Sigma)

            # Posterior for vec(B)
            post_precision = np.kron(Sigma_inv, prior_precision) + np.kron(Sigma_inv, XtX)
            post_var = np.linalg.inv(post_precision)
            post_mean_vec = post_var @ (
                np.kron(Sigma_inv, prior_precision) @ prior_mean.flatten() +
                (np.kron(Sigma_inv, X.T) @ Y.flatten())
            )
            B_vec = np.random.multivariate_normal(post_mean_vec, post_var)
            B = B_vec.reshape(n_coefs, k)

            # Draw Sigma | B, Y
            residuals = Y - X @ B
            scale = residuals.T @ residuals
            df = T - n_coefs
            Sigma = invwishart.rvs(df=df, scale=scale)

            # Store after burn-in
            if draw >= self.n_burn:
                B_draws[draw - self.n_burn] = B
                Sigma_draws[draw - self.n_burn] = Sigma

        self.coef_posterior = B_draws
        self.sigma_posterior = Sigma_draws

    def forecast(self, h=12, return_draws=False):
        """
        Generate h-step ahead forecasts.

        Parameters:
        -----------
        h : int
            Forecast horizon
        return_draws : bool
            If True, return all posterior draws; else return mean and intervals

        Returns:
        --------
        If return_draws:
            forecasts : np.array, shape (n_draws, h, k)
        Else:
            dict with 'mean', 'lower', 'upper' (80% interval)
        """
        n_draws = self.coef_posterior.shape[0]
        k = self.k

        # Get last observations for conditioning
        last_obs = self.Y[-self.n_lags:][::-1].flatten()  # Reversed for lag order

        forecast_draws = np.zeros((n_draws, h, k))

        for d in range(n_draws):
            B = self.coef_posterior[d]
            Sigma = self.sigma_posterior[d]

            # Iterative forecasting
            current_lags = last_obs.copy()

            for t in range(h):
                # Construct regressor
                x_t = np.concatenate([[1], current_lags[:k * self.n_lags]])

                # Point forecast
                mean_t = x_t @ B

                # Add shock
                shock = np.random.multivariate_normal(np.zeros(k), Sigma)
                forecast_t = mean_t + shock

                forecast_draws[d, t] = forecast_t

                # Update lags
                current_lags = np.concatenate([forecast_t, current_lags[:-k]])

        if return_draws:
            return forecast_draws
        else:
            return {
                'mean': np.mean(forecast_draws, axis=0),
                'lower_10': np.percentile(forecast_draws, 10, axis=0),
                'upper_90': np.percentile(forecast_draws, 90, axis=0),
                'lower_5': np.percentile(forecast_draws, 5, axis=0),
                'upper_95': np.percentile(forecast_draws, 95, axis=0),
            }

    def forecast_point(self, h=12):
        """Point forecast using posterior mean coefficients."""
        B = self.coef_mean
        last_obs = self.Y[-self.n_lags:][::-1].flatten()

        forecasts = np.zeros((h, self.k))
        current_lags = last_obs.copy()

        for t in range(h):
            x_t = np.concatenate([[1], current_lags[:self.k * self.n_lags]])
            forecasts[t] = x_t @ B
            current_lags = np.concatenate([forecasts[t], current_lags[:-self.k]])

        return forecasts
```

---

## Hyperparameter Selection

### Grid Search over Lambda

```python
LAMBDA_GRID = {
    "lambda1": [0.05, 0.1, 0.2, 0.5],  # Overall tightness
    "lambda3": [1.0, 2.0],              # Lag decay
    "n_lags": [1, 2, 3, 4],             # VAR order
}
```

### Selection Criterion

Use **marginal likelihood** (Bayesian model comparison) or **cross-validated RMSE**.

```python
def compute_marginal_likelihood(bvar, Y):
    """
    Approximate marginal likelihood for hyperparameter selection.
    Uses Laplace approximation.
    """
    # ... implementation
    pass

def rolling_cv_rmse(Y, lambda1, lambda3, n_lags, n_folds=5):
    """
    Time-series cross-validation RMSE.
    """
    T = len(Y)
    fold_size = T // (n_folds + 1)
    rmse_list = []

    for fold in range(n_folds):
        train_end = fold_size * (fold + 2)
        test_end = min(train_end + 12, T)

        Y_train = Y[:train_end]
        Y_test = Y[train_end:test_end]

        bvar = BayesianVAR(n_lags=n_lags, lambda1=lambda1, lambda3=lambda3)
        bvar.fit(Y_train)
        forecasts = bvar.forecast_point(h=len(Y_test))

        rmse = np.sqrt(np.mean((forecasts[:, 0] - Y_test[:, 0]) ** 2))
        rmse_list.append(rmse)

    return np.mean(rmse_list)
```

---

## File Structure

```
reserves_project/scripts/academic/
├── run_bvar.py                    ← Main execution script
├── models/
│   ├── __init__.py
│   ├── bvar.py                    ← BayesianVAR class
│   ├── bvar_diagnostics.py        ← Convergence checks
│   └── bvar_forecaster.py         ← Wrapper for pipeline
```

## Output Structure

```
data/forecast_results_academic/bvar/
├── bvar_forecasts_{varset}.csv           ← Point forecasts
├── bvar_density_{varset}.csv             ← Percentile forecasts
├── bvar_posterior_summary_{varset}.json  ← Coefficient posteriors
├── bvar_hyperparams_{varset}.json        ← Selected lambdas
├── bvar_rolling_backtest_{varset}.csv    ← Rolling evaluation
└── figures/
    ├── bvar_fan_chart_{varset}.png
    ├── bvar_coefficient_posterior.png
    └── bvar_hyperparameter_search.png
```

---

## Execution Plan

### Step 1: Data Preparation
- Load each variable set from `data/forecast_prep_academic/`
- Ensure all variables are stationary (difference if needed)
- Standardize for numerical stability

### Step 2: Hyperparameter Selection
- Grid search over `LAMBDA_GRID`
- Use rolling CV on training data
- Select best (lambda1, lambda3, n_lags) per variable set

### Step 3: Model Estimation
- Fit BVAR with optimal hyperparameters
- Run Gibbs sampler (5000 draws, 1000 burn-in)
- Check convergence (trace plots, R-hat)

### Step 4: Forecasting
- Generate point forecasts (posterior mean)
- Generate density forecasts (percentile bands)
- Save all outputs

### Step 5: Rolling Backtest
- Expanding window, refit every 12 months
- Compute MAE, RMSE, MASE for each model-varset combo

---

## Validation Criteria

### Pre-Execution Checklist
- [x] Variable sets created (Spec 01 complete)
- [x] `scipy`, `numpy` available
- [x] Sufficient memory for Gibbs sampling

### Post-Execution Validation
- [x] Gibbs sampler converged (R-hat < 1.1)
- [x] Forecasts are reasonable (no explosions)
- [x] Hyperparameter selection documented
- [x] Rolling backtest complete for all variable sets

---

## Execution Log

### Pre-Execution
| Check | Status | Notes |
|-------|--------|-------|
| Spec 01 complete | :white_check_mark: | All 5 variable sets ready |
| Dependencies installed | :white_check_mark: | scipy, numpy available |
| Memory adequate | :white_check_mark: | Execution successful |

### Execution
| Step | Status | Timestamp | Notes |
|------|--------|-----------|-------|
| Load variable sets | :white_check_mark: | 2026-02-10 09:30 | All 5 sets loaded |
| Hyperparameter search (parsimonious) | :white_check_mark: | 2026-02-10 09:35 | Best: n_lags=2, lambda1=0.05 |
| Hyperparameter search (bop) | :white_check_mark: | 2026-02-10 09:42 | Best: n_lags=4, lambda1=0.2 |
| Hyperparameter search (monetary) | :white_check_mark: | 2026-02-10 09:48 | Best: n_lags=4, lambda1=0.1 |
| Hyperparameter search (pca) | :white_check_mark: | 2026-02-10 09:55 | Best: n_lags=4, lambda1=0.5 |
| Hyperparameter search (full) | :white_check_mark: | 2026-02-10 10:05 | Best: n_lags=1, lambda1=0.05 |
| Fit final models | :white_check_mark: | 2026-02-10 10:08 | 5000 draws, 1000 burn-in |
| Generate forecasts | :white_check_mark: | 2026-02-10 10:08 | 12-month horizon |
| Rolling backtest | :white_check_mark: | 2026-02-10 10:08 | 12-month refit interval |
| Convergence diagnostics | :white_check_mark: | 2026-02-10 10:08 | All R-hat < 1.01 |

### Post-Execution
| Validation | Status | Notes |
|------------|--------|-------|
| Convergence OK | :white_check_mark: | Max R-hat = 1.0014 (pca) |
| Forecasts reasonable | :white_check_mark: | No explosive forecasts |
| All varsets complete | :white_check_mark: | 5/5 completed |

---

## Results Section

### Hyperparameter Selection

Grid search over 32 combinations (lambda1 x lambda3 x n_lags) with 5-fold time-series CV.

| Variable Set | Best lambda1 | Best lambda3 | Best Lags | CV RMSE | R-hat Max |
|--------------|---------|---------|-----------|---------|-----------|
| Parsimonious | 0.05 | 1.0 | 2 | 1433.09 | 1.0003 |
| BoP | 0.2 | 1.0 | 4 | 1151.32 | 1.0011 |
| Monetary | 0.1 | 1.0 | 4 | 585.29 | 1.0007 |
| PCA | 0.5 | 1.0 | 4 | 1722.23 | 1.0014 |
| Full | 0.05 | 1.0 | 1 | 1163.36 | 1.0003 |

**Key Findings:**
- All models converged (R-hat < 1.01)
- Monetary set achieved lowest CV RMSE (585.29)
- Tight priors (lambda1 = 0.05-0.2) preferred for most sets
- Linear lag decay (lambda3 = 1.0) universally selected
- Full model required only 1 lag (maximum shrinkage for high dimensions)

### Rolling Backtest Results (h=1, 1-step ahead)

| Variable Set | Split | MAE | RMSE | MASE | 80% Coverage |
|--------------|-------|-----|------|------|--------------|
| Parsimonious | validation | 524.28 | 678.00 | 1.27 | 61.1% |
| Parsimonious | test | 426.99 | 510.04 | 1.87 | 80.8% |
| BoP | validation | 578.14 | 725.25 | 1.40 | 66.7% |
| BoP | test | 435.29 | 511.28 | 1.91 | 80.8% |
| Monetary | validation | 671.88 | 798.64 | 1.24 | 50.0% |
| Monetary | test | 621.81 | 904.01 | 2.98 | 76.9% |
| PCA | validation | 713.42 | 833.80 | 1.27 | 56.5% |
| PCA | test | 673.74 | 809.30 | 2.89 | 80.0% |
| Full | validation | 701.61 | 820.28 | 1.29 | 50.0% |
| Full | test | 559.77 | 850.43 | 2.68 | 84.6% |

### Multi-Horizon Backtest (Test Set)

| Variable Set | h=1 RMSE | h=3 RMSE | h=6 RMSE | h=12 RMSE |
|--------------|----------|----------|----------|-----------|
| Parsimonious | 510.04 | 860.61 | 1321.11 | 2483.05 |
| BoP | 511.28 | 830.93 | 1231.55 | 2323.84 |
| Monetary | 904.01 | 1435.93 | 2050.92 | 4714.31 |
| PCA | 809.30 | 1372.78 | 2020.44 | N/A |
| Full | 850.43 | 1291.97 | 1753.34 | 3669.16 |

**Best performing by horizon:**
- h=1: Parsimonious (510.04)
- h=3: BoP (830.93)
- h=6: BoP (1231.55)
- h=12: BoP (2323.84)

### Comparison with Existing Models

BVAR Minnesota prior provides:
1. **Regularization**: Shrinkage controls overfitting vs unrestricted VAR
2. **Density Forecasts**: Full posterior enables probabilistic intervals
3. **Convergence**: All models converged (R-hat < 1.01, ESS > 1000)

**Relative to standard VECM:**
- Better short-horizon (h=1-3) performance for small variable sets
- BoP specification competitive across all horizons
- Monetary specification struggles at longer horizons (h=12)

---

## References

- Litterman, R. (1986). Forecasting with Bayesian Vector Autoregressions—Five Years of Experience. Journal of Business & Economic Statistics.
- Doan, T., Litterman, R., & Sims, C. (1984). Forecasting and Conditional Projection Using Realistic Prior Distributions. Econometric Reviews.
- Bańbura, M., Giannone, D., & Reichlin, L. (2010). Large Bayesian Vector Auto Regressions. Journal of Applied Econometrics.

---

## Change Log

| Date | Version | Changes |
|------|---------|---------|
| 2026-02-10 | 1.0 | Initial specification |
| 2026-02-10 | 2.0 | Implementation complete, results populated |

