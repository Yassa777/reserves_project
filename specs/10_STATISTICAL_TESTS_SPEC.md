# Specification 10: Statistical Tests for Forecast Evaluation
## Diebold-Mariano, Model Confidence Set, and Density Evaluation

**Version:** 1.1
**Created:** 2026-02-10
**Status:** ✅ COMPLETE
**Phase:** 4 (Evaluation)
**Dependencies:** 09_DMA_DMS (all forecasts must be complete)
**Blocks:** 11_ROBUSTNESS_TABLES

---

## Objective

Implement rigorous statistical tests for forecast comparison:
1. **Diebold-Mariano (DM) Test** - Pairwise forecast accuracy comparison
2. **Model Confidence Set (MCS)** - Simultaneous comparison of all models
3. **Density Forecast Evaluation** - CRPS, Log Score, PIT
4. **Encompassing Tests** - Does one forecast contain all information in another?

---

## 1. Diebold-Mariano Test

### Theory

Tests whether two forecasts have equal predictive accuracy:
- H₀: E[d_t] = 0 where d_t = L(e₁_t) - L(e₂_t)
- H₁: E[d_t] ≠ 0

For squared error loss: d_t = e₁_t² - e₂_t²

### Implementation

```python
import numpy as np
from scipy import stats

def diebold_mariano_test(actual, forecast1, forecast2,
                         loss_fn="squared", h=1, alternative="two-sided"):
    """
    Diebold-Mariano test for equal predictive accuracy.

    Parameters:
    -----------
    actual : np.array
        Actual values
    forecast1, forecast2 : np.array
        Two competing forecasts
    loss_fn : str or callable
        "squared", "absolute", or custom function
    h : int
        Forecast horizon (for HAC standard error)
    alternative : str
        "two-sided", "less", or "greater"

    Returns:
    --------
    dict with:
        - dm_statistic: DM test statistic
        - p_value: p-value
        - mean_loss_diff: Average loss differential
        - conclusion: Interpretation string
    """
    # Compute errors
    e1 = actual - forecast1
    e2 = actual - forecast2

    # Loss differential
    if loss_fn == "squared":
        d = e1**2 - e2**2
    elif loss_fn == "absolute":
        d = np.abs(e1) - np.abs(e2)
    else:
        d = loss_fn(e1) - loss_fn(e2)

    # Remove NaN
    d = d[~np.isnan(d)]
    n = len(d)

    if n < 10:
        return {"error": "Insufficient observations"}

    # Mean and variance with HAC (Newey-West)
    mean_d = np.mean(d)

    # Autocovariances
    gamma = np.zeros(h)
    for j in range(h):
        gamma[j] = np.mean((d[j:] - mean_d) * (d[:n-j] - mean_d))

    # HAC variance
    var_d = gamma[0] + 2 * np.sum(gamma[1:])
    var_d = max(var_d, 1e-10)  # Ensure positive

    # Standard error of mean
    se = np.sqrt(var_d / n)

    # DM statistic
    dm_stat = mean_d / se

    # p-value
    if alternative == "two-sided":
        p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    elif alternative == "less":
        p_value = stats.norm.cdf(dm_stat)
    else:  # greater
        p_value = 1 - stats.norm.cdf(dm_stat)

    # Interpretation
    if p_value < 0.01:
        significance = "***"
    elif p_value < 0.05:
        significance = "**"
    elif p_value < 0.10:
        significance = "*"
    else:
        significance = ""

    if mean_d < 0:
        better = "Forecast 1"
    else:
        better = "Forecast 2"

    return {
        "dm_statistic": dm_stat,
        "p_value": p_value,
        "mean_loss_diff": mean_d,
        "se": se,
        "n_obs": n,
        "better_forecast": better if p_value < 0.10 else "No significant difference",
        "significance": significance,
    }


def dm_test_matrix(actual, forecasts_dict, loss_fn="squared", h=1):
    """
    Compute DM test for all pairs of forecasts.

    Returns:
    --------
    dm_stats : pd.DataFrame
        Matrix of DM statistics (row vs column)
    p_values : pd.DataFrame
        Matrix of p-values
    """
    import pandas as pd

    model_names = list(forecasts_dict.keys())
    n_models = len(model_names)

    dm_stats = np.zeros((n_models, n_models))
    p_values = np.zeros((n_models, n_models))

    for i, m1 in enumerate(model_names):
        for j, m2 in enumerate(model_names):
            if i == j:
                dm_stats[i, j] = np.nan
                p_values[i, j] = np.nan
            else:
                result = diebold_mariano_test(
                    actual, forecasts_dict[m1], forecasts_dict[m2],
                    loss_fn=loss_fn, h=h
                )
                dm_stats[i, j] = result.get("dm_statistic", np.nan)
                p_values[i, j] = result.get("p_value", np.nan)

    return (
        pd.DataFrame(dm_stats, index=model_names, columns=model_names),
        pd.DataFrame(p_values, index=model_names, columns=model_names)
    )
```

### Harvey-Leybourne-Newbold Correction

```python
def dm_test_hln(actual, forecast1, forecast2, h=1):
    """
    DM test with HLN small-sample correction.

    Uses t-distribution instead of normal.
    """
    result = diebold_mariano_test(actual, forecast1, forecast2, h=h)

    if "error" in result:
        return result

    n = result["n_obs"]

    # HLN correction factor
    correction = np.sqrt((n + 1 - 2*h + h*(h-1)/n) / n)
    dm_corrected = result["dm_statistic"] * correction

    # Use t-distribution
    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_corrected), df=n-1))

    result["dm_statistic_hln"] = dm_corrected
    result["p_value_hln"] = p_value

    return result
```

---

## 2. Model Confidence Set (MCS)

### Theory

Hansen, Lunde & Nason (2011):
- Sequentially eliminate inferior models
- Returns set of models not significantly worse than best
- Controls familywise error rate

### Implementation

```python
def model_confidence_set(actual, forecasts_dict, loss_fn="squared",
                         alpha=0.10, bootstrap_reps=1000):
    """
    Model Confidence Set procedure.

    Parameters:
    -----------
    actual : np.array
        Actual values
    forecasts_dict : dict
        {model_name: forecast_array}
    loss_fn : str
        Loss function
    alpha : float
        Significance level
    bootstrap_reps : int
        Number of bootstrap replications

    Returns:
    --------
    dict with:
        - mcs: list of models in confidence set
        - eliminated: list of eliminated models (in order)
        - p_values: p-value for each elimination
    """
    from scipy.stats import bootstrap

    # Compute losses
    if loss_fn == "squared":
        losses = {m: (actual - f)**2 for m, f in forecasts_dict.items()}
    else:
        losses = {m: np.abs(actual - f) for m, f in forecasts_dict.items()}

    remaining = list(forecasts_dict.keys())
    eliminated = []
    p_values = []

    while len(remaining) > 1:
        # Compute pairwise loss differentials
        n = len(actual)
        n_models = len(remaining)

        loss_matrix = np.column_stack([losses[m] for m in remaining])
        mean_losses = np.nanmean(loss_matrix, axis=0)

        # Range statistic (T_R)
        t_range = np.max(mean_losses) - np.min(mean_losses)

        # Bootstrap distribution
        def bootstrap_range(data, axis):
            return np.max(np.mean(data, axis=axis)) - np.min(np.mean(data, axis=axis))

        boot_ranges = np.zeros(bootstrap_reps)
        for b in range(bootstrap_reps):
            idx = np.random.choice(n, n, replace=True)
            boot_losses = loss_matrix[idx]
            # Center losses
            centered = boot_losses - np.mean(boot_losses, axis=0)
            boot_ranges[b] = np.max(np.mean(centered, axis=0)) - np.min(np.mean(centered, axis=0))

        # p-value
        p_value = np.mean(boot_ranges >= t_range)
        p_values.append(p_value)

        if p_value >= alpha:
            # Cannot reject: remaining models form MCS
            break
        else:
            # Eliminate worst model
            worst_idx = np.argmax(mean_losses)
            worst_model = remaining[worst_idx]
            eliminated.append(worst_model)
            remaining.remove(worst_model)

    return {
        "mcs": remaining,
        "eliminated": eliminated,
        "p_values": p_values,
        "alpha": alpha,
    }


def mcs_summary_table(mcs_result, forecasts_dict, actual):
    """
    Create summary table of MCS results.
    """
    import pandas as pd

    all_models = list(forecasts_dict.keys())

    summary = []
    for m in all_models:
        fc = forecasts_dict[m]
        rmse = np.sqrt(np.nanmean((actual - fc)**2))
        mae = np.nanmean(np.abs(actual - fc))
        in_mcs = m in mcs_result["mcs"]

        if m in mcs_result["eliminated"]:
            elim_order = mcs_result["eliminated"].index(m) + 1
        else:
            elim_order = None

        summary.append({
            "model": m,
            "rmse": rmse,
            "mae": mae,
            "in_mcs": in_mcs,
            "elimination_order": elim_order,
        })

    return pd.DataFrame(summary).sort_values("rmse")
```

---

## 3. Density Forecast Evaluation

### Continuous Ranked Probability Score (CRPS)

```python
def crps_empirical(actual, forecast_samples):
    """
    CRPS from ensemble/MCMC samples.

    Parameters:
    -----------
    actual : float
        Realized value
    forecast_samples : np.array
        Samples from predictive distribution

    Returns:
    --------
    crps : float
    """
    n = len(forecast_samples)
    sorted_samples = np.sort(forecast_samples)

    # CRPS formula
    crps = 0.0
    for i, x in enumerate(sorted_samples):
        weight = (2 * i + 1 - n) / n
        crps += weight * (x - actual)

    crps = 2 * np.mean(np.abs(forecast_samples - actual)) - np.mean(
        np.abs(forecast_samples[:, None] - forecast_samples[None, :])
    )

    return crps


def crps_normal(actual, mean, std):
    """
    CRPS for Gaussian predictive distribution.

    Closed-form solution.
    """
    z = (actual - mean) / std
    crps = std * (
        z * (2 * stats.norm.cdf(z) - 1) +
        2 * stats.norm.pdf(z) -
        1 / np.sqrt(np.pi)
    )
    return crps


def compute_crps_series(actuals, forecast_means, forecast_stds):
    """
    Compute CRPS for a series of forecasts.
    """
    n = len(actuals)
    crps_values = np.zeros(n)

    for t in range(n):
        if np.isnan(forecast_means[t]) or np.isnan(forecast_stds[t]):
            crps_values[t] = np.nan
        else:
            crps_values[t] = crps_normal(
                actuals[t], forecast_means[t], forecast_stds[t]
            )

    return crps_values
```

### Log Score

```python
def log_score_normal(actual, mean, std):
    """
    Log score (negative log predictive density).

    Lower is better.
    """
    return -stats.norm.logpdf(actual, loc=mean, scale=std)


def compute_log_score_series(actuals, forecast_means, forecast_stds):
    """
    Compute log scores for a series.
    """
    n = len(actuals)
    log_scores = np.zeros(n)

    for t in range(n):
        if np.isnan(forecast_means[t]) or np.isnan(forecast_stds[t]):
            log_scores[t] = np.nan
        else:
            log_scores[t] = log_score_normal(
                actuals[t], forecast_means[t], forecast_stds[t]
            )

    return log_scores
```

### Probability Integral Transform (PIT)

```python
def compute_pit(actuals, forecast_means, forecast_stds):
    """
    Probability Integral Transform.

    If forecasts are calibrated, PIT should be uniform.
    """
    pit = stats.norm.cdf(actuals, loc=forecast_means, scale=forecast_stds)
    return pit


def pit_histogram_test(pit_values, n_bins=10):
    """
    Chi-squared test for PIT uniformity.
    """
    # Remove NaN
    pit_clean = pit_values[~np.isnan(pit_values)]
    n = len(pit_clean)

    # Observed frequencies
    observed, _ = np.histogram(pit_clean, bins=n_bins, range=(0, 1))

    # Expected (uniform)
    expected = n / n_bins

    # Chi-squared statistic
    chi2 = np.sum((observed - expected)**2 / expected)
    p_value = 1 - stats.chi2.cdf(chi2, df=n_bins - 1)

    return {
        "chi2_statistic": chi2,
        "p_value": p_value,
        "reject_uniformity": p_value < 0.05,
        "observed_frequencies": observed,
    }
```

---

## 4. Forecast Encompassing Tests

```python
def forecast_encompassing_test(actual, forecast1, forecast2):
    """
    Test whether forecast1 encompasses forecast2.

    H₀: Optimal combination weight on forecast2 = 0
    (forecast1 contains all information)

    Regression: y = α + λ₁*f₁ + λ₂*f₂ + ε
    Test: λ₂ = 0
    """
    import statsmodels.api as sm

    # Prepare data
    valid = ~(np.isnan(actual) | np.isnan(forecast1) | np.isnan(forecast2))
    y = actual[valid]
    X = np.column_stack([np.ones(valid.sum()), forecast1[valid], forecast2[valid]])

    # OLS regression
    model = sm.OLS(y, X)
    results = model.fit()

    # Test λ₂ = 0
    t_stat = results.tvalues[2]
    p_value = results.pvalues[2]

    return {
        "lambda1": results.params[1],
        "lambda2": results.params[2],
        "t_statistic_lambda2": t_stat,
        "p_value": p_value,
        "forecast1_encompasses": p_value > 0.05,
        "r_squared": results.rsquared,
    }
```

---

## File Structure

```
reserves_project/scripts/academic/
├── run_statistical_tests.py       ← Main execution
├── tests/
│   ├── diebold_mariano.py         ← DM test
│   ├── model_confidence_set.py    ← MCS
│   ├── density_evaluation.py      ← CRPS, log score, PIT
│   └── encompassing.py            ← Encompassing tests
```

## Output Structure

```
data/statistical_tests/
├── dm_test_matrix.csv             ← Pairwise DM statistics
├── dm_pvalues_matrix.csv          ← Pairwise p-values
├── mcs_results.json               ← Model confidence set
├── crps_summary.csv               ← CRPS by model
├── log_scores.csv                 ← Log scores by model
├── pit_tests.json                 ← PIT uniformity tests
├── encompassing_tests.csv         ← Encompassing results
└── figures/
    ├── dm_heatmap.png
    ├── mcs_elimination.png
    ├── crps_comparison.png
    └── pit_histograms.png
```

---

## Academic Presentation

### DM Test Table (Example)

```latex
\begin{table}[htbp]
\caption{Diebold-Mariano Test Results (Squared Error Loss)}
\begin{tabular}{lcccccc}
\toprule
& ARIMA & VECM & MS-VAR & MS-VECM & BVAR & DMA \\
\midrule
ARIMA & - & -0.45 & 1.23 & 1.45* & 0.89 & 1.67** \\
VECM & 0.45 & - & 1.56* & 1.78** & 1.12 & 2.01** \\
MS-VAR & -1.23 & -1.56* & - & 0.34 & -0.23 & 0.89 \\
\bottomrule
\end{tabular}
\end{table}
```

### MCS Table (Example)

```latex
\begin{table}[htbp]
\caption{Model Confidence Set ($\alpha = 0.10$)}
\begin{tabular}{lccc}
\toprule
Model & RMSE & In MCS & Elimination Order \\
\midrule
DMA & 245.3 & \checkmark & - \\
MS-VECM & 254.6 & \checkmark & - \\
BVAR & 267.8 & \checkmark & - \\
MS-VAR & 281.2 & & 4 \\
ARIMA & 312.4 & & 3 \\
VECM & 456.7 & & 1 \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Execution Log

### Pre-Execution
| Check | Status | Notes |
|-------|--------|-------|
| All forecasts available | ✅ | From Specs 03-09 |
| Forecast variances available | ✅ | BVAR density forecasts |
| scipy installed | ✅ | |

### Execution
| Step | Status | Timestamp | Notes |
|------|--------|-----------|-------|
| Compute DM test matrix | ✅ | 2026-02-10 16:25 | 11x11 pairwise matrix |
| Run MCS | ✅ | 2026-02-10 16:25 | 7 models in 90% MCS |
| Compute CRPS | ✅ | 2026-02-10 16:25 | 5 BVAR variants evaluated |
| Compute log scores | ✅ | 2026-02-10 16:25 | 5 BVAR variants |
| PIT tests | ✅ | 2026-02-10 16:25 | Chi-squared tests complete |
| Encompassing tests | ✅ | 2026-02-10 16:25 | Full pairwise matrix |
| Generate tables/plots | ✅ | 2026-02-10 16:25 | 3 figures generated |

### Post-Execution
| Validation | Status | Notes |
|------------|--------|-------|
| DM tests consistent | ✅ | HLN correction applied |
| MCS non-empty | ✅ | 7 models in set |
| PITs reasonable | ⚠️ | Non-uniform for all BVAR variants |

---

## Results Section

### DM Test Results

Key findings from Diebold-Mariano tests (HLN-corrected, squared error loss):

**vs Naive Benchmark:**
- VECM, MS-VAR, MS-VECM: Significantly **worse** than Naive (p < 0.001)
- ARIMA: Not significantly different from Naive
- BVAR, EqualWeight, MSE-Weight, GR-Convex: Significantly **better** than Naive
- DMA, DMS: Not significantly different from Naive

**vs DMA Benchmark:**
- ARIMA, MS-VAR, MS-VECM: Significantly worse than DMA
- BVAR, MSE-Weight: Significantly worse than DMA (i.e., DMA is better)
- Naive, VECM, DMS, EqualWeight, GR-Convex: Not significantly different

### MCS Results

**Models in 90% Model Confidence Set:**
1. MSE-Weight (RMSE = 778)
2. BVAR (RMSE = 1,491)
3. EqualWeight (RMSE = 1,774)
4. GR-Convex (RMSE = 1,774)
5. DMS (RMSE = 1,990)
6. DMA (RMSE = 2,114)
7. Naive (RMSE = 2,131)

**Eliminated models (in order):**
1. MS-VECM (p = 0.001)
2. MS-VAR (p = 0.001)
3. VECM (p = 0.029)
4. ARIMA (p = 0.031)

### Density Forecast Ranking

| Model | Mean CRPS | Mean Log Score | Coverage 90% | PIT Uniform? |
|-------|-----------|----------------|--------------|--------------|
| BVAR_pca | 857.38 | 8.83 | 83.33% | No |
| BVAR_parsimonious | 934.33 | 9.09 | 83.33% | No |
| BVAR_bop | 1067.13 | 9.27 | 83.33% | No |
| BVAR_full | 1163.66 | 9.17 | 83.33% | No |
| BVAR_monetary | 1251.48 | 9.77 | 25.00% | No |

Note: PIT non-uniformity suggests density forecasts may be miscalibrated.

### Key Academic Findings

1. **Best point forecast model**: MSE-Weight combination (RMSE = 777.70)
2. **Simple forecasts competitive**: Naive benchmark in 90% MCS
3. **MS models underperform**: Both MS-VAR and MS-VECM eliminated first from MCS
4. **DMA provides value**: DMA is in MCS and not significantly worse than best models
5. **Combination helps**: EqualWeight and MSE-Weight combinations in MCS

---

## References

- Diebold, F.X. & Mariano, R.S. (1995). Comparing Predictive Accuracy. Journal of Business & Economic Statistics.
- Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the Equality of Prediction Mean Squared Errors. International Journal of Forecasting.
- Hansen, P.R., Lunde, A., & Nason, J.M. (2011). The Model Confidence Set. Econometrica.
- Gneiting, T. & Raftery, A.E. (2007). Strictly Proper Scoring Rules, Prediction, and Estimation. JASA.

---

## Change Log

| Date | Version | Changes |
|------|---------|---------|
| 2026-02-10 | 1.0 | Initial specification |
| 2026-02-10 | 1.1 | Implementation complete - all tests executed |

## Implementation Files

```
reserves_project/scripts/academic/tests/
├── __init__.py                    # Module exports
├── diebold_mariano.py             # DM test with HLN correction
├── model_confidence_set.py        # MCS procedure (Hansen et al. 2011)
├── density_evaluation.py          # CRPS, log score, PIT tests
└── encompassing.py                # Forecast encompassing tests

reserves_project/scripts/academic/
└── run_statistical_tests.py       # Main execution script

data/statistical_tests/
├── dm_test_matrix.csv             # Pairwise DM statistics
├── dm_pvalues_matrix.csv          # Pairwise p-values
├── dm_formatted_table.csv         # Publication-ready table
├── mcs_results.json               # MCS membership and elimination order
├── mcs_summary.csv                # Summary table with RMSE
├── mcs_pvalues.csv                # MCS p-values for all models
├── encompassing_tests.csv         # Pairwise encompassing results
├── encompassing_matrix.csv        # Formatted encompassing matrix
├── optimal_weights.json           # Optimal combination weights
├── density_evaluation.csv         # CRPS/Log Score summary
└── figures/
    ├── dm_heatmap.png             # DM statistics heatmap
    ├── mcs_results.png            # MCS visualization
    └── forecast_comparison.png    # Time series comparison
```

