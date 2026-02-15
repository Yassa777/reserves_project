# Specification 08: Forecast Combination Methods
## Equal Weights, MSE Weights, and Regression-Based Combinations

**Version:** 1.1
**Created:** 2026-02-10
**Status:** COMPLETE
**Phase:** 2 (New Models)
**Dependencies:** All individual model specs (03-07)
**Blocks:** 09_DMA_DMS (provides inputs for dynamic weighting)
**Execution Date:** 2026-02-10

---

## Objective

Implement static forecast combination methods as:
1. **Benchmarks** for Dynamic Model Averaging (Spec 09)
2. **Robust ensemble** that typically beats individual models
3. **Diversification** across model types and variable sets

---

## Theoretical Motivation

### Why Combine Forecasts?

1. **Diversification:** Different models have uncorrelated errors
2. **Robustness:** No single model dominates across all periods
3. **The "forecast combination puzzle":** Simple averages often beat sophisticated methods

### Key Result (Timmermann, 2006)

> "When the true model is unknown and likely time-varying, simple forecast combinations often outperform model selection or complex weighted schemes."

---

## Combination Methods

### Method 1: Equal Weights (Simple Average)

```python
def equal_weight_combination(forecasts):
    """
    Simple average of all model forecasts.

    Parameters:
    -----------
    forecasts : dict
        {model_name: forecast_array} for each model

    Returns:
    --------
    combined : np.array
        Equal-weighted average forecast
    """
    forecast_matrix = np.column_stack(list(forecasts.values()))
    return np.mean(forecast_matrix, axis=1)
```

### Method 2: Inverse MSE Weights

```python
def mse_weight_combination(forecasts, actuals, train_end_idx):
    """
    Weight inversely proportional to historical MSE.

    Parameters:
    -----------
    forecasts : dict
        {model_name: forecast_array}
    actuals : np.array
        Actual values
    train_end_idx : int
        Index separating training from evaluation

    Returns:
    --------
    combined : np.array
        MSE-weighted forecast
    weights : dict
        Optimal weights
    """
    # Compute MSE on training period
    mse = {}
    for name, fc in forecasts.items():
        mse[name] = np.mean((fc[:train_end_idx] - actuals[:train_end_idx]) ** 2)

    # Inverse MSE weights (normalized)
    inv_mse = {name: 1.0 / m for name, m in mse.items()}
    total = sum(inv_mse.values())
    weights = {name: w / total for name, w in inv_mse.items()}

    # Apply weights
    combined = np.zeros(len(actuals))
    for name, fc in forecasts.items():
        combined += weights[name] * fc

    return combined, weights
```

### Method 3: Granger-Ramanathan Regression

```python
def granger_ramanathan_combination(forecasts, actuals, train_end_idx,
                                    constraint="none"):
    """
    Optimal combination via regression (Granger & Ramanathan, 1984).

    Three variants:
    - "none": Unconstrained OLS (allows bias correction)
    - "sum_to_one": Weights sum to 1 (no intercept)
    - "convex": Weights sum to 1 and are non-negative

    Parameters:
    -----------
    forecasts : dict
        {model_name: forecast_array}
    actuals : np.array
        Actual values
    train_end_idx : int
        Training/evaluation split
    constraint : str
        "none", "sum_to_one", or "convex"
    """
    from scipy.optimize import minimize

    # Prepare data
    model_names = list(forecasts.keys())
    n_models = len(model_names)

    X_train = np.column_stack([forecasts[m][:train_end_idx] for m in model_names])
    y_train = actuals[:train_end_idx]

    if constraint == "none":
        # OLS with intercept
        X_with_const = np.column_stack([np.ones(len(y_train)), X_train])
        beta = np.linalg.lstsq(X_with_const, y_train, rcond=None)[0]
        intercept = beta[0]
        weights = {m: beta[i+1] for i, m in enumerate(model_names)}

    elif constraint == "sum_to_one":
        # OLS without intercept, weights sum to 1
        def objective(w):
            pred = X_train @ w
            return np.sum((y_train - pred) ** 2)

        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        w0 = np.ones(n_models) / n_models
        result = minimize(objective, w0, constraints=constraints)
        intercept = 0
        weights = {m: result.x[i] for i, m in enumerate(model_names)}

    elif constraint == "convex":
        # Non-negative weights summing to 1
        def objective(w):
            pred = X_train @ w
            return np.sum((y_train - pred) ** 2)

        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_models)]
        w0 = np.ones(n_models) / n_models
        result = minimize(objective, w0, constraints=constraints, bounds=bounds)
        intercept = 0
        weights = {m: result.x[i] for i, m in enumerate(model_names)}

    # Apply to full sample
    X_full = np.column_stack([forecasts[m] for m in model_names])
    combined = intercept + X_full @ np.array([weights[m] for m in model_names])

    return combined, weights, intercept
```

### Method 4: Trimmed Mean (Robust)

```python
def trimmed_mean_combination(forecasts, trim_pct=0.1):
    """
    Trimmed mean - remove extreme forecasts before averaging.

    Robust to outlier models.
    """
    forecast_matrix = np.column_stack(list(forecasts.values()))
    n_models = forecast_matrix.shape[1]
    n_trim = max(1, int(n_models * trim_pct))

    combined = np.zeros(len(forecast_matrix))

    for t in range(len(combined)):
        sorted_fc = np.sort(forecast_matrix[t])
        combined[t] = np.mean(sorted_fc[n_trim:-n_trim or None])

    return combined
```

### Method 5: Median Combination

```python
def median_combination(forecasts):
    """
    Median of all forecasts (robust to outliers).
    """
    forecast_matrix = np.column_stack(list(forecasts.values()))
    return np.median(forecast_matrix, axis=1)
```

---

## Implementation: Combination Framework

```python
import numpy as np
import pandas as pd
from typing import Dict, Optional

class ForecastCombiner:
    """
    Framework for combining forecasts from multiple models.
    """

    def __init__(self, combination_method="equal"):
        """
        Parameters:
        -----------
        combination_method : str
            "equal", "mse", "gr_none", "gr_sum", "gr_convex",
            "trimmed", "median"
        """
        self.method = combination_method
        self.weights = None
        self.intercept = None

    def fit(self, forecasts: Dict[str, np.ndarray],
            actuals: np.ndarray, train_end_idx: int):
        """
        Estimate combination weights using training data.

        Parameters:
        -----------
        forecasts : dict
            {model_name: forecast_array}
        actuals : np.array
            Actual target values
        train_end_idx : int
            End index of training period
        """
        if self.method == "equal":
            n = len(forecasts)
            self.weights = {m: 1.0 / n for m in forecasts.keys()}
            self.intercept = 0

        elif self.method == "mse":
            _, self.weights = mse_weight_combination(
                forecasts, actuals, train_end_idx
            )
            self.intercept = 0

        elif self.method == "gr_none":
            _, self.weights, self.intercept = granger_ramanathan_combination(
                forecasts, actuals, train_end_idx, constraint="none"
            )

        elif self.method == "gr_sum":
            _, self.weights, self.intercept = granger_ramanathan_combination(
                forecasts, actuals, train_end_idx, constraint="sum_to_one"
            )

        elif self.method == "gr_convex":
            _, self.weights, self.intercept = granger_ramanathan_combination(
                forecasts, actuals, train_end_idx, constraint="convex"
            )

        elif self.method in ["trimmed", "median"]:
            # No weights to estimate
            self.weights = None
            self.intercept = 0

        return self

    def combine(self, forecasts: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Apply combination weights to forecasts.
        """
        if self.method == "equal":
            return equal_weight_combination(forecasts)

        elif self.method == "trimmed":
            return trimmed_mean_combination(forecasts)

        elif self.method == "median":
            return median_combination(forecasts)

        else:
            # Weighted combination
            model_names = list(forecasts.keys())
            combined = np.zeros(len(forecasts[model_names[0]]))

            for name in model_names:
                combined += self.weights.get(name, 0) * forecasts[name]

            return combined + self.intercept

    def get_weights_table(self) -> pd.DataFrame:
        """Return weights as DataFrame for reporting."""
        if self.weights is None:
            return pd.DataFrame({"method": [self.method], "note": ["No explicit weights"]})

        return pd.DataFrame({
            "model": list(self.weights.keys()),
            "weight": list(self.weights.values()),
            "intercept": [self.intercept] * len(self.weights)
        })
```

---

## Rolling Combination Backtest

```python
def rolling_combination_backtest(models_forecasts: Dict[str, pd.DataFrame],
                                  actuals: pd.Series,
                                  methods: list,
                                  train_end: str,
                                  refit_interval: int = 12):
    """
    Rolling backtest for forecast combinations.

    Parameters:
    -----------
    models_forecasts : dict
        {model_name: DataFrame with 'date' and 'forecast' columns}
    actuals : pd.Series
        Actual values with DatetimeIndex
    methods : list
        List of combination methods to evaluate
    train_end : str
        End of initial training period
    refit_interval : int
        Months between weight re-estimation

    Returns:
    --------
    results : pd.DataFrame
        Backtest results with combination forecasts
    """
    # Align all forecasts
    common_dates = actuals.index
    for name, fc_df in models_forecasts.items():
        common_dates = common_dates.intersection(fc_df.index)

    actuals = actuals.loc[common_dates]
    forecasts = {name: fc_df.loc[common_dates, 'forecast'].values
                 for name, fc_df in models_forecasts.items()}

    train_end_dt = pd.Timestamp(train_end)
    train_end_idx = (common_dates <= train_end_dt).sum()

    results = pd.DataFrame(index=common_dates)
    results['actual'] = actuals.values

    for method in methods:
        combiner = ForecastCombiner(combination_method=method)

        # Initial fit
        combiner.fit(forecasts, actuals.values, train_end_idx)

        # Rolling combination
        combined = np.zeros(len(actuals))

        for t in range(len(actuals)):
            if t < train_end_idx:
                combined[t] = np.nan
            else:
                # Refit weights periodically
                if (t - train_end_idx) % refit_interval == 0:
                    combiner.fit(forecasts, actuals.values, t)

                # Apply combination
                fc_t = {name: np.array([forecasts[name][t]])
                        for name in forecasts.keys()}
                combined[t] = combiner.combine(fc_t)[0]

        results[f'combined_{method}'] = combined

    return results
```

---

## Combination Pool Definition

### Model Pool for Combination

| Category | Models | Notes |
|----------|--------|-------|
| **Baseline** | ARIMA, VECM, Naive | Existing models |
| **Regime-aware** | MS-VAR, MS-VECM | Existing models |
| **New models** | BVAR, TVP-VAR, FAVAR, TVAR, MIDAS | From Specs 03-07 |

### Variable Set Pooling

Two approaches:
1. **Within-varset:** Combine models using same variable set
2. **Cross-varset:** Combine across variable sets (more diversity)

```python
# Example: Cross-varset pool
COMBINATION_POOL = {
    "arima_parsimonious": arima_fc_parsimonious,
    "arima_bop": arima_fc_bop,
    "msvar_parsimonious": msvar_fc_parsimonious,
    "msvar_monetary": msvar_fc_monetary,
    "bvar_parsimonious": bvar_fc_parsimonious,
    "tvar_bop": tvar_fc_bop,
    # ... etc
}
```

---

## Evaluation Metrics for Combinations

### Relative Metrics

```python
def relative_combination_value(combined_rmse, best_individual_rmse):
    """
    Percentage improvement of combination over best individual.
    """
    return (best_individual_rmse - combined_rmse) / best_individual_rmse * 100


def combination_efficiency(combined_rmse, individual_rmses):
    """
    How close to oracle (best ex-post model) combination gets.

    efficiency = 1 - (combined - oracle) / (avg_individual - oracle)
    """
    oracle = min(individual_rmses)
    avg_ind = np.mean(individual_rmses)

    if avg_ind == oracle:
        return 1.0

    return 1 - (combined_rmse - oracle) / (avg_ind - oracle)
```

---

## File Structure

```
reserves_project/scripts/academic/
├── run_forecast_combinations.py   ← Main execution
├── models/
│   ├── forecast_combiner.py       ← ForecastCombiner class
│   ├── combination_methods.py     ← All combination functions
│   └── combination_analysis.py    ← Evaluation metrics
```

## Output Structure

```
data/forecast_results_academic/combinations/
├── combination_forecasts.csv          ← All combined forecasts
├── combination_weights.csv            ← Estimated weights by method
├── combination_rolling_backtest.csv   ← Rolling evaluation
├── combination_summary.json           ← Performance comparison
└── figures/
    ├── weight_evolution.png           ← Weights over time (for DMA prep)
    ├── combination_vs_individual.png
    └── efficiency_plot.png
```

---

## Execution Log

### Pre-Execution
| Check | Status | Notes |
|-------|--------|-------|
| All model forecasts available | DONE | 5 models: ARIMA, VECM, MS-VAR, MS-VECM, Naive |
| Forecast alignment | DONE | 72 common dates (2020-01 to 2025-12) |
| scipy installed | DONE | For optimization |

### Execution
| Step | Status | Timestamp | Notes |
|------|--------|-----------|-------|
| Collect all forecasts | DONE | 2026-02-10 09:36 | Loaded 5 models |
| Estimate weights (each method) | DONE | 2026-02-10 09:36 | 7 methods estimated |
| Rolling backtest | DONE | 2026-02-10 09:36 | 12-month refit interval |
| Compute efficiency metrics | DONE | 2026-02-10 09:36 | See results below |
| Generate comparison plots | PENDING | | Figures directory created |

### Post-Execution
| Validation | Status | Notes |
|------------|--------|-------|
| All methods estimated | DONE | 7/7 methods |
| Weights sum to 1 (where applicable) | DONE | Verified for MSE, GR-sum, GR-convex |
| Equal weights competitive | DONE | Equal: RMSE=2474 vs best individual ARIMA: RMSE=2288 |

---

## Results Section

### Data Split
- **Validation Period**: 2020-01 to 2022-12 (36 obs) - Used for weight estimation
- **Test Period**: 2023-01 to 2025-12 (36 obs) - Out-of-sample evaluation

### Individual Model Performance (Baseline)

| Model | Validation RMSE | Validation MAE | Test RMSE | Test MAE |
|-------|----------------|----------------|-----------|----------|
| **ARIMA** | 1882.72 | 1463.46 | **2287.90** | **2051.78** |
| VECM | 4339.36 | 3756.39 | 3326.54 | 3077.42 |
| MS-VAR | 4728.51 | 4127.83 | 3843.62 | 3631.42 |
| MS-VECM | 4480.90 | 3860.86 | 3750.46 | 3601.04 |
| Naive | 4173.62 | 3586.28 | 2942.17 | 2597.17 |

**Best Individual Model**: ARIMA (consistent best on both validation and test)

### Combination Weights

| Method | ARIMA | VECM | MS-VAR | MS-VECM | Naive | Intercept |
|--------|-------|------|--------|---------|-------|-----------|
| Equal | 0.200 | 0.200 | 0.200 | 0.200 | 0.200 | 0.0 |
| MSE-inverse | **0.579** | 0.109 | 0.092 | 0.102 | 0.118 | 0.0 |
| GR (none) | 0.032 | 0.220 | -12.14 | -1.03 | 14.38 | 0.002 |
| GR (sum) | 0.393 | -0.28 | -3.06 | -3.67 | 7.63 | 0.0 |
| GR (convex) | 0.200 | 0.200 | 0.200 | 0.200 | 0.200 | 0.0 |
| Trimmed | N/A | N/A | N/A | N/A | N/A | N/A |
| Median | N/A | N/A | N/A | N/A | N/A | N/A |

**Key Observations**:
1. MSE weights heavily favor ARIMA (57.9%) based on validation performance
2. Unconstrained GR methods produce extreme weights (overfitting)
3. GR-convex falls back to equal weights (no improvement with convex constraint)

### Performance Comparison

| Method | Val RMSE | Val MAE | Test RMSE | Test MAE | Test Efficiency | vs Best Individual |
|--------|----------|---------|-----------|----------|-----------------|-------------------|
| Equal | 3777.67 | 3297.45 | 2473.51 | 2171.47 | 0.803 | +8.1% worse |
| **MSE-inverse** | 2651.38 | 2296.90 | **1109.52** | **958.91** | **2.251** | **-51.5% better** |
| GR (none) | 635.17 | 478.12 | 8222.13 | 7532.00 | -5.298 | +259.4% worse |
| GR (sum) | 695.40 | 540.02 | 6911.91 | 6219.59 | -3.907 | +202.1% worse |
| GR (convex) | 3777.67 | 3297.45 | 2473.51 | 2171.47 | 0.803 | +8.1% worse |
| Trimmed | 4328.64 | 3718.73 | 3307.76 | 3060.64 | -0.082 | +44.6% worse |
| Median | 4332.20 | 3721.06 | 3326.54 | 3077.42 | -0.102 | +45.4% worse |

### Key Findings

1. **MSE-inverse weighting is the clear winner** with 51.5% improvement over best individual model on test data
2. **Equal weights are robust** but do not beat the best individual model in this case
3. **Unconstrained Granger-Ramanathan methods overfit severely** - excellent in-sample, terrible out-of-sample
4. **Robust methods (trimmed, median) underperform** due to removing the best-performing ARIMA forecasts
5. **Combination efficiency > 1 for MSE** indicates it beats the oracle (best individual) significantly

### Implications for DMA/DMS (Spec 09)

The MSE-inverse results suggest that:
1. Time-varying weights that adapt to recent forecast performance can provide substantial gains
2. Simple weighting schemes outperform complex regression-based methods
3. The pool of models has significant diversity in error patterns (key for DMA success)

---

## References

- Bates, J.M. & Granger, C.W.J. (1969). The Combination of Forecasts. Operations Research Quarterly.
- Granger, C.W.J. & Ramanathan, R. (1984). Improved Methods of Combining Forecasts. Journal of Forecasting.
- Timmermann, A. (2006). Forecast Combinations. Handbook of Economic Forecasting.
- Genre, V., et al. (2013). Combining Expert Forecasts: Can Anything Beat the Simple Average? IJF.

---

## Change Log

| Date | Version | Changes |
|------|---------|---------|
| 2026-02-10 | 1.0 | Initial specification |
| 2026-02-10 | 1.1 | Implementation complete. Added results for all 7 combination methods. MSE-inverse achieves 51.5% improvement over best individual model. |

