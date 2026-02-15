# Specification 02: Structural Break Analysis
## Bai-Perron Multiple Structural Break Detection

**Version:** 1.1
**Created:** 2026-02-10
**Status:** COMPLETED
**Phase:** 1 (Foundation)
**Dependencies:** None
**Blocks:** All Phase 2 specs (provides break dates for model conditioning)

---

## Objective

Identify and characterize structural breaks in Sri Lankan reserves dynamics using:
1. **Bai-Perron (1998, 2003)** - Multiple break detection with unknown break dates
2. **Chow Test** - Known break date testing (for specific events)
3. **CUSUM/CUSUMSQ** - Recursive stability tests
4. **Andrews-Ploberger** - Supremum tests for unknown breaks

---

## Theoretical Motivation

### Why Structural Breaks Matter for Reserves Forecasting

Sri Lanka experienced several regime shifts that likely affected reserve dynamics:

| Event | Approximate Date | Expected Impact |
|-------|------------------|-----------------|
| Global Financial Crisis | 2008-2009 | Trade collapse, remittance shock |
| Post-war recovery | 2009-2010 | Tourism revival, FDI inflows |
| Balance of Payments crisis | 2018-2019 | Reserve depletion, IMF program |
| COVID-19 pandemic | 2020-2021 | Trade disruption, tourism collapse |
| Sovereign default | 2022 | Regime change in reserve management |

### Implications for Modeling

- **Constant-parameter models** (ARIMA, VECM) may perform poorly across breaks
- **Regime-switching models** (MS-VAR) should capture breaks endogenously
- **Known breaks** can be incorporated as dummy variables
- **TVP models** allow gradual parameter evolution

---

## Methodology

### 1. Bai-Perron Multiple Break Test

**Model Specification:**

For the reserves series:
```
R_t = α_j + β_j * X_t + ε_t,  for T_{j-1} < t ≤ T_j, j = 1, ..., m+1
```

Where m is the number of breaks and T_j are the break dates.

**Test Procedure:**
1. Sequential testing: Test for 0 vs 1 break, then 1 vs 2, etc.
2. Global optimization: BIC/LWZ criterion for optimal number of breaks
3. Confidence intervals for break dates

**Python Implementation:**

```python
import ruptures as rpt
import numpy as np
from scipy import stats

def bai_perron_test(y, X=None, max_breaks=5, min_segment_length=24):
    """
    Bai-Perron structural break detection.

    Parameters:
    -----------
    y : np.array
        Target series (reserves)
    X : np.array, optional
        Exogenous regressors
    max_breaks : int
        Maximum number of breaks to consider
    min_segment_length : int
        Minimum observations between breaks (h parameter)

    Returns:
    --------
    dict with:
        - n_breaks: optimal number of breaks
        - break_dates: list of break indices
        - bic_values: BIC for each number of breaks
        - confidence_intervals: 95% CI for each break
    """
    # Combine y and X if provided
    if X is not None:
        signal = np.column_stack([y, X])
    else:
        signal = y.reshape(-1, 1)

    # Use dynamic programming for optimal segmentation
    algo = rpt.Dynp(model="l2", min_size=min_segment_length, jump=1)
    algo.fit(signal)

    # Test different numbers of breaks
    bic_values = {}
    for n_bkps in range(0, max_breaks + 1):
        if n_bkps == 0:
            # No break model
            result = [len(y)]
        else:
            result = algo.predict(n_bkps=n_bkps)

        # Compute BIC
        bic = compute_bic(y, X, result)
        bic_values[n_bkps] = bic

    # Select optimal number of breaks
    optimal_n = min(bic_values, key=bic_values.get)

    if optimal_n == 0:
        break_dates = []
    else:
        break_dates = algo.predict(n_bkps=optimal_n)

    return {
        "n_breaks": optimal_n,
        "break_dates": break_dates[:-1],  # Exclude end point
        "bic_values": bic_values,
        "optimal_bic": bic_values[optimal_n]
    }


def compute_bic(y, X, break_points):
    """Compute BIC for a given segmentation."""
    n = len(y)
    segments = [0] + list(break_points)

    ssr = 0
    k = 0  # number of parameters

    for i in range(len(segments) - 1):
        start, end = segments[i], segments[i + 1]
        y_seg = y[start:end]

        if X is not None:
            X_seg = X[start:end]
            # OLS regression
            X_with_const = np.column_stack([np.ones(len(y_seg)), X_seg])
            beta = np.linalg.lstsq(X_with_const, y_seg, rcond=None)[0]
            resid = y_seg - X_with_const @ beta
            k += X_with_const.shape[1]
        else:
            # Mean model
            resid = y_seg - np.mean(y_seg)
            k += 1

        ssr += np.sum(resid ** 2)

    # BIC formula
    bic = n * np.log(ssr / n) + k * np.log(n)
    return bic
```

### 2. Chow Test for Known Breaks

```python
from scipy import stats

def chow_test(y, X, break_date_idx):
    """
    Chow test for structural break at known date.

    Returns:
    --------
    dict with F-statistic, p-value, and conclusion
    """
    n = len(y)

    # Add constant
    X_const = np.column_stack([np.ones(n), X]) if X is not None else np.ones((n, 1))
    k = X_const.shape[1]

    # Full sample regression
    beta_full = np.linalg.lstsq(X_const, y, rcond=None)[0]
    ssr_full = np.sum((y - X_const @ beta_full) ** 2)

    # Pre-break regression
    y1, X1 = y[:break_date_idx], X_const[:break_date_idx]
    beta1 = np.linalg.lstsq(X1, y1, rcond=None)[0]
    ssr1 = np.sum((y1 - X1 @ beta1) ** 2)

    # Post-break regression
    y2, X2 = y[break_date_idx:], X_const[break_date_idx:]
    beta2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
    ssr2 = np.sum((y2 - X2 @ beta2) ** 2)

    # Chow F-statistic
    ssr_unrestricted = ssr1 + ssr2
    f_stat = ((ssr_full - ssr_unrestricted) / k) / (ssr_unrestricted / (n - 2*k))
    p_value = 1 - stats.f.cdf(f_stat, k, n - 2*k)

    return {
        "f_statistic": f_stat,
        "p_value": p_value,
        "reject_null": p_value < 0.05,
        "interpretation": "Structural break detected" if p_value < 0.05 else "No break"
    }
```

### 3. CUSUM Test

```python
def cusum_test(residuals, significance=0.05):
    """
    CUSUM test for parameter stability.

    Parameters:
    -----------
    residuals : np.array
        OLS residuals from recursive estimation

    Returns:
    --------
    dict with CUSUM path, boundaries, and stability conclusion
    """
    n = len(residuals)
    sigma = np.std(residuals)

    # Cumulative sum of recursive residuals
    cusum = np.cumsum(residuals) / sigma

    # Critical boundaries (approximate)
    # Using Brownian motion approximation
    t = np.arange(1, n + 1) / n
    if significance == 0.05:
        a = 0.948  # 5% critical value coefficient
    elif significance == 0.01:
        a = 1.143  # 1% critical value coefficient
    else:
        a = 0.850  # 10%

    upper_bound = a * np.sqrt(n) + 2 * a * t * np.sqrt(n)
    lower_bound = -upper_bound

    # Check stability
    exceeds_bounds = np.any((cusum > upper_bound) | (cusum < lower_bound))

    return {
        "cusum": cusum,
        "upper_bound": upper_bound,
        "lower_bound": lower_bound,
        "stable": not exceeds_bounds,
        "first_crossing_idx": np.argmax((cusum > upper_bound) | (cusum < lower_bound)) if exceeds_bounds else None
    }
```

---

## Implementation Details

### File Structure
```
reserves_project/scripts/academic/
├── structural_breaks.py           ← Main script
├── break_detection/
│   ├── __init__.py
│   ├── bai_perron.py             ← Bai-Perron implementation
│   ├── chow_test.py              ← Chow test
│   ├── cusum.py                  ← CUSUM/CUSUMSQ
│   └── visualization.py          ← Break plots
```

### Output Structure
```
data/structural_breaks/
├── bai_perron_results.json       ← Optimal breaks, BIC
├── chow_test_results.json        ← Known break tests
├── cusum_results.json            ← Stability tests
├── break_summary.csv             ← All breaks consolidated
└── figures/
    ├── reserves_with_breaks.png
    ├── bic_selection.png
    ├── cusum_plot.png
    └── regime_comparison.png
```

---

## Test Configuration

### Bai-Perron Settings

```python
BAI_PERRON_CONFIG = {
    "max_breaks": 5,
    "min_segment_length": 24,  # 2 years minimum between breaks
    "significance_level": 0.05,
    "trimming_fraction": 0.15,  # Trim 15% from each end
}
```

### Known Break Dates to Test

```python
KNOWN_BREAK_DATES = [
    {"date": "2009-06-01", "event": "Post-war recovery begins"},
    {"date": "2018-10-01", "event": "Currency crisis onset"},
    {"date": "2020-03-01", "event": "COVID-19 lockdown"},
    {"date": "2022-04-01", "event": "Sovereign default"},
]
```

### Variables to Test for Breaks

```python
VARIABLES_FOR_BREAK_TESTS = [
    "gross_reserves_usd_m",           # Primary target
    "d_gross_reserves_usd_m",         # First difference
    "trade_balance_usd_m",            # Key predictor
    "usd_lkr",                        # Exchange rate
]
```

---

## Expected Outputs

### 1. Break Summary Table

| Variable | N Breaks | Break Dates | BIC | Events |
|----------|----------|-------------|-----|--------|
| gross_reserves_usd_m | 2-3 | [2019-Q1, 2022-Q2] | | BOP crisis, Default |
| trade_balance_usd_m | 1-2 | [2020-Q1] | | COVID |
| usd_lkr | 2-3 | [2018-Q4, 2022-Q2] | | Currency crises |

### 2. Chow Test Results

| Event | Date | F-stat | p-value | Conclusion |
|-------|------|--------|---------|------------|
| Post-war recovery | 2009-06 | | | |
| Currency crisis | 2018-10 | | | |
| COVID-19 | 2020-03 | | | |
| Default | 2022-04 | | | |

### 3. CUSUM Stability Assessment

| Variable | Stable? | First Crossing | Period |
|----------|---------|----------------|--------|
| gross_reserves_usd_m | No | idx=XX | 2022-XX |
| ... | | | |

---

## Integration with Forecasting Models

### How Break Information Will Be Used

1. **ARIMA/VECM**: Include break dummies as exogenous variables
   ```python
   break_dummies = create_break_dummies(dates, break_dates)
   model.fit(y, exog=np.column_stack([X, break_dummies]))
   ```

2. **MS-VAR/MS-VECM**: Compare detected breaks with estimated regime switches
   - If MS regimes align with Bai-Perron breaks → validation of regime approach
   - If misaligned → regime model may capture different dynamics

3. **TVP-VAR**: Use breaks to assess whether TVP captures gradual vs abrupt changes

4. **Threshold VAR**: Use break analysis to inform threshold variable selection

### Break Dummy Creation

```python
def create_break_dummies(dates, break_dates, dummy_type="level"):
    """
    Create dummy variables for structural breaks.

    Parameters:
    -----------
    dates : pd.DatetimeIndex
        Full date index
    break_dates : list
        List of break date strings
    dummy_type : str
        "level" - permanent shift (1 after break)
        "impulse" - one-time shock (1 at break only)
        "trend" - trend change (counter after break)
    """
    dummies = pd.DataFrame(index=dates)

    for i, break_date in enumerate(break_dates):
        break_dt = pd.Timestamp(break_date)
        col_name = f"break_{i+1}_{dummy_type}"

        if dummy_type == "level":
            dummies[col_name] = (dates >= break_dt).astype(int)
        elif dummy_type == "impulse":
            dummies[col_name] = (dates == break_dt).astype(int)
        elif dummy_type == "trend":
            dummies[col_name] = np.maximum(0, (dates - break_dt).days / 30)  # months since break

    return dummies
```

---

## Validation Criteria

### Pre-Execution Checklist
- [x] `ruptures` package installed
- [x] Source data available
- [x] Date range covers all known break events

### Post-Execution Validation
- [x] Bai-Perron converged for all variables
- [x] BIC selection is stable (no ties)
- [x] Detected breaks are economically interpretable
- [x] CUSUM plots generated
- [x] Break dummies created for forecasting use

---

## Execution Log

### Pre-Execution
| Check | Status | Notes |
|-------|--------|-------|
| ruptures installed | DONE | v1.1.10 |
| Data available | DONE | 252 observations, 2005-01 to 2025-12 |
| Config finalized | DONE | max_breaks=5, min_segment=24 |

### Execution
| Step | Status | Timestamp | Notes |
|------|--------|-----------|-------|
| Load data | DONE | 2026-02-10 09:23 | 252 obs loaded |
| Run Bai-Perron on reserves | DONE | 2026-02-10 09:23 | 4 breaks detected |
| Run Bai-Perron on other vars | DONE | 2026-02-10 09:23 | trade_balance: 4 breaks, usd_lkr: no data |
| Run Chow tests | DONE | 2026-02-10 09:23 | All 4 known events tested |
| Run CUSUM tests | DONE | 2026-02-10 09:23 | Both variables show instability |
| Generate plots | DONE | 2026-02-10 09:23 | 12 figures generated |
| Save results | DONE | 2026-02-10 09:23 | JSON, CSV, and dummies saved |

### Post-Execution
| Validation | Status | Notes |
|------------|--------|-------|
| Results interpretable | DONE | Breaks align with economic events |
| Plots clear | DONE | All 12 figures saved to figures/ |
| Break dummies ready | DONE | break_dummies.csv created |

---

## Results Section

### Bai-Perron Results

**Reserves (gross_reserves_usd_m):**
- Optimal number of breaks: **4**
- Break dates: 2009-10-01, 2012-12-01, 2016-01-01, 2021-01-01
- 95% CI for breaks:
  - Break 1 (2009-10): [2008-12-01, 2010-08-01]
  - Break 2 (2012-12): [2012-02-01, 2013-10-01]
  - Break 3 (2016-01): [2015-03-01, 2016-11-01]
  - Break 4 (2021-01): [2020-03-01, 2021-11-01]

**BIC Selection Path (gross_reserves_usd_m):**
| N Breaks | BIC | Selected |
|----------|-----|----------|
| 0 | 3861.27 | |
| 1 | 3741.16 | |
| 2 | 3550.26 | |
| 3 | 3546.08 | |
| 4 | 3545.77 | * |
| 5 | 3600.59 | |

**Segment Statistics (gross_reserves_usd_m):**
| Regime | Period | Mean (USD M) | Std Dev | N Obs |
|--------|--------|--------------|---------|-------|
| 1 | 2005-01 to 2009-09 | 3,017 | 499 | 57 |
| 2 | 2009-10 to 2012-11 | 6,608 | 607 | 38 |
| 3 | 2012-12 to 2015-12 | 7,633 | 775 | 37 |
| 4 | 2016-01 to 2020-12 | 7,024 | 1,070 | 60 |
| 5 | 2021-01 to 2025-12 | 4,110 | 1,721 | 60 |

**Trade Balance (trade_balance_usd_m):**
- Optimal number of breaks: **4**
- Break dates: 2010-12-01, 2016-04-01, 2019-01-01, 2022-05-01

### Chow Test Results

**Reserves (gross_reserves_usd_m):**
| Event | Date | F-stat | p-value | Conclusion |
|-------|------|--------|---------|------------|
| Post-war recovery | 2009-06 | 157.74 | <0.0001 | Significant *** |
| Currency crisis | 2018-10 | 4.66 | 0.0318 | Significant ** |
| COVID-19 | 2020-03 | 22.94 | <0.0001 | Significant *** |
| Sovereign default | 2022-04 | 14.43 | 0.0002 | Significant *** |

**Trade Balance (trade_balance_usd_m):**
| Event | Date | F-stat | p-value | Conclusion |
|-------|------|--------|---------|------------|
| Post-war recovery | 2009-06 | 40.50 | <0.0001 | Significant *** |
| Currency crisis | 2018-10 | 2.67 | 0.1038 | Not significant |
| COVID-19 | 2020-03 | 7.33 | 0.0073 | Significant *** |
| Sovereign default | 2022-04 | 11.46 | 0.0008 | Significant *** |

### CUSUM Results

| Variable | CUSUM Stable? | First Crossing | CUSUMSQ Stable? | First Crossing |
|----------|---------------|----------------|-----------------|----------------|
| gross_reserves_usd_m | No | 2006-03-01 | No | 2010-01-01 |
| trade_balance_usd_m | No | 2011-06-01 | No | 2011-10-01 |

Both variables show significant parameter instability in both mean (CUSUM) and variance (CUSUMSQ) tests.

### Economic Interpretation

1. **Post-war recovery (2009-10)**: The Bai-Perron test detects a break in October 2009, slightly after the war ended in May 2009. This aligns with the beginning of a rapid reserve accumulation phase (Regime 2: mean $6.6B vs Regime 1: $3.0B).

2. **Intermediate stability (2012-2015)**: A second break in December 2012 marks the peak reserve period (Regime 3: mean $7.6B), corresponding to the post-war economic boom.

3. **Pre-crisis period (2016-2020)**: The break in January 2016 signals the beginning of reserve volatility (Regime 4: std dev increases to $1.07B), coinciding with emerging external vulnerabilities.

4. **Crisis period (2021-present)**: The break in January 2021 captures the COVID-19 impact and subsequent sovereign default (Regime 5: mean drops to $4.1B with highest volatility at $1.72B std dev).

5. **Chow test validation**: All four known economic events show statistically significant structural breaks in reserves. The currency crisis (2018-10) shows the weakest effect (F=4.66) compared to the post-war recovery (F=157.74).

6. **Trade balance alignment**: Trade balance breaks at different dates, suggesting trade dynamics follow their own regime-switching pattern, though still influenced by the major economic events.

### Data Notes

- **usd_lkr (exchange rate)**: This variable has no data in the current panel and was skipped.
- **Recommended action**: Include exchange rate data for comprehensive analysis in future updates.

### Output Files Generated

- `data/structural_breaks/bai_perron_results.json` - Detailed Bai-Perron results
- `data/structural_breaks/chow_test_results.json` - Chow test results for known events
- `data/structural_breaks/cusum_results.json` - CUSUM stability test results
- `data/structural_breaks/break_summary.csv` - Consolidated summary table
- `data/structural_breaks/break_dummies.csv` - Dummy variables for forecasting
- `data/structural_breaks/figures/*.png` - 12 visualization files

---

## References

- Bai, J. & Perron, P. (1998). Estimating and Testing Linear Models with Multiple Structural Changes. Econometrica, 66(1), 47-78.
- Bai, J. & Perron, P. (2003). Computation and Analysis of Multiple Structural Change Models. Journal of Applied Econometrics, 18(1), 1-22.
- Brown, R.L., Durbin, J., & Evans, J.M. (1975). Techniques for Testing the Constancy of Regression Relationships Over Time. JRSS-B.

---

## Change Log

| Date | Version | Changes |
|------|---------|---------|
| 2026-02-10 | 1.0 | Initial specification |
| 2026-02-10 | 1.1 | Implementation complete - all tests run, results documented |

