# Specification 07: Mixed Data Sampling (MIDAS)
## Exploiting High-Frequency Exchange Rate Data

**Version:** 1.1
**Created:** 2026-02-10
**Status:** 🟢 COMPLETE
**Phase:** 2 (New Models)
**Dependencies:** 01_VARIABLE_SETS
**Blocks:** 09_DMA_DMS

---

## Objective

Implement MIDAS regression to exploit higher-frequency data for reserves forecasting:
1. Use **daily/weekly exchange rate** data to predict monthly reserves
2. Capture intra-month dynamics that monthly data misses
3. Apply flexible weighting schemes (Almon, Beta, exponential)
4. Compare with standard monthly-only models

---

## Theoretical Motivation

### The Frequency Mismatch Problem

- Reserves data: **monthly** (reported by CBSL)
- Exchange rate: Available **daily**
- Trade data: **monthly**
- The exchange rate contains valuable high-frequency information about intervention pressure

### MIDAS Solution

Instead of aggregating daily data to monthly (losing information), weight high-frequency observations:

```
R_t^{(m)} = α + β · B(L^{1/m}; θ) · X_t^{(d)} + ε_t

Where:
  R_t^{(m)} = monthly reserves
  X_t^{(d)} = daily exchange rate data
  B(L^{1/m}; θ) = polynomial weighting function
  m = frequency ratio (e.g., 22 trading days per month)
```

---

## Weighting Schemes

### 1. Exponential Almon Weights

```python
def exp_almon_weights(n_lags, theta1, theta2):
    """
    Exponential Almon polynomial weights.

    w_k = exp(theta1 * k + theta2 * k^2) / sum(exp(...))
    """
    k = np.arange(1, n_lags + 1)
    raw_weights = np.exp(theta1 * k + theta2 * k ** 2)
    return raw_weights / raw_weights.sum()
```

### 2. Beta Weights

```python
def beta_weights(n_lags, alpha, beta_param):
    """
    Beta polynomial weights (Ghysels et al., 2007).

    Based on beta distribution density.
    """
    k = np.linspace(1e-5, 1 - 1e-5, n_lags)
    raw_weights = k ** (alpha - 1) * (1 - k) ** (beta_param - 1)
    return raw_weights / raw_weights.sum()
```

### 3. Step Function (Unrestricted)

```python
def step_weights(n_lags, step_size=5):
    """
    Step function weights (piecewise constant).

    Groups high-frequency obs into blocks.
    """
    n_steps = n_lags // step_size
    weights = np.zeros(n_lags)

    for i in range(n_steps):
        start = i * step_size
        end = min((i + 1) * step_size, n_lags)
        weights[start:end] = 1.0 / n_steps

    weights /= weights.sum()
    return weights
```

---

## Implementation

### Core MIDAS Class

```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize

class MIDAS:
    """
    Mixed Data Sampling Regression.

    Handles frequency mismatch between LHS (monthly) and RHS (daily) variables.
    """

    def __init__(self, weight_type="exp_almon", n_hf_lags=22, n_lf_lags=3):
        """
        Parameters:
        -----------
        weight_type : str
            "exp_almon", "beta", or "step"
        n_hf_lags : int
            Number of high-frequency lags (days) per low-frequency period
        n_lf_lags : int
            Number of low-frequency lags (months)
        """
        self.weight_type = weight_type
        self.n_hf_lags = n_hf_lags
        self.n_lf_lags = n_lf_lags

        self.theta = None
        self.beta = None
        self.intercept = None

    def _get_weights(self, theta):
        """Compute weights given parameters."""
        if self.weight_type == "exp_almon":
            return exp_almon_weights(self.n_hf_lags, theta[0], theta[1])
        elif self.weight_type == "beta":
            return beta_weights(self.n_hf_lags, theta[0], theta[1])
        elif self.weight_type == "step":
            return step_weights(self.n_hf_lags)
        else:
            raise ValueError(f"Unknown weight type: {self.weight_type}")

    def _aggregate_hf_data(self, X_hf, dates_lf, weights):
        """
        Aggregate high-frequency data to low-frequency using MIDAS weights.

        Parameters:
        -----------
        X_hf : pd.Series
            High-frequency data with DatetimeIndex
        dates_lf : pd.DatetimeIndex
            Low-frequency dates
        weights : np.array
            MIDAS weights

        Returns:
        --------
        X_agg : np.array
            Aggregated high-frequency regressors
        """
        n_lf = len(dates_lf)
        X_agg = np.zeros((n_lf, self.n_lf_lags + 1))

        for t, date_lf in enumerate(dates_lf):
            for lag in range(self.n_lf_lags + 1):
                # Get month-lag start date
                lag_date = date_lf - pd.DateOffset(months=lag)
                month_start = lag_date.replace(day=1)
                month_end = (month_start + pd.DateOffset(months=1)) - pd.DateOffset(days=1)

                # Get HF data for this month
                mask = (X_hf.index >= month_start) & (X_hf.index <= month_end)
                hf_values = X_hf[mask].values

                if len(hf_values) >= self.n_hf_lags:
                    # Apply weights to last n_hf_lags observations
                    hf_values = hf_values[-self.n_hf_lags:]
                    X_agg[t, lag] = np.dot(weights, hf_values)
                elif len(hf_values) > 0:
                    # Proportional weights for shorter months
                    adj_weights = weights[:len(hf_values)]
                    adj_weights = adj_weights / adj_weights.sum()
                    X_agg[t, lag] = np.dot(adj_weights, hf_values)
                else:
                    X_agg[t, lag] = np.nan

        return X_agg

    def fit(self, Y_lf, X_hf, X_lf=None):
        """
        Fit MIDAS regression.

        Parameters:
        -----------
        Y_lf : pd.Series
            Low-frequency target (monthly reserves)
        X_hf : pd.Series
            High-frequency regressor (daily exchange rate)
        X_lf : pd.DataFrame, optional
            Additional low-frequency regressors
        """
        # Initial theta
        if self.weight_type in ["exp_almon", "beta"]:
            theta_init = np.array([0.0, -0.01])
        else:
            theta_init = np.array([])

        # Objective function
        def objective(theta):
            weights = self._get_weights(theta)
            X_agg = self._aggregate_hf_data(X_hf, Y_lf.index, weights)

            # Combine with LF regressors
            if X_lf is not None:
                X_combined = np.column_stack([X_agg, X_lf.values])
            else:
                X_combined = X_agg

            # Add intercept
            X_combined = np.column_stack([np.ones(len(Y_lf)), X_combined])

            # Remove NaN rows
            valid = ~np.isnan(X_combined).any(axis=1) & ~np.isnan(Y_lf.values)
            X_valid = X_combined[valid]
            Y_valid = Y_lf.values[valid]

            # OLS
            try:
                beta = np.linalg.lstsq(X_valid, Y_valid, rcond=None)[0]
                resid = Y_valid - X_valid @ beta
                ssr = np.sum(resid ** 2)
            except:
                ssr = 1e10

            return ssr

        # Optimize
        if len(theta_init) > 0:
            result = minimize(objective, theta_init, method='Nelder-Mead')
            self.theta = result.x
        else:
            self.theta = theta_init

        # Final estimation
        weights = self._get_weights(self.theta)
        X_agg = self._aggregate_hf_data(X_hf, Y_lf.index, weights)

        if X_lf is not None:
            X_combined = np.column_stack([X_agg, X_lf.values])
        else:
            X_combined = X_agg

        X_combined = np.column_stack([np.ones(len(Y_lf)), X_combined])

        valid = ~np.isnan(X_combined).any(axis=1) & ~np.isnan(Y_lf.values)
        self.valid_mask = valid
        X_valid = X_combined[valid]
        Y_valid = Y_lf.values[valid]

        self.beta = np.linalg.lstsq(X_valid, Y_valid, rcond=None)[0]
        self.intercept = self.beta[0]
        self.hf_coefs = self.beta[1:self.n_lf_lags + 2]

        # Fitted values and residuals
        self.fitted = X_valid @ self.beta
        self.residuals = Y_valid - self.fitted

        # Store for forecasting
        self.Y_lf = Y_lf
        self.X_hf = X_hf
        self.X_lf = X_lf
        self.weights = weights

        return self

    def forecast(self, h=1, X_hf_future=None, X_lf_future=None):
        """
        Generate h-step ahead forecast.

        Parameters:
        -----------
        h : int
            Forecast horizon (months)
        X_hf_future : pd.Series, optional
            Future high-frequency data (if available)
        X_lf_future : pd.DataFrame, optional
            Future low-frequency regressors

        Returns:
        --------
        forecasts : np.array
        """
        forecasts = np.zeros(h)

        for t in range(h):
            # Get future date
            last_date = self.Y_lf.index[-1]
            future_date = last_date + pd.DateOffset(months=t + 1)

            if X_hf_future is not None:
                # Use provided future HF data
                X_agg = self._aggregate_hf_data(
                    X_hf_future,
                    pd.DatetimeIndex([future_date]),
                    self.weights
                )[0]
            else:
                # Use last observed HF data (naive)
                X_agg = self._aggregate_hf_data(
                    self.X_hf,
                    pd.DatetimeIndex([last_date]),
                    self.weights
                )[0]

            # Combine regressors
            x_t = np.concatenate([[1], X_agg])

            if X_lf_future is not None and t < len(X_lf_future):
                x_t = np.concatenate([x_t, X_lf_future.iloc[t].values])
            elif self.X_lf is not None:
                x_t = np.concatenate([x_t, self.X_lf.iloc[-1].values])

            forecasts[t] = x_t @ self.beta

        return forecasts

    def get_weight_plot_data(self):
        """Return data for plotting MIDAS weights."""
        return {
            "weights": self.weights,
            "lags": np.arange(1, self.n_hf_lags + 1),
            "theta": self.theta,
            "weight_type": self.weight_type
        }
```

---

## Data Preparation

### High-Frequency Exchange Rate

```python
def prepare_hf_exchange_rate(daily_file_path, start_date, end_date):
    """
    Load and prepare daily exchange rate data.

    Returns:
    --------
    pd.Series with daily USD/LKR rates
    """
    # Load daily data
    df = pd.read_csv(daily_file_path, parse_dates=['date'], index_col='date')

    # Select date range
    df = df.loc[start_date:end_date]

    # Forward-fill weekends/holidays
    df = df.resample('D').ffill()

    # Compute returns (for stationarity)
    df['usd_lkr_return'] = np.log(df['usd_lkr']).diff()

    return df['usd_lkr_return'].dropna()
```

### Aligning Frequencies

```python
def align_midas_data(Y_monthly, X_daily, X_monthly_exog=None):
    """
    Align monthly target with daily regressors.

    Ensures proper date handling for MIDAS.
    """
    # Ensure monthly is end-of-month
    Y_monthly.index = Y_monthly.index.to_period('M').to_timestamp('M')

    # Daily should cover all monthly periods
    first_month = Y_monthly.index.min() - pd.DateOffset(months=3)
    last_day = Y_monthly.index.max()

    X_daily_aligned = X_daily.loc[first_month:last_day]

    return Y_monthly, X_daily_aligned, X_monthly_exog
```

---

## Model Variations

### 1. U-MIDAS (Unrestricted MIDAS)

When n_hf_lags is small, can estimate unrestricted:

```python
class UMIDAS:
    """
    Unrestricted MIDAS - direct OLS on all HF lags.
    """

    def __init__(self, n_hf_lags=5, n_lf_lags=3):
        self.n_hf_lags = n_hf_lags
        self.n_lf_lags = n_lf_lags

    def fit(self, Y_lf, X_hf):
        # Aggregate HF data into blocks
        # Estimate by OLS without weight restrictions
        pass
```

### 2. MIDAS-AR

Include autoregressive terms for reserves:

```python
class MIDAS_AR:
    """
    MIDAS with autoregressive low-frequency lags.

    R_t = α + ρ*R_{t-1} + β*B(L)*X_t^{(d)} + ε_t
    """

    def __init__(self, n_ar_lags=1, **midas_kwargs):
        self.n_ar_lags = n_ar_lags
        self.midas = MIDAS(**midas_kwargs)

    def fit(self, Y_lf, X_hf):
        # Add lagged Y to regressors
        Y_lagged = pd.concat([Y_lf.shift(i) for i in range(1, self.n_ar_lags + 1)], axis=1)
        self.midas.fit(Y_lf, X_hf, X_lf=Y_lagged)
        return self
```

---

## Comparison with Monthly-Only Models

| Model | Data Frequency | Weights | Parameters |
|-------|---------------|---------|------------|
| MIDAS (Almon) | Daily + Monthly | Smooth polynomial | 2 (theta) + k |
| MIDAS (Beta) | Daily + Monthly | Flexible hump | 2 (alpha, beta) + k |
| ARIMA | Monthly only | None | ARMA orders |
| VAR | Monthly only | None | k²p |

### Information Gain Metric

```python
def midas_information_gain(midas_rmse, monthly_rmse):
    """
    Compute relative RMSE improvement from using HF data.
    """
    return (monthly_rmse - midas_rmse) / monthly_rmse * 100
```

---

## File Structure

```
reserves_project/scripts/academic/
├── run_midas.py                   ← Main execution
├── models/
│   ├── midas.py                   ← MIDAS class
│   ├── midas_weights.py           ← Weight functions
│   └── midas_data_prep.py         ← Frequency alignment
```

## Output Structure

```
data/forecast_results_academic/midas/
├── midas_forecasts.csv
├── midas_weights.csv                  ← Estimated weights
├── midas_coefficients.json
├── midas_rolling_backtest.csv
└── figures/
    ├── midas_weight_functions.png
    ├── midas_vs_monthly.png
    └── intramonth_dynamics.png
```

---

## Data Requirements

### High-Frequency Data Needed

| Variable | Frequency | Source | Coverage |
|----------|-----------|--------|----------|
| USD/LKR rate | Daily | CBSL/Bloomberg | 2008-2025 |
| VIX | Daily | CBOE | 2008-2025 |
| Oil prices | Daily | EIA | 2008-2025 |

### Note on Data Availability

If daily data is not available, can use:
- **Weekly averages** (m = 4-5 per month)
- **Bi-weekly** data

---

## Execution Log

### Pre-Execution
| Check | Status | Notes |
|-------|--------|-------|
| Spec 01 complete | ✅ | Variable sets prepared |
| Daily exchange rate data | ✅ | 1169 daily obs (2021-01-12 to 2025-12-19) |
| scipy installed | ✅ | For optimization |

### Execution
| Step | Status | Timestamp | Notes |
|------|--------|-----------|-------|
| Load daily data | ✅ | 2026-02-10 | USD/LKR from CBSL |
| Fit MIDAS (Almon) | ✅ | 2026-02-10 | θ₁=-1.40, θ₂=0.047 |
| Fit MIDAS (Beta) | ✅ | 2026-02-10 | α=0.81, β=6.04 |
| Fit MIDAS-AR | ✅ | 2026-02-10 | AR(1) + Almon weights |
| Compare with monthly | ✅ | 2026-02-10 | MIDAS-AR beats in-sample |
| Rolling backtest | ✅ | 2026-02-10 | 23 origins, h=1,3 |

### Post-Execution
| Validation | Status | Notes |
|------------|--------|-------|
| Weights reasonable | ✅ | Declining from recent lags |
| Improvement over monthly | ⚠️ | In-sample yes, out-of-sample no |
| Forecasts stable | ✅ | No divergence observed |

---

## Results Section

### Data Summary
- **Daily Exchange Rate**: 1,169 observations (2021-01-12 to 2025-12-19)
- **Monthly Reserves Overlap**: 50 observations (2021-02 to 2025-03)
- **Estimation Sample**: 47 valid observations after lag adjustments
- **Backtest Period**: 2023-01 to 2025-03 (23 forecast origins)

### Weight Estimates

| Weight Type | θ₁ / α | θ₂ / β | Shape |
|-------------|--------|--------|-------|
| Exp Almon | -1.40 | 0.047 | Strong recent-lag weight (70% on day 1) |
| Beta | 0.81 | 6.04 | Declining from recent, similar to Almon |

**Interpretation**: Both weight functions concentrate weight heavily on the most recent trading days, suggesting that short-term FX movements are most informative for reserves forecasting.

### In-Sample Model Comparison

| Model | R² | RMSE | Notes |
|-------|-----|------|-------|
| AR(3) Baseline | 0.932 | 421.0 | Monthly data only |
| MIDAS (Exp Almon) | 0.254 | 1394.9 | Without AR terms |
| MIDAS (Beta) | 0.285 | 1365.5 | Without AR terms |
| **MIDAS-AR** | **0.938** | **401.3** | AR(1) + daily FX |

### Information Gain (In-Sample)
| Comparison | MIDAS-AR RMSE | Baseline RMSE | Improvement % |
|------------|---------------|---------------|---------------|
| MIDAS-AR vs AR(3) | 401.3 | 421.0 | **4.7%** |

### Rolling Backtest Results (Out-of-Sample)

| Model | Horizon | MAE | RMSE | N |
|-------|---------|-----|------|---|
| AR Baseline | h=1 | 453.3 | 548.4 | 23 |
| AR Baseline | h=3 | 776.3 | 900.2 | 23 |
| MIDAS-AR (Almon) | h=1 | 636.7 | 719.5 | 23 |
| MIDAS-AR (Almon) | h=3 | 913.4 | 1075.0 | 23 |
| MIDAS-AR (Beta) | h=1 | 633.3 | 713.9 | 23 |
| MIDAS-AR (Beta) | h=3 | 910.0 | 1074.8 | 23 |

### Key Finding: Limited Out-of-Sample Gains

The MIDAS-AR model shows **in-sample improvement** (4.7% lower RMSE) but **underperforms out-of-sample** (-31% at h=1, -19% at h=3). This is a common finding in MIDAS literature and can be attributed to:

1. **Limited overlap period**: Only 50 monthly observations with daily FX data
2. **Turbulent period**: 2021-2025 includes Sri Lanka's economic crisis with extreme FX volatility
3. **Overfitting risk**: Daily data adds noise in volatile regimes
4. **Structural breaks**: FX-reserves relationship may have changed during crisis

### Recommendations

1. **Use with caution**: MIDAS adds value for in-sample fit but may not improve forecasts
2. **Consider ensemble**: Combine MIDAS-AR with simpler models via DMA/DMS
3. **Regime-dependent**: MIDAS may work better in stable periods
4. **Expand data**: Future analysis with longer daily history may show different results

---

## References

- Ghysels, E., Santa-Clara, P., & Valkanov, R. (2006). Predicting Volatility: Getting the Most out of Return Data. Review of Financial Studies.
- Ghysels, E., Sinko, A., & Valkanov, R. (2007). MIDAS Regressions: Further Results and New Directions. Econometric Reviews.
- Andreou, E., Ghysels, E., & Kourtellos, A. (2013). Should Macroeconomic Forecasters Use Daily Financial Data and How? Journal of Business & Economic Statistics.

---

## Change Log

| Date | Version | Changes |
|------|---------|---------|
| 2026-02-10 | 1.0 | Initial specification |
| 2026-02-10 | 1.1 | Implementation complete - Added results, findings, recommendations |

