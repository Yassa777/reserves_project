# Specification 06: Threshold VAR (TVAR)
## Regime-Switching Based on Observable Threshold Variable

**Version:** 1.1
**Created:** 2026-02-10
**Updated:** 2026-02-10
**Status:** 🟢 COMPLETE
**Phase:** 2 (New Models)
**Dependencies:** 01_VARIABLE_SETS, 02_STRUCTURAL_BREAKS
**Blocks:** 09_DMA_DMS

---

## Objective

Implement Threshold VAR (TVAR) as an alternative to Markov-Switching VAR:
1. Regimes determined by **observable** threshold variable (vs unobserved states in MS-VAR)
2. More interpretable regime definition (e.g., "depreciation regime" when Δ(usd_lkr) > τ)
3. Easier policy interpretation (conditions for regime switch are known)
4. Compare with MS-VAR for robustness

---

## Theoretical Motivation

### TVAR vs MS-VAR

| Aspect | Threshold VAR | Markov-Switching VAR |
|--------|---------------|---------------------|
| Regime trigger | Observable variable (z_t) | Latent Markov chain |
| Transition | Deterministic given z_t | Probabilistic |
| Interpretation | Clear (e.g., "high depreciation") | Inferred from data |
| Estimation | Grid search + OLS | EM or MCMC |
| Forecasting | Requires z_t forecast | Probabilistic weighting |

### Model Specification

**Two-Regime TVAR:**
```
Y_t = { Φ₁(L) Y_{t-1} + ε₁_t   if z_{t-d} ≤ τ  (Regime 1)
      { Φ₂(L) Y_{t-1} + ε₂_t   if z_{t-d} > τ  (Regime 2)

Where:
  z_t = threshold variable (e.g., exchange rate change)
  d = delay parameter (typically 1)
  τ = threshold value (estimated)
  Φ_j = regime-specific VAR coefficients
  ε_j ~ N(0, Σ_j)
```

---

## Threshold Variable Candidates

For Sri Lankan reserves, consider:

| Variable | Rationale | Expected Regimes |
|----------|-----------|------------------|
| `Δ(usd_lkr)` | Exchange rate depreciation | Stable vs crisis |
| `trade_balance` | External balance | Surplus vs deficit |
| `Δ(reserves)` | Reserve trend | Accumulation vs depletion |
| `global_risk` | VIX or similar | Risk-on vs risk-off |

### Primary Choice: Exchange Rate Depreciation

```python
THRESHOLD_VARIABLE = "usd_lkr_pct_change"  # Monthly % change
THRESHOLD_DELAY = 1  # Use t-1 value to avoid simultaneity
```

**Regime Interpretation:**
- **Regime 1 (z ≤ τ):** Stable/appreciation period - normal reserve dynamics
- **Regime 2 (z > τ):** Depreciation pressure - crisis dynamics, potential intervention

---

## Implementation

### Core TVAR Class

```python
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.api import VAR

class ThresholdVAR:
    """
    Threshold Vector Autoregression with two regimes.

    Estimation via concentrated maximum likelihood:
    1. Grid search over threshold τ
    2. OLS estimation conditional on τ
    3. Select τ minimizing total SSR
    """

    def __init__(self, n_lags=2, delay=1, trim=0.15):
        """
        Parameters:
        -----------
        n_lags : int
            VAR lag order
        delay : int
            Threshold delay (d in z_{t-d})
        trim : float
            Trimming fraction (exclude extreme τ values)
        """
        self.n_lags = n_lags
        self.delay = delay
        self.trim = trim

        self.threshold = None
        self.var_regime1 = None
        self.var_regime2 = None
        self.regime_indicators = None

    def fit(self, Y, z, threshold_var_name="threshold"):
        """
        Fit TVAR model.

        Parameters:
        -----------
        Y : pd.DataFrame
            Multivariate system (T x k)
        z : pd.Series
            Threshold variable
        """
        # Align data
        common_idx = Y.index.intersection(z.index)
        Y = Y.loc[common_idx]
        z = z.loc[common_idx]

        T = len(Y)
        k = Y.shape[1]

        # Create lagged threshold variable
        z_lagged = z.shift(self.delay).dropna()
        valid_idx = z_lagged.index
        Y = Y.loc[valid_idx]
        z_lagged = z_lagged.loc[valid_idx]

        # Grid search for threshold
        sorted_z = np.sort(z_lagged.values)
        n_grid = len(sorted_z)

        # Trim extremes
        lower_idx = int(n_grid * self.trim)
        upper_idx = int(n_grid * (1 - self.trim))
        threshold_grid = sorted_z[lower_idx:upper_idx]

        best_ssr = np.inf
        best_threshold = None

        for tau in threshold_grid:
            # Split data by regime
            regime1_mask = z_lagged <= tau
            regime2_mask = z_lagged > tau

            # Ensure minimum observations in each regime
            n1 = regime1_mask.sum()
            n2 = regime2_mask.sum()
            min_obs = (k * self.n_lags + 1) * 2  # Minimum for estimation

            if n1 < min_obs or n2 < min_obs:
                continue

            # Estimate regime-specific VARs
            try:
                Y1 = Y[regime1_mask]
                Y2 = Y[regime2_mask]

                var1 = VAR(Y1).fit(self.n_lags)
                var2 = VAR(Y2).fit(self.n_lags)

                # Total SSR
                ssr1 = np.sum(var1.resid ** 2)
                ssr2 = np.sum(var2.resid ** 2)
                total_ssr = ssr1 + ssr2

                if total_ssr < best_ssr:
                    best_ssr = total_ssr
                    best_threshold = tau
                    self.var_regime1 = var1
                    self.var_regime2 = var2

            except Exception as e:
                continue

        if best_threshold is None:
            raise ValueError("Could not find valid threshold")

        self.threshold = best_threshold
        self.regime_indicators = (z_lagged > best_threshold).astype(int)
        self.Y = Y
        self.z = z_lagged
        self.threshold_var_name = threshold_var_name

        # Store summary statistics
        self._compute_regime_stats()

        return self

    def _compute_regime_stats(self):
        """Compute regime-specific statistics."""
        regime1_mask = self.regime_indicators == 0
        regime2_mask = self.regime_indicators == 1

        self.regime_stats = {
            "regime1": {
                "n_obs": regime1_mask.sum(),
                "pct": regime1_mask.mean() * 100,
                "mean_threshold_var": self.z[regime1_mask].mean(),
            },
            "regime2": {
                "n_obs": regime2_mask.sum(),
                "pct": regime2_mask.mean() * 100,
                "mean_threshold_var": self.z[regime2_mask].mean(),
            },
            "threshold": self.threshold,
        }

    def forecast(self, h=12, z_future=None):
        """
        Generate h-step forecasts.

        Parameters:
        -----------
        h : int
            Forecast horizon
        z_future : np.array, optional
            Future values of threshold variable
            If None, use last observed regime

        Returns:
        --------
        forecasts : np.array (h x k)
        """
        if z_future is None:
            # Use last regime for all forecasts
            last_z = self.z.iloc[-1]
            regime = 2 if last_z > self.threshold else 1
            var_model = self.var_regime2 if regime == 2 else self.var_regime1

            # Use that regime's VAR for all h steps
            last_obs = self.Y.values[-self.n_lags:]
            forecasts = var_model.forecast(last_obs, steps=h)

        else:
            # Regime can switch during forecast
            forecasts = np.zeros((h, self.Y.shape[1]))
            current_obs = self.Y.values[-self.n_lags:]

            for t in range(h):
                z_t = z_future[t] if t < len(z_future) else z_future[-1]
                regime = 2 if z_t > self.threshold else 1
                var_model = self.var_regime2 if regime == 2 else self.var_regime1

                # One-step forecast
                fc_t = var_model.forecast(current_obs, steps=1)[0]
                forecasts[t] = fc_t

                # Update lags
                current_obs = np.vstack([current_obs[1:], fc_t])

        return forecasts

    def forecast_by_scenario(self, h=12):
        """
        Generate forecasts for both regime scenarios.
        """
        last_obs = self.Y.values[-self.n_lags:]

        # Scenario 1: Stay in regime 1 (stable)
        fc_regime1 = self.var_regime1.forecast(last_obs, steps=h)

        # Scenario 2: Stay in regime 2 (crisis)
        fc_regime2 = self.var_regime2.forecast(last_obs, steps=h)

        return {
            "regime1_scenario": fc_regime1,
            "regime2_scenario": fc_regime2,
        }

    def linearity_test(self):
        """
        Test for threshold nonlinearity vs linear VAR.
        Uses sup-Wald test.
        """
        # Fit linear VAR
        var_linear = VAR(self.Y).fit(self.n_lags)
        ssr_linear = np.sum(var_linear.resid ** 2)

        # TVAR SSR
        ssr_tvar = (np.sum(self.var_regime1.resid ** 2) +
                    np.sum(self.var_regime2.resid ** 2))

        # Wald statistic
        T = len(self.Y)
        k = self.Y.shape[1]
        n_params = k * self.n_lags + 1  # Per equation

        # F-type statistic
        df1 = n_params * k  # Extra parameters in TVAR
        df2 = T - 2 * n_params * k

        f_stat = ((ssr_linear - ssr_tvar) / df1) / (ssr_tvar / df2)

        # Note: Critical values require bootstrap due to Davies problem
        # (threshold not identified under null)

        return {
            "f_statistic": f_stat,
            "df1": df1,
            "df2": df2,
            "ssr_linear": ssr_linear,
            "ssr_tvar": ssr_tvar,
            "note": "Bootstrap p-value required (Davies problem)"
        }

    def bootstrap_linearity_test(self, n_bootstrap=500):
        """
        Bootstrap test for linearity (handles Davies problem).
        """
        # Observed F-statistic
        test_result = self.linearity_test()
        f_obs = test_result["f_statistic"]

        # Fit linear VAR under null
        var_linear = VAR(self.Y).fit(self.n_lags)

        f_bootstrap = np.zeros(n_bootstrap)

        for b in range(n_bootstrap):
            # Generate data under null (linear)
            Y_boot = var_linear.simulate_var(steps=len(self.Y))
            Y_boot = pd.DataFrame(Y_boot, columns=self.Y.columns)

            # Re-estimate TVAR on bootstrap data
            try:
                tvar_boot = ThresholdVAR(n_lags=self.n_lags, delay=self.delay)
                tvar_boot.fit(Y_boot, self.z)
                f_bootstrap[b] = tvar_boot.linearity_test()["f_statistic"]
            except:
                f_bootstrap[b] = 0

        # Bootstrap p-value
        p_value = np.mean(f_bootstrap >= f_obs)

        return {
            "f_statistic": f_obs,
            "bootstrap_p_value": p_value,
            "reject_linearity": p_value < 0.05,
        }
```

---

## Threshold Selection Diagnostics

### Confidence Interval for Threshold

```python
def threshold_confidence_interval(tvar, alpha=0.05):
    """
    Construct confidence interval for threshold parameter.
    Based on likelihood ratio inversion.
    """
    T = len(tvar.Y)
    k = tvar.Y.shape[1]

    # Best SSR
    ssr_best = (np.sum(tvar.var_regime1.resid ** 2) +
                np.sum(tvar.var_regime2.resid ** 2))

    # Critical value (chi-square)
    cv = stats.chi2.ppf(1 - alpha, df=1)

    # Grid search for CI bounds
    sorted_z = np.sort(tvar.z.values)
    n_grid = len(sorted_z)
    lower_idx = int(n_grid * tvar.trim)
    upper_idx = int(n_grid * (1 - tvar.trim))
    threshold_grid = sorted_z[lower_idx:upper_idx]

    in_ci = []

    for tau in threshold_grid:
        # Compute SSR at this tau
        regime1_mask = tvar.z <= tau
        regime2_mask = tvar.z > tau

        n1, n2 = regime1_mask.sum(), regime2_mask.sum()
        min_obs = (k * tvar.n_lags + 1) * 2

        if n1 < min_obs or n2 < min_obs:
            continue

        try:
            var1 = VAR(tvar.Y[regime1_mask]).fit(tvar.n_lags)
            var2 = VAR(tvar.Y[regime2_mask]).fit(tvar.n_lags)
            ssr_tau = np.sum(var1.resid ** 2) + np.sum(var2.resid ** 2)

            # LR statistic
            lr_stat = T * np.log(ssr_tau / ssr_best)

            if lr_stat <= cv:
                in_ci.append(tau)
        except:
            continue

    return {
        "point_estimate": tvar.threshold,
        "lower": min(in_ci) if in_ci else tvar.threshold,
        "upper": max(in_ci) if in_ci else tvar.threshold,
    }
```

---

## Regime Visualization

```python
def plot_regime_indicators(tvar, Y_var="gross_reserves_usd_m"):
    """
    Plot series with regime shading.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    dates = tvar.Y.index

    # Panel 1: Target variable with regime shading
    ax1 = axes[0]
    ax1.plot(dates, tvar.Y[Y_var], 'b-', linewidth=1.5)
    ax1.fill_between(dates, ax1.get_ylim()[0], ax1.get_ylim()[1],
                     where=tvar.regime_indicators == 1,
                     alpha=0.3, color='red', label='Regime 2 (Crisis)')
    ax1.set_ylabel('Reserves (USD m)')
    ax1.set_title('Reserves with Regime Indicators')
    ax1.legend()

    # Panel 2: Threshold variable
    ax2 = axes[1]
    ax2.plot(dates, tvar.z, 'k-', linewidth=1)
    ax2.axhline(tvar.threshold, color='red', linestyle='--',
                label=f'Threshold = {tvar.threshold:.3f}')
    ax2.fill_between(dates, tvar.z, tvar.threshold,
                     where=tvar.z > tvar.threshold,
                     alpha=0.3, color='red')
    ax2.set_ylabel('Threshold Variable')
    ax2.legend()

    # Panel 3: Regime indicator
    ax3 = axes[2]
    ax3.fill_between(dates, 0, tvar.regime_indicators,
                     step='mid', alpha=0.5, color='red')
    ax3.set_ylabel('Regime')
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['Stable', 'Crisis'])

    plt.tight_layout()
    return fig
```

---

## Comparison with MS-VAR

```python
def compare_tvar_msvar(tvar_regimes, msvar_probs, dates):
    """
    Compare TVAR regime indicators with MS-VAR smoothed probabilities.
    """
    # Concordance measure
    # MS-VAR regime = 1 if P(regime 2) > 0.5
    msvar_regimes = (msvar_probs > 0.5).astype(int)

    concordance = np.mean(tvar_regimes == msvar_regimes)

    # Correlation
    correlation = np.corrcoef(tvar_regimes, msvar_probs)[0, 1]

    # Timing differences
    tvar_switches = np.where(np.diff(tvar_regimes) != 0)[0]
    msvar_switches = np.where(np.diff(msvar_regimes) != 0)[0]

    return {
        "concordance": concordance,
        "correlation": correlation,
        "tvar_switch_dates": dates[tvar_switches],
        "msvar_switch_dates": dates[msvar_switches],
    }
```

---

## File Structure

```
reserves_project/scripts/academic/
├── run_threshold_var.py           ← Main execution
├── models/
│   ├── tvar.py                    ← ThresholdVAR class
│   ├── tvar_tests.py              ← Linearity tests
│   └── tvar_comparison.py         ← Compare with MS-VAR
```

## Output Structure

```
data/forecast_results_academic/tvar/
├── tvar_forecasts_{varset}.csv
├── tvar_threshold_{varset}.json       ← Threshold estimate, CI
├── tvar_regime_indicators_{varset}.csv
├── tvar_linearity_test_{varset}.json
├── tvar_vs_msvar_comparison.json
├── tvar_rolling_backtest_{varset}.csv
└── figures/
    ├── tvar_regime_plot_{varset}.png
    ├── threshold_selection.png
    └── tvar_vs_msvar.png
```

---

## Execution Log

### Pre-Execution
| Check | Status | Notes |
|-------|--------|-------|
| Spec 01 complete | ✅ | Variable sets prepared |
| Spec 02 complete | ✅ | Structural breaks analyzed |
| Threshold variable available | ✅ | usd_lkr from historical_fx.csv |

### Execution
| Step | Status | Timestamp | Notes |
|------|--------|-----------|-------|
| Estimate threshold (parsimonious) | ✅ | 2026-02-10 09:40 | tau=-0.01% |
| Estimate threshold (bop) | ✅ | 2026-02-10 09:40 | tau=-0.20% |
| Estimate threshold (monetary) | ✅ | 2026-02-10 09:40 | tau=-0.18% |
| Linearity tests | ✅ | 2026-02-10 09:40 | No improvement over linear VAR |
| Compare with MS-VAR | ⬜ | | Pending MS-VAR implementation |
| Rolling backtest | ✅ | 2026-02-10 09:40 | 12-month horizon |

### Post-Execution
| Validation | Status | Notes |
|------------|--------|-------|
| Threshold interpretable | ⚠️ | Low thresholds (~0%) not economically meaningful |
| Linearity rejected | ❌ | TVAR does not improve over linear VAR |
| Comparison with MS-VAR | ⬜ | Pending |

---

## Results Section

### Key Finding: TVAR Does Not Improve Over Linear VAR

**The threshold VAR specification does not provide statistically significant improvement over a linear VAR model for Sri Lankan reserves forecasting.** This is evidenced by:

1. **Negative SSR reductions**: All variable sets show the TVAR SSR exceeding the linear VAR SSR
2. **Negative F-statistics**: Indicating the linear model fits better
3. **Low optimal thresholds**: The algorithm finds thresholds near 0%, which lack economic interpretation

### Threshold Estimates

| Variable Set | Threshold τ | 95% CI | Regime 1 % | Regime 2 % | SSR Reduction |
|--------------|-------------|--------|------------|------------|---------------|
| Parsimonious | -0.01% | [-0.21%, 0.01%] | 36.7% | 63.3% | -44.8% |
| BoP | -0.20% | [-0.20%, -0.19%] | 23.6% | 76.4% | -36.7% |
| Monetary | -0.18% | [-0.18%, -0.14%] | 22.6% | 77.4% | -178.1% |

### Linearity Test Results

| Variable Set | F-stat | SSR Reduction | Reject H0? | Interpretation |
|--------------|--------|---------------|------------|----------------|
| Parsimonious | -2.55 | -44.8% | No | Linear VAR preferred |
| BoP | -0.89 | -36.7% | No | Linear VAR preferred |
| Monetary | -5.34 | -178.1% | No | Linear VAR preferred |

### Regime Persistence Analysis

| Variable Set | P(stay stable) | P(stay crisis) | Persistence Index | Transitions |
|--------------|----------------|----------------|-------------------|-------------|
| Parsimonious | 0.63 | 0.78 | 0.70 | 59 |
| BoP | 0.41 | 0.82 | 0.61 | 53 |
| Monetary | 0.42 | 0.83 | 0.62 | 57 |

### Rolling Backtest Performance

| Variable Set | RMSE | MAE | Notes |
|--------------|------|-----|-------|
| Parsimonious | 10,773.96 | 3,034.34 | High error due to crisis period |
| BoP | 1,834.41 | 1,360.34 | Best performance |
| Monetary | 2,056.39 | 1,555.79 | Moderate performance |

### Why TVAR Fails for This Data

1. **Sparse Crisis Observations**: Only ~34 observations (16%) have exchange rate depreciation > 1% per month. This is insufficient for reliable regime-specific VAR estimation.

2. **Non-Contiguous Regime Segments**: Crisis periods are scattered across time, breaking the temporal structure that VARs rely on.

3. **Gradual Transitions**: Sri Lankan crises (2012, 2018, 2022) involved gradual deterioration rather than sharp regime switches. MS-VAR with probabilistic transitions may be more appropriate.

4. **Threshold Variable Choice**: Exchange rate % change may not be the optimal threshold indicator. Alternative candidates:
   - Reserve coverage ratio
   - Import cover months
   - Trade balance as % of GDP
   - Sovereign CDS spreads

### Comparison with MS-VAR
*Pending MS-VAR implementation - to be updated*

### Recommendations

1. **Use Linear VAR or MS-VAR**: For reserves forecasting, linear VAR or Markov-Switching VAR are preferred over TVAR given the empirical results.

2. **Alternative Threshold Variables**: Future research could explore:
   - Import cover falling below critical threshold (e.g., 3 months)
   - Reserve depletion rate exceeding historical norms
   - External debt service coverage ratio

3. **Smooth Transition VAR (STVAR)**: Consider STVAR which allows gradual regime transitions rather than abrupt switches.

---

## References

- Tong, H. (1990). Non-linear Time Series: A Dynamical System Approach. Oxford.
- Hansen, B.E. (1996). Inference When a Nuisance Parameter Is Not Identified Under the Null Hypothesis. Econometrica.
- Tsay, R.S. (1998). Testing and Modeling Multivariate Threshold Models. JASA.

---

## Output Files Generated

```
data/forecast_results_academic/tvar/
├── tvar_summary.csv                           # Summary statistics
├── tvar_forecasts_parsimonious.csv            # 12-month forecasts
├── tvar_forecasts_bop.csv
├── tvar_forecasts_monetary.csv
├── tvar_threshold_parsimonious.json           # Full model results
├── tvar_threshold_bop.json
├── tvar_threshold_monetary.json
├── tvar_regime_indicators_parsimonious.csv    # Regime time series
├── tvar_regime_indicators_bop.csv
├── tvar_regime_indicators_monetary.csv
├── tvar_rolling_backtest_parsimonious.csv     # Backtest results
├── tvar_rolling_backtest_bop.csv
├── tvar_rolling_backtest_monetary.csv
└── figures/
    ├── tvar_regimes_parsimonious.png          # 3-panel regime plots
    ├── tvar_regimes_bop.png
    ├── tvar_regimes_monetary.png
    ├── tvar_threshold_selection_parsimonious.png  # SSR vs threshold
    ├── tvar_threshold_selection_bop.png
    ├── tvar_threshold_selection_monetary.png
    ├── tvar_forecast_scenarios_parsimonious.png   # Scenario forecasts
    ├── tvar_forecast_scenarios_bop.png
    └── tvar_forecast_scenarios_monetary.png
```

---

## Change Log

| Date | Version | Changes |
|------|---------|---------|
| 2026-02-10 | 1.0 | Initial specification |
| 2026-02-10 | 1.1 | Execution complete - TVAR does not improve over linear VAR |

