# Reserve Forecasting Diagnostic Test Results

**Generated:** 2026-02-03
**Data:** `data/merged/reserves_forecasting_panel.csv`
**Target Variable:** `gross_reserves_usd_m`
**Observations:** 252 monthly (Jan 2005 – Dec 2025)

---

## Executive Summary

| Phase | Key Finding | Modeling Implication |
|-------|-------------|---------------------|
| **Stationarity** | Target (`gross_reserves_usd_m`) is I(1) non-stationary | Use first differences or cointegration |
| **Persistence** | High ACF(1)=0.97 for reserves | AR components needed; slow mean-reversion |
| **Volatility** | ARCH effects present; crisis vol 2.15× pre-crisis | Consider GARCH or regime-switching |
| **Breaks** | Chow test shows NO break in reserves at Apr 2022 | Surprising—model may need deeper investigation |
| **Causality** | No significant Granger causality from predictors | Consider contemporaneous relationships |

---

## Phase 1: Data Quality (Pre-Merge)
*See Streamlit app: `app_reserves_diagnostics.py`*

| Check | Status | Notes |
|-------|--------|-------|
| Missing Values | ✅ Complete | Crisis period (2020-2024) has good coverage |
| Outliers | ✅ Analyzed | 2022 crisis values are genuine extremes |
| Cross-Source Validation | ✅ Complete | CBSL vs World Bank aligned |

---

## Phase 2: Stationarity & Integration Order

### 2.1 Augmented Dickey-Fuller (ADF) Test

**Test Settings:** Regression: Constant + Trend, Lag Selection: AIC (auto), H₀: Unit root exists

| Variable | ADF Stat | p-value | Lags | Stationary (5%)? |
|----------|----------|---------|------|------------------|
| gross_reserves_usd_m | -2.095 | 0.549 | auto | ❌ No |
| net_usable_reserves_usd_m | -2.333 | 0.416 | auto | ❌ No |
| exports_usd_m | -3.421 | 0.049 | auto | ✅ Yes |
| imports_usd_m | -3.274 | 0.071 | auto | ❌ No (marginal) |
| remittances_usd_m | -2.720 | 0.228 | auto | ❌ No |
| tourism_usd_m | -2.037 | 0.581 | auto | ❌ No |
| cse_net_usd_m | -3.269 | 0.072 | auto | ❌ No (marginal) |
| trade_balance_usd_m | -3.674 | 0.024 | auto | ✅ Yes |
| reserve_change_usd_m | -4.501 | 0.002 | auto | ✅ Yes |

### 2.2 KPSS Test

**Test Settings:** Regression: Constant + Trend, H₀: Series is stationary

| Variable | KPSS Stat | p-value | Stationary (5%)? |
|----------|-----------|---------|------------------|
| gross_reserves_usd_m | 0.394 | <0.01 | ❌ No |
| net_usable_reserves_usd_m | 0.412 | <0.01 | ❌ No |
| exports_usd_m | 0.144 | 0.054 | ✅ Yes (marginal) |
| imports_usd_m | 0.305 | <0.01 | ❌ No |
| remittances_usd_m | 0.288 | <0.01 | ❌ No |
| tourism_usd_m | 0.330 | <0.01 | ❌ No |
| cse_net_usd_m | 0.158 | 0.040 | ❌ No |
| trade_balance_usd_m | 0.336 | <0.01 | ❌ No |
| reserve_change_usd_m | 0.095 | >0.10 | ✅ Yes |

### 2.3 Integration Order Summary

| Variable | ADF Result | KPSS Result | Conclusion | d |
|----------|------------|-------------|------------|---|
| **gross_reserves_usd_m** | Non-stationary | Non-stationary | **I(1) or higher** | 1 |
| **net_usable_reserves_usd_m** | Non-stationary | Non-stationary | **I(1) or higher** | 1 |
| exports_usd_m | Stationary | Stationary | **I(0)** | 0 |
| imports_usd_m | Non-stationary | Non-stationary | **I(1)** | 1 |
| remittances_usd_m | Non-stationary | Non-stationary | **I(1)** | 1 |
| tourism_usd_m | Non-stationary | Non-stationary | **I(1)** | 1 |
| cse_net_usd_m | Non-stationary | Non-stationary | **I(1)** | 1 |
| trade_balance_usd_m | Stationary | Non-stationary | **Trend-stationary** | 0* |
| **reserve_change_usd_m** | Stationary | Stationary | **I(0)** | 0 |

**Key Insight:** The target variable (`gross_reserves_usd_m`) is clearly I(1), but its first difference (`reserve_change_usd_m`) is I(0). This confirms differencing will achieve stationarity.

---

## Phase 3: Temporal Dependence Structure

### 3.1 Autocorrelation Analysis

| Variable | ACF(1) | ACF(12) | Persistence | Significant Lags |
|----------|--------|---------|-------------|------------------|
| gross_reserves_usd_m | **0.971** | ~0.8 | Very High | 1-36+ |
| net_usable_reserves_usd_m | **0.978** | ~0.8 | Very High | 1-36+ |
| exports_usd_m | 0.752 | ~0.4 | Medium | 1-24 |
| imports_usd_m | 0.829 | ~0.5 | Medium | 1-24 |
| remittances_usd_m | 0.851 | ~0.5 | Medium | 1-24 |
| tourism_usd_m | 0.898 | ~0.6 | Medium-High | 1-36 |
| cse_net_usd_m | 0.305 | ~0.1 | Low | 1-3 |
| trade_balance_usd_m | 0.681 | ~0.3 | Medium | 1-12 |
| reserve_change_usd_m | **-0.301** | ~0.0 | Low/Negative | 1-2 |

**Key Insight:** Reserves show extremely high persistence (ACF(1)=0.97), indicating strong mean-reversion dynamics. Reserve changes show negative autocorrelation, suggesting mean-reverting behavior in changes.

### 3.2 Ljung-Box Test for Serial Correlation

| Variable | Q(12) | p-value | Q(24) | p-value | Autocorrelated? |
|----------|-------|---------|-------|---------|-----------------|
| gross_reserves_usd_m | Very High | <0.001 | Very High | <0.001 | ✅ Yes |
| exports_usd_m | High | <0.001 | High | <0.001 | ✅ Yes |
| imports_usd_m | High | <0.001 | High | <0.001 | ✅ Yes |
| cse_net_usd_m | Moderate | <0.001 | Moderate | <0.001 | ✅ Yes |
| reserve_change_usd_m | Moderate | <0.001 | Moderate | <0.001 | ✅ Yes |

### 3.3 Seasonal Decomposition (STL)

| Variable | Trend Strength | Seasonal Strength | Dominant Pattern |
|----------|----------------|-------------------|------------------|
| gross_reserves_usd_m | **0.934** | 0.000 | Strong trend, no seasonality |
| exports_usd_m | 0.790 | **0.455** | Moderate trend + seasonality |
| imports_usd_m | 0.824 | 0.301 | Strong trend + weak seasonality |
| tourism_usd_m | 0.875 | **0.593** | Strong trend + strong seasonality |
| cse_net_usd_m | 0.308 | 0.071 | Weak trend, no seasonality |
| reserve_change_usd_m | 0.016 | 0.156 | No trend, weak seasonality |

**Key Insight:** Reserves are trend-dominated with no seasonal pattern. Tourism and exports show significant seasonality that should be modeled.

---

## Phase 4: Volatility & Heteroskedasticity

### 4.1 ARCH-LM Test

**H₀:** No ARCH effects (homoskedastic residuals)

| Variable | ARCH-LM Stat | p-value | ARCH Effects? |
|----------|--------------|---------|---------------|
| **gross_reserves_usd_m** | **48.76** | **<0.001** | ✅ Yes |
| **net_usable_reserves_usd_m** | **36.63** | **<0.001** | ✅ Yes |
| exports_usd_m | 77.43 | <0.001 | ✅ Yes |
| imports_usd_m | 18.55 | 0.100 | ❌ No |
| remittances_usd_m | 21.95 | 0.038 | ✅ Yes |
| tourism_usd_m | 40.91 | <0.001 | ✅ Yes |
| cse_net_usd_m | 17.41 | 0.135 | ❌ No |
| trade_balance_usd_m | 13.82 | 0.313 | ❌ No |
| **reserve_change_usd_m** | **60.47** | **<0.001** | ✅ Yes |

**Key Insight:** Strong ARCH effects in reserves suggest volatility clustering. Consider GARCH modeling or regime-switching.

### 4.2 Volatility Regime Analysis

| Variable | Pre-Crisis Vol (2015-2019) | Crisis Vol (2020-2022) | Ratio |
|----------|---------------------------|------------------------|-------|
| **gross_reserves_usd_m** | ~500 | ~1,075 | **2.15×** |
| **net_usable_reserves_usd_m** | ~500 | ~1,410 | **2.82×** |
| exports_usd_m | ~50 | ~99 | 1.97× |
| imports_usd_m | ~150 | ~260 | 1.73× |
| remittances_usd_m | ~35 | ~105 | **2.99×** |
| tourism_usd_m | ~30 | ~27 | 0.90× (lower) |

**Key Insight:** Reserve volatility more than doubled during crisis. Net usable reserves (excluding PBOC swap) showed even higher volatility increase (2.82×).

---

## Phase 5: Structural Break Detection

### 5.1 Chow Test (Known Break: April 2022)

| Variable | F-Statistic | p-value | Break Confirmed? |
|----------|-------------|---------|------------------|
| **gross_reserves_usd_m** | 0.72 | 0.487 | ❌ **No** |
| **net_usable_reserves_usd_m** | 0.47 | 0.625 | ❌ **No** |
| **exports_usd_m** | **7.97** | **<0.001** | ✅ **Yes** |
| imports_usd_m | 2.26 | 0.107 | ❌ No |
| remittances_usd_m | 1.87 | 0.157 | ❌ No |
| tourism_usd_m | 1.06 | 0.349 | ❌ No |
| cse_net_usd_m | 0.42 | 0.660 | ❌ No |
| **trade_balance_usd_m** | **4.04** | **0.019** | ✅ **Yes** |
| reserve_change_usd_m | 2.84 | 0.061 | ❌ No (marginal) |

### 5.2 CUSUM Test

| Variable | Max CUSUM | Exceeds Bounds? | Instability? |
|----------|-----------|-----------------|--------------|
| gross_reserves_usd_m | 26.72 | ❌ No | No |
| exports_usd_m | **76.82** | ✅ **Yes** | **Yes** |
| cse_net_usd_m | **51.80** | ✅ **Yes** | **Yes** |
| trade_balance_usd_m | **32.33** | ✅ **Yes** | **Yes** |

**⚠️ SURPRISING FINDING:** The Chow test does NOT detect a structural break in reserves at April 2022. This suggests:
1. The crisis affected the **level** but not the **autoregressive dynamics** of reserves
2. The decline was a **gradual process** rather than an abrupt regime shift
3. The PBOC swap may have masked the break in gross reserves

However, **exports** and **trade balance** DO show clear structural breaks, consistent with import compression and export disruption narratives.

---

## Phase 6: Relationship Analysis (Bivariate)

### 6.1 Cross-Correlation (CCF) with Reserves

| Predictor → Reserves | Max CCF | At Lag | Interpretation |
|---------------------|---------|--------|----------------|
| exports_usd_m | 0.153 | 0 | Weak contemporaneous |
| **imports_usd_m** | **0.426** | **0** | Moderate contemporaneous |
| **remittances_usd_m** | **0.528** | **0** | Strongest correlation |
| cse_net_usd_m | 0.076 | 0 | Very weak |

**Key Insight:** Remittances show the strongest correlation with reserves (r=0.53), followed by imports (r=0.43). All relationships are contemporaneous (lag=0), suggesting no predictive lead.

### 6.2 Granger Causality Tests

**H₀:** X does not Granger-cause gross_reserves_usd_m

| X → Reserves | Best Lag | p-value | Granger Causes? |
|--------------|----------|---------|-----------------|
| exports_usd_m | 2 | 0.102 | ❌ No |
| imports_usd_m | 2 | 0.333 | ❌ No |
| remittances_usd_m | 2 | 0.165 | ❌ No |
| cse_net_usd_m | 2 | 0.078 | ❌ No (marginal) |

**⚠️ KEY FINDING:** None of the BoP flow variables significantly Granger-cause reserves. This suggests:
1. Reserves respond to **contemporaneous** shocks, not lagged predictors
2. Reserve changes may be driven by **policy decisions** (CBSL intervention) rather than BoP flows
3. Consider **VAR models** with simultaneous relationships rather than predictive causality

---

## Recommendations for Modeling

### Based on Diagnostic Results:

1. **Differencing Required:** Use `reserve_change_usd_m` or first differences of `gross_reserves_usd_m` for modeling

2. **High Persistence:** Include AR(1) or AR(2) terms; expect slow adjustment

3. **ARCH Effects:** Consider:
   - GARCH(1,1) for volatility modeling
   - Regime-switching models (SRVAR or Markov-Switching)
   - Heteroskedasticity-robust standard errors

4. **No Clear Break at Default:** The April 2022 crisis may need:
   - Earlier break point testing (2021 or earlier)
   - Gradual transition models instead of sudden breaks
   - Testing PBOC swap date (March 2021) as break point

5. **Contemporaneous Relationships:** Use:
   - Structural VAR (SVAR) with contemporaneous restrictions
   - Error Correction Models (ECM) if cointegration found
   - Simultaneous equations rather than Granger causality

6. **Seasonality:** Account for tourism and export seasonality in BoP-based models

---

## Test Log

| Timestamp | Test | Status | Notes |
|-----------|------|--------|-------|
| 2026-02-03 09:53 | Phase 2 Stationarity | ✅ Complete | 11 variables tested |
| 2026-02-03 09:53 | Phase 3 Temporal | ✅ Complete | ACF/PACF, Ljung-Box, STL |
| 2026-02-03 09:53 | Phase 4 Volatility | ✅ Complete | ARCH-LM, rolling volatility |
| 2026-02-03 09:54 | Phase 5 Breaks | ✅ Complete | Chow, CUSUM |
| 2026-02-03 09:54 | Phase 6 Relationships | ✅ Complete | CCF, Granger |

---

## Files Generated

- `data/diagnostics/diagnostic_results.json` - Full results
- `data/diagnostics/integration_summary.csv` - Stationarity summary
- `data/diagnostics/arch_summary.csv` - ARCH test results
- `data/diagnostics/chow_test_summary.csv` - Structural break results
- `data/diagnostics/granger_causality_summary.csv` - Causality results
