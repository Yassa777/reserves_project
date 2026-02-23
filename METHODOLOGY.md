# Methodology

This document describes the complete methodology for forecasting Sri Lanka's foreign exchange reserves using a Markov-Switching VAR approach with comprehensive model comparison.

## Table of Contents

1. [Data Pipeline](#1-data-pipeline)
2. [Diagnostic Framework](#2-diagnostic-framework)
3. [Variable Set Specifications](#3-variable-set-specifications)
4. [Forecasting Models](#4-forecasting-models)
5. [Evaluation Framework](#5-evaluation-framework)
6. [Statistical Testing](#6-statistical-testing)
7. [Scenario Analysis](#7-scenario-analysis)
8. [Robustness Checks](#8-robustness-checks)

---

## 1. Data Pipeline

### 1.1 Data Sources

| Source | Variables | Frequency |
|--------|-----------|-----------|
| Central Bank of Sri Lanka | Gross reserves, net usable reserves | Monthly |
| CBSL External Sector | Exports, imports, remittances, tourism | Monthly |
| Monetary Authority | USD/LKR exchange rate, M2 money supply | Monthly |
| Colombo Stock Exchange | Net foreign portfolio flows | Monthly |

### 1.2 Dataset Construction

**Output**: `data/merged/reserves_forecasting_panel.csv`
- **Period**: January 2005 - December 2025 (252 observations)
- **Frequency**: Monthly (month-start indexing)

**Core Variables**:

| Variable | Description | Units |
|----------|-------------|-------|
| `gross_reserves_usd_m` | Foreign exchange reserves (TARGET) | USD millions |
| `exports_usd_m` | Merchandise exports | USD millions |
| `imports_usd_m` | Merchandise imports | USD millions |
| `remittances_usd_m` | Worker remittances | USD millions |
| `tourism_usd_m` | Tourism receipts | USD millions |
| `usd_lkr` | Exchange rate | LKR per USD |
| `m2_usd_m` | Broad money (USD equivalent) | USD millions |
| `trade_balance_usd_m` | Exports minus imports | USD millions |
| `cse_net_usd_m` | Net CSE foreign flows | USD millions |

**Missing Data Strategy**:
- Forward-fill with 3-month limit
- Drop remaining missing values
- Minimum 20 observations required per series

### 1.3 Train/Validation/Test Split

```
Training:    2005-01 to 2019-12  (180 months)
Validation:  2020-01 to 2022-12  (36 months)
Test:        2023-01 to 2025-12  (36 months)
```

This split ensures the crisis period (2020-2022) and post-default period (2023+) are held out for true out-of-sample evaluation.

---

## 2. Diagnostic Framework

A 9-phase diagnostic pipeline characterizes the time-series properties of the data.

### Phase 1: Data Quality
- Coverage analysis (non-null observations)
- Constant series detection
- Minimum observation thresholds

### Phase 2: Stationarity Testing

| Test | Null Hypothesis | Purpose |
|------|-----------------|---------|
| ADF | Unit root exists | Confirm I(1) vs I(0) |
| KPSS | Series is stationary | Complement ADF |
| Zivot-Andrews | Unit root with break | Allow structural break |

**Decision Rule**: Series is I(0) if ADF p < 0.05 AND KPSS p > 0.05

**Finding**: Target variable `gross_reserves_usd_m` is I(1) - requires first-differencing.

### Phase 3: Temporal Dynamics
- ACF/PACF analysis
- Lag significance testing
- **Finding**: ACF(1) = 0.97 indicates high persistence

### Phase 4: Volatility (ARCH Effects)
- ARCH-LM test for conditional heteroskedasticity
- **Finding**: Significant ARCH effects present (p < 0.05)

### Phase 5: Structural Breaks

| Test | Method | Application |
|------|--------|-------------|
| Chow | Known breakpoint | Test April 2022 (sovereign default) |
| CUSUM | Endogenous | Recursive residual analysis |

**Surprising Finding**: No statistically significant break at April 2022 default date.

### Phase 6: Granger Causality
- VAR(p) specification with AIC lag selection
- Tests predictive content of BoP flows for reserves
- **Finding**: No significant Granger causality from BoP components

### Phase 7: Cointegration

| Test | System | Output |
|------|--------|--------|
| Engle-Granger | Pairwise | Cointegration t-statistics |
| Johansen | Multivariate | Cointegration rank r |

**ECM Suitability**: Error-correction term coefficient < 0 and significant

### Phase 8: Structural VAR
- Cholesky decomposition for identification
- Impulse response analysis
- Sign restriction validation

### Phase 9: Multiple Breaks (Bai-Perron)
- Endogenous multiple break detection
- Tests stability across the full sample

---

## 3. Variable Set Specifications

Five variable sets enable systematic comparison of specification choices.

### 3.1 Parsimonious (Recommended)

```python
variables = ["gross_reserves_usd_m", "trade_balance_usd_m", "usd_lkr"]
```

**Rationale**: Minimal economically-motivated set. Trade balance captures current account; exchange rate captures valuation and intervention.

### 3.2 Balance of Payments (BoP)

```python
variables = ["gross_reserves_usd_m", "exports_usd_m", "imports_usd_m",
             "remittances_usd_m", "tourism_usd_m"]
```

**Rationale**: Disaggregated current account components. Excludes exchange rate to avoid BoP identity endogeneity.

### 3.3 Monetary Policy

```python
variables = ["gross_reserves_usd_m", "usd_lkr", "m2_usd_m"]
```

**Rationale**: Policy intervention channel. CBSL's USD operations affect both reserves and money supply.

### 3.4 PCA (Data-Driven)

```python
source_vars = ["exports", "imports", "remittances", "tourism",
               "usd_lkr", "m2_usd_m", "cse_net", "trade_balance"]
n_components = 3  # Fit on training data only
variables = ["gross_reserves_usd_m", "PC1", "PC2", "PC3"]
```

**Rationale**: Dimensionality reduction prevents overfitting while retaining information.

### 3.5 Full (Kitchen Sink)

```python
variables = ["gross_reserves_usd_m"] + [all 8 predictors]
```

**Rationale**: Overfitting benchmark. Should underperform parsimonious sets out-of-sample.

---

## 4. Forecasting Models

### 4.1 Benchmark Models

#### Naive (Random Walk)
```
ŷ_{t+h} = y_t
```
Assumes reserves follow a random walk. Baseline for comparison.

#### Seasonal Naive
```
ŷ_{t+h} = y_{t-12+h}
```
Uses same month from previous year.

### 4.2 Classical Time Series

#### ARIMA with Exogenous Variables
- **Specification**: SARIMAX(p, 1, q) with exogenous regressors
- **Order Selection**: AIC-based grid search over p, q ∈ {0, 1, 2, 3}
- **Estimation**: Maximum likelihood via Kalman filter

#### Vector Error Correction Model (VECM)
- **Specification**:
  ```
  Δy_t = αβ'y_{t-1} + Γ₁Δy_{t-1} + ... + Γ_{k-1}Δy_{t-k+1} + ε_t
  ```
- **Cointegration Rank**: Determined by Johansen trace test
- **Lag Order**: AIC/BIC selection on VAR in levels

### 4.3 Bayesian VAR (BVAR)

**Prior**: Minnesota (Litterman) prior
- Own first lag: prior mean = 1 (persistence)
- Cross-variable and higher lags: prior mean = 0
- Tightness parameter λ₁ = 0.2

**Estimation**: Gibbs sampling
- 5,000 posterior draws
- 1,000 burn-in samples

**Forecasting**: Posterior predictive distribution
- Point forecast: posterior mean
- Intervals: posterior quantiles

### 4.4 Markov-Switching VAR (MS-VAR)

**Specification**: Two-regime VAR with state-dependent parameters
```
y_t = μ(s_t) + A₁(s_t)y_{t-1} + ... + Aₚ(s_t)y_{t-p} + ε_t(s_t)
s_t ∈ {0, 1}  (low-volatility, high-volatility regimes)
```

**Estimation**: Expectation-Maximization (EM) algorithm
- E-step: Forward-backward algorithm for regime probabilities
- M-step: Regime-weighted least squares for coefficients

**Transition Matrix**:
```
P = [p₀₀  p₀₁]    where pᵢⱼ = P(sₜ = j | sₜ₋₁ = i)
    [p₁₀  p₁₁]
```

**Regime Initialization**: Volatility-based
- Rolling 6-month standard deviation
- High-volatility regime if rolling_std > median(rolling_std)

**Forecasting Modes**:
1. **Free**: Regime probabilities evolve according to transition matrix
2. **Locked**: Fix regime probabilities at last observed value
3. **Path**: Specify explicit regime sequence

### 4.5 Machine Learning Models (Optional)

#### XGBoost
- **Features**: Lags (1, 2, 3, 6, 12), rolling means/stds, momentum
- **Hyperparameters**: max_depth=4, learning_rate=0.1, n_estimators=100

#### LSTM
- **Architecture**: 2-layer LSTM (128, 64 units) with dropout
- **Training**: MinMax scaling, early stopping on validation loss

---

## 5. Evaluation Framework

### 5.1 Rolling-Origin Evaluation

```
For each origin t in [TRAIN_END, TEST_END]:
    1. Train model on [START, t]
    2. Generate forecasts for horizons h ∈ {1, 3, 6, 12}
    3. Record forecast errors
    4. If (t - last_refit) >= refit_interval: refit model
```

**Refit Interval**: Every 12 months (expanding window)

### 5.2 Point Forecast Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| MAE | mean(\|yₜ - ŷₜ\|) | Average absolute error |
| RMSE | √mean((yₜ - ŷₜ)²) | Penalizes large errors |
| MAPE | mean(\|yₜ - ŷₜ\| / \|yₜ\|) | Percentage error |
| MASE | MAE / mean(\|Δy\|) | Scaled by naive baseline |

### 5.3 Directional Accuracy

```
Direction Accuracy = mean(sign(Δŷₜ) == sign(Δyₜ))
```

Critical for policy: correctly predicting whether reserves will rise or fall.

### 5.4 Asymmetric Loss

```
L(e) = { 2|e|  if e < 0 (underprediction)
       { 1|e|  if e ≥ 0 (overprediction)
```

**Rationale**: Underpredicting reserves is more costly for policymakers (false sense of security).

### 5.5 Density Forecast Metrics

| Metric | Description |
|--------|-------------|
| CRPS | Continuous Ranked Probability Score |
| Log Score | Log-likelihood of predictive density at realized value |

---

## 6. Statistical Testing

### 6.1 Diebold-Mariano Test

**Purpose**: Test equal predictive accuracy between two models

**Hypotheses**:
```
H₀: E[L(e₁ₜ)] = E[L(e₂ₜ)]  (equal expected loss)
H₁: E[L(e₁ₜ)] ≠ E[L(e₂ₜ)]  (different expected loss)
```

**Test Statistic**:
```
DM = d̄ / √(V̂(d̄))
```
where d̄ = mean(L(e₁ₜ) - L(e₂ₜ))

**Variance Estimation**: Newey-West HAC with bandwidth h-1

**Small-Sample Correction**: Harvey-Leybourne-Newbold (1997)
- Uses t-distribution with T-1 degrees of freedom

### 6.2 Model Confidence Set (MCS)

**Purpose**: Identify set of statistically indistinguishable best models

**Algorithm** (Hansen, Lunde & Nason, 2011):
1. Start with all models in the set
2. Test null hypothesis of equal predictive ability
3. If rejected: eliminate worst-performing model
4. Repeat until EPA cannot be rejected
5. Surviving models form the 90% Model Confidence Set

**Bootstrap**: Stationary block bootstrap (1,000 replications)

### 6.3 Forecast Encompassing

**Purpose**: Test if one forecast encompasses all information in another

**Regression**:
```
yₜ = α·ŷ₁ₜ + (1-α)·ŷ₂ₜ + εₜ
H₀: α = 1  (Model 1 encompasses Model 2)
```

---

## 7. Scenario Analysis

### 7.1 Framework

Conditional forecasting using MS-VARX with specified exogenous variable paths.

**Shock Representation**: Multiplicative factors
- 1.0 = baseline (no change)
- 0.85 = 15% decline
- 1.20 = 20% increase

**Shock Profiles**:
- **Ramp**: Gradual linear adjustment over horizon
- **Step**: Immediate permanent shift
- **Impulse**: One-time shock that fades

### 7.2 Policy Scenarios

| Scenario | Key Shocks | Economic Interpretation |
|----------|------------|------------------------|
| **Baseline** | None | Current trajectory |
| **LKR Depreciation 10%** | usd_lkr × 1.10 | Moderate FX pressure |
| **LKR Depreciation 20%** | usd_lkr × 1.20 | Severe currency crisis |
| **Export Shock** | exports × 0.85, trade_bal × 0.70 | Global demand collapse |
| **Remittance Decline** | remittances × 0.80 | Gulf employment crisis |
| **Tourism Recovery** | tourism × 1.25 | Post-COVID boom |
| **Oil Price Shock** | imports × 1.15, trade_bal × 0.75 | Energy price surge |
| **IMF Delay** | usd_lkr × 1.08, imports × 0.95 | Disbursement postponed |
| **Combined Adverse** | Multiple adverse shocks | Stress test |
| **Combined Upside** | Multiple favorable shocks | Best-case scenario |

### 7.3 Conditional Forecasting Method

```
1. Fit MS-VARX on historical data
2. Generate baseline exogenous paths (naive or ARIMA extrapolation)
3. Apply scenario shocks to exogenous paths
4. Solve for endogenous variables conditional on shocked paths
5. Aggregate regime-weighted forecasts
```

---

## 8. Robustness Checks

### 8.1 Subsample Analysis

| Period | Dates | Economic Context |
|--------|-------|------------------|
| Pre-Crisis | 2012-2018 | Stable growth |
| Crisis | 2019-2022 | COVID + default |
| Post-Default | 2023-2025 | IMF program |
| COVID Only | 2020-2021 | Pandemic shock |

### 8.2 Variable Set Sensitivity

Compare model rankings across all 5 variable sets to assess:
- Stability of MS-VAR superiority
- Sensitivity to specification choices
- Overfitting in larger sets

### 8.3 Horizon Analysis

Evaluate performance separately for each horizon:
- h=1: Near-term (operational)
- h=3: Short-term (tactical)
- h=6: Medium-term (strategic)
- h=12: Long-term (planning)

### 8.4 Regime Robustness

- Test 2 vs 3 regimes
- Alternative regime initialization methods
- Sensitivity to EM convergence criteria

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         RAW DATA                                │
│   (CBSL, Trade Statistics, Monetary Authority, CSE)            │
└─────────────────────────┬───────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA MERGING                                  │
│   reserves_forecasting_panel.csv (252 obs × 11 vars)           │
└─────────────────────────┬───────────────────────────────────────┘
                          ▼
         ┌────────────────┴────────────────┐
         ▼                                 ▼
┌─────────────────┐              ┌─────────────────────────┐
│   DIAGNOSTICS   │              │   VARIABLE SETS         │
│   (9 Phases)    │              │   (5 specifications)    │
│   - Stationarity│              │   - Parsimonious        │
│   - Cointegration│             │   - BoP                 │
│   - Breaks      │              │   - Monetary            │
│   - ARCH        │              │   - PCA                 │
└─────────────────┘              │   - Full                │
                                 └───────────┬─────────────┘
                                             ▼
                          ┌──────────────────┴──────────────────┐
                          ▼                                     ▼
              ┌───────────────────┐              ┌───────────────────┐
              │  FORECASTING      │              │  SCENARIO         │
              │  MODELS           │              │  ANALYSIS         │
              │  - Naive          │              │  - MS-VARX        │
              │  - ARIMA          │              │  - 10 scenarios   │
              │  - VECM           │              │  - Conditional    │
              │  - BVAR           │              │    forecasts      │
              │  - MS-VAR         │              └─────────┬─────────┘
              │  - XGBoost        │                        │
              │  - LSTM           │                        │
              └─────────┬─────────┘                        │
                        ▼                                  │
              ┌───────────────────┐                        │
              │  ROLLING-ORIGIN   │                        │
              │  EVALUATION       │                        │
              │  - 4 horizons     │                        │
              │  - 5 metrics      │                        │
              └─────────┬─────────┘                        │
                        ▼                                  │
              ┌───────────────────┐                        │
              │  STATISTICAL      │                        │
              │  TESTS            │                        │
              │  - Diebold-Mariano│                        │
              │  - MCS            │                        │
              │  - Encompassing   │                        │
              └─────────┬─────────┘                        │
                        ▼                                  ▼
              ┌─────────────────────────────────────────────┐
              │              ROBUSTNESS CHECKS              │
              │  - Subsample analysis                       │
              │  - Variable set sensitivity                 │
              │  - Horizon analysis                         │
              └─────────────────────────────────────────────┘
                                  ▼
              ┌─────────────────────────────────────────────┐
              │           PUBLICATION OUTPUTS               │
              │  - LaTeX tables                             │
              │  - Figures                                  │
              │  - Model rankings                           │
              └─────────────────────────────────────────────┘
```

---

## Key References

| Method | Reference |
|--------|-----------|
| ADF Test | Dickey & Fuller (1979); MacKinnon (1996) |
| KPSS Test | Kwiatkowski et al. (1992) |
| Johansen Cointegration | Johansen (1991) |
| VECM | Engle & Granger (1987) |
| Minnesota Prior (BVAR) | Litterman (1986); Doan et al. (1984) |
| Markov-Switching | Hamilton (1989); Krolzig (1997) |
| Diebold-Mariano Test | Diebold & Mariano (1995) |
| HLN Correction | Harvey, Leybourne & Newbold (1997) |
| Model Confidence Set | Hansen, Lunde & Nason (2011) |
| Bai-Perron Breaks | Bai & Perron (2003) |

---

## File Structure Reference

```
reserves_project/
├── reserves_project/
│   ├── config/
│   │   ├── paths.py          # Directory configuration
│   │   └── varsets.py        # Variable set definitions
│   ├── diagnostics/          # Phase 1-9 implementations
│   ├── models/
│   │   ├── bvar.py           # Bayesian VAR
│   │   ├── ms_switching_var.py # MS-VAR with EM
│   │   └── ml_models.py      # XGBoost, LSTM
│   ├── eval/
│   │   ├── metrics.py        # Forecast metrics
│   │   ├── diebold_mariano.py
│   │   ├── model_confidence_set.py
│   │   └── unified_evaluator.py
│   ├── scenarios/
│   │   ├── definitions.py    # Scenario specifications
│   │   └── msvarx.py         # Conditional forecasting
│   ├── robustness/           # Robustness analysis modules
│   └── pipelines/            # Main execution scripts
└── data/
    ├── merged/               # Processed datasets
    ├── diagnostics/          # Diagnostic outputs
    ├── forecast_prep/        # Model-ready datasets
    └── outputs/              # Run-specific results
```
