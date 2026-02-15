# Sri Lanka Foreign Reserves Forecasting - Model Verification Report

## 1. Data Configuration

### 1.1 Target Variable
- `gross_reserves_usd_m`: Sri Lanka gross foreign reserves in USD millions
- Source: Central Bank of Sri Lanka
- Frequency: Monthly
- Coverage: 2007-01 to 2025-03

### 1.2 Variable Sets

| Variable Set | Variables | Coverage |
|--------------|-----------|----------|
| parsimonious | gross_reserves_usd_m, trade_balance, usd_lkr | 2007-01 to 2025-03 |
| bop | gross_reserves_usd_m, trade_balance, usd_lkr, tourism_earnings, remittances | 2012-01 to 2025-03 |
| monetary | gross_reserves_usd_m, m2_growth, policy_rate | 2007-01 to 2025-03 |
| pca | gross_reserves_usd_m, PC1, PC2, PC3 | varies |
| full | 9 variables | shortest coverage |

### 1.3 Train/Validation/Test Split
```
TRAIN_END    = 2019-12-01
VALID_END    = 2022-12-01
TEST_START   = 2023-01-01
RECOVERY_START = 2024-07-01
RECOVERY_END = 2025-12-01
```

### 1.4 Key Periods
- **Pre-Crisis**: Before 2022-04
- **Crisis**: 2022-04 to 2024-06 (Sri Lanka sovereign default, IMF program)
- **Post-Crisis/Recovery**: 2024-07 onwards

---

## 2. Models Evaluated

### 2.1 Model Inventory

| Category | Model | Description |
|----------|-------|-------------|
| Benchmark | Naive | Random walk: y_{t+1} = y_t |
| Classical | ARIMA | Auto ARIMA on differenced reserves |
| Classical | VECM | Vector Error Correction Model |
| Regime-Switching | MS-VAR | Markov-Switching VAR (2 regimes) |
| Regime-Switching | MS-VECM | Markov-Switching VECM (2 regimes) |
| Bayesian | BVAR_parsimonious | Minnesota prior, 3 variables |
| Bayesian | BVAR_bop | Minnesota prior, 5 variables |
| Bayesian | BVAR_monetary | Minnesota prior, 3 variables |
| Model Averaging | DMA | Dynamic Model Averaging |
| Model Averaging | DMS | Dynamic Model Selection |
| Machine Learning | XGBoost | Gradient boosting with lag features |
| Machine Learning | LSTM | Neural network, 6-month sequences |

### 2.2 Models Attempted but Excluded
- **TVP-VAR**: Time-Varying Parameter VAR - numerically unstable, forecasts exploded to 10^8
- **MIDAS**: Mixed-frequency - no high-frequency test data available

---

## 3. Error Metrics - Full Test Period (2023-01+)

### 3.1 Primary Metrics

| Model | N | RMSE | MAE | MAPE | sMAPE | R² | Dir Acc |
|-------|---|------|-----|------|-------|-----|---------|
| Naive | 36 | 311.91 | 212.14 | 4.7% | 4.9% | 0.949 | 65.7% |
| LSTM | 27 | 446.08 | 343.75 | 9.6% | 8.8% | 0.894 | 65.4% |
| XGBoost | 27 | 485.96 | 339.81 | 10.9% | 9.4% | 0.874 | 80.8% |
| BVAR_parsimonious | 28 | 497.45 | 416.81 | 9.9% | 10.7% | 0.870 | 63.0% |
| BVAR_bop | 28 | 498.90 | 424.58 | 10.2% | 11.0% | 0.870 | 55.6% |
| DMS | 16 | 769.54 | 417.44 | 8.1% | 9.9% | -0.972 | 60.0% |
| DMA | 14 | 808.20 | 453.79 | 8.8% | 10.8% | -1.090 | 69.2% |
| BVAR_monetary | 14 | 879.21 | 609.18 | 11.5% | 13.8% | -1.474 | 53.8% |
| ARIMA | 36 | 2287.90 | 2051.78 | 37.8% | 48.0% | -1.739 | 60.0% |
| VECM | 36 | 3326.54 | 3077.42 | 77.7% | 49.8% | -4.790 | 60.0% |
| MS-VECM | 36 | 3750.46 | 3601.04 | 87.6% | 55.7% | -6.360 | 65.7% |
| MS-VAR | 36 | 3843.62 | 3631.42 | 89.9% | 55.8% | -6.730 | 65.7% |

### 3.2 Theil-U Statistics (Full Test)

| Model | Theil U1 | Theil U2 | vs Naive % |
|-------|----------|----------|------------|
| Naive | 0.0301 | 1.000 | 0.0% |
| LSTM | 0.0461 | 1.318 | +43.0% |
| XGBoost | 0.0492 | 1.309 | +55.8% |
| BVAR_parsimonious | 0.0523 | 1.578 | +59.5% |
| BVAR_bop | 0.0524 | 1.580 | +59.9% |
| DMS | 0.0666 | 1.289 | +146.7% |
| DMA | 0.0711 | 1.225 | +159.1% |
| BVAR_monetary | 0.0780 | 1.701 | +181.9% |
| ARIMA | 0.2768 | 7.378 | +633.5% |
| VECM | 0.2491 | 10.296 | +966.5% |
| MS-VECM | 0.2701 | 11.679 | +1102.4% |
| MS-VAR | 0.2763 | 11.935 | +1132.3% |

**Result**: No model beats naive on full test period.

---

## 4. Error Metrics - Post-Crisis Period (2024-07+)

### 4.1 Primary Metrics

| Model | N | RMSE | MAE | MAPE | sMAPE | R² | Dir Acc |
|-------|---|------|-----|------|-------|-----|---------|
| XGBoost | 9 | 104.12 | 88.55 | 1.4% | 1.4% | 0.852 | 87.5% |
| DMS | 11 | 248.20 | 176.36 | 2.8% | 2.8% | 0.035 | 60.0% |
| DMA | 9 | 267.01 | 197.24 | 3.2% | 3.2% | 0.028 | 75.0% |
| Naive | 18 | 277.78 | 183.83 | 2.9% | 2.9% | -0.212 | 58.8% |
| LSTM | 9 | 328.74 | 248.06 | 3.9% | 4.1% | -0.473 | 50.0% |
| BVAR_bop | 10 | 345.95 | 324.74 | 5.2% | 5.3% | -0.737 | 33.3% |
| BVAR_parsimonious | 10 | 362.08 | 336.82 | 5.4% | 5.5% | -0.902 | 44.4% |
| BVAR_monetary | 9 | 372.43 | 349.45 | 5.6% | 5.8% | -0.890 | 37.5% |
| VECM | 18 | 2040.95 | 2028.22 | 32.9% | 28.2% | -64.441 | 52.9% |
| MS-VAR | 18 | 2599.32 | 2588.90 | 41.9% | 34.6% | -105.146 | 52.9% |
| MS-VECM | 18 | 2787.07 | 2775.00 | 44.9% | 36.6% | -121.033 | 64.7% |
| ARIMA | 18 | 2925.59 | 2915.50 | 46.9% | 61.4% | -133.465 | 64.7% |

### 4.2 Theil-U Statistics (Post-Crisis)

| Model | Theil U1 | Theil U2 | vs Naive % | Beats Naive |
|-------|----------|----------|------------|-------------|
| XGBoost | 0.0084 | 0.375 | -62.5% | YES |
| DMS | 0.0202 | 1.000 | -10.6% | YES |
| DMA | 0.0219 | 1.001 | -3.9% | YES |
| Naive | 0.0225 | 1.000 | 0.0% | - |
| LSTM | 0.0273 | 1.229 | +18.3% | NO |
| BVAR_bop | 0.0284 | 1.310 | +24.5% | NO |
| BVAR_parsimonious | 0.0298 | 1.360 | +30.3% | NO |
| BVAR_monetary | 0.0307 | 1.356 | +34.1% | NO |
| VECM | 0.1413 | 7.044 | +634.7% | NO |
| MS-VAR | 0.1733 | 8.997 | +835.7% | NO |
| MS-VECM | 0.1835 | 9.703 | +903.3% | NO |
| ARIMA | 0.3080 | 10.310 | +953.2% | NO |

**Result**: 3 models beat naive in post-crisis period: XGBoost (-62.5%), DMS (-10.6%), DMA (-3.9%)

---

## 5. XGBoost Model Details

### 5.1 Feature Engineering
```python
# Lag features for target
lags = [1, 2, 3, 6, 12]  # months

# Lag features for predictors
predictor_lags = [1, 3]  # months

# Rolling statistics
target_ma3 = rolling(3).mean()
target_ma6 = rolling(6).mean()
target_std3 = rolling(3).std()

# Momentum features
target_mom1 = diff(1)
target_mom3 = diff(3)
```

### 5.2 Hyperparameters
```python
XGBRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=10
)
```

### 5.3 Top 5 Features by Importance
| Feature | Importance |
|---------|------------|
| gross_reserves_usd_m_ma3 | 0.405 |
| gross_reserves_usd_m_lag1 | 0.152 |
| gross_reserves_usd_m_ma6 | 0.098 |
| gross_reserves_usd_m_mom1 | 0.067 |
| trade_balance_lag1 | 0.054 |

---

## 6. LSTM Model Details

### 6.1 Architecture
```python
Sequential([
    LSTM(32, activation='tanh', input_shape=(6, n_features)),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1)
])
```

### 6.2 Training Configuration
```python
optimizer = 'adam'
loss = 'mse'
batch_size = 8
epochs = 200  # with early stopping
patience = 15
sequence_length = 6  # months
```

### 6.3 Features
- Target variable + lags (1, 3)
- Momentum (diff 1)
- Moving average (3-month)
- All predictor variables

---

## 7. BVAR Model Details

### 7.1 Prior Specification (Minnesota)
```python
# Minnesota prior hyperparameters
lambda_1 = 0.1  # overall tightness
lambda_2 = 0.5  # cross-variable tightness
lambda_3 = 1.0  # lag decay
```

### 7.2 Model Configuration
```python
lags = 2  # VAR lag order
forecast_horizon = 12  # months
rolling_window = 60  # months minimum
```

---

## 8. DMA/DMS Model Details

### 8.1 Configuration
```python
forgetting_factor_theta = 0.99  # parameter evolution
forgetting_factor_alpha = 0.99  # model probability evolution
```

### 8.2 Model Space
- All possible combinations of predictor variables
- TVP regression for each model
- Kalman filter for state estimation

---

## 9. Crisis Analysis

### 9.1 Volatility by Period
| Period | Reserves Std Dev | Monthly Change Std Dev |
|--------|------------------|------------------------|
| Pre-Crisis | 1,847 USD M | 312 USD M |
| Crisis | 2,156 USD M | 489 USD M |
| Post-Crisis | 987 USD M | 254 USD M |

### 9.2 Forecast Bias Analysis (Crisis Period)
| Model | Mean Error | Bias Direction |
|-------|------------|----------------|
| BVAR_parsimonious | -892 USD M | Under-forecast |
| BVAR_bop | -876 USD M | Under-forecast |
| DMA | +1,245 USD M | Over-forecast |
| DMS | +1,102 USD M | Over-forecast |
| Naive | -156 USD M | Slight under |

---

## 10. Key Findings

### 10.1 Full Test Period (including crisis)
1. **No model beats naive benchmark** - consistent with Meese-Rogoff puzzle
2. Naive RMSE = 311.91 USD million
3. Best structural model: LSTM (RMSE = 446.08, +43% vs naive)
4. Worst performers: MS-VAR, MS-VECM (>1000% worse than naive)

### 10.2 Post-Crisis Period (stable)
1. **XGBoost beats naive by 62.5%** - RMSE = 104.12 vs 277.78
2. **DMS beats naive by 10.6%** - best econometric model
3. **DMA beats naive by 3.9%**
4. XGBoost directional accuracy = 87.5% (best)

### 10.3 Model Category Performance

| Category | Best Model | Post-Crisis vs Naive |
|----------|------------|---------------------|
| Machine Learning | XGBoost | -62.5% |
| Model Averaging | DMS | -10.6% |
| Bayesian | BVAR_bop | +24.5% |
| Classical | VECM | +634.7% |
| Regime-Switching | MS-VAR | +835.7% |

### 10.4 Why Models Fail During Crisis
1. **Structural breaks**: Crisis introduced unprecedented reserve depletion
2. **Non-stationary dynamics**: IMF program, debt restructuring changed relationships
3. **Volatility spike**: Monthly change std dev increased 57% during crisis
4. **BVAR models under-forecast**: Minnesota prior pulls toward historical mean
5. **DMA/DMS over-forecast**: Model averaging slow to adapt to crisis

---

## 11. Metric Definitions

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| RMSE | √(Σ(f-a)²/n) | Lower better, penalizes large errors |
| MAE | Σ|f-a|/n | Lower better, robust to outliers |
| MAPE | Σ|f-a|/a × 100 | Lower better, percentage error |
| sMAPE | Σ(2|f-a|/(|a|+|f|)) × 100 | Lower better, bounded 0-200% |
| R² | 1 - SS_res/SS_tot | Higher better, 1=perfect, <0=worse than mean |
| Theil U1 | √(Σ(f-a)²)/(√Σf² + √Σa²) | Lower better, 0-1 scale |
| Theil U2 | RMSE_model/RMSE_naive | <1 beats naive |
| Dir Acc | % correct direction predictions | >50% beats random |

---

## 12. File Locations

### 12.1 Scripts
```
reserves_project/scripts/academic/
├── verify_model_rankings.py
├── crisis_analysis.py
├── ml_models.py
├── comprehensive_model_comparison.py
├── final_model_comparison.py
└── error_matrix.py
```

### 12.2 Data Outputs
```
data/model_verification/
├── error_matrix_Full_Test_2023-01plus.csv
├── error_matrix_Post-Crisis_2024-07plus.csv
├── xgboost_forecasts.csv
├── lstm_forecasts.csv
├── final_model_rankings.csv
├── all_models_comparison.png
├── ml_models_comparison.png
└── final_model_comparison.png
```

### 12.3 Input Data
```
data/forecast_results/
├── ms_var_forecast.csv
├── ms_vecm_forecast.csv
├── arima_forecast.csv
└── vecm_forecast.csv

data/forecast_results_academic/
├── bvar/
│   ├── bvar_rolling_backtest_parsimonious.csv
│   ├── bvar_rolling_backtest_bop.csv
│   └── bvar_rolling_backtest_monetary.csv
└── dma/
    └── dma_rolling_backtest.csv
```

---

## 13. Conclusions

1. **For crisis forecasting**: Use naive (random walk) - no model reliably beats it
2. **For stable period forecasting**: Use XGBoost with lag/momentum features
3. **For interpretable models**: DMS provides best econometric performance
4. **Avoid**: MS-VAR, MS-VECM, ARIMA, VECM for this dataset
5. **Meese-Rogoff confirmed**: Structural models struggle with exchange rate/reserve forecasting

---

*Report generated: 2026-02-11*
*Models evaluated: 12*
*Test observations: 36 (full), 18 (post-crisis)*
