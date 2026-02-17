# Data Quality and Feature Engineering Methodology

## 1. Data Sources

| Source File | Status | Rows | Date Range | Missing Values |
|-------------|--------|------|------------|----------------|
| historical_reserves.csv | OK | 240 | 2005-01 to 2024-12 | 0 |
| historical_fx.csv | OK | 240 | 2005-01 to 2024-12 | 0 |
| monthly_imports_usd.csv | OK | 225 | 2007-01 to 2025-11 | 0 |
| monthly_exports_usd.csv | OK | 224 | 2007-01 to 2025-11 | 0 |
| remittances_monthly.csv | OK | 202 | 2009-01 to 2025-10 | 0 |
| tourism_earnings_monthly.csv | OK | 192 | 2009-01 to 2024-12 | 0 |
| monetary_aggregates_monthly.csv | OK | 355 | 1995-12 to 2025-09 | 0 |
| D12_reserves.csv | OK | 18 | 2013-11 to 2015-04 | 0 |


## 2. Missing Data Strategy

Configuration from `config.py`:
```python
MISSING_STRATEGY = {
    "method": "ffill_limit",
    "limit": 3,
    "drop_remaining": True,
}
```

**Process:**
1. Forward-fill missing values with a maximum limit of 3 consecutive periods
2. Drop any remaining rows with missing values
3. Applied separately to each variable set

## 3. Train/Validation/Test Split

| Split | End Date | Purpose |
|-------|----------|---------|
| Train | 2019-12-01 | Model estimation |
| Validation | 2022-12-01 | Hyperparameter tuning |
| Test | 2025-03-01+ | Out-of-sample evaluation |

## 4. Variable Set Quality Summary

| Variable Set | Variables | Total Obs | Missing | Zeros | Outliers |
|--------------|-----------|-----------|---------|-------|----------|
| parsimonious | 3 | 219 | 0 | 0 | 0 |
| bop | 5 | 195 | 0 | 8 | 3 |
| monetary | 3 | 221 | 0 | 0 | 9 |
| pca | 4 | 130 | 0 | 0 | 3 |
| full | 7 | 137 | 0 | 0 | 7 |


## 5. Feature Engineering

### 5.1 ARIMA Dataset
- Target transformations: diff(1), log, log_diff(1), pct_change
- Exogenous variables: as specified per variable set

### 5.2 VECM Dataset
- Level variables for cointegration analysis
- Johansen test for cointegrating rank
- Error correction term (ECT) computed from first cointegrating vector
- Differenced variables for VAR component

### 5.3 MS-VAR Dataset
- First differences of all variables
- Standardization based on training period statistics
- Regime initialization flag based on rolling volatility

### 5.4 Machine Learning Features (XGBoost)
- Lag features: 1, 2, 3, 6, 12 months
- Rolling statistics: MA(3), MA(6), STD(3)
- Momentum features: diff(1), diff(3)

### 5.5 LSTM Features
- Sequence length: 6 months
- Features: target + lags + momentum + MA
- MinMax scaling applied

## 6. Quality Checks Performed

1. **Date continuity**: Verified no gaps in monthly series
2. **Unit consistency**: All monetary values in USD millions
3. **Outlier detection**: Flagged values beyond 3 standard deviations
4. **Zero handling**: Monitored zero values (may indicate data issues)
5. **Stationarity**: ADF and KPSS tests on each variable
6. **Cointegration**: Johansen trace test for multivariate systems

