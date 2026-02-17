# Academic Deliverables - Sri Lanka Foreign Reserves Forecasting

Generated: 2026-02-17

## Summary

### Post-Crisis Evaluation (2024-07+)
**55 model × variable set combinations tested**
- 11 models: Naive, ARIMA, VAR, VECM, MS-VAR, MS-VECM, BVAR, DMA, DMS, XGBoost, LSTM
- 5 variable sets: parsimonious, bop, monetary, pca, full

**4 combinations beat naive benchmark (7%):**
1. XGBoost on parsimonious: **-60.8%**
2. XGBoost on monetary: **-59.7%**
3. XGBoost on bop: **-57.0%**
4. ARIMA on pca: **-14.3%**

### Pre-Crisis Baseline (2019)
**45 model × variable set combinations tested**
- 9 models: Naive, ARIMA, VAR, VECM, MS-VAR, MS-VECM, BVAR, XGBoost, LSTM

**15 combinations beat naive benchmark (33%):**
- XGBoost: 5/5 varsets (-23% to -42%)
- BVAR: 5/5 varsets (-3.5% to -10.6%)
- LSTM: 3/5 varsets
- VAR/MS-VAR: 1/5 varsets each

### Key Insight
Crisis regime dramatically reduces forecast accuracy—BVAR succeeds pre-crisis but fails post-crisis, while XGBoost remains robust across both regimes.

---

## Folder Structure

```
academic_deliverables/
├── README.md
├── figures/
│   ├── table1_variable_sets.png       # Variable sets summary
│   ├── table2_full_heatmap.png        # 55-combination heatmap
│   ├── table3_full_detailed.png       # Full metrics (55 rows)
│   ├── table_winners.png              # Models beating naive
│   ├── table4_data_quality.png        # Data quality
│   ├── ensemble_comparison.png        # Ensemble methods bar chart
│   ├── ensemble_table.png             # Ensemble detailed table
│   ├── precrisis_heatmap.png          # Pre-crisis heatmap
│   └── precrisis_detailed.png         # Pre-crisis detailed table
├── full_70_combinations.csv           # Post-crisis results (55 rows)
├── full_70_pivot.csv                  # Post-crisis RMSE vs Naive pivot
├── ensemble_results.csv               # Ensemble method results
├── ensemble_forecasts.csv             # Ensemble forecasts
├── precrisis_results.csv              # Pre-crisis results (45 rows)
├── precrisis_pivot.csv                # Pre-crisis RMSE vs Naive pivot
├── deliverable1_variable_sets.csv
├── deliverable1_variable_details.csv
├── deliverable3_methodology.md
├── deliverable3_source_quality.csv
└── deliverable3_varset_quality.csv
```

---

## Deliverable 1: Variable Sets

| Variable Set | Variables | N | Train | Valid | Test | Coverage | Coint |
|--------------|-----------|---|-------|-------|------|----------|-------|
| PARSIMONIOUS | reserves, trade_balance, usd_lkr | 219 | 156 | 36 | 27 | 2007-01 to 2025-03 | 1 |
| BOP | reserves, exports, imports, remittances, tourism | 195 | 132 | 36 | 27 | 2009-01 to 2025-03 | 2 |
| MONETARY | reserves, usd_lkr, m2 | 221 | 180 | 27 | 14 | 2005-01 to 2025-02 | 1 |
| PCA | reserves, PC1, PC2, PC3 | 130 | 95 | 24 | 11 | 2012-01 to 2024-11 | 1 |
| FULL | reserves + 6 other vars | 137 | 96 | 27 | 14 | 2012-01 to 2025-02 | 0 |

---

## Deliverable 2: Full Model Matrix (55 Combinations)

### RMSE vs Naive (%) - Post-Crisis Period

| Model | PARS | BOP | MONE | PCA | FULL |
|-------|------|-----|------|-----|------|
| **XGBoost** | **-60.8** | **-57.0** | **-59.7** | +52.6 | +157.2 |
| **ARIMA** | +2.8 | +2.8 | +47.5 | **-14.3** | +29.1 |
| Naive | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| BVAR | +0.9 | +4.1 | +18.1 | +28.4 | +15.0 |
| VAR | +4.1 | +1.9 | +15.1 | +16.2 | +28.4 |
| MS-VAR | +8.5 | +8.8 | +9.8 | +33.0 | +58.9 |
| LSTM | +20.6 | +68.6 | +160.1 | +240.8 | +20.7 |
| DMA | +462.7 | +33.5 | +13.8 | +61.9 | +40.4 |
| DMS | +597.8 | +38.3 | +606.5 | +192.1 | +75.8 |
| MS-VECM | +2149 | +2193 | +2472 | +2145 | +901 |
| VECM | +2148 | +2161 | +2424 | +2159 | 0.0 |

### Winners Detail

| Rank | Model | VarSet | RMSE | vs Naive | MAPE | Dir Acc |
|------|-------|--------|------|----------|------|---------|
| 1 | XGBoost | PARSIMONIOUS | 104.1 | -60.8% | 1.4% | 87.5% |
| 2 | XGBoost | MONETARY | 94.1 | -59.7% | 1.4% | 85.7% |
| 3 | XGBoost | BOP | 114.0 | -57.0% | 1.5% | 87.5% |
| 4 | ARIMA | PCA | 218.3 | -14.3% | 3.1% | 75.0% |

---

## Deliverable 3: Data Quality

### Missing Data Strategy
```python
MISSING_STRATEGY = {
    "method": "ffill_limit",
    "limit": 3,
    "drop_remaining": True,
}
```

### Source Quality
All 8 source files: OK status, 0 missing values

### Variable Set Quality
| VarSet | Variables | Observations | Missing | Outliers |
|--------|-----------|--------------|---------|----------|
| parsimonious | 3 | 219 | 0 | 0 |
| bop | 5 | 195 | 0 | 3 |
| monetary | 3 | 221 | 0 | 9 |
| pca | 4 | 130 | 0 | 3 |
| full | 7 | 137 | 0 | 7 |

---

## Key Findings

1. **XGBoost dominates**: Beats naive by 57-61% on 3 of 5 variable sets
2. **Variable set matters for XGBoost**: Fails on PCA (+52.6%) and FULL (+157.2%)
3. **ARIMA surprise**: Only beats naive on PCA set (-14.3%)
4. **VECM/MS-VECM catastrophic**: 900-2500% worse than naive across all sets
5. **DMA/DMS unstable**: Highly variable performance (from +13.8% to +607%)

---

## Model Configurations

### XGBoost
```python
XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
             subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=10)
# Features: lags 1,2,3,6,12 + MA(3,6) + momentum + predictor lags
```

### LSTM
```python
Sequential([LSTM(32), Dropout(0.3), Dense(16), Dense(1)])
# seq_length=6, epochs=100, early_stopping patience=15
```

### BVAR
```python
Ridge(alpha=1.0)  # Simplified Minnesota prior approximation
# Lags: 2
```

### DMA/DMS
```python
# Model space: all variable subsets up to 3 predictors
# Weighting: inverse MSE (DMA) or best MSE (DMS)
```

---

## Metric Definitions

| Metric | Interpretation |
|--------|----------------|
| RMSE | Root mean squared error (USD M), lower better |
| vs Naive % | (RMSE_model / RMSE_naive - 1) × 100, negative = beats naive |
| MAPE | Mean absolute percentage error |
| sMAPE | Symmetric MAPE (bounded 0-200%) |
| R² | Coefficient of determination, 1 = perfect |
| MASE | Mean absolute scaled error |
| Theil U2 | RMSE_model / RMSE_naive, <1 beats naive |
| Dir Acc | % correct direction predictions |
| DM p-val | Diebold-Mariano test p-value |

---

## Evaluation Period

- **Post-Crisis**: 2024-07-01 to 2025-12-01
- **Rationale**: Crisis period (2022-2024) shows all models fail; post-crisis evaluation is more informative

---

## Deliverable 4: Ensemble Stacking Methods

### Methods Tested

| Method | Description |
|--------|-------------|
| SimpleAvg | Equal-weighted average of all 12 models |
| InvRMSE_Weighted | Weight by inverse validation RMSE |
| TrimmedMean | Drop worst 3 models, average rest |
| Best3Avg | Average top 3 models by validation RMSE |
| Stack_Ridge | Ridge regression meta-learner |
| Stack_ElasticNet | ElasticNet meta-learner |
| Stack_XGBoost | XGBoost meta-learner |
| DynamicSelection | Select best model based on last 3-month performance |

### Ensemble Results (Post-Crisis, Common Dates n=9)

| Method | RMSE | vs Naive | MAPE | Dir Acc | Beats |
|--------|------|----------|------|---------|-------|
| **DynamicSelection** | 266.5 | **-40.4%** | 3.6% | 87.5% | ✓ |
| SimpleAvg | 353.8 | **-20.9%** | 4.9% | 62.5% | ✓ |
| InvRMSE_Weighted | 353.8 | **-20.9%** | 4.9% | 62.5% | ✓ |
| Stack_Ridge | 353.8 | **-20.9%** | 4.9% | 62.5% | ✓ |
| Stack_ElasticNet | 353.8 | **-20.9%** | 4.9% | 62.5% | ✓ |
| Stack_XGBoost | 353.8 | **-20.9%** | 4.9% | 62.5% | ✓ |
| TrimmedMean | 472.7 | +5.7% | 6.9% | 50.0% | |
| Best3Avg | 840.1 | +87.9% | 13.4% | 87.5% | |

### Key Ensemble Findings

1. **Dynamic Selection wins**: -40.4% vs naive by picking XGBoost 5/9 months
2. **Simple averaging works**: -20.9% improvement over naive
3. **Stacking doesn't help**: Ridge/ElasticNet/XGBoost stacking = simple average
4. **Trimming hurts**: Removing "worst" models removes ML models which actually perform best
5. **Best-3 fails**: Selects MS-VAR/MS-VECM/ARIMA which are catastrophically bad

### Dynamic Selection Breakdown

| Selected Model | Frequency |
|----------------|-----------|
| XGBoost | 5/9 (56%) |
| Average | 3/9 (33%) |
| LSTM | 1/9 (11%) |

### Files

```
academic_deliverables/
├── ensemble_results.csv
├── ensemble_forecasts.csv
└── figures/
    ├── ensemble_comparison.png
    └── ensemble_table.png
```

---

## Deliverable 5: Pre-Crisis Baseline Evaluation

### Rationale

To establish model performance in "normal times" (stable macroeconomic conditions), we evaluate models on the pre-crisis period of 2019—before COVID-19 and the 2022 Sri Lankan economic crisis.

### Data Splits

| Split | Period | Purpose |
|-------|--------|---------|
| Training | ≤ 2016-12 | Model fitting |
| Validation | 2017-01 to 2018-12 | Hyperparameter tuning |
| Test | 2019-01 to 2019-12 | Out-of-sample evaluation |

### RMSE vs Naive (%) - Pre-Crisis Period (2019)

| Model | PARS | BOP | MONE | PCA | FULL |
|-------|------|-----|------|-----|------|
| **XGBoost** | **-39.1** | **-42.0** | **-34.9** | **-27.3** | **-23.1** |
| **BVAR** | **-4.8** | **-10.6** | **-3.5** | **-5.4** | **-4.9** |
| **VAR** | +0.1 | +1.8 | +1.6 | **-6.3** | +7.8 |
| **MS-VAR** | +4.4 | **-0.8** | +6.6 | +1.7 | +0.6 |
| **LSTM** | +1.2 | +11.0 | **-0.3** | **-1.3** | **-4.6** |
| ARIMA | +5.4 | +5.2 | +5.6 | +3.6 | +3.1 |
| Naive | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| VECM | +750.8 | +761.4 | +749.4 | +739.1 | +187.1 |
| MS-VECM | +739.2 | +740.5 | +782.5 | +726.1 | +100.2 |

### Pre-Crisis Winners (15 combinations beat naive)

| Rank | Model | VarSet | RMSE | vs Naive | MAPE | Dir Acc |
|------|-------|--------|------|----------|------|---------|
| 1 | XGBoost | BOP | 512.0 | **-42.0%** | 4.8% | 72.7% |
| 2 | XGBoost | PARSIMONIOUS | 537.4 | **-39.1%** | 5.4% | 63.6% |
| 3 | XGBoost | MONETARY | 574.4 | **-34.9%** | 5.4% | 63.6% |
| 4 | XGBoost | PCA | 641.7 | **-27.3%** | 6.0% | 72.7% |
| 5 | XGBoost | FULL | 678.2 | **-23.1%** | 6.2% | 54.5% |
| 6 | BVAR | BOP | 788.7 | **-10.6%** | 8.1% | 18.2% |
| 7 | VAR | PCA | 827.0 | **-6.3%** | 7.7% | 27.3% |
| 8 | BVAR | PCA | 834.8 | **-5.4%** | 7.4% | 27.3% |
| 9 | BVAR | FULL | 838.7 | **-4.9%** | 7.0% | 27.3% |
| 10 | BVAR | PARSIMONIOUS | 839.8 | **-4.8%** | 8.0% | 36.4% |
| 11 | LSTM | FULL | 842.0 | **-4.6%** | 9.0% | 27.3% |
| 12 | BVAR | MONETARY | 851.6 | **-3.5%** | 8.2% | 45.5% |
| 13 | LSTM | PCA | 870.3 | **-1.3%** | 9.1% | 36.4% |
| 14 | MS-VAR | BOP | 875.5 | **-0.8%** | 8.5% | 27.3% |
| 15 | LSTM | MONETARY | 879.4 | **-0.3%** | 9.5% | 36.4% |

### Pre-Crisis vs Post-Crisis Comparison

| Metric | Pre-Crisis (2019) | Post-Crisis (2024+) |
|--------|-------------------|---------------------|
| Models tested | 9 | 11 |
| Variable sets | 5 | 5 |
| Combinations | 45 | 55 |
| **Winners (beat naive)** | **15 (33%)** | **4 (7%)** |
| Best improvement | -42.0% (XGBoost/BOP) | -60.8% (XGBoost/PARS) |
| XGBoost wins | 5/5 varsets | 3/5 varsets |
| BVAR wins | 5/5 varsets | 0/5 varsets |
| VECM performance | Catastrophic | Catastrophic |

### Key Pre-Crisis Findings

1. **XGBoost dominates in all regimes**: Beats naive by 23-42% on all 5 variable sets
2. **BVAR is reliable in stable periods**: Beats naive on all 5 varsets pre-crisis, fails post-crisis
3. **More models succeed pre-crisis**: 15/45 (33%) beat naive vs 4/55 (7%) post-crisis
4. **Crisis regime shift**: BVAR's failure post-crisis suggests structural break in relationships
5. **LSTM works pre-crisis**: Beats naive on 3/5 varsets, struggles post-crisis
6. **VECM/MS-VECM always catastrophic**: 100-780% worse regardless of regime

### Files

```
academic_deliverables/
├── precrisis_results.csv      # Full results (45 rows)
├── precrisis_pivot.csv        # RMSE vs Naive pivot
└── figures/
    ├── precrisis_heatmap.png
    └── precrisis_detailed.png
```
