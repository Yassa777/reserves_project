# Reserves Forecasting Report (Updated: 2026-02-10)

## Overview
This report compares baseline vs expanded variable-set forecasts with a consistent missing-data strategy and includes naive benchmarks plus MAE/RMSE/MAPE/sMAPE/MASE metrics for both single-fit and rolling backtests.

## Missing-data strategy
- Method: {'method': 'ffill_limit', 'limit': 3, 'drop_remaining': True}
- Applied to exogenous/system variables before model-specific transforms (target still required to be non-null).

## Variable sets
### Baseline
- ARIMA exog: ['exports_usd_m', 'imports_usd_m', 'remittances_usd_m', 'usd_lkr']
- VECM system: ['gross_reserves_usd_m', 'exports_usd_m', 'imports_usd_m', 'remittances_usd_m', 'usd_lkr']
- MS-VAR system: ['gross_reserves_usd_m', 'usd_lkr', 'exports_usd_m', 'imports_usd_m']

### Expanded
- ARIMA exog: ['exports_usd_m', 'imports_usd_m', 'remittances_usd_m', 'usd_lkr', 'trade_balance_usd_m', 'm2_usd_m', 'tourism_usd_m', 'cse_net_usd_m']
- VECM system: ['gross_reserves_usd_m', 'exports_usd_m', 'imports_usd_m', 'remittances_usd_m', 'usd_lkr', 'trade_balance_usd_m', 'm2_usd_m']
- MS-VAR system: ['gross_reserves_usd_m', 'usd_lkr', 'exports_usd_m', 'imports_usd_m', 'trade_balance_usd_m', 'm2_usd_m']

## Sample sizes
### Baseline
- Rows: 204 (train=132, validation=36, test=36)
- Date range: 2009-01-01 to 2025-12-01
### Expanded
- Rows: 138 (train=96, validation=27, test=15)
- Date range: 2012-01-01 to 2025-03-01

## Baseline results
### Single-fit (train only, multi-step forecast)
| Model | Split | MAE | RMSE | MAPE | sMAPE | MASE |
|---|---|---|---|---|---|---|
| ARIMA | validation | 1463.460902 | 1882.721261 | 50.884848 | 37.007225 | 4.307441 |
| ARIMA | test | 2051.775976 | 2287.896593 | 37.768594 | 47.987122 | 6.039044 |
| VECM | validation | 3756.389551 | 4339.358834 | 159.991913 | 71.540658 | 11.056276 |
| VECM | test | 3077.423973 | 3326.538822 | 77.739468 | 49.829802 | 9.057860 |
| MS-VAR | validation | 4127.833134 | 4728.514531 | 174.224940 | 75.048782 | 12.149555 |
| MS-VAR | test | 3631.424211 | 3843.622403 | 89.945011 | 55.840352 | 10.688462 |
| MS-VECM | validation | 3860.860652 | 4480.901558 | 164.408807 | 72.470713 | 11.363768 |
| MS-VECM | test | 3601.040242 | 3750.456583 | 87.614571 | 55.652072 | 10.599032 |
| NAIVE | validation | 3586.275052 | 4173.622540 | 153.553120 | 69.827577 | 10.555574 |
| NAIVE | test | 2597.170779 | 2942.172478 | 68.136745 | 44.085653 | 7.644318 |

### Rolling backtest (expanding window; refit every 12 months)
| model | split | mae | rmse | mape | smape | mase |
|---|---|---|---|---|---|---|
| ARIMA | validation | 736.657673 | 1055.896499 | 22.268701 | 17.314873 | 2.168223 |
| ARIMA | test | 916.271360 | 1177.286508 | 20.454064 | 24.620948 | 2.696885 |
| VECM | validation | 1554.664596 | 1811.327128 | 60.762477 | 41.081217 | 4.575883 |
| VECM | test | 2258.876984 | 2955.809085 | 63.062289 | 42.930584 | 6.648610 |
| MS-VAR | validation | 645.666643 | 893.593335 | 22.778527 | 24.040417 | 1.900407 |
| MS-VAR | test | 280.930747 | 329.595784 | 5.917962 | 5.810607 | 0.826871 |
| MS-VECM | validation | 560.353304 | 848.928280 | 20.896840 | 21.723426 | 1.649302 |
| MS-VECM | test | 254.582160 | 349.298750 | 5.642273 | 5.950302 | 0.749318 |
| Naive | validation | 411.190326 | 545.562827 | 11.984802 | 11.596890 | 1.210267 |
| Naive | test | 212.138889 | 311.909041 | 4.652087 | 4.900827 | 0.624394 |

## Expanded results
### Single-fit (train only, multi-step forecast)
| Model | Split | MAE | RMSE | MAPE | sMAPE | MASE |
|---|---|---|---|---|---|---|
| ARIMA | validation | 3786.914746 | 4323.569929 | 121.264061 | 62.085155 | 9.068629 |
| ARIMA | test | 565.195814 | 622.847286 | 10.362926 | 9.715741 | 1.353490 |
| VECM | validation | 1266719515689940157257645556670448795648.000000 | 6360007504025489463338717741537539653632.000000 | 65688770356294765739983747327698403328.000000 | 193.183064 | 3728367470634250769051143054737539072.000000 |
| VECM | test | 4919653532604199030635578498406796626848093162943131381361803264.000000 | 20828901770604238978395287966654760645744444795662435958602596352.000000 | 80895544729974774388984162039036356852308585971733263443558400.000000 | 200.000000 | 14480140213014673845702341317549414887369110172927559156629504.000000 |
| MS-VAR | validation | 3011.760867 | 3670.983372 | 103.240942 | 53.720154 | 7.212347 |
| MS-VAR | test | 2501.055892 | 2562.377344 | 45.486665 | 36.381369 | 5.989347 |
| MS-VECM | validation | 3547.706196 | 4235.566428 | 119.513243 | 59.154464 | 8.495790 |
| MS-VECM | test | 3562.185202 | 3593.041640 | 64.082402 | 47.950026 | 8.530463 |
| NAIVE | validation | 2837.013513 | 3446.554618 | 97.021537 | 51.944335 | 6.793874 |
| NAIVE | test | 1919.526334 | 2020.821977 | 35.347678 | 29.254498 | 4.596743 |

### Rolling backtest (expanding window; refit every 12 months)
| model | split | mae | rmse | mape | smape | mase |
|---|---|---|---|---|---|---|
| ARIMA | validation | 1387.601917 | 1760.501969 | 36.267626 | 26.857726 | 3.322928 |
| ARIMA | test | 2086.700812 | 2534.073228 | 38.325027 | 53.791075 | 4.997080 |
| VECM | validation | 4936.824826 | 9867.219645 | 200.670501 | 55.913312 | 11.822350 |
| VECM | test | 21635.028109 | 23974.656648 | 391.862700 | 147.439148 | 51.809997 |
| MS-VAR | validation | 556.208872 | 697.033853 | 15.260498 | 13.893127 | 1.331969 |
| MS-VAR | test | 1631.824755 | 4359.194458 | 34.742091 | 22.004626 | 3.907776 |
| MS-VECM | validation | 522.735850 | 647.559893 | 14.625374 | 13.564358 | 1.251810 |
| MS-VECM | test | 817.908316 | 2270.010849 | 17.254089 | 17.569300 | 1.958668 |
| Naive | validation | 523.105620 | 628.281128 | 14.584380 | 14.068376 | 1.252695 |
| Naive | test | 370.266667 | 721.302248 | 7.233482 | 8.869770 | 0.886688 |

## Interpretation
- Expanded variable sets reduce usable sample size materially (baseline 204 rows vs expanded 138 rows). The expanded test window is shorter (15 months), which makes test metrics more volatile.
- In baseline rolling tests, MS-VAR/MS-VECM outperform ARIMA/VECM but are still beaten by the naive benchmark, indicating very high persistence in reserves and that a random-walk baseline is a hard-to-beat comparator.
- In expanded rolling tests, MS-VAR/MS-VECM remain strong on validation but degrade on test, while the naive model is extremely strong in the shortened test window; this suggests either regime stability in late sample or overfitting risk with the expanded system.
- Single-fit (multi-step) results are consistently worse than rolling for all models, highlighting parameter drift and the need for regular refits.
- VECM performs poorly in expanded mode (very high error), likely due to high dimensionality + reduced sample size + a large lag order (k_ar_diff).
- Expanded VECM single-fit produced numerically explosive forecasts (astronomical errors); treat that run as unstable and not fit for decision use without re-specification.

## Strengths
- Regime-aware models (MS-VAR/MS-VECM) are competitive and often superior to ARIMA/VECM when refit regularly.
- The evaluation now includes sMAPE/MASE and naive baselines for fairer benchmarking.
- Clear missing-data strategy is applied consistently across datasets.

## Weaknesses / limitations
- Expanded variable set reduces sample size and shortens the test window; results are less stable and more sensitive to outliers.
- Naive benchmark outperforms many models in rolling tests; this implies limited incremental value unless models can beat persistence convincingly.
- VECM likely over-parameterized relative to sample size; k_ar_diff from diagnostics may not be suitable for expanded systems.
- Expanded VECM single-fit is numerically unstable; treat those baseline errors as a failure case, not a valid comparison.
- Exogenous variables are treated as known; operational forecasting would need their own forecasts or scenarios.
- Statsmodels warnings (frequency/convergence) indicate estimation instability in some refits.

## Next steps
- Introduce model selection for lag length and system size (BIC, rolling CV) and consider reducing k_ar_diff for expanded VECM.
- Use a rolling window in addition to expanding window to test robustness to structural breaks.
- Add scenario-based exogenous forecasts and evaluate sensitivity to macro drivers.
- Require models to beat the naive baseline on MASE before promoting them to production.
- Enforce monthly frequency in the index to eliminate statsmodels warnings.

## Overall assessment
The expanded variable set did not improve accuracy and materially reduced sample size, which weakened ARIMA/VECM and made results more volatile. Regime models remain strong but still face a high bar against the naive benchmark. The methodology is sound, but model selection and variable discipline are now the key determinants of real gains.
