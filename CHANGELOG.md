# Changelog

## 2026-02-19
- Added a unified rolling-origin evaluator with a common forecast API and exportable summary tables.
- Added exogenous forecast options (`naive` and univariate `arima`) plus `forecast` vs `actual` exog modes.
- Added probabilistic scoring (Gaussian CRPS, log score) and interval coverage diagnostics.
- Fixed leakage in VECM preparation (Johansen/ECT computed on training only) and in baseline VECM rank estimation.
- Fixed leakage in ML preprocessing (XGBoost rolling stats/momentum shifted; LSTM scaler fit on train only).
- Added package initializers for `scripts/` and `scripts/forecasting_models/`.
- Added MS-VAR, MS-VECM, and LSTM adapters to the unified evaluator (optional flags).
- Added multi-varset runner `scripts/forecasting_models/run_unified_evaluations.py`.
- Updated robustness spec to reference unified evaluator outputs.
- Tuned BVAR draw defaults in the unified evaluator (1000/200) for rolling-origin feasibility.
- Reduced default LSTM training load in unified evaluator (epochs 60, batch 16, patience 5).
- Wired robustness table generator to prefer unified rolling-origin outputs when available.
- Added ARIMA order selection for rolling backtests and unified evaluator (AIC grid).
- Added varset-specific VECM lag selection (VAR order on training) and kept rank estimation on training.
- Added time-series CV tuning script for XGBoost/LSTM with JSON outputs.
- Added imputation benchmarking script (masked data + sensitivity by missingness rate).
- Added policy-relevant asymmetric loss to unified summaries.
- Added LocalLevelSV (local-level state-space) and BoPIdentity forecasters with CLI flags.
- Added Policy Loss column to Table 1 LaTeX output when available.
- Extended statistical tests to support unified rolling-origin outputs (DM/MCS).
- Added support for tuned XGBoost/LSTM parameter JSONs that store `best_params`.
