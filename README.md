# Reserves Forecasting Project

Sri Lanka reserve level forecasting with comprehensive diagnostic framework.

## Structure

```
reserves_project/
├── reserves_project/              # Core Python package (pip/console entry points)
├── apps/                           # Streamlit dashboards
│   ├── run_diagnostics.py          # Modular Streamlit diagnostics app (Phase 1-9)
│   ├── app_reserves_diagnostics.py # Monolithic diagnostics app
│   ├── app_reserve_adequacy.py     # Reserve adequacy metrics
│   └── app_data_dictionary.py      # Data source documentation
├── scripts/                        # Analysis scripts
│   ├── run_diagnostics.py          # Orchestrates all diagnostic phases
│   ├── prepare_forecasting_data.py # Builds model-ready datasets for forecasting
│   ├── run_forecasting_models.py   # Runs baseline forecasts (ARIMA/VECM/MS proxies)
│   ├── run_rolling_backtests.py    # Rolling backtests for model comparison
│   └── diagnostics_phases/         # Modular phase-specific diagnostics
│       ├── phase2_stationarity.py
│       ├── phase3_temporal.py
│       ├── phase4_volatility.py
│       ├── phase5_breaks.py
│       ├── phase6_relationships.py
│       ├── phase7_cointegration.py
│       ├── phase8_svar.py
│       └── phase9_multiple_breaks.py
├── data/
│   ├── external/                   # Raw source data
│   ├── merged/                     # Processed panels
│   │   ├── reserves_forecasting_panel.csv  # Main forecasting panel
│   │   └── slfsi_monthly_panel.csv         # Full FSI panel
│   └── diagnostics/                # Test results
│       ├── diagnostic_results.json
│       ├── variable_quality_summary.csv
│       ├── integration_summary.csv
│       ├── arch_summary.csv
│       ├── chow_test_summary.csv
│       ├── granger_causality_summary.csv
│       ├── cointegration_engle_granger_summary.csv
│       ├── ecm_suitability_summary.csv
│       ├── johansen_summary.csv
│       ├── vecm_suitability_summary.csv
│       ├── svar_exogeneity_summary.csv
│       ├── svar_sign_restriction_summary.csv
│       ├── svar_model_summary.csv
│       └── bai_perron_summary.csv
│   └── forecast_prep/              # Prepared datasets for forecasting models
│       ├── arima_prep_dataset.csv
│       ├── vecm_levels_dataset.csv
│       ├── ms_vecm_state_dataset.csv
│       ├── ms_var_raw_dataset.csv
│       ├── ms_var_scaled_dataset.csv
│       ├── model_readiness_summary.csv
│       └── forecast_prep_metadata.json
│   └── forecast_results/           # Forecast outputs and summaries
│       ├── arima_forecast.csv
│       ├── vecm_forecast.csv
│       ├── ms_var_forecast.csv
│       ├── ms_vecm_forecast.csv
│       └── forecast_model_summary.json
│       ├── rolling_backtests.csv
│       ├── rolling_backtest_summary.csv
│       └── rolling_backtest_metadata.json
└── docs/                           # Documentation
    ├── DIAGNOSTIC_TEST_RESULTS.md  # Complete test results
    └── complete_data_dictionary.md # Variable definitions
```

Note: Core logic is now in the `reserves_project/` package. Legacy `scripts/` remain as thin shims for backward compatibility.

## Quick Start

```bash
# Install editable package (for console scripts)
pip install -e .

# Unified rolling-origin evaluation
reserves-unified --exog-mode forecast --exog-forecast naive --include-ms --include-lstm --include-llsv --include-bop

# Statistical tests (DM/MCS) on unified outputs
reserves-stat-tests --use-unified --unified-varset parsimonious --unified-horizon 1 --unified-split test

# Robustness tables
reserves-tables

# Tune ML models
reserves-tune-ml --varset parsimonious

# Run diagnostics
cd reserves_project
python scripts/run_diagnostics.py

# Build forecasting-ready datasets (ARIMA, VECM, MS-VAR, MS-VECM)
python scripts/prepare_forecasting_data.py

# (New) Diagnostics + forecasting prep via console scripts
reserves-diagnostics
reserves-forecast-prep --varset baseline

# Baseline forecasts + rolling backtests
reserves-forecast-baselines --varset baseline
reserves-rolling-backtests --varset baseline --refit-interval 12

# Optional: write outputs under data/outputs/<run-id>/
reserves-forecast-baselines --varset baseline --run-id 2026-02-23_baselines
reserves-rolling-backtests --varset baseline --refit-interval 12 --run-id 2026-02-23_baselines

# Run baseline forecasting models (ARIMA, VECM, regime proxies)
python scripts/run_forecasting_models.py

# Rolling backtests (expanding window)
python scripts/run_rolling_backtests.py

# Launch modular dashboard (from reserves_project directory)
streamlit run apps/run_diagnostics.py -- 9
```

## Key Findings

1. **Target Variable:** `gross_reserves_usd_m` is I(1) - requires differencing
2. **Persistence:** ACF(1) = 0.97 - very high persistence
3. **ARCH Effects:** Volatility clustering present
4. **⚠️ Surprising:** No structural break at April 2022 default
5. **Causality:** No Granger causality from BoP flows

## Data Coverage

- **Date Range:** Jan 2005 - Dec 2025 (252 monthly observations)
- **Target:** `gross_reserves_usd_m` (USD millions)
- **Key Predictors:** exports, imports, remittances, tourism, CSE flows
