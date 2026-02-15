# Reserve Adequacy Forecasting Methodology and Implementation

This document describes the end-to-end methodology, technical decisions, implementation details, outputs, and limitations for the reserve adequacy forecasting module added to the SL-FSI reserve dashboard.

## 1) Objective and Scope
- Primary goal: construct a forward-looking *required reserve path* using IMF ARA components, then compare actual/forecasted reserves to that requirement to estimate adequacy gaps.
- Secondary goal: stress-test the adequacy path with three transparent scenarios (baseline, downside, upside) and check robustness using import cover and Greenspan-Guidotti (GG).
- Focus: forecast *requirements* rather than *actual reserves* to avoid overreliance on balance-of-payments modeling that is not fully available in the dataset.

## 2) Data Inputs and Frequency
- Quarterly frequency is used because key inputs (short-term debt, portfolio liabilities) are quarterly.
- Inputs (from existing loaders in `app_reserve_adequacy.py`):
  - Reserves (monthly -> quarterly end): `gross_reserves_usd_m`
  - Exports (monthly -> quarterly sum -> annualized)
  - Broad money M2 (monthly -> quarterly end -> USD via FX)
  - Short-term external debt (quarterly, USD)
  - Portfolio liabilities (quarterly, USD)
  - Imports (monthly -> quarterly sum for import cover)

## 3) Core Metric: IMF ARA Requirement
- Formula (quarterly):
  - ARA = 0.05 * Annual Exports + 0.05 * M2 (USD) + 0.30 * Short-term Debt + 0.15 * Portfolio Liabilities
- Adequacy thresholds:
  - Minimum requirement: 100% of ARA
  - Comfortable range: 150% of ARA
- Outputs:
  - `ara_total`, `ara_ratio`, `required_100`, `required_150`
  - Adequacy gaps: `gap_to_100`, `gap_to_150`

## 4) Forecasting Strategy (Component-Based)
### 4.1 Rationale
- A joint multivariate model (VAR/BVAR) is not adopted due to:
  - Limited quarterly sample size (post-2012)
  - Mixed dynamics and structural breaks (COVID, default)
  - Parameter explosion and instability
- A component-based, growth-extrapolation approach is chosen to maximize interpretability and reproducibility.

### 4.2 Method
- Each component is projected independently with a median quarterly growth rate:
  - Compute QoQ growth over a recent lookback window (default 8 quarters).
  - Use median QoQ growth to reduce sensitivity to outliers.
  - Apply scenario multipliers (see Section 5).
- Forecast horizon (default 8 quarters) is user-configurable.

### 4.3 Implementation
- Utility functions added in `app_reserve_adequacy.py`:
  - `to_quarterly(df, value_col, how)`: resamples monthly data to quarterly.
  - `forecast_with_growth(series_q, value_col, horizon, lookback, scenario_multiplier, start_date)`: median QoQ growth forecast.
  - `prepare_quarterly_imports(monthly_imports)`: quarterly import totals.
  - `enrich_with_import_cover(df, imports_q)`: import cover and net reserves with PBOC swap.
  - `build_forecast_scenarios(...)`: builds the history panel + scenario forecast panels.

## 5) Scenario Design
Three scenarios are implemented as multipliers on the median QoQ growth rate:
- Baseline: no adjustment (multipliers = 1.0).
- Downside: weaker external earnings and higher debt/import growth.
- Upside: stronger export performance and slower debt/liability growth.

Default multipliers (can be tuned in `SCENARIO_SHOCKS`):
- Exports: 0.9 / 1.0 / 1.05 (Downside / Baseline / Upside)
- M2: 0.98 / 1.0 / 1.02
- Short-term debt: 1.05 / 1.0 / 0.95
- Portfolio liabilities: 1.05 / 1.0 / 0.95
- Reserves: 0.97 / 1.0 / 1.02
- Imports: 1.05 / 1.0 / 0.98

## 6) Robustness Metrics
- Import cover (months):
  - Import cover = Reserves / (Quarterly imports / 3)
  - Net import cover = (Reserves - PBOC swap) / (Quarterly imports / 3)
- Greenspan-Guidotti ratio:
  - GG = Reserves / Short-term debt (USD)
- Threshold lines are plotted:
  - Import cover: 3 and 6 months
  - GG: 1.0

## 7) Implementation in the Dashboard
- A new "Forecasts" tab is added to `app_reserve_adequacy.py`.
- UI controls:
  - `horizon` slider (4 to 12 quarters)
  - `lookback` slider (4 to 12 quarters)
  - scenario selector
- Outputs (per scenario):
  - ARA ratio chart (history + forecast)
  - Reserves vs required band (100-150% ARA)
  - Robustness charts (import cover + GG)
  - Scenario summary table at forecast horizon end
  - CSV download for forecast data

## 8) Results and Outputs (What the Module Produces)
The module produces the following outputs in the UI:
- A time-varying required reserve band (100-150% ARA).
- An adequacy gap series for each scenario:
  - gap to 100% ARA (minimum requirement)
  - gap to 150% ARA (comfortable requirement)
- Scenario comparison table that summarizes:
  - ARA ratio at horizon end
  - adequacy gaps (USD)
  - import cover and GG ratio at horizon end
- Robustness panels to confirm whether adequacy is supported by traditional metrics.

## 9) Gaps, Weaknesses, and Limitations
- **No structural BoP model**: reserves are not directly modeled from full balance-of-payments flows.
- **Component independence**: component forecasts are univariate and ignore dynamic co-movement.
- **Structural breaks**: 2020-2022 regime shifts can bias growth-based forecasts.
- **Sample size**: quarterly data provides only ~50 observations post-2012, limiting model complexity.
- **Portfolio liability coverage**: data frequency and coverage can create gaps in early periods.
- **Swap encumbrance**: PBOC swap adjustment is a fixed assumption; actual usability is uncertain.

## 10) Future Enhancements
- **Scenario calibration**: tie scenario shocks to historical percentiles (25th/50th/75th growth).
- **BVAR with shrinkage**: test a minimal multivariate model for exports, imports, and debt.
- **Alternative reserve definitions**: include SDDS reserve template for net usable reserves.
- **Policy reaction functions**: link reserves to exchange rate regime or intervention rules.
- **Uncertainty bands**: provide confidence intervals using bootstrap growth distributions.
- **Automatic gap alerts**: flag quarters where adequacy gap crosses thresholds.

## 11) Files and Entry Points
- Primary implementation: `app_reserve_adequacy.py`
- Progress log: `DOCS/RESERVE_FORECAST_PROGRESS.md`
- Data definitions: `DOCS/DATA_DICTIONARY_RESERVE_ADEQUACY.md`
