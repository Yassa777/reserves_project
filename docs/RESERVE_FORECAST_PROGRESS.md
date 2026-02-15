# Reserve Adequacy Forecasting Progress Log

This file tracks iterative work on adding ARA component projections, adequacy gaps, and multi-scenario forecasts to the reserve adequacy dashboard.

## 2026-XX-XX - Session Start
- Aligned on approach: forecast ARA components (exports, M2, short-term debt, portfolio liabilities) and import cover/GG robustness with three scenarios (baseline, downside, upside).
- Decision: use simple growth-based quarterly projections with scenario multipliers to stay reproducible and avoid heavy model dependencies.
- Plan: add forecast utilities and a dedicated Forecast tab in `app_reserve_adequacy.py`; provide cross-scenario summary and downloadable data.

## 2026-01-26 - Forecast tab and scenarios
- Added scenario shocks (baseline/downside/upside) and growth-based quarterly forecasters for ARA components, reserves, and imports.
- New helper utilities to aggregate to quarterly, project series, and compute adequacy bands, GG ratio, and import cover with PBOC swap adjustment.
- Inserted a Forecasts tab in `app_reserve_adequacy.py` with horizon/lookback controls, scenario comparison table, charts (ARA ratio, reserves vs required band, import cover, GG), and CSV downloads.
- Default scenario shocks (multipliers on median QoQ growth): **Baseline** (1.0 across), **Downside** (exports 0.9, M2 0.98, ST debt 1.05, portfolio 1.05, reserves 0.97, imports 1.05), **Upside** (exports 1.05, M2 1.02, ST debt 0.95, portfolio 0.95, reserves 1.02, imports 0.98). Tweak `SCENARIO_SHOCKS` in `app_reserve_adequacy.py` or use a longer/shorter lookback for sensitivity.
- Recommended starting settings in UI: horizon = 8 quarters, lookback = 8 quarters; use the scenario table at the bottom of the Forecast tab to compare adequacy gaps at the horizon end.
