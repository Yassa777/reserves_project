# Data Dictionary: Reserve Adequacy Threshold Analysis

**Project:** Sri Lanka Financial Stress Index (SL-FSI) - Reserve Adequacy Module
**Purpose:** Document all raw data sources, column definitions, and derived metrics for the Reserve Level Adequacy Threshold research
**Last Updated:** January 2026
**Primary Data Source:** Central Bank of Sri Lanka (CBSL)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Data Source Files](#2-data-source-files)
3. [Column Definitions by Category](#3-column-definitions-by-category)
4. [Derived/Calculated Metrics](#4-derivedcalculated-metrics)
5. [Data Coverage Summary](#5-data-coverage-summary)
6. [References and URLs](#6-references-and-urls)

---

## 1. Overview

This data dictionary documents all variables used in the Reserve Adequacy Threshold Analysis, which aims to identify ex-ante thresholds that would have signaled Sri Lanka's 2022 sovereign default crisis. The analysis covers multiple reserve adequacy benchmarks including Import Cover, Greenspan-Guidotti Ratio, and IMF ARA Metric.

### Key Event Timeline
- **April 12, 2022:** Sri Lanka announces sovereign default
- **Research Question:** At what reserve level did the crisis become inevitable?
- **Data Range:** 2005 - 2025 (with varying coverage by series)

---

## 2. Data Source Files

### 2.1 Primary Source Files (Reserve Adequacy Focus)

| File | Source Table | Frequency | Data Range | Description |
|------|--------------|-----------|------------|-------------|
| `reserve_assets_monthly_cbsl.csv` | CBSL Historical Data Series | Monthly | Nov 2013 - Dec 2025 | Official reserve asset components |
| `central_govt_debt_quarterly.csv` | CBSL SDDS | Quarterly | 2000 - Q3 2025 | Central government debt by type |
| `external_debt_usd_quarterly.csv` | CBSL Table 2.12 | Quarterly | Q4 2012 - Q3 2025 | External debt in USD |
| `iip_quarterly_2025.csv` | CBSL IIP Release | Quarterly | Q4 2012 - Q3 2025 | International Investment Position |
| `monetary_aggregates_monthly.csv` | CBSL Table 4.02 | Monthly | Dec 1995 - Sep 2025 | M0 and M2 money supply |
| `reserve_money_velocity_monthly.csv` | CBSL Table 4.11 | Monthly | Jan 2003 - Dec 2025 | Reserve money and multipliers |
| `monthly_imports_usd.csv` | CBSL Table 2.04 | Monthly | Jan 2007 - Nov 2025 | Merchandise imports |
| `monthly_exports_usd.csv` | CBSL Table 2.02 | Monthly | Jan 2007 - Nov 2025 | Merchandise exports |
| `tourism_earnings_monthly.csv` | CBSL Table 2.14.1 | Monthly | Jan 2009 - Latest | Tourism receipts |
| `remittances_monthly.csv` | CBSL Table 2.14.2 | Monthly | Jan 2009 - Latest | Workers' remittances |
| `cse_flows_monthly.csv` | CBSL Table 2.14.3 | Monthly | Jan 2012 - Latest | CSE portfolio flows |

### 2.2 Consolidated Output File

| File | Description |
|------|-------------|
| `slfsi_monthly_panel.csv` | Merged monthly panel with all indicators (data/merged/) |

---

## 3. Column Definitions by Category

### 3.1 Reserve Assets

**Source File:** `data/external/reserve_assets_monthly_cbsl.csv`
**CBSL Source:** Official Reserve Assets - Historical Data Series
**URL:** https://www.cbsl.gov.lk/en/statistics/statistical-tables/external-sector

| Column | Definition | Unit | Notes |
|--------|------------|------|-------|
| `date` | Month-end date | Date | Format: YYYY-MM-DD |
| `gross_reserves_usd_m` | Total official reserve assets | USD millions | Sum of all components below |
| `fx_reserves_usd_m` | Foreign currency reserves | USD millions | Primarily held in USD, EUR, GBP |
| `imf_position_usd_m` | Reserve position in IMF | USD millions | Sri Lanka's quota contribution |
| `sdrs_usd_m` | Special Drawing Rights | USD millions | IMF-allocated SDRs |
| `gold_usd_m` | Monetary gold holdings | USD millions | Valued at market prices |
| `other_reserves_usd_m` | Other reserve assets | USD millions | Residual category |
| `source` | Data provenance marker | String | "CBSL Official Reserve Assets Historical" |

**Formula:**
```
gross_reserves_usd_m = fx_reserves_usd_m + imf_position_usd_m + sdrs_usd_m + gold_usd_m + other_reserves_usd_m
```

---

### 3.2 Central Government Debt (LKR)

**Source File:** `data/external/central_govt_debt_quarterly.csv`
**CBSL Source:** Central Government Debt SDDS (Special Data Dissemination Standard)
**URL:** https://www.cbsl.gov.lk/en/statistics/sdds-sri-lanka

| Column | Definition | Unit | Notes |
|--------|------------|------|-------|
| `date` | Quarter-end date | Date | Format: YYYY-MM-DD |
| `total_debt_lkr_m` | Total central government debt | LKR millions | Domestic + Foreign |
| `domestic_debt_lkr_m` | Total domestic debt | LKR millions | Rupee-denominated obligations |
| `domestic_short_term_lkr_m` | Short-term domestic debt | LKR millions | Maturity < 12 months (T-bills, etc.) |
| `domestic_medium_long_lkr_m` | Medium/long-term domestic debt | LKR millions | Maturity ≥ 12 months (T-bonds, etc.) |
| `foreign_debt_lkr_m` | Total foreign debt (LKR equivalent) | LKR millions | Converted at prevailing FX rate |
| `foreign_short_term_lkr_m` | Short-term foreign debt | LKR millions | Maturity < 12 months |
| `total_short_term_lkr_m` | Combined short-term debt | LKR millions | domestic_short_term + foreign_short_term |
| `source` | Data provenance marker | String | "CBSL Central Government Debt SDDS" |

**Formula:**
```
total_debt_lkr_m = domestic_debt_lkr_m + foreign_debt_lkr_m
domestic_debt_lkr_m = domestic_short_term_lkr_m + domestic_medium_long_lkr_m
total_short_term_lkr_m = domestic_short_term_lkr_m + foreign_short_term_lkr_m
```

---

### 3.3 External Debt (USD)

**Source File:** `data/external/external_debt_usd_quarterly.csv`
**CBSL Source:** Table 2.12 - Outstanding External Debt
**URL:** https://www.cbsl.gov.lk/en/statistics/statistical-tables/external-sector

| Column | Definition | Unit | Notes |
|--------|------------|------|-------|
| `date` | Quarter-end date | Date | Format: YYYY-MM-DD |
| `govt_total_usd_m` | Total government external debt | USD millions | All government foreign obligations |
| `govt_short_term_usd_m` | Government short-term external debt | USD millions | **Key for Greenspan-Guidotti** |
| `govt_long_term_usd_m` | Government long-term external debt | USD millions | Maturity ≥ 12 months |
| `central_bank_usd_m` | Central bank external debt | USD millions | CBSL foreign obligations |
| `other_sectors_usd_m` | Other sectors' external debt | USD millions | SOEs, private sector |
| `total_short_term_usd_m` | Combined short-term external debt | USD millions | All sectors, maturity < 12 months |

---

### 3.4 International Investment Position (IIP)

**Source File:** `data/external/iip_quarterly_2025.csv`
**CBSL Source:** International Investment Position Release
**URL:** https://www.cbsl.gov.lk/en/statistics/statistical-tables/external-sector

| Column | Definition | Unit | Notes |
|--------|------------|------|-------|
| `date` | Quarter-end date | Date | Format: YYYY-MM-DD |
| `portfolio_equity` | Portfolio investment - equity liabilities | USD millions | Foreign holdings of SL equities |
| `portfolio_debt` | Portfolio investment - debt securities | USD millions | Foreign holdings of SL bonds |
| `reserve_assets` | Official reserve assets (backup) | USD millions | Cross-check with reserve_assets file |
| `total_liabilities` | Total external liabilities | USD millions | All foreign claims on SL |

**Note:** For IMF ARA calculation, portfolio liabilities = `portfolio_equity + portfolio_debt`

---

### 3.5 Monetary Aggregates (M0, M2)

**Source File:** `data/external/monetary_aggregates_monthly.csv`
**CBSL Source:** Table 4.02 - Monetary Aggregates
**URL:** https://www.cbsl.gov.lk/en/statistics/statistical-tables/monetary-sector

| Column | Definition | Unit | Notes |
|--------|------------|------|-------|
| `date` | Month-end date | Date | Format: YYYY-MM-DD |
| `reserve_money_m0_lkr_m` | Reserve money (M0) | LKR millions | Currency + bank reserves at CBSL |
| `broad_money_m2_lkr_m` | Broad money (M2) | LKR millions | **Key for IMF ARA calculation** |

**Definition of M2:**
```
M2 = Currency in circulation + Demand deposits + Time deposits + Savings deposits
```

---

### 3.6 Reserve Money and Multipliers

**Source File:** `data/external/reserve_money_velocity_monthly.csv`
**CBSL Source:** Table 4.11 - Reserve Money, Multipliers and Velocity of Money
**URL:** https://www.cbsl.gov.lk/en/statistics/statistical-tables/monetary-sector

| Column | Definition | Unit | Notes |
|--------|------------|------|-------|
| `date` | Month-end date | Date | Format: YYYY-MM-DD |
| `reserve_money_total_lkr_m` | Total reserve money | LKR millions | Monetary base |
| `money_multiplier_m1` | M1 money multiplier | Ratio | M1 / Reserve Money |

**Formula:**
```
money_multiplier_m1 = M1 / Reserve Money
```

---

### 3.7 Trade Data

#### Imports
**Source File:** `data/external/monthly_imports_usd.csv`
**CBSL Source:** Table 2.04 - Monthly Trade Statistics (Imports)

| Column | Definition | Unit | Notes |
|--------|------------|------|-------|
| `date` | Month-end date | Date | Format: YYYY-MM-DD |
| `imports_usd_m` | Monthly merchandise imports | USD millions | CIF basis |

#### Exports
**Source File:** `data/external/monthly_exports_usd.csv`
**CBSL Source:** Table 2.02 - Monthly Trade Statistics (Exports)

| Column | Definition | Unit | Notes |
|--------|------------|------|-------|
| `date` | Month-end date | Date | Format: YYYY-MM-DD |
| `exports_usd_m` | Monthly merchandise exports | USD millions | FOB basis |

---

### 3.8 External Flows

#### Tourism Earnings
**Source File:** `data/external/tourism_earnings_monthly.csv`
**CBSL Source:** Table 2.14.1 - Tourism Earnings

| Column | Definition | Unit | Notes |
|--------|------------|------|-------|
| `date` | Month-end date | Date | Format: YYYY-MM-DD |
| `tourism_earnings_usd_m` | Monthly tourism receipts | USD millions | Travel services credit |

#### Workers' Remittances
**Source File:** `data/external/remittances_monthly.csv`
**CBSL Source:** Table 2.14.2 - Workers' Remittances

| Column | Definition | Unit | Notes |
|--------|------------|------|-------|
| `date` | Month-end date | Date | Format: YYYY-MM-DD |
| `remittances_usd_m` | Monthly remittance inflows | USD millions | Personal transfers from abroad |

#### Colombo Stock Exchange Portfolio Flows
**Source File:** `data/external/cse_flows_monthly.csv`
**CBSL Source:** Table 2.14.3 - CSE Portfolio Investment Flows

| Column | Definition | Unit | Notes |
|--------|------------|------|-------|
| `date` | Month-end date | Date | Format: YYYY-MM-DD |
| `cse_inflows_usd_m` | Foreign portfolio inflows | USD millions | Purchases by non-residents |
| `cse_outflows_usd_m` | Foreign portfolio outflows | USD millions | Sales by non-residents |
| `cse_net_usd_m` | Net portfolio flows | USD millions | Inflows - Outflows |

**Formula:**
```
cse_net_usd_m = cse_inflows_usd_m - cse_outflows_usd_m
```

---

## 4. Derived/Calculated Metrics

### 4.1 Import Cover Ratio

**Location:** Merged panel (`slfsi_monthly_panel.csv`)
**Column:** `import_cover_months`

**Formula:**
```
Import Cover (months) = Gross Reserves (USD) / Monthly Imports (USD)
```

**Thresholds:**
| Level | Months | Interpretation |
|-------|--------|----------------|
| Comfortable | ≥ 6 | Adequate buffer for external shocks |
| IMF Minimum | ≥ 3 | Minimum acceptable level |
| Warning | < 2 | Elevated risk of reserves depletion |
| Critical | < 1 | Imminent crisis, insufficient cover |

---

### 4.2 Net Usable Reserves (PBOC Adjustment)

**Location:** Merged panel (`slfsi_monthly_panel.csv`)
**Column:** `net_usable_reserves_usd_m`

**Formula:**
```
Net Usable Reserves = Gross Reserves - PBOC Swap Amount
```

**PBOC Swap Details:**
- **Amount:** USD 1,500 million
- **Start Date:** March 1, 2021
- **Status:** Encumbered (subject to conditionalities, not freely usable)

**Rationale:** The China-Sri Lanka bilateral currency swap was activated in March 2021 but is subject to usage conditions and must be repaid. It inflates gross reserves without providing equivalent liquidity.

---

### 4.3 Net Import Cover

**Location:** Merged panel (`slfsi_monthly_panel.csv`)
**Column:** `net_import_cover_months`

**Formula:**
```
Net Import Cover (months) = Net Usable Reserves (USD) / Monthly Imports (USD)
```

---

### 4.4 Greenspan-Guidotti Ratio

**Calculated in:** `app_reserve_adequacy.py`

**Formula:**
```
GG Ratio = Gross Reserves (USD) / Short-term External Debt (USD)
```

**Data Sources:**
- Numerator: `gross_reserves_usd_m` from reserve assets
- Denominator: `govt_short_term_usd_m` from external debt (quarterly)

**Threshold:** GG ≥ 1.0 indicates ability to service all short-term debt without external financing.

**Conversion for LKR-denominated debt:**
```
Short-term Debt (USD) = total_short_term_lkr_m / usd_lkr
```

---

### 4.5 IMF Assessing Reserve Adequacy (ARA) Metric

**Calculated in:** `app_reserve_adequacy.py`

**Formula:**
```
ARA = 5% × Annual Exports + 5% × Broad Money (M2 in USD) + 30% × Short-term Debt + 15% × Portfolio Liabilities
```

**Component Breakdown:**

| Component | Weight | Source Column | Transformation |
|-----------|--------|---------------|----------------|
| Annual Exports | 5% | `exports_usd_m` | Sum quarterly, annualize (×4) |
| Broad Money M2 | 5% | `broad_money_m2_lkr_m` | Convert to USD using `usd_lkr` |
| Short-term Debt | 30% | `govt_short_term_usd_m` | Direct use (quarterly) |
| Portfolio Liabilities | 15% | `portfolio_equity + portfolio_debt` | Sum from IIP (quarterly) |

**ARA Ratio:**
```
ARA Ratio = Actual Reserves / ARA Requirement
```

**Threshold:** ARA Ratio ≥ 100% indicates adequate reserves; 100-150% is the recommended range.

---

### 4.6 Real Policy Rate

**Location:** Merged panel (`slfsi_monthly_panel.csv`)
**Column:** `real_policy_rate`

**Formula:**
```
Real Policy Rate = Nominal Policy Rate - Inflation (YoY)
```

**Data Sources:**
- `policy_ceiling` (CBSL Standing Lending Facility Rate)
- `ncpi_yoy_pct` (National CPI, year-on-year percentage change)

---

## 5. Data Coverage Summary

### 5.1 Coverage by Series (Crisis Period: 2020-2024)

| Metric | Start Date | End Date | Frequency | Crisis Coverage |
|--------|------------|----------|-----------|-----------------|
| Gross Reserves | Nov 2013 | Dec 2025 | Monthly | ✅ Complete |
| Government Debt (LKR) | Dec 2000 | Q3 2025 | Quarterly | ✅ Complete |
| External Debt (USD) | Q4 2012 | Q3 2025 | Quarterly | ✅ Complete |
| Portfolio Liabilities | Q4 2012 | Q3 2025 | Quarterly | ✅ Complete |
| Broad Money M2 | Dec 1995 | Sep 2025 | Monthly | ✅ Complete |
| Imports | Jan 2007 | Nov 2025 | Monthly | ✅ Complete |
| Exports | Jan 2007 | Nov 2025 | Monthly | ✅ Complete |
| Tourism | Jan 2009 | Nov 2025 | Monthly | ✅ Complete |
| Remittances | Jan 2009 | Nov 2025 | Monthly | ✅ Complete |
| CSE Flows | Jan 2012 | Dec 2025 | Monthly | ✅ Complete |
| FX Rate (USD/LKR) | Jan 2005 | Dec 2025 | Daily/Monthly | ✅ Complete |

### 5.2 Data Limitations

1. **Quarterly vs Monthly:** Debt data is quarterly, requiring interpolation or quarterly-only analysis
2. **PBOC Swap Opacity:** Exact terms and utilization of the swap are not publicly disclosed
3. **Single Crisis Sample:** Cannot fully backtest against 2008-2009 or 2011-2012 stress periods due to data gaps
4. **Reporting Lags:** Official CBSL data published with 1-2 month lag

---

## 6. References and URLs

### 6.1 Primary Data Sources

| Source | URL |
|--------|-----|
| CBSL Statistical Tables | https://www.cbsl.gov.lk/en/statistics/statistical-tables |
| CBSL External Sector Statistics | https://www.cbsl.gov.lk/en/statistics/statistical-tables/external-sector |
| CBSL Monetary Sector Statistics | https://www.cbsl.gov.lk/en/statistics/statistical-tables/monetary-sector |
| CBSL SDDS (Debt Data) | https://www.cbsl.gov.lk/en/statistics/sdds-sri-lanka |
| CBSL Data Library | https://www.cbsl.gov.lk/en/statistics/statistical-data-library |

### 6.2 Specific CBSL Tables Referenced

| Table | Description | Data Category |
|-------|-------------|---------------|
| Table 2.02 | Monthly Exports | Trade |
| Table 2.04 | Monthly Imports | Trade |
| Table 2.12 | Outstanding External Debt | External Debt |
| Table 2.14.1 | Tourism Earnings | External Flows |
| Table 2.14.2 | Workers' Remittances | External Flows |
| Table 2.14.3 | CSE Portfolio Flows | External Flows |
| Table 4.02 | Monetary Aggregates | Money Supply |
| Table 4.11 | Reserve Money, Multipliers, Velocity | Money Supply |
| IIP Release | International Investment Position | External Position |
| SDDS | Central Government Debt | Government Debt |
| Historical Series | Official Reserve Assets | Reserves |

### 6.3 Methodology References

| Metric | Reference |
|--------|-----------|
| Greenspan-Guidotti Rule | Greenspan, A. (1999). "Currency reserves and debt." IMF/World Bank remarks |
| IMF ARA Metric | IMF (2011). "Assessing Reserve Adequacy." IMF Policy Paper |
| Import Cover | IMF traditional metric; 3-month minimum guideline |

---

## Appendix: Column Header Cross-Reference

This table maps the column headers specified in the research plan to their actual file locations:

| Research Plan Header | Actual Column | File Location |
|---------------------|---------------|---------------|
| `total_debt_lkr_m` | `total_debt_lkr_m` | central_govt_debt_quarterly.csv |
| `domestic_debt_lkr_m` | `domestic_debt_lkr_m` | central_govt_debt_quarterly.csv |
| `domestic_short_term_lkr_m` | `domestic_short_term_lkr_m` | central_govt_debt_quarterly.csv |
| `domestic_medium_long_lkr_m` | `domestic_medium_long_lkr_m` | central_govt_debt_quarterly.csv |
| `foreign_debt_lkr_m` | `foreign_debt_lkr_m` | central_govt_debt_quarterly.csv |
| `foreign_short_term_lkr_m` | `foreign_short_term_lkr_m` | central_govt_debt_quarterly.csv |
| `total_short_term_lkr_m` | `total_short_term_lkr_m` | central_govt_debt_quarterly.csv |
| `portfolio_liabilities` | `portfolio_equity + portfolio_debt` | iip_quarterly_2025.csv (derived) |
| `gross_reserves` | `gross_reserves_usd_m` | reserve_assets_monthly_cbsl.csv |
| `import_cover_months` | `import_cover_months` | slfsi_monthly_panel.csv (derived) |
| `Reserve_money_total_lkr` | `reserve_money_total_lkr_m` | reserve_money_velocity_monthly.csv |
| `money_multiplier_m1` | `money_multiplier_m1` | reserve_money_velocity_monthly.csv |
| `cse_inflows_usd_m` | `cse_inflows_usd_m` | cse_flows_monthly.csv |
| `cse_outflows_usd_m` | `cse_outflows_usd_m` | cse_flows_monthly.csv |
| `cse_net_usd_m` | `cse_net_usd_m` | cse_flows_monthly.csv |
| `remittances_usd_m` | `remittances_usd_m` | remittances_monthly.csv |
| `tourism_earnings_usd_m` | `tourism_earnings_usd_m` | tourism_earnings_monthly.csv |
| `reserve_money_m0_lkr_m` | `reserve_money_m0_lkr_m` | monetary_aggregates_monthly.csv |
| `broad_money_m2_lkr_m` | `broad_money_m2_lkr_m` | monetary_aggregates_monthly.csv |
| `gross_reserves_usd_m` | `gross_reserves_usd_m` | reserve_assets_monthly_cbsl.csv |
| `fx_reserves_usd_m` | `fx_reserves_usd_m` | reserve_assets_monthly_cbsl.csv |
| `imf_position_usd_m` | `imf_position_usd_m` | reserve_assets_monthly_cbsl.csv |
| `sdrs_usd_m` | `sdrs_usd_m` | reserve_assets_monthly_cbsl.csv |
| `gold_usd_m` | `gold_usd_m` | reserve_assets_monthly_cbsl.csv |
| `other_reserves_usd_m` | `other_reserves_usd_m` | reserve_assets_monthly_cbsl.csv |
| `imports_usd_m` | `imports_usd_m` | monthly_imports_usd.csv |
| `import_cover` | `import_cover_months` | slfsi_monthly_panel.csv (derived) |
| `net_reserves` | `net_usable_reserves_usd_m` | slfsi_monthly_panel.csv (derived) |
| `net_import_cover` | `net_import_cover_months` | slfsi_monthly_panel.csv (derived) |
| `govt_total_usd_m` | `govt_total_usd_m` | external_debt_usd_quarterly.csv |

---

## 7. Variables Used in Final Benchmarking Analysis

This section documents the specific variables, transformations, and thresholds used in the final reserve adequacy benchmarking analysis (January 2026). See `DOCS/RESERVE_ADEQUACY_BENCHMARKING.md` for full results.

### 7.1 Benchmark 1: Import Cover Ratio

**Purpose:** Measures how many months of imports can be financed by current reserves.

| Variable | Source File | Column Used | Transformation |
|----------|-------------|-------------|----------------|
| Gross Reserves | `reserve_assets_monthly_cbsl.csv` | `gross_reserves_usd_m` | None (monthly) |
| Monthly Imports | `monthly_imports_usd.csv` | `imports_usd_m` | None (monthly) |
| Import Cover | Calculated | — | `gross_reserves_usd_m / imports_usd_m` |

**Thresholds Applied:**
| Threshold | Value | Lead Time (2022 Default) |
|-----------|-------|--------------------------|
| Comfortable | ≥ 6 months | — |
| IMF Minimum | ≥ 3 months | First breach: Mar 2017 (62 mo before) |
| Warning | < 2 months | First breach: Jul 2021 (9 mo before) |
| Critical | < 1 month | First breach: Nov 2021 (5 mo before) |

**Net Import Cover Adjustment:**
```python
# PBOC swap activated March 2021
pboc_swap = 1500  # USD millions
net_reserves = gross_reserves - pboc_swap  # for dates >= 2021-03-01
net_import_cover = net_reserves / imports_usd_m
```

---

### 7.2 Benchmark 2: Greenspan-Guidotti Ratio

**Purpose:** Measures ability to repay all short-term external debt without new financing.

| Variable | Source File | Column Used | Transformation |
|----------|-------------|-------------|----------------|
| Gross Reserves | `reserve_assets_monthly_cbsl.csv` | `gross_reserves_usd_m` | Quarterly resampling (last) |
| Short-term Debt (USD) | `external_debt_usd_quarterly.csv` | `govt_short_term_usd_m` | None (quarterly) |
| GG Ratio | Calculated | — | `gross_reserves / short_term_debt` |

**Formula:**
```python
# Quarterly reserves
reserves_q = reserves.set_index('date').resample('Q').last().reset_index()

# Merge with debt
gg = reserves_q.merge(ext_debt[['date', 'govt_short_term_usd_m']], on='date')
gg['gg_ratio'] = gg['gross_reserves_usd_m'] / gg['govt_short_term_usd_m']
```

**Thresholds Applied:**
| Threshold | Value | Interpretation |
|-----------|-------|----------------|
| Adequate | ≥ 1.0 | Can cover all ST debt |
| Near-breach | 1.0 - 1.5 | Elevated rollover risk |
| Breach | < 1.0 | Cannot cover ST debt |

**Key Finding:** Minimum GG ratio was 1.02 (Q3 2021) - near-breach but no technical breach.

---

### 7.3 Benchmark 3: IMF ARA Metric (Full)

**Purpose:** Comprehensive reserve adequacy assessment using IMF's weighted formula.

| Component | Weight | Source File | Column Used | Transformation |
|-----------|--------|-------------|-------------|----------------|
| Annual Exports | 5% | `monthly_exports_usd.csv` | `exports_usd_m` | Sum quarterly × 4 (annualize) |
| Broad Money M2 | 5% | `monetary_aggregates_monthly.csv` | `broad_money_m2_lkr_m` | Quarterly last, convert to USD |
| Short-term Debt | 30% | `external_debt_usd_quarterly.csv` | `govt_short_term_usd_m` | None (quarterly) |
| Portfolio Liabilities | 15% | `iip_quarterly_2025.csv` | `portfolio_equity` + `portfolio_debt` | Sum components |

**Formulas:**
```python
# M2 to USD conversion
m2_q = m2.set_index('date').resample('Q').last().reset_index()
m2_q = m2_q.merge(fx_q[['date', 'usd_lkr']], on='date')
m2_q['m2_usd_m'] = m2_q['broad_money_m2_lkr_m'] / m2_q['usd_lkr']

# Annualize exports
exports_q = exports.set_index('date').resample('Q').sum().reset_index()
exports_q['annual_exports'] = exports_q['exports_usd_m'] * 4

# Portfolio liabilities
iip['portfolio_liabilities'] = iip['portfolio_equity'].fillna(0) + iip['portfolio_debt'].fillna(0)

# ARA calculation
ara['ara_exports'] = 0.05 * ara['annual_exports']
ara['ara_m2'] = 0.05 * ara['m2_usd_m']
ara['ara_debt'] = 0.30 * ara['govt_short_term_usd_m']
ara['ara_portfolio'] = 0.15 * ara['portfolio_liabilities']
ara['ara_total'] = ara['ara_exports'] + ara['ara_m2'] + ara['ara_debt'] + ara['ara_portfolio']
ara['ara_ratio'] = ara['gross_reserves_usd_m'] / ara['ara_total']
```

**Thresholds Applied:**
| Threshold | Value | Interpretation |
|-----------|-------|----------------|
| Comfortable | ≥ 150% | Strong buffer |
| Adequate | 100-150% | Acceptable |
| Breach | < 100% | Inadequate reserves |

**Key Finding:** ARA breached 100% in Q3 2021 (71%) - 6 months before default.

---

### 7.4 Supplementary Variables Used

| Variable | Source File | Column | Usage |
|----------|-------------|--------|-------|
| USD/LKR Exchange Rate | `slfsi_monthly_panel.csv` | `usd_lkr` | M2 currency conversion |
| Tourism Earnings | `tourism_earnings_monthly.csv` | `tourism_earnings_usd_m` | External flows context |
| Remittances | `remittances_monthly.csv` | `remittances_usd_m` | External flows context |
| CSE Net Flows | `cse_flows_monthly.csv` | `cse_net_usd_m` | Portfolio flow proxy |
| Reserve Money | `reserve_money_velocity_monthly.csv` | `reserve_money_total_lkr_m` | Monetary context |
| Money Multiplier | `reserve_money_velocity_monthly.csv` | `money_multiplier_m1` | Monetary context |

---

### 7.5 Crisis Event Dates Used for Validation

| Date | Event | Used For |
|------|-------|----------|
| 2019-04-21 | Easter Sunday bombings | Pre-crisis baseline |
| 2020-03-01 | COVID-19 pandemic begins | External shock marker |
| 2021-03-01 | PBOC swap activated | Net reserves adjustment |
| 2021-07-01 | Food emergency declared | Early warning validation |
| 2021-09-01 | Economic emergency declared | Early warning validation |
| **2022-04-12** | **Sovereign default announced** | **Primary validation event** |
| 2022-07-05 | Wickremesinghe becomes president | Political transition marker |
| 2023-03-20 | IMF Extended Fund Facility approved | Recovery marker |

---

### 7.6 Scripts and Output Files

**Analysis Scripts:**
| Script | Purpose |
|--------|---------|
| `parse_imports_iip.py` | Parse CBSL imports, exports, and IIP data |
| `parse_cbsl_tables.py` | Parse CBSL monetary and external data |
| `analyze_benchmarks.py` | Calculate threshold breaches and lead times |
| `app_reserve_adequacy.py` | Streamlit dashboard for visualization |

**Output Files Generated:**
| File | Content |
|------|---------|
| `DOCS/RESERVE_ADEQUACY_BENCHMARKING.md` | Full analysis results |
| `data/external/monthly_imports_usd.csv` | Parsed monthly imports |
| `data/external/monthly_exports_usd.csv` | Parsed monthly exports |
| `data/external/iip_quarterly_2025.csv` | Updated IIP with portfolio liabilities |

---

### 7.7 Key Metrics Summary Table

| Metric | Formula | Data Frequency | Coverage | Primary Threshold |
|--------|---------|----------------|----------|-------------------|
| Import Cover | Reserves / Imports | Monthly | Nov 2013 - Nov 2025 | ≥ 3 months |
| Net Import Cover | (Reserves - PBOC) / Imports | Monthly | Mar 2021 - Nov 2025 | ≥ 3 months |
| Greenspan-Guidotti | Reserves / ST Debt | Quarterly | Q4 2012 - Q3 2025 | ≥ 1.0 |
| IMF ARA Ratio | Reserves / ARA | Quarterly | Q4 2012 - Q3 2025 | ≥ 100% |

---

*Section added: January 2026*
*Analysis documented in: RESERVE_ADEQUACY_BENCHMARKING.md*

---

## 8. Net Foreign Assets (NFA) Series

This section documents the Net Foreign Assets data series, which provides a complementary view to gross reserves by capturing the **net position** (assets minus liabilities) of the banking system's foreign exchange holdings.

### 8.1 Why NFA Matters for Reserve Adequacy

**Gross Reserves** show only the asset side of the central bank's forex position. **Net Foreign Assets** reveal the true picture by accounting for foreign liabilities. During Sri Lanka's 2022 crisis:
- Gross reserves appeared low but positive in late 2021
- NFA of the Monetary Authority turned **deeply negative**, indicating foreign liabilities exceeded foreign assets
- This distinction is critical for assessing true reserve adequacy

### 8.2 NFA Data Sources

#### 8.2.1 Net Foreign Assets - Banking System (Aggregate)

**Source File:** `data/external/money_supply_monthly_clean.csv`
**CBSL Source:** Money Supply - Monetary Aggregates and Related Components
**Table Reference:** CBSL eResearch Data Library
**URL:** https://www.cbsl.lk/eresearch/

| Column | Definition | Unit | Notes |
|--------|------------|------|-------|
| `date` | Month-end date | Date | Format: YYYY-MM-DD |
| `net_foreign_assets` | Total NFA of banking system | LKR millions | Monetary Authority + Commercial Banks |

**Data Range:** Jan 2005 - Oct 2024 (with gaps 2009-2020)
**Key Observation:** NFA dropped from +427,759 LKR million (Oct 2010) to -2,195,171 LKR million (Apr 2022)

---

#### 8.2.2 Net Foreign Assets - Detailed Breakdown

**Source File:** `data/raw/net_foreign_assets.xls` (HTML format)
**CBSL Source:** CBSL eResearch - Money Supply Foreign Assets of the Banking System
**URL:** https://www.cbsl.lk/eresearch/

| Series | Column Name | Unit | Data Range | Definition |
|--------|-------------|------|------------|------------|
| NFA - Monetary Authorities | `nfa_monetary_auth_lkr_b` | LKR billion | Jan 2021 - Feb 2024 | CBSL's net foreign asset position |
| NFA - Commercial Banks (Total) | `nfa_comm_banks_total_lkr_b` | LKR billion | Jan 2021 - Feb 2024 | All commercial banks' NFA |
| NFA - Commercial Banks (DBUs) | `nfa_comm_banks_dbu_lkr_b` | LKR billion | Jan 2021 - Feb 2024 | Domestic Banking Units only |
| NFA - Commercial Banks (OBUs) | `nfa_comm_banks_obu_lkr_b` | LKR billion | Jan 2021 - Feb 2024 | Offshore Banking Units only |

**Formulas:**
```
NFA (Monetary Authorities) = CBSL Foreign Assets - CBSL Foreign Liabilities
NFA (Commercial Banks) = Commercial Bank Foreign Assets - Commercial Bank Foreign Liabilities
NFA (Total Banking System) = NFA (Monetary Auth) + NFA (Commercial Banks)
```

---

#### 8.2.3 Processed NFA with REER

**Source File:** `data/processed/D14_reer_nfa.csv`
**Description:** Combined dataset with NFA and Real Effective Exchange Rate

| Column | Definition | Unit | Notes |
|--------|------------|------|-------|
| `date` | Month-end date | Date | Format: YYYY-MM-DD |
| `Net_Foreign_Assets_of_the_Monetary_Autho` | CBSL NFA | LKR millions | Negative values in parentheses or with minus |
| `Net_Foreign_Assets_of_the_Commercial_Ban` | Commercial Banks NFA | LKR millions | DBUs + OBUs combined |
| `Real_Effective_Exchange_Rate_Index` | REER Index | Index (base=100) | Competitiveness indicator |

**Data Range:** Jan 2010 - Jul 2025 (NFA: Jan 2021 - Feb 2024)

---

### 8.3 NFA Crisis Timeline

| Date | Monetary Auth NFA (LKR B) | Commercial Banks NFA (LKR B) | Event |
|------|---------------------------|------------------------------|-------|
| Jan 2021 | +417.9 | -739.0 | Pre-PBOC swap baseline |
| Jul 2021 | +10.3 | -709.6 | Reserves depleting rapidly |
| Aug 2021 | -83.9 | -617.1 | **NFA turns negative** |
| Apr 2022 | -1,462.2 | -732.9 | Default month peak negative |
| Dec 2022 | -1,613.9 | -153.0 | Commercial banks recover faster |
| Feb 2024 | -686.1 | +365.9 | Recovery underway |

---

## 9. IMF SDDS Reserve Data Template (Table 2.15.2)

### 9.1 Overview

The Reserve Data Template follows the IMF's Special Data Dissemination Standard (SDDS) and provides comprehensive reserve adequacy information including **predetermined short-term drains** on reserves.

**Source File:** `data/manual_extraction/table2.15.2_20251231_e.xlsx`
**CBSL Source:** Reserve Data Template - Historical
**URL:** https://www.cbsl.gov.lk/en/statistics/statistical-tables/external-sector

### 9.2 Official Reserve Assets Composition (as of latest)

| Component | Amount (USD mn) | Share of Total |
|-----------|-----------------|----------------|
| **Total Official Reserve Assets** | 7,005.11 | 100% |
| Foreign Currency Reserves | 6,001.30 | 85.7% |
| → Securities | 3,287.28 | 46.9% |
| → Deposits (other central banks) | 957.24 | 13.7% |
| → Deposits (foreign banks) | 1,756.78 | 25.1% |
| Reserve Position in IMF | 73.47 | 1.0% |
| Special Drawing Rights (SDRs) | 15.50 | 0.2% |
| Gold | 913.87 | 13.0% |
| Other Reserve Assets | 0.98 | <0.1% |

### 9.3 Predetermined Short-Term Net Drains

These are scheduled outflows that will reduce usable reserves within the next 12 months:

| Drain Type | Total (USD mn) | < 1 month | 1-3 months | 3-12 months |
|------------|----------------|-----------|------------|-------------|
| **FC Loans Principal Outflows** | -3,212.31 | -177.40 | -401.13 | -2,633.78 |
| **FC Loans Interest Outflows** | -1,252.41 | -44.37 | -255.37 | -952.67 |
| **Forward/Futures Short Positions** | -2,063.00 | -212.87 | -147.00 | -1,703.13 |
| **Reverse Repo Inflows** | +819.37 | +819.37 | — | — |

**Net Predetermined Drains (12-month):** Approximately USD 5.7 billion in outflows

### 9.4 Net Usable Reserves Calculation

For true reserve adequacy, the IMF recommends:

```
Net Usable Reserves = Gross Official Reserves - Predetermined Drains - Encumbered Assets

Where:
- Predetermined Drains: Scheduled principal + interest payments
- Encumbered Assets: PBOC swap ($1.5B), gold held as collateral, etc.
```

### 9.5 Data Limitations and Notes

1. **SWAP Rollover Assumption:** Per CBSL footnotes, "A major share of SWAP outstanding will be rolled over" — drains may be lower than scheduled
2. **ACU Balances:** Reserves include Asian Clearing Union balances which have specific usage constraints
3. **Gold Liquidity:** While counted in reserves, gold liquidation takes time and may not be immediately usable
4. **Reporting Frequency:** Template updated monthly with ~1 month lag

---

## 10. Data Gaps and Bridging Strategy

### 10.1 Primary Data Gap: Gross Reserves Pre-2013

**Challenge:** Official Reserve Assets monthly data only available from Nov 2013

**Potential Bridging Sources:**
| Source | Coverage | Limitation |
|--------|----------|------------|
| IIP Reserve Assets | Q4 2012 onwards | Quarterly only |
| CBSL Annual Reports | 1950s onwards | Annual only, manual extraction |
| IMF IFS Database | 1960 onwards | Monthly, but requires subscription |
| World Bank WDI | 1960 onwards | Annual only |

### 10.2 NFA as Complementary Indicator

**Advantage:** NFA from money supply tables available back to 2005 (monthly)

**Use Case:** While not a direct substitute for gross reserves, NFA trends can:
- Signal deteriorating forex position earlier than gross reserves
- Capture commercial bank vulnerabilities missed by central bank reserves alone
- Provide cross-check for reserve data accuracy

### 10.3 Recommended Actions

1. **Short-term:** Use existing Nov 2013+ reserve data for primary analysis
2. **Medium-term:** Extract annual reserve data from CBSL Annual Reports for 2005-2013
3. **Long-term:** Request IMF IFS database access for complete monthly series

---

## 11. References and URLs (Updated)

### 11.1 New Data Sources Added

| Source | URL | Data Category |
|--------|-----|---------------|
| CBSL eResearch Data Library | https://www.cbsl.lk/eresearch/ | NFA, Monetary Survey |
| Reserve Data Template Historical | https://www.cbsl.gov.lk/en/statistics/statistical-tables/external-sector | SDDS Reserve Template |
| CBSL Reserve Position Clarification | https://www.cbsl.gov.lk/en/reserve-position | Reserve composition methodology |

### 11.2 Additional CBSL Tables Identified

| Table | Description | Data Category | Coverage |
|-------|-------------|---------------|----------|
| Table 2.15.1 | Reserve Data Template - Latest | SDDS Reserves | Current month |
| Table 2.15.2 | Reserve Data Template - Historical | SDDS Reserves | Historical |
| Money Supply - NFA | Foreign Assets of Banking System | NFA breakdown | 2021-2024 (detailed) |
| Monetary Survey | Banking sector monetary aggregates | M2, NFA | Dec 1995 onwards |

---

*Section added: January 2026*
*Net Foreign Assets analysis supports Reserve Adequacy Threshold Research*

---

*Document generated for SL-FSI Reserve Adequacy Threshold Analysis*
