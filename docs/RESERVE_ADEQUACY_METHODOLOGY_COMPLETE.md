# Reserve Adequacy Analysis: Complete Methodology

**Sri Lanka Financial Stress Index Project (SL-FSI)**
**Date:** January 2026
**Scope:** Reserve adequacy thresholds, forecasting, and structural BoP modelling

---

## 1. Introduction and Research Motivation

Sri Lanka's sovereign default of April 12, 2022 exposed the inadequacy of headline gross reserve figures as predictors of external crisis. Gross official reserves, as reported by the Central Bank of Sri Lanka (CBSL), included approximately USD 1.5 billion in encumbered assets from a bilateral currency swap with the People's Bank of China (PBOC), masking the true liquidity position of the country. This research constructs a multi-benchmark reserve adequacy framework, evaluates its ex-ante predictive power against the 2022 default, and extends it with forward-looking forecasting using both traditional extrapolation and a structural Balance of Payments (BoP) accumulation model.

The analysis is implemented as an interactive Streamlit dashboard (`app_reserve_adequacy.py`) comprising eight analytical tabs, of which this document covers the reserve adequacy thresholds (Tabs 1-4) and the associated computational infrastructure.

### 1.1 Research Questions

1. Which reserve adequacy benchmarks would have provided actionable early warning of the 2022 crisis, and with what lead time?
2. Can a composite reserve requirement path be projected forward using IMF ARA components under multiple stress scenarios?
3. Does adjusting for reserve quality (PBOC swap encumbrance) materially alter the adequacy assessment?
4. Can a structural BoP flow model produce more economically grounded reserve projections than univariate growth extrapolation?

---

## 2. Data Sources and Construction

### 2.1 Primary Data Sources

All data originates from the Central Bank of Sri Lanka (CBSL) statistical tables, accessed via the CBSL Data Library and Statistical Tables portal. The table below summarizes each series:

| Series | CBSL Table | File | Frequency | Range |
|--------|-----------|------|-----------|-------|
| Official Reserve Assets | Historical Data Series | `reserve_assets_monthly_cbsl.csv` | Monthly | Nov 2013 - Dec 2025 |
| Merchandise Exports | Table 2.02 | `monthly_exports_usd.csv` | Monthly | Jan 2007 - Nov 2025 |
| Merchandise Imports | Table 2.04 | `monthly_imports_usd.csv` | Monthly | Jan 2007 - Nov 2025 |
| External Debt (USD) | Table 2.12 | `external_debt_usd_quarterly.csv` | Quarterly | Q4 2012 - Q3 2025 |
| Tourism Earnings | Table 2.14.1 | `tourism_earnings_monthly.csv` | Monthly | Jan 2009 - Dec 2024 |
| Workers' Remittances | Table 2.14.2 | `remittances_monthly.csv` | Monthly | Jan 2009 - Oct 2025 |
| CSE Portfolio Flows | Table 2.14.3 | `cse_flows_monthly.csv` | Monthly | Jan 2012 - Dec 2025 |
| Monetary Aggregates | Table 4.02 | `monetary_aggregates_monthly.csv` | Monthly | Dec 1995 - Sep 2025 |
| Int'l Investment Position | IIP Release | `iip_quarterly_2025.csv` | Quarterly | Q4 2012 - Q3 2025 |
| Exchange Rate (USD/LKR) | Merged Panel | `slfsi_monthly_panel.csv` | Monthly | Jan 2005 - Dec 2025 |

### 2.2 Data Cleaning and Feature Engineering

#### 2.2.1 Temporal Alignment

A persistent data engineering challenge arose from inconsistent date conventions across CBSL releases. Reserve assets and trade data use end-of-month dates; external debt uses end-of-quarter dates; but the International Investment Position (IIP) uses first-of-quarter-month dates (e.g., `2025-09-01` instead of `2025-09-30`). This misalignment caused portfolio liabilities to produce all-NaN values on inner joins. The fix normalises IIP dates to end-of-quarter using:

```python
pl['date'] = pl['date'] + pd.offsets.QuarterEnd(0)
```

This maps any date within a quarter to that quarter's final day, enabling correct merges with other quarterly series.

#### 2.2.2 Forward-Filling for Trailing Quarters

Monthly series (reserves, M2) extend further in time than quarterly series (external debt, IIP). When reserves data extends to Q4 2025 but M2 and debt data only reach Q3 2025, the final quarter would have NaN components, producing a spurious ARA ratio. Forward-filling (`ffill`) is applied to M2, short-term debt, and portfolio liabilities so that the most recent observation carries through to the trailing quarter. This is econometrically defensible: these stock variables are slow-moving and the most recent observation is the best available estimate.

#### 2.2.3 Exports Annualisation

The IMF ARA formula requires annual exports. Early implementations used `quarterly_sum * 4`, which overstates annualised exports in quarters with only 2 months of data and understates in recovery quarters. This was replaced with a rolling four-quarter sum:

```python
exports_q['annual_exports_usd_m'] = exports_q['quarterly_exports_usd_m'].rolling(4, min_periods=1).sum()
```

With `min_periods=1`, the first available quarters use however many periods are available, producing a progressively better estimate as data accumulates.

#### 2.2.4 M2 Currency Conversion

Broad money (M2) is reported in LKR millions by CBSL. Conversion to USD is required for the ARA formula. End-of-quarter USD/LKR exchange rates are used:

```
M2_USD = M2_LKR / USD_LKR
```

The choice of end-of-quarter spot rate (rather than period average) aligns with the stock nature of M2 — it represents a balance sheet position at a point in time, not a flow.

#### 2.2.5 PBOC Swap Adjustment

The CBSL activated a CNY 10 billion bilateral currency swap with the PBOC on March 19, 2021. This adds approximately USD 1,500 million to gross official reserves, but the IMF considers this conditionally usable — available only when gross reserves excluding the swap exceed three months of the previous year's imports. All net reserve calculations subtract USD 1,500M for dates on or after March 1, 2021:

```
Net Reserves = Gross Reserves - 1,500   (for date >= 2021-03-01)
```

---

## 3. Reserve Adequacy Benchmarks

### 3.1 Import Cover Ratio

**Definition:**
$$\text{Import Cover (months)} = \frac{\text{Gross Reserves (USD)}}{\text{Monthly Imports (USD)}}$$

**Economic rationale:** Measures the external payment buffer — how many months a country can sustain its import bill without additional foreign exchange inflows. The IMF traditionally recommends a minimum of three months.

**Thresholds applied:**

| Level | Months | Interpretation |
|-------|--------|----------------|
| Comfortable | >= 6 | Adequate external buffer |
| IMF Minimum | >= 3 | Minimum acceptable |
| Warning | < 2 | Elevated depletion risk |
| Critical | < 1 | Imminent crisis |

**Net Import Cover** adjusts the numerator by subtracting the PBOC swap. This metric first breached the critical threshold (< 1 month) in November 2021 — five months before the April 2022 default.

### 3.2 Greenspan-Guidotti Ratio

**Definition:**
$$\text{GG Ratio} = \frac{\text{Gross Reserves (USD)}}{\text{Short-term External Debt (USD)}}$$

**Economic rationale:** Proposed independently by Alan Greenspan (1999) and Pablo Guidotti (1999), this ratio captures rollover risk. A ratio below 1.0 implies that a country cannot cover its short-term obligations maturing within 12 months without access to external financing — a classic trigger for sudden-stop crises.

**Implementation detail:** Short-term external debt uses `total_short_term_usd_m` from `external_debt_usd_quarterly.csv`, which encompasses government, central bank, and other sector obligations. This is preferred over `govt_short_term_usd_m` alone because rollover risk applies to all external obligors.

**Finding:** The GG ratio reached a minimum of 1.02 in Q3 2021 — a near-breach that coincided with the economic emergency declaration, providing six months of lead time before default.

### 3.3 IMF Assessing Reserve Adequacy (ARA) Metric

**Definition:**
$$\text{ARA} = 0.05 \times X + 0.05 \times M_2 + 0.30 \times D_{ST} + 0.15 \times L_P$$

where $X$ = annual exports (USD), $M_2$ = broad money (USD), $D_{ST}$ = short-term external debt (USD), and $L_P$ = portfolio liabilities (USD).

**Economic rationale (IMF 2011, 2013):** Each component proxies a distinct channel of reserve drain during exchange market pressure (EMP) episodes:

| Component | Weight | Drain Channel |
|-----------|--------|---------------|
| Exports (5%) | Current account shock — sudden drop in export earnings from terms-of-trade deterioration |
| M2 (5%) | Capital flight — domestic residents converting local currency deposits into foreign assets |
| Short-term Debt (30%) | Rollover failure — inability to refinance maturing obligations |
| Portfolio Liabilities (15%) | Portfolio outflows — non-resident investors liquidating equity and bond holdings |

The weights are calibrated from cross-country EMP episodes (IMF, 2011). Sri Lanka is classified under the managed floating exchange rate regime, for which the IMF applies the same weights as fixed regimes.

**Adequacy thresholds:** The IMF recommends reserves in the range of 100-150% of ARA. Below 100% indicates inadequate reserves; above 150% indicates comfortable reserves.

**Implementation:**
- `calculate_imf_ara()` in `app_reserve_adequacy.py` performs the full calculation
- Quarterly frequency is dictated by the lowest-frequency inputs (debt, portfolio liabilities)
- Portfolio liabilities are computed as the sum of portfolio equity and portfolio debt from the IIP

### 3.4 Net ARA Ratio (New)

**Definition:**
$$\text{Net ARA Ratio} = \frac{\text{Net Reserves (excl. PBOC swap)}}{\text{ARA Requirement}}$$

**Rationale:** The standard ARA ratio uses gross reserves in the numerator, which for Sri Lanka includes the PBOC swap. The IMF's own programme documents (Country Report No. 25/339) note that the swap "becomes available once the GIR without it is above 3 months of the previous year's imports." The net ARA ratio provides the adequacy picture as seen by the IMF.

**Latest values (Q4 2025):** Gross ARA = 182.9%, Net ARA = 142.7%. The 40 percentage point gap represents the PBOC swap distortion. While gross reserves appear comfortable, net reserves are only marginally above the 150% comfort threshold. This has material implications for policy: the CBSL's true reserve buffer is substantially thinner than headline figures suggest.

**Pre-2021 behaviour:** Before the swap was activated (March 2021), net and gross ARA ratios are identical. The two lines diverge from Q1 2021 onward, providing a visual measure of the swap's distorting effect.

---

## 4. Early Warning Performance

### 4.1 Lead Time Analysis

The table below summarises the first breach date, lead time relative to the April 12, 2022 default, and assessment of each benchmark's early warning utility:

| Benchmark | First Breach | Lead Time | Specificity |
|-----------|-------------|-----------|-------------|
| Import Cover < 2 months | July 2021 | 9 months | High |
| IMF ARA < 100% | Q3 2021 | 6 months | High |
| GG Ratio < 1.5 | Q3 2021 | 6 months | Medium |
| Import Cover < 1 month | November 2021 | 5 months | Very High |

The optimal early warning window is July-September 2021, when multiple benchmarks simultaneously signalled distress. This six-to-nine month lead time would have been sufficient for preemptive IMF engagement, debt restructuring initiation, or emergency import controls.

### 4.2 Backtesting Against the 2018 Currency Crisis

The LKR depreciated approximately 19% against USD in 2018. Reserve adequacy benchmarks did not breach crisis thresholds during this period (minimum import cover: 3.70 months; minimum ARA ratio: 221%; minimum GG ratio: 4.29), correctly identifying this as a currency confidence shock rather than a solvency crisis. This non-breach demonstrates acceptable specificity — the framework does not generate false positives for currency events that do not threaten external solvency.

---

## 5. Forecasting Framework

### 5.1 Design Philosophy

Two approaches to reserve forecasting are possible:

1. **Requirement forecasting:** Project the ARA requirement forward and compare against actual/extrapolated reserves. This was the original approach.
2. **Structural accumulation forecasting:** Project reserves themselves from underlying BoP flows. This was added in the January 2026 enhancement.

Both are implemented. The ARA requirement components (exports, M2, short-term debt, portfolio liabilities) are always projected using growth extrapolation. The reserve level itself is now projected using the structural BoP model when sufficient data is available, with automatic fallback to growth extrapolation.

### 5.2 Component Growth Extrapolation

**Method:** For each ARA component, the median quarter-on-quarter (QoQ) growth rate over a lookback window (default: 8 quarters) is computed and applied recursively:

$$v_{t+1} = v_t \times (1 + g_{\text{median}} \times m_s)$$

where $v_t$ is the component value at quarter $t$, $g_{\text{median}}$ is the median QoQ growth rate over the lookback window, and $m_s$ is the scenario multiplier.

**Rationale for median over mean:** The median is robust to outlier quarters (e.g., a one-off IMF tranche disbursement or COVID-era import collapse). With only ~50 quarterly observations, outlier sensitivity is a material concern.

**Rationale against VAR/BVAR:** A joint multivariate model was considered but rejected due to: (a) limited quarterly sample size (post-2012); (b) structural breaks from COVID and the 2022 default; (c) parameter explosion relative to degrees of freedom; (d) reduced interpretability and reproducibility for the dashboard context. The component-based approach sacrifices cross-variable dynamics but gains transparency and robustness.

### 5.3 Scenario Design

Three scenarios are applied as multipliers on the median QoQ growth rate:

| Component | Downside | Baseline | Upside |
|-----------|----------|----------|--------|
| Exports | 0.90 | 1.00 | 1.05 |
| M2 | 0.98 | 1.00 | 1.02 |
| Short-term Debt | 1.05 | 1.00 | 0.95 |
| Portfolio Liabilities | 1.05 | 1.00 | 0.95 |
| Imports | 1.05 | 1.00 | 0.98 |
| Remittances | 0.92 | 1.00 | 1.05 |
| Tourism | 0.85 | 1.00 | 1.10 |
| CSE Net Flows | 0.80 | 1.00 | 1.10 |

The downside scenario reflects weaker external demand, elevated debt rollover pressure, and capital outflows. The upside reflects stronger trade performance, controlled liability growth, and portfolio inflows. Multipliers are intentionally conservative (0.8-1.1 range) to avoid implausible extrapolation.

### 5.4 Growth Rate Capping

A maximum QoQ growth rate of 5% (approximately 22% annualised) is applied to BoP component projections. This was necessitated by tourism earnings data, which exhibited a median QoQ growth rate of 27.1% over the most recent 8 quarters — an artifact of the post-COVID recovery from a near-zero base. Without capping, tourism earnings would be projected to reach USD 5.6 billion per quarter within 8 quarters, an implausible figure exceeding Sri Lanka's total annual tourism receipts. The 5% cap allows robust growth while preventing extrapolation of transient recovery dynamics.

---

## 6. Structural BoP Reserve Accumulation Model

### 6.1 Theoretical Foundation

The balance of payments identity states that the change in official reserves equals the current account balance plus the capital/financial account balance (excluding reserves) plus net errors and omissions:

$$\Delta R = CA + KFA + EO$$

In practice, we decompose the observable portion of the current account and capital account flows and calibrate a residual term from historical data:

$$\Delta R_t = \underbrace{(X_t + \text{Rem}_t + \text{Tour}_t)}_{\text{inflows}} - \underbrace{M_t}_{\text{outflows}} + \underbrace{\text{CSE}_t}_{\text{portfolio}} + \underbrace{\varepsilon}_{\text{residual}}$$

where $X_t$ = merchandise exports, $\text{Rem}_t$ = workers' remittances, $\text{Tour}_t$ = tourism earnings, $M_t$ = merchandise imports, $\text{CSE}_t$ = net CSE portfolio flows, and $\varepsilon$ = calibrated residual.

### 6.2 The Residual Term

The residual captures all flows not explicitly modelled:
- **Foreign Direct Investment (FDI) inflows**
- **Debt service outflows** (ISB coupon payments, bilateral/multilateral principal repayments, IMF credit repayment)
- **IMF programme disbursements** (Extended Fund Facility tranches)
- **Central bank FX interventions** (sterilisation operations)
- **Valuation changes** (gold price movements, SDR revaluation, cross-currency effects)
- **Net errors and omissions**

**Calibration method:** For each historical quarter, the residual is computed as:

$$\varepsilon_t = \Delta R_t - (X_t + \text{Rem}_t + \text{Tour}_t - M_t + \text{CSE}_t)$$

The median of the most recent `lookback` quarters (default: 8) is used as the constant residual for all forecast quarters. The median (rather than mean) provides robustness to exceptional quarters such as a large IMF tranche disbursement.

**Calibrated value (Q4 2025):** $\varepsilon = -317$ million USD per quarter. The negative sign indicates that Sri Lanka's unmodelled outflows (primarily debt service) exceed its unmodelled inflows (primarily FDI and IMF tranches). This is consistent with the country's post-restructuring debt service obligations.

### 6.3 Reserve Accumulation

For each forecast quarter, reserve levels are accumulated:

$$R_{t+1} = R_t + \Delta R_{t+1}$$

starting from the last observed reserve level. Each BoP component is projected independently using the capped growth extrapolation method with scenario-specific multipliers.

### 6.4 Advantages Over Growth Extrapolation

The structural BoP model offers several advantages over the prior growth-based reserve projection:

1. **Economic interpretability:** Each component of reserve change has a clear economic meaning. Remittances can be shocked independently of tourism; imports can be stressed via oil price assumptions.
2. **Scenario coherence:** Downside scenarios naturally produce lower reserves because import drains exceed export inflows, rather than simply applying an arbitrary 0.97 multiplier to reserves.
3. **Decomposition analysis:** A stacked bar chart visualises the quarterly flow composition, enabling identification of which components drive reserve accumulation or depletion.
4. **Remittance inclusion:** Workers' remittances (~USD 6.7 billion annually, 53% of export earnings) are now explicitly modelled as a first-class inflow component, addressing a gap identified in the initial methodology review.

### 6.5 Limitations

1. **Constant residual:** The residual is treated as time-invariant. In reality, debt service is lumpy (ISB maturities cluster in specific quarters) and IMF disbursements follow programme review schedules. A decomposed residual separating known debt service from stochastic flows would improve accuracy.
2. **Component independence:** BoP components are projected independently, ignoring co-movement (e.g., higher oil prices simultaneously increase import costs and Gulf-state remittances).
3. **Growth rate cap:** The 5% QoQ cap is a pragmatic heuristic. A mean-reverting or logistic saturation model would better handle post-crisis recovery dynamics.
4. **No FDI modelling:** FDI is absorbed into the residual rather than modelled from determinants (investment climate indicators, sovereign ratings, etc.).

---

## 7. Implementation Architecture

### 7.1 Code Structure

All reserve adequacy logic resides in `app_reserve_adequacy.py` (approximately 1,960 lines). The architecture follows a layered pattern:

**Layer 1 — Constants and Configuration**
- `PBOC_SWAP_USD_M = 1500`: Encumbered swap amount
- `SCENARIO_SHOCKS`: Dictionary of scenario multipliers for 11 components
- Date constants for crisis period, default date, and IMF programme approval

**Layer 2 — Data Loaders** (13 `@st.cache_data` functions)
Each loader reads a CSV from `data/external/`, handles missing files gracefully (returns `None`), and parses dates.

**Layer 3 — Computation Functions**
| Function | Purpose |
|----------|---------|
| `calculate_imf_ara()` | Computes ARA metric, gross and net ARA ratios |
| `calculate_greenspan_guidotti()` | Computes GG ratio |
| `to_quarterly()` | Resamples monthly series to quarterly |
| `forecast_with_growth()` | Median QoQ growth extrapolation with optional cap |
| `prepare_quarterly_imports()` | Aggregates monthly imports to quarterly sums |
| `prepare_quarterly_bop_components()` | Aggregates all monthly BoP flows to quarterly |
| `calibrate_bop_residual()` | Computes median BoP residual over lookback window |
| `forecast_reserves_bop()` | Structural BoP reserve accumulation projection |
| `enrich_with_import_cover()` | Adds import cover and net reserve metrics |
| `build_forecast_scenarios()` | Orchestrates scenario panels with BoP or fallback |

**Layer 4 — UI/Visualisation** (`main()` function)
Eight tabs rendered using Streamlit and Plotly. Reserve adequacy occupies Tabs 1 (Overview), 2 (Reserve Metrics), 3 (IMF ARA), and 4 (Forecasts & Scenarios).

### 7.2 Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Quarterly frequency | End-of-quarter alignment | Dictated by debt and IIP data availability |
| Median growth | Over mean or OLS trend | Robust to outliers in short sample |
| Forward-fill | For trailing incomplete quarters | Avoids NaN-driven ratio spikes |
| 5% QoQ cap | Applied only to BoP components | Prevents post-recovery extrapolation; ARA requirement components uncapped |
| Constant residual | Median over lookback | Robust to IMF tranche outliers; no maturity schedule data available |
| PBOC swap as constant | USD 1,500M for all dates >= 2021-03 | Swap has been rolled over consistently; exact utilisation not disclosed |

---

## 8. Findings

### 8.1 Current Reserve Adequacy Position (Q4 2025)

| Metric | Value | Assessment |
|--------|-------|------------|
| Gross Reserves | USD 6,825M | — |
| Net Reserves (excl. PBOC) | USD 5,325M | — |
| Import Cover (gross) | 5.2 months | Above IMF minimum |
| Import Cover (net) | 4.1 months | Above minimum but below comfortable |
| Greenspan-Guidotti Ratio | ~13.4 | Comfortable |
| Gross ARA Ratio | 182.9% | Above 150% comfortable |
| Net ARA Ratio | 142.7% | Between adequate and comfortable |

### 8.2 ARA Component Breakdown (Q4 2025)

| Component | Value (USD M) | Weight | Contribution |
|-----------|--------------|--------|-------------|
| 5% x Annual Exports ($12,413M) | 621 | 16.6% | Current account buffer |
| 5% x M2 ($45,079M) | 2,254 | 60.4% | Capital flight buffer |
| 30% x ST Debt ($510M) | 153 | 4.1% | Rollover buffer |
| 15% x Portfolio Liabilities ($4,694M) | 704 | 18.9% | Portfolio outflow buffer |
| **ARA Total** | **3,732** | **100%** | — |

M2 dominates the ARA requirement at 60.4%, consistent with the IMF's observation that the M2 component can be disproportionately large for some countries. Short-term external debt contributes only 4.1%, reflecting successful post-restructuring reduction in short-term obligations.

### 8.3 Forecast Results (8-Quarter Horizon, Baseline)

| Metric | Current (Q4 2025) | Projected (Q4 2027) | Direction |
|--------|-------------------|---------------------|-----------|
| Gross Reserves | $6,825M | $6,753M | Slight decline |
| Net Reserves | $5,325M | $5,253M | Slight decline |
| Gross ARA Ratio | 182.9% | 152.5% | Converging toward comfort threshold |
| Net ARA Ratio | 142.7% | 118.6% | Approaching adequacy boundary |
| Import Cover | ~5.2 months | ~5.6 months | Stable |

The baseline BoP model projects reserves initially declining to ~USD 5,775M (Q3 2026) before recovering, driven by the negative residual (debt service drain) partially offset by growing export and remittance inflows. The net ARA ratio converges toward 100% over the forecast horizon, suggesting that without continued reserve accumulation, Sri Lanka's adequacy position could come under pressure.

### 8.4 BoP Residual Interpretation

The calibrated residual of -USD 317M per quarter (approximately -USD 1.3 billion annually) reflects Sri Lanka's post-restructuring debt service burden. This is consistent with the IMF SDDS Reserve Data Template (Table 2.15.2), which reports approximately USD 5.7 billion in predetermined 12-month foreign currency drains, partially offset by IMF programme disbursements and FDI inflows.

---

## 9. Gaps, Limitations, and Future Work

### 9.1 Current Limitations

1. **Constant residual calibration:** The BoP residual treats unmodelled flows as time-invariant. Known debt maturity schedules (ISB maturities, bilateral repayments) are lumpy and could be explicitly modelled using the external debt quarterly data.
2. **Component independence:** No cross-variable dynamics are captured (e.g., oil price → imports + Gulf remittances; exchange rate depreciation → import costs + M2 in USD).
3. **Post-recovery extrapolation:** Tourism's 27% QoQ median growth reflects recovery dynamics, not sustainable long-run growth. The 5% cap is a pragmatic fix; a saturation or mean-reversion model would be more principled.
4. **Single-country, single-crisis validation:** The early warning framework is validated against only the 2022 default. Pre-2013 crises (2008-09, 2011-12) cannot be backtested due to data availability.
5. **PBOC swap treatment:** The swap is treated as a fixed USD 1,500M encumbrance. In practice, the swap amount fluctuates with CNY/USD exchange rates and rollover terms.
6. **No confidence intervals:** Forecasts are point estimates without uncertainty bands.

### 9.2 Planned Enhancements

1. **Residual decomposition:** Split the residual into a known debt service component (from external debt maturity structure) and a stochastic remainder. This would produce quarter-specific residuals for periods with known ISB maturities.
2. **Bootstrap confidence intervals:** Resample historical growth rates to generate forecast distributions rather than point estimates.
3. **Scenario calibration from percentiles:** Tie scenario multipliers to historical growth rate percentiles (25th/50th/75th) rather than arbitrary values.
4. **Reserve quality scorecard:** Combine net ARA ratio, import cover, GG ratio, reserve trajectory, and FX share into a composite health score with traffic-light classification (healthy / stress / crisis).
5. **Policy reaction functions:** Link reserve forecasts to exchange rate regime assumptions and central bank intervention rules.

---

## 10. References

1. IMF (2011). "Assessing Reserve Adequacy." IMF Policy Paper.
2. IMF (2013). "Assessing Reserve Adequacy — Further Considerations." IMF Policy Paper.
3. IMF (2025). "Sri Lanka: Fourth Review Under the Extended Arrangement." Country Report No. 25/339.
4. Greenspan, A. (1999). "Currency Reserves and Debt." Remarks at World Bank Conference.
5. Guidotti, P. (1999). Remarks at G-33 seminar, Bonn.
6. Setser, B. (2019). "It Is Time To Scrap the IMF's Reserve Adequacy Metric." Council on Foreign Relations.
7. IMF (2013). "International Reserves and Foreign Currency Liquidity: Guidelines for a Data Template." IMF Statistics Department.
8. Central Bank of Sri Lanka. Statistical Tables — External Sector. https://www.cbsl.gov.lk/en/statistics/statistical-tables/external-sector
9. Central Bank of Sri Lanka. Workers' Remittances. https://www.cbsl.gov.lk/en/workers-remittances

---

*Document generated: January 2026*
*Implementation: `app_reserve_adequacy.py` in the SL-FSI repository*
