# Specification 01: Variable Sets Definition
## Parsimonious, BoP-Focused, Monetary Policy, and PCA Variable Sets

**Version:** 1.1
**Created:** 2026-02-10
**Status:** :white_check_mark: COMPLETED
**Phase:** 1 (Foundation)
**Dependencies:** None
**Blocks:** All Phase 2 specs

---

## Objective

Define multiple theoretically-motivated variable sets for systematic comparison:
1. **Parsimonious Core** - Minimal economically-motivated set
2. **BoP-Focused** - Balance of Payments drivers
3. **Monetary Policy** - Policy intervention channel
4. **PCA-Reduced** - Data-driven dimensionality reduction
5. **Kitchen Sink** - All available variables (benchmark for overfitting)

---

## Theoretical Motivation

### Reserve Accumulation Dynamics

Foreign exchange reserves change according to:

```
ΔR = CA + KA + VA + EI

Where:
  CA = Current Account (exports - imports + remittances + tourism + investment income)
  KA = Capital Account (FDI, portfolio flows, CSE flows)
  VA = Valuation Adjustments (exchange rate changes on non-USD reserves)
  EI = Exceptional Items (IMF drawings, swap lines, etc.)
```

### Variable Selection Rationale

| Variable | Economic Channel | Expected Sign | Persistence |
|----------|-----------------|---------------|-------------|
| `exports_usd_m` | CA inflow | + | I(1) |
| `imports_usd_m` | CA outflow | - | I(1) |
| `trade_balance_usd_m` | Net CA (goods) | + | I(1) |
| `remittances_usd_m` | CA inflow (services) | + | I(1) |
| `tourism_usd_m` | CA inflow (services) | + | I(1) |
| `cse_net_usd_m` | KA flow (portfolio) | +/- | I(0)/I(1) |
| `usd_lkr` | Valuation + intervention | - (depreciation) | I(1) |
| `m2_usd_m` | Monetary policy stance | - (expansion) | I(1) |

---

## Variable Set Definitions

### Set 1: Parsimonious Core (`varset_parsimonious`)

**Rationale:** Minimum viable set capturing the core reserve dynamics with maximum data availability.

```python
VARSET_PARSIMONIOUS = {
    "name": "parsimonious",
    "target": "gross_reserves_usd_m",
    "arima_exog": ["trade_balance_usd_m", "usd_lkr"],
    "vecm_system": ["gross_reserves_usd_m", "trade_balance_usd_m", "usd_lkr"],
    "var_system": ["gross_reserves_usd_m", "trade_balance_usd_m", "usd_lkr"],
    "description": "Minimal set: net trade + exchange rate",
    "expected_obs": 200,  # Approximate
}
```

**Economic Logic:**
- Trade balance = primary current account driver
- Exchange rate = captures intervention + valuation effects
- Only 3 variables → robust estimation even with limited data

---

### Set 2: Balance of Payments (`varset_bop`)

**Rationale:** Full current account representation without capital flows.

```python
VARSET_BOP = {
    "name": "bop",
    "target": "gross_reserves_usd_m",
    "arima_exog": ["exports_usd_m", "imports_usd_m", "remittances_usd_m", "tourism_usd_m"],
    "vecm_system": ["gross_reserves_usd_m", "exports_usd_m", "imports_usd_m",
                    "remittances_usd_m", "tourism_usd_m"],
    "var_system": ["gross_reserves_usd_m", "exports_usd_m", "imports_usd_m",
                   "remittances_usd_m"],
    "description": "Current account flow decomposition",
    "expected_obs": 180,
}
```

**Economic Logic:**
- Disaggregated flows allow modeling differential dynamics
- Tourism and remittances are major Sri Lankan inflows
- Excludes exchange rate (endogeneity concerns in BoP identity)

---

### Set 3: Monetary Policy (`varset_monetary`)

**Rationale:** Central bank intervention channel focus.

```python
VARSET_MONETARY = {
    "name": "monetary",
    "target": "gross_reserves_usd_m",
    "arima_exog": ["usd_lkr", "m2_usd_m"],
    "vecm_system": ["gross_reserves_usd_m", "usd_lkr", "m2_usd_m"],
    "var_system": ["gross_reserves_usd_m", "usd_lkr", "m2_usd_m"],
    "description": "Monetary policy and exchange rate intervention",
    "expected_obs": 190,
}
```

**Economic Logic:**
- CBSL intervenes via USD sales/purchases affecting reserves
- M2 growth signals monetary stance (sterilization capacity)
- Exchange rate reflects intervention pressure

---

### Set 4: PCA-Reduced (`varset_pca`)

**Rationale:** Data-driven dimensionality reduction to capture common factors.

```python
VARSET_PCA = {
    "name": "pca",
    "target": "gross_reserves_usd_m",
    "source_vars": ["exports_usd_m", "imports_usd_m", "remittances_usd_m",
                    "tourism_usd_m", "usd_lkr", "m2_usd_m", "cse_net_usd_m",
                    "trade_balance_usd_m"],
    "n_components": 3,  # Determined by scree plot / Kaiser criterion
    "arima_exog": ["PC1", "PC2", "PC3"],
    "vecm_system": ["gross_reserves_usd_m", "PC1", "PC2", "PC3"],
    "var_system": ["gross_reserves_usd_m", "PC1", "PC2", "PC3"],
    "description": "Principal components of all macro variables",
    "expected_obs": 140,  # Limited by shortest series
}
```

**Implementation Notes:**
- Standardize all source variables before PCA
- Use training sample only for PCA fit (avoid look-ahead bias)
- Transform validation/test using training loadings
- Interpret components post-hoc via loadings

---

### Set 5: Kitchen Sink (`varset_full`)

**Rationale:** Benchmark for overfitting detection. Should perform worse than parsimonious sets.

```python
VARSET_FULL = {
    "name": "full",
    "target": "gross_reserves_usd_m",
    "arima_exog": ["exports_usd_m", "imports_usd_m", "remittances_usd_m",
                   "tourism_usd_m", "usd_lkr", "m2_usd_m", "cse_net_usd_m",
                   "trade_balance_usd_m"],
    "vecm_system": ["gross_reserves_usd_m", "exports_usd_m", "imports_usd_m",
                    "remittances_usd_m", "usd_lkr", "m2_usd_m", "trade_balance_usd_m"],
    "var_system": ["gross_reserves_usd_m", "exports_usd_m", "imports_usd_m",
                   "remittances_usd_m", "usd_lkr", "m2_usd_m"],
    "description": "All available variables (overfitting benchmark)",
    "expected_obs": 138,
}
```

---

## Implementation Details

### File Structure
```
reserves_project/scripts/academic/
├── prepare_variable_sets.py      ← Main script
├── variable_sets/
│   ├── __init__.py
│   ├── config.py                 ← Variable set definitions
│   ├── pca_builder.py            ← PCA extraction logic
│   └── validators.py             ← Data availability checks
```

### Output Structure
```
data/forecast_prep_academic/
├── varset_parsimonious/
│   ├── arima_dataset.csv
│   ├── vecm_levels.csv
│   ├── var_system.csv
│   └── metadata.json
├── varset_bop/
│   └── ...
├── varset_monetary/
│   └── ...
├── varset_pca/
│   ├── pca_loadings.csv          ← Component loadings
│   ├── pca_variance_explained.csv
│   └── ...
└── varset_full/
    └── ...
```

### PCA Implementation

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def build_pca_factors(df, source_vars, n_components, train_end):
    """
    Extract PCA factors using training data only.

    Parameters:
    -----------
    df : pd.DataFrame
        Full dataset with DatetimeIndex
    source_vars : list
        Variables to extract factors from
    n_components : int
        Number of principal components
    train_end : str
        End date for training period

    Returns:
    --------
    factors_df : pd.DataFrame
        DataFrame with PC1, PC2, ... columns
    loadings : pd.DataFrame
        Component loadings matrix
    variance_explained : np.array
        Variance explained by each component
    """
    # Split data
    train_data = df.loc[:train_end, source_vars].dropna()
    full_data = df[source_vars].dropna()

    # Fit scaler and PCA on training only
    scaler = StandardScaler()
    scaler.fit(train_data)

    pca = PCA(n_components=n_components)
    pca.fit(scaler.transform(train_data))

    # Transform full dataset
    scaled_full = scaler.transform(full_data)
    factors = pca.transform(scaled_full)

    # Create output DataFrame
    factor_cols = [f"PC{i+1}" for i in range(n_components)]
    factors_df = pd.DataFrame(
        factors,
        index=full_data.index,
        columns=factor_cols
    )

    # Loadings for interpretation
    loadings = pd.DataFrame(
        pca.components_.T,
        index=source_vars,
        columns=factor_cols
    )

    return factors_df, loadings, pca.explained_variance_ratio_
```

### Data Availability Validation

```python
def validate_variable_set(df, varset_config, min_obs=100):
    """
    Check if variable set has sufficient observations.

    Returns:
    --------
    dict with:
        - valid: bool
        - n_obs: int
        - missing_vars: list
        - date_range: tuple
    """
    all_vars = set(varset_config.get("arima_exog", []) +
                   varset_config.get("vecm_system", []))

    missing = [v for v in all_vars if v not in df.columns]
    if missing:
        return {"valid": False, "missing_vars": missing}

    subset = df[list(all_vars)].dropna()
    n_obs = len(subset)

    return {
        "valid": n_obs >= min_obs,
        "n_obs": n_obs,
        "missing_vars": [],
        "date_range": (subset.index.min(), subset.index.max())
    }
```

---

## Validation Criteria

### Pre-Execution Checklist
- [ ] Source data available at `data/merged/reserves_forecasting_panel.csv`
- [ ] All variable names confirmed in source data
- [ ] Minimum observation thresholds defined

### Post-Execution Validation
- [ ] All 5 variable sets created
- [ ] Each set has metadata.json with:
  - Variable list
  - Observation count
  - Date range
  - Missing data handling applied
- [ ] PCA set includes:
  - Loadings matrix
  - Variance explained
  - Scree plot saved
- [ ] Sample size comparison table generated

---

## Expected Outputs

### 1. Variable Set Summary Table

| Variable Set | ARIMA Vars | VECM Vars | Observations | Date Range |
|--------------|------------|-----------|--------------|------------|
| Parsimonious | 2 | 3 | ~200 | 2008-2025 |
| BoP | 4 | 5 | ~180 | 2009-2025 |
| Monetary | 2 | 3 | ~190 | 2008-2025 |
| PCA | 3 | 4 | ~140 | 2012-2025 |
| Full | 8 | 7 | ~138 | 2012-2025 |

### 2. PCA Interpretation Table

| Component | Var Explained | Top Loadings | Interpretation |
|-----------|---------------|--------------|----------------|
| PC1 | ~45% | exports(+), imports(+), trade_bal(-) | Trade scale factor |
| PC2 | ~25% | usd_lkr(+), m2_usd(-) | Monetary conditions |
| PC3 | ~15% | tourism(+), remittances(+) | Service inflows |

### 3. Configuration File Update

Update `forecasting_prep/config.py` to include all variable sets as selectable options.

---

## Execution Log

### Pre-Execution
| Check | Status | Notes |
|-------|--------|-------|
| Source data exists | :white_check_mark: | reserves_forecasting_panel.csv (252 obs) |
| Dependencies installed | :white_check_mark: | sklearn, pandas, numpy |
| Output directories created | :white_check_mark: | data/forecast_prep_academic/ |

### Execution
| Step | Status | Timestamp | Notes |
|------|--------|-----------|-------|
| Load source data | :white_check_mark: | 2026-02-10 09:13 | 252 obs, repaired usd_lkr and m2_usd_m |
| Validate parsimonious set | :white_check_mark: | 2026-02-10 09:13 | 213 obs available |
| Validate BoP set | :white_check_mark: | 2026-02-10 09:13 | 189 obs available |
| Validate monetary set | :white_check_mark: | 2026-02-10 09:13 | 214 obs available |
| Build PCA factors | :white_check_mark: | 2026-02-10 09:13 | 3 components, 79.5% variance |
| Validate full set | :white_check_mark: | 2026-02-10 09:13 | 130 obs available |
| Save all datasets | :white_check_mark: | 2026-02-10 09:13 | All 5 varsets saved |
| Generate summary tables | :white_check_mark: | 2026-02-10 09:13 | variable_set_summary.csv |

### Post-Execution
| Validation | Status | Notes |
|------------|--------|-------|
| All sets created | :white_check_mark: | 5/5 variable sets |
| Metadata complete | :white_check_mark: | metadata.json in each varset |
| PCA outputs saved | :white_check_mark: | loadings, variance, scree, interpretation |
| Config updated | :white_check_mark: | variable_sets/config.py |

---

## Results Section

### Final Variable Set Statistics

| Variable Set | ARIMA Vars | VECM Vars | VAR Vars | Final Obs | Train | Valid | Test | Date Range |
|--------------|------------|-----------|----------|-----------|-------|-------|------|------------|
| Parsimonious | 2 | 3 | 3 | 219 | 156 | 36 | 27 | 2007-01 to 2025-03 |
| BoP | 4 | 5 | 4 | 195 | 132 | 36 | 27 | 2009-01 to 2025-03 |
| Monetary | 2 | 3 | 3 | 221 | 180 | 27 | 14 | 2005-01 to 2025-02 |
| PCA | 3 | 4 | 4 | 130 | 95 | 24 | 11 | 2012-01 to 2024-11 |
| Full | 8 | 7 | 6 | 137 | 96 | 27 | 14 | 2012-01 to 2025-02 |

### PCA Results

**Variance Explained:**
- PC1: 46.1%
- PC2: 20.7%
- PC3: 12.6%
- Cumulative: 79.5%

**Component Loadings:**

| Variable | PC1 | PC2 | PC3 |
|----------|-----|-----|-----|
| exports_usd_m | +0.35 | +0.28 | +0.37 |
| imports_usd_m | +0.37 | -0.48 | -0.07 |
| remittances_usd_m | +0.31 | -0.15 | +0.53 |
| tourism_usd_m | +0.44 | +0.08 | +0.03 |
| usd_lkr | +0.42 | +0.25 | -0.20 |
| m2_usd_m | +0.47 | +0.12 | -0.10 |
| cse_net_usd_m | -0.15 | -0.33 | +0.66 |
| trade_balance_usd_m | -0.17 | +0.69 | +0.31 |

**Component Interpretation:**

| Component | Var Explained | Top Loadings | Interpretation |
|-----------|---------------|--------------|----------------|
| PC1 | 46.1% | m2_usd(+), tourism(+), usd_lkr(+), imports(+) | Trade scale / monetary conditions factor |
| PC2 | 20.7% | trade_balance(+), imports(-), cse_net(-) | Trade balance factor |
| PC3 | 12.6% | cse_net(+), remittances(+), exports(+) | Capital flows / service inflows |

### Issues Encountered

1. **Missing usd_lkr data**: The `reserves_forecasting_panel.csv` had empty `usd_lkr` column due to date alignment issues. Resolved by merging data from `historical_fx.csv` (240 observations).

2. **Missing m2_usd_m data**: Computed from `m2_lkr_m / usd_lkr` after repairing exchange rate data.

3. **Limited PCA/Full observations**: These variable sets require all variables to be non-missing, limiting data to 2012-01 onwards when CSE net flows become available (166 obs in source, 130 after complete cases).

### Output Files

```
data/forecast_prep_academic/
├── variable_set_summary.csv
├── varset_parsimonious/
│   ├── arima_dataset.csv
│   ├── vecm_levels.csv
│   ├── var_system.csv
│   └── metadata.json
├── varset_bop/
│   └── [same structure]
├── varset_monetary/
│   └── [same structure]
├── varset_pca/
│   ├── arima_dataset.csv
│   ├── vecm_levels.csv
│   ├── var_system.csv
│   ├── metadata.json
│   ├── pca_loadings.csv
│   ├── pca_variance_explained.csv
│   ├── pca_interpretation.csv
│   └── pca_scree_data.csv
└── varset_full/
    └── [same structure]
```

---

## References

- Stock, J.H. & Watson, M.W. (2002). Forecasting Using Principal Components. JASA.
- Bai, J. & Ng, S. (2002). Determining the Number of Factors. Econometrica.

---

## Change Log

| Date | Version | Changes |
|------|---------|---------|
| 2026-02-10 | 1.0 | Initial specification |
| 2026-02-10 | 1.1 | Implementation complete - all 5 variable sets created with PCA factors |

