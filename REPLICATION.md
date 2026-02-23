# Replication Guide

This document provides step-by-step instructions to reproduce all results from the Sri Lanka Reserves Forecasting study.

## System Requirements

- **Python**: 3.10 or higher
- **OS**: macOS, Linux, or Windows (WSL recommended)
- **Memory**: 8GB RAM minimum (16GB recommended for full model suite)
- **Time**: Full replication takes approximately 30-60 minutes

## Quick Start (One Command)

```bash
# Clone, install, and run full replication
cd reserves_project
pip install -e .
slreserves replicate --run-id replication_$(date +%Y%m%d)
```

This runs all steps below automatically. For granular control, follow the detailed instructions.

---

## Step 1: Environment Setup

### 1.1 Clone the Repository

```bash
git clone <repository-url>
cd SL-FSI/reserves_project
```

### 1.2 Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

### 1.3 Install Dependencies

**Option A: pip (recommended)**
```bash
# Core dependencies only
pip install -e .

# With ML models (XGBoost, LSTM)
pip install -e ".[ml]"

# Full development environment
pip install -e ".[all]"

# Or use requirements files directly
pip install -r requirements.txt        # Core only
pip install -r requirements-full.txt   # Including ML
```

**Option B: conda**
```bash
# Core environment
conda env create -f environment.yml
conda activate slreserves

# Full environment (including ML)
conda env create -f environment-full.yml
conda activate slreserves-full

# Then install the package
pip install -e .
```

**Verify installation:**
```bash
slreserves --help
```

### 1.4 Verify Data Files

The following data files should exist:

```
data/merged/reserves_forecasting_panel.csv   # Main forecasting panel (252 obs)
data/merged/slfsi_monthly_panel.csv          # Full FSI panel
data/external/                               # Raw source data
```

---

## Step 2: Time-Series Diagnostics

Run comprehensive diagnostics to validate data quality and select appropriate models.

```bash
slreserves diagnostics
```

**Expected outputs:**

| File | Description |
|------|-------------|
| `data/diagnostics/diagnostic_results.json` | All test results |
| `data/diagnostics/integration_summary.csv` | ADF/PP/KPSS tests |
| `data/diagnostics/arch_summary.csv` | ARCH-LM tests |
| `data/diagnostics/chow_test_summary.csv` | Structural break tests |
| `data/diagnostics/johansen_summary.csv` | Cointegration rank |

**Key findings to verify:**
- Target variable `gross_reserves_usd_m` is I(1)
- ACF(1) ≈ 0.97 (high persistence)
- ARCH effects present in residuals
- No structural break detected at April 2022 default (surprising but validated)

---

## Step 3: Model Evaluation

### 3.1 Run Rolling-Origin Evaluation

```bash
# Full evaluation across all variable sets with MS-VAR
slreserves evaluate \
    --varsets parsimonious,bop,monetary,pca,full \
    --horizons 1,3,6,12 \
    --include-ms \
    --include-llsv \
    --include-bop \
    --run-id evaluation_v1
```

**Expected outputs:**

```
data/outputs/evaluation_v1/forecast_results_unified/
├── rolling_origin_forecasts_parsimonious.csv
├── rolling_origin_forecasts_bop.csv
├── rolling_origin_forecasts_monetary.csv
├── rolling_origin_forecasts_pca.csv
├── rolling_origin_forecasts_full.csv
├── rolling_origin_summary_*.csv
└── manifest.json
```

### 3.2 Verify Model Rankings

Expected rankings (h=1, parsimonious variable set):

| Rank | Model | RMSE | Direction Accuracy |
|------|-------|------|-------------------|
| 1 | MS-VAR | ~180 | ~62% |
| 2 | LLSV | ~195 | ~58% |
| 3 | XGBoost | ~210 | ~55% |
| 4 | BVAR | ~220 | ~52% |
| 5 | Naive | ~250 | ~50% |

---

## Step 4: Statistical Tests

Run Diebold-Mariano and Model Confidence Set tests.

```bash
# For each variable set
for VARSET in parsimonious bop monetary pca full; do
    slreserves tests --varset $VARSET --horizon 1 --run-id evaluation_v1
done
```

**Expected outputs:**

```
data/outputs/evaluation_v1/statistical_tests/
├── diebold_mariano_parsimonious_h1.csv
├── model_confidence_set_parsimonious_h1.csv
└── ...
```

**Key hypothesis tests:**
- DM test: MS-VAR vs Naive → p < 0.05 (reject equal accuracy)
- MCS: MS-VAR included at 90% confidence level

---

## Step 5: Scenario Analysis

Generate policy-relevant scenario forecasts.

```bash
slreserves scenarios \
    --varset parsimonious \
    --horizon 12 \
    --scenarios all \
    --run-id scenarios_v1
```

**Expected outputs:**

```
data/outputs/scenarios_v1/scenario_analysis/
├── msvarx_scenario_summary.csv
├── msvarx_scenario_paths.csv
├── msvarx_scenario_exog_paths.csv
└── msvarx_scenario_fan_chart.png
```

**Expected scenario results (12-month horizon):**

| Scenario | End Level (USD M) | Change (%) |
|----------|-------------------|------------|
| Combined Upside | ~8,500 | +30% |
| Tourism Recovery | ~7,800 | +19% |
| Baseline | ~7,300 | +12% |
| Combined Adverse | ~6,600 | +1% |

---

## Step 6: Generate Publication Tables

```bash
slreserves tables --run-id evaluation_v1
```

**Expected outputs:**

```
data/outputs/evaluation_v1/tables/
├── table1_main_results.tex
├── table2_robustness.tex
├── table3_dm_tests.tex
└── appendix_*.tex
```

---

## Output Verification Checklist

After completing all steps, verify these key results:

### Model Performance
- [ ] MS-VAR achieves lowest RMSE on parsimonious variable set
- [ ] Directional accuracy > 50% for MS-VAR at h=1
- [ ] All models beat naive baseline (except potentially at long horizons)

### Diagnostics
- [ ] Target series is I(1) (ADF p > 0.05 for levels, p < 0.05 for differences)
- [ ] ARCH effects present (p < 0.05 in ARCH-LM test)
- [ ] Johansen test indicates 1-2 cointegrating relations

### Scenarios
- [ ] Combined Upside shows highest end-level
- [ ] Combined Adverse shows lowest end-level
- [ ] Baseline falls between adverse and upside

---

## Troubleshooting

### Import Errors
```bash
# Ensure package is installed in editable mode
pip install -e .
```

### Memory Issues
```bash
# Run with reduced model set
slreserves evaluate --exclude-bvar --horizons 1,3
```

### Missing Data
```bash
# Check data directory structure
ls -la data/merged/
ls -la data/external/
```

### Streamlit Dashboard Issues
```bash
# Launch dashboard manually
slreserves dashboard --port 8502
```

---

## Individual Command Reference

For fine-grained control, use the underlying commands:

```bash
# Diagnostics
reserves-diagnostics

# Forecast preparation
reserves-forecast-prep --varset parsimonious

# Baseline models only
reserves-forecast-baselines --varset parsimonious

# Rolling backtests
reserves-rolling-backtests --varset parsimonious --refit-interval 12

# Unified evaluation (recommended)
reserves-unified --include-ms --varsets parsimonious

# Statistical tests
reserves-stat-tests --use-unified --unified-varset parsimonious --unified-horizon 1

# ML hyperparameter tuning
reserves-tune-ml --varset parsimonious --model xgb

# Scenario analysis
reserves-scenarios --varset parsimonious --scenarios combined_adverse,combined_upside

# Publication tables
reserves-tables
```

---

## Data Dictionary

| Variable | Description | Units |
|----------|-------------|-------|
| `gross_reserves_usd_m` | Foreign exchange reserves (target) | USD millions |
| `exports_usd_m` | Monthly merchandise exports | USD millions |
| `imports_usd_m` | Monthly merchandise imports | USD millions |
| `remittances_usd_m` | Worker remittances | USD millions |
| `tourism_usd_m` | Tourism receipts | USD millions |
| `usd_lkr` | USD/LKR exchange rate | LKR per USD |
| `m2_usd_m` | Broad money supply | USD millions |
| `trade_balance_usd_m` | Exports - Imports | USD millions |

---

## Citation

If you use this code or data, please cite:

```bibtex
@article{reserves_forecasting_2026,
  title={Forecasting Sri Lanka's Foreign Exchange Reserves: A Markov-Switching VAR Approach},
  author={...},
  journal={...},
  year={2026}
}
```

---

## Contact

For questions about replication, please open an issue on the repository or contact [email].
