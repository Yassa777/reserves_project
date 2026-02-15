# Master Pipeline Specification
## Reserves Forecasting Enhancement Pipeline

**Version:** 1.1
**Created:** 2026-02-10
**Updated:** 2026-02-10
**Status:** 🟢 ALL PHASES COMPLETE - PIPELINE FINISHED

---

## Pipeline Overview

This master specification orchestrates the execution of all enhancement specs for the reserves forecasting project. The pipeline is organized into **phases** with clear dependencies.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PIPELINE EXECUTION GRAPH                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PHASE 1: FOUNDATION (Parallel)                                              │
│  ┌──────────────┐  ┌──────────────────┐  ┌─────────────────┐                │
│  │ 01_VARIABLE  │  │ 02_STRUCTURAL    │  │ (Existing Base  │                │
│  │    SETS      │  │    BREAKS        │  │   Models OK)    │                │
│  └──────┬───────┘  └────────┬─────────┘  └────────┬────────┘                │
│         │                   │                     │                          │
│         ▼                   ▼                     ▼                          │
│  PHASE 2: NEW MODELS (Parallel after Phase 1)                               │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌──────────┐        │
│  │ 03_BVAR│ │04_TVP  │ │05_FACTR│ │06_THRES│ │07_MIDAS│ │08_COMBINE│        │
│  └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘ └────┬─────┘        │
│      │          │          │          │          │           │               │
│      └──────────┴──────────┴──────────┴──────────┴───────────┘               │
│                                   │                                          │
│                                   ▼                                          │
│  PHASE 3: MODEL INTEGRATION                                                  │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │                     09_DMA_DMS                               │            │
│  │         (Requires all models from Phase 2)                   │            │
│  └──────────────────────────┬──────────────────────────────────┘            │
│                             │                                                │
│                             ▼                                                │
│  PHASE 4: EVALUATION                                                         │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │                  10_STATISTICAL_TESTS                        │            │
│  │     (Diebold-Mariano, Model Confidence Set, Density)         │            │
│  └──────────────────────────┬──────────────────────────────────┘            │
│                             │                                                │
│                             ▼                                                │
│  PHASE 5: SYNTHESIS                                                          │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │                  11_ROBUSTNESS_TABLES                        │            │
│  │            (Final academic-ready output)                     │            │
│  └─────────────────────────────────────────────────────────────┘            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase Definitions

### PHASE 1: Foundation (Parallel Execution)

| Spec | Description | Dependencies | Estimated Time |
|------|-------------|--------------|----------------|
| `01_VARIABLE_SETS_SPEC.md` | Define parsimonious, BoP, monetary, PCA variable sets | None | 2-3 hours |
| `02_STRUCTURAL_BREAKS_SPEC.md` | Bai-Perron structural break detection | None | 2-3 hours |

**Parallelization:** Both specs can execute simultaneously.

**Exit Criteria:**
- [x] All variable sets defined and saved to config
- [x] Structural breaks identified and documented
- [x] Break dates available for model conditioning

---

### PHASE 2: New Models (Parallel Execution)

| Spec | Description | Dependencies | Estimated Time |
|------|-------------|--------------|----------------|
| `03_BVAR_SPEC.md` | Bayesian VAR with Minnesota prior | Phase 1 complete | 3-4 hours |
| `04_TVP_VAR_SPEC.md` | Time-varying parameter VAR | Phase 1 complete | 4-5 hours |
| `05_FACTOR_VAR_SPEC.md` | Factor-augmented VAR (FAVAR) | Phase 1 complete | 3-4 hours |
| `06_THRESHOLD_VAR_SPEC.md` | Threshold VAR (exchange rate regime) | Phase 1 complete | 3-4 hours |
| `07_MIDAS_SPEC.md` | Mixed-frequency data sampling | Phase 1 complete | 4-5 hours |
| `08_FORECAST_COMBINATION_SPEC.md` | Equal-weight, MSE-weight, GR combination | Phase 1 complete | 2-3 hours |

**Parallelization:** All six specs can execute simultaneously after Phase 1.

**Exit Criteria:**
- [x] Each model produces forecasts for all variable sets
- [x] Rolling backtest results saved
- [x] Model diagnostics documented

---

### PHASE 3: Model Integration (Sequential)

| Spec | Description | Dependencies | Estimated Time |
|------|-------------|--------------|----------------|
| `09_DMA_DMS_SPEC.md` | Dynamic Model Averaging/Selection | All Phase 2 models | 4-5 hours |

**Exit Criteria:**
- [x] DMA time-varying weights computed
- [x] DMS model selection sequence saved
- [x] Combined forecasts generated

---

### PHASE 4: Evaluation (Sequential)

| Spec | Description | Dependencies | Estimated Time |
|------|-------------|--------------|----------------|
| `10_STATISTICAL_TESTS_SPEC.md` | DM tests, MCS, density evaluation | Phase 3 complete | 3-4 hours |

**Exit Criteria:**
- [ ] Pairwise DM test p-values computed
- [ ] Model Confidence Set identified
- [ ] CRPS and log scores calculated

---

### PHASE 5: Synthesis (Sequential)

| Spec | Description | Dependencies | Estimated Time |
|------|-------------|--------------|----------------|
| `11_ROBUSTNESS_TABLES_SPEC.md` | Academic-ready tables and figures | Phase 4 complete | 3-4 hours |

**Exit Criteria:**
- [x] Publication-ready tables generated
- [x] Robustness checks across subsamples
- [x] LaTeX-formatted output available

---

## Execution Tracking

### Phase 1 Status
| Spec | Status | Started | Completed | Notes |
|------|--------|---------|-----------|-------|
| 01_VARIABLE_SETS | ✅ Complete | 2026-02-10 09:08 | 2026-02-10 09:13 | 5 variable sets created, PCA: 79.5% var explained |
| 02_STRUCTURAL_BREAKS | ✅ Complete | 2026-02-10 09:15 | 2026-02-10 09:23 | 4 breaks detected in reserves, all events significant |

### Phase 2 Status
| Spec | Status | Started | Completed | Notes |
|------|--------|---------|-----------|-------|
| 03_BVAR | ✅ Complete | 2026-02-10 09:30 | 2026-02-10 10:08 | 5 varsets, best h=1 RMSE=510 (parsimonious) |
| 04_TVP_VAR | ✅ Complete | 2026-02-10 11:37 | 2026-02-10 11:38 | 3 varsets, coefficients align with breaks |
| 05_FACTOR_VAR | ✅ Complete | 2026-02-10 09:35 | 2026-02-10 09:40 | FAVAR with 3 PCs, Theil-U > 1 |
| 06_THRESHOLD_VAR | ✅ Complete | 2026-02-10 09:40 | 2026-02-10 09:45 | Negative result: linear VAR preferred |
| 07_MIDAS | ✅ Complete | 2026-02-10 09:36 | 2026-02-10 09:45 | Daily FX, in-sample +4.7%, OOS underperforms |
| 08_FORECAST_COMBINATION | ✅ Complete | 2026-02-10 09:36 | 2026-02-10 09:40 | MSE-inverse: **-51.5% vs best individual** |

### Phase 3 Status
| Spec | Status | Started | Completed | Notes |
|------|--------|---------|-----------|-------|
| 09_DMA_DMS | ✅ Complete | 2026-02-10 14:40 | 2026-02-10 14:48 | α=1.0 optimal, BVAR models dominate weights |

### Phase 4 Status
| Spec | Status | Started | Completed | Notes |
|------|--------|---------|-----------|-------|
| 10_STATISTICAL_TESTS | ✅ Complete | 2026-02-10 15:00 | 2026-02-10 16:25 | 7 models in 90% MCS, MS models eliminated |

### Phase 5 Status
| Spec | Status | Started | Completed | Notes |
|------|--------|---------|-----------|-------|
| 11_ROBUSTNESS_TABLES | ✅ Complete | 2026-02-10 16:30 | 2026-02-10 16:40 | 6 main tables + 3 appendix, 4 pub figures, MSE-Weight best |

---

## Quick Reference: Spec Files

```
reserves_project/specs/
├── 00_MASTER_PIPELINE_SPEC.md      ← You are here
├── 01_VARIABLE_SETS_SPEC.md        ← Variable set definitions
├── 02_STRUCTURAL_BREAKS_SPEC.md    ← Bai-Perron analysis
├── 03_BVAR_SPEC.md                 ← Bayesian VAR
├── 04_TVP_VAR_SPEC.md              ← Time-varying parameter VAR
├── 05_FACTOR_VAR_SPEC.md           ← Factor-augmented VAR
├── 06_THRESHOLD_VAR_SPEC.md        ← Threshold VAR
├── 07_MIDAS_SPEC.md                ← Mixed-frequency models
├── 08_FORECAST_COMBINATION_SPEC.md ← Combination strategies
├── 09_DMA_DMS_SPEC.md              ← Dynamic Model Averaging
├── 10_STATISTICAL_TESTS_SPEC.md    ← DM tests, MCS, density
└── 11_ROBUSTNESS_TABLES_SPEC.md    ← Final academic output
```

---

## Global Configuration

### Target Variable
```python
TARGET = "gross_reserves_usd_m"
```

### Sample Splits
```python
TRAIN_END = "2019-12-01"   # End of training
VALID_END = "2022-12-01"   # End of validation
TEST_END  = "2025-12-01"   # End of test
```

### Evaluation Metrics
- **Point Forecasts:** MAE, RMSE, MAPE, sMAPE, MASE
- **Density Forecasts:** CRPS, Log Score, PIT histograms
- **Relative Tests:** Diebold-Mariano, Model Confidence Set

### Output Directories
```
data/
├── forecast_prep_academic/     ← Enhanced variable sets
├── forecast_results_academic/  ← All model forecasts
├── statistical_tests/          ← DM tests, MCS results
└── robustness/                 ← Subsample analysis
```

---

## Execution Commands

### Run Full Pipeline
```bash
# Phase 1 (parallel)
python scripts/academic/prepare_variable_sets.py &
python scripts/academic/structural_breaks.py &
wait

# Phase 2 (parallel)
python scripts/academic/run_bvar.py &
python scripts/academic/run_tvp_var.py &
python scripts/academic/run_factor_var.py &
python scripts/academic/run_threshold_var.py &
python scripts/academic/run_midas.py &
python scripts/academic/run_forecast_combinations.py &
wait

# Phase 3
python scripts/academic/run_dma_dms.py

# Phase 4
python scripts/academic/run_statistical_tests.py

# Phase 5
python scripts/academic/generate_robustness_tables.py
```

---

## Change Log

| Date | Version | Changes |
|------|---------|---------|
| 2026-02-10 | 1.0 | Initial specification created |

