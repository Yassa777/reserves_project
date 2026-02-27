# Implementation Backlog (Phases 1-2)

Date: 2026-02-25
Scope: Strengthen intellectual contribution and methodological credibility before manuscript revision.

## Core Research Questions To Operationalize

1. What do we learn about macro forecasting methodology and reserve dynamics that is new?
2. Why does MS-VAR on disaggregated BoP flows outperform alternatives?
3. Is the gain driven by architecture (regime-switching), information content (disaggregation), or both?

## Non-Negotiable Engineering Constraints

1. No look-ahead leakage in any rolling forecast path.
2. Cross-model and cross-varset comparisons must report common-window metrics.
3. Every manuscript number must map to a deterministic pipeline artifact.
4. Any headline model gap must include uncertainty (confidence interval or hypothesis test).

## Phase 1: Evaluation Integrity + Disentangling (P0)

### P1.1 Common-Window Comparability

Goal: Make model and varset comparisons apples-to-apples.

Code changes:
1. Extend [unified_evaluator.py](/Users/dim/Desktop/Spill%20Projects/SL-FSI/reserves_project/reserves_project/eval/unified_evaluator.py):
   - Add `window_mode` argument (`full`, `common_dates`).
   - Add post-processing helper to filter each `(split, horizon)` slice to intersection of available `forecast_date` across selected models.
   - Add metadata columns in summaries: `window_mode`, `effective_start`, `effective_end`, `n_common_dates`.
2. Extend [run_unified_evaluations.py](/Users/dim/Desktop/Spill%20Projects/SL-FSI/reserves_project/reserves_project/pipelines/run_unified_evaluations.py):
   - Add CLI flag `--window-mode full|common_dates`.
   - Persist both raw forecast panel and common-window summary.
3. Add a shared helper module:
   - `reserves_project/eval/windowing.py` for deterministic date-window filtering.

Acceptance tests:
1. Add `tests/test_windowing.py`:
   - Verifies intersection logic with staggered model coverage.
   - Verifies no dates are added or reordered.
2. Add `tests/test_unified_common_window.py`:
   - For synthetic panel, confirm `n` is equal across models within same `(split, horizon)` when `window_mode=common_dates`.

Done definition:
1. Summaries include both full-sample and common-window outputs.
2. Manuscript tables can be regenerated from common-window outputs without manual edits.

---

### P1.2 Leakage Guardrails And Legacy Path Hygiene

Goal: Eliminate silent look-ahead paths and prevent regressions.

Code changes:
1. Fix history initialization in legacy MS paths:
   - [regime_var_model.py](/Users/dim/Desktop/Spill%20Projects/SL-FSI/reserves_project/reserves_project/forecasting_models/regime_var_model.py):
     replace `joined[variables].iloc[-ar_order:]` with origin-consistent training history.
   - [ms_vecm_model.py](/Users/dim/Desktop/Spill%20Projects/SL-FSI/reserves_project/reserves_project/forecasting_models/ms_vecm_model.py):
     replace `df[y_cols].iloc[-ar_order:]` with training-consistent history.
2. Add runtime assertions in [run_rolling_backtests.py](/Users/dim/Desktop/Spill%20Projects/SL-FSI/reserves_project/reserves_project/pipelines/run_rolling_backtests.py):
   - Ensure forecast inputs only use rows dated `<= origin`.
   - Track optional debug columns (`history_end_date`, `origin_date`) in intermediate frames.
3. Add leakage utility:
   - `reserves_project/eval/leakage_checks.py` with reusable checks for origin integrity.

Acceptance tests:
1. Add `tests/test_leakage_checks.py` with synthetic date-indexed panels.
2. Add regression tests in `tests/test_models.py` for legacy forecast wrappers:
   - Ensure first validation forecast does not depend on future test rows.

Done definition:
1. All rolling and legacy forecasting tests pass with leakage checks enabled.
2. No code path seeds forecast recursion from end-of-sample when origin is earlier.

---

### P1.3 Explicit Model-vs-Information Disentangling

Goal: Quantify architecture effect, information effect, and interaction effect.

Code changes:
1. Add analysis module:
   - `reserves_project/eval/disentangling.py`
   - Inputs: unified forecast panel for `MS-VAR` and `XGBoost` over varsets `parsimonious` and `bop`.
   - Outputs:
     - 2x2 RMSE matrix
     - architecture effect (holding info constant)
     - information effect (holding model constant)
     - interaction term (difference-in-differences)
     - optional DM tests on paired forecast errors.
2. Add pipeline:
   - `reserves_project/pipelines/run_disentangling_analysis.py`
   - CLI: varsets/models/horizon/split/bootstraps/output-dir.
3. Add CLI entrypoint:
   - `reserves_project/cli/disentangling.py`
   - wire into [pyproject.toml](/Users/dim/Desktop/Spill%20Projects/SL-FSI/reserves_project/pyproject.toml) and [main.py](/Users/dim/Desktop/Spill%20Projects/SL-FSI/reserves_project/reserves_project/cli/main.py).

Acceptance tests:
1. Add `tests/test_disentangling.py`:
   - Synthetic 2x2 error panel with known effect decomposition.
   - Verifies interaction is zero in additive synthetic case.
2. Add smoke test for CLI pipeline execution with temporary CSV fixtures.

Done definition:
1. One command produces publishable disentangling table and machine-readable CSV.
2. Paper claim about "specification vs information content" is directly supported by outputs.

---

### P1.4 Crisis-Stratified Evaluation

Goal: Align evaluation with the paper's crisis-forecasting contribution.

Code changes:
1. Add config:
   - `reserves_project/config/evaluation_segments.py`
   - Define default windows (`crisis`, `tranquil`, `all`) with explicit dates.
2. Extend `summarize_results` in [unified_evaluator.py](/Users/dim/Desktop/Spill%20Projects/SL-FSI/reserves_project/reserves_project/eval/unified_evaluator.py):
   - Compute metrics by segment.
3. Add optional segment report generation in [generate_robustness_tables.py](/Users/dim/Desktop/Spill%20Projects/SL-FSI/reserves_project/reserves_project/pipelines/generate_robustness_tables.py).

Acceptance tests:
1. Add `tests/test_evaluation_segments.py`:
   - Validate segment masks and edge dates.
2. Add summary test:
   - Verify pooled metric equals weighted average of segment metrics when coverage is complete.

Done definition:
1. Every main model has metrics reported for crisis and non-crisis windows.
2. Manuscript can claim crisis-specific advantage with explicit evidence.

---

### P1.5 Uncertainty Around Headline Gaps

Goal: Avoid point-estimate-only claims.

Code changes:
1. Add bootstrap utilities:
   - `reserves_project/eval/bootstrap_ci.py` for RMSE/MAE difference intervals.
2. Integrate into [run_statistical_tests.py](/Users/dim/Desktop/Spill%20Projects/SL-FSI/reserves_project/reserves_project/pipelines/run_statistical_tests.py):
   - Output CI tables for key model pairs.
3. Add table artifact:
   - `model_gap_confidence_intervals.csv`.

Acceptance tests:
1. Add `tests/test_bootstrap_ci.py`:
   - Deterministic seed behavior.
   - Coverage sanity check on synthetic data with known mean gap.

Done definition:
1. Main manuscript gap statements include interval estimates and not just raw RMSE deltas.

## Phase 2: Regime Mechanism + Information-Loss Explanation (P1)

### P2.1 MS-VAR Estimation Diagnostics

Goal: Make regime model estimation transparent and auditable.

Code changes:
1. Extend [ms_switching_var.py](/Users/dim/Desktop/Spill%20Projects/SL-FSI/reserves_project/reserves_project/models/ms_switching_var.py):
   - Store `loglik_path_`, `converged_`, `n_iter_`.
   - Preserve optional initialization diagnostics (`init_states_summary_`).
   - Add helper methods:
     - `expected_durations()`
     - `classification_certainty()`
2. Add serialization helper:
   - `reserves_project/models/msvar_diagnostics.py` (convert model state to tidy DataFrames).

Acceptance tests:
1. Extend `tests/test_models.py`:
   - Check monotonic/non-degenerate likelihood path behavior.
   - Check transition rows sum to 1.
   - Check durations are positive finite values.

Done definition:
1. MS-VAR fit emits diagnostics sufficient for appendix tables and convergence reporting.

---

### P2.2 Regime Characterization Pipeline

Goal: Produce smoothed probabilities, transition matrices, and regime duration tables for manuscript.

Code changes:
1. Add pipeline:
   - `reserves_project/pipelines/run_regime_characterization.py`
   - Inputs: varset, train/validation/test cutoff, ar_order.
   - Outputs:
     - `regime_smoothed_probabilities.csv` (date x regime probabilities)
     - `regime_transition_matrix.csv`
     - `regime_durations.csv`
     - `regime_classification_certainty.csv`
     - `regime_fit_diagnostics.json`
2. Add CLI entrypoint and wire into [pyproject.toml](/Users/dim/Desktop/Spill%20Projects/SL-FSI/reserves_project/pyproject.toml).

Acceptance tests:
1. Add `tests/test_regime_characterization_pipeline.py`:
   - Confirms files are created and probability rows sum to ~1.

Done definition:
1. Regime tables/figures in manuscript are generated directly from pipeline outputs.

---

### P2.3 Regime-Conditional Impulse Response Framework

Goal: Explain mechanism, not only predictive ranking.

Code changes:
1. Add IRF module:
   - `reserves_project/eval/msvar_irf.py`
   - Compute regime-conditional generalized IRFs from fitted MS-VAR parameters.
2. Add pipeline:
   - `reserves_project/pipelines/run_msvar_irf_analysis.py`
   - Outputs:
     - `msvar_irf_regime0.csv`
     - `msvar_irf_regime1.csv`
     - summary comparison table for shock persistence/amplitude.

Acceptance tests:
1. Add `tests/test_msvar_irf.py`:
   - Shape checks for horizons x variables x shocks.
   - Basic stability sanity checks (finite values, deterministic with seed where applicable).

Done definition:
1. Manuscript can report how shock propagation differs by regime and why crisis dynamics are captured better.

---

### P2.4 Formal Information-Loss Under Aggregation

Goal: Test whether aggregating components into composite variables loses predictive crisis information.

Code changes:
1. Add module:
   - `reserves_project/eval/information_loss.py`
   - Compare:
     - disaggregated BoP component models
     - aggregated flow proxy models
   - Compute forecast-error impact and a formal test statistic (nested predictive loss comparison).
2. Add pipeline:
   - `reserves_project/pipelines/run_information_loss_tests.py`
   - Output table for manuscript mechanism section.

Acceptance tests:
1. Add `tests/test_information_loss.py`:
   - Synthetic setting where aggregation is known to hide opposing component shocks.
   - Verify test detects degradation under aggregation.

Done definition:
1. Mechanistic argument about aggregation loss is supported by formal empirical test output.

---

### P2.5 Link Mechanism To Performance Differential

Goal: Join disentangling + regime diagnostics + IRFs into one coherent explanation.

Code changes:
1. Add synthesis pipeline:
   - `reserves_project/pipelines/run_mechanism_synthesis.py`
   - Merge outputs from P1.3, P2.2, P2.3, P2.4.
   - Produce concise table mapping each mechanism metric to forecast gain.

Acceptance tests:
1. Add `tests/test_mechanism_synthesis.py`:
   - Verify join keys and no missing critical columns.

Done definition:
1. A single artifact can directly support manuscript narrative in contribution section and discussion.

## Recommended Execution Order

1. P1.1 common-window comparability.
2. P1.2 leakage guardrails.
3. P1.3 disentangling analysis.
4. P1.4 crisis stratification.
5. P1.5 uncertainty intervals.
6. P2.1 MS-VAR diagnostics.
7. P2.2 regime characterization.
8. P2.3 IRF framework.
9. P2.4 information-loss tests.
10. P2.5 mechanism synthesis.

## Validation Gate Before Manuscript Update

Required checks:
1. `pytest tests/ -q` passes.
2. Unified evaluation rerun succeeds with `--window-mode common_dates`.
3. Disentangling pipeline output includes all four cells of the 2x2 design.
4. Regime characterization outputs include transition matrix, smoothed probabilities, and durations.
5. No manuscript table value is hand-edited without corresponding source artifact.

## Immediate Next Build Slice (Start Here)

1. Implement P1.1 and P1.2 together in a single PR.
2. Add tests for windowing and leakage in same PR.
3. Re-run unified outputs for `parsimonious` and `bop` only to validate runtime.
4. Then implement P1.3 disentangling pipeline.
