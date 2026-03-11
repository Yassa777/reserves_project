# Super Prompt 1: Analytical & Code Changes

## Context

This prompt targets the forecasting codebase at `/reserves_project/` and the manuscript at `/manuscript/manuscript_v2.tex`. The paper forecasts Sri Lanka's foreign exchange reserves using multiple model families (ARIMA, BVAR, MS-VAR, MS-VECM, XGBoost, LSTM, DMA/DMS) across five variable sets. The headline finding is that MS-VAR dominates with ~75% RMSE reduction over the naïve benchmark. Four analytical improvements are required to strengthen this claim against reviewer scrutiny.

---

## Change 1: Multi-Start Estimation Stability Check for MS-VAR

### Problem
The MS-VAR estimation (Table 10 in the manuscript) reports `Converged: No (max iterations)`. With a complex 5-variable BoP system on ~193 differenced observations, the sharp regime separability (mean max state probability 0.997, entropy 0.012) may reflect a local maximum of the likelihood or initialization sensitivity rather than stable structural dynamics. A reviewer will flag this as undermining the paper's central claim.

### What to build
A **multi-start estimation stability analysis** that fits the MS-VAR from N different random initialization seeds and reports how stable the results are.

### Implementation specification

1. **Modify `MarkovSwitchingVAR` in `reserves_project/models/ms_switching_var.py`:**
   - The current `_initialize_params()` method (line 73) accepts an `init_states` array and initializes the transition matrix with 0.9 on the diagonal (line 115). Add an optional `random_state: int | None` parameter to the `fit()` method. When provided, use it to seed a `numpy.random.RandomState` that generates a random `init_states` vector (drawing from `{0, 1, ..., n_regimes-1}`) and random perturbations to the initial transition matrix diagonal (e.g., uniform on `[0.7, 0.95]` instead of fixed 0.9).
   - The `fit()` method (line 195) currently uses `max_iter=50` and `tol=1e-4`. Ensure that `max_iter` and `tol` can be overridden at fit time or construction time so the multi-start harness can increase iteration limits (e.g., to 200-500).

2. **Create a new module `reserves_project/robustness/multistart_msvar.py`:**
   - Implement a function `run_multistart_estimation(n_starts: int = 20, max_iter: int = 200, varset: str = "bop", ...)` that:
     a. Loads the BoP variable set data via `load_varset_levels("bop")`.
     b. For each start `s` in `1..n_starts`:
        - Fits `MarkovSwitchingVAR(n_regimes=2, ar_order=1, max_iter=max_iter)` with `random_state=s`.
        - Records: final log-likelihood, converged (bool), number of iterations, transition matrix (2x2), expected durations, regime assignment shares, mean max probability, entropy.
     c. Also fits one run using the existing deterministic initialization (the `_build_init_states` volatility-threshold method from `run_regime_characterization.py` line 27) as the "baseline" seed.
     d. Returns a DataFrame with one row per start plus the baseline.
   - Implement a function `multistart_stability_summary(results_df)` that computes:
     - Mean, std, min, max of log-likelihood across starts.
     - Number/share of starts that converged.
     - Mean, std of each transition probability (p00, p01, p10, p11).
     - Mean, std of expected durations for each regime.
     - Mean, std of regime assignment shares.
     - Correlation between the smoothed regime probability vectors across starts (to check if different starts identify the same regimes or permuted versions). Handle label-switching: if regime labels are permuted in some starts, detect this (e.g., if p00 and p11 are swapped) and relabel before computing statistics.
   - Implement `multistart_forecast_stability(n_starts, ...)` that:
     - For each start, runs the full rolling-origin forecast evaluation (using the same procedure as `RollingOriginEvaluator` in `unified_evaluator.py`) and records RMSE on the test period.
     - Returns summary statistics of RMSE across starts (mean, std, min, max, range).
     - This demonstrates that forecast performance is not sensitive to initialization.

3. **Create a CLI entry point** `reserves_project/pipelines/run_multistart_stability.py`:
   - Arguments: `--n-starts` (default 20), `--max-iter` (default 200), `--varset` (default "bop"), `--ar-order` (default 1), `--run-id`, `--output-dir`.
   - Calls the functions above, saves:
     - `multistart_estimation_results.csv` (one row per start)
     - `multistart_stability_summary.json` (aggregated statistics)
     - `multistart_forecast_rmse.csv` (RMSE per start, if forecast stability is run)
   - Register CLI entry point in `setup.cfg` / `pyproject.toml` consistent with existing patterns (e.g., `reserves-multistart`).

4. **Update `manuscript_v2.tex`:**
   - In Section 6.2 (Regime Characterisation), add a paragraph reporting the multi-start results. State how many of N starts converged, the range of log-likelihood values, and whether the regime transition probabilities and forecast RMSE are stable across starts. If they are stable, this insulates the "Converged: No" caveat.
   - Add a new table (or extend Table 10) with multi-start summary statistics: columns for log-likelihood (mean±std), p00 (mean±std), p11 (mean±std), duration0 (mean±std), duration1 (mean±std), RMSE (mean±std).
   - Update the existing caveat paragraph (currently around line 496 which says "multi-start estimation... should be standard practice") to reference the actual results rather than simply recommending it.

### Acceptance criteria
- Multi-start analysis runs end-to-end from CLI.
- At least 20 random starts are evaluated.
- Results demonstrate whether the headline MS-VAR findings (transition probabilities, regime durations, forecast RMSE) are robust to initialization — or if they are not, the manuscript acknowledges this clearly.
- Tests pass (`pytest tests/`).

---

## Change 2: Random Walk Benchmark Precision

### Problem
The manuscript forecasts the **first difference** of reserves (Section 3.2, line 139: "the first-differenced series... is therefore the primary forecast target") but presents results in **levels** (Figure 1, scenario Table 14). The "Random Walk" benchmark is implemented as `NaiveForecaster` in `unified_evaluator.py`, but the manuscript does not explicitly state whether this is a random walk in levels (predicting zero change) or in growth rates. Accuracy metrics like MAPE and MASE behave very differently depending on the target (changes near zero vs. levels in the thousands). This ambiguity makes the "beating the random walk" claim vulnerable.

### What to build
Explicit documentation and verification of the benchmark transformation pipeline.

### Implementation specification

1. **Audit `NaiveForecaster` in `unified_evaluator.py`:**
   - Read the full `NaiveForecaster` class and trace exactly what it predicts. Document: does it predict the last observed level? The last observed change? Zero change?
   - Read how the `RollingOriginEvaluator` handles the naïve forecast — specifically, does it compute errors in differences or levels? How are multi-step (h=3, 6, 12) level forecasts reconstructed from change forecasts?
   - Trace the full pipeline from `predict()` → error computation → metrics (RMSE, MAPE, MASE in `metrics.py`). Determine unambiguously:
     a. What space are forecasts generated in? (differences or levels)
     b. What space are errors computed in? (differences or levels)
     c. For multi-step forecasts, how are levels reconstructed? (cumulative sum of predicted changes? direct level forecast?)
     d. Is the naïve benchmark transformed identically to the structural models?

2. **Add an explicit benchmark transformation test** in `tests/`:
   - Create `tests/test_benchmark_transformation.py` that:
     a. Constructs a simple synthetic reserve series with known dynamics.
     b. Runs both the NaiveForecaster and MS-VAR through the full evaluation pipeline.
     c. Verifies that the naïve forecast at h=1 is equivalent to a random walk in levels (i.e., ŷ_{t+1} = y_t, implying predicted change = 0).
     d. Verifies that multi-step naïve forecasts are consistent (h=3 naïve should be y_t for all 3 steps if it's a RW in levels, or cumulated zero changes).
     e. Verifies that MAPE/MASE are computed on the same target space for both the naïve and structural models.

3. **Update `manuscript_v2.tex`:**
   - In Section 3.2 (Dependent Variable, around line 139), add an explicit statement: "The naïve random walk benchmark predicts [zero change / last observed level / etc.] at all horizons. Multi-step level forecasts for all models are reconstructed by [cumulative summation of predicted h-step changes starting from the last observed level / direct iterated forecasts / etc.]. All accuracy metrics (RMSE, MAE, MAPE, MASE) are computed on [levels / changes], ensuring that the benchmark and structural models are evaluated on an identical target."
   - In Section 4.6 (Evaluation Framework, around line 238), add a sentence clarifying that the benchmark is transformed identically to all other models before metric computation.

### Acceptance criteria
- A clear, documented answer to the four questions in step 1.
- A passing test suite that verifies benchmark/model evaluation parity.
- Manuscript text that makes the benchmark definition indisputable.

---

## Change 3: Effective Sample Alignment Transparency

### Problem
Table 1 shows uneven data coverage: reserves has 252 observations, tourism has 192, CSE flows has 166. If the MS-VAR on the BoP set (which includes tourism) starts its effective estimation sample in ~2009 (when tourism data begins), while XGBoost on the Monetary set (which doesn't include tourism) starts in ~2005, the performance difference could partly reflect the different training sample rather than genuine model superiority. The manuscript does not currently explain how this ragged-edge problem is handled.

### What to build
A clear accounting of effective sample sizes per model-varset combination, and manuscript text disclosing this.

### Implementation specification

1. **Audit `load_varset_levels()` in `unified_evaluator.py` and `prepare_forecasting_data.py`:**
   - Trace what happens when a variable set is loaded with variables of different lengths. Does `dropna()` trim to the shortest variable? Is there imputation? Is each model given the maximum available data for its variable set?
   - For each of the 5 variable sets, determine the effective start date and number of usable observations after differencing and lag construction.

2. **Create `reserves_project/eval/sample_alignment_report.py`:**
   - Implement `generate_sample_alignment_table(varsets: list[str])` that:
     a. For each variable set, loads the data, applies the same preprocessing as the evaluation pipeline (differencing, lag construction, dropna).
     b. Reports: variable set name, variables included, earliest observation date, latest observation date, effective number of training observations (up to 2019-12), effective number of total observations.
   - Returns a DataFrame suitable for inclusion in the manuscript as a supplementary table.

3. **Add a sample alignment note to the manuscript's Table 2** (Variable Set Specifications, line 145):
   - Add a column "Effective N (train)" showing the number of training observations available for each variable set.
   - Alternatively, add a note below Table 2 stating: "The effective training sample for the BoP set begins in [date] due to the later availability of tourism earnings and remittances data, yielding [N] observations versus [M] for the Parsimonious and Monetary sets."

4. **In the manuscript body** (Section 5 or Section 6.1, near the DiD discussion):
   - Add a sentence acknowledging: "Performance comparisons across variable sets should be interpreted with the caveat that effective training samples differ: the BoP set provides [N] pre-crisis observations versus [M] for the Parsimonious set, owing to the later availability of tourism and remittance data."

### Acceptance criteria
- A concrete table showing the effective sample start date and N for each variable set.
- Manuscript text that discloses this clearly enough that a reviewer cannot raise it as an unacknowledged confound.

---

## Change 4: Statistical Significance Modulation

### Problem
The test period is only 36 months (2023–2025). Diebold-Mariano tests and the Model Confidence Set rely on asymptotic theory that may not hold with so few observations — especially for overlapping multi-step forecasts (h=3, 6, 12) where the effective number of independent observations is even lower. Block-bootstrap procedures for the MCS are highly sensitive to block length in short samples. The very small p-values reported (e.g., p < 0.0001) may overstate confidence.

### What to build
Additional emphasis on rank stability and economic magnitude, with appropriate caveats on the statistical tests.

### Implementation specification

1. **Add a rolling-origin rank stability analysis** in `reserves_project/robustness/rank_stability.py`:
   - Implement `compute_rolling_rank_stability(forecasts_df, metric="rmse", window=12)` that:
     a. Takes the rolling-origin forecast results (one row per origin date per model).
     b. For overlapping windows of `window` months within the test period, computes model rankings by RMSE.
     c. Returns: a rank matrix (origins × models), mean rank per model, rank standard deviation per model, and the number of origins where each model was ranked 1st.
   - This supplements the existing split robustness (which uses 3 fixed splits) with a more granular origin-by-origin stability check.

2. **Compute effective degrees of freedom for DM tests:**
   - In `reserves_project/eval/diebold_mariano.py`, add a function `effective_sample_size(n_obs: int, horizon: int) -> float` that computes the approximate effective independent sample size for h-step-ahead overlapping forecasts: `n_eff ≈ n_obs / h` (or a more sophisticated HAC-based estimate). This helps the reader understand why p-values for h=12 should be interpreted cautiously.
   - Add this as a column in the DM test output.

3. **Update `manuscript_v2.tex`:**
   - In Section 5.2 (Statistical Significance, around line 282), add a paragraph after the DM/MCS results:
     > "These tests should be interpreted with appropriate caution given the short evaluation window. With 36 test-period observations and overlapping multi-step forecasts, the effective number of independent observations is approximately [36/h] at horizon h. The block-bootstrap MCS procedure is sensitive to block length in such short samples. We therefore emphasise two complementary pieces of evidence that do not depend on asymptotic inference: first, the rank ordering is perfectly stable across three alternative train/validation splits (Table 15: MS-VAR holds rank 1 with zero rank variance); second, the economic magnitude of the error reduction is large (RMSE improvement of [X] USD million, representing [Y]% of mean reserve levels over the test period)."
   - Compute the economic magnitude: RMSE of ~312 vs ~1179 is an improvement of ~867 USD million. Against mean test-period reserves (from Table 1, the mean is ~5450 overall but the test period level is ~5000-6800), this is ~13-17% of the reserve level. State this explicitly.
   - In the abstract and introduction, where the "30 of 36 DM comparisons significant" claim appears, add a qualifier: "though the short evaluation window (36 months) warrants caution in interpreting individual p-values."

4. **Add the rolling-origin rank table to the manuscript:**
   - Present the rank stability results (from step 1) as a supplementary table or as an addition to the existing Table 15 discussion. Show that MS-VAR is ranked 1st at [X] of [Y] individual rolling origins, not just across 3 fixed splits.

### Acceptance criteria
- Rolling-origin rank stability analysis runs and produces a summary table.
- Effective sample size is computed and reported for DM tests.
- Manuscript text appropriately caveats the statistical significance while emphasising the more robust evidence (rank stability, economic magnitude).
- The paper's core claim remains strong but is now defended on grounds that don't depend on fragile p-values.

---

## General Implementation Notes

- **Testing**: All new code must have corresponding tests in `tests/`. Follow existing patterns (see `tests/test_diebold_mariano.py`, `tests/test_regime_characterization_pipeline.py`).
- **CLI consistency**: New CLI entry points should follow the pattern in `reserves_project/pipelines/` with argparse and the `write_run_manifest` / `write_latest_pointer` utilities.
- **Data paths**: Use `reserves_project.config.paths.DATA_DIR` for all data I/O. Output to `data/outputs/<run-id>/` when `--run-id` is provided.
- **Manuscript changes**: All LaTeX edits should be made to `manuscript/manuscript_v2.tex`. Preserve existing table/figure labels. New tables should follow the existing `\begin{table}[htbp]` pattern with `\begin{tablenotes}`.
- **Do not break existing functionality**: Run `pytest tests/` after all changes to verify nothing is broken.
