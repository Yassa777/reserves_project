# Super Prompt 2: Manuscript Discrepancy Audit & Corrections

## Context

This prompt targets `manuscript/manuscript_v2.tex`. A thorough numerical audit has identified **9 discrepancies** — 2 flagged by the author, 7 additional ones found during verification. Each is documented below with the exact location, the error, the correct value (where determinable), and the required fix. **All fixes are to the manuscript text/tables only** (no code changes needed for this prompt).

The discrepancies range from critical (wrong sign on the interaction term, mislabeled table rows) to minor (rounding differences). All should be corrected before submission.

---

## Discrepancy 1: Table 10 — Regime Share Inconsistency [Author-identified]

### Location
Table 10 (`\label{tab:regime_char}`), approximately line 468–490.

### The error
The "Share of observations" row reports:
- Regime 0 (Crisis): 74.6%
- Regime 1 (Accumulation): 25.4%

But the **ergodic (long-run) distribution** implied by the self-transition probabilities tells a different story. With p00 = 0.943 and p11 = 0.983:
- Ergodic share of Regime 0 = (1 − p11) / (2 − p00 − p11) = 0.017 / 0.074 ≈ **23.0%**
- Ergodic share of Regime 1 = (1 − p00) / (2 − p00 − p11) = 0.057 / 0.074 ≈ **77.0%**

The reported shares (74.6% / 25.4%) are approximately the **inverse** of the ergodic distribution. This is internally inconsistent: a regime with expected duration 57.2 months should dominate the sample, not the regime with expected duration 17.6 months.

### Likely explanation
The shares are computed from smoothed-state classification frequencies (i.e., how many observations the Viterbi/smoothed path assigns to each regime), not from the ergodic distribution. This is valid — but if so, it suggests either:
(a) The regime labels are swapped (Regime 0 is actually the long-duration accumulation regime, not crisis), or
(b) The sample is dominated by the crisis regime due to the specific sample period, and the share reflects actual classification rather than steady-state.

Given the non-convergence of the EM algorithm noted in the same table, option (a) — label swapping — is plausible and must be ruled out.

### Required fix
1. **Verify the regime labeling** in the code. In `reserves_project/models/ms_switching_var.py`, the `classification_certainty()` method (line 251) computes shares from `assigned = probs.argmax(axis=1)`. Check whether the regime labeled "0" in the code corresponds to the regime labeled "Crisis" in the manuscript. Cross-reference against the estimated mean vectors for each regime: the crisis regime should have negative mean reserve changes, while the accumulation regime should have positive or near-zero mean changes.
2. **If labels are correct**: Add an explicit note to Table 10 stating: "Share of observations reflects smoothed-state classification frequencies, not the ergodic (long-run) distribution. The ergodic shares implied by the transition probabilities are approximately 23% (Crisis) and 77% (Accumulation), differing from the classification shares because the sample period 2005–2025 includes an extended crisis episode (2019–2022)."
3. **If labels are swapped**: Fix all regime references throughout the manuscript (Table 10, Section 6.2, Section 6.3, and all prose discussing "crisis" and "accumulation" regimes). This is a more extensive fix.

---

## Discrepancy 2: Table 15 — Restricted Model Set Not Stated [Author-identified]

### Location
Table 15 (`\label{tab:split_robustness}`), approximately line 637–655.

### The error
Table 15 shows 5 models: MS-VAR, DMA, MS-VECM, DMS, XGBoost. The note says "Only the top five models shown; remaining models (ARIMA, Naïve, BVAR, VECM) have RMSE > 1,200 in all splits."

But Table 7 (BoP specification, `\label{tab:bop_horizon}`) shows LSTM at RMSE 479.7 for h=1 — substantially better than XGBoost at 886.1. If this is the same specification, LSTM should rank above XGBoost and should appear in the "top five."

XGBoost is shown as rank 5 with Rank SD 0.00, implying it is always last among the shown models. But if LSTM were included, XGBoost might be rank 6, and LSTM would enter the top 5. This makes the table potentially misleading.

### Likely explanation
LSTM may have been intentionally excluded from the split robustness analysis — for example, because it was not re-tuned for each split, or because its training is non-deterministic, or because it was added to the pipeline after the split robustness runs. But this exclusion is not stated.

### Required fix
1. **If LSTM was intentionally excluded**: Change the table note to: "Only the top five models shown from the split-robust evaluation set. LSTM is excluded from split robustness analysis due to [reason: e.g., sensitivity to hyperparameter tuning across different training windows / computational cost of re-estimation]. Remaining models (ARIMA, Naïve, BVAR, VECM) have RMSE > 1,200 in all splits."
2. **If LSTM should be included**: Re-run the split robustness analysis with LSTM, update the table to show 6 models (or keep 5 but include LSTM and remove the lowest-ranked), and update rank statistics accordingly.
3. **Also consider BoPIdentity** (RMSE 1,136.4 in Table 7) — it too outperforms several "remaining models" but is not mentioned. The note should account for all models in the BoP specification.

---

## Discrepancy 3: Table 9 — Architecture Effect Numerical Error [NEW]

### Location
Table 9 (`\label{tab:did}`), table notes, approximately line 452–456. Also appears in the prose at line 458 and in the abstract at line 52.

### The error
The table note states: "Architecture effect (MS-VAR advantage, averaged across varsets) ≈ 464.5 RMSE points."

Using the formula from Section 4.4 (line 208) and the values in the same table:
```
α̂ = (1/2) × [(R_XGB,Parsim − R_MSVAR,Parsim) + (R_XGB,BoP − R_MSVAR,BoP)]
   = (1/2) × [(640.6 − 311.8) + (886.1 − 315.2)]
   = (1/2) × [328.8 + 570.9]
   = (1/2) × 899.7
   = 449.85
```

**Correct value: ≈ 449.9, not 464.5.** Discrepancy of 14.6 points.

Note: This can also be computed as the difference of row means: 763.4 − 313.5 = 449.9 (confirming the row means in the table are correct but the stated architecture effect is not).

### Required fix
1. Change "≈ 464.5" to "≈ 449.9" in the Table 9 notes.
2. Update any prose that references this number (line 458: "approximately 464 RMSE points" → "approximately 450 RMSE points").
3. Check the abstract (line 52): it says "DiD ≈ 173.8" which is also wrong (see Discrepancy 4). The architecture effect number should be consistent everywhere.

---

## Discrepancy 4: Table 9 — DiD Interaction Term Wrong Sign and Magnitude [NEW — CRITICAL]

### Location
Table 9 notes (line 454), prose at line 458, and abstract at line 52.

### The error
The table note states: "Interaction (DiD) ≈ 173.8, indicating that the architecture gain is differentially larger in the parsimonious specification."

Using the formula from Section 4.4 (line 218):
```
δ̂ = (R_XGB,Parsim − R_XGB,BoP) − (R_MSVAR,Parsim − R_MSVAR,BoP)
   = (640.6 − 886.1) − (311.8 − 315.2)
   = (−245.5) − (−3.4)
   = −242.1
```

**Correct value: ≈ −242.1, not +173.8.** This is wrong in both sign and magnitude.

### Interpretation consequences
The **negative** DiD means: XGBoost's performance degrades much more than MS-VAR's when moving from Parsimonious to BoP. In other words, the BoP information hurts XGBoost far more than it hurts (or helps) MS-VAR. The prose at line 458 says the interaction is "positive and large, indicating that the model architecture and information set are complements for MS-VAR but substitutes for XGBoost." The qualitative interpretation is actually still defensible (architecture and information interact asymmetrically), but the sign should be negative, and the phrasing needs adjustment.

Alternative computation (reversing the differencing order):
```
(R_XGB,BoP − R_MSVAR,BoP) − (R_XGB,Parsim − R_MSVAR,Parsim)
= (886.1 − 315.2) − (640.6 − 311.8)
= 570.9 − 328.8
= +242.1
```
This gives +242.1 (not 173.8 either way). The value 173.8 does not emerge from any combination of the four RMSE values in the table.

### Required fix
1. **Recompute the DiD** from the actual data (not just the table values — check the underlying CSV outputs to verify the RMSE values in the table are correct).
2. **If the table RMSE values are correct**: Replace "DiD ≈ 173.8" with the correct value (−242.1 or +242.1 depending on the chosen sign convention). State the sign convention explicitly.
3. **Update all prose** referencing the DiD value:
   - Table 9 notes (line 454)
   - Line 458: "DiD ≈ 173.8" → correct value
   - Abstract line 52: "a significant interaction term (DiD ≈ 173.8)" → correct value
4. **Adjust the interpretation** at line 458 to be consistent with the correct sign. If δ̂ < 0: "The negative interaction indicates that moving to the BoP specification degrades XGBoost performance (by 245.5 RMSE points) while barely affecting MS-VAR (3.4 RMSE points), confirming that the richer information in disaggregated flows is exploitable only by the regime-switching architecture."

---

## Discrepancy 5: Table 10 — Expected Duration Does Not Match Self-Transition Probability [NEW]

### Location
Table 10 (`\label{tab:regime_char}`), line 472.

### The error
The table states:
- Regime 1 (Accumulation): p11 = 0.983, Expected duration = 57.2 months

Using the formula stated in the table notes: Expected duration = 1/(1 − p_ii):
```
1 / (1 − 0.983) = 1 / 0.017 = 58.82 months
```

**Correct value: ≈ 58.8 months, not 57.2.** Discrepancy of 1.6 months.

For Regime 0: 1/(1 − 0.943) = 1/0.057 = 17.54. The table says 17.6, which is close enough to be a rounding issue (the actual p00 might be 0.9432 with more decimal places, giving 1/0.0568 = 17.6).

### Required fix
1. **Check the raw transition matrix** in `data/regime_characterization/regime_transition_matrix.csv` to get the full-precision p11 value.
2. **If p11 = 0.983 is correct (to 3 decimal places)**: Change 57.2 to 58.8 in the table.
3. **If the duration of 57.2 is correct**: The self-transition probability should be reported as p11 = 1 − 1/57.2 ≈ 0.9825, and the table should show 0.983 (which rounds correctly from 0.9825). In this case, the values are consistent at 3 decimal places — but add a note that "expected durations are computed from full-precision transition probabilities; reported probabilities are rounded."
4. **Also check and fix the abstract** (line 54): "expected durations of 17.6 and 57.2 months" — update if the duration changes.

---

## Discrepancy 6: Abstract Cancellation Index vs. Table 11 [NEW — Minor]

### Location
Abstract (line 52) vs. Table 11 (`\label{tab:info_loss}`), line 514.

### The error
- Abstract: "mean cancellation index ≈ 0.062"
- Table 11, "All" period: CI = 0.063

These should match. The table value is the authoritative source.

### Required fix
Change the abstract from "0.062" to "0.063" (or change the table, if the abstract value is from a different computation).

---

## Discrepancy 7: Table 12 — "All (2020–2025)" Row Shows Test-Period-Only Values [NEW — CRITICAL]

### Location
Table 12 (`\label{tab:crisis_segment}`), line 552.

### The error
The table has three rows:
- Crisis (2020–2022): MS-VAR 367.1, Naïve 1338.9
- Post-default (2023–2025): MS-VAR (BoP) 260.6, Naïve 1350.8
- All (2020–2025): MS-VAR 311.8, Naïve 1178.9

The "All (2020–2025)" row shows MS-VAR = 311.8 and Naïve = 1178.9. But these are **exactly** the test-period-only (2023–2025) numbers from Table 3 (Parsimonious specification, h=1).

If "All (2020–2025)" truly covers both the validation period (2020–2022, 36 months) and the test period (2023–2025, 36 months), the RMSE should be a combined value across 72 months of forecasts — **not** identical to the 36-month test period numbers.

### Likely explanation
Either:
(a) The "All" label is wrong and should say "Test (2023–2025)" or "Post-default (2023–2025, Parsimonious)", or
(b) The numbers are wrong and should reflect the combined 2020–2025 RMSE.

### Required fix
1. **Verify** by checking the underlying forecast CSV files: compute RMSE for MS-VAR and Naïve over the full 2020–2025 window.
2. **If the numbers are test-period-only**: Relabel the row to "Test (2023–2025)" and note the specification (Parsimonious).
3. **If the label is correct**: Replace the numbers with the actual 2020–2025 combined RMSE.
4. **Note the specification inconsistency**: The first two rows appear to reference BoP-specification results (260.6 for MS-VAR BoP in the Post-default row), while the "All" row uses Parsimonious-specification results (311.8). State the specification for each row explicitly.

---

## Discrepancy 8: Section 5.2 — DM/MCS Specification Ambiguity [NEW]

### Location
Section 5.2, approximately line 307–309.

### The error
The text says: "Across the full nine-model set (including DMA and DMS), 30 of 36 pairwise DM comparisons are significant at the 10% level, and 28 at the 1% level."

DMA and DMS are defined only for the **BoP specification** (Section 4.5, line 226: "The constituent model pool comprises all seven non-DMA/DMS models evaluated on the BoP specification"). But this statement appears in a section that has been discussing the **Parsimonious specification** (Tables 3 and 4).

The reader cannot tell whether these 36 comparisons are:
(a) All within the BoP specification (9 BoP models), or
(b) Mixed across specifications, or
(c) Some other configuration.

### Required fix
Add a clarifying phrase: "Across the full nine-model set **in the BoP specification** (including DMA and DMS), 30 of 36..." — or if the comparisons mix specifications, state this explicitly.

---

## Discrepancy 9: Tables 5 and 6 Exclude MS-VAR/MS-VECM Without Clear Justification [NEW]

### Location
Table 5 (`\label{tab:horizon}`), line 313–333, and Table 6 (`\label{tab:val_vs_test}`), line 339–361.

### The issue
Both tables present Parsimonious specification results but exclude MS-VAR and MS-VECM. Table 5's note redirects to Table 7 "for the BoP specification." Table 6 provides no explanation for the exclusion.

This is inconsistent because:
- Table 3 (also Parsimonious specification) **does** include MS-VAR (311.8) and MS-VECM (357.7).
- If Parsimonious MS-VAR results exist at h=1, they presumably exist at h=3, 6, 12 as well — so why are they excluded from the horizon table?

A reviewer may suspect that MS-VAR Parsimonious results at longer horizons are unfavorable and were selectively omitted.

### Required fix
Either:
1. **Include MS-VAR and MS-VECM in Tables 5 and 6** for the Parsimonious specification (the data must exist since h=1 results are in Table 3), OR
2. **Add an explicit note** explaining the exclusion: e.g., "MS-VAR and MS-VECM Parsimonious results are omitted from horizon and period tables to avoid redundancy with the BoP results (Table 7), which represent the preferred specification for regime-switching models due to the richer structural identification of component flows." This at least makes the editorial choice transparent.

---

## Summary of All Discrepancies

| # | Table/Section | Severity | Issue | Fix complexity |
|---|---|---|---|---|
| 1 | Table 10 | HIGH | Regime shares inconsistent with transition probabilities | Verify labels + add note |
| 2 | Table 15 | MEDIUM | LSTM excluded without explanation | Add note or include LSTM |
| 3 | Table 9 notes | HIGH | Architecture effect: 464.5 should be 449.9 | Change number |
| 4 | Table 9 notes + abstract | **CRITICAL** | DiD: 173.8 should be ±242.1 (wrong sign + magnitude) | Recompute + rewrite interpretation |
| 5 | Table 10 | MEDIUM | Duration 57.2 doesn't match p11=0.983 (should be 58.8) | Verify precision + fix |
| 6 | Abstract vs Table 11 | LOW | CI 0.062 vs 0.063 | Align numbers |
| 7 | Table 12 | **CRITICAL** | "All (2020–2025)" shows test-period-only values | Relabel or recompute |
| 8 | Section 5.2 | MEDIUM | DM/MCS specification (BoP vs Parsimonious) unclear | Add specification label |
| 9 | Tables 5, 6 | MEDIUM | MS-VAR excluded without clear justification | Add note or include results |

### Recommended fix order
1. **Discrepancy 4** (DiD — critical, affects abstract, body, and table)
2. **Discrepancy 7** (Table 12 mislabeling — critical)
3. **Discrepancy 1** (Regime shares — high, may require code verification)
4. **Discrepancy 3** (Architecture effect — high, straightforward number fix)
5. **Discrepancy 5** (Duration — medium, verify raw data)
6. **Discrepancy 2** (Table 15 LSTM — medium, editorial choice)
7. **Discrepancy 8** (DM specification — medium, add one phrase)
8. **Discrepancy 9** (Tables 5/6 exclusion — medium, editorial choice)
9. **Discrepancy 6** (CI rounding — low, one-character fix)

---

## Verification Process

For each discrepancy, the implementation should:

1. **Check the source data** in `data/outputs/` (forecast CSVs, regime characterization outputs, statistical test matrices) to verify whether the manuscript values or the recomputed values are correct.
2. **Fix the manuscript** once the correct values are established.
3. **After all fixes**, do a final pass through the manuscript searching for any instance of the old incorrect values (e.g., grep for "464", "173.8", "57.2", "0.062") to ensure no stale references remain.
4. **Cross-check the abstract** against all tables — every number in the abstract should match a specific table cell.
