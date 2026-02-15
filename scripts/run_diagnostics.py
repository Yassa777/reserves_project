#!/usr/bin/env python3
"""Run reserves forecasting diagnostic phases and persist outputs."""

from __future__ import annotations

from datetime import datetime
import warnings

from diagnostics_phases import (
    KEY_VARIABLES,
    PHASE6_PREDICTORS,
    TARGET_VARIABLE,
    build_variable_quality,
    load_panel,
    run_phase2,
    run_phase3,
    run_phase4,
    run_phase5,
    run_phase6,
    run_phase7,
    run_phase8,
    run_phase9,
    save_outputs,
)

warnings.filterwarnings("ignore")


def run_all_diagnostics(verbose=True):
    if verbose:
        print("=" * 60)
        print("RESERVE FORECASTING DIAGNOSTIC TESTS")
        print("=" * 60)
        print(f"Started: {datetime.now()}")
        print("\nLoading data...")

    df = load_panel()
    if verbose:
        print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")

    quality_df = build_variable_quality(df, KEY_VARIABLES)
    usable_vars = quality_df.loc[quality_df["is_usable"], "variable"].tolist()
    skipped = quality_df.loc[~quality_df["is_usable"], ["variable", "status"]]

    if verbose:
        print(f"\nTesting {len(usable_vars)} usable variables: {usable_vars}")
        if not skipped.empty:
            print("\nSkipped variables:")
            for _, row in skipped.iterrows():
                print(f"  - {row['variable']}: {row['status']}")

    results = {
        "metadata": {
            "timestamp": str(datetime.now()),
            "data_file": "reserves_forecasting_panel.csv",
            "n_obs": len(df),
            "date_range": f"{df.index.min()} to {df.index.max()}",
            "variables_requested": KEY_VARIABLES,
            "variables_tested": usable_vars,
            "variables_skipped": skipped.to_dict(orient="records"),
        },
        "phase2_stationarity": {},
        "phase3_temporal": {},
        "phase4_volatility": {},
        "phase5_breaks": {},
        "phase6_relationships": {},
        "phase7_cointegration": {},
        "phase8_svar": {},
        "phase9_multiple_breaks": {},
    }

    if verbose:
        print("\n" + "=" * 40)
        print("PHASE 2: STATIONARITY TESTS")
        print("=" * 40)
    results["phase2_stationarity"] = run_phase2(df, usable_vars, verbose=verbose)

    if verbose:
        print("\n" + "=" * 40)
        print("PHASE 3: TEMPORAL DEPENDENCE")
        print("=" * 40)
    results["phase3_temporal"] = run_phase3(df, usable_vars, verbose=verbose)

    if verbose:
        print("\n" + "=" * 40)
        print("PHASE 4: VOLATILITY")
        print("=" * 40)
    results["phase4_volatility"] = run_phase4(df, usable_vars, verbose=verbose)

    if verbose:
        print("\n" + "=" * 40)
        print("PHASE 5: STRUCTURAL BREAKS")
        print("=" * 40)
    results["phase5_breaks"] = run_phase5(df, usable_vars, verbose=verbose)

    predictors = [p for p in PHASE6_PREDICTORS if p in usable_vars]
    if TARGET_VARIABLE not in usable_vars:
        predictors = []

    if verbose:
        print("\n" + "=" * 40)
        print("PHASE 6: RELATIONSHIPS")
        print("=" * 40)

    if predictors:
        results["phase6_relationships"] = run_phase6(df, TARGET_VARIABLE, predictors, verbose=verbose)
    else:
        results["phase6_relationships"] = {"cross_correlation": [], "granger_causality": []}

    if verbose:
        print("\n" + "=" * 40)
        print("PHASE 7: COINTEGRATION & ECM/VECM")
        print("=" * 40)
    results["phase7_cointegration"] = run_phase7(
        df,
        usable_vars,
        results["phase2_stationarity"],
        verbose=verbose,
    )

    if verbose:
        print("\n" + "=" * 40)
        print("PHASE 8: EXOGENEITY & SVAR")
        print("=" * 40)
    results["phase8_svar"] = run_phase8(
        df,
        usable_vars,
        results["phase2_stationarity"],
        verbose=verbose,
    )

    if verbose:
        print("\n" + "=" * 40)
        print("PHASE 9: MULTIPLE STRUCTURAL BREAKS")
        print("=" * 40)
    results["phase9_multiple_breaks"] = run_phase9(df, usable_vars, verbose=verbose)

    if verbose:
        print("\n" + "=" * 40)
        print("SAVING RESULTS")
        print("=" * 40)

    summary_dfs = save_outputs(results, quality_df)

    if verbose:
        print("  Saved to data/diagnostics/diagnostic_results.json")
        print("  Saved variable_quality_summary.csv")
        print("  Saved integration_summary.csv")
        print("  Saved arch_summary.csv")
        print("  Saved chow_test_summary.csv")
        print("  Saved granger_causality_summary.csv")
        print("  Saved cointegration_engle_granger_summary.csv")
        print("  Saved ecm_suitability_summary.csv")
        print("  Saved johansen_summary.csv")
        print("  Saved vecm_suitability_summary.csv")
        print("  Saved svar_exogeneity_summary.csv")
        print("  Saved svar_sign_restriction_summary.csv")
        print("  Saved svar_model_summary.csv")
        print("  Saved bai_perron_summary.csv")
        print(f"\nCompleted: {datetime.now()}")
        print("=" * 60)

    return results, summary_dfs


if __name__ == "__main__":
    run_all_diagnostics(verbose=True)
