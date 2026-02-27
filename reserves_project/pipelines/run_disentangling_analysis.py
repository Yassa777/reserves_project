#!/usr/bin/env python3
"""Run 2x2 model-vs-information disentangling analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reserves_project.config.paths import DATA_DIR
from reserves_project.eval.disentangling import (
    build_2x2_aligned_panel,
    bootstrap_two_by_two_effects,
    compute_rmse_matrix,
    compute_two_by_two_effects,
    load_unified_forecasts_for_disentangling,
    run_disentangling_dm_tests,
)
from reserves_project.utils.run_manifest import write_latest_pointer, write_run_manifest


def _parse_csv_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="Run 2x2 model-vs-information disentangling analysis.")
    parser.add_argument("--varsets", default="parsimonious,bop", help="Exactly two varsets (comma-separated).")
    parser.add_argument("--models", default="MS-VAR,XGBoost", help="Exactly two models (comma-separated).")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--split", default="test")
    parser.add_argument("--input-dir", default="data/forecast_results_unified")
    parser.add_argument("--output-dir", default="data/disentangling")
    parser.add_argument("--min-obs", type=int, default=12)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--skip-dm", action="store_true")
    parser.add_argument("--run-id", default=None, help="Optional run ID to nest outputs in data/outputs/<run-id>/.")
    parser.add_argument("--output-root", default=None, help="Optional output root (overrides --run-id).")
    args = parser.parse_args()

    varsets = _parse_csv_arg(args.varsets)
    models = _parse_csv_arg(args.models)
    if len(varsets) != 2:
        raise ValueError(f"--varsets must contain exactly 2 entries; got {varsets}")
    if len(models) != 2:
        raise ValueError(f"--models must contain exactly 2 entries; got {models}")

    output_root = None
    if args.output_root:
        output_root = Path(args.output_root)
    elif args.run_id:
        output_root = DATA_DIR / "outputs" / args.run_id

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if output_root is not None:
        output_dir = output_root / "disentangling"
    output_dir.mkdir(parents=True, exist_ok=True)

    forecasts = load_unified_forecasts_for_disentangling(
        input_dir=input_dir,
        varsets=varsets,
        models=models,
        horizon=args.horizon,
        split=args.split,
    )
    if forecasts.empty:
        raise RuntimeError("No forecasts available after filtering. Check varsets/models/horizon/split.")

    aligned = build_2x2_aligned_panel(
        forecasts_long=forecasts,
        models=models,
        varsets=varsets,
    )
    if len(aligned) < args.min_obs:
        raise RuntimeError(
            f"Only {len(aligned)} aligned observations available; require at least {args.min_obs}."
        )

    rmse_long = compute_rmse_matrix(aligned, models=models, varsets=varsets)
    rmse_wide = rmse_long.pivot(index="model", columns="varset", values="rmse").reindex(
        index=models,
        columns=varsets,
    )
    effects = compute_two_by_two_effects(rmse_long, models=models, varsets=varsets)

    effects_ci = bootstrap_two_by_two_effects(
        aligned_panel=aligned,
        models=models,
        varsets=varsets,
        n_bootstrap=args.bootstrap_reps,
        seed=args.random_seed,
    )
    if not effects_ci.empty:
        effects_out = effects.merge(effects_ci, on="effect", how="left")
    else:
        effects_out = effects.copy()

    if args.skip_dm:
        dm_results = None
    else:
        dm_results = run_disentangling_dm_tests(
            aligned_panel=aligned,
            models=models,
            varsets=varsets,
            horizon=args.horizon,
        )

    aligned.to_csv(output_dir / "disentangling_aligned_panel.csv", index=False)
    rmse_long.to_csv(output_dir / "disentangling_rmse_long.csv", index=False)
    rmse_wide.to_csv(output_dir / "disentangling_rmse_matrix.csv")
    effects_out.to_csv(output_dir / "disentangling_effects.csv", index=False)
    if dm_results is not None:
        dm_results.to_csv(output_dir / "disentangling_dm_tests.csv", index=False)

    summary = {
        "varsets": varsets,
        "models": models,
        "horizon": int(args.horizon),
        "split": args.split,
        "n_aligned_obs": int(len(aligned)),
        "bootstrap_reps": int(args.bootstrap_reps),
        "random_seed": int(args.random_seed),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "skip_dm": bool(args.skip_dm),
    }
    with open(output_dir / "disentangling_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    write_run_manifest(output_dir, summary)
    if args.run_id and output_root is not None:
        write_latest_pointer(DATA_DIR / "outputs", args.run_id, output_root)

    print(f"Aligned observations: {len(aligned)}")
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
