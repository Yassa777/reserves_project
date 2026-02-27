#!/usr/bin/env python3
"""Run formal information-loss tests under aggregation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reserves_project.config.evaluation_segments import DEFAULT_SEGMENT_ORDER, normalize_segment_keys
from reserves_project.config.paths import DATA_DIR
from reserves_project.eval.information_loss import (
    compute_cancellation_index,
    evaluate_information_loss_by_segment,
    load_aligned_aggregation_forecasts,
    summarize_information_loss,
)
from reserves_project.eval.unified_evaluator import load_varset_levels
from reserves_project.utils.run_manifest import write_latest_pointer, write_run_manifest


def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def _parse_component_signs(value: str | None) -> dict[str, float]:
    # Format: exports_usd_m:+1,imports_usd_m:-1,...
    if not value:
        return {
            "exports_usd_m": 1.0,
            "imports_usd_m": -1.0,
            "remittances_usd_m": 1.0,
            "tourism_usd_m": 1.0,
            "cse_net_usd_m": 1.0,
        }
    out: dict[str, float] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid component sign spec: {item}")
        key, val = item.split(":", 1)
        out[key.strip()] = float(val.strip())
    return out


def main():
    parser = argparse.ArgumentParser(description="Run information-loss tests under variable aggregation.")
    parser.add_argument("--aggregated-varset", default="parsimonious")
    parser.add_argument("--disaggregated-varset", default="bop")
    parser.add_argument("--models", default=None, help="Optional comma-separated models to test.")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--split", default="test")
    parser.add_argument("--loss", choices=["squared", "absolute"], default="squared")
    parser.add_argument("--segments", default=",".join(DEFAULT_SEGMENT_ORDER), help="Comma-separated segment keys.")
    parser.add_argument("--min-obs-per-model", type=int, default=12)
    parser.add_argument("--component-signs", default=None, help="Component signs, e.g. exports:+1,imports:-1")
    parser.add_argument("--input-dir", default="data/forecast_results_unified")
    parser.add_argument("--output-dir", default="data/information_loss")
    parser.add_argument("--run-id", default=None, help="Optional run ID to nest outputs in data/outputs/<run-id>/.")
    parser.add_argument("--output-root", default=None, help="Optional output root (overrides --run-id).")
    args = parser.parse_args()

    segments = normalize_segment_keys(_parse_csv(args.segments), default=DEFAULT_SEGMENT_ORDER)
    models = _parse_csv(args.models) if args.models else None
    component_signs = _parse_component_signs(args.component_signs)

    output_root = None
    if args.output_root:
        output_root = Path(args.output_root)
    elif args.run_id:
        output_root = DATA_DIR / "outputs" / args.run_id

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if output_root is not None:
        output_dir = output_root / "information_loss"
    output_dir.mkdir(parents=True, exist_ok=True)

    aligned = load_aligned_aggregation_forecasts(
        input_dir=input_dir,
        aggregated_varset=args.aggregated_varset,
        disaggregated_varset=args.disaggregated_varset,
        horizon=args.horizon,
        split=args.split,
        models=models,
        min_obs_per_model=args.min_obs_per_model,
    )
    if aligned.empty:
        raise RuntimeError("No aligned forecast pairs available for information-loss test.")

    model_segment = evaluate_information_loss_by_segment(
        aligned=aligned,
        segment_keys=segments,
        loss=args.loss,
        horizon=args.horizon,
    )
    if model_segment.empty:
        raise RuntimeError("No model-segment results computed; check segment/date coverage.")

    disagg_levels = load_varset_levels(args.disaggregated_varset).sort_index()
    cancellation = compute_cancellation_index(disagg_levels, component_signs=component_signs)
    summary = summarize_information_loss(
        model_segment_results=model_segment,
        cancellation_index=cancellation,
        segment_keys=segments,
    )

    aligned.to_csv(output_dir / "information_loss_aligned_forecasts.csv", index=False)
    model_segment.to_csv(output_dir / "information_loss_model_segment_tests.csv", index=False)
    cancellation.to_csv(output_dir / "information_loss_cancellation_index.csv", index=False)
    summary.to_csv(output_dir / "information_loss_segment_summary.csv", index=False)

    diagnostics = {
        "aggregated_varset": args.aggregated_varset,
        "disaggregated_varset": args.disaggregated_varset,
        "models_requested": models,
        "models_tested": sorted(model_segment["model"].dropna().unique().tolist()),
        "horizon": int(args.horizon),
        "split": args.split,
        "loss": args.loss,
        "segments": segments,
        "min_obs_per_model": int(args.min_obs_per_model),
        "component_signs": component_signs,
        "n_aligned_rows": int(len(aligned)),
        "output_dir": str(output_dir),
    }
    with open(output_dir / "information_loss_diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

    write_run_manifest(output_dir, diagnostics)
    if args.run_id and output_root is not None:
        write_latest_pointer(DATA_DIR / "outputs", args.run_id, output_root)

    print(f"Aligned rows: {len(aligned)}")
    print(f"Saved information-loss outputs in {output_dir}")


if __name__ == "__main__":
    main()
