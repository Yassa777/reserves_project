#!/usr/bin/env python3
"""Run unified rolling-origin evaluations across variable sets."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd

from reserves_project.config.paths import DATA_DIR
from reserves_project.config.varsets import TARGET_VAR, TRAIN_END, VALID_END, VARSET_ORDER
from reserves_project.config.evaluation_segments import DEFAULT_SEGMENT_ORDER, normalize_segment_keys
from reserves_project.eval.dma import augment_with_dma_dms
from reserves_project.eval.unified_evaluator import (
    ARIMAExogForecaster,
    NaiveExogForecaster,
    RollingOriginEvaluator,
    build_models,
    load_varset_levels,
    summarize_results,
)
from reserves_project.utils.run_manifest import write_run_manifest, write_latest_pointer


@dataclass
class SplitSpec:
    """Evaluation split definition."""

    label: str
    train_end: pd.Timestamp
    valid_end: pd.Timestamp


def _parse_split_specs(
    split_specs_raw: str | None,
    default_train_end: pd.Timestamp,
    default_valid_end: pd.Timestamp,
) -> List[SplitSpec]:
    """
    Parse split specs in format:
    `label:YYYY-MM-DD:YYYY-MM-DD,label2:YYYY-MM-DD:YYYY-MM-DD`
    """

    if not split_specs_raw:
        return [SplitSpec("baseline", default_train_end, default_valid_end)]

    specs: List[SplitSpec] = []
    seen = set()
    raw_items = [item.strip() for item in split_specs_raw.split(",") if item.strip()]
    for idx, item in enumerate(raw_items):
        parts = [p.strip() for p in item.split(":") if p.strip()]
        if len(parts) == 3:
            label, train_end_raw, valid_end_raw = parts
        elif len(parts) == 2:
            label = f"split_{idx + 1}"
            train_end_raw, valid_end_raw = parts
        else:
            raise ValueError(
                "Invalid split spec entry. Use label:train_end:valid_end, "
                f"got '{item}'."
            )
        if label in seen:
            raise ValueError(f"Duplicate split label '{label}'.")
        seen.add(label)
        specs.append(
            SplitSpec(
                label=label,
                train_end=pd.Timestamp(train_end_raw),
                valid_end=pd.Timestamp(valid_end_raw),
            )
        )
    return specs


def _load_params(path: str | None):
    if not path:
        return None
    import json

    with open(path, "r") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "best_params" in payload:
        return payload.get("best_params") or None
    return payload


def _save_unified_outputs(
    directory: Path,
    varset: str,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    summary_segments: pd.DataFrame,
    summary_full: pd.DataFrame,
    summary_common: pd.DataFrame,
    dma_weights: pd.DataFrame | None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    results.to_csv(directory / f"rolling_origin_forecasts_{varset}.csv", index=False)
    summary.to_csv(directory / f"rolling_origin_summary_{varset}.csv", index=False)
    summary_segments.to_csv(directory / f"rolling_origin_summary_segments_{varset}.csv", index=False)
    summary_full.to_csv(directory / f"rolling_origin_summary_full_{varset}.csv", index=False)
    summary_common.to_csv(directory / f"rolling_origin_summary_common_dates_{varset}.csv", index=False)
    if dma_weights is not None and not dma_weights.empty:
        dma_weights.to_csv(directory / f"dma_weights_{varset}.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Run unified evaluator across variable sets")
    parser.add_argument("--varsets", default=",".join(VARSET_ORDER), help="Comma-separated varset list")
    parser.add_argument("--refit-interval", type=int, default=12)
    parser.add_argument("--horizons", default="1,3,6,12")
    parser.add_argument("--exog-mode", choices=["forecast", "actual"], default="forecast")
    parser.add_argument("--exog-forecast", choices=["naive", "arima"], default="naive")
    parser.add_argument("--include-ms", action="store_true")
    parser.add_argument("--include-lstm", action="store_true")
    parser.add_argument("--include-llsv", action="store_true")
    parser.add_argument("--include-bop", action="store_true")
    parser.add_argument("--exclude-bvar", action="store_true")
    parser.add_argument("--exclude-xgb", action="store_true")
    parser.add_argument("--include-xgb-quantile", action="store_true")
    parser.add_argument("--include-dma", action="store_true")
    parser.add_argument("--dma-alpha", type=float, default=0.99)
    parser.add_argument("--dma-warmup-periods", type=int, default=12)
    parser.add_argument("--dma-variance-window", type=int, default=24)
    parser.add_argument("--dma-min-model-obs", type=int, default=24)
    parser.add_argument("--dma-model-pool", default=None, help="Comma-separated model names for DMA pool")
    parser.add_argument("--window-mode", choices=["full", "common_dates"], default="full")
    parser.add_argument("--segments", default=",".join(DEFAULT_SEGMENT_ORDER), help="Comma-separated segment keys")
    parser.add_argument("--train-end", default=str(TRAIN_END.date()))
    parser.add_argument("--valid-end", default=str(VALID_END.date()))
    parser.add_argument(
        "--split-specs",
        default=None,
        help=(
            "Optional split robustness specs: "
            "label:train_end:valid_end,label2:train_end:valid_end"
        ),
    )
    parser.add_argument("--xgb-params", default=None, help="Path to JSON with tuned XGBoost params")
    parser.add_argument("--lstm-params", default=None, help="Path to JSON with tuned LSTM params")
    parser.add_argument("--output-dir", default="data/forecast_results_unified")
    parser.add_argument("--run-id", default=None, help="Optional run ID to nest outputs in data/outputs/<run-id>/.")
    parser.add_argument("--output-root", default=None, help="Optional output root (overrides --run-id).")
    args = parser.parse_args()

    varsets = [v.strip() for v in args.varsets.split(",") if v.strip()]
    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
    segments = normalize_segment_keys([s.strip() for s in args.segments.split(",") if s.strip()])
    dma_pool = [m.strip() for m in args.dma_model_pool.split(",") if m.strip()] if args.dma_model_pool else None

    output_root = None
    if args.output_root:
        output_root = Path(args.output_root)
    elif args.run_id:
        output_root = DATA_DIR / "outputs" / args.run_id

    output_dir = Path(args.output_dir)
    if output_root is not None:
        output_dir = output_root / "forecast_results_unified"
    output_dir.mkdir(parents=True, exist_ok=True)

    xgb_params = _load_params(args.xgb_params)
    lstm_params = _load_params(args.lstm_params)

    split_specs = _parse_split_specs(
        split_specs_raw=args.split_specs,
        default_train_end=pd.Timestamp(args.train_end),
        default_valid_end=pd.Timestamp(args.valid_end),
    )
    primary_label = split_specs[0].label

    split_summary_blocks = []
    dma_weight_blocks = []

    for split_spec in split_specs:
        split_dir = output_dir if len(split_specs) == 1 else (output_dir / "splits" / split_spec.label)
        split_dir.mkdir(parents=True, exist_ok=True)

        for varset in varsets:
            df = load_varset_levels(varset).sort_index()
            exog_cols = [c for c in df.columns if c != TARGET_VAR]

            models = build_models(
                TARGET_VAR,
                exog_cols,
                include_bvar=not args.exclude_bvar,
                include_ms=args.include_ms,
                include_lstm=args.include_lstm,
                include_xgb=not args.exclude_xgb,
                include_xgb_quantile=args.include_xgb_quantile,
                include_llsv=args.include_llsv,
                include_bop=args.include_bop,
                xgb_params=xgb_params,
                lstm_params=lstm_params,
            )

            exog_forecaster = ARIMAExogForecaster() if args.exog_forecast == "arima" else NaiveExogForecaster()
            evaluator = RollingOriginEvaluator(
                data=df,
                target_col=TARGET_VAR,
                exog_cols=exog_cols,
                models=models,
                horizons=horizons,
                train_end=split_spec.train_end,
                valid_end=split_spec.valid_end,
                refit_interval=args.refit_interval,
                exog_mode=args.exog_mode,
                exog_forecaster=exog_forecaster,
            )

            results = evaluator.run()
            dma_weights = pd.DataFrame()
            if args.include_dma:
                results, dma_weights = augment_with_dma_dms(
                    results=results,
                    alpha=args.dma_alpha,
                    variance_window=args.dma_variance_window,
                    warmup_periods=args.dma_warmup_periods,
                    min_model_obs=args.dma_min_model_obs,
                    model_pool=dma_pool,
                )
                if not dma_weights.empty:
                    dma_meta = dma_weights.copy()
                    dma_meta["varset"] = varset
                    dma_meta["split_label"] = split_spec.label
                    dma_meta["train_end"] = split_spec.train_end
                    dma_meta["valid_end"] = split_spec.valid_end
                    dma_weight_blocks.append(dma_meta)

            train_series = df.loc[df.index <= split_spec.train_end, TARGET_VAR]
            summary = summarize_results(
                results,
                train_series,
                window_mode=args.window_mode,
                segment_keys=["all"],
            )
            summary_segments = summarize_results(
                results,
                train_series,
                window_mode=args.window_mode,
                segment_keys=segments,
            )
            summary_full = summarize_results(
                results,
                train_series,
                window_mode="full",
                segment_keys=["all"],
            )
            summary_common = summarize_results(
                results,
                train_series,
                window_mode="common_dates",
                segment_keys=["all"],
            )

            _save_unified_outputs(
                directory=split_dir,
                varset=varset,
                results=results,
                summary=summary,
                summary_segments=summary_segments,
                summary_full=summary_full,
                summary_common=summary_common,
                dma_weights=dma_weights,
            )

            if len(split_specs) > 1 and split_spec.label == primary_label:
                _save_unified_outputs(
                    directory=output_dir,
                    varset=varset,
                    results=results,
                    summary=summary,
                    summary_segments=summary_segments,
                    summary_full=summary_full,
                    summary_common=summary_common,
                    dma_weights=dma_weights,
                )

            split_block = summary.copy()
            split_block["varset"] = varset
            split_block["split_label"] = split_spec.label
            split_block["train_end"] = split_spec.train_end
            split_block["valid_end"] = split_spec.valid_end
            split_summary_blocks.append(split_block)

            print(
                f"Saved {varset} ({split_spec.label}): "
                f"{len(results)} forecasts, {results['model'].nunique()} models"
            )

    if split_summary_blocks:
        split_summary = pd.concat(split_summary_blocks, ignore_index=True)
        split_summary.to_csv(output_dir / "split_robustness_summary.csv", index=False)

        split_specs_df = pd.DataFrame(
            [
                {
                    "split_label": s.label,
                    "train_end": s.train_end,
                    "valid_end": s.valid_end,
                    "is_primary": s.label == primary_label,
                }
                for s in split_specs
            ]
        )
        split_specs_df.to_csv(output_dir / "split_robustness_specs.csv", index=False)

        key = split_summary[
            (split_summary["split"] == "test")
            & (split_summary["horizon"] == 1)
        ].copy()
        if "segment" in key.columns:
            key = key[key["segment"] == "all"]
        if not key.empty:
            key["rank_within_split_varset"] = key.groupby(["split_label", "varset"])["rmse"].rank(method="min")
            key.to_csv(output_dir / "split_robustness_h1_test.csv", index=False)

            best = (
                key.sort_values(["split_label", "varset", "rmse"])
                .groupby(["split_label", "varset"], as_index=False)
                .first()
                .rename(
                    columns={
                        "model": "best_model",
                        "rmse": "best_rmse",
                        "mae": "best_mae",
                        "mape": "best_mape",
                    }
                )
            )
            best.to_csv(output_dir / "split_robustness_best_models.csv", index=False)

            stability = (
                key.groupby(["varset", "model"], as_index=False)
                .agg(
                    mean_rmse=("rmse", "mean"),
                    std_rmse=("rmse", "std"),
                    mean_rank=("rank_within_split_varset", "mean"),
                    std_rank=("rank_within_split_varset", "std"),
                    n_splits=("split_label", "nunique"),
                )
                .sort_values(["varset", "mean_rank", "mean_rmse"])
            )
            stability.to_csv(output_dir / "split_robustness_model_stability.csv", index=False)

    if dma_weight_blocks:
        dma_weights_all = pd.concat(dma_weight_blocks, ignore_index=True)
        dma_weights_all.to_csv(output_dir / "dma_weights.csv", index=False)

    config = {
        "varsets": varsets,
        "split_specs": [
            {
                "label": s.label,
                "train_end": str(s.train_end.date()),
                "valid_end": str(s.valid_end.date()),
            }
            for s in split_specs
        ],
        "primary_split_label": primary_label,
        "refit_interval": args.refit_interval,
        "horizons": horizons,
        "exog_mode": args.exog_mode,
        "exog_forecast": args.exog_forecast,
        "include_ms": args.include_ms,
        "include_lstm": args.include_lstm,
        "include_llsv": args.include_llsv,
        "include_bop": args.include_bop,
        "include_dma": args.include_dma,
        "dma_alpha": args.dma_alpha,
        "dma_warmup_periods": args.dma_warmup_periods,
        "dma_variance_window": args.dma_variance_window,
        "dma_min_model_obs": args.dma_min_model_obs,
        "dma_model_pool": dma_pool,
        "exclude_bvar": args.exclude_bvar,
        "exclude_xgb": args.exclude_xgb,
        "include_xgb_quantile": args.include_xgb_quantile,
        "window_mode": args.window_mode,
        "segments": segments,
        "xgb_params": xgb_params,
        "lstm_params": lstm_params,
        "output_dir": str(output_dir),
    }
    write_run_manifest(output_dir, config)
    if args.run_id and output_root is not None:
        write_latest_pointer(DATA_DIR / "outputs", args.run_id, output_root)

    print(f"Outputs in {output_dir}")


if __name__ == "__main__":
    main()

