#!/usr/bin/env python3
"""Run unified rolling-origin evaluations across all variable sets."""

from __future__ import annotations

import argparse
from pathlib import Path

from reserves_project.config.varsets import (
    TARGET_VAR,
    TRAIN_END,
    VALID_END,
    VARSET_ORDER,
)
from reserves_project.eval.unified_evaluator import (
    ARIMAExogForecaster,
    NaiveExogForecaster,
    RollingOriginEvaluator,
    build_models,
    load_varset_levels,
    summarize_results,
)


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
    parser.add_argument("--xgb-params", default=None, help="Path to JSON with tuned XGBoost params")
    parser.add_argument("--lstm-params", default=None, help="Path to JSON with tuned LSTM params")
    parser.add_argument("--output-dir", default="data/forecast_results_unified")
    args = parser.parse_args()

    varsets = [v.strip() for v in args.varsets.split(",") if v.strip()]
    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]

    exog_forecaster = ARIMAExogForecaster() if args.exog_forecast == "arima" else NaiveExogForecaster()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _load_params(path: str | None):
        if not path:
            return None
        import json
        with open(path, "r") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and "best_params" in payload:
            return payload.get("best_params") or None
        return payload

    xgb_params = _load_params(args.xgb_params)
    lstm_params = _load_params(args.lstm_params)

    for varset in varsets:
        df = load_varset_levels(varset)
        df = df.sort_index()
        exog_cols = [c for c in df.columns if c != TARGET_VAR]

        models = build_models(
            TARGET_VAR,
            exog_cols,
            include_bvar=not args.exclude_bvar,
            include_ms=args.include_ms,
            include_lstm=args.include_lstm,
            include_xgb=not args.exclude_xgb,
            include_llsv=args.include_llsv,
            include_bop=args.include_bop,
            xgb_params=xgb_params,
            lstm_params=lstm_params,
        )

        evaluator = RollingOriginEvaluator(
            data=df,
            target_col=TARGET_VAR,
            exog_cols=exog_cols,
            models=models,
            horizons=horizons,
            train_end=TRAIN_END,
            valid_end=VALID_END,
            refit_interval=args.refit_interval,
            exog_mode=args.exog_mode,
            exog_forecaster=exog_forecaster,
        )

        results = evaluator.run()
        summary = summarize_results(results, df.loc[df.index <= TRAIN_END, TARGET_VAR])

        results.to_csv(output_dir / f"rolling_origin_forecasts_{varset}.csv", index=False)
        summary.to_csv(output_dir / f"rolling_origin_summary_{varset}.csv", index=False)
        print(f"Saved {varset}: {len(results)} forecasts")

    print(f"Outputs in {output_dir}")


if __name__ == "__main__":
    main()
