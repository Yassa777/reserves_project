#!/usr/bin/env python3
"""Run conditional scenario analysis using MS-VARX."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from reserves_project.config.paths import DATA_DIR
from reserves_project.config.varsets import TARGET_VAR, VARIABLE_SETS
from reserves_project.eval.unified_evaluator import load_varset_levels
from reserves_project.scenarios.definitions import POLICY_SCENARIOS, Scenario
from reserves_project.scenarios.paths import build_baseline_exog_path, build_scenario_exog_path
from reserves_project.scenarios.msvarx import fit_msvarx, forecast_msvarx
from reserves_project.utils.run_manifest import write_run_manifest, write_latest_pointer


def _parse_cols(value: str | None) -> list[str]:
    if not value:
        return []
    return [c.strip() for c in value.split(",") if c.strip()]


def _resolve_cols(varset: str, df_cols: list[str], endog_cols: list[str], exog_cols: list[str]) -> tuple[list[str], list[str]]:
    if endog_cols:
        endog = [c for c in endog_cols if c in df_cols]
    else:
        config = VARIABLE_SETS.get(varset, {})
        endog = config.get("scenario_endog") or [TARGET_VAR]
        endog = [c for c in endog if c in df_cols]
    if not endog:
        endog = [TARGET_VAR]

    if exog_cols:
        exog = [c for c in exog_cols if c in df_cols]
    else:
        config = VARIABLE_SETS.get(varset, {})
        exog = config.get("scenario_exog")
        if exog:
            exog = [c for c in exog if c in df_cols]
        else:
            exog = [c for c in df_cols if c not in endog and c != "split"]
    return endog, exog


def _select_scenarios(keys: str | None) -> dict[str, Scenario]:
    if not keys or keys.lower() == "all":
        return POLICY_SCENARIOS
    wanted = [k.strip() for k in keys.split(",") if k.strip()]
    scenarios = {}
    for key in wanted:
        if key not in POLICY_SCENARIOS:
            raise ValueError(f"Unknown scenario key: {key}")
        scenarios[key] = POLICY_SCENARIOS[key]
    return scenarios


def _regime_path_from_arg(value: str | None) -> np.ndarray | None:
    if not value:
        return None
    vals = [v.strip() for v in value.split(",") if v.strip()]
    return np.asarray([int(v) for v in vals], dtype=int)


def main():
    parser = argparse.ArgumentParser(description="MS-VARX scenario analysis")
    parser.add_argument("--varset", default="parsimonious")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--ar-order", type=int, default=2)
    parser.add_argument("--exog-forecast", choices=["naive", "arima"], default="naive")
    parser.add_argument("--scenarios", default="all", help="Comma list of scenario keys (default all)")
    parser.add_argument("--endog-cols", default=None, help="Comma-separated endogenous columns")
    parser.add_argument("--exog-cols", default=None, help="Comma-separated exogenous columns")
    parser.add_argument("--regime-mode", choices=["free", "locked", "path"], default="free")
    parser.add_argument("--regime-path", default=None, help="Comma list of regime indices for path mode")
    parser.add_argument("--output-dir", default="data/scenario_analysis")
    parser.add_argument("--run-id", default=None, help="Optional run ID to nest outputs in data/outputs/<run-id>/.")
    parser.add_argument("--output-root", default=None, help="Optional output root (overrides --run-id).")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    output_root = None
    if args.output_root:
        output_root = Path(args.output_root)
    elif args.run_id:
        output_root = DATA_DIR / "outputs" / args.run_id

    output_dir = Path(args.output_dir)
    if output_root is not None:
        output_dir = output_root / "scenario_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_varset_levels(args.varset)
    df = df.sort_index()
    cols = [c for c in df.columns if c != "split"]

    endog_cols = _parse_cols(args.endog_cols)
    exog_cols = _parse_cols(args.exog_cols)
    endog_cols, exog_cols = _resolve_cols(args.varset, cols, endog_cols, exog_cols)

    scenarios = _select_scenarios(args.scenarios)

    msvar, diff_data = fit_msvarx(
        df,
        endog_cols=endog_cols,
        exog_cols=exog_cols,
        ar_order=args.ar_order,
    )

    regime_probs = msvar.smoothed_probs_[-1] if msvar.smoothed_probs_ is not None else None
    regime_path = _regime_path_from_arg(args.regime_path) if args.regime_mode == "path" else None

    baseline_exog = None
    if exog_cols:
        baseline_exog = build_baseline_exog_path(df[exog_cols], exog_cols, args.horizon, method=args.exog_forecast)

    summary_rows = []
    path_frames = []
    exog_frames = []

    for key, scenario in scenarios.items():
        scenario = Scenario(
            name=scenario.name,
            description=scenario.description,
            horizon_months=args.horizon,
            shocks=scenario.shocks,
            profile=scenario.profile,
        )

        exog_future = None
        if baseline_exog is not None and not baseline_exog.empty:
            exog_future = build_scenario_exog_path(baseline_exog, scenario)
            exog_out = exog_future.copy()
            exog_out["scenario"] = scenario.name
            exog_frames.append(exog_out.reset_index().rename(columns={"index": "date"}))

        forecast = forecast_msvarx(
            msvar,
            diff_data,
            df,
            target_col=TARGET_VAR,
            endog_cols=endog_cols,
            exog_cols=exog_cols,
            exog_future_levels=exog_future,
            horizon=args.horizon,
            regime_mode=args.regime_mode,
            regime_path=regime_path,
            regime_probs=regime_probs,
        )

        last_level = float(df[TARGET_VAR].iloc[-1])
        end_level = float(forecast.iloc[-1])
        total_change = end_level - last_level
        pct_change = (total_change / last_level) * 100 if last_level != 0 else np.nan
        avg_monthly = total_change / args.horizon if args.horizon else np.nan

        summary_rows.append({
            "scenario": scenario.name,
            "description": scenario.description,
            "start_level": last_level,
            "end_level": end_level,
            "total_change": total_change,
            "pct_change": pct_change,
            "avg_monthly_change": avg_monthly,
            "min_level": float(forecast.min()),
            "max_level": float(forecast.max()),
        })

        path_frame = forecast.to_frame(name=TARGET_VAR)
        path_frame["scenario"] = scenario.name
        path_frames.append(path_frame.reset_index().rename(columns={"index": "date"}))

    summary_df = pd.DataFrame(summary_rows).sort_values("end_level", ascending=False)
    summary_path = output_dir / "msvarx_scenario_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    paths_df = pd.concat(path_frames, ignore_index=True)
    paths_path = output_dir / "msvarx_scenario_paths.csv"
    paths_df.to_csv(paths_path, index=False)

    if exog_frames:
        exog_df = pd.concat(exog_frames, ignore_index=True)
        exog_df.to_csv(output_dir / "msvarx_scenario_exog_paths.csv", index=False)

    if not args.no_figures:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(12, 7))
            for scenario in summary_df["scenario"]:
                series = paths_df[paths_df["scenario"] == scenario].set_index("date")[TARGET_VAR]
                linestyle = "-" if scenario.lower() == "baseline" else "--"
                ax.plot(series.index, series.values, linestyle=linestyle, linewidth=1.8, label=scenario)
            ax.set_title("MS-VARX Scenario Paths")
            ax.set_xlabel("Date")
            ax.set_ylabel("Gross Reserves (USD million)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=9)
            fig.tight_layout()
            fig.savefig(output_dir / "msvarx_scenario_fan_chart.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
        except Exception:
            pass

    config = {
        "varset": args.varset,
        "horizon": args.horizon,
        "ar_order": args.ar_order,
        "exog_forecast": args.exog_forecast,
        "scenarios": list(scenarios.keys()),
        "endog_cols": endog_cols,
        "exog_cols": exog_cols,
        "regime_mode": args.regime_mode,
        "regime_path": regime_path.tolist() if regime_path is not None else None,
        "output_dir": str(output_dir),
    }
    write_run_manifest(output_dir, config)
    if args.run_id and output_root is not None:
        write_latest_pointer(DATA_DIR / "outputs", args.run_id, output_root)

    print(f"Saved: {summary_path}")
    print(f"Saved: {paths_path}")
    if exog_frames:
        print(f"Saved: {output_dir / 'msvarx_scenario_exog_paths.csv'}")
    print(f"Outputs in {output_dir}")


if __name__ == "__main__":
    main()
