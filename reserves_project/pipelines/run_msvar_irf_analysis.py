#!/usr/bin/env python3
"""Run regime-conditional MS-VAR impulse-response analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reserves_project.config.paths import DATA_DIR
from reserves_project.config.varsets import TARGET_VAR, TRAIN_END, VALID_END
from reserves_project.eval.msvar_irf import generalized_irf, girf_to_long_df, summarize_regime_comparison
from reserves_project.eval.unified_evaluator import load_varset_levels
from reserves_project.models.ms_switching_var import MarkovSwitchingVAR
from reserves_project.models.msvar_diagnostics import fit_diagnostics_dict
from reserves_project.utils.run_manifest import write_latest_pointer, write_run_manifest


def _build_init_states(target_diff: pd.Series, n_regimes: int) -> tuple[np.ndarray, dict]:
    rolling_vol = target_diff.rolling(6).std()
    if rolling_vol.notna().sum() == 0:
        init_states = np.zeros(len(target_diff), dtype=int)
        return init_states, {"method": "zeros", "threshold": None, "n_regimes": int(n_regimes)}

    vol_filled = rolling_vol.fillna(float(rolling_vol.median()))
    if n_regimes == 1:
        init_states = np.zeros(len(target_diff), dtype=int)
        return init_states, {"method": "single_regime", "threshold": None, "n_regimes": 1}

    if n_regimes == 2:
        threshold = float(vol_filled.quantile(0.75))
        init_states = (vol_filled > threshold).astype(int).values
        return init_states, {"method": "rolling_vol_threshold", "threshold": threshold, "n_regimes": 2}

    quantiles = np.linspace(0.0, 1.0, n_regimes + 1)
    bins = np.quantile(vol_filled.values, quantiles)
    bins = np.unique(bins)
    if len(bins) < 2:
        init_states = np.zeros(len(target_diff), dtype=int)
        return init_states, {"method": "quantile_fallback_zeros", "threshold": None, "n_regimes": int(n_regimes)}

    init_states = np.digitize(vol_filled.values, bins[1:-1], right=True).astype(int)
    init_states = np.clip(init_states, 0, n_regimes - 1)
    return init_states, {"method": "rolling_vol_quantile_bins", "threshold": None, "n_regimes": int(n_regimes)}


def _fit_window(
    df: pd.DataFrame,
    fit_split: str,
    train_end: pd.Timestamp,
    valid_end: pd.Timestamp,
    fit_start: pd.Timestamp | None,
    fit_end: pd.Timestamp | None,
) -> pd.DataFrame:
    out = df.copy()
    if fit_split == "train":
        out = out.loc[out.index <= train_end]
    elif fit_split == "train_valid":
        out = out.loc[out.index <= valid_end]
    elif fit_split == "full":
        pass
    else:
        raise ValueError(f"Unknown fit_split: {fit_split}")

    if fit_start is not None:
        out = out.loc[out.index >= fit_start]
    if fit_end is not None:
        out = out.loc[out.index <= fit_end]
    return out


def main():
    parser = argparse.ArgumentParser(description="Run regime-conditional MS-VAR IRF analysis.")
    parser.add_argument("--varset", default="parsimonious")
    parser.add_argument("--n-regimes", type=int, default=2)
    parser.add_argument("--ar-order", type=int, default=1)
    parser.add_argument("--max-horizon", type=int, default=24)
    parser.add_argument("--fit-split", choices=["train", "train_valid", "full"], default="full")
    parser.add_argument("--train-end", default=str(TRAIN_END.date()))
    parser.add_argument("--valid-end", default=str(VALID_END.date()))
    parser.add_argument("--fit-start", default=None, help="Optional YYYY-MM-DD fit start date.")
    parser.add_argument("--fit-end", default=None, help="Optional YYYY-MM-DD fit end date.")
    parser.add_argument("--difference-irf", action="store_true", help="Report IRFs in differenced units (non-cumulative).")
    parser.add_argument("--output-dir", default="data/msvar_irf")
    parser.add_argument("--run-id", default=None, help="Optional run ID to nest outputs in data/outputs/<run-id>/.")
    parser.add_argument("--output-root", default=None, help="Optional output root (overrides --run-id).")
    args = parser.parse_args()

    if args.n_regimes < 1:
        raise ValueError("--n-regimes must be >= 1")
    if args.max_horizon < 0:
        raise ValueError("--max-horizon must be >= 0")

    train_end = pd.Timestamp(args.train_end)
    valid_end = pd.Timestamp(args.valid_end)
    fit_start = pd.Timestamp(args.fit_start) if args.fit_start else None
    fit_end = pd.Timestamp(args.fit_end) if args.fit_end else None

    output_root = None
    if args.output_root:
        output_root = Path(args.output_root)
    elif args.run_id:
        output_root = DATA_DIR / "outputs" / args.run_id

    output_dir = Path(args.output_dir)
    if output_root is not None:
        output_dir = output_root / "msvar_irf"
    output_dir.mkdir(parents=True, exist_ok=True)

    levels = load_varset_levels(args.varset).sort_index()
    levels = _fit_window(
        levels,
        fit_split=args.fit_split,
        train_end=train_end,
        valid_end=valid_end,
        fit_start=fit_start,
        fit_end=fit_end,
    )
    levels = levels.dropna()
    if levels.empty:
        raise RuntimeError("No observations available after applying fit window and dropping missing values.")

    system_cols = [TARGET_VAR] + [c for c in levels.columns if c != TARGET_VAR]
    levels = levels[system_cols]
    diffs = levels.diff().dropna()
    if len(diffs) < max(20, args.ar_order + 2):
        raise RuntimeError(
            f"Insufficient differenced observations for MS-VAR: {len(diffs)} rows "
            f"(need at least {max(20, args.ar_order + 2)})."
        )

    init_states, init_meta = _build_init_states(diffs[TARGET_VAR], n_regimes=args.n_regimes)
    model = MarkovSwitchingVAR(n_regimes=args.n_regimes, ar_order=args.ar_order)
    model.fit(diffs.values, init_states=init_states)

    cumulative = not args.difference_irf
    regime_frames = []
    for regime in range(args.n_regimes):
        irf = generalized_irf(
            model=model,
            regime=regime,
            max_horizon=args.max_horizon,
            cumulative=cumulative,
        )
        df = girf_to_long_df(
            girf=irf,
            var_names=system_cols,
            regime=regime,
            cumulative=cumulative,
        )
        df.to_csv(output_dir / f"msvar_irf_regime{regime}.csv", index=False)
        regime_frames.append(df)

    if len(regime_frames) >= 2:
        comp = summarize_regime_comparison(regime_frames[0], regime_frames[1])
        comp.to_csv(output_dir / "msvar_irf_regime_comparison.csv", index=False)
    else:
        comp = pd.DataFrame()

    diagnostics = fit_diagnostics_dict(model)
    diagnostics.update(
        {
            "varset": args.varset,
            "fit_split": args.fit_split,
            "fit_start": str(fit_start.date()) if fit_start is not None else None,
            "fit_end": str(fit_end.date()) if fit_end is not None else None,
            "train_end": str(train_end.date()),
            "valid_end": str(valid_end.date()),
            "n_levels_obs": int(len(levels)),
            "n_diff_obs": int(len(diffs)),
            "target_col": TARGET_VAR,
            "system_cols": system_cols,
            "max_horizon": int(args.max_horizon),
            "response_type": "difference" if args.difference_irf else "cumulative",
            "init_state_method": init_meta,
            "comparison_table_rows": int(len(comp)),
        }
    )
    with open(output_dir / "msvar_irf_diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

    config = {
        "varset": args.varset,
        "n_regimes": args.n_regimes,
        "ar_order": args.ar_order,
        "max_horizon": args.max_horizon,
        "fit_split": args.fit_split,
        "train_end": str(train_end.date()),
        "valid_end": str(valid_end.date()),
        "fit_start": str(fit_start.date()) if fit_start is not None else None,
        "fit_end": str(fit_end.date()) if fit_end is not None else None,
        "difference_irf": bool(args.difference_irf),
        "output_dir": str(output_dir),
    }
    write_run_manifest(output_dir, config, extra={"n_obs": int(len(diffs))})
    if args.run_id and output_root is not None:
        write_latest_pointer(DATA_DIR / "outputs", args.run_id, output_root)

    print(f"Saved MS-VAR IRF outputs in {output_dir}")


if __name__ == "__main__":
    main()
