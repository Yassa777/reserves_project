#!/usr/bin/env python3
"""Run regime characterization diagnostics for fitted MS-VAR models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reserves_project.config.paths import DATA_DIR
from reserves_project.config.varsets import TARGET_VAR, TRAIN_END, VALID_END
from reserves_project.eval.unified_evaluator import load_varset_levels
from reserves_project.models.ms_switching_var import MarkovSwitchingVAR
from reserves_project.models.msvar_diagnostics import (
    classification_certainty_df,
    expected_durations_df,
    fit_diagnostics_dict,
    smoothed_probabilities_df,
    transition_matrix_df,
)
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
    parser = argparse.ArgumentParser(description="Run regime characterization for MS-VAR.")
    parser.add_argument("--varset", default="parsimonious")
    parser.add_argument("--n-regimes", type=int, default=2)
    parser.add_argument("--ar-order", type=int, default=1)
    parser.add_argument("--fit-split", choices=["train", "train_valid", "full"], default="full")
    parser.add_argument("--train-end", default=str(TRAIN_END.date()))
    parser.add_argument("--valid-end", default=str(VALID_END.date()))
    parser.add_argument("--fit-start", default=None, help="Optional YYYY-MM-DD fit start date.")
    parser.add_argument("--fit-end", default=None, help="Optional YYYY-MM-DD fit end date.")
    parser.add_argument("--output-dir", default="data/regime_characterization")
    parser.add_argument("--run-id", default=None, help="Optional run ID to nest outputs in data/outputs/<run-id>/.")
    parser.add_argument("--output-root", default=None, help="Optional output root (overrides --run-id).")
    args = parser.parse_args()

    if args.n_regimes < 1:
        raise ValueError("--n-regimes must be >= 1")

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
        output_dir = output_root / "regime_characterization"
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

    smoothed_dates = diffs.index[args.ar_order :]
    probs_df = smoothed_probabilities_df(
        model,
        dates=smoothed_dates,
        train_end=train_end,
        valid_end=valid_end,
    )
    transition_df = transition_matrix_df(model)
    durations_df = expected_durations_df(model)
    certainty_df = classification_certainty_df(model)

    probs_df.to_csv(output_dir / "regime_smoothed_probabilities.csv", index=False)
    transition_df.to_csv(output_dir / "regime_transition_matrix.csv", index=False)
    durations_df.to_csv(output_dir / "regime_durations.csv", index=False)
    certainty_df.to_csv(output_dir / "regime_classification_certainty.csv", index=False)

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
            "n_smoothed_obs": int(len(smoothed_dates)),
            "target_col": TARGET_VAR,
            "system_cols": system_cols,
            "init_state_method": init_meta,
        }
    )
    with open(output_dir / "regime_fit_diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

    config = {
        "varset": args.varset,
        "n_regimes": args.n_regimes,
        "ar_order": args.ar_order,
        "fit_split": args.fit_split,
        "train_end": str(train_end.date()),
        "valid_end": str(valid_end.date()),
        "fit_start": str(fit_start.date()) if fit_start is not None else None,
        "fit_end": str(fit_end.date()) if fit_end is not None else None,
        "output_dir": str(output_dir),
    }
    write_run_manifest(output_dir, config, extra={"n_obs": int(len(smoothed_dates))})
    if args.run_id and output_root is not None:
        write_latest_pointer(DATA_DIR / "outputs", args.run_id, output_root)

    print(f"Saved regime characterization outputs in {output_dir}")


if __name__ == "__main__":
    main()
