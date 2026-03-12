#!/usr/bin/env python3
"""Run 2x2 model-vs-information disentangling analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from reserves_project.config.paths import DATA_DIR
from reserves_project.eval.disentangling import (
    build_2x2_aligned_panel,
    build_pairwise_aligned_panel,
    bootstrap_relative_rmse_reduction,
    bootstrap_two_by_two_effects,
    compute_relative_rmse_reduction,
    compute_rmse_matrix,
    compute_two_by_two_effects,
    load_unified_forecasts_for_disentangling,
    run_disentangling_dm_tests,
)
from reserves_project.utils.run_manifest import write_latest_pointer, write_run_manifest


def _parse_csv_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _fmt_num(value: float, decimals: int = 1) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.{decimals}f}"


def _fmt_ci(lower: float, upper: float, decimals: int = 1) -> str:
    if pd.isna(lower) or pd.isna(upper):
        return "--"
    return f"[{float(lower):.{decimals}f}, {float(upper):.{decimals}f}]"


def _fmt_month(value) -> str:
    return pd.Timestamp(value).strftime("%Y:%m")


def _fmt_window(start_value, end_value) -> str:
    return f"{_fmt_month(start_value)}--{_fmt_month(end_value)}"


def _pretty_varset_name(varset: str) -> str:
    mapping = {
        "parsimonious": "Parsimonious",
        "bop": "BoP",
        "monetary": "Monetary",
        "pca": "PCA",
        "full": "Full",
    }
    return mapping.get(varset, varset)


def _pretty_effect_name(effect: str) -> str:
    mapping = {
        "architecture_effect_avg": "Architecture average",
        "architecture_effect_at_parsimonious": "Architecture at Parsimonious",
        "architecture_effect_at_bop": "Architecture at BoP",
        "information_effect_avg": "Information average",
        "information_effect_at_MS-VAR": "Information at MS-VAR",
        "information_effect_at_XGBoost": "Information at XGBoost",
        "interaction_did": "Interaction (DiD)",
    }
    return mapping.get(effect, effect.replace("_", " "))


def _generate_main_rmse_table(
    rmse_wide: pd.DataFrame,
    aligned_panel: pd.DataFrame,
    varsets: list[str],
) -> str:
    left_varset, right_varset = varsets
    row_means = rmse_wide.mean(axis=1)
    col_means = rmse_wide.mean(axis=0)
    window = _fmt_window(aligned_panel["forecast_date"].min(), aligned_panel["forecast_date"].max())
    n_obs = int(len(aligned_panel))

    latex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Factorial Decomposition of Architecture and Information Effects}",
        r"\label{tab:did}",
        r"\begin{tabular}{lcc|c}",
        r"\toprule",
        rf"& {_pretty_varset_name(left_varset)} & {_pretty_varset_name(right_varset)} & Row Mean \\",
        r"\midrule",
    ]

    for model in rmse_wide.index:
        latex.append(
            f"{model} & "
            f"{_fmt_num(rmse_wide.loc[model, left_varset])} & "
            f"{_fmt_num(rmse_wide.loc[model, right_varset])} & "
            f"{_fmt_num(row_means.loc[model])} \\\\"
        )

    latex.extend(
        [
            r"\midrule",
            f"Column Mean & {_fmt_num(col_means.loc[left_varset])} & {_fmt_num(col_means.loc[right_varset])} & \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{tablenotes}",
            r"\small",
            (
                r"\item \textit{Notes:} Entries are RMSE values computed on the common-support "
                rf"$h=1$ post-default window ({window}; $n={n_obs}$). "
                r"Appendix Table~\ref{tab:disentangling_uncertainty} reports stationary block-bootstrap "
                r"confidence intervals for the architecture, information, and interaction contrasts."
            ),
            r"\end{tablenotes}",
            r"\end{table}",
        ]
    )
    return "\n".join(latex) + "\n"


def _generate_disentangling_uncertainty_table(
    effects_out: pd.DataFrame,
    aligned_panel: pd.DataFrame,
) -> str:
    window = _fmt_window(aligned_panel["forecast_date"].min(), aligned_panel["forecast_date"].max())
    n_obs = int(len(aligned_panel))
    block_length = int(effects_out["block_length"].dropna().iloc[0]) if "block_length" in effects_out and effects_out["block_length"].notna().any() else "--"
    n_bootstrap = int(effects_out["n_bootstrap"].dropna().iloc[0]) if "n_bootstrap" in effects_out and effects_out["n_bootstrap"].notna().any() else "--"
    ci_level = int(round(100 * float(effects_out["ci_level"].dropna().iloc[0]))) if "ci_level" in effects_out and effects_out["ci_level"].notna().any() else 95
    order = [
        "architecture_effect_avg",
        "architecture_effect_at_parsimonious",
        "architecture_effect_at_bop",
        "information_effect_avg",
        "information_effect_at_MS-VAR",
        "information_effect_at_XGBoost",
        "interaction_did",
    ]
    display = effects_out.set_index("effect").loc[order].reset_index()

    latex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Uncertainty Around the Architecture-vs-Information Decomposition}",
        r"\label{tab:disentangling_uncertainty}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        rf"Contrast & Estimate & {ci_level}\% CI & Excludes 0 \\",
        r"\midrule",
    ]

    for _, row in display.iterrows():
        latex.append(
            f"{_pretty_effect_name(row['effect'])} & "
            f"{_fmt_num(row['value'])} & "
            f"{_fmt_ci(row.get('ci_lower'), row.get('ci_upper'))} & "
            f"{'Yes' if bool(row.get('ci_excludes_zero')) else 'No'} \\\\"
        )

    latex.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{tablenotes}",
            r"\footnotesize",
            (
                r"\item \textit{Notes:} Positive architecture effects mean XGBoost has higher RMSE than "
                r"MS-VAR, so larger values favour the regime-switching architecture. Positive information "
                r"effects mean the BoP specification raises RMSE relative to the parsimonious specification "
                r"under the paper's sign convention. Confidence intervals use a stationary block bootstrap "
                rf"over rolling-origin dates ({window}; $n={n_obs}$, {n_bootstrap} replications, block length {block_length})."
            ),
            r"\end{tablenotes}",
            r"\end{table}",
        ]
    )
    return "\n".join(latex) + "\n"


def _generate_headline_reduction_table(headline_df: pd.DataFrame) -> str:
    ci_level = int(round(100 * float(headline_df["ci_level"].dropna().iloc[0]))) if "ci_level" in headline_df and headline_df["ci_level"].notna().any() else 95
    n_bootstrap = int(headline_df["n_bootstrap"].dropna().iloc[0]) if "n_bootstrap" in headline_df and headline_df["n_bootstrap"].notna().any() else "--"

    latex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Uncertainty Around Benchmark RMSE Reductions}",
        r"\label{tab:headline_rmse_uncertainty}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        rf"Variable set & Window & $n$ & RMSE (MS-VAR) & RMSE (Na\"{{\i}}ve) & Gain & Reduction (\%) & {ci_level}\% CI \\",
        r"\midrule",
    ]

    for _, row in headline_df.iterrows():
        latex.append(
            f"{_pretty_varset_name(row['varset'])} & "
            f"{row['window']} & "
            f"{int(row['n_obs'])} & "
            f"{_fmt_num(row['rmse_model'])} & "
            f"{_fmt_num(row['rmse_benchmark'])} & "
            f"{_fmt_num(row['rmse_gain'])} & "
            f"{_fmt_num(row['pct_reduction'])} & "
            f"{_fmt_ci(row['pct_reduction_ci_lower'], row['pct_reduction_ci_upper'])} \\\\"
        )

    latex.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{tablenotes}",
            r"\footnotesize",
            (
                r"\item \textit{Notes:} Gain equals RMSE(Na\"{\i}ve) $-$ RMSE(MS-VAR). "
                rf"Confidence intervals use a stationary block bootstrap over each variable set's observed "
                rf"rolling-origin test window ({n_bootstrap} replications, block length $\lceil T^{{1/3}} \rceil$ for each row)."
            ),
            r"\end{tablenotes}",
            r"\end{table}",
        ]
    )
    return "\n".join(latex) + "\n"


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
    parser.add_argument("--block-length", type=int, default=None)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--headline-model", default=None, help="Model for benchmark-RMSE reduction; defaults to first --models entry.")
    parser.add_argument("--benchmark-model", default="Naive", help="Benchmark model for headline RMSE reduction.")
    parser.add_argument("--headline-varsets", default=None, help="Varsets for benchmark-RMSE reduction table; defaults to --varsets.")
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
    if not (0.0 < args.ci_level < 1.0):
        raise ValueError(f"--ci-level must lie strictly between 0 and 1; got {args.ci_level}")

    headline_model = args.headline_model or models[0]
    benchmark_model = args.benchmark_model
    headline_varsets = _parse_csv_arg(args.headline_varsets) if args.headline_varsets else list(varsets)
    if not headline_varsets:
        raise ValueError("No headline varsets supplied.")
    if headline_model == benchmark_model:
        raise ValueError("--headline-model and --benchmark-model must differ.")

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
        block_length=args.block_length,
        seed=args.random_seed,
        ci=args.ci_level,
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

    headline_rows: list[dict] = []
    for varset in headline_varsets:
        pairwise_forecasts = load_unified_forecasts_for_disentangling(
            input_dir=input_dir,
            varsets=[varset],
            models=[headline_model, benchmark_model],
            horizon=args.horizon,
            split=args.split,
        )
        if pairwise_forecasts.empty:
            raise RuntimeError(
                f"No forecasts available for headline comparison in varset '{varset}' "
                f"with models {headline_model} and {benchmark_model}."
            )

        pairwise_aligned = build_pairwise_aligned_panel(
            forecasts_long=pairwise_forecasts,
            models=[headline_model, benchmark_model],
        )
        if len(pairwise_aligned) < args.min_obs:
            raise RuntimeError(
                f"Only {len(pairwise_aligned)} aligned observations available for headline comparison "
                f"in varset '{varset}'; require at least {args.min_obs}."
            )

        row = compute_relative_rmse_reduction(
            aligned_panel=pairwise_aligned,
            headline_model=headline_model,
            benchmark_model=benchmark_model,
        )
        row.update(
            bootstrap_relative_rmse_reduction(
                aligned_panel=pairwise_aligned,
                headline_model=headline_model,
                benchmark_model=benchmark_model,
                n_bootstrap=args.bootstrap_reps,
                block_length=args.block_length,
                seed=args.random_seed,
                ci=args.ci_level,
            )
        )
        row.update(
            {
                "varset": varset,
                "window_start": str(pd.Timestamp(pairwise_aligned["forecast_date"].min()).date()),
                "window_end": str(pd.Timestamp(pairwise_aligned["forecast_date"].max()).date()),
                "window": _fmt_window(
                    pairwise_aligned["forecast_date"].min(),
                    pairwise_aligned["forecast_date"].max(),
                ),
            }
        )
        headline_rows.append(row)

    headline_df = pd.DataFrame(headline_rows)
    headline_df["varset"] = pd.Categorical(headline_df["varset"], categories=headline_varsets, ordered=True)
    headline_df = headline_df.sort_values("varset").reset_index(drop=True)
    headline_df["varset"] = headline_df["varset"].astype(str)

    aligned.to_csv(output_dir / "disentangling_aligned_panel.csv", index=False)
    rmse_long.to_csv(output_dir / "disentangling_rmse_long.csv", index=False)
    rmse_wide.to_csv(output_dir / "disentangling_rmse_matrix.csv")
    effects_out.to_csv(output_dir / "disentangling_effects.csv", index=False)
    headline_df.to_csv(output_dir / "headline_rmse_reductions.csv", index=False)
    if dm_results is not None:
        dm_results.to_csv(output_dir / "disentangling_dm_tests.csv", index=False)

    table_main = _generate_main_rmse_table(
        rmse_wide=rmse_wide,
        aligned_panel=aligned,
        varsets=varsets,
    )
    table_uncertainty = _generate_disentangling_uncertainty_table(
        effects_out=effects_out,
        aligned_panel=aligned,
    )
    table_headline = _generate_headline_reduction_table(headline_df=headline_df)

    (output_dir / "table_did_rmse.tex").write_text(table_main)
    (output_dir / "table_a9_disentangling_uncertainty.tex").write_text(table_uncertainty)
    (output_dir / "table_a10_headline_rmse_uncertainty.tex").write_text(table_headline)

    summary = {
        "varsets": varsets,
        "models": models,
        "horizon": int(args.horizon),
        "split": args.split,
        "n_aligned_obs": int(len(aligned)),
        "bootstrap_reps": int(args.bootstrap_reps),
        "block_length": int(effects_out["block_length"].dropna().iloc[0]) if "block_length" in effects_out and effects_out["block_length"].notna().any() else None,
        "ci_level": float(args.ci_level),
        "random_seed": int(args.random_seed),
        "headline_model": headline_model,
        "benchmark_model": benchmark_model,
        "headline_varsets": headline_varsets,
        "aligned_window_start": str(pd.Timestamp(aligned["forecast_date"].min()).date()),
        "aligned_window_end": str(pd.Timestamp(aligned["forecast_date"].max()).date()),
        "headline_reductions": headline_df.to_dict(orient="records"),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "skip_dm": bool(args.skip_dm),
        "generated_tables": [
            "table_did_rmse.tex",
            "table_a9_disentangling_uncertainty.tex",
            "table_a10_headline_rmse_uncertainty.tex",
        ],
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
