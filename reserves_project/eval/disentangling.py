"""Model-vs-information disentangling utilities for 2x2 forecast designs."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd

from reserves_project.eval.diebold_mariano import dm_test_hln


def load_unified_forecasts_for_disentangling(
    input_dir: Path | str,
    varsets: Sequence[str],
    models: Sequence[str],
    horizon: int = 1,
    split: str = "test",
) -> pd.DataFrame:
    """Load and stack unified forecast files for requested varsets/models."""
    input_dir = Path(input_dir)
    rows = []

    for varset in varsets:
        path = input_dir / f"rolling_origin_forecasts_{varset}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing unified forecast file for varset '{varset}': {path}")

        df = pd.read_csv(path, parse_dates=["forecast_date"])
        required_cols = {"model", "forecast_date", "horizon", "split", "actual", "forecast"}
        missing = required_cols.difference(df.columns)
        if missing:
            raise KeyError(f"File {path} missing required columns: {sorted(missing)}")

        df = df[(df["horizon"] == horizon) & (df["split"] == split) & (df["model"].isin(models))]
        if df.empty:
            continue
        df = df[["forecast_date", "model", "actual", "forecast"]].copy()
        df["varset"] = varset
        rows.append(df)

    if not rows:
        return pd.DataFrame(columns=["forecast_date", "model", "actual", "forecast", "varset"])

    out = pd.concat(rows, ignore_index=True)
    return out


def build_2x2_aligned_panel(
    forecasts_long: pd.DataFrame,
    models: Sequence[str],
    varsets: Sequence[str],
) -> pd.DataFrame:
    """Build a date-aligned panel with common support across all 2x2 cells."""
    if len(models) != 2:
        raise ValueError("Disentangling requires exactly 2 models.")
    if len(varsets) != 2:
        raise ValueError("Disentangling requires exactly 2 varsets.")

    if forecasts_long.empty:
        return pd.DataFrame(columns=["forecast_date", "actual"])

    required_cols = {"forecast_date", "model", "varset", "actual", "forecast"}
    missing = required_cols.difference(forecasts_long.columns)
    if missing:
        raise KeyError(f"Missing required columns for panel build: {sorted(missing)}")

    df = forecasts_long.copy()
    df["forecast_date"] = pd.to_datetime(df["forecast_date"])

    # Ensure each required cell exists at least once.
    required_cells = set(product(models, varsets))
    present_cells = set(zip(df["model"], df["varset"]))
    missing_cells = required_cells.difference(present_cells)
    if missing_cells:
        raise ValueError(f"Missing required model/varset cells: {sorted(missing_cells)}")

    actual = (
        df.groupby("forecast_date", as_index=True)["actual"]
        .mean()
        .rename("actual")
    )

    pivot = df.pivot_table(
        index="forecast_date",
        columns=["model", "varset"],
        values="forecast",
        aggfunc="mean",
    )

    required_cols_multi = pd.MultiIndex.from_tuples(list(required_cells), names=["model", "varset"])
    missing_multi = required_cols_multi.difference(pivot.columns)
    if len(missing_multi):
        raise ValueError(f"Missing required model/varset forecast columns: {missing_multi.tolist()}")

    pivot = pivot[required_cols_multi]
    merged = pd.concat([actual, pivot], axis=1).dropna()
    merged = merged.sort_index()

    # Flatten multi-index columns for downstream use.
    flat_cols = ["actual"]
    for model, varset in required_cols_multi:
        flat_cols.append(f"forecast__{model}__{varset}")
    merged.columns = flat_cols

    out = merged.reset_index().rename(columns={"index": "forecast_date"})
    return out


def _rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    err = np.asarray(actual, dtype=float) - np.asarray(forecast, dtype=float)
    return float(np.sqrt(np.mean(err ** 2)))


def compute_rmse_matrix(
    aligned_panel: pd.DataFrame,
    models: Sequence[str],
    varsets: Sequence[str],
) -> pd.DataFrame:
    """Compute RMSE for each model/varset cell on common aligned dates."""
    if aligned_panel.empty:
        return pd.DataFrame(columns=["model", "varset", "rmse", "n"])

    actual = aligned_panel["actual"].values
    rows = []
    for model, varset in product(models, varsets):
        col = f"forecast__{model}__{varset}"
        if col not in aligned_panel.columns:
            raise KeyError(f"Missing aligned forecast column: {col}")
        rows.append(
            {
                "model": model,
                "varset": varset,
                "rmse": _rmse(actual, aligned_panel[col].values),
                "n": int(len(aligned_panel)),
            }
        )
    return pd.DataFrame(rows)


def compute_two_by_two_effects(
    rmse_matrix: pd.DataFrame,
    models: Sequence[str],
    varsets: Sequence[str],
) -> pd.DataFrame:
    """Compute architecture/info effects and interaction (difference-in-differences)."""
    if len(models) != 2:
        raise ValueError("Effects require exactly 2 models.")
    if len(varsets) != 2:
        raise ValueError("Effects require exactly 2 varsets.")

    m0, m1 = models
    v0, v1 = varsets

    pivot = rmse_matrix.pivot(index="model", columns="varset", values="rmse")
    for m, v in product(models, varsets):
        if m not in pivot.index or v not in pivot.columns:
            raise ValueError(f"RMSE matrix missing ({m}, {v}) cell.")

    r00 = float(pivot.loc[m0, v0])
    r01 = float(pivot.loc[m0, v1])
    r10 = float(pivot.loc[m1, v0])
    r11 = float(pivot.loc[m1, v1])

    architecture_v0 = r10 - r00
    architecture_v1 = r11 - r01
    information_m0 = r01 - r00
    information_m1 = r11 - r10

    rows = [
        {
            "effect": f"architecture_effect_at_{v0}",
            "value": architecture_v0,
            "definition": f"RMSE({m1},{v0}) - RMSE({m0},{v0})",
        },
        {
            "effect": f"architecture_effect_at_{v1}",
            "value": architecture_v1,
            "definition": f"RMSE({m1},{v1}) - RMSE({m0},{v1})",
        },
        {
            "effect": "architecture_effect_avg",
            "value": float((architecture_v0 + architecture_v1) / 2.0),
            "definition": "Average architecture contrast across varsets",
        },
        {
            "effect": f"information_effect_at_{m0}",
            "value": information_m0,
            "definition": f"RMSE({m0},{v1}) - RMSE({m0},{v0})",
        },
        {
            "effect": f"information_effect_at_{m1}",
            "value": information_m1,
            "definition": f"RMSE({m1},{v1}) - RMSE({m1},{v0})",
        },
        {
            "effect": "information_effect_avg",
            "value": float((information_m0 + information_m1) / 2.0),
            "definition": "Average information contrast across models",
        },
        {
            "effect": "interaction_did",
            "value": float(information_m1 - information_m0),
            "definition": "Difference-in-differences interaction term",
        },
    ]
    return pd.DataFrame(rows)


def bootstrap_two_by_two_effects(
    aligned_panel: pd.DataFrame,
    models: Sequence[str],
    varsets: Sequence[str],
    n_bootstrap: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> pd.DataFrame:
    """Bootstrap confidence intervals for disentangling effects."""
    if n_bootstrap <= 0:
        return pd.DataFrame(columns=["effect", "ci_lower", "ci_upper", "boot_std", "n_bootstrap"])
    if aligned_panel.empty:
        return pd.DataFrame(columns=["effect", "ci_lower", "ci_upper", "boot_std", "n_bootstrap"])

    rng = np.random.default_rng(seed)
    n = len(aligned_panel)
    effect_draws: Dict[str, list[float]] = {}

    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        sample = aligned_panel.iloc[sample_idx].reset_index(drop=True)
        rmse = compute_rmse_matrix(sample, models=models, varsets=varsets)
        eff = compute_two_by_two_effects(rmse, models=models, varsets=varsets)
        for _, row in eff.iterrows():
            effect_draws.setdefault(row["effect"], []).append(float(row["value"]))

    alpha = 1.0 - ci
    rows = []
    for effect, draws in effect_draws.items():
        values = np.asarray(draws, dtype=float)
        rows.append(
            {
                "effect": effect,
                "ci_lower": float(np.quantile(values, alpha / 2)),
                "ci_upper": float(np.quantile(values, 1 - alpha / 2)),
                "boot_std": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                "n_bootstrap": int(n_bootstrap),
            }
        )
    return pd.DataFrame(rows)


def run_disentangling_dm_tests(
    aligned_panel: pd.DataFrame,
    models: Sequence[str],
    varsets: Sequence[str],
    horizon: int = 1,
) -> pd.DataFrame:
    """Run DM tests for model and information contrasts on aligned dates."""
    if aligned_panel.empty:
        return pd.DataFrame(
            columns=[
                "comparison_type",
                "anchor",
                "forecast_1",
                "forecast_2",
                "n_obs",
                "dm_statistic",
                "p_value",
                "mean_loss_diff",
                "better_forecast",
                "significance",
            ]
        )

    actual = aligned_panel["actual"].values
    m0, m1 = models
    v0, v1 = varsets

    rows = []
    # Model contrasts within each varset.
    for varset in varsets:
        f1_name = f"{m0}|{varset}"
        f2_name = f"{m1}|{varset}"
        f1 = aligned_panel[f"forecast__{m0}__{varset}"].values
        f2 = aligned_panel[f"forecast__{m1}__{varset}"].values
        res = dm_test_hln(actual, f1, f2, loss_fn="squared", h=horizon)
        rows.append(
            {
                "comparison_type": "model",
                "anchor": varset,
                "forecast_1": f1_name,
                "forecast_2": f2_name,
                "n_obs": int(res.get("n_obs", 0)),
                "dm_statistic": res.get("dm_statistic"),
                "p_value": res.get("p_value"),
                "mean_loss_diff": res.get("mean_loss_diff"),
                "better_forecast": res.get("better_forecast"),
                "significance": res.get("significance"),
            }
        )

    # Information contrasts within each model.
    for model in models:
        f1_name = f"{model}|{v0}"
        f2_name = f"{model}|{v1}"
        f1 = aligned_panel[f"forecast__{model}__{v0}"].values
        f2 = aligned_panel[f"forecast__{model}__{v1}"].values
        res = dm_test_hln(actual, f1, f2, loss_fn="squared", h=horizon)
        rows.append(
            {
                "comparison_type": "information",
                "anchor": model,
                "forecast_1": f1_name,
                "forecast_2": f2_name,
                "n_obs": int(res.get("n_obs", 0)),
                "dm_statistic": res.get("dm_statistic"),
                "p_value": res.get("p_value"),
                "mean_loss_diff": res.get("mean_loss_diff"),
                "better_forecast": res.get("better_forecast"),
                "significance": res.get("significance"),
            }
        )

    return pd.DataFrame(rows)
