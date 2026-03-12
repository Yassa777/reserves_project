"""Model-vs-information disentangling utilities for 2x2 forecast designs."""

from __future__ import annotations

from itertools import product
from math import ceil
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from reserves_project.eval.diebold_mariano import dm_test_hln

DEFAULT_BOOTSTRAP_METHOD = "stationary_block"
DEFAULT_CI_LEVEL = 0.95


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


def _ordered_model_varset_index(models: Sequence[str], varsets: Sequence[str]) -> pd.MultiIndex:
    """Return a deterministic model/varset column order."""
    return pd.MultiIndex.from_tuples(
        [(model, varset) for model in models for varset in varsets],
        names=["model", "varset"],
    )


def _validate_common_actuals(actual_panel: pd.DataFrame, tol: float = 1e-8) -> None:
    """Ensure aligned cells share the same realized value on each date."""
    if actual_panel.empty:
        return

    spread = actual_panel.max(axis=1) - actual_panel.min(axis=1)
    bad_dates = spread[spread > tol]
    if not bad_dates.empty:
        first_bad = bad_dates.index[0]
        raise ValueError(
            "Aligned actuals differ across cells on "
            f"{pd.Timestamp(first_bad).date()}: max spread {bad_dates.iloc[0]:.6f}"
        )


def _resolve_block_length(n_obs: int, block_length: int | None = None) -> int:
    """Resolve stationary-bootstrap block length using the MCS rule of thumb."""
    if n_obs <= 0:
        return 0
    if block_length is None:
        return max(1, int(ceil(n_obs ** (1 / 3))))
    return max(1, min(int(block_length), int(n_obs)))


def _stationary_bootstrap_indices(
    n_obs: int,
    rng: np.random.Generator,
    block_length: int,
) -> np.ndarray:
    """Draw stationary-bootstrap indices with circular wraparound."""
    if n_obs <= 0:
        return np.array([], dtype=int)
    if n_obs == 1:
        return np.array([0], dtype=int)

    p_new_block = 1.0 / float(block_length)
    indices = np.empty(n_obs, dtype=int)
    indices[0] = int(rng.integers(0, n_obs))

    for t in range(1, n_obs):
        if rng.random() < p_new_block:
            indices[t] = int(rng.integers(0, n_obs))
        else:
            indices[t] = (indices[t - 1] + 1) % n_obs

    return indices


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

    actual_pivot = df.pivot_table(
        index="forecast_date",
        columns=["model", "varset"],
        values="actual",
        aggfunc="mean",
    )
    forecast_pivot = df.pivot_table(
        index="forecast_date",
        columns=["model", "varset"],
        values="forecast",
        aggfunc="mean",
    )

    required_cols_multi = _ordered_model_varset_index(models=models, varsets=varsets)
    missing_actual = required_cols_multi.difference(actual_pivot.columns)
    if len(missing_actual):
        raise ValueError(f"Missing required model/varset actual columns: {missing_actual.tolist()}")
    missing_forecast = required_cols_multi.difference(forecast_pivot.columns)
    if len(missing_forecast):
        raise ValueError(f"Missing required model/varset forecast columns: {missing_forecast.tolist()}")

    actual_pivot = actual_pivot[required_cols_multi]
    forecast_pivot = forecast_pivot[required_cols_multi]

    common_panel = pd.concat(
        {
            "actual": actual_pivot,
            "forecast": forecast_pivot,
        },
        axis=1,
    ).dropna()
    common_panel = common_panel.sort_index()

    actual_complete = common_panel["actual"]
    forecast_complete = common_panel["forecast"]
    _validate_common_actuals(actual_complete)

    merged = pd.concat(
        [
            actual_complete.iloc[:, 0].rename("actual"),
            forecast_complete,
        ],
        axis=1,
    )

    # Flatten multi-index columns for downstream use.
    flat_cols = ["actual"]
    for model, varset in required_cols_multi:
        flat_cols.append(f"forecast__{model}__{varset}")
    merged.columns = flat_cols

    out = merged.reset_index().rename(columns={"index": "forecast_date"})
    return out


def build_pairwise_aligned_panel(
    forecasts_long: pd.DataFrame,
    models: Sequence[str],
) -> pd.DataFrame:
    """Build a date-aligned panel with common support for a model pair."""
    if len(models) != 2:
        raise ValueError("Pairwise alignment requires exactly 2 models.")
    if forecasts_long.empty:
        return pd.DataFrame(columns=["forecast_date", "actual"])

    required_cols = {"forecast_date", "model", "actual", "forecast"}
    missing = required_cols.difference(forecasts_long.columns)
    if missing:
        raise KeyError(f"Missing required columns for pairwise panel: {sorted(missing)}")

    df = forecasts_long.copy()
    df["forecast_date"] = pd.to_datetime(df["forecast_date"])

    required_models = set(models)
    present_models = set(df["model"])
    missing_models = required_models.difference(present_models)
    if missing_models:
        raise ValueError(f"Missing required models: {sorted(missing_models)}")

    actual_pivot = df.pivot_table(
        index="forecast_date",
        columns="model",
        values="actual",
        aggfunc="mean",
    )
    forecast_pivot = df.pivot_table(
        index="forecast_date",
        columns="model",
        values="forecast",
        aggfunc="mean",
    )
    actual_pivot = actual_pivot[models]
    forecast_pivot = forecast_pivot[models]

    common_panel = pd.concat(
        {
            "actual": actual_pivot,
            "forecast": forecast_pivot,
        },
        axis=1,
    ).dropna()
    common_panel = common_panel.sort_index()

    actual_complete = common_panel["actual"]
    forecast_complete = common_panel["forecast"]
    _validate_common_actuals(actual_complete)

    merged = pd.concat(
        [
            actual_complete.iloc[:, 0].rename("actual"),
            forecast_complete,
        ],
        axis=1,
    )
    merged.columns = ["actual"] + [f"forecast__{model}" for model in models]
    return merged.reset_index().rename(columns={"index": "forecast_date"})


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


def compute_relative_rmse_reduction(
    aligned_panel: pd.DataFrame,
    headline_model: str,
    benchmark_model: str,
) -> Dict[str, float]:
    """Compute RMSE levels and relative reduction for a pairwise comparison."""
    if aligned_panel.empty:
        raise ValueError("Cannot compute RMSE reduction on an empty panel.")

    actual = aligned_panel["actual"].to_numpy(dtype=float)
    rmse_model = _rmse(actual, aligned_panel[f"forecast__{headline_model}"].to_numpy(dtype=float))
    rmse_benchmark = _rmse(actual, aligned_panel[f"forecast__{benchmark_model}"].to_numpy(dtype=float))
    rmse_gain = float(rmse_benchmark - rmse_model)
    rmse_ratio = float(rmse_model / rmse_benchmark) if rmse_benchmark else np.nan
    pct_reduction = float(100.0 * (1.0 - rmse_ratio)) if np.isfinite(rmse_ratio) else np.nan

    return {
        "headline_model": headline_model,
        "benchmark_model": benchmark_model,
        "n_obs": int(len(aligned_panel)),
        "rmse_model": rmse_model,
        "rmse_benchmark": rmse_benchmark,
        "rmse_gain": rmse_gain,
        "rmse_ratio": rmse_ratio,
        "pct_reduction": pct_reduction,
    }


def bootstrap_two_by_two_effects(
    aligned_panel: pd.DataFrame,
    models: Sequence[str],
    varsets: Sequence[str],
    n_bootstrap: int = 1000,
    block_length: int | None = None,
    seed: int = 42,
    ci: float = DEFAULT_CI_LEVEL,
) -> pd.DataFrame:
    """Stationary block-bootstrap confidence intervals for disentangling effects."""
    if n_bootstrap <= 0:
        return pd.DataFrame(
            columns=[
                "effect",
                "ci_lower",
                "ci_upper",
                "boot_std",
                "n_bootstrap",
                "block_length",
                "ci_level",
                "bootstrap_method",
                "ci_excludes_zero",
            ]
        )
    if aligned_panel.empty:
        return pd.DataFrame(
            columns=[
                "effect",
                "ci_lower",
                "ci_upper",
                "boot_std",
                "n_bootstrap",
                "block_length",
                "ci_level",
                "bootstrap_method",
                "ci_excludes_zero",
            ]
        )

    rng = np.random.default_rng(seed)
    n = len(aligned_panel)
    resolved_block_length = _resolve_block_length(n_obs=n, block_length=block_length)
    effect_draws: Dict[str, list[float]] = {}

    for _ in range(n_bootstrap):
        sample_idx = _stationary_bootstrap_indices(
            n_obs=n,
            rng=rng,
            block_length=resolved_block_length,
        )
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
                "block_length": int(resolved_block_length),
                "ci_level": float(ci),
                "bootstrap_method": DEFAULT_BOOTSTRAP_METHOD,
                "ci_excludes_zero": bool(
                    (float(np.quantile(values, alpha / 2)) > 0.0)
                    or (float(np.quantile(values, 1 - alpha / 2)) < 0.0)
                ),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_relative_rmse_reduction(
    aligned_panel: pd.DataFrame,
    headline_model: str,
    benchmark_model: str,
    n_bootstrap: int = 1000,
    block_length: int | None = None,
    seed: int = 42,
    ci: float = DEFAULT_CI_LEVEL,
) -> Dict[str, float]:
    """Stationary block-bootstrap confidence intervals for RMSE reduction claims."""
    if aligned_panel.empty:
        return {}
    if n_bootstrap <= 0:
        return {}

    rng = np.random.default_rng(seed)
    n = len(aligned_panel)
    resolved_block_length = _resolve_block_length(n_obs=n, block_length=block_length)

    metric_draws: Dict[str, list[float]] = {
        "rmse_model": [],
        "rmse_benchmark": [],
        "rmse_gain": [],
        "rmse_ratio": [],
        "pct_reduction": [],
    }

    for _ in range(n_bootstrap):
        sample_idx = _stationary_bootstrap_indices(
            n_obs=n,
            rng=rng,
            block_length=resolved_block_length,
        )
        sample = aligned_panel.iloc[sample_idx].reset_index(drop=True)
        metrics = compute_relative_rmse_reduction(
            aligned_panel=sample,
            headline_model=headline_model,
            benchmark_model=benchmark_model,
        )
        for metric, value in metrics.items():
            if metric in metric_draws:
                metric_draws[metric].append(float(value))

    alpha = 1.0 - ci
    out: Dict[str, float] = {
        "n_bootstrap": int(n_bootstrap),
        "block_length": int(resolved_block_length),
        "ci_level": float(ci),
        "bootstrap_method": DEFAULT_BOOTSTRAP_METHOD,
    }
    for metric, draws in metric_draws.items():
        values = np.asarray(draws, dtype=float)
        out[f"{metric}_ci_lower"] = float(np.quantile(values, alpha / 2))
        out[f"{metric}_ci_upper"] = float(np.quantile(values, 1 - alpha / 2))
        out[f"{metric}_boot_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan

    out["rmse_gain_ci_excludes_zero"] = bool(
        (out["rmse_gain_ci_lower"] > 0.0) or (out["rmse_gain_ci_upper"] < 0.0)
    )
    out["pct_reduction_ci_excludes_zero"] = bool(
        (out["pct_reduction_ci_lower"] > 0.0) or (out["pct_reduction_ci_upper"] < 0.0)
    )
    return out


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
