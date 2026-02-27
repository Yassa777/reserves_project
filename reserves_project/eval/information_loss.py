"""Information-loss diagnostics under aggregation for forecast models."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from reserves_project.config.evaluation_segments import normalize_segment_keys, segment_date_mask
from reserves_project.eval.diebold_mariano import dm_test_hln


def load_aligned_aggregation_forecasts(
    input_dir: Path | str,
    aggregated_varset: str,
    disaggregated_varset: str,
    horizon: int = 1,
    split: str = "test",
    models: Sequence[str] | None = None,
    min_obs_per_model: int = 1,
) -> pd.DataFrame:
    """Load and align forecast pairs for aggregated vs disaggregated varsets."""
    input_dir = Path(input_dir)
    path_agg = input_dir / f"rolling_origin_forecasts_{aggregated_varset}.csv"
    path_dis = input_dir / f"rolling_origin_forecasts_{disaggregated_varset}.csv"
    if not path_agg.exists():
        raise FileNotFoundError(f"Missing aggregated forecast file: {path_agg}")
    if not path_dis.exists():
        raise FileNotFoundError(f"Missing disaggregated forecast file: {path_dis}")

    agg = pd.read_csv(path_agg, parse_dates=["forecast_date"])
    dis = pd.read_csv(path_dis, parse_dates=["forecast_date"])

    required_cols = {"model", "forecast_date", "horizon", "split", "actual", "forecast"}
    for name, df in [("aggregated", agg), ("disaggregated", dis)]:
        missing = required_cols.difference(df.columns)
        if missing:
            raise KeyError(f"{name} forecast file missing columns: {sorted(missing)}")

    agg = agg[(agg["horizon"] == horizon) & (agg["split"] == split)].copy()
    dis = dis[(dis["horizon"] == horizon) & (dis["split"] == split)].copy()
    if agg.empty or dis.empty:
        return pd.DataFrame(
            columns=[
                "forecast_date",
                "model",
                "actual",
                "forecast_aggregated",
                "forecast_disaggregated",
            ]
        )

    models_agg = set(agg["model"].dropna().unique())
    models_dis = set(dis["model"].dropna().unique())
    common_models = sorted(models_agg.intersection(models_dis))
    if models is not None:
        requested = [m for m in models if m in common_models]
        common_models = requested

    if not common_models:
        return pd.DataFrame(
            columns=[
                "forecast_date",
                "model",
                "actual",
                "forecast_aggregated",
                "forecast_disaggregated",
            ]
        )

    agg = agg[agg["model"].isin(common_models)].copy()
    dis = dis[dis["model"].isin(common_models)].copy()

    agg = agg[["forecast_date", "model", "actual", "forecast"]].rename(
        columns={
            "actual": "actual_aggregated",
            "forecast": "forecast_aggregated",
        }
    )
    dis = dis[["forecast_date", "model", "actual", "forecast"]].rename(
        columns={
            "actual": "actual_disaggregated",
            "forecast": "forecast_disaggregated",
        }
    )

    merged = agg.merge(dis, on=["forecast_date", "model"], how="inner")
    if merged.empty:
        return pd.DataFrame(
            columns=[
                "forecast_date",
                "model",
                "actual",
                "forecast_aggregated",
                "forecast_disaggregated",
            ]
        )

    merged["actual"] = merged[["actual_aggregated", "actual_disaggregated"]].mean(axis=1)
    merged = merged.drop(columns=["actual_aggregated", "actual_disaggregated"])
    merged = merged.dropna(subset=["actual", "forecast_aggregated", "forecast_disaggregated"])
    merged = merged.sort_values(["model", "forecast_date"])

    if min_obs_per_model > 1:
        counts = merged.groupby("model")["forecast_date"].count()
        keep = counts[counts >= min_obs_per_model].index
        merged = merged[merged["model"].isin(keep)].copy()

    return merged.reset_index(drop=True)


def _loss(actual: np.ndarray, forecast: np.ndarray, loss: str) -> np.ndarray:
    err = np.asarray(actual, dtype=float) - np.asarray(forecast, dtype=float)
    if loss == "squared":
        return err ** 2
    if loss == "absolute":
        return np.abs(err)
    raise ValueError(f"Unsupported loss function: {loss}")


def evaluate_information_loss_by_segment(
    aligned: pd.DataFrame,
    segment_keys: Iterable[str] | None = None,
    loss: str = "squared",
    horizon: int = 1,
) -> pd.DataFrame:
    """Evaluate aggregation information loss by model and segment."""
    segments = normalize_segment_keys(segment_keys, default=["all", "crisis", "tranquil"])
    if aligned.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "segment",
                "n_obs",
                "rmse_aggregated",
                "rmse_disaggregated",
                "mae_aggregated",
                "mae_disaggregated",
                "mean_loss_diff_agg_minus_disagg",
                "rmse_improvement_pct_disagg_vs_agg",
                "dm_statistic_one_sided",
                "p_value_one_sided",
                "dm_statistic_two_sided",
                "p_value_two_sided",
                "significance_two_sided",
            ]
        )

    rows = []
    for model, model_df in aligned.groupby("model", sort=False):
        dates = pd.to_datetime(model_df["forecast_date"])
        for segment in segments:
            mask = segment_date_mask(dates, segment)
            seg = model_df.loc[mask].copy()
            if seg.empty:
                continue

            actual = seg["actual"].to_numpy(dtype=float)
            f_agg = seg["forecast_aggregated"].to_numpy(dtype=float)
            f_dis = seg["forecast_disaggregated"].to_numpy(dtype=float)

            loss_agg = _loss(actual, f_agg, loss=loss)
            loss_dis = _loss(actual, f_dis, loss=loss)
            mean_loss_diff = float(np.mean(loss_agg - loss_dis))

            rmse_agg = float(np.sqrt(np.mean((actual - f_agg) ** 2)))
            rmse_dis = float(np.sqrt(np.mean((actual - f_dis) ** 2)))
            mae_agg = float(np.mean(np.abs(actual - f_agg)))
            mae_dis = float(np.mean(np.abs(actual - f_dis)))
            if np.isclose(rmse_agg, 0.0):
                rmse_improvement_pct = np.nan
            else:
                rmse_improvement_pct = float((rmse_agg - rmse_dis) / rmse_agg * 100.0)

            # One-sided: H1 aggregated has larger loss (information loss).
            dm_one = dm_test_hln(
                actual=actual,
                forecast1=f_agg,
                forecast2=f_dis,
                loss_fn=loss,
                h=horizon,
                alternative="greater",
            )
            dm_two = dm_test_hln(
                actual=actual,
                forecast1=f_agg,
                forecast2=f_dis,
                loss_fn=loss,
                h=horizon,
                alternative="two-sided",
            )

            rows.append(
                {
                    "model": model,
                    "segment": segment,
                    "n_obs": int(len(seg)),
                    "rmse_aggregated": rmse_agg,
                    "rmse_disaggregated": rmse_dis,
                    "mae_aggregated": mae_agg,
                    "mae_disaggregated": mae_dis,
                    "mean_loss_diff_agg_minus_disagg": mean_loss_diff,
                    "rmse_improvement_pct_disagg_vs_agg": rmse_improvement_pct,
                    "dm_statistic_one_sided": dm_one.get("dm_statistic"),
                    "p_value_one_sided": dm_one.get("p_value"),
                    "dm_statistic_two_sided": dm_two.get("dm_statistic"),
                    "p_value_two_sided": dm_two.get("p_value"),
                    "significance_two_sided": dm_two.get("significance"),
                }
            )

    return pd.DataFrame(rows)


def compute_cancellation_index(
    level_df: pd.DataFrame,
    component_signs: dict[str, float],
) -> pd.DataFrame:
    """Compute date-level cancellation index for disaggregated flows."""
    cols = [c for c in component_signs if c in level_df.columns]
    if not cols:
        return pd.DataFrame(columns=["date", "signed_net_flow", "gross_abs_flow", "cancellation_index", "info_loss_potential"])

    signed_parts = []
    for col in cols:
        signed_parts.append(float(component_signs[col]) * level_df[col])
    signed_sum = signed_parts[0].copy()
    for series in signed_parts[1:]:
        signed_sum = signed_sum.add(series, fill_value=0.0)

    gross_abs = pd.Series(np.zeros(len(level_df)), index=level_df.index, dtype=float)
    for col in cols:
        gross_abs = gross_abs.add(level_df[col].abs(), fill_value=0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        cancellation_index = np.where(gross_abs.values > 0, np.abs(signed_sum.values) / gross_abs.values, np.nan)
    info_loss = 1.0 - cancellation_index

    out = pd.DataFrame(
        {
            "date": pd.DatetimeIndex(level_df.index),
            "signed_net_flow": signed_sum.values,
            "gross_abs_flow": gross_abs.values,
            "cancellation_index": cancellation_index,
            "info_loss_potential": info_loss,
        }
    )
    return out


def summarize_information_loss(
    model_segment_results: pd.DataFrame,
    cancellation_index: pd.DataFrame | None = None,
    segment_keys: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Build segment-level summary table across models."""
    segments = normalize_segment_keys(segment_keys, default=["all", "crisis", "tranquil"])
    rows = []

    for segment in segments:
        subset = model_segment_results[model_segment_results["segment"] == segment].copy()
        if subset.empty:
            continue

        row = {
            "segment": segment,
            "n_models": int(subset["model"].nunique()),
            "mean_rmse_improvement_pct_disagg_vs_agg": float(subset["rmse_improvement_pct_disagg_vs_agg"].mean()),
            "share_models_disagg_better_rmse": float((subset["rmse_disaggregated"] < subset["rmse_aggregated"]).mean()),
            "share_models_significant_info_loss_10pct": float((subset["p_value_one_sided"] < 0.10).mean()),
            "mean_loss_diff_agg_minus_disagg": float(subset["mean_loss_diff_agg_minus_disagg"].mean()),
        }

        if cancellation_index is not None and not cancellation_index.empty:
            dates = pd.to_datetime(cancellation_index["date"])
            mask = segment_date_mask(dates, segment)
            csub = cancellation_index.loc[mask]
            if len(csub):
                row["mean_cancellation_index"] = float(csub["cancellation_index"].mean())
                row["mean_info_loss_potential"] = float(csub["info_loss_potential"].mean())
                row["n_dates_cancellation"] = int(len(csub))
            else:
                row["mean_cancellation_index"] = np.nan
                row["mean_info_loss_potential"] = np.nan
                row["n_dates_cancellation"] = 0

        rows.append(row)

    return pd.DataFrame(rows)
