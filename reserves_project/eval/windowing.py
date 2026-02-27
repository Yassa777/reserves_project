"""Utilities for enforcing comparable evaluation windows."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import pandas as pd


GroupKey = Tuple[object, ...]


def _coerce_group_key(key: object) -> GroupKey:
    if isinstance(key, tuple):
        return key
    return (key,)


def compute_common_dates_by_group(
    results: pd.DataFrame,
    group_cols: Iterable[str] = ("split", "horizon"),
    model_col: str = "model",
    date_col: str = "forecast_date",
) -> Dict[GroupKey, pd.DatetimeIndex]:
    """Return date intersections across models within each group."""
    if results.empty:
        return {}

    group_cols = tuple(group_cols)
    required = [*group_cols, model_col, date_col, "actual", "forecast"]
    missing = [col for col in required if col not in results.columns]
    if missing:
        raise KeyError(f"Missing required columns for common-window computation: {missing}")

    df = results.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df[df["actual"].notna() & df["forecast"].notna()]

    intersections: Dict[GroupKey, pd.DatetimeIndex] = {}
    for group_key, subset in df.groupby(list(group_cols), sort=False):
        model_dates = []
        for _, model_subset in subset.groupby(model_col, sort=False):
            dates = pd.DatetimeIndex(model_subset[date_col].dropna().unique())
            if len(dates):
                model_dates.append(set(dates.to_pydatetime()))

        key = _coerce_group_key(group_key)
        if not model_dates:
            intersections[key] = pd.DatetimeIndex([])
            continue

        common = set.intersection(*model_dates)
        intersections[key] = pd.DatetimeIndex(sorted(common))

    return intersections


def filter_to_common_window(
    results: pd.DataFrame,
    group_cols: Iterable[str] = ("split", "horizon"),
    model_col: str = "model",
    date_col: str = "forecast_date",
) -> pd.DataFrame:
    """Filter rows to groupwise common forecast dates across models."""
    if results.empty:
        return results.copy()

    group_cols = tuple(group_cols)
    intersections = compute_common_dates_by_group(
        results=results,
        group_cols=group_cols,
        model_col=model_col,
        date_col=date_col,
    )

    df = results.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    filtered_parts = []
    for group_key, subset in df.groupby(list(group_cols), sort=False):
        key = _coerce_group_key(group_key)
        common_dates = intersections.get(key, pd.DatetimeIndex([]))
        if len(common_dates) == 0:
            continue
        filtered_parts.append(subset[subset[date_col].isin(common_dates)].copy())

    if not filtered_parts:
        return df.iloc[0:0].copy()

    filtered = pd.concat(filtered_parts, ignore_index=True)
    return filtered
