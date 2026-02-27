"""Helpers for detecting temporal leakage in rolling forecasts."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def assert_no_future_in_history(
    history_index: Iterable,
    origin_date: pd.Timestamp,
    context: str = "",
) -> None:
    """Raise if history contains timestamps after the forecast origin."""
    idx = pd.DatetimeIndex(pd.to_datetime(list(history_index)))
    if len(idx) == 0:
        return
    max_date = idx.max()
    origin_date = pd.Timestamp(origin_date)
    if max_date > origin_date:
        label = f" ({context})" if context else ""
        raise ValueError(
            f"Temporal leakage detected{label}: history_end={max_date.date()} > "
            f"origin={origin_date.date()}."
        )


def history_debug_info(history_index: Iterable, origin_date: pd.Timestamp) -> dict:
    """Return trace columns for diagnostics/logging."""
    idx = pd.DatetimeIndex(pd.to_datetime(list(history_index)))
    history_end = idx.max() if len(idx) else pd.NaT
    return {
        "origin_date": pd.Timestamp(origin_date),
        "history_end_date": history_end,
    }
