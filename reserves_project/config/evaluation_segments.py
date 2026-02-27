"""Evaluation segment definitions for crisis-vs-tranquil reporting."""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

# Explicit windows for segment-aware evaluation.
EVALUATION_SEGMENTS = OrderedDict(
    {
        "all": {
            "label": "All",
            "ranges": [("1900-01-01", "2100-12-01")],
        },
        "crisis": {
            "label": "Crisis",
            # COVID + sovereign default stress period.
            "ranges": [("2020-03-01", "2023-12-01")],
        },
        "tranquil": {
            "label": "Tranquil",
            # Complementary windows outside crisis.
            "ranges": [("1900-01-01", "2020-02-01"), ("2024-01-01", "2100-12-01")],
        },
    }
)

DEFAULT_SEGMENT_ORDER = ["all", "crisis", "tranquil"]


def normalize_segment_keys(
    segment_keys: Iterable[str] | None,
    default: Sequence[str] | None = None,
) -> list[str]:
    """Validate and normalize requested segment keys."""
    if segment_keys is None:
        keys = list(default if default is not None else DEFAULT_SEGMENT_ORDER)
    else:
        keys = [str(k).strip() for k in segment_keys if str(k).strip()]
    if not keys:
        raise ValueError("At least one evaluation segment must be provided.")

    unknown = [k for k in keys if k not in EVALUATION_SEGMENTS]
    if unknown:
        raise KeyError(f"Unknown evaluation segment(s): {unknown}")
    return keys


def segment_date_mask(
    dates: pd.Series | pd.DatetimeIndex | Sequence,
    segment_key: str,
    segments: Mapping[str, dict] | None = None,
) -> np.ndarray:
    """Return boolean mask for dates belonging to a segment."""
    segments = segments or EVALUATION_SEGMENTS
    if segment_key not in segments:
        raise KeyError(f"Unknown evaluation segment: {segment_key}")

    date_idx = pd.DatetimeIndex(pd.to_datetime(dates))
    mask = np.zeros(len(date_idx), dtype=bool)

    for start_str, end_str in segments[segment_key]["ranges"]:
        start = pd.Timestamp(start_str)
        end = pd.Timestamp(end_str)
        mask |= (date_idx >= start) & (date_idx <= end)
    return mask


__all__ = [
    "EVALUATION_SEGMENTS",
    "DEFAULT_SEGMENT_ORDER",
    "normalize_segment_keys",
    "segment_date_mask",
]
