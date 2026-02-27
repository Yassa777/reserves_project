"""Configuration and constants for the reserves project."""

from .paths import PROJECT_ROOT, DATA_DIR
from .varsets import (
    TARGET_VAR,
    TRAIN_END,
    VALID_END,
    VARSET_ORDER,
    VARSET_PARSIMONIOUS,
    VARSET_BOP,
    VARSET_MONETARY,
    VARSET_PCA,
    VARSET_FULL,
)
from .evaluation_segments import (
    EVALUATION_SEGMENTS,
    DEFAULT_SEGMENT_ORDER,
    normalize_segment_keys,
    segment_date_mask,
)

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "TARGET_VAR",
    "TRAIN_END",
    "VALID_END",
    "VARSET_ORDER",
    "VARSET_PARSIMONIOUS",
    "VARSET_BOP",
    "VARSET_MONETARY",
    "VARSET_PCA",
    "VARSET_FULL",
    "EVALUATION_SEGMENTS",
    "DEFAULT_SEGMENT_ORDER",
    "normalize_segment_keys",
    "segment_date_mask",
]
