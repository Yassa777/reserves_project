"""I/O helpers for forecasting preparation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import OUTPUT_DIR


def save_dataframe(df: pd.DataFrame, filename: str, output_dir: Path | None = None) -> str:
    base_dir = output_dir if output_dir is not None else OUTPUT_DIR
    path = base_dir / filename
    df.to_csv(path, index=False)
    return str(path)


def save_metadata(
    metadata: dict[str, Any],
    filename: str = "forecast_prep_metadata.json",
    output_dir: Path | None = None,
) -> str:
    base_dir = output_dir if output_dir is not None else OUTPUT_DIR
    path = base_dir / filename
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    return str(path)
