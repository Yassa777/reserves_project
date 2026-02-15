"""Load prepared forecasting datasets."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DIAG_DIR = PROJECT_ROOT / "data" / "diagnostics"


def _dir_suffix(varset: str | None) -> str:
    if not varset or varset == "baseline":
        return ""
    return f"_{varset}"


def get_prep_dir(varset: str | None = None) -> Path:
    return PROJECT_ROOT / "data" / f"forecast_prep{_dir_suffix(varset)}"


def get_results_dir(varset: str | None = None) -> Path:
    path = PROJECT_ROOT / "data" / f"forecast_results{_dir_suffix(varset)}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_prep_csv(filename: str, varset: str | None = None) -> pd.DataFrame:
    path = get_prep_dir(varset) / filename
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def load_prep_metadata(varset: str | None = None) -> dict:
    path = get_prep_dir(varset) / "forecast_prep_metadata.json"
    with open(path, "r") as f:
        return json.load(f)


def load_johansen_rank() -> int:
    path = DIAG_DIR / "johansen_summary.csv"
    if not path.exists():
        return 1
    df = pd.read_csv(path)
    if "rank_trace_5pct" in df.columns and not df.empty:
        return max(1, int(df["rank_trace_5pct"].iloc[0]))
    return 1
