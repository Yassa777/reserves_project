"""Load prepared forecasting datasets."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.api import VAR

from forecasting_prep.config import TRAIN_END

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


def estimate_johansen_rank(
    levels_df: pd.DataFrame,
    train_end: pd.Timestamp | None = None,
    k_ar_diff: int = 2,
) -> int:
    """Estimate Johansen cointegration rank using training data only."""
    if train_end is None:
        train_end = TRAIN_END
    if "date" in levels_df.columns:
        df = levels_df.set_index("date")
    else:
        df = levels_df.copy()

    cols = [c for c in df.columns if c != "split"]
    train = df.loc[df.index <= train_end, cols].dropna()
    if len(train) < max(30, k_ar_diff + 5):
        return 1

    try:
        joh = coint_johansen(train, det_order=0, k_ar_diff=max(1, k_ar_diff))
        trace_stats = joh.lr1
        crit_vals = joh.cvt[:, 1]  # 5% critical values
        rank = int(sum(stat > cv for stat, cv in zip(trace_stats, crit_vals)))
        max_rank = max(1, len(cols) - 1)
        return max(1, min(rank, max_rank))
    except Exception:
        return 1


def estimate_k_ar_diff(
    levels_df: pd.DataFrame,
    train_end: pd.Timestamp | None = None,
    max_lags: int = 6,
) -> int:
    """Estimate VAR lag order (k_ar_diff) using training data only."""
    if train_end is None:
        train_end = TRAIN_END
    if "date" in levels_df.columns:
        df = levels_df.set_index("date")
    else:
        df = levels_df.copy()

    cols = [c for c in df.columns if c != "split"]
    train = df.loc[df.index <= train_end, cols].dropna()
    if len(train) < max(30, max_lags + 1):
        return 2
    try:
        sel = VAR(train).select_order(maxlags=max_lags)
        chosen = None
        if hasattr(sel, "selected_orders") and sel.selected_orders:
            chosen = sel.selected_orders.get("aic") or sel.selected_orders.get("bic")
        if chosen is None and hasattr(sel, "aic"):
            chosen = sel.aic
        if chosen is None:
            return 2
        chosen = int(chosen)
        return max(1, chosen - 1)
    except Exception:
        return 2
