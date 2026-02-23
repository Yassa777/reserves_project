"""Phase 9: Bai-Perron style multiple break diagnostics (BIC-selected)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import MIN_OBS_BREAKS, PHASE9_BREAK_VARS


def _segment_rss_factory(values: np.ndarray):
    n = len(values)
    x = np.arange(n, dtype=float)

    c_x = np.concatenate([[0.0], np.cumsum(x)])
    c_xx = np.concatenate([[0.0], np.cumsum(x * x)])
    c_y = np.concatenate([[0.0], np.cumsum(values)])
    c_yy = np.concatenate([[0.0], np.cumsum(values * values)])
    c_xy = np.concatenate([[0.0], np.cumsum(x * values)])

    def seg_rss(i: int, j: int) -> float:
        # Segment is [i, j), j exclusive.
        nseg = j - i
        if nseg < 2:
            return np.inf

        sx = c_x[j] - c_x[i]
        sxx = c_xx[j] - c_xx[i]
        sy = c_y[j] - c_y[i]
        syy = c_yy[j] - c_yy[i]
        sxy = c_xy[j] - c_xy[i]

        xtx = np.array([[nseg, sx], [sx, sxx]], dtype=float)
        det = np.linalg.det(xtx)
        if np.isclose(det, 0.0):
            return np.inf

        xty = np.array([sy, sxy], dtype=float)
        beta = np.linalg.solve(xtx, xty)
        rss = float(syy - beta @ xty)
        return max(rss, 1e-12)

    return seg_rss


def _bai_perron_select(series: pd.Series, min_seg: int = 24, max_breaks: int = 5):
    y = series.dropna().values.astype(float)
    idx = series.dropna().index
    n = len(y)

    if n < max(MIN_OBS_BREAKS, 2 * min_seg):
        return {
            "error": "Insufficient observations for multiple-break detection",
            "effective_nobs": int(n),
        }

    max_segments = min(max_breaks + 1, n // min_seg)
    seg_rss = _segment_rss_factory(y)

    # Precompute feasible RSS for all segments.
    rss = np.full((n + 1, n + 1), np.inf)
    for i in range(n):
        j_start = i + min_seg
        for j in range(j_start, n + 1):
            rss[i, j] = seg_rss(i, j)

    dp = np.full((max_segments + 1, n + 1), np.inf)
    prev = np.full((max_segments + 1, n + 1), -1, dtype=int)
    dp[0, 0] = 0.0

    for s in range(1, max_segments + 1):
        j_min = s * min_seg
        for j in range(j_min, n + 1):
            i_min = (s - 1) * min_seg
            i_max = j - min_seg
            best_val = np.inf
            best_i = -1
            for i in range(i_min, i_max + 1):
                cand = dp[s - 1, i] + rss[i, j]
                if cand < best_val:
                    best_val = cand
                    best_i = i
            dp[s, j] = best_val
            prev[s, j] = best_i

    model_rows = []
    for s in range(1, max_segments + 1):
        total_rss = dp[s, n]
        if not np.isfinite(total_rss):
            continue
        k_params = 2 * s + (s - 1)  # intercept + trend per segment + break locations
        bic = n * math.log(total_rss / n) + k_params * math.log(n)
        model_rows.append(
            {
                "segments": int(s),
                "break_count": int(s - 1),
                "rss": float(total_rss),
                "bic": float(bic),
            }
        )

    if not model_rows:
        return {
            "error": "No feasible segmentation found",
            "effective_nobs": int(n),
        }

    best_model = min(model_rows, key=lambda r: r["bic"])
    best_s = int(best_model["segments"])

    # Backtrack break locations.
    breaks = []
    j = n
    s = best_s
    while s > 1:
        i = int(prev[s, j])
        if i <= 0:
            break
        breaks.append(i)
        j = i
        s -= 1
    breaks = sorted(breaks)

    no_break = min(model_rows, key=lambda r: r["break_count"])
    rss_reduction = 0.0
    if no_break["rss"] > 0:
        rss_reduction = (no_break["rss"] - best_model["rss"]) / no_break["rss"]

    break_dates = [str(idx[b].date()) for b in breaks if b < len(idx)]

    return {
        "effective_nobs": int(n),
        "optimal_break_count": int(best_model["break_count"]),
        "break_indices": breaks,
        "break_dates": break_dates,
        "bic_no_break": round(float(no_break["bic"]), 4),
        "bic_optimal": round(float(best_model["bic"]), 4),
        "rss_reduction_pct": round(float(rss_reduction * 100), 2),
        "multiple_breaks_detected": bool(best_model["break_count"] >= 2),
        "model_selection": [
            {
                "segments": int(r["segments"]),
                "break_count": int(r["break_count"]),
                "rss": round(float(r["rss"]), 4),
                "bic": round(float(r["bic"]), 4),
            }
            for r in model_rows
        ],
    }


def run_phase9(df: pd.DataFrame, usable_vars: list[str], verbose=True):
    variables = [v for v in PHASE9_BREAK_VARS if v in usable_vars]

    results = []
    for var in variables:
        out = _bai_perron_select(df[var])
        out["variable"] = var
        results.append(out)

        if verbose:
            print(f"\n  Bai-Perron {var}:")
            print(
                f"    breaks={out.get('optimal_break_count', 'N/A')}, "
                f"dates={out.get('break_dates', [])}, "
                f"bic_opt={out.get('bic_optimal', 'N/A')}"
            )

    return {"bai_perron": results}
