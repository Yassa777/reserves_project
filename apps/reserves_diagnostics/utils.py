"""Utility functions for data analysis."""

import pandas as pd
import numpy as np
from scipy import stats

from .config import CRISIS_START, CRISIS_END


def compute_missing_analysis(df, date_col, value_cols):
    """Compute missing value statistics."""
    if df is None or value_cols is None:
        return None

    results = []
    for col in value_cols:
        if col not in df.columns:
            continue

        total = len(df)
        missing = df[col].isna().sum()
        missing_pct = (missing / total) * 100 if total > 0 else 0

        # Check for gaps in crisis period
        if date_col in df.columns:
            crisis_df = df[(df[date_col] >= CRISIS_START) & (df[date_col] <= CRISIS_END)]
            crisis_missing = crisis_df[col].isna().sum() if len(crisis_df) > 0 else 0
            crisis_total = len(crisis_df)
        else:
            crisis_missing = 0
            crisis_total = 0

        results.append({
            "Column": col,
            "Total Rows": total,
            "Missing": missing,
            "Missing %": round(missing_pct, 1),
            "Crisis Period Missing": crisis_missing,
            "Crisis Period Total": crisis_total
        })

    return pd.DataFrame(results)


def compute_outlier_analysis(df, value_cols, method="iqr", threshold=1.5):
    """Detect outliers using IQR or Z-score method."""
    if df is None or value_cols is None:
        return None

    results = []
    for col in value_cols:
        if col not in df.columns:
            continue

        series = df[col].dropna()
        if len(series) < 10:
            continue

        if method == "iqr":
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            outliers = series[(series < lower) | (series > upper)]
        else:  # z-score
            z_scores = np.abs(stats.zscore(series))
            outliers = series[z_scores > threshold]

        results.append({
            "Column": col,
            "N": len(series),
            "Outliers": len(outliers),
            "Outlier %": round((len(outliers) / len(series)) * 100, 2),
            "Min": round(series.min(), 2),
            "Max": round(series.max(), 2),
            "Mean": round(series.mean(), 2),
            "Std": round(series.std(), 2)
        })

    return pd.DataFrame(results)


def compute_distribution_stats(df, value_cols):
    """Compute distribution statistics including normality tests."""
    if df is None or value_cols is None:
        return None

    results = []
    for col in value_cols:
        if col not in df.columns:
            continue

        series = df[col].dropna()
        if len(series) < 20:
            continue

        # Normality test (Jarque-Bera)
        try:
            jb_stat, jb_pval = stats.jarque_bera(series)
        except:
            jb_stat, jb_pval = np.nan, np.nan

        results.append({
            "Column": col,
            "N": len(series),
            "Mean": round(series.mean(), 2),
            "Median": round(series.median(), 2),
            "Std": round(series.std(), 2),
            "Skewness": round(series.skew(), 3),
            "Kurtosis": round(series.kurtosis(), 3),
            "JB Stat": round(jb_stat, 2) if not np.isnan(jb_stat) else "N/A",
            "JB p-value": round(jb_pval, 4) if not np.isnan(jb_pval) else "N/A",
            "Normal?": "Yes" if jb_pval > 0.05 else "No" if not np.isnan(jb_pval) else "N/A"
        })

    return pd.DataFrame(results)


def get_coverage_info(df, date_col):
    """Get date coverage information."""
    if df is None or date_col not in df.columns:
        return None

    df[date_col] = pd.to_datetime(df[date_col])

    return {
        "start": df[date_col].min(),
        "end": df[date_col].max(),
        "records": len(df),
        "date_range_days": (df[date_col].max() - df[date_col].min()).days
    }
