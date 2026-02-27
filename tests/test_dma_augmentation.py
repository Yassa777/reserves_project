"""Tests for DMA/DMS augmentation over unified forecast outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from reserves_project.eval.dma import augment_with_dma_dms


def _synthetic_results() -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=8, freq="MS")
    rows = []
    actual = np.linspace(100.0, 107.0, len(dates))
    model_a = actual + np.array([0.5, -0.2, 0.1, -0.3, 0.2, -0.1, 0.4, -0.2])
    model_b = actual + np.array([1.0, -0.8, 0.9, -0.7, 1.1, -0.9, 0.8, -0.6])

    for i, dt in enumerate(dates):
        for model_name, forecast in [("ModelA", model_a[i]), ("ModelB", model_b[i])]:
            rows.append(
                {
                    "model": model_name,
                    "forecast_origin": dt - pd.DateOffset(months=1),
                    "forecast_date": dt,
                    "horizon": 1,
                    "split": "test",
                    "actual": actual[i],
                    "forecast": forecast,
                    "std": np.nan,
                    "lower_80": np.nan,
                    "upper_80": np.nan,
                    "lower_95": np.nan,
                    "upper_95": np.nan,
                    "crps": np.nan,
                    "log_score": np.nan,
                }
            )
    return pd.DataFrame(rows)


def test_augment_with_dma_dms_appends_models_and_weights():
    base = _synthetic_results()

    augmented, weights = augment_with_dma_dms(
        base,
        alpha=0.99,
        warmup_periods=2,
        min_model_obs=3,
    )

    assert set(["DMA", "DMS"]).issubset(set(augmented["model"].unique()))

    dma_rows = augmented[augmented["model"] == "DMA"].sort_values("forecast_date")
    dms_rows = augmented[augmented["model"] == "DMS"].sort_values("forecast_date")
    assert len(dma_rows) == len(dms_rows) == 8
    assert dma_rows["forecast"].notna().all()
    assert dms_rows["forecast"].notna().all()

    assert not weights.empty
    assert set(["date", "horizon", "ModelA", "ModelB"]).issubset(set(weights.columns))
    assert (weights[["ModelA", "ModelB"]].sum(axis=1) > 0.999).all()
    assert (weights[["ModelA", "ModelB"]].sum(axis=1) < 1.001).all()

