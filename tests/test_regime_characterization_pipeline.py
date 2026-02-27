"""Tests for regime characterization pipeline outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from reserves_project.pipelines import run_regime_characterization


def _synthetic_levels(n: int = 72) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    dates = pd.date_range("2018-01-01", periods=n, freq="MS")

    eps1 = rng.normal(0, 20, n)
    eps2 = rng.normal(0, 10, n)
    eps3 = rng.normal(0, 15, n)

    reserves = 6000 + np.cumsum(30 + eps1)
    trade = -300 + np.cumsum(eps2)
    fx = 150 + np.cumsum(0.2 + eps3 * 0.02)

    df = pd.DataFrame(
        {
            "gross_reserves_usd_m": reserves,
            "trade_balance_usd_m": trade,
            "usd_lkr": fx,
        },
        index=dates,
    )
    return df


def test_regime_characterization_pipeline_outputs(tmp_path, monkeypatch):
    levels = _synthetic_levels()
    out_dir = tmp_path / "regime_outputs"

    monkeypatch.setattr(run_regime_characterization, "load_varset_levels", lambda varset: levels.copy())
    monkeypatch.setattr(
        "sys.argv",
        [
            "reserves-regime-characterization",
            "--varset=parsimonious",
            "--n-regimes=2",
            "--ar-order=1",
            "--fit-split=full",
            f"--output-dir={out_dir}",
        ],
    )

    run_regime_characterization.main()

    expected_files = [
        out_dir / "regime_smoothed_probabilities.csv",
        out_dir / "regime_transition_matrix.csv",
        out_dir / "regime_durations.csv",
        out_dir / "regime_classification_certainty.csv",
        out_dir / "regime_fit_diagnostics.json",
        out_dir / "run_manifest.json",
    ]
    for path in expected_files:
        assert path.exists(), f"Missing expected output file: {path}"

    probs = pd.read_csv(out_dir / "regime_smoothed_probabilities.csv")
    regime_prob_cols = [c for c in probs.columns if c.startswith("regime_") and c.endswith("_prob")]
    assert len(regime_prob_cols) == 2
    row_sums = probs[regime_prob_cols].sum(axis=1)
    assert np.allclose(row_sums.values, 1.0, atol=1e-5)

    transition = pd.read_csv(out_dir / "regime_transition_matrix.csv")
    assert set(transition.columns) == {"from_regime", "to_regime", "transition_probability"}
    row_totals = transition.groupby("from_regime")["transition_probability"].sum()
    assert np.allclose(row_totals.values, 1.0, atol=1e-6)

    durations = pd.read_csv(out_dir / "regime_durations.csv")
    assert (durations["expected_duration_months"] > 0).all()

    with open(out_dir / "regime_fit_diagnostics.json", "r") as f:
        diagnostics = json.load(f)
    assert "loglik_path" in diagnostics
    assert len(diagnostics["loglik_path"]) >= 1
    assert diagnostics["n_regimes"] == 2
