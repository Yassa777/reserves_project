"""Tests for regime-conditional MS-VAR IRF analysis."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from reserves_project.eval.msvar_irf import generalized_irf, girf_to_long_df, summarize_regime_comparison
from reserves_project.models.ms_switching_var import MarkovSwitchingVAR
from reserves_project.pipelines import run_msvar_irf_analysis


def _synthetic_diff_data(n: int = 180, k: int = 3, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 1.0, size=(n, k))
    # Introduce mild autocorrelation for a non-trivial VAR structure.
    for t in range(1, n):
        data[t] += 0.35 * data[t - 1]
    return data


def _fit_model(seed: int = 123) -> MarkovSwitchingVAR:
    y = _synthetic_diff_data(seed=seed)
    # Deterministic init states to keep fit deterministic in tests.
    init_states = (y[:, 0] > np.median(y[:, 0])).astype(int)
    model = MarkovSwitchingVAR(n_regimes=2, ar_order=1, max_iter=40, tol=1e-6)
    model.fit(y, init_states=init_states)
    return model


def test_generalized_irf_shape_and_finite():
    model = _fit_model()
    irf = generalized_irf(model, regime=0, max_horizon=12, cumulative=False)

    assert irf.shape == (13, 3, 3)
    assert np.isfinite(irf).all()

    df = girf_to_long_df(irf, var_names=["a", "b", "c"], regime=0, cumulative=False)
    assert len(df) == 13 * 3 * 3
    assert set(df.columns) == {"regime", "horizon", "response_variable", "shock_variable", "response", "response_type"}


def test_generalized_irf_deterministic_given_same_fit_setup():
    m1 = _fit_model(seed=123)
    m2 = _fit_model(seed=123)

    irf1 = generalized_irf(m1, regime=1, max_horizon=10, cumulative=True)
    irf2 = generalized_irf(m2, regime=1, max_horizon=10, cumulative=True)
    assert np.allclose(irf1, irf2, atol=1e-10)


def test_regime_comparison_summary_columns():
    model = _fit_model()
    var_names = ["a", "b", "c"]
    r0 = girf_to_long_df(generalized_irf(model, regime=0, max_horizon=8), var_names, regime=0)
    r1 = girf_to_long_df(generalized_irf(model, regime=1, max_horizon=8), var_names, regime=1)

    summary = summarize_regime_comparison(r0, r1)
    expected_cols = {
        "shock_variable",
        "response_variable",
        "peak_abs_response_regime0",
        "peak_abs_response_regime1",
        "peak_abs_delta_regime1_minus_regime0",
        "half_life_horizon_regime0",
        "half_life_horizon_regime1",
        "half_life_delta_regime1_minus_regime0",
    }
    assert set(summary.columns) == expected_cols
    assert len(summary) == len(var_names) * len(var_names)


def test_msvar_irf_pipeline_outputs(tmp_path, monkeypatch):
    rng = np.random.default_rng(99)
    dates = pd.date_range("2017-01-01", periods=84, freq="MS")
    levels = pd.DataFrame(
        {
            "gross_reserves_usd_m": 6000 + np.cumsum(30 + rng.normal(0, 20, len(dates))),
            "trade_balance_usd_m": -300 + np.cumsum(rng.normal(0, 10, len(dates))),
            "usd_lkr": 150 + np.cumsum(0.1 + rng.normal(0, 1, len(dates))),
        },
        index=dates,
    )
    out_dir = tmp_path / "msvar_irf_outputs"

    monkeypatch.setattr(run_msvar_irf_analysis, "load_varset_levels", lambda varset: levels.copy())
    monkeypatch.setattr(
        "sys.argv",
        [
            "reserves-msvar-irf",
            "--varset=parsimonious",
            "--n-regimes=2",
            "--ar-order=1",
            "--max-horizon=8",
            "--fit-split=full",
            f"--output-dir={out_dir}",
        ],
    )

    run_msvar_irf_analysis.main()

    expected = [
        out_dir / "msvar_irf_regime0.csv",
        out_dir / "msvar_irf_regime1.csv",
        out_dir / "msvar_irf_regime_comparison.csv",
        out_dir / "msvar_irf_diagnostics.json",
        out_dir / "run_manifest.json",
    ]
    for path in expected:
        assert path.exists(), f"Missing output file: {path}"

    with open(out_dir / "msvar_irf_diagnostics.json", "r") as f:
        diagnostics = json.load(f)
    assert diagnostics["max_horizon"] == 8
    assert diagnostics["n_regimes"] == 2
