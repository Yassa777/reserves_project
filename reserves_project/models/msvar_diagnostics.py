"""Serialization helpers for MS-VAR diagnostics outputs."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from reserves_project.models.ms_switching_var import MarkovSwitchingVAR


def smoothed_probabilities_df(
    model: MarkovSwitchingVAR,
    dates: pd.DatetimeIndex,
    train_end: pd.Timestamp | None = None,
    valid_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build tidy smoothed regime probability DataFrame."""
    if model.smoothed_probs_ is None:
        raise RuntimeError("Model has no smoothed probabilities; run fit() first.")
    if len(dates) != model.smoothed_probs_.shape[0]:
        raise ValueError(
            f"Date length mismatch: {len(dates)} dates vs {model.smoothed_probs_.shape[0]} probability rows."
        )

    probs = model.smoothed_probs_
    out = pd.DataFrame(index=pd.DatetimeIndex(dates))
    for r in range(probs.shape[1]):
        out[f"regime_{r}_prob"] = probs[:, r]
    out["most_likely_regime"] = probs.argmax(axis=1).astype(int)
    out["max_probability"] = probs.max(axis=1)

    if train_end is not None and valid_end is not None:
        train_end = pd.Timestamp(train_end)
        valid_end = pd.Timestamp(valid_end)
        out["split"] = "test"
        out.loc[out.index <= valid_end, "split"] = "validation"
        out.loc[out.index <= train_end, "split"] = "train"

    out = out.reset_index().rename(columns={"index": "date"})
    return out


def transition_matrix_df(model: MarkovSwitchingVAR) -> pd.DataFrame:
    """Transition matrix as tidy long-form table."""
    if model.transition_ is None:
        raise RuntimeError("Model has no transition matrix; run fit() first.")
    rows = []
    for i in range(model.transition_.shape[0]):
        for j in range(model.transition_.shape[1]):
            rows.append(
                {
                    "from_regime": int(i),
                    "to_regime": int(j),
                    "transition_probability": float(model.transition_[i, j]),
                }
            )
    return pd.DataFrame(rows)


def expected_durations_df(model: MarkovSwitchingVAR) -> pd.DataFrame:
    """Expected duration table by regime."""
    if model.transition_ is None:
        raise RuntimeError("Model has no transition matrix; run fit() first.")
    durations = model.expected_durations()
    rows = []
    for r, duration in enumerate(durations):
        rows.append(
            {
                "regime": int(r),
                "persistence_probability": float(model.transition_[r, r]),
                "expected_duration_months": float(duration),
            }
        )
    return pd.DataFrame(rows)


def classification_certainty_df(model: MarkovSwitchingVAR) -> pd.DataFrame:
    """Flatten classification certainty summary for CSV export."""
    cert = model.classification_certainty()
    rows = []
    for key, value in cert.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                rows.append({"metric": f"{key}.{subkey}", "value": float(subvalue)})
        else:
            rows.append({"metric": key, "value": float(value)})
    return pd.DataFrame(rows)


def fit_diagnostics_dict(model: MarkovSwitchingVAR) -> Dict[str, Any]:
    """Serializable model fit diagnostics."""
    return {
        "n_regimes": int(model.n_regimes),
        "ar_order": int(model.ar_order),
        "max_iter": int(model.max_iter),
        "tol": float(model.tol),
        "converged": bool(model.converged_) if model.converged_ is not None else None,
        "n_iter": int(model.n_iter_),
        "loglik": float(model.loglik_) if model.loglik_ is not None else None,
        "loglik_path": [float(v) for v in (model.loglik_path_ or [])],
        "init_states_summary": model.init_states_summary_,
    }
