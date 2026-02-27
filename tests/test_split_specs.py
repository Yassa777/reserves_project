"""Tests for split-spec parsing in unified evaluation pipeline."""

from __future__ import annotations

import pandas as pd

from reserves_project.pipelines.run_unified_evaluations import _parse_split_specs


def test_parse_split_specs_default_baseline():
    specs = _parse_split_specs(
        split_specs_raw=None,
        default_train_end=pd.Timestamp("2019-12-01"),
        default_valid_end=pd.Timestamp("2022-12-01"),
    )
    assert len(specs) == 1
    assert specs[0].label == "baseline"
    assert specs[0].train_end == pd.Timestamp("2019-12-01")
    assert specs[0].valid_end == pd.Timestamp("2022-12-01")


def test_parse_split_specs_multiple():
    specs = _parse_split_specs(
        split_specs_raw="baseline:2019-12-01:2022-12-01,early:2018-12-01:2021-12-01",
        default_train_end=pd.Timestamp("2019-12-01"),
        default_valid_end=pd.Timestamp("2022-12-01"),
    )
    assert [s.label for s in specs] == ["baseline", "early"]
    assert specs[1].train_end == pd.Timestamp("2018-12-01")
    assert specs[1].valid_end == pd.Timestamp("2021-12-01")

