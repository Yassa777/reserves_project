"""Compatibility shim for scenario utilities."""

from reserves_project.scenarios.definitions import Scenario, POLICY_SCENARIOS
from reserves_project.scenarios.paths import build_baseline_exog_path, build_scenario_exog_path, diff_from_last

__all__ = [
    "Scenario",
    "POLICY_SCENARIOS",
    "build_baseline_exog_path",
    "build_scenario_exog_path",
    "diff_from_last",
]
