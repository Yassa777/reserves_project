"""Scenario analysis module for policy-relevant reserve forecasting."""

from .scenario_framework import (
    Scenario,
    ScenarioEngine,
    MSVARScenarioAnalyzer,
    BoPScenarioAnalyzer,
    POLICY_SCENARIOS,
    REGIME_SCENARIOS,
    create_scenario_fan_chart,
    run_scenario_analysis,
)

__all__ = [
    "Scenario",
    "ScenarioEngine",
    "MSVARScenarioAnalyzer",
    "BoPScenarioAnalyzer",
    "POLICY_SCENARIOS",
    "REGIME_SCENARIOS",
    "create_scenario_fan_chart",
    "run_scenario_analysis",
]
