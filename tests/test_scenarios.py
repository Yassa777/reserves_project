"""Tests for scenario definitions and framework."""

import pytest

from reserves_project.scenarios.definitions import (
    Scenario,
    POLICY_SCENARIOS,
    DEFAULT_SHOCK_VARS,
    SUPPORTED_PROFILES,
)


class TestScenarioClass:
    """Tests for Scenario dataclass."""

    def test_basic_scenario_creation(self):
        """Basic scenario creation should work."""
        scenario = Scenario(
            name="Test Scenario",
            description="A test scenario",
            horizon_months=12,
            shocks={"exports_usd_m": 0.9},
        )

        assert scenario.name == "Test Scenario"
        assert scenario.description == "A test scenario"
        assert scenario.horizon_months == 12
        assert scenario.shocks == {"exports_usd_m": 0.9}

    def test_default_values(self):
        """Default values should be set correctly."""
        scenario = Scenario(name="Minimal", description="Minimal scenario")

        assert scenario.horizon_months == 12
        assert scenario.shocks == {}
        assert scenario.profile == "ramp"

    def test_normalized_shocks_fills_defaults(self):
        """normalized_shocks should fill missing variables with 1.0."""
        scenario = Scenario(
            name="Partial",
            description="Partial shocks",
            shocks={"exports_usd_m": 0.85}
        )

        normalized = scenario.normalized_shocks()

        # Specified shock should be preserved
        assert normalized["exports_usd_m"] == 0.85

        # Other default vars should be 1.0
        for var in DEFAULT_SHOCK_VARS:
            if var != "exports_usd_m":
                assert normalized[var] == 1.0

    def test_normalized_shocks_custom_vars(self):
        """normalized_shocks with custom variable list."""
        scenario = Scenario(
            name="Custom",
            description="Custom vars",
            shocks={"var_a": 0.9}
        )

        normalized = scenario.normalized_shocks(variables=["var_a", "var_b", "var_c"])

        assert normalized == {"var_a": 0.9, "var_b": 1.0, "var_c": 1.0}

    def test_normalized_profile(self):
        """normalized_profile should validate profile type."""
        scenario_valid = Scenario(name="Valid", description="", profile="step")
        scenario_invalid = Scenario(name="Invalid", description="", profile="unknown")

        assert scenario_valid.normalized_profile() == "step"
        assert scenario_invalid.normalized_profile() == "ramp"  # Falls back to default


class TestPolicyScenarios:
    """Tests for predefined policy scenarios."""

    def test_all_scenarios_exist(self):
        """All expected policy scenarios should be defined."""
        expected_scenarios = [
            "baseline",
            "lkr_depreciation_10pct",
            "lkr_depreciation_20pct",
            "export_shock_negative",
            "remittance_shock",
            "tourism_recovery",
            "oil_price_shock",
            "imf_tranche_delay",
            "combined_adverse",
            "combined_upside",
        ]

        for key in expected_scenarios:
            assert key in POLICY_SCENARIOS, f"Missing scenario: {key}"

    def test_baseline_has_no_shocks(self):
        """Baseline scenario should have empty shocks."""
        baseline = POLICY_SCENARIOS["baseline"]

        assert baseline.shocks == {}
        assert baseline.name == "Baseline"

    def test_combined_adverse_has_multiple_shocks(self):
        """Combined adverse should shock multiple variables."""
        adverse = POLICY_SCENARIOS["combined_adverse"]

        assert len(adverse.shocks) >= 3
        # All shocks should be unfavorable
        for var, mult in adverse.shocks.items():
            if "import" in var:
                assert mult > 1.0  # Higher imports is bad
            elif "usd_lkr" in var:
                assert mult > 1.0  # Depreciation is bad
            else:
                assert mult < 1.0  # Lower exports/remittances/tourism is bad

    def test_combined_upside_has_favorable_shocks(self):
        """Combined upside should have favorable shocks."""
        upside = POLICY_SCENARIOS["combined_upside"]

        assert len(upside.shocks) >= 3
        # All shocks should be favorable
        for var, mult in upside.shocks.items():
            if "import" in var:
                assert mult < 1.0  # Lower imports is good
            elif "usd_lkr" in var:
                assert mult < 1.0  # Appreciation is good
            else:
                assert mult > 1.0  # Higher exports/remittances/tourism is good

    def test_all_scenarios_have_valid_profiles(self):
        """All scenarios should have valid profiles."""
        for key, scenario in POLICY_SCENARIOS.items():
            profile = scenario.normalized_profile()
            assert profile in SUPPORTED_PROFILES, f"{key} has invalid profile: {profile}"

    def test_depreciation_scenarios_shock_fx(self):
        """Depreciation scenarios should shock usd_lkr."""
        dep_10 = POLICY_SCENARIOS["lkr_depreciation_10pct"]
        dep_20 = POLICY_SCENARIOS["lkr_depreciation_20pct"]

        assert "usd_lkr" in dep_10.shocks
        assert dep_10.shocks["usd_lkr"] == 1.10

        assert "usd_lkr" in dep_20.shocks
        assert dep_20.shocks["usd_lkr"] == 1.20

    def test_all_scenarios_have_descriptions(self):
        """All scenarios should have non-empty descriptions."""
        for key, scenario in POLICY_SCENARIOS.items():
            assert scenario.description, f"{key} missing description"
            assert len(scenario.description) > 10


class TestShockMultipliers:
    """Tests for shock multiplier conventions."""

    def test_multiplier_1_is_baseline(self):
        """Multiplier of 1.0 should mean no change."""
        scenario = Scenario(
            name="No Change",
            description="",
            shocks={"exports_usd_m": 1.0}
        )

        normalized = scenario.normalized_shocks(variables=["exports_usd_m"])
        assert normalized["exports_usd_m"] == 1.0

    def test_multiplier_below_1_is_decline(self):
        """Multiplier < 1.0 should mean decline."""
        scenario = Scenario(
            name="Decline",
            description="",
            shocks={"exports_usd_m": 0.85}  # 15% decline
        )

        assert scenario.shocks["exports_usd_m"] < 1.0

    def test_multiplier_above_1_is_increase(self):
        """Multiplier > 1.0 should mean increase."""
        scenario = Scenario(
            name="Increase",
            description="",
            shocks={"tourism_usd_m": 1.25}  # 25% increase
        )

        assert scenario.shocks["tourism_usd_m"] > 1.0
