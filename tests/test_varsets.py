"""Tests for variable set configuration."""

import pandas as pd
import pytest

from reserves_project.config.varsets import (
    TARGET_VAR,
    TRAIN_END,
    VALID_END,
    VARSET_PARSIMONIOUS,
    VARSET_BOP,
    VARSET_MONETARY,
    VARSET_PCA,
    VARSET_FULL,
    VARIABLE_SETS,
    VARSET_ORDER,
    get_varset,
)


class TestCoreConfiguration:
    """Tests for core configuration values."""

    def test_target_variable_defined(self):
        """Target variable should be defined."""
        assert TARGET_VAR == "gross_reserves_usd_m"

    def test_train_end_is_timestamp(self):
        """Train end should be a timestamp."""
        assert isinstance(TRAIN_END, pd.Timestamp)
        assert TRAIN_END.year == 2019
        assert TRAIN_END.month == 12

    def test_valid_end_is_timestamp(self):
        """Validation end should be a timestamp."""
        assert isinstance(VALID_END, pd.Timestamp)
        assert VALID_END.year == 2022
        assert VALID_END.month == 12

    def test_valid_end_after_train_end(self):
        """Validation period should come after training."""
        assert VALID_END > TRAIN_END


class TestVariableSetStructure:
    """Tests for variable set structure."""

    @pytest.mark.parametrize("varset", [
        VARSET_PARSIMONIOUS,
        VARSET_BOP,
        VARSET_MONETARY,
        VARSET_PCA,
        VARSET_FULL,
    ])
    def test_varset_has_required_fields(self, varset):
        """Each varset should have required fields."""
        required_fields = ["name", "target", "arima_exog", "vecm_system"]

        for field in required_fields:
            assert field in varset, f"Missing field: {field} in {varset.get('name', 'unknown')}"

    @pytest.mark.parametrize("varset", [
        VARSET_PARSIMONIOUS,
        VARSET_BOP,
        VARSET_MONETARY,
        VARSET_PCA,
        VARSET_FULL,
    ])
    def test_target_in_vecm_system(self, varset):
        """Target variable should be in VECM system."""
        assert varset["target"] in varset["vecm_system"]

    @pytest.mark.parametrize("varset", [
        VARSET_PARSIMONIOUS,
        VARSET_BOP,
        VARSET_MONETARY,
        VARSET_PCA,
        VARSET_FULL,
    ])
    def test_arima_exog_not_contain_target(self, varset):
        """ARIMA exog should not contain target variable."""
        assert varset["target"] not in varset["arima_exog"]


class TestParsimoniousVarset:
    """Tests specific to parsimonious variable set."""

    def test_parsimonious_is_minimal(self):
        """Parsimonious should have minimal variables."""
        assert len(VARSET_PARSIMONIOUS["arima_exog"]) <= 3
        assert len(VARSET_PARSIMONIOUS["vecm_system"]) <= 4

    def test_parsimonious_contains_fx(self):
        """Parsimonious should include exchange rate."""
        assert "usd_lkr" in VARSET_PARSIMONIOUS["arima_exog"]

    def test_parsimonious_contains_trade(self):
        """Parsimonious should include trade balance."""
        assert "trade_balance_usd_m" in VARSET_PARSIMONIOUS["arima_exog"]


class TestBopVarset:
    """Tests specific to BoP variable set."""

    def test_bop_contains_major_flows(self):
        """BoP should contain major current account flows."""
        expected_vars = ["exports_usd_m", "imports_usd_m", "remittances_usd_m"]

        for var in expected_vars:
            assert var in VARSET_BOP["arima_exog"], f"Missing: {var}"

    def test_bop_excludes_fx(self):
        """BoP should typically exclude exchange rate (to avoid identity)."""
        # This is a design choice - BoP flows shouldn't include FX
        # to avoid BoP identity endogeneity
        assert "usd_lkr" not in VARSET_BOP["arima_exog"]


class TestFullVarset:
    """Tests specific to full/kitchen-sink variable set."""

    def test_full_is_largest(self):
        """Full varset should have the most variables."""
        varsets = [VARSET_PARSIMONIOUS, VARSET_BOP, VARSET_MONETARY, VARSET_FULL]
        full_size = len(VARSET_FULL["arima_exog"])

        for vs in varsets[:-1]:
            assert full_size >= len(vs["arima_exog"])


class TestVarsetRegistry:
    """Tests for VARIABLE_SETS registry."""

    def test_all_varsets_registered(self):
        """All varsets should be in VARIABLE_SETS dict."""
        expected_names = ["parsimonious", "bop", "monetary", "pca", "full"]

        for name in expected_names:
            assert name in VARIABLE_SETS, f"Missing varset: {name}"

    def test_varset_order_matches_registry(self):
        """VARSET_ORDER should match VARIABLE_SETS keys."""
        for name in VARSET_ORDER:
            assert name in VARIABLE_SETS


class TestGetVarset:
    """Tests for get_varset helper function."""

    def test_get_valid_varset(self):
        """get_varset should return correct varset."""
        pars = get_varset("parsimonious")
        assert pars["name"] == "parsimonious"

    def test_get_invalid_varset_raises(self):
        """get_varset should raise for unknown varset."""
        with pytest.raises(KeyError):
            get_varset("nonexistent")

    def test_get_varset_case_sensitive(self):
        """Varset names should be case-sensitive."""
        # Should work with correct case
        get_varset("parsimonious")

        # Should fail with wrong case
        with pytest.raises(KeyError):
            get_varset("PARSIMONIOUS")
