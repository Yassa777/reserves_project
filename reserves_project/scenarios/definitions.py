"""Scenario definitions for conditional reserve forecasting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


DEFAULT_SHOCK_VARS = [
    "exports_usd_m",
    "imports_usd_m",
    "remittances_usd_m",
    "tourism_usd_m",
    "usd_lkr",
    "m2_usd_m",
    "trade_balance_usd_m",
]

SUPPORTED_PROFILES = {"ramp", "step", "impulse"}


@dataclass
class Scenario:
    """Scenario definition with variable-level shocks (multipliers)."""

    name: str
    description: str
    horizon_months: int = 12
    shocks: Dict[str, float] = field(default_factory=dict)
    profile: str = "ramp"  # ramp | step | impulse

    def normalized_shocks(self, variables: List[str] | None = None) -> Dict[str, float]:
        if variables is None:
            variables = DEFAULT_SHOCK_VARS
        out = {var: 1.0 for var in variables}
        out.update(self.shocks)
        return out

    def normalized_profile(self) -> str:
        profile = (self.profile or "ramp").lower()
        return profile if profile in SUPPORTED_PROFILES else "ramp"


POLICY_SCENARIOS: Dict[str, Scenario] = {
    "baseline": Scenario(
        name="Baseline",
        description="Current trajectory continues",
        horizon_months=12,
        shocks={},
        profile="ramp",
    ),
    "lkr_depreciation_10pct": Scenario(
        name="LKR Depreciation 10%",
        description="Exchange rate depreciates 10% over horizon",
        horizon_months=12,
        shocks={"usd_lkr": 1.10},
    ),
    "lkr_depreciation_20pct": Scenario(
        name="LKR Depreciation 20%",
        description="Severe exchange rate pressure",
        horizon_months=12,
        shocks={"usd_lkr": 1.20},
    ),
    "export_shock_negative": Scenario(
        name="Export Shock (-15%)",
        description="Global demand contraction hits exports",
        horizon_months=12,
        shocks={"exports_usd_m": 0.85, "trade_balance_usd_m": 0.70},
    ),
    "remittance_shock": Scenario(
        name="Remittance Decline (-20%)",
        description="Gulf employment crisis or hawala channel disruption",
        horizon_months=12,
        shocks={"remittances_usd_m": 0.80},
    ),
    "tourism_recovery": Scenario(
        name="Tourism Recovery (+25%)",
        description="Post-COVID tourism boom",
        horizon_months=12,
        shocks={"tourism_usd_m": 1.25},
    ),
    "oil_price_shock": Scenario(
        name="Oil Price Shock",
        description="Import bill rises due to energy prices",
        horizon_months=12,
        shocks={"imports_usd_m": 1.15, "trade_balance_usd_m": 0.75},
    ),
    "imf_tranche_delay": Scenario(
        name="IMF Tranche Delay",
        description="Expected disbursement delayed 6 months",
        horizon_months=12,
        shocks={"usd_lkr": 1.08, "imports_usd_m": 0.95},
    ),
    "combined_adverse": Scenario(
        name="Combined Adverse",
        description="Multiple simultaneous shocks (stress test)",
        horizon_months=12,
        shocks={
            "exports_usd_m": 0.90,
            "remittances_usd_m": 0.85,
            "tourism_usd_m": 0.80,
            "imports_usd_m": 1.10,
            "usd_lkr": 1.15,
        },
    ),
    "combined_upside": Scenario(
        name="Combined Upside",
        description="Favorable conditions across the board",
        horizon_months=12,
        shocks={
            "exports_usd_m": 1.10,
            "remittances_usd_m": 1.10,
            "tourism_usd_m": 1.20,
            "imports_usd_m": 0.95,
            "usd_lkr": 0.95,
        },
    ),
}

