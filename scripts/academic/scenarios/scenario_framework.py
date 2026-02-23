"""
Scenario Analysis Framework for Reserve Forecasting
====================================================

Provides policy-relevant scenario analysis capabilities:

1. CONDITIONAL FORECASTS
   - Given a path for exogenous variables, forecast reserves
   - Useful for: "What if exports grow 5% faster?"

2. REGIME SCENARIOS (MS-VAR specific)
   - Force model into "crisis" or "recovery" regime
   - Useful for: "What if we enter another crisis?"

3. STRESS TESTING
   - Historical shock replay (e.g., 2022 crisis)
   - Hypothetical adverse scenarios (2-sigma shocks)

4. BOP COMPONENT SCENARIOS
   - Decompose reserve changes by source
   - Simulate individual BoP line shocks

5. FAN CHARTS WITH SCENARIOS
   - Baseline + upside + downside paths
   - Policy-relevant confidence bands

Author: Academic Pipeline
Date: 2026-02
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_DIR / "forecast_results_unified"
OUTPUT_DIR = DATA_DIR / "scenario_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Scenario Definitions
# =============================================================================

@dataclass
class Scenario:
    """A scenario definition with shocks to exogenous variables."""
    name: str
    description: str
    horizon_months: int = 12
    shocks: Dict[str, float] = field(default_factory=dict)
    # Shock values are multipliers: 1.0 = baseline, 0.9 = -10%, 1.1 = +10%

    def __post_init__(self):
        # Default all shocks to 1.0 (no change)
        default_vars = [
            'exports_usd_m', 'imports_usd_m', 'remittances_usd_m',
            'tourism_usd_m', 'usd_lkr', 'm2_usd_m', 'trade_balance_usd_m'
        ]
        for var in default_vars:
            if var not in self.shocks:
                self.shocks[var] = 1.0


# Pre-defined policy scenarios
POLICY_SCENARIOS = {
    "baseline": Scenario(
        name="Baseline",
        description="Current trajectory continues",
        horizon_months=12,
        shocks={}
    ),

    "lkr_depreciation_10pct": Scenario(
        name="LKR Depreciation 10%",
        description="Exchange rate depreciates 10% over horizon",
        horizon_months=12,
        shocks={"usd_lkr": 1.10}
    ),

    "lkr_depreciation_20pct": Scenario(
        name="LKR Depreciation 20%",
        description="Severe exchange rate pressure",
        horizon_months=12,
        shocks={"usd_lkr": 1.20}
    ),

    "export_shock_negative": Scenario(
        name="Export Shock (-15%)",
        description="Global demand contraction hits exports",
        horizon_months=12,
        shocks={"exports_usd_m": 0.85, "trade_balance_usd_m": 0.70}
    ),

    "remittance_shock": Scenario(
        name="Remittance Decline (-20%)",
        description="Gulf employment crisis or hawala channel disruption",
        horizon_months=12,
        shocks={"remittances_usd_m": 0.80}
    ),

    "tourism_recovery": Scenario(
        name="Tourism Recovery (+25%)",
        description="Post-COVID tourism boom",
        horizon_months=12,
        shocks={"tourism_usd_m": 1.25}
    ),

    "oil_price_shock": Scenario(
        name="Oil Price Shock",
        description="Import bill rises due to energy prices",
        horizon_months=12,
        shocks={"imports_usd_m": 1.15, "trade_balance_usd_m": 0.75}
    ),

    "imf_tranche_delay": Scenario(
        name="IMF Tranche Delay",
        description="Expected disbursement delayed 6 months",
        horizon_months=12,
        shocks={
            "usd_lkr": 1.08,  # Confidence shock
            "imports_usd_m": 0.95,  # Import compression
        }
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
        }
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
        }
    ),
}


# Regime scenarios for MS-VAR
REGIME_SCENARIOS = {
    "stay_recovery": {
        "name": "Sustained Recovery",
        "description": "Economy remains in recovery regime",
        "regime_path": [1] * 12,  # Regime 1 = recovery
        "transition_override": None,
    },
    "crisis_return": {
        "name": "Crisis Return",
        "description": "Economy reverts to crisis regime",
        "regime_path": [0] * 12,  # Regime 0 = crisis
        "transition_override": None,
    },
    "gradual_transition": {
        "name": "Gradual Transition",
        "description": "Slow transition from recovery to crisis",
        "regime_path": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "transition_override": None,
    },
    "volatile": {
        "name": "Volatile Regime Switching",
        "description": "Frequent regime changes",
        "regime_path": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        "transition_override": None,
    },
}


# =============================================================================
# Scenario Engine
# =============================================================================

class ScenarioEngine:
    """
    Engine for running scenario analysis on trained models.

    Supports:
    - Conditional forecasting given exogenous paths
    - Regime-conditional MS-VAR forecasts
    - Stress testing with historical shocks
    """

    def __init__(self, varset: str = "parsimonious"):
        self.varset = varset
        self.data = self._load_data()
        self.last_observation = self.data.index[-1]

    def _load_data(self) -> pd.DataFrame:
        """Load the variable set data."""
        path = DATA_DIR / "forecast_prep_academic" / f"varset_{self.varset}" / "vecm_levels.csv"
        df = pd.read_csv(path, parse_dates=['date'])
        df = df.set_index('date').sort_index()
        return df

    def generate_scenario_paths(
        self,
        scenario: Scenario,
        base_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Generate exogenous variable paths for a scenario.

        Parameters
        ----------
        scenario : Scenario
            The scenario definition with shocks
        base_date : pd.Timestamp, optional
            Starting date (defaults to last observation)

        Returns
        -------
        pd.DataFrame
            DataFrame with projected paths for each variable
        """
        if base_date is None:
            base_date = self.last_observation

        # Get last known values
        last_values = self.data.loc[base_date].to_dict()

        # Generate future dates
        future_dates = pd.date_range(
            start=base_date + pd.DateOffset(months=1),
            periods=scenario.horizon_months,
            freq='MS'
        )

        # Build paths with linear interpolation of shocks
        paths = {}
        for var in self.data.columns:
            if var == 'gross_reserves_usd_m':
                continue  # This is what we're forecasting

            base_value = last_values.get(var, np.nan)
            shock = scenario.shocks.get(var, 1.0)

            # Linear interpolation from 1.0 to shock over horizon
            shock_path = np.linspace(1.0, shock, scenario.horizon_months)
            paths[var] = base_value * shock_path

        return pd.DataFrame(paths, index=future_dates)

    def compute_historical_shocks(
        self,
        reference_period: Tuple[str, str] = ("2022-01", "2022-12"),
    ) -> Dict[str, float]:
        """
        Compute historical shock magnitudes for stress testing.

        Parameters
        ----------
        reference_period : tuple
            (start_date, end_date) for shock calculation

        Returns
        -------
        dict
            Shock multipliers based on historical changes
        """
        start, end = pd.Timestamp(reference_period[0]), pd.Timestamp(reference_period[1])

        # Get values at start and end of period
        start_values = self.data.loc[:start].iloc[-1]
        end_values = self.data.loc[:end].iloc[-1]

        # Compute percentage changes
        shocks = {}
        for var in self.data.columns:
            if var == 'gross_reserves_usd_m':
                continue
            if start_values[var] != 0:
                shocks[var] = end_values[var] / start_values[var]
            else:
                shocks[var] = 1.0

        return shocks

    def generate_stress_scenario(
        self,
        sigma: float = 2.0,
        direction: str = "adverse",
    ) -> Scenario:
        """
        Generate a stress scenario based on historical volatility.

        Parameters
        ----------
        sigma : float
            Number of standard deviations for shock
        direction : str
            "adverse" (reserves decline) or "favorable" (reserves increase)

        Returns
        -------
        Scenario
            Generated stress scenario
        """
        # Compute historical volatility (monthly changes)
        changes = self.data.pct_change().dropna()
        volatility = changes.std()

        # For adverse: exports down, imports up, FX depreciation
        # For favorable: opposite
        sign = -1 if direction == "adverse" else 1

        shocks = {}
        for var in self.data.columns:
            if var == 'gross_reserves_usd_m':
                continue

            # Determine shock direction based on variable type
            if var in ['exports_usd_m', 'remittances_usd_m', 'tourism_usd_m']:
                # Inflows: adverse = decrease
                shocks[var] = 1 + sign * sigma * volatility.get(var, 0.05)
            elif var in ['imports_usd_m']:
                # Outflows: adverse = increase
                shocks[var] = 1 - sign * sigma * volatility.get(var, 0.05)
            elif var == 'usd_lkr':
                # FX: adverse = depreciation (higher value)
                shocks[var] = 1 - sign * sigma * volatility.get(var, 0.03)
            else:
                shocks[var] = 1.0

        return Scenario(
            name=f"{sigma}σ {direction.title()} Stress",
            description=f"Historical {sigma}-sigma shock in {direction} direction",
            horizon_months=12,
            shocks=shocks
        )


# =============================================================================
# MS-VAR Regime-Conditional Forecasts
# =============================================================================

class MSVARScenarioAnalyzer:
    """
    Scenario analysis specific to Markov-Switching VAR.

    Enables:
    - Regime-conditional forecasts
    - Transition probability scenarios
    - Crisis probability paths
    """

    def __init__(self, varset: str = "parsimonious"):
        self.varset = varset
        self.results = self._load_results()

    def _load_results(self) -> Optional[pd.DataFrame]:
        """Load MS-VAR forecast results."""
        path = RESULTS_DIR / f"rolling_origin_forecasts_{self.varset}.csv"
        if path.exists():
            df = pd.read_csv(path, parse_dates=['origin', 'target_date'])
            return df[df['model'] == 'MS-VAR']
        return None

    def get_regime_probabilities(self) -> pd.DataFrame:
        """
        Extract smoothed regime probabilities from MS-VAR.

        Returns
        -------
        pd.DataFrame
            Regime probabilities over time
        """
        # TODO(human): This requires access to the fitted MS-VAR model
        # Placeholder for now
        pass

    def conditional_forecast_by_regime(
        self,
        regime: int,
        horizon: int = 12,
    ) -> pd.DataFrame:
        """
        Generate forecasts conditional on staying in a specific regime.

        Parameters
        ----------
        regime : int
            0 = crisis, 1 = recovery
        horizon : int
            Forecast horizon in months

        Returns
        -------
        pd.DataFrame
            Conditional forecasts with confidence intervals
        """
        # TODO(human): Implement conditional simulation from MS-VAR
        pass


# =============================================================================
# BoP Component Decomposition
# =============================================================================

class BoPScenarioAnalyzer:
    """
    Balance of Payments component-level scenario analysis.

    Decomposes reserve changes into:
    - Current account (exports - imports + remittances + tourism)
    - Financial account (portfolio flows, FDI)
    - Valuation effects (FX changes on non-USD reserves)
    """

    def __init__(self):
        self.data = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        """Load BoP variable set."""
        path = DATA_DIR / "forecast_prep_academic" / "varset_bop" / "vecm_levels.csv"
        df = pd.read_csv(path, parse_dates=['date'])
        return df.set_index('date').sort_index()

    def decompose_reserve_change(
        self,
        start_date: str,
        end_date: str,
    ) -> Dict[str, float]:
        """
        Decompose historical reserve change by BoP component.

        Returns contribution of each component to total reserve change.
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        start_vals = self.data.loc[:start].iloc[-1]
        end_vals = self.data.loc[:end].iloc[-1]

        # Reserve change
        reserve_change = end_vals['gross_reserves_usd_m'] - start_vals['gross_reserves_usd_m']

        # Component changes (cumulative over period)
        period_data = self.data.loc[start:end]

        decomposition = {
            "total_reserve_change": reserve_change,
            "exports_contribution": period_data['exports_usd_m'].sum() - start_vals['exports_usd_m'] * len(period_data),
            "imports_contribution": -(period_data['imports_usd_m'].sum() - start_vals['imports_usd_m'] * len(period_data)),
            "remittances_contribution": period_data['remittances_usd_m'].sum() - start_vals['remittances_usd_m'] * len(period_data),
            "tourism_contribution": period_data['tourism_usd_m'].sum() - start_vals['tourism_usd_m'] * len(period_data),
        }

        # Residual (valuation + financial account + errors)
        explained = sum(v for k, v in decomposition.items() if k != "total_reserve_change")
        decomposition["residual"] = reserve_change - explained

        return decomposition

    def project_with_component_shocks(
        self,
        scenario: Scenario,
        base_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Project reserves by shocking individual BoP components.

        This builds up reserve forecasts from BoP identity:
        ΔReserves ≈ Exports - Imports + Remittances + Tourism + Other
        """
        if base_date is None:
            base_date = self.data.index[-1]
        else:
            base_date = pd.Timestamp(base_date)

        # Get monthly averages from recent history
        recent = self.data.loc[base_date - pd.DateOffset(months=12):base_date]
        monthly_avg = recent.mean()

        # Generate future dates
        future_dates = pd.date_range(
            start=base_date + pd.DateOffset(months=1),
            periods=scenario.horizon_months,
            freq='MS'
        )

        # Build projections with shocks
        projections = []
        current_reserves = self.data.loc[base_date, 'gross_reserves_usd_m']

        for i, date in enumerate(future_dates):
            # Interpolate shock (linear from 1.0 to target)
            shock_frac = (i + 1) / scenario.horizon_months

            # Project each component
            exports = monthly_avg['exports_usd_m'] * (1 + shock_frac * (scenario.shocks.get('exports_usd_m', 1.0) - 1))
            imports = monthly_avg['imports_usd_m'] * (1 + shock_frac * (scenario.shocks.get('imports_usd_m', 1.0) - 1))
            remittances = monthly_avg['remittances_usd_m'] * (1 + shock_frac * (scenario.shocks.get('remittances_usd_m', 1.0) - 1))
            tourism = monthly_avg['tourism_usd_m'] * (1 + shock_frac * (scenario.shocks.get('tourism_usd_m', 1.0) - 1))

            # BoP identity (simplified)
            monthly_flow = exports - imports + remittances + tourism

            # Update reserves
            current_reserves = current_reserves + monthly_flow

            projections.append({
                'date': date,
                'gross_reserves_usd_m': current_reserves,
                'exports_usd_m': exports,
                'imports_usd_m': imports,
                'remittances_usd_m': remittances,
                'tourism_usd_m': tourism,
                'monthly_flow': monthly_flow,
            })

        return pd.DataFrame(projections).set_index('date')


# =============================================================================
# Visualization
# =============================================================================

def create_scenario_fan_chart(
    baseline: pd.Series,
    scenarios: Dict[str, pd.Series],
    title: str = "Reserve Scenarios",
    output_path: Optional[Path] = None,
):
    """
    Create fan chart with baseline and scenario paths.

    Parameters
    ----------
    baseline : pd.Series
        Baseline forecast path
    scenarios : dict
        Dictionary of scenario name -> forecast path
    title : str
        Chart title
    output_path : Path, optional
        Where to save the figure
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot baseline
    ax.plot(baseline.index, baseline.values, 'k-', linewidth=2.5, label='Baseline')

    # Color map for scenarios
    colors = {
        'adverse': 'red',
        'upside': 'green',
        'downside': 'orange',
        'stress': 'darkred',
    }

    for name, path in scenarios.items():
        color = 'blue'
        for key, c in colors.items():
            if key in name.lower():
                color = c
                break
        ax.plot(path.index, path.values, '--', linewidth=1.5, label=name, color=color, alpha=0.7)

    # Formatting
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Gross Reserves (USD million)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Add horizontal lines for key thresholds
    ax.axhline(3000, color='red', linestyle=':', alpha=0.5, label='Critical (3 months imports)')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# =============================================================================
# Main Execution
# =============================================================================

def run_scenario_analysis(varset: str = "parsimonious"):
    """Run comprehensive scenario analysis."""
    print("="*70)
    print("SCENARIO ANALYSIS FOR RESERVE FORECASTING")
    print("="*70)

    # Initialize engine
    engine = ScenarioEngine(varset=varset)

    print(f"\nVariable set: {varset}")
    print(f"Last observation: {engine.last_observation}")

    # Generate paths for each policy scenario
    print("\n### Policy Scenarios ###")
    for key, scenario in POLICY_SCENARIOS.items():
        paths = engine.generate_scenario_paths(scenario)
        print(f"\n{scenario.name}:")
        print(f"  Description: {scenario.description}")
        print(f"  Key shocks: {scenario.shocks}")

    # Historical shock analysis
    print("\n### Historical Shock Analysis ###")
    crisis_shocks = engine.compute_historical_shocks(("2022-01", "2022-12"))
    print("2022 Crisis shocks (multipliers):")
    for var, shock in crisis_shocks.items():
        print(f"  {var}: {shock:.2f}x")

    # Generate stress scenario
    print("\n### Stress Scenarios ###")
    stress_adverse = engine.generate_stress_scenario(sigma=2.0, direction="adverse")
    print(f"2σ Adverse scenario: {stress_adverse.shocks}")

    # BoP decomposition
    print("\n### BoP Decomposition ###")
    bop_analyzer = BoPScenarioAnalyzer()
    decomp = bop_analyzer.decompose_reserve_change("2022-01", "2022-12")
    print("2022 Reserve Change Decomposition:")
    for component, value in decomp.items():
        print(f"  {component}: {value:,.0f}")

    print("\n" + "="*70)
    print("Scenario analysis complete.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_scenario_analysis()
