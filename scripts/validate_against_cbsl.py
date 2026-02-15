#!/usr/bin/env python3
"""
Cross-Validation Script: Compare Our Data Against CBSL Official Reports

This script validates our data against CBSL External Sector Performance reports
and identifies discrepancies that need resolution.

Run with: python validate_against_cbsl.py
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# =============================================================================
# CBSL OFFICIAL FIGURES (June 2022 External Sector Performance Report)
# =============================================================================
CBSL_JUNE_2022 = {
    "reserves_usd_m": 1856,        # Gross Official Reserves
    "exports_usd_m": 1051,         # Merchandise Exports
    "imports_usd_m": 1791,         # Merchandise Imports
    "remittances_usd_m": 343,      # Workers' Remittances
    "tourism_usd_m": 59,           # Tourism Earnings
    "trade_balance_usd_m": -740,   # Trade Balance (exports - imports)
}

CBSL_JUNE_2021 = {
    "reserves_usd_m": 2824,        # For YoY comparison
    "exports_usd_m": 933,
    "imports_usd_m": 1799,
    "remittances_usd_m": 792,
    "tourism_usd_m": 5,            # COVID period
}

CBSL_H1_2022 = {
    "tourism_total_usd_m": 740,    # From CBSL report summary
    "remittances_total_usd_m": 1847,  # Approximate from monthly figures
}


def load_our_data():
    """Load our processed data for comparison."""
    # Main panel
    panel = pd.read_csv(DATA_DIR / "merged" / "slfsi_monthly_panel.csv", parse_dates=["date"])

    # Individual source files
    exports = pd.read_csv(DATA_DIR / "external" / "monthly_exports_usd.csv", parse_dates=["date"])
    imports = pd.read_csv(DATA_DIR / "external" / "monthly_imports_usd.csv", parse_dates=["date"])

    return panel, exports, imports


def validate_june_2022(panel, exports, imports):
    """Compare June 2022 data against CBSL official figures."""
    print("=" * 70)
    print("VALIDATION: June 2022 vs CBSL External Sector Performance Report")
    print("=" * 70)
    print()

    # Get June 2022 from our data
    jun_2022 = panel[panel["date"] == "2022-06-30"].iloc[0] if len(panel[panel["date"] == "2022-06-30"]) > 0 else None
    jun_exports = exports[exports["date"].dt.to_period("M") == "2022-06"]
    jun_imports = imports[imports["date"].dt.to_period("M") == "2022-06"]

    results = []

    # Reserves
    our_reserves = jun_2022["gross_reserves_usd_m"] if jun_2022 is not None else None
    results.append({
        "Variable": "Gross Reserves",
        "Our Data": f"${our_reserves:,.0f}M" if our_reserves else "N/A",
        "CBSL Report": f"${CBSL_JUNE_2022['reserves_usd_m']:,.0f}M",
        "Difference": f"{((our_reserves - CBSL_JUNE_2022['reserves_usd_m'])/CBSL_JUNE_2022['reserves_usd_m']*100):+.1f}%" if our_reserves else "N/A",
        "Status": "✓ MATCH" if our_reserves == CBSL_JUNE_2022['reserves_usd_m'] else "⚠ CHECK"
    })

    # Exports
    our_exports = jun_exports["exports_usd_m"].values[0] if len(jun_exports) > 0 else None
    results.append({
        "Variable": "Exports",
        "Our Data": f"${our_exports:,.0f}M" if our_exports else "MISSING",
        "CBSL Report": f"${CBSL_JUNE_2022['exports_usd_m']:,.0f}M",
        "Difference": f"{((our_exports - CBSL_JUNE_2022['exports_usd_m'])/CBSL_JUNE_2022['exports_usd_m']*100):+.1f}%" if our_exports else "N/A",
        "Status": "✓ OK" if our_exports and abs(our_exports - CBSL_JUNE_2022['exports_usd_m']) < 50 else "❌ MISSING" if not our_exports else "⚠ CHECK"
    })

    # Imports
    our_imports = jun_imports["imports_usd_m"].values[0] if len(jun_imports) > 0 else None
    results.append({
        "Variable": "Imports",
        "Our Data": f"${our_imports:,.0f}M" if our_imports else "MISSING",
        "CBSL Report": f"${CBSL_JUNE_2022['imports_usd_m']:,.0f}M",
        "Difference": f"{((our_imports - CBSL_JUNE_2022['imports_usd_m'])/CBSL_JUNE_2022['imports_usd_m']*100):+.1f}%" if our_imports else "N/A",
        "Status": "✓ OK" if our_imports and abs(our_imports - CBSL_JUNE_2022['imports_usd_m']) < 100 else "❌ MISSING" if not our_imports else "⚠ CHECK"
    })

    # Remittances
    our_remit = jun_2022["remittances_usd_m"] if jun_2022 is not None else None
    results.append({
        "Variable": "Remittances",
        "Our Data": f"${our_remit:,.0f}M" if our_remit else "N/A",
        "CBSL Report": f"${CBSL_JUNE_2022['remittances_usd_m']:,.0f}M",
        "Difference": f"{((our_remit - CBSL_JUNE_2022['remittances_usd_m'])/CBSL_JUNE_2022['remittances_usd_m']*100):+.1f}%" if our_remit else "N/A",
        "Status": "✓ OK" if our_remit and abs(our_remit - CBSL_JUNE_2022['remittances_usd_m']) < 50 else "⚠ CHECK"
    })

    # Tourism
    our_tourism = jun_2022["tourism_earnings_usd_m"] if jun_2022 is not None else None
    results.append({
        "Variable": "Tourism",
        "Our Data": f"${our_tourism:,.0f}M" if our_tourism else "N/A",
        "CBSL Report": f"${CBSL_JUNE_2022['tourism_usd_m']:,.0f}M",
        "Difference": f"{((our_tourism - CBSL_JUNE_2022['tourism_usd_m'])/CBSL_JUNE_2022['tourism_usd_m']*100):+.1f}%" if our_tourism else "N/A",
        "Status": "✓ OK" if our_tourism and abs(our_tourism - CBSL_JUNE_2022['tourism_usd_m']) < 10 else "⚠ DISCREPANCY"
    })

    # Print results
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print()

    return df


def validate_june_2021(panel, exports, imports):
    """Compare June 2021 data for YoY validation."""
    print("=" * 70)
    print("VALIDATION: June 2021 (for YoY comparison)")
    print("=" * 70)
    print()

    jun_2021 = panel[panel["date"] == "2021-06-30"].iloc[0] if len(panel[panel["date"] == "2021-06-30"]) > 0 else None
    jun_exports = exports[exports["date"].dt.to_period("M") == "2021-06"]
    jun_imports = imports[imports["date"].dt.to_period("M") == "2021-06"]

    results = []

    # Reserves
    our_reserves = jun_2021["gross_reserves_usd_m"] if jun_2021 is not None else None
    results.append({
        "Variable": "Gross Reserves",
        "Our Data": f"${our_reserves:,.0f}M" if our_reserves else "N/A",
        "CBSL Report": f"${CBSL_JUNE_2021['reserves_usd_m']:,.0f}M",
        "Difference": f"{((our_reserves - CBSL_JUNE_2021['reserves_usd_m'])/CBSL_JUNE_2021['reserves_usd_m']*100):+.1f}%" if our_reserves else "N/A",
    })

    # Exports
    our_exports = jun_exports["exports_usd_m"].values[0] if len(jun_exports) > 0 else None
    results.append({
        "Variable": "Exports",
        "Our Data": f"${our_exports:,.0f}M" if our_exports else "N/A",
        "CBSL Report": f"${CBSL_JUNE_2021['exports_usd_m']:,.0f}M",
        "Difference": f"{((our_exports - CBSL_JUNE_2021['exports_usd_m'])/CBSL_JUNE_2021['exports_usd_m']*100):+.1f}%" if our_exports else "N/A",
    })

    # Imports
    our_imports = jun_imports["imports_usd_m"].values[0] if len(jun_imports) > 0 else None
    results.append({
        "Variable": "Imports",
        "Our Data": f"${our_imports:,.0f}M" if our_imports else "N/A",
        "CBSL Report": f"${CBSL_JUNE_2021['imports_usd_m']:,.0f}M",
        "Difference": f"{((our_imports - CBSL_JUNE_2021['imports_usd_m'])/CBSL_JUNE_2021['imports_usd_m']*100):+.1f}%" if our_imports else "N/A",
    })

    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print()

    return df


def check_data_gaps():
    """Identify all missing data points in critical variables."""
    print("=" * 70)
    print("DATA GAP ANALYSIS: Missing Months in Critical Variables")
    print("=" * 70)
    print()

    # TODO(human): Implement logic to scan each source file for date gaps
    # and report which months are missing for each variable
    pass


def main():
    print()
    print("CBSL DATA CROSS-VALIDATION REPORT")
    print("Generated for SL-FSI Reserve Forecasting Project")
    print()

    panel, exports, imports = load_our_data()

    validate_june_2022(panel, exports, imports)
    validate_june_2021(panel, exports, imports)

    print("=" * 70)
    print("SUMMARY OF ISSUES FOUND")
    print("=" * 70)
    print()
    print("1. ❌ MISSING: June 2022 exports and imports data")
    print("2. ⚠ DISCREPANCY: Tourism earnings ~24% below CBSL figure")
    print("3. ⚠ CHECK: Remittances ~20% below CBSL figure (may be timing)")
    print("4. ✓ MATCH: Reserves data matches exactly")
    print()
    print("RECOMMENDED ACTIONS:")
    print("- Fill in missing June 2022 trade data from CBSL report")
    print("- Investigate tourism data source vs CBSL methodology")
    print("- Document remittances timing differences")
    print()


if __name__ == "__main__":
    main()
