#!/usr/bin/env python3
"""Synthesize mechanism evidence into a concise forecast-gain mapping table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reserves_project.config.paths import DATA_DIR
from reserves_project.eval.mechanism_synthesis import (
    build_mechanism_detail_table,
    build_mechanism_synthesis_table,
    load_disentangling_metrics,
    load_information_loss_metrics,
    load_irf_metrics,
    load_regime_metrics,
)
from reserves_project.utils.run_manifest import write_latest_pointer, write_run_manifest


def _resolve_component_dir(
    output_root: Path | None,
    provided_path: str,
    default_leaf: str,
) -> Path:
    path = Path(provided_path)
    if output_root is not None and provided_path == f"data/{default_leaf}":
        return output_root / default_leaf
    return path


def main():
    parser = argparse.ArgumentParser(description="Run mechanism synthesis across pipeline outputs.")
    parser.add_argument("--disentangling-dir", default="data/disentangling")
    parser.add_argument("--regime-dir", default="data/regime_characterization")
    parser.add_argument("--irf-dir", default="data/msvar_irf")
    parser.add_argument("--information-loss-dir", default="data/information_loss")
    parser.add_argument("--output-dir", default="data/mechanism_synthesis")
    parser.add_argument("--run-id", default=None, help="Optional run ID to nest outputs in data/outputs/<run-id>/.")
    parser.add_argument("--output-root", default=None, help="Optional output root (overrides --run-id).")
    args = parser.parse_args()

    output_root = None
    if args.output_root:
        output_root = Path(args.output_root)
    elif args.run_id:
        output_root = DATA_DIR / "outputs" / args.run_id

    dis_dir = _resolve_component_dir(output_root, args.disentangling_dir, "disentangling")
    reg_dir = _resolve_component_dir(output_root, args.regime_dir, "regime_characterization")
    irf_dir = _resolve_component_dir(output_root, args.irf_dir, "msvar_irf")
    info_dir = _resolve_component_dir(output_root, args.information_loss_dir, "information_loss")

    output_dir = Path(args.output_dir)
    if output_root is not None:
        output_dir = output_root / "mechanism_synthesis"
    output_dir.mkdir(parents=True, exist_ok=True)

    info_metrics, info_detail = load_information_loss_metrics(info_dir)
    dis_metrics, dis_detail = load_disentangling_metrics(
        dis_dir,
        aggregated_varset=info_metrics.get("aggregated_varset"),
        disaggregated_varset=info_metrics.get("disaggregated_varset"),
    )
    reg_metrics, reg_detail = load_regime_metrics(reg_dir)
    irf_metrics, irf_detail = load_irf_metrics(irf_dir)

    synthesis = build_mechanism_synthesis_table(
        disentangling_metrics=dis_metrics,
        regime_metrics=reg_metrics,
        irf_metrics=irf_metrics,
        information_loss_metrics=info_metrics,
    )
    detail = build_mechanism_detail_table(dis_detail, reg_detail, irf_detail, info_detail)

    synthesis.to_csv(output_dir / "mechanism_synthesis_table.csv", index=False)
    detail.to_csv(output_dir / "mechanism_synthesis_detail.csv", index=False)

    diagnostics = {
        "disentangling_dir": str(dis_dir),
        "regime_dir": str(reg_dir),
        "irf_dir": str(irf_dir),
        "information_loss_dir": str(info_dir),
        "output_dir": str(output_dir),
        "n_synthesis_rows": int(len(synthesis)),
        "n_detail_rows": int(len(detail)),
        "missing_mechanism_values": int(synthesis["mechanism_value"].isna().sum()),
        "missing_linked_gain_values": int(synthesis["linked_forecast_gain_value"].isna().sum()),
    }
    with open(output_dir / "mechanism_synthesis_diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

    write_run_manifest(output_dir, diagnostics)
    if args.run_id and output_root is not None:
        write_latest_pointer(DATA_DIR / "outputs", args.run_id, output_root)

    print(f"Synthesis rows: {len(synthesis)}")
    print(f"Detail rows: {len(detail)}")
    print(f"Saved mechanism synthesis outputs in {output_dir}")


if __name__ == "__main__":
    main()
