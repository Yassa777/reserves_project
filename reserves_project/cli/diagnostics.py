"""Run diagnostics pipeline."""

from reserves_project.pipelines.run_diagnostics import run_all_diagnostics


def main() -> None:
    run_all_diagnostics(verbose=True, write_manifest=True)
