"""Compatibility shim for diagnostics pipeline."""

from reserves_project.pipelines.run_diagnostics import run_all_diagnostics

if __name__ == "__main__":
    run_all_diagnostics(verbose=True, write_manifest=True)
