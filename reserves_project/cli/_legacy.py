"""Helpers to run legacy scripts while refactor is in progress."""

from __future__ import annotations

import importlib
import sys
from typing import Callable

from reserves_project.config.paths import PROJECT_ROOT


def _ensure_project_root() -> None:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def run_legacy_main(module_path: str) -> Callable[[], None]:
    """Return a callable that runs module_path.main() after path setup."""
    def _runner() -> None:
        _ensure_project_root()
        module = importlib.import_module(module_path)
        if not hasattr(module, "main"):
            raise AttributeError(f"{module_path} has no main()")
        module.main()
    return _runner
