"""Run manifest utilities for reproducibility."""

from __future__ import annotations

import json
import hashlib
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import sys


def _hash_config(config: Dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def write_run_manifest(
    output_dir: Path,
    config: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write a run manifest JSON to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "config_hash": _hash_config(config),
        "config": config,
        "argv": sys.argv,
    }
    if extra:
        manifest["extra"] = extra

    path = output_dir / "run_manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return path


def write_latest_pointer(
    outputs_dir: Path,
    run_id: str,
    output_root: Path,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write latest pointer for run-id based outputs."""
    outputs_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "run_id": run_id,
        "output_root": str(output_root),
    }
    if extra:
        payload["extra"] = extra

    path = outputs_dir / "latest.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path
