"""Marks a cached output with the inputs that built it, so a later call can tell it's stale. No
fingerprint file present is treated as a match (never invalidates a cache it didn't itself write)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _path(out_dir: Path) -> Path:
    return Path(str(out_dir) + ".fingerprint.json")


def exists(out_dir: Path) -> bool:
    return _path(out_dir).exists()


def matches(out_dir: Path, current: dict[str, Any]) -> bool:
    """True only if a fingerprint is present AND equals `current` exactly."""
    path = _path(out_dir)
    if not path.exists():
        return False
    try:
        stored = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return stored == current


def write(out_dir: Path, current: dict[str, Any]) -> None:
    _path(out_dir).write_text(json.dumps(current, indent=2, sort_keys=True))
