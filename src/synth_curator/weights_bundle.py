"""Precomputed weights: skip the tournament (steps 4-6) and curate straight from an already-known
set of per-class metric weights - either your own, from a previous run (`weights_path=`), or one
shipped with the library (`shipped_weights=`). Both use the exact same bundle: a directory with

    weights.json   {class_name: {metric_name: weight}} - the same shape derive_weights() produces
    meta.json      what produced those weights - see meta_schema.json for the fields

`meta.json`'s `classes`/`metrics` are the only fields SynthCurator actually enforces (they're
cross-checked against weights.json here, and against your own configured classes/metrics in
pipeline.py); everything else in meta.json (model, formulas, provenance, ...) is informational -
there's no way for the library to judge whether a different model or domain will transfer well,
only you can.
"""
from __future__ import annotations

import json
import logging
from importlib import resources
from pathlib import Path
from typing import Any

from .exceptions import SynthCuratorError

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1


def _validate_bundle(weights: dict, meta: dict, label: str) -> None:
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise SynthCuratorError(
            f"{label}: meta.json schema_version={meta.get('schema_version')!r} isn't supported "
            f"here (expected {SCHEMA_VERSION}) - this bundle needs a different synth-curator version")
    missing = [k for k in ("classes", "metrics") if k not in meta]
    if missing:
        raise SynthCuratorError(f"{label}: meta.json is missing required key(s) {missing}")
    if set(weights) != set(meta["classes"]):
        raise SynthCuratorError(
            f"{label}: weights.json's classes {sorted(weights)} don't match meta.json's classes "
            f"{sorted(meta['classes'])} - this bundle is internally inconsistent")
    weights_metrics = {m for per_class in weights.values() for m in per_class}
    if weights_metrics != set(meta["metrics"]):
        raise SynthCuratorError(
            f"{label}: weights.json's metrics {sorted(weights_metrics)} don't match meta.json's "
            f"metrics {sorted(meta['metrics'])} - this bundle is internally inconsistent")
    _warn_on_version_mismatch(meta, label)


def _warn_on_version_mismatch(meta: dict, label: str) -> None:
    from . import __version__
    built_with = meta.get("provenance", {}).get("synth_curator_version")
    if built_with and built_with != __version__:
        log.warning(
            "%s: computed with synth-curator %s, you have %s installed - metric implementations "
            "can change between versions even when weight values wouldn't reflect it, so treat "
            "this as a prompt to double check, not an error", label, built_with, __version__)


def load_weights_path(path: str | Path) -> tuple[dict, dict]:
    """Bring your own: `path` is a directory containing weights.json + meta.json."""
    path = Path(path)
    label = f"weights_path={path}"
    weights_json, meta_json = path / "weights.json", path / "meta.json"
    if not weights_json.is_file():
        raise SynthCuratorError(f"{label}: weights.json not found")
    if not meta_json.is_file():
        raise SynthCuratorError(f"{label}: meta.json not found")
    weights, meta = json.loads(weights_json.read_text()), json.loads(meta_json.read_text())
    _validate_bundle(weights, meta, label)
    return weights, meta


def _shipped_root() -> Any:
    return resources.files("synth_curator").joinpath("shipped")


def list_shipped_weights() -> list[str]:
    """Names usable as `shipped_weights=...`."""
    root = _shipped_root()
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def load_shipped_weights(name: str) -> tuple[dict, dict]:
    label = f"shipped_weights={name!r}"
    bundle_dir = _shipped_root().joinpath(name)
    if not bundle_dir.is_dir():
        raise SynthCuratorError(f"{label} not found - available: {list_shipped_weights()}")
    weights = json.loads(bundle_dir.joinpath("weights.json").read_text())
    meta = json.loads(bundle_dir.joinpath("meta.json").read_text())
    _validate_bundle(weights, meta, label)
    return weights, meta
