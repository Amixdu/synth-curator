"""Curation outputs: manifest CSV, block log, weights-used JSON.

Port of create_csv_manifest (metric_pipeline/dataset_curation/main.py) and generate_evaluation_log.
Write the manifest BEFORE the log: the manifest is the deliverable, and a log-first write once
destroyed a completed curation.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_manifest(selected: pd.DataFrame, out_csv: Path, image_dir: Path | None = None) -> Path:
    """CSV with source_path, target_class, safe_filename."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    src = selected["filename"].map(lambda f: str(image_dir / f) if image_dir else str(f))
    pd.DataFrame({
        "source_path": src.values,
        "target_class": selected["target_class_label"].values,
        "safe_filename": selected["filename"].astype(str).str.replace("/", "_").str.replace("\\", "_").values,
    }).to_csv(out_csv, index=False)
    return out_csv


def write_block_log(blocks: list[dict], path: Path) -> None:
    """Per-image scores + block-level utilities for every drawn block."""
    frames = []
    for b in blocks:
        d = b["df"].copy()
        d["Block_U_inst"], d["Block_U_dist"], d["Block_S_Final"] = b["U_inst"], b["U_dist"], b["S_Final"]
        for k, v in b["distributional_scores"].items():
            d[k] = v
        frames.append(d)
    log_df = pd.concat(frames, ignore_index=True)
    core = [c for c in ["filename", "Block_ID", "target_class_label", "S_Composite", "Block_S_Final",
                        "Block_U_inst", "Block_U_dist"] if c in log_df.columns]
    log_df[core + [c for c in log_df.columns if c not in core]].to_csv(path, index=False)


def write_weights_used(path: Path, weights: dict, params: dict) -> None:
    from .. import __version__
    path.write_text(json.dumps({"algorithm": "v15", "library_version": __version__,
                                "params": params, "weights": weights}, indent=2))
