"""Step 7. The v15 framework: weights + metadata + features -> best curated set.

safety filter -> per-class instance scores -> group distributional scores -> composite ->
draw_blocks (temperature) -> [block_search: per-class distributional eval + select best] -> write.
Port of v15 run_for_arch. Manifest is written BEFORE the logs.

Metadata CSV needs: filename, target_class_label, the `group_by` columns, and one column per
instance metric that carries weight.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..core.block_search import (
    evaluate_distributional_utility_per_class, select_optimal_block_per_class, write_class_log)
from ..core.groups import compute_group_scores
from ..core.output import write_block_log, write_manifest, write_weights_used
from ..core.registry import is_distributional, is_instance, props
from ..core.safety import apply_safety_filter
from ..core.sampling import draw_blocks
from ..core.scoring import calculate_per_class_score
from ..exceptions import MissingMetricsError, SynthCuratorError

log = logging.getLogger(__name__)


def curate(metadata_csv: Path, features_dir: Path, weights: dict, out_dir: Path, *,
           target_distribution: dict[str, float], num_images: int, temperature: float,
           group_by: list[str], block_search: bool, num_blocks: int,
           safety_max_similarity_threshold: float | None, seed: int,
           image_dir: Path | None = None, min_samples: dict | None = None,
           per_class_normalization: bool = True) -> Path:
    out_dir = Path(out_dir)
    manifest = out_dir / f"seed_{seed}.csv"
    if manifest.exists():
        log.info("%s already exists, skipping", manifest)
        return manifest
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = sorted({m for w in weights.values() for m in w})
    for m in all_metrics:
        props(m)  # unknown metric -> clear error
    num_blocks = num_blocks if block_search else 1

    df = pd.read_csv(metadata_csv)
    used_inst = sorted({m for w in weights.values() for m, v in w.items() if v > 1e-9 and is_instance(m)})
    missing = [m for m in used_inst if m not in df.columns]
    if missing:
        raise MissingMetricsError(f"pool lacks weighted instance metric columns: {missing}")
    for c in ("filename", "target_class_label"):
        if c not in df.columns:
            raise SynthCuratorError(f"metadata CSV lacks required column {c!r}")
    keep = ["filename", "target_class_label", *group_by, *all_metrics]
    df = df[[c for c in dict.fromkeys(keep) if c in df.columns]].copy()

    if safety_max_similarity_threshold is not None:
        df = apply_safety_filter(df, features_dir, safety_max_similarity_threshold,
                                 target_distribution, num_images,
                                 cache_path=out_dir.parent / ".safety_maxsim.csv")

    group_scores = None
    if group_by:
        absent = [c for c in group_by if c not in df.columns]
        if absent:
            log.warning("group_by columns %s not in metadata - group terms disabled", absent)
        else:
            dist_metrics = sorted({m for w in weights.values() for m, v in w.items()
                                   if v > 1e-9 and is_distributional(m)})
            if dist_metrics:
                group_scores = compute_group_scores(df, features_dir, group_by, dist_metrics,
                                                    cache_path=out_dir.parent / ".group_scores.csv")
            else:
                log.info("no distributional metric carries weight - group terms unused")

    df = calculate_per_class_score(df, weights, group_scores, group_by,
                                   per_class_bounds=per_class_normalization)
    blocks = draw_blocks(df, target_distribution, num_images, num_blocks, temperature, seed)

    if block_search:
        blocks = evaluate_distributional_utility_per_class(
            blocks, weights, features_dir, sorted(target_distribution), min_samples)
        optimal_df, blocks = select_optimal_block_per_class(blocks)
        write_class_log(blocks, out_dir / f".class_log_seed{seed}.csv")
    else:
        b = blocks[0]
        b["U_inst"] = b["df"]["S_Composite"].mean()
        b["U_dist"] = 0.0          # no per-block U_dist: the group term already acted during sampling
        b["S_Final"] = b["U_inst"]
        b["distributional_scores"] = {}
        optimal_df = b["df"]

    write_manifest(optimal_df, manifest, image_dir)            # deliverable first
    write_block_log(blocks, out_dir / f".block_log_seed{seed}.csv")
    write_weights_used(out_dir / f"weights_used_seed{seed}.json", weights, {
        "target_distribution": target_distribution, "num_images": num_images, "temperature": temperature,
        "group_by": group_by, "block_search": block_search, "num_blocks": num_blocks,
        "safety_max_similarity_threshold": safety_max_similarity_threshold, "seed": seed})
    log.info("curated %d images -> %s", len(optimal_df), manifest)
    return manifest
