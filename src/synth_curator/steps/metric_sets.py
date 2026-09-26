"""Step 4. Curate one dataset per metric (a "metric set") plus a random control, for the tournament.

Each metric set isolates ONE metric, so its downstream score says how useful that metric alone is.
    instance metric       top-N per class by the metric (deterministic)
    distributional metric best generation group by that metric (needs `group_by`), then a random
                          per-class draw from that group (fixed seed)
    random_control        a random per-class draw; a fresh draw per tournament seed (the baseline
                          that random_anchored weights are measured against)
Port of metric_pipeline/dataset_curation/main.py. Returns {metric_set: {seed: manifest_csv}}.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..core.groups import compute_group_raw
from ..core.output import write_manifest
from ..core.registry import is_distributional, props
from ..core.sampling import class_counts
from ..core.scoring import group_key

log = logging.getLogger(__name__)

RANDOM_METRIC_SET = "random_control"
DIST_METRIC_SET_SEED = 42   # distributional metric sets draw from the best group with a fixed seed


def _top_n_per_class(df: pd.DataFrame, metric: str, counts: dict[str, int]) -> pd.DataFrame:
    ascending = props(metric)["optimize"] == "min"
    valid = df.dropna(subset=[metric])
    parts = []
    for class_name, n in counts.items():
        cdf = valid[valid["target_class_label"] == class_name]
        if len(cdf) < n:
            log.warning("[%s] need %d images for class %s, have %d", metric, n, class_name, len(cdf))
        parts.append(cdf.sort_values(by=metric, ascending=ascending).head(n))
    return pd.concat(parts)


def _random_per_class(df: pd.DataFrame, counts: dict[str, int], seed: int) -> pd.DataFrame:
    parts = []
    for class_name, n in counts.items():
        cdf = df[df["target_class_label"] == class_name]
        parts.append(cdf.sample(n=min(n, len(cdf)), random_state=seed))
    return pd.concat(parts)


def curate_metric_sets(metadata_csv: Path, features_dir: Path, out_dir: Path, metrics: list[str],
                       target_distribution: dict[str, float], block_size: int, group_by: list[str],
                       seeds: list[int], image_dir: Path | None = None) -> dict[str, dict[int, Path]]:
    out_dir = Path(out_dir)
    df = pd.read_csv(metadata_csv)
    counts = class_counts(target_distribution, block_size)
    metric_sets: dict[str, dict[int, Path]] = {}

    def write(name: str, seed: int, selected: pd.DataFrame, tag: str = "") -> Path:
        path = out_dir / name / f"{name}{tag}.csv"
        if not path.exists():
            write_manifest(selected, path, image_dir)
        return path

    dist_metrics = [m for m in metrics if is_distributional(m)]
    group_raw = None
    if dist_metrics:
        if group_by and all(c in df.columns for c in group_by):
            group_raw = compute_group_raw(df, features_dir, group_by, dist_metrics)
        else:
            log.warning("distributional metric sets %s skipped: they need group_by columns present in the metadata",
                        dist_metrics)

    for m in metrics:
        if is_distributional(m):
            if group_raw is None:
                continue
            best = group_raw[m].idxmin()                       # all distributional metrics: lower is better
            pool = df[group_key(df, group_by) == best]
            path = write(m, 0, _random_per_class(pool, counts, DIST_METRIC_SET_SEED))
        else:
            if m not in df.columns:
                log.warning("metric set %s skipped: no such column in the metadata", m)
                continue
            path = write(m, 0, _top_n_per_class(df, m, counts))
        metric_sets[m] = {s: path for s in seeds}               # one fixed dataset, reused for every seed

    metric_sets[RANDOM_METRIC_SET] = {s: write(RANDOM_METRIC_SET, s, _random_per_class(df, counts, s), f"_seed{s}")
                                      for s in seeds}
    return metric_sets
