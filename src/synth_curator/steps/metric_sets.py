"""Step 4. Curate one dataset per metric (a "metric set") plus a random control, for the tournament.

Each metric set isolates ONE metric, so its downstream score says how useful that metric alone is.
    instance metric       top-N per class by the metric (deterministic - curated once, trained once,
                          nothing for random_seeds_per_metric_set to average out)
    distributional metric best generation group by that metric (needs `dist_group_by`), then a
                          fresh random per-class draw from that group, per tournament seed
    random_control        a random per-class draw; a fresh draw per tournament seed (the baseline
                          that random_anchored weights are measured against)
Distributional metrics and random_control are both randomized, so both get one manifest per seed
in `seeds` - that's what `random_seeds_per_metric_set` actually repeats. Instance metrics are
deterministic, so they get exactly one manifest regardless of how many seeds are configured.
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
from ..exceptions import SynthCuratorError

log = logging.getLogger(__name__)

RANDOM_METRIC_SET = "random_control"


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
                       target_distribution: dict[str, float], num_images: int, dist_group_by: list[str],
                       seeds: list[int], image_dir: Path | None = None) -> dict[str, dict[int, Path]]:
    out_dir = Path(out_dir)
    df = pd.read_csv(metadata_csv)
    counts = class_counts(target_distribution, num_images)
    metric_sets: dict[str, dict[int, Path]] = {}

    def write(name: str, seed: int, selected: pd.DataFrame, tag: str = "") -> Path:
        path = out_dir / name / f"{name}{tag}.csv"
        if not path.exists():
            write_manifest(selected, path, image_dir)
        return path

    dist_metrics = [m for m in metrics if is_distributional(m)]
    group_raw = None
    if dist_metrics:
        if dist_group_by and all(c in df.columns for c in dist_group_by):
            if group_key(df, dist_group_by).nunique() < 2:
                raise SynthCuratorError(
                    f"dist_group_by {dist_group_by} produces only one group across the whole pool - "
                    f"{dist_metrics} need >=2 real generation groups to compare, so they can't do "
                    "anything here. Drop them from metrics, or fix dist_group_by.")
            group_raw = compute_group_raw(df, features_dir, dist_group_by, dist_metrics)
        else:
            log.warning("distributional metric sets %s skipped: they need dist_group_by columns present in the metadata",
                        dist_metrics)

    for m in metrics:
        if is_distributional(m):
            if group_raw is None:
                continue
            best = group_raw[m].idxmin()                       # all distributional metrics: lower is better
            pool = df[group_key(df, dist_group_by) == best]
            # randomized, like random_control - a fresh draw per seed, not one fixed dataset
            metric_sets[m] = {s: write(m, s, _random_per_class(pool, counts, s), f"_seed{s}") for s in seeds}
        else:
            if m not in df.columns:
                log.warning("metric set %s skipped: no such column in the metadata", m)
                continue
            # deterministic - one dataset, trained once, regardless of how many seeds are configured
            metric_sets[m] = {seeds[0]: write(m, seeds[0], _top_n_per_class(df, m, counts))}

    metric_sets[RANDOM_METRIC_SET] = {s: write(RANDOM_METRIC_SET, s, _random_per_class(df, counts, s), f"_seed{s}")
                                      for s in seeds}
    return metric_sets
