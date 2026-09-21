"""Step 7. The v15 framework: weights + metadata + features -> best curated set.

safety filter -> per-class instance scores -> group distributional scores -> composite ->
draw_blocks (temperature) -> [block_search: per-class distributional eval + select best] -> write.
Port of v15 run_for_arch. Write the manifest BEFORE the log (a log-first write once destroyed a finished run).
"""
from pathlib import Path


def curate(metadata_csv: Path, features_dir: Path, weights: dict, out_dir: Path, *,
           target_distribution: dict[str, float], block_size: int, temperature: float,
           group_by: list[str], block_search: bool, num_blocks: int,
           safety_max_similarity_threshold: float | None, seed: int) -> Path:
    raise NotImplementedError
