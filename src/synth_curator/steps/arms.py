"""Step 4. Curate one dataset per metric ("13 arms") plus a random control.

Each arm ranks the pool by a single metric only (top-N per class, matching the target class
distribution). Distributional metrics use the best-config + random-draw route.
Port of final_rerun single-metric curation (configs .../1_curate/2_best_metric.yml, metric_pipeline/dataset_curation).
Returns {arm_name: manifest_csv}.
"""
from pathlib import Path


def curate_arms(metadata_csv: Path, out_dir: Path, target_distribution: dict[str, float],
                block_size: int, seed: int) -> dict[str, Path]:
    raise NotImplementedError
