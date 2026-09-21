"""Step 3. Run the 13-metric portfolio over the pool -> metadata CSV (one row per image, one column per metric).

Port of metric_pipeline/run_metric_pipeline.py + metric_calculation_scripts/calculate_*.py, but
in-process (no subprocess / temp YAML). Each metric is one function in this file (or a small
`metrics/` package if it grows: LPIPS/CLIP/BRISQUE need optional heavy deps, import them lazily).
"""
from pathlib import Path


def compute_metrics(synthetic_dir: Path, real_dir: Path, features_dir: Path, out_csv: Path) -> Path:
    raise NotImplementedError
