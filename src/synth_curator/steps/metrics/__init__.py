"""Step 3. Compute the per-image (instance) metrics -> metadata CSV (one row per image, one column per metric).

Port of metric_pipeline/run_metric_pipeline.py + metric_calculation_scripts/calculate_*.py, in-process.
Distributional metrics (FID/KID/SWD) are NOT columns: they are scored per generation group at curation time.

Metrics are computed by a handful of "steps" (CLIP yields 3 columns). Each step skips work that is
already in the CSV, and the CSV is saved as it goes, so an interrupted run resumes.

    paired.py         PSNR, SSIM, FSIM, LPIPS   compared to the nearest real image (knn_map.json)
    single_image.py   BRISQUE, CLIP             one image at a time
    from_features.py  FS-CS, realism, diversity, Inception Score   from the step-2 feature tensors

Base metadata: an existing CSV if you have one (filename, target_class_label, + group columns),
else built by scanning `synthetic_dir/<class_name>/**/<image>` (class = first folder).
"""
from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ...core.registry import is_instance
from ...exceptions import SynthCuratorError
from ..standardize import list_images
from . import from_features, paired, single_image

log = logging.getLogger(__name__)


@dataclass
class MetricContext:
    synthetic_dir: Path
    real_std_dir: Path
    features_dir: Path
    out_csv: Path
    epsilon: float = 1e-6
    checkpoint_every: int = 1000

    def save(self, df: pd.DataFrame) -> None:
        tmp = self.out_csv.with_suffix(f".tmp{os.getpid()}")
        df.to_csv(tmp, index=False)
        os.replace(tmp, self.out_csv)


# step name -> (columns it fills, function(df, ctx) -> df)
STEPS: dict[str, tuple[list[str], Callable]] = {
    "psnr": (["psnr_score"], paired.psnr),
    "ssim": (["ssim_score"], paired.ssim),
    "fsim": (["fsim_score"], paired.fsim),
    "lpips": (["lpips_score"], paired.lpips),
    "brisque": (["brisqe_score"], single_image.brisque),
    "clip": (["clip_similarity_cancer", "clip_similarity_artifact", "clip_log_odds_score"], single_image.clip),
    "fscs": (["fscs_score"], from_features.fscs),
    "diversity": (["diversity_score"], from_features.diversity),
    "realism": (["realism_score"], from_features.realism),
    "inception_score": (["is_contribution_score"], from_features.inception_score),
}
STEP_OF_METRIC = {col: step for step, (cols, _) in STEPS.items() for col in cols}


def build_metadata(synthetic_dir: Path) -> pd.DataFrame:
    """filename (relative), target_class_label (= first folder under synthetic_dir)."""
    rows = []
    for p in list_images(synthetic_dir):
        rel = p.relative_to(synthetic_dir)
        if len(rel.parts) < 2:
            raise SynthCuratorError(
                f"{p}: expected <class_name>/.../<image>. Pass a metadata CSV (filename, target_class_label) instead.")
        rows.append({"filename": str(rel), "target_class_label": rel.parts[0]})
    if not rows:
        raise SynthCuratorError(f"no images under {synthetic_dir}")
    return pd.DataFrame(rows)


def compute_metrics(synthetic_dir: Path, real_std_dir: Path, features_dir: Path, out_csv: Path,
                    metrics: list[str], base_metadata: Path | None = None, **ctx_kwargs) -> Path:
    out_csv = Path(out_csv)
    if out_csv.exists():
        df = pd.read_csv(out_csv)
    elif base_metadata is not None:
        df = pd.read_csv(base_metadata)
    else:
        df = build_metadata(Path(synthetic_dir))
    for c in ("filename", "target_class_label"):
        if c not in df.columns:
            raise SynthCuratorError(f"metadata lacks required column {c!r}")

    wanted = [m for m in metrics if is_instance(m)]
    unknown = [m for m in wanted if m not in STEP_OF_METRIC]
    if unknown:
        raise SynthCuratorError(f"no calculator for metrics {unknown}")

    ctx = MetricContext(Path(synthetic_dir), Path(real_std_dir), Path(features_dir), out_csv, **ctx_kwargs)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    for step in dict.fromkeys(STEP_OF_METRIC[m] for m in wanted):
        cols, fn = STEPS[step]
        if all(c in df.columns and df[c].notna().all() for c in cols):
            log.info("%s already complete, skipping", step)
            continue
        for c in cols:
            if c not in df.columns:
                df[c] = float("nan")
        log.info("computing %s", step)
        df = fn(df, ctx)
        ctx.save(df)
    if not out_csv.exists():      # no metric needed computing: still hand back a complete metadata file
        ctx.save(df)
    return out_csv
