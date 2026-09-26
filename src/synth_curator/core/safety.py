"""Optional memorization/similarity pre-filter (absolute threshold).

Port of v14 compute_max_similarity_to_real + apply_safety_filter (absolute mode only; the
relative percentile mode is not carried over).

Per-image max cosine similarity to ANY real image in DINOv2 space is the inner term of FSMS
(FSMS = 1 - mean_i max_j cos(synth_i, real_j)); high = close to a real image = memorization risk.
The threshold is ABSOLUTE and GLOBAL (not per class, not a percentile): a memorization breach
is a property of the image, not its rank within the pool. The thesis calibrated 0.97 against
ISIC same-lesion pairs (96.2% precision); calibrate your own for another domain.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
import torch

from ..exceptions import SynthCuratorError
from .features import load_features, stack_real, stack_synthetic
from .sampling import class_counts

log = logging.getLogger(__name__)

SIM_BATCH = 512


def compute_max_similarity_to_real(df: pd.DataFrame, features_dir: Path,
                                   cache_path: Path | None = None) -> pd.Series:
    if cache_path is not None and Path(cache_path).exists():
        cached = pd.read_csv(cache_path).set_index("filename")["max_sim_to_real"]
        mapped = df["filename"].map(cached)
        if mapped.notna().all():
            log.info("max-similarity loaded from cache %s", cache_path)
            return mapped.astype(float)
        log.info("cache %s does not cover this pool - recomputing", cache_path)

    real = stack_real(load_features(features_dir, "real", "dino"))
    real_n = real / (real.norm(dim=1, keepdim=True) + 1e-8)
    synth = stack_synthetic(load_features(features_dir, "synthetic", "dino"), df["filename"])
    synth_n = synth / (synth.norm(dim=1, keepdim=True) + 1e-8)

    sims = torch.cat([(synth_n[i:i + SIM_BATCH] @ real_n.T).max(dim=1).values
                      for i in range(0, len(synth_n), SIM_BATCH)]).numpy()

    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(cache_path.suffix + f".tmp{os.getpid()}")
        pd.DataFrame({"filename": df["filename"].values, "max_sim_to_real": sims}).to_csv(tmp, index=False)
        os.replace(tmp, cache_path)
    return pd.Series(sims, index=df.index, dtype=float)


def apply_safety_filter(df: pd.DataFrame, features_dir: Path, threshold: float,
                        target_distribution: dict[str, float], block_size: int,
                        cache_path: Path | None = None) -> pd.DataFrame:
    """Drop every image with max_sim_to_real >= threshold; fail early if a class is starved below the draw's quota."""
    df = df.assign(max_sim_to_real=compute_max_similarity_to_real(df, features_dir, cache_path))
    out = df[df["max_sim_to_real"] < threshold].copy()

    need = class_counts(target_distribution, block_size)
    short = {c: (int((out["target_class_label"] == c).sum()), n) for c, n in need.items()
             if int((out["target_class_label"] == c).sum()) < n}
    if short:
        raise SynthCuratorError(
            "safety filter left classes below the count the draw needs: "
            + ", ".join(f"{c} has {h}, needs {n}" for c, (h, n) in short.items())
            + ". Raise the threshold, or curate a smaller block.")
    log.info("safety filter (>= %g): pool %d -> %d images", threshold, len(df), len(out))
    return out
