"""Metric registry: name -> type, optimize direction, and for distributional metrics the
family (fid|kid|swd) and feature space (dino|inception) they are computed on.

Port of METRIC_PROPERTIES in framework_utils/utils.py. Mutable on purpose: add your own entry
(and a calculator in steps/metrics) to extend the portfolio.
"""
from ..exceptions import SynthCuratorError


def _inst(optimize: str) -> dict:
    return {"type": "instance", "optimize": optimize}


def _dist(family: str, space: str) -> dict:
    return {"type": "distributional", "optimize": "min", "family": family, "space": space}


METRIC_PROPERTIES: dict[str, dict] = {
    "psnr_score": _inst("max"),
    "ssim_score": _inst("max"),
    "fsim_score": _inst("max"),
    "lpips_score": _inst("min"),
    "brisqe_score": _inst("min"),
    "fscs_score": _inst("max"),
    "clip_similarity_cancer": _inst("max"),
    "clip_similarity_artifact": _inst("min"),
    "clip_log_odds_score": _inst("max"),
    "diversity_score": _inst("max"),
    "realism_score": _inst("min"),
    "is_contribution_score": _inst("max"),
    "fid_inception": _dist("fid", "inception"),
    "fid_dino": _dist("fid", "dino"),
    "kid_dino": _dist("kid", "dino"),
    "swd_dino": _dist("swd", "dino"),
}

DEFAULT_DIST_MIN_SAMPLES = {"fid": 2048, "kid": 256, "swd": 64}


def props(metric: str) -> dict:
    try:
        return METRIC_PROPERTIES[metric]
    except KeyError:
        raise SynthCuratorError(f"unknown metric {metric!r}; known: {sorted(METRIC_PROPERTIES)}") from None


def is_distributional(metric: str) -> bool:
    return props(metric)["type"] == "distributional"


def is_instance(metric: str) -> bool:
    return props(metric)["type"] == "instance"


# The thesis portfolio (fid_dino is registered but was not part of the locked configuration).
DEFAULT_METRICS = [
    "clip_log_odds_score", "fscs_score", "clip_similarity_cancer", "clip_similarity_artifact",
    "lpips_score", "fsim_score", "brisqe_score", "psnr_score", "ssim_score",
    "diversity_score", "realism_score", "is_contribution_score",
    "fid_inception", "kid_dino", "swd_dino",
]
