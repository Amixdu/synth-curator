"""FID / KID / SWD (+ precision/recall helpers) on feature tensors.

Port of metric_pipeline/torch_metrics.py; math unchanged.
"""
from __future__ import annotations

import logging

import numpy as np
import torch
from scipy import linalg

log = logging.getLogger(__name__)


def get_stats(features: torch.Tensor, eps: float = 1e-6):
    """Mean and regularized covariance of a feature set (float64)."""
    mu = torch.mean(features, dim=0, dtype=torch.float64)
    sigma = torch.cov(features.to(torch.double).T)
    sigma = sigma + torch.eye(sigma.shape[0], device=features.device, dtype=torch.float64) * eps
    return mu, sigma


def calculate_fid(mu1: torch.Tensor, sigma1: torch.Tensor,
                  mu2: torch.Tensor, sigma2: torch.Tensor) -> float:
    """Frechet distance between two Gaussians (synthetic first, real second). Lower is better."""
    mu1_np = mu1.cpu().numpy().astype(np.float64)
    sigma1_np = sigma1.cpu().numpy().astype(np.float64)
    mu2_np = mu2.cpu().numpy().astype(np.float64)
    sigma2_np = sigma2.cpu().numpy().astype(np.float64)

    diff = mu1_np - mu2_np
    mean_term = diff.dot(diff)

    try:
        covmean = linalg.sqrtm(sigma1_np @ sigma2_np)
    except linalg.LinAlgError:
        log.warning("linalg.sqrtm failed to converge; returning inf")
        return float("inf")

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    if not np.isfinite(covmean).all():
        log.warning("sqrtm returned non-finite values; retrying with a small diagonal offset")
        offset = np.eye(sigma1_np.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma1_np + offset) @ (sigma2_np + offset))
        if np.iscomplexobj(covmean):
            covmean = covmean.real

    fid = mean_term + np.trace(sigma1_np + sigma2_np - 2 * covmean)
    return 0.0 if fid < 0 else float(fid)


def calculate_swd(features1, features2, n_projections=128, generator=None) -> float:
    """Sliced Wasserstein distance; equal row counts required. Pass a seeded generator for reproducibility."""
    features1 = features1.cpu().to(torch.float32)
    features2 = features2.cpu().to(torch.float32)
    projections = torch.randn(features1.shape[1], n_projections, generator=generator)
    projections = projections / torch.linalg.norm(projections, dim=0, keepdim=True)
    sorted1, _ = torch.sort(features1 @ projections, dim=0)
    sorted2, _ = torch.sort(features2 @ projections, dim=0)
    return torch.mean(torch.abs(sorted1 - sorted2)).item()


def _polynomial_kernel(X, Y, degree=3, gamma=None, coef0=1):
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    return (X @ Y.T * gamma + coef0) ** degree


def calculate_kid(features1, features2) -> float:
    """Kernel Inception Distance (polynomial kernel)."""
    features1 = features1.cpu().to(torch.float32)
    features2 = features2.cpu().to(torch.float32)
    k_xx = _polynomial_kernel(features1, features1)
    k_yy = _polynomial_kernel(features2, features2)
    k_xy = _polynomial_kernel(features1, features2)
    return (k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()).item()


def calculate_per_image_kl_divergence(probabilities: torch.Tensor, eps: float = 1e-16) -> torch.Tensor:
    """Per-image KL(p(y|x) || p(y)): the per-image term of the Inception Score."""
    eps_t = torch.tensor(eps, device=probabilities.device, dtype=torch.float32)
    p_y = probabilities.mean(dim=0, keepdim=True)
    return torch.sum(probabilities * (torch.log(probabilities + eps_t) - torch.log(p_y + eps_t)), dim=1)


def calculate_precision_recall_scores(real_features, synth_features, k=3) -> dict:
    """Global precision/recall plus per-image realism/coverage (k-NN distances). Uses the GPU if present."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real_features = real_features.to(device)
    synth_features = synth_features.to(device)

    def kth(dists, k_):
        return torch.kthvalue(dists, k=k_, dim=1)[0]

    per_image_realism = kth(torch.cdist(synth_features, real_features, p=2), k)
    per_image_coverage = kth(torch.cdist(real_features, synth_features, p=2), k)
    real_manifold = kth(torch.cdist(real_features, real_features, p=2), k + 1)

    precision = (per_image_realism.unsqueeze(1) <= real_manifold.unsqueeze(0)).any(dim=1).float().mean().item()
    recall = (per_image_coverage.unsqueeze(1) <= real_manifold.unsqueeze(0)).any(dim=1).float().mean().item()
    return {
        "precision": precision,
        "recall": recall,
        "per_image_realism_scores": per_image_realism.cpu(),
        "per_image_coverage_scores": per_image_coverage.cpu(),
    }
