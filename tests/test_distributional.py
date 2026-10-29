"""Hand-computable cases for core/distributional.py (FID/KID/SWD + the IS/precision-recall helpers)."""
import math

import pytest
import torch

from synth_curator.core.distributional import (
    calculate_fid, calculate_kid, calculate_per_image_kl_divergence,
    calculate_precision_recall_scores, calculate_swd, get_stats,
)


def test_get_stats_mean_and_covariance():
    feats = torch.tensor([[0.0, 0.0], [2.0, 4.0]])
    mu, sigma = get_stats(feats)
    assert mu.tolist() == pytest.approx([1.0, 2.0])
    # torch.cov of these two rows: var=2 each dim, cov=4 (population n-1 formula over 2 samples)
    assert sigma[0, 0].item() == pytest.approx(2.0, abs=2e-6)   # +1e-6 eps regularization
    assert sigma[1, 1].item() == pytest.approx(8.0, abs=2e-6)


def test_fid_is_zero_for_identical_stats():
    feats = torch.randn(50, 4)
    mu, sigma = get_stats(feats)
    assert calculate_fid(mu, sigma, mu, sigma) == pytest.approx(0.0, abs=1e-6)


def test_fid_hand_computed_2d_gaussians():
    # mu1=(0,0), cov1=I; mu2=(3,4), cov2=I (independent dims, both variance 1)
    # -> FID = |mu1-mu2|^2 + trace(I + I - 2*sqrt(I*I)) = (9+16) + 0 = 25
    # (scipy's sqrtm has a documented edge-case quirk on literal 1x1 matrices - use 2x2 here,
    # which matches real usage: feature spaces are always multi-dimensional.)
    mu1, sigma1 = torch.tensor([0.0, 0.0]), torch.eye(2, dtype=torch.float64)
    mu2, sigma2 = torch.tensor([3.0, 4.0]), torch.eye(2, dtype=torch.float64)
    assert calculate_fid(mu1, sigma1, mu2, sigma2) == pytest.approx(25.0, abs=1e-6)


def test_fid_hand_computed_different_variances():
    # mu equal (0,0); cov1=I, cov2=4*I -> FID = 0 + trace(I + 4I - 2*sqrt(4*I)) = trace(5I - 4I) = trace(I) = 2
    mu = torch.tensor([0.0, 0.0])
    sigma1, sigma2 = torch.eye(2, dtype=torch.float64), torch.eye(2, dtype=torch.float64) * 4
    assert calculate_fid(mu, sigma1, mu, sigma2) == pytest.approx(2.0, abs=1e-6)


def test_kid_is_zero_for_identical_sample_sets():
    feats = torch.randn(30, 6)
    assert calculate_kid(feats, feats) == pytest.approx(0.0, abs=1e-6)


def test_kid_is_nonzero_for_different_distributions():
    a = torch.zeros(20, 4)
    b = torch.ones(20, 4) * 10
    assert calculate_kid(a, b) != pytest.approx(0.0, abs=1e-3)


def test_swd_is_zero_for_identical_sample_sets():
    feats = torch.randn(40, 5)
    gen = torch.Generator().manual_seed(0)
    assert calculate_swd(feats, feats, generator=gen) == pytest.approx(0.0, abs=1e-6)


def test_swd_is_deterministic_given_a_seeded_generator():
    a, b = torch.randn(20, 4), torch.randn(20, 4)
    v1 = calculate_swd(a, b, generator=torch.Generator().manual_seed(7))
    v2 = calculate_swd(a, b, generator=torch.Generator().manual_seed(7))
    assert v1 == v2


def test_swd_hand_computed_constant_shift():
    # every projection is a unit vector; shifting every row by the same constant c along every
    # dimension shifts every projected+sorted value by c * sum(unit_vector) for that projection,
    # so the mean abs difference is exactly the mean of |c * column-sum of the projection matrix|.
    n, d, c = 10, 3, 2.0
    a = torch.zeros(n, d)
    b = torch.full((n, d), c)
    projections = torch.randn(d, 128, generator=torch.Generator().manual_seed(1))
    projections = projections / torch.linalg.norm(projections, dim=0, keepdim=True)
    expected = torch.abs(c * projections.sum(dim=0)).mean().item()
    got = calculate_swd(a, b, generator=torch.Generator().manual_seed(1))
    assert got == pytest.approx(expected, abs=1e-5)


def test_per_image_kl_divergence_hand_computed():
    probs = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    kl = calculate_per_image_kl_divergence(probs, eps=1e-12)
    # p_y = mean = [2/3, 1/3]; row [1,0] -> log(1/(2/3)) = log(3/2); row [0,1] -> log(1/(1/3)) = log(3)
    assert kl[0].item() == pytest.approx(math.log(3 / 2), abs=1e-4)
    assert kl[1].item() == pytest.approx(math.log(3 / 2), abs=1e-4)
    assert kl[2].item() == pytest.approx(math.log(3), abs=1e-4)


def test_precision_recall_scores_perfect_overlap():
    feats = torch.randn(20, 4)
    out = calculate_precision_recall_scores(feats, feats, k=3)
    assert out["precision"] == pytest.approx(1.0)
    assert out["recall"] == pytest.approx(1.0)
    assert out["per_image_realism_scores"].shape == (20,)
    assert out["per_image_coverage_scores"].shape == (20,)


def test_precision_recall_scores_disjoint_manifolds_score_low():
    real = torch.randn(20, 4)
    synth = torch.randn(20, 4) + 1000.0     # far away in feature space
    out = calculate_precision_recall_scores(real, synth, k=3)
    assert out["precision"] == pytest.approx(0.0)
    assert out["recall"] == pytest.approx(0.0)
