"""steps/metrics/paired.py and single_image.py - direct calculator tests.

PSNR/SSIM only need scikit-image (installed here) and are tested for real. FSIM (phasepack),
LPIPS (lpips), BRISQUE (brisque) and CLIP (transformers) need deps not installed in this
environment; those tests are gated behind pytest.importorskip so they run wherever the deps are
available and skip cleanly here.
"""
import json
import math

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("skimage")

from synth_curator.steps.metrics import MetricContext, paired, single_image  # noqa: E402


def test_psnr_pair_returns_none_on_shape_mismatch():
    a = np.zeros((10, 10, 3), dtype=np.uint8)
    b = np.zeros((5, 5, 3), dtype=np.uint8)
    assert paired.psnr_pair(a, b) is None


def test_psnr_pair_identical_images_is_100():
    a = np.zeros((8, 8, 3), dtype=np.uint8)
    assert paired.psnr_pair(a, a.copy()) == 100.0


def test_psnr_pair_hand_computed():
    real = np.zeros((4, 4, 3), dtype=np.uint8)
    synth = np.full((4, 4, 3), 10, dtype=np.uint8)
    # mse = 100 -> psnr = 20*log10(255/sqrt(100)) = 20*log10(25.5)
    expected = 20 * math.log10(255.0 / 10.0)
    assert paired.psnr_pair(synth, real) == pytest.approx(expected, abs=1e-6)


def test_ssim_pair_identical_images_is_one():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
    assert paired.ssim_pair(img, img.copy()) == pytest.approx(1.0, abs=1e-6)


def test_ssim_pair_different_images_scores_lower_than_identical():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
    b = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
    assert paired.ssim_pair(a, b) < paired.ssim_pair(a, a.copy())


@pytest.fixture
def paired_ctx(tmp_path):
    from PIL import Image
    synth_dir, real_dir, feat_dir = tmp_path / "synth", tmp_path / "real_std", tmp_path / "feat"
    synth_dir.mkdir(); real_dir.mkdir(); feat_dir.mkdir()
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
    (synth_dir / "A").mkdir()
    Image.fromarray(img).save(synth_dir / "A" / "s0.png")
    Image.fromarray(img).save(real_dir / "r0.png")            # identical -> best-case scores
    (feat_dir / "knn_map.json").write_text(json.dumps({"A/s0.png": ["r0.png"]}))
    df = pd.DataFrame({"filename": ["A/s0.png"], "target_class_label": ["A"]})
    ctx = MetricContext(synth_dir, real_dir, feat_dir, tmp_path / "m.csv")
    return df, ctx


def test_fsim_needs_phasepack(paired_ctx):
    pytest.importorskip("phasepack")
    df, ctx = paired_ctx
    df["fsim_score"] = float("nan")
    out = paired.fsim(df, ctx)
    assert out.loc[0, "fsim_score"] == pytest.approx(1.0, abs=1e-3)   # identical images -> perfect FSIM


def test_lpips_needs_lpips_package(paired_ctx):
    pytest.importorskip("lpips")
    df, ctx = paired_ctx
    df["lpips_score"] = float("nan")
    out = paired.lpips(df, ctx)
    assert out.loc[0, "lpips_score"] == pytest.approx(0.0, abs=1e-2)   # identical images -> ~0 distance


def test_brisque_needs_brisque_package(tmp_path):
    pytest.importorskip("brisque")
    from PIL import Image
    synth_dir = tmp_path / "synth" / "A"
    synth_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(synth_dir / "s0.png")
    df = pd.DataFrame({"filename": ["A/s0.png"], "target_class_label": ["A"], "brisqe_score": [float("nan")]})
    ctx = MetricContext(tmp_path / "synth", tmp_path, tmp_path, tmp_path / "m.csv")
    out = single_image.brisque(df, ctx)
    assert np.isfinite(out.loc[0, "brisqe_score"])


def test_clip_needs_transformers(tmp_path):
    pytest.importorskip("transformers")
    from PIL import Image
    synth_dir = tmp_path / "synth" / "A"
    synth_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(synth_dir / "s0.png")
    df = pd.DataFrame({"filename": ["A/s0.png"], "target_class_label": ["A"],
                       "clip_similarity_cancer": [float("nan")], "clip_similarity_artifact": [float("nan")],
                       "clip_log_odds_score": [float("nan")]})
    ctx = MetricContext(tmp_path / "synth", tmp_path, tmp_path, tmp_path / "m.csv")
    out = single_image.clip(df, ctx)
    assert np.isfinite(out.loc[0, "clip_log_odds_score"])
