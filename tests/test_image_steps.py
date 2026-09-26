"""Image-touching steps on tiny generated images. Skipped when the [full] extras are absent."""
import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("PIL")
pytest.importorskip("skimage")

from PIL import Image  # noqa: E402

from synth_curator.exceptions import SynthCuratorError  # noqa: E402
from synth_curator.steps import features, standardize  # noqa: E402
from synth_curator.steps.metrics import compute_metrics  # noqa: E402


def _img(path, seed, size=(80, 60)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.random.default_rng(seed).integers(0, 255, (*size[::-1], 3), dtype=np.uint8)).save(path)


def test_standardize_center_crops_resizes_and_flattens(tmp_path):
    _img(tmp_path / "real" / "A" / "x.jpg", 0)
    out = standardize.standardize(tmp_path / "real", tmp_path / "std", 32)
    assert Image.open(out / "x.png").size == (32, 32)


def test_standardize_rejects_duplicate_stems(tmp_path):
    _img(tmp_path / "real" / "A" / "x.jpg", 0)
    _img(tmp_path / "real" / "B" / "x.jpg", 1)
    with pytest.raises(SynthCuratorError, match="duplicate"):
        standardize.standardize(tmp_path / "real", tmp_path / "std", 32)


@pytest.fixture
def world(tmp_path):
    """3 real images (class A/B), 4 synthetic; synthetic 0 is a copy of real 0 so its nearest real is identical."""
    real, synth = tmp_path / "real", tmp_path / "synth"
    for i, cls in enumerate(["A", "A", "B"]):
        _img(real / cls / f"r{i}.png", i, (64, 64))
    for i, cls in enumerate(["A", "A", "B", "B"]):
        _img(synth / cls / f"s{i}.png", 10 + i, (64, 64))
    Image.open(real / "A" / "r0.png").save(synth / "A" / "s0.png")
    std = standardize.standardize(real, tmp_path / "std", 64)

    feats = tmp_path / "feat"
    feats.mkdir()
    rng = np.random.default_rng(0)
    rvec = {f"r{i}.png": torch.randn(8) for i in range(3)}
    svec = {f"{c}/s{i}.png": torch.randn(8) for i, c in enumerate(["A", "A", "B", "B"])}
    svec["A/s0.png"] = rvec["r0.png"].clone()                       # identical feature -> nearest is r0
    torch.save(rvec, feats / "real_dino.pt")
    torch.save(svec, feats / "synthetic_dino.pt")
    torch.save({k: torch.softmax(torch.randn(5), 0) for k in svec}, feats / "synthetic_inception_probs.pt")
    features.build_knn_map(feats)
    features._write_real_labels(real, feats / "real_labels.csv")
    return {"synth": synth, "std": std, "feats": feats, "out": tmp_path / "m.csv"}


def test_knn_map_and_labels(world):
    import json
    knn = json.loads((world["feats"] / "knn_map.json").read_text())
    assert knn["A/s0.png"] == ["r0.png"]
    assert set(pd.read_csv(world["feats"] / "real_labels.csv")["class"]) == {"A", "B"}


def test_compute_metrics_end_to_end_and_resume(world):
    m = ["psnr_score", "ssim_score", "fscs_score", "realism_score", "diversity_score", "is_contribution_score"]
    csv = compute_metrics(world["synth"], world["std"], world["feats"], world["out"], m)
    df = pd.read_csv(csv).set_index("filename")
    assert list(df.target_class_label) == ["A", "A", "B", "B"]          # built by scanning <class>/<image>
    assert df.loc["A/s0.png", "psnr_score"] == 100.0                     # identical image -> mse 0
    assert df.loc["A/s0.png", "ssim_score"] == pytest.approx(1.0)
    assert (df.realism_score >= 0).all()                                 # k=3 of 3 reals: values checked in test_metrics_from_features
    assert df[m].notna().all().all()
    before = csv.stat().st_mtime_ns
    compute_metrics(world["synth"], world["std"], world["feats"], world["out"], m)   # nothing missing -> no work
    assert csv.stat().st_mtime_ns == before
