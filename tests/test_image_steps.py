"""Image-touching steps on tiny generated images. Skipped when the [full] extras are absent."""
import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("PIL")
pytest.importorskip("skimage")

from PIL import Image

from synth_curator.exceptions import SynthCuratorError
from synth_curator.steps import features, standardize
from synth_curator.steps.metrics import compute_metrics


def _img(path, seed, size=(80, 60)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.random.default_rng(seed).integers(0, 255, (*size[::-1], 3), dtype=np.uint8)).save(path)


def test_flatten_name_joins_the_full_relative_path(tmp_path):
    root = tmp_path / "real"
    assert standardize.flatten_name(root / "patient_001" / "x.jpg", root) == "patient_001__x"


def test_flatten_name_of_a_flat_file_is_just_its_stem(tmp_path):
    root = tmp_path / "real"
    assert standardize.flatten_name(root / "x.jpg", root) == "x"


def test_flatten_name_drops_only_the_final_suffix(tmp_path):
    root = tmp_path / "real"
    assert standardize.flatten_name(root / "a.b" / "x.jpg", root) == "a.b__x"


def test_standardize_center_crops_resizes_and_flattens(tmp_path):
    _img(tmp_path / "real" / "A" / "x.jpg", 0)
    out = standardize.standardize(tmp_path / "real", tmp_path / "std", 32)
    assert Image.open(out / "A__x.png").size == (32, 32)


def test_standardize_same_filename_in_different_subfolders_does_not_collide(tmp_path):
    """Regression: two source images with the same bare name but different parent folders (e.g.
    per-patient subfolders) must NOT collide just because their filenames match."""
    _img(tmp_path / "real" / "patient_001" / "x.jpg", 0)
    _img(tmp_path / "real" / "patient_002" / "x.jpg", 1)
    out = standardize.standardize(tmp_path / "real", tmp_path / "std", 32)
    assert {p.name for p in out.iterdir()} == {"patient_001__x.png", "patient_002__x.png"}


def test_standardize_rejects_a_true_flattened_name_collision(tmp_path):
    """Two source images that genuinely flatten to the same name (same folder, same stem,
    different source extension) must still be caught - only the false collision is fixed."""
    _img(tmp_path / "real" / "A" / "x.jpg", 0)
    _img(tmp_path / "real" / "A" / "x.png", 1)
    with pytest.raises(SynthCuratorError, match="same name"):
        standardize.standardize(tmp_path / "real", tmp_path / "std", 32)


def test_standardize_can_rebuild_into_an_already_populated_out_dir(tmp_path):
    """Regression: os.rename() can't replace a non-empty directory, so calling standardize() a
    second time at a different resolution against the SAME out_dir used to crash."""
    _img(tmp_path / "real" / "A" / "x.jpg", 0)
    standardize.standardize(tmp_path / "real", tmp_path / "std", 32)
    out = standardize.standardize(tmp_path / "real", tmp_path / "std", 64)
    assert Image.open(out / "A__x.png").size == (64, 64)


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
    # keys must match standardize()'s actual flattened output names (real/A/r0.png -> "A__r0.png")
    rvec = {f"{cls}__r{i}.png": torch.randn(8) for i, cls in enumerate(["A", "A", "B"])}
    svec = {f"{c}/s{i}.png": torch.randn(8) for i, c in enumerate(["A", "A", "B", "B"])}
    svec["A/s0.png"] = rvec["A__r0.png"].clone()                    # identical feature -> nearest is r0
    torch.save(rvec, feats / "real_dino.pt")
    torch.save(svec, feats / "synthetic_dino.pt")
    torch.save({k: torch.softmax(torch.randn(5), 0) for k in svec}, feats / "synthetic_inception_probs.pt")
    features.build_knn_map(feats)
    features._write_real_labels(real, feats / "real_labels.csv")
    return {"synth": synth, "std": std, "feats": feats, "out": tmp_path / "m.csv"}


def test_knn_map_and_labels(world):
    import json
    knn = json.loads((world["feats"] / "knn_map.json").read_text())
    assert knn["A/s0.png"] == ["A__r0.png"]
    labels = pd.read_csv(world["feats"] / "real_labels.csv")
    assert set(labels["class"]) == {"A", "B"}
    assert set(labels["filename"]) == {"A__r0.png", "A__r1.png", "B__r2.png"}   # matches rvec's keys


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
