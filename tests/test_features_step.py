"""steps/features.py orchestration (skip-per-file logic, knn map, real labels) with the DINOv2/
InceptionV3 embedders monkeypatched out (no network / transformers needed to exercise the wiring)."""
import numpy as np
import pytest
import torch

pytest.importorskip("PIL")

from synth_curator.steps import features  # noqa: E402


def _img(path, seed, size=(24, 24)):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.random.default_rng(seed).integers(0, 255, (*size[::-1], 3), dtype=np.uint8)).save(path)


@pytest.fixture
def world(tmp_path):
    real_std, synth, real_src = tmp_path / "real_std", tmp_path / "synth", tmp_path / "real_src"
    for i in range(2):
        _img(real_std / f"r{i}.png", i)
        _img(real_src / "A" / f"r{i}.png", i)
    for cls in ("A", "B"):
        _img(synth / cls / "s0.png", 10)
    return {"real_std": real_std, "synth": synth, "real_src": real_src, "out": tmp_path / "feat"}


@pytest.fixture(autouse=True)
def fake_embedders(monkeypatch):
    monkeypatch.setattr(features, "_dino_embedder", lambda: (lambda imgs: (torch.randn(len(imgs), 8),)))

    def fake_inception():
        def embed(imgs):
            return torch.randn(len(imgs), 16), torch.softmax(torch.randn(len(imgs), 5), dim=1)
        return embed
    monkeypatch.setattr(features, "_inception_embedder", fake_inception)


def test_extract_features_writes_every_expected_file(world):
    out = features.extract_features(world["real_std"], world["synth"], world["out"], world["real_src"])
    for role in ("real", "synthetic"):
        for space in ("dino", "inception", "inception_probs"):
            assert (out / f"{role}_{space}.pt").exists()
    assert (out / "knn_map.json").exists()
    assert (out / "real_labels.csv").exists()


def test_extract_features_skips_everything_on_second_call(world, monkeypatch):
    features.extract_features(world["real_std"], world["synth"], world["out"], world["real_src"])

    def boom():
        raise AssertionError("embedder should not be constructed: every output already exists")

    monkeypatch.setattr(features, "_dino_embedder", boom)
    monkeypatch.setattr(features, "_inception_embedder", boom)
    out = features.extract_features(world["real_std"], world["synth"], world["out"], world["real_src"])
    assert out == world["out"]


def test_extract_features_without_real_dir_skips_real_labels(world):
    out = features.extract_features(world["real_std"], world["synth"], world["out"], real_dir=None)
    assert not (out / "real_labels.csv").exists()
