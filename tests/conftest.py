"""Shared fixtures/helpers. Tiny CPU-only images and features. No network, no GPU, no heavy deps."""
import numpy as np
import pandas as pd
import pytest
import torch

from synth_curator import model_from_function
from synth_curator.steps import features, standardize

CLASSES = ["A", "B", "C"]
TARGET = {"A": 50.0, "B": 30.0, "C": 20.0}


def make_image(path, seed, size=(32, 32)):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.random.default_rng(seed).integers(0, 255, (*size[::-1], 3), dtype=np.uint8)).save(path)


def build_image_pool(root, real_per_class=2, synthetic_per_class=None):
    """<real_dir>/<class>/*.png and <synthetic_dir>/<class>/*.png for CLASSES.

    Real filenames are globally unique across classes: standardize() flattens all real images
    into one directory keyed by filename stem, so per-class counters would collide.
    """
    synthetic_per_class = synthetic_per_class or {"A": 12, "B": 8, "C": 6}
    real_dir, synthetic_dir = root / "real", root / "synthetic"
    seed = 0
    real_idx = 0
    for cls in CLASSES:
        for _ in range(real_per_class):
            make_image(real_dir / cls / f"r{real_idx}.png", seed, (32, 32)); seed += 1; real_idx += 1
        for i in range(synthetic_per_class[cls]):
            make_image(synthetic_dir / cls / f"s{i}.png", seed, (32, 32)); seed += 1
    return real_dir, synthetic_dir


def fake_extract_features(std_dir, synthetic_dir, out_dir, real_dir):
    """Stand-in for steps.features.extract_features: random tensors instead of real DINOv2/InceptionV3
    (which need network + heavy deps not available here). Everything else (knn map, real labels) is
    the real, network-free code - only the model inference is faked."""
    out_dir.mkdir(parents=True, exist_ok=True)
    real_names = [p.name for p in standardize.list_images(std_dir)]
    synthetic_names = [str(p.relative_to(synthetic_dir)) for p in standardize.list_images(synthetic_dir)]
    for role, names in (("real", real_names), ("synthetic", synthetic_names)):
        torch.save({n: torch.randn(8) for n in names}, out_dir / f"{role}_dino.pt")
        torch.save({n: torch.randn(8) for n in names}, out_dir / f"{role}_inception.pt")
        torch.save({n: torch.softmax(torch.randn(5), 0) for n in names}, out_dir / f"{role}_inception_probs.pt")
    features.build_knn_map(out_dir)
    features._write_real_labels(real_dir, out_dir / "real_labels.csv")
    return out_dir


def fake_model(accuracy_for=lambda manifest: 0.8):
    """A Model whose accuracy can vary by which metric_set's manifest it's training on."""
    def train_and_evaluate(manifest, seed):
        rng = np.random.default_rng(seed)
        m = pd.read_csv(manifest)
        acc = accuracy_for(manifest)
        t = m["target_class"].values
        p = np.where(rng.random(len(t)) < acc, t, rng.choice(CLASSES, len(t)))
        return pd.DataFrame({"unique_image_id": m["safe_filename"], "true_label": t, "predicted_label": p})
    return model_from_function(train_and_evaluate)


def cli_fake_trainer():
    """Importable via a dotted `trainer:` config path (tests.conftest:cli_fake_trainer)."""
    return fake_model()


class _WithProduction:
    """Wraps a Model with a train_production(manifest) -> dict, for testing the CLI's
    --train-production flag and Python-side ProductionModel behavior."""

    def __init__(self, base):
        self._base = base

    def train_and_evaluate(self, manifest, seed):
        return self._base.train_and_evaluate(manifest, seed)

    def train_production(self, manifest):
        return {"trained_on": str(manifest)}


def cli_fake_trainer_with_production():
    """Importable via a dotted `trainer:` config path (tests.conftest:cli_fake_trainer_with_production)."""
    return _WithProduction(fake_model())


def cli_fake_scoring(true_labels, pred_labels, classes):
    """Importable via a dotted `weights.scoring:` config path (tests.conftest:cli_fake_scoring)."""
    return {c: 0.0 for c in classes}


@pytest.fixture
def image_pool(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    return {"real_dir": real_dir, "synthetic_dir": synthetic_dir, "dir": tmp_path}
