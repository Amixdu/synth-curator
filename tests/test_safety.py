"""core/safety.py - hand-computed max-similarity-to-real + the absolute threshold filter."""
import pandas as pd
import pytest
import torch

from synth_curator.core.safety import (
    apply_safety_filter,
    compute_max_similarity_to_real,
)
from synth_curator.exceptions import SynthCuratorError


@pytest.fixture
def feats(tmp_path):
    features_dir = tmp_path / "feat"
    features_dir.mkdir()
    real = {"r0.png": torch.tensor([1.0, 0.0]), "r1.png": torch.tensor([0.0, 1.0])}
    synth = {
        "A/s0.png": torch.tensor([1.0, 0.0]),        # identical to r0 -> max sim 1.0
        "A/s1.png": torch.tensor([0.6, 0.8]),        # unit vector -> sim(r0)=0.6, sim(r1)=0.8 -> max 0.8
        "B/s0.png": torch.tensor([-1.0, 0.0]),       # sim(r0)=-1, sim(r1)=0 -> max 0.0
    }
    torch.save(real, features_dir / "real_dino.pt")
    torch.save(synth, features_dir / "synthetic_dino.pt")
    return features_dir


@pytest.fixture
def df():
    return pd.DataFrame({
        "filename": ["A/s0.png", "A/s1.png", "B/s0.png"],
        "target_class_label": ["A", "A", "B"],
    })


def test_max_similarity_hand_computed(feats, df):
    sims = compute_max_similarity_to_real(df, feats)
    assert sims.tolist() == pytest.approx([1.0, 0.8, 0.0], abs=1e-6)


def test_max_similarity_uses_cache_when_it_covers_the_pool(feats, df, tmp_path):
    cache = tmp_path / "cache.csv"
    pd.DataFrame({"filename": df["filename"], "max_sim_to_real": [0.11, 0.22, 0.33]}).to_csv(cache, index=False)
    sims = compute_max_similarity_to_real(df, feats, cache_path=cache)
    assert sims.tolist() == pytest.approx([0.11, 0.22, 0.33])   # cache wins, features are never touched


def test_max_similarity_recomputes_when_cache_does_not_cover_pool(feats, df, tmp_path):
    cache = tmp_path / "cache.csv"
    pd.DataFrame({"filename": ["some/other.png"], "max_sim_to_real": [0.5]}).to_csv(cache, index=False)
    sims = compute_max_similarity_to_real(df, feats, cache_path=cache)
    assert sims.tolist() == pytest.approx([1.0, 0.8, 0.0], abs=1e-6)


def test_apply_safety_filter_drops_images_at_or_above_threshold(feats, df):
    out = apply_safety_filter(df, feats, threshold=0.9, target_distribution={"A": 1, "B": 1},
                              num_images=2)
    assert set(out["filename"]) == {"A/s1.png", "B/s0.png"}     # A/s0.png (sim=1.0 >= 0.9) is dropped


def test_apply_safety_filter_raises_when_a_class_is_starved_below_the_draw(feats, df):
    # threshold 0.5 drops A/s0.png (1.0) too, leaving class A with only 1 image; ask for 2
    with pytest.raises(SynthCuratorError, match="below the count the draw needs"):
        apply_safety_filter(df, feats, threshold=0.5, target_distribution={"A": 100, "B": 0}, num_images=2)
