"""core/groups.py - group distributional scores computed live from feature tensors."""
import pandas as pd
import pytest
import torch

from synth_curator.core.groups import compute_group_raw, compute_group_scores


@pytest.fixture
def feats(tmp_path):
    features_dir = tmp_path / "feat"
    features_dir.mkdir()
    real = {f"r{i}.png": torch.tensor([float(i), float(i)]) for i in range(6)}   # 0..5 on the diagonal
    synth = {}
    for i in range(6):
        synth[f"close/s{i}.png"] = torch.tensor([float(i), float(i)])           # identical to real -> ~0 distance
        synth[f"far/s{i}.png"] = torch.tensor([float(i) + 100.0, float(i) + 100.0])  # far away
    torch.save(real, features_dir / "real_dino.pt")
    torch.save(synth, features_dir / "synthetic_dino.pt")
    return features_dir


@pytest.fixture
def df():
    rows = []
    for g in ("close", "far"):
        for i in range(6):
            rows.append({"filename": f"{g}/s{i}.png", "grp": g})
    return pd.DataFrame(rows)


def test_compute_group_raw_close_group_scores_lower_than_far_group(feats, df):
    raw = compute_group_raw(df, feats, ["grp"], ["kid_dino", "fid_dino", "swd_dino"])
    assert list(raw.index) == ["close", "far"]
    for m in ("kid_dino", "fid_dino", "swd_dino"):
        assert raw.loc["close", m] < raw.loc["far", m]
        assert raw.loc["close", m] == pytest.approx(0.0, abs=1e-3)


def test_compute_group_scores_normalizes_so_best_raw_gets_score_one(feats, df):
    scores = compute_group_scores(df, feats, ["grp"], ["kid_dino"])
    assert scores.loc["close", "kid_dino"] == pytest.approx(1.0)     # lowest raw (best) -> 1.0
    assert scores.loc["far", "kid_dino"] == pytest.approx(0.0)       # highest raw (worst) -> 0.0


def test_compute_group_scores_caches_to_disk_and_reuses(feats, df, tmp_path):
    cache = tmp_path / "group_scores.csv"
    first = compute_group_scores(df, feats, ["grp"], ["kid_dino"], cache_path=cache)
    assert cache.exists()

    # corrupt the underlying features; a cache hit must not touch them
    (feats / "synthetic_dino.pt").unlink()
    second = compute_group_scores(df, feats, ["grp"], ["kid_dino"], cache_path=cache)
    pd.testing.assert_frame_equal(first, second)


def test_compute_group_scores_recomputes_when_cache_lacks_requested_metric(feats, df, tmp_path):
    cache = tmp_path / "group_scores.csv"
    compute_group_scores(df, feats, ["grp"], ["kid_dino"], cache_path=cache)
    out = compute_group_scores(df, feats, ["grp"], ["kid_dino", "fid_dino"], cache_path=cache)
    assert {"kid_dino", "fid_dino"}.issubset(out.columns)
