"""core/block_search.py - the block_search=True path (off by default, but must still work)."""
import pandas as pd
import pytest
import torch

from synth_curator.core.block_search import (
    evaluate_distributional_utility_per_class, select_optimal_block_per_class, write_class_log,
)


@pytest.fixture
def blocks_and_features(tmp_path):
    features_dir = tmp_path / "feat"
    features_dir.mkdir()

    real = {"r0.png": torch.tensor([0.0, 0.0]), "r1.png": torch.tensor([0.0, 1.0])}
    synth = {
        "b1/A/0.png": torch.tensor([0.0, 0.0]),      # block 1: identical to real -> tiny KID
        "b1/A/1.png": torch.tensor([0.0, 1.0]),
        "b2/A/0.png": torch.tensor([50.0, 50.0]),    # block 2: far from real -> large KID
        "b2/A/1.png": torch.tensor([50.0, 51.0]),
    }
    torch.save(real, features_dir / "real_dino.pt")
    torch.save(synth, features_dir / "synthetic_dino.pt")
    pd.DataFrame({"filename": ["r0.png", "r1.png"], "class": ["A", "A"]}).to_csv(
        features_dir / "real_labels.csv", index=False)

    def block(idx, prefix):
        df = pd.DataFrame({
            "filename": [f"{prefix}/A/0.png", f"{prefix}/A/1.png"],
            "target_class_label": ["A", "A"],
            "S_Composite": [0.5, 0.5],
        })
        return {"idx": idx, "df": df}

    blocks = [block(1, "b1"), block(2, "b2")]
    weights = {"A": {"kid_dino": 1.0}, "B": {}}    # B carries no distributional weight -> skipped
    return blocks, weights, features_dir


def test_close_block_scores_higher_than_far_block(blocks_and_features):
    blocks, weights, features_dir = blocks_and_features
    blocks = evaluate_distributional_utility_per_class(blocks, weights, features_dir, ["A", "B"],
                                                        min_samples={"kid": 2, "fid": 2, "swd": 2})
    by_idx = {b["idx"]: b for b in blocks}
    assert by_idx[1]["U_dist"] > by_idx[2]["U_dist"]
    assert by_idx[1]["U_dist_by_class"]["A"] == pytest.approx(1.0, abs=1e-6)   # best raw -> normalized 1.0
    assert by_idx[2]["U_dist_by_class"]["A"] == pytest.approx(0.0, abs=1e-6)
    assert "B" not in by_idx[1]["U_dist_by_class"]           # class B had no distributional weight


def test_select_optimal_block_per_class_picks_the_argmax(blocks_and_features):
    blocks, weights, features_dir = blocks_and_features
    blocks = evaluate_distributional_utility_per_class(blocks, weights, features_dir, ["A", "B"],
                                                        min_samples={"kid": 2, "fid": 2, "swd": 2})
    optimal_df, blocks = select_optimal_block_per_class(blocks)
    assert set(optimal_df["filename"]) == {"b1/A/0.png", "b1/A/1.png"}
    for b in blocks:
        assert b["S_Final"] == b["U_dist"]
        assert b["U_inst"] == pytest.approx(0.5)   # mean of S_Composite


def test_no_distributional_metrics_gives_zero_u_dist_for_every_block(blocks_and_features):
    blocks, _, features_dir = blocks_and_features
    blocks = evaluate_distributional_utility_per_class(blocks, {"A": {}, "B": {}}, features_dir, ["A", "B"])
    assert all(b["U_dist"] == 0.0 for b in blocks)
    assert all(b["U_dist_by_class"] == {} for b in blocks)


def test_class_skipped_when_pool_below_min_samples(blocks_and_features):
    blocks, weights, features_dir = blocks_and_features
    # only 2 real/synth samples available; require 256 (the KID default) -> class A must be skipped
    blocks = evaluate_distributional_utility_per_class(blocks, weights, features_dir, ["A", "B"])
    assert all(b["U_dist"] == 0.0 and b["U_dist_by_class"] == {} for b in blocks)


def test_write_class_log_has_one_row_per_block_per_class_per_metric(blocks_and_features, tmp_path):
    blocks, weights, features_dir = blocks_and_features
    blocks = evaluate_distributional_utility_per_class(blocks, weights, features_dir, ["A", "B"],
                                                        min_samples={"kid": 2, "fid": 2, "swd": 2})
    path = tmp_path / "class_log.csv"
    write_class_log(blocks, path)
    df = pd.read_csv(path)
    assert len(df) == 2       # 2 blocks x 1 class (A) x 1 metric (kid_dino)
    assert set(df["metric"]) == {"kid_dino"}
    assert set(df["target_class_label"]) == {"A"}
    assert {"Block_ID", "raw", "normalized", "U_dist_class", "U_dist_block"} <= set(df.columns)


def test_write_class_log_writes_nothing_when_no_rows(tmp_path):
    path = tmp_path / "class_log.csv"
    write_class_log([{"idx": 1, "dist_detail": []}], path)
    assert not path.exists()
