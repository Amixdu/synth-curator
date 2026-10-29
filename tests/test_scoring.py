import pandas as pd
import pytest

from synth_curator.core.scoring import (
    calculate_per_class_score,
    calculate_robust_bounds,
    calculate_robust_score,
    group_key,
)


def test_robust_score_is_normalized_and_clipped():
    s = pd.Series(list(range(101)) + [10_000])          # one extreme outlier
    vmin, vmax = calculate_robust_bounds(s)
    out = calculate_robust_score(s, vmin, vmax, "max")
    assert out.min() == 0 and out.max() == 1


def test_min_direction_inverts():
    s = pd.Series([1.0, 2.0, 3.0])
    assert calculate_robust_score(s, 1.0, 3.0, "min").tolist() == pytest.approx([1.0, 0.5, 0.0])


def test_constant_series_does_not_divide_by_zero():
    s = pd.Series([5.0] * 10)
    assert calculate_robust_score(s, 5.0, 5.0, "max").isna().sum() == 0


def test_group_key_joins_columns_with_pipe():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    assert group_key(df, ["a", "b"]).tolist() == ["1|x", "2|y"]


def test_per_class_score_single_instance_metric_matches_robust_score():
    df = pd.DataFrame({"target_class_label": ["A"] * 5, "psnr_score": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out = calculate_per_class_score(df.copy(), {"A": {"psnr_score": 1.0}}, None, [])
    vmin, vmax = calculate_robust_bounds(df["psnr_score"])
    expected = calculate_robust_score(df["psnr_score"], vmin, vmax, "max")
    assert out["S_Composite"].tolist() == pytest.approx(expected.tolist())
    assert out["psnr_score_norm"].tolist() == pytest.approx(expected.tolist())


def test_per_class_score_weights_two_instance_metrics():
    df = pd.DataFrame({
        "target_class_label": ["A"] * 3,
        "psnr_score": [0.0, 5.0, 10.0],       # max: normalized [0, 0.5, 1.0]
        "lpips_score": [0.0, 5.0, 10.0],      # min: normalized [1.0, 0.5, 0.0]
    })
    out = calculate_per_class_score(df.copy(), {"A": {"psnr_score": 0.5, "lpips_score": 0.5}}, None, [])
    assert out["S_Composite"].tolist() == pytest.approx([0.5, 0.5, 0.5])


def test_per_class_score_only_weights_usable_metrics_and_renormalizes():
    """A weight for a metric column absent from df is dropped; the remaining weight is renormalized."""
    df = pd.DataFrame({"target_class_label": ["A"] * 3, "psnr_score": [0.0, 5.0, 10.0]})
    out = calculate_per_class_score(df.copy(), {"A": {"psnr_score": 0.5, "ssim_score": 0.5}}, None, [])
    vmin, vmax = calculate_robust_bounds(df["psnr_score"])
    expected = calculate_robust_score(df["psnr_score"], vmin, vmax, "max")
    assert out["S_Composite"].tolist() == pytest.approx(expected.tolist())   # psnr_score alone gets 100% after renorm


def test_per_class_score_includes_group_distributional_term():
    df = pd.DataFrame({"target_class_label": ["A", "A"], "grp": ["g1", "g2"], "psnr_score": [0.0, 10.0]})
    group_scores = pd.DataFrame({"kid_dino": [1.0, 0.0]}, index=["g1", "g2"])
    out = calculate_per_class_score(df.copy(), {"A": {"psnr_score": 0.5, "kid_dino": 0.5}}, group_scores, ["grp"])
    # row0: group g1 -> kid term 1.0, psnr normalized 0.0 -> 0.5*0 + 0.5*1 = 0.5
    # row1: group g2 -> kid term 0.0, psnr normalized 1.0 -> 0.5*1 + 0.5*0 = 0.5
    assert out["S_Composite"].tolist() == pytest.approx([0.5, 0.5])


def test_per_class_score_zero_usable_weight_leaves_score_zero():
    df = pd.DataFrame({"target_class_label": ["A"] * 2, "psnr_score": [0.0, 10.0]})
    out = calculate_per_class_score(df.copy(), {"A": {"ssim_score": 1.0}}, None, [])
    assert out["S_Composite"].tolist() == [0.0, 0.0]


def test_per_class_score_only_scores_rows_of_weighted_classes():
    df = pd.DataFrame({"target_class_label": ["A", "B"], "psnr_score": [0.0, 10.0]})
    out = calculate_per_class_score(df.copy(), {"A": {"psnr_score": 1.0}}, None, [])
    assert out.loc[out.target_class_label == "B", "S_Composite"].tolist() == [0.0]
