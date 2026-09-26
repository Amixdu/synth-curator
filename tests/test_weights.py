import numpy as np
import pandas as pd
import pytest

from synth_curator.exceptions import WeightsError
from synth_curator.steps.weights import derive_weights, fbeta_per_class

CLASSES = ["A", "B"]


def test_f2_hand_computed():
    # class A: TP=1 FN=1 FP=1 -> 5*1 / (5*1 + 4*1 + 1) = 0.5
    out = fbeta_per_class(["A", "A", "B"], ["A", "B", "A"], CLASSES)
    assert out["A"] == pytest.approx(0.5)


def test_int_labels_index_sorted_classes():
    assert fbeta_per_class([0, 1], [0, 1], CLASSES) == {"A": 1.0, "B": 1.0}


def test_fbeta_beta_is_configurable():
    # class A: TP=1 FN=1 FP=1. F1 = 2*1 / (2*1 + 1 + 1) = 0.5; F2 (default) = 0.5 too by
    # coincidence at this beta, so use a case where F1 and F2 clearly diverge: TP=1 FN=2 FP=0.
    out_f1 = fbeta_per_class(["A", "A", "A"], ["A", "B", "B"], CLASSES, beta=1.0)
    out_f2 = fbeta_per_class(["A", "A", "A"], ["A", "B", "B"], CLASSES, beta=2.0)
    assert out_f1["A"] == pytest.approx((2 * 1) / (2 * 1 + 2 + 0))       # F1 = 0.5
    assert out_f2["A"] == pytest.approx((5 * 1) / (5 * 1 + 4 * 2 + 0))   # F2 = 0.3846...
    assert out_f1["A"] != pytest.approx(out_f2["A"])


def _preds(accuracy):
    rng = np.random.default_rng(0)
    t = rng.choice(CLASSES, 400)
    p = np.where(rng.random(400) < accuracy, t, rng.choice(CLASSES, 400))
    return pd.DataFrame({"true_label": t, "predicted_label": p})


def test_weights_sum_to_one_and_random_anchored_zeroes_losers():
    preds = {"lpips_score": _preds(0.95), "psnr_score": _preds(0.3), "random_control": _preds(0.6)}
    w = derive_weights(preds, CLASSES, formula="random_anchored", dist_formula=None)
    for cls in CLASSES:
        assert sum(w[cls].values()) == pytest.approx(1.0)
        assert w[cls]["psnr_score"] == 0.0 and w[cls]["lpips_score"] == 1.0


def test_worst_anchored_never_zeroes():
    preds = {"lpips_score": _preds(0.95), "psnr_score": _preds(0.3)}
    w = derive_weights(preds, CLASSES, formula="worst_anchored", dist_formula=None)
    assert all(v > 0 for v in w["A"].values())


def test_random_anchored_requires_random_arm():
    with pytest.raises(WeightsError):
        derive_weights({"lpips_score": _preds(0.9)}, CLASSES, formula="random_anchored")


def test_unknown_formula_raises_weights_error():
    with pytest.raises(WeightsError, match="unknown formula"):
        derive_weights({"lpips_score": _preds(0.9), "random_control": _preds(0.5)},
                       CLASSES, formula="not_a_formula")


def test_dist_formula_redistributes_only_the_distributional_share():
    preds = {
        "lpips_score": _preds(0.9),    # instance
        "psnr_score": _preds(0.3),     # instance (worst -> smallest but nonzero share)
        "kid_dino": _preds(0.9),       # distributional, beats random
        "swd_dino": _preds(0.2),       # distributional, loses to random -> zeroed by the re-split
        "random_control": _preds(0.6),
    }
    baseline = derive_weights(preds, CLASSES, formula="worst_anchored", dist_formula=None)
    out = derive_weights(preds, CLASSES, formula="worst_anchored", dist_formula="random_anchored")
    for cls in CLASSES:
        # instance weights are untouched by the distributional re-split
        assert out[cls]["lpips_score"] == pytest.approx(baseline[cls]["lpips_score"])
        assert out[cls]["psnr_score"] == pytest.approx(baseline[cls]["psnr_score"])
        # the distributional SHARE (sum of kid+swd) is preserved by the re-split...
        base_share = baseline[cls]["kid_dino"] + baseline[cls]["swd_dino"]
        new_share = out[cls]["kid_dino"] + out[cls]["swd_dino"]
        assert new_share == pytest.approx(base_share)
        # ...but within that share, swd_dino (loses to random) is zeroed and kid_dino gets it all
        assert out[cls]["swd_dino"] == pytest.approx(0.0)
        assert out[cls]["kid_dino"] == pytest.approx(base_share)
        assert sum(out[cls].values()) == pytest.approx(1.0)


def test_active_metrics_filters_which_metric_sets_are_used():
    preds = {"lpips_score": _preds(0.9), "psnr_score": _preds(0.1), "random_control": _preds(0.5)}
    w = derive_weights(preds, CLASSES, formula="worst_anchored", dist_formula=None,
                       active_metrics=["lpips_score"])
    assert set(w["A"]) == {"lpips_score"}


def _overall_accuracy(true_labels, pred_labels, classes):
    """A user-written scoring function, standing in for whatever custom metric someone plugs into
    `scoring` - accuracy has no real per-class variant, so this broadcasts one number to every
    class, same as the library would if it shipped one (it doesn't - this is on the caller)."""
    import numpy as np
    t = np.asarray(true_labels) == np.asarray(pred_labels)
    return {c: float(t.mean()) for c in classes}


def test_derive_weights_scoring_override_replaces_fbeta_and_ignores_beta():
    preds = {"lpips_score": _preds(0.9), "psnr_score": _preds(0.3), "random_control": _preds(0.6)}
    w_scoring = derive_weights(preds, CLASSES, formula="worst_anchored", dist_formula=None,
                               scoring=_overall_accuracy, beta=999)   # beta must be ignored here
    w_direct = derive_weights(preds, CLASSES, formula="worst_anchored", dist_formula=None,
                              scoring=_overall_accuracy)
    assert w_scoring == w_direct
    # _overall_accuracy gives every class the same score per metric set, so worst_anchored
    # (score - min + 1e-4) must give both classes the exact same weights for every metric.
    assert w_scoring["A"] == w_scoring["B"]


def test_derive_weights_beta_changes_the_result():
    # two metrics whose per-class F1 ranking flips their F2 ranking would be a stronger test, but
    # even with _preds' random imbalance, a different beta must at least change the raw weights.
    preds = {"lpips_score": _preds(0.9), "psnr_score": _preds(0.3), "random_control": _preds(0.6)}
    w_f2 = derive_weights(preds, CLASSES, formula="worst_anchored", dist_formula=None, beta=2.0)
    w_f1 = derive_weights(preds, CLASSES, formula="worst_anchored", dist_formula=None, beta=1.0)
    assert w_f2 != w_f1
    assert sum(w_f1["A"].values()) == pytest.approx(1.0)


def test_seeds_are_averaged():
    a, b = _preds(0.9), _preds(0.3)
    stacked = pd.concat([a.assign(seed=0), b.assign(seed=1)])
    w = derive_weights({"lpips_score": stacked, "psnr_score": _preds(0.6)}, CLASSES,
                       formula="worst_anchored", dist_formula=None)
    assert sum(w["A"].values()) == pytest.approx(1.0)
