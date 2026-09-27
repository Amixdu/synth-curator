"""steps/tournament.py - training orchestration, caching and prediction validation."""
import pandas as pd
import pytest

from synth_curator.exceptions import SynthCuratorError
from synth_curator.model_interface import model_from_function
from synth_curator.steps import tournament


def _manifest(tmp_path, name="m.csv"):
    path = tmp_path / name
    pd.DataFrame({"source_path": ["a.png"], "target_class": ["A"], "safe_filename": ["a.png"]}).to_csv(
        path, index=False)
    return path


def test_run_caches_predictions_and_does_not_retrain_on_second_call(tmp_path):
    calls = []

    def fn(manifest, seed):
        calls.append(seed)
        return pd.DataFrame({"unique_image_id": ["a.png"], "true_label": ["A"], "predicted_label": ["A"]})

    manifest = _manifest(tmp_path)
    metric_sets = {"lpips_score": {0: manifest}}
    out_dir = tmp_path / "tournament"

    result1 = tournament.run(metric_sets, model_from_function(fn), out_dir, [0])
    assert calls == [0]
    assert (out_dir / "lpips_score_seed0_predictions.csv").exists()

    result2 = tournament.run(metric_sets, model_from_function(fn), out_dir, [0])
    assert calls == [0]     # second run: cache hit, model never invoked again
    pd.testing.assert_frame_equal(
        result1["lpips_score"].reset_index(drop=True), result2["lpips_score"].reset_index(drop=True))


def test_run_propagates_synth_curator_error_for_bad_predictions(tmp_path):
    def fn(manifest, seed):
        return pd.DataFrame({"true_label": ["A"]})    # missing unique_image_id, predicted_label

    manifest = _manifest(tmp_path)
    with pytest.raises(SynthCuratorError, match="missing columns"):
        tournament.run({"lpips_score": {0: manifest}}, model_from_function(fn), tmp_path / "tournament", [0])


def test_run_logs_when_reusing_a_cached_prediction(tmp_path, caplog):
    def fn(manifest, seed):
        return pd.DataFrame({"unique_image_id": ["a.png"], "true_label": ["A"], "predicted_label": ["A"]})

    manifest = _manifest(tmp_path)
    metric_sets = {"lpips_score": {0: manifest}}
    out_dir = tmp_path / "tournament"
    tournament.run(metric_sets, model_from_function(fn), out_dir, [0])

    with caplog.at_level("INFO"):
        tournament.run(metric_sets, model_from_function(fn), out_dir, [0])
    assert "reusing cached predictions" in caplog.text
    assert "lpips_score" in caplog.text


def test_force_retrain_bypasses_the_cache(tmp_path):
    calls = []

    def fn(manifest, seed):
        calls.append(seed)
        return pd.DataFrame({"unique_image_id": ["a.png"], "true_label": ["A"], "predicted_label": ["A"]})

    manifest = _manifest(tmp_path)
    metric_sets = {"lpips_score": {0: manifest}}
    out_dir = tmp_path / "tournament"

    tournament.run(metric_sets, model_from_function(fn), out_dir, [0])
    assert calls == [0]

    tournament.run(metric_sets, model_from_function(fn), out_dir, [0], force_retrain=True)
    assert calls == [0, 0]   # retrained despite the existing cache file

    tournament.run(metric_sets, model_from_function(fn), out_dir, [0])
    assert calls == [0, 0]   # back to normal: the freshly-overwritten cache is reused again


def test_run_trains_once_per_seed_per_metric_set(tmp_path):
    calls = []

    def fn(manifest, seed):
        calls.append((manifest.name, seed))
        return pd.DataFrame({"unique_image_id": ["a.png"], "true_label": ["A"], "predicted_label": ["A"]})

    m1, m2 = _manifest(tmp_path, "m1.csv"), _manifest(tmp_path, "m2.csv")
    metric_sets = {"lpips_score": {0: m1, 1: m1}, "random_control": {0: m2, 1: m2}}
    result = tournament.run(metric_sets, model_from_function(fn), tmp_path / "tournament", [0, 1])
    assert sorted(calls) == [("m1.csv", 0), ("m1.csv", 1), ("m2.csv", 0), ("m2.csv", 1)]
    assert set(result["lpips_score"]["seed"]) == {0, 1}
    assert len(result["lpips_score"]) == 2
