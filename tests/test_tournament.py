"""steps/tournament.py - training orchestration, caching and prediction validation."""
import pandas as pd
import pytest

from synth_curator.exceptions import SynthCuratorError
from synth_curator.model_interface import model_from_function
from synth_curator.steps import tournament
from tests.conftest import cli_fake_failing_model_factory, cli_fake_model_factory, cli_fake_recording_model_factory


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

    result1 = tournament.run(metric_sets, model_from_function(fn), out_dir)
    assert calls == [0]
    assert (out_dir / "lpips_score_seed0_predictions.csv").exists()

    result2 = tournament.run(metric_sets, model_from_function(fn), out_dir)
    assert calls == [0]     # second run: cache hit, model never invoked again
    pd.testing.assert_frame_equal(
        result1["lpips_score"].reset_index(drop=True), result2["lpips_score"].reset_index(drop=True))


def test_run_propagates_synth_curator_error_for_bad_predictions(tmp_path):
    def fn(manifest, seed):
        return pd.DataFrame({"true_label": ["A"]})    # missing unique_image_id, predicted_label

    manifest = _manifest(tmp_path)
    with pytest.raises(SynthCuratorError, match="missing columns"):
        tournament.run({"lpips_score": {0: manifest}}, model_from_function(fn), tmp_path / "tournament")


def test_run_logs_when_reusing_a_cached_prediction(tmp_path, caplog):
    def fn(manifest, seed):
        return pd.DataFrame({"unique_image_id": ["a.png"], "true_label": ["A"], "predicted_label": ["A"]})

    manifest = _manifest(tmp_path)
    metric_sets = {"lpips_score": {0: manifest}}
    out_dir = tmp_path / "tournament"
    tournament.run(metric_sets, model_from_function(fn), out_dir)

    with caplog.at_level("INFO"):
        tournament.run(metric_sets, model_from_function(fn), out_dir)
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

    tournament.run(metric_sets, model_from_function(fn), out_dir)
    assert calls == [0]

    tournament.run(metric_sets, model_from_function(fn), out_dir, force_retrain=True)
    assert calls == [0, 0]   # retrained despite the existing cache file

    tournament.run(metric_sets, model_from_function(fn), out_dir)
    assert calls == [0, 0]   # back to normal: the freshly-overwritten cache is reused again


def test_run_trains_once_per_seed_per_metric_set(tmp_path):
    calls = []

    def fn(manifest, seed):
        calls.append((manifest.name, seed))
        return pd.DataFrame({"unique_image_id": ["a.png"], "true_label": ["A"], "predicted_label": ["A"]})

    m1, m2 = _manifest(tmp_path, "m1.csv"), _manifest(tmp_path, "m2.csv")
    metric_sets = {"lpips_score": {0: m1, 1: m1}, "random_control": {0: m2, 1: m2}}
    result = tournament.run(metric_sets, model_from_function(fn), tmp_path / "tournament")
    assert sorted(calls) == [("m1.csv", 0), ("m1.csv", 1), ("m2.csv", 0), ("m2.csv", 1)]
    assert set(result["lpips_score"]["seed"]) == {0, 1}
    assert len(result["lpips_score"]) == 2


def test_run_parallel_trains_every_job_and_caches_predictions(tmp_path):
    m1, m2 = _manifest(tmp_path, "m1.csv"), _manifest(tmp_path, "m2.csv")
    metric_sets = {"lpips_score": {0: m1, 1: m1}, "random_control": {0: m2, 1: m2}}
    out_dir = tmp_path / "tournament"

    result = tournament.run_parallel(metric_sets, cli_fake_model_factory, [0, 1], out_dir)

    assert set(result["lpips_score"]["seed"]) == {0, 1}
    assert len(result["lpips_score"]) == 2
    for name, seed in (("lpips_score", 0), ("lpips_score", 1), ("random_control", 0), ("random_control", 1)):
        assert (out_dir / f"{name}_seed{seed}_predictions.csv").exists()


def test_run_parallel_reuses_cached_predictions_on_a_second_call(tmp_path):
    manifest = _manifest(tmp_path)
    metric_sets = {"lpips_score": {0: manifest}}
    out_dir = tmp_path / "tournament"

    result1 = tournament.run_parallel(metric_sets, cli_fake_model_factory, [0], out_dir)
    cache = out_dir / "lpips_score_seed0_predictions.csv"
    written_at = cache.stat().st_mtime_ns

    result2 = tournament.run_parallel(metric_sets, cli_fake_model_factory, [0], out_dir)
    assert cache.stat().st_mtime_ns == written_at   # not rewritten: cache hit, no worker spawned
    pd.testing.assert_frame_equal(
        result1["lpips_score"].reset_index(drop=True), result2["lpips_score"].reset_index(drop=True))


def test_run_parallel_starts_one_worker_per_gpu_id(tmp_path, monkeypatch):
    """Every id in gpu_ids gets its own worker process (each builds its model up front, before
    pulling any job) - independent of how job-count fairness happens to land, which real wall-clock
    process-startup timing makes nondeterministic to assert on directly."""
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()
    monkeypatch.setenv("SYNTH_CURATOR_TEST_MARKER_DIR", str(marker_dir))

    manifest = _manifest(tmp_path)
    tournament.run_parallel({"lpips_score": {0: manifest}}, cli_fake_recording_model_factory, [0, 1],
                            tmp_path / "tournament")

    assert {p.name for p in marker_dir.iterdir()} == {"0", "1"}


def test_run_parallel_raises_synth_curator_error_when_a_worker_job_fails(tmp_path):
    manifest = _manifest(tmp_path)
    with pytest.raises(SynthCuratorError, match="boom"):
        tournament.run_parallel(
            {"lpips_score": {0: manifest}}, cli_fake_failing_model_factory, [0], tmp_path / "tournament")
