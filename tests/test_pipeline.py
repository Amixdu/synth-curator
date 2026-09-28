"""End-to-end on real (tiny) images: the full linear path, steps 1-7, with a fake model.

Feature extraction (DINOv2/InceptionV3) needs network + heavy deps not available here, so it's
replaced with `conftest.fake_extract_features` - random tensors instead of real embeddings, but
everything else (standardize, metrics, curate_metric_sets, tournament, weights, curate) is real.
"""
import json

import pandas as pd
import pytest

from synth_curator import SynthCurator, model_from_function
from synth_curator.exceptions import SynthCuratorError

from .conftest import TARGET, build_image_pool, build_weights_bundle, fake_extract_features, fake_model, make_image

pytest.importorskip("PIL")

METRICS = ["fscs_score", "realism_score", "diversity_score", "is_contribution_score"]  # feature-based only
KW = dict(metrics=METRICS, dist_group_by=[], num_images=15, safety_max_similarity_threshold=None)


def _make(tmp_path, workdir="run", **overrides):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    sc = SynthCurator(tmp_path / workdir, TARGET, model=fake_model(), synthetic_dir=synthetic_dir,
                      real_dir=real_dir, resolution=32, **{**KW, **overrides})
    std_dir = sc.standardize()
    fake_extract_features(std_dir, synthetic_dir, sc.features_dir, real_dir)
    return sc


def test_full_path_runs_linearly_and_curates(tmp_path):
    sc = _make(tmp_path, random_seeds_per_metric_set=2)
    manifest = sc.run()

    w = json.loads((tmp_path / "run" / "weights.json").read_text())
    assert set(w) == set(TARGET)
    assert all(abs(sum(cw.values()) - 1) < 1e-9 or sum(cw.values()) == 0 for cw in w.values())

    df = pd.read_csv(manifest)
    assert len(df) == 15 and df.target_class.value_counts().to_dict() == {"A": 8, "B": 4, "C": 3}
    assert list(df.columns) == ["source_path", "target_class", "safe_filename"]

    used = json.loads((manifest.parent / "weights_used_seed0.json").read_text())
    assert used["algorithm"] == "v15"

    # METRICS is instance-only (feature-based) - deterministic, so each trains once regardless of
    # random_seeds_per_metric_set; only random_control (randomized) actually repeats, x2 seeds here.
    preds = list((tmp_path / "run" / "tournament").glob("*_predictions.csv"))
    assert len(preds) == len(METRICS) * 1 + 1 * 2


def test_tournament_resumes_from_cache(tmp_path):
    calls = []

    def counting(manifest, seed):
        calls.append(seed)
        return fake_model().train_and_evaluate(manifest, seed)

    real_dir, synthetic_dir = build_image_pool(tmp_path)

    def make():
        sc = SynthCurator(tmp_path / "run", TARGET, model=model_from_function(counting),
                          synthetic_dir=synthetic_dir, real_dir=real_dir, resolution=32, **KW)
        std_dir = sc.standardize()
        fake_extract_features(std_dir, synthetic_dir, sc.features_dir, real_dir)
        return sc

    sc = make()
    metric_sets = sc.curate_metric_sets(sc.compute_metrics(sc.standardize(), sc.features_dir), sc.features_dir)
    predictions = sc.run_tournament(metric_sets)
    sc.derive_weights(predictions)
    first = len(calls)

    sc2 = make()
    metric_sets2 = sc2.curate_metric_sets(sc2.compute_metrics(sc2.standardize(), sc2.features_dir), sc2.features_dir)
    predictions2 = sc2.run_tournament(metric_sets2)
    sc2.derive_weights(predictions2)
    assert first == 5 and len(calls) == first          # 4 metrics + random, second run trains nothing


def test_model_is_required(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    with pytest.raises(SynthCuratorError, match="a model is required"):
        SynthCurator(tmp_path / "r", TARGET, synthetic_dir=synthetic_dir, real_dir=real_dir, resolution=32, **KW)


def test_from_config_full_path(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"""
workdir: {tmp_path / 'run'}
target_distribution: {{A: 50, B: 30, C: 20}}
inputs: {{synthetic_dir: {synthetic_dir}, real_dir: {real_dir}, resolution: 32}}
metrics: {{instance: {METRICS}}}
curation: {{num_images: 15, dist_group_by: [], safety_max_similarity_threshold: null}}
""")
    sc = SynthCurator.from_config(cfg, model=fake_model())
    std_dir = sc.standardize()
    fake_extract_features(std_dir, synthetic_dir, sc.features_dir, real_dir)
    assert len(pd.read_csv(sc.run())) == 15


def test_config_model_block_wins_over_python_model_kwarg(tmp_path):
    """from_config: the config's own model: block wins if present, even when a Python model= is
    also passed. Proven by giving Python a model that explodes if it's ever actually used."""
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"""
workdir: {tmp_path / 'run'}
target_distribution: {{A: 50, B: 30, C: 20}}
inputs: {{synthetic_dir: {synthetic_dir}, real_dir: {real_dir}, resolution: 32}}
model: {{trainer: tests.conftest:cli_fake_trainer}}
metrics: {{instance: {METRICS}}}
curation: {{num_images: 15, dist_group_by: [], safety_max_similarity_threshold: null}}
""")

    def exploding(manifest, seed):
        raise AssertionError("the Python model= kwarg must not be used: the config's model: block wins")

    sc = SynthCurator.from_config(cfg, model=model_from_function(exploding))
    std_dir = sc.standardize()
    fake_extract_features(std_dir, synthetic_dir, sc.features_dir, real_dir)
    assert len(pd.read_csv(sc.run())) == 15   # would have raised AssertionError if the wrong model ran


def test_from_config_missing_required_inputs_is_a_clear_error(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"""
workdir: {tmp_path / 'run'}
target_distribution: {{A: 50, B: 30, C: 20}}
inputs: {{synthetic_dir: {synthetic_dir}, real_dir: {real_dir}}}
model: {{trainer: tests.conftest:cli_fake_trainer}}
""")
    with pytest.raises(SynthCuratorError, match="inputs.synthetic_dir.*inputs.real_dir.*inputs.resolution"):
        SynthCurator.from_config(cfg)


def test_from_config_missing_workdir_is_a_clear_error_not_a_keyerror(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"""
target_distribution: {{A: 50, B: 30, C: 20}}
inputs: {{synthetic_dir: {synthetic_dir}, real_dir: {real_dir}, resolution: 32}}
model: {{trainer: tests.conftest:cli_fake_trainer}}
""")
    with pytest.raises(SynthCuratorError, match="workdir") as excinfo:
        SynthCurator.from_config(cfg)
    assert not isinstance(excinfo.value, KeyError)


def test_from_config_missing_target_distribution_is_a_clear_error_not_a_keyerror(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"""
workdir: {tmp_path / 'run'}
inputs: {{synthetic_dir: {synthetic_dir}, real_dir: {real_dir}, resolution: 32}}
model: {{trainer: tests.conftest:cli_fake_trainer}}
""")
    with pytest.raises(SynthCuratorError, match="target_distribution") as excinfo:
        SynthCurator.from_config(cfg)
    assert not isinstance(excinfo.value, KeyError)


def test_from_config_model_block_without_trainer_is_a_clear_error_not_a_keyerror(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"""
workdir: {tmp_path / 'run'}
target_distribution: {{A: 50, B: 30, C: 20}}
inputs: {{synthetic_dir: {synthetic_dir}, real_dir: {real_dir}, resolution: 32}}
model: {{params: {{}}}}
""")
    with pytest.raises(SynthCuratorError, match="trainer") as excinfo:
        SynthCurator.from_config(cfg)
    assert not isinstance(excinfo.value, KeyError)


def test_from_config_model_gpu_ids_builds_a_factory_and_a_plain_model(tmp_path):
    """train_production() needs a plain model=..., and the CLI never has a Python object to pass -
    so from_config builds one too, the same way a worker would (bound to gpu_ids[0]), whenever no
    model=... was already given."""
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"""
workdir: {tmp_path / 'run'}
target_distribution: {{A: 50, B: 30, C: 20}}
inputs: {{synthetic_dir: {synthetic_dir}, real_dir: {real_dir}, resolution: 32}}
metrics: {{instance: {METRICS}}}
curation: {{num_images: 15, dist_group_by: [], safety_max_similarity_threshold: null}}
model: {{trainer: tests.conftest:cli_fake_model_factory, gpu_ids: [0, 1]}}
""")
    sc = SynthCurator.from_config(cfg)
    assert sc.model is not None and sc.model.gpu_id == 0   # bound to gpu_ids[0]
    assert sc.gpu_ids == [0, 1]
    assert sc.model_factory(1).gpu_id == 1


def test_from_config_model_gpu_ids_keeps_an_explicit_python_model_arg(tmp_path):
    """An explicitly-passed model= must win over auto-building one from trainer/params - it's
    already the exact instance the caller wants for train_production()."""
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"""
workdir: {tmp_path / 'run'}
target_distribution: {{A: 50, B: 30, C: 20}}
inputs: {{synthetic_dir: {synthetic_dir}, real_dir: {real_dir}, resolution: 32}}
metrics: {{instance: {METRICS}}}
curation: {{num_images: 15, dist_group_by: [], safety_max_similarity_threshold: null}}
model: {{trainer: tests.conftest:cli_fake_model_factory, gpu_ids: [0]}}
""")
    deploy_model = fake_model()
    sc = SynthCurator.from_config(cfg, model=deploy_model)
    assert sc.model is deploy_model
    assert sc.gpu_ids == [0]


def test_from_config_model_gpu_ids_train_production_works_without_a_python_model_arg(tmp_path):
    """The concrete case this all exists for: `synth-curator config.yaml --train-production` never
    has a Python model= to pass, so this has to work from the config alone."""
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"""
workdir: {tmp_path / 'run'}
target_distribution: {{A: 50, B: 30, C: 20}}
inputs: {{synthetic_dir: {synthetic_dir}, real_dir: {real_dir}, resolution: 32}}
metrics: {{instance: {METRICS}}}
curation: {{num_images: 15, dist_group_by: [], safety_max_similarity_threshold: null}}
model: {{trainer: tests.conftest:cli_fake_model_factory, gpu_ids: [0]}}
""")
    sc = SynthCurator.from_config(cfg)   # no model= - exactly what the CLI does
    std_dir = sc.standardize()
    fake_extract_features(std_dir, synthetic_dir, sc.features_dir, real_dir)
    assert sc.train_production() == {"gpu_id": 0, "trained_on": str(sc.run())}


def test_from_config_production_gpu_id_overrides_gpu_ids_zero(tmp_path):
    """production_gpu_id lets the final training run use a different GPU than any of the
    tournament's (e.g. a bigger one) - not just whichever happens to be first in gpu_ids."""
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"""
workdir: {tmp_path / 'run'}
target_distribution: {{A: 50, B: 30, C: 20}}
inputs: {{synthetic_dir: {synthetic_dir}, real_dir: {real_dir}, resolution: 32}}
metrics: {{instance: {METRICS}}}
curation: {{num_images: 15, dist_group_by: [], safety_max_similarity_threshold: null}}
model: {{trainer: tests.conftest:cli_fake_model_factory, gpu_ids: [0, 1], production_gpu_id: 7}}
""")
    sc = SynthCurator.from_config(cfg)
    assert sc.model.gpu_id == 7
    assert sc.gpu_ids == [0, 1]         # the tournament pool is unaffected


def test_from_config_production_gpu_id_without_gpu_ids_is_a_clear_error(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"""
workdir: {tmp_path / 'run'}
target_distribution: {{A: 50, B: 30, C: 20}}
inputs: {{synthetic_dir: {synthetic_dir}, real_dir: {real_dir}, resolution: 32}}
metrics: {{instance: {METRICS}}}
curation: {{num_images: 15, dist_group_by: [], safety_max_similarity_threshold: null}}
model: {{trainer: tests.conftest:cli_fake_model_factory, production_gpu_id: 7}}
""")
    with pytest.raises(SynthCuratorError, match="production_gpu_id needs model.gpu_ids"):
        SynthCurator.from_config(cfg)


def test_random_seeds_per_metric_set_must_be_at_least_one(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    with pytest.raises(SynthCuratorError, match="random_seeds_per_metric_set"):
        SynthCurator(tmp_path / "r", TARGET, model=fake_model(), synthetic_dir=synthetic_dir,
                    real_dir=real_dir, resolution=32, random_seeds_per_metric_set=0)


def test_metric_beta_must_be_positive(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    with pytest.raises(SynthCuratorError, match="metric_beta"):
        SynthCurator(tmp_path / "r", TARGET, model=fake_model(), synthetic_dir=synthetic_dir,
                    real_dir=real_dir, resolution=32, metric_beta=0, **KW)


def test_metric_beta_is_forwarded_to_derive_weights(tmp_path, monkeypatch):
    from synth_curator import pipeline as pipeline_mod

    sc = _make(tmp_path, metric_beta=3.5)
    seen = {}

    def spy(predictions, classes, formula, dist_formula, metrics, beta, scoring):
        seen["beta"], seen["scoring"] = beta, scoring
        return {cls: {} for cls in classes}

    monkeypatch.setattr(pipeline_mod.weights, "derive_weights", spy)
    sc.derive_weights({})
    assert seen["beta"] == 3.5 and seen["scoring"] is None


def test_scoring_must_be_callable(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    with pytest.raises(SynthCuratorError, match="scoring"):
        SynthCurator(tmp_path / "r", TARGET, model=fake_model(), synthetic_dir=synthetic_dir,
                    real_dir=real_dir, resolution=32, scoring="not_a_function", **KW)


def test_scoring_override_is_forwarded_to_derive_weights(tmp_path, monkeypatch):
    from synth_curator import pipeline as pipeline_mod

    def my_scoring(true, pred, classes):
        return {c: 0.0 for c in classes}

    sc = _make(tmp_path, scoring=my_scoring)
    seen = {}

    def spy(predictions, classes, formula, dist_formula, metrics, beta, scoring):
        seen["scoring"] = scoring
        return {cls: {} for cls in classes}

    monkeypatch.setattr(pipeline_mod.weights, "derive_weights", spy)
    sc.derive_weights({})
    assert seen["scoring"] is my_scoring


def test_from_config_reads_weights_scoring_by_dotted_path(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"""
workdir: {tmp_path / 'run'}
target_distribution: {{A: 50, B: 30, C: 20}}
inputs: {{synthetic_dir: {synthetic_dir}, real_dir: {real_dir}, resolution: 32}}
metrics: {{instance: {METRICS}}}
weights: {{scoring: tests.conftest:cli_fake_scoring}}
curation: {{num_images: 15, dist_group_by: [], safety_max_similarity_threshold: null}}
""")
    sc = SynthCurator.from_config(cfg, model=fake_model())
    from tests.conftest import cli_fake_scoring
    assert sc.scoring is cli_fake_scoring


def test_from_config_reads_weights_beta(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"""
workdir: {tmp_path / 'run'}
target_distribution: {{A: 50, B: 30, C: 20}}
inputs: {{synthetic_dir: {synthetic_dir}, real_dir: {real_dir}, resolution: 32}}
metrics: {{instance: {METRICS}}}
weights: {{beta: 1.5}}
curation: {{num_images: 15, dist_group_by: [], safety_max_similarity_threshold: null}}
""")
    sc = SynthCurator.from_config(cfg, model=fake_model())
    assert sc.metric_beta == 1.5


def test_init_rejects_an_unknown_metric(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    with pytest.raises(SynthCuratorError, match="unknown metric"):
        SynthCurator(tmp_path / "r", TARGET, model=fake_model(), synthetic_dir=synthetic_dir,
                    real_dir=real_dir, resolution=32, metrics=["not_a_real_metric"])


def test_standardize_step_skips_real_work_on_second_call(tmp_path, monkeypatch):
    from synth_curator import pipeline as pipeline_mod
    sc = _make(tmp_path)
    std_dir = sc.workdir / "real_standardized"
    assert std_dir.exists()

    def boom(*a, **kw):
        raise AssertionError("standardize() should skip real work when its output already exists")

    monkeypatch.setattr(pipeline_mod.standardize, "standardize", boom)
    assert sc.standardize() == std_dir


def test_extract_features_step_skips_real_work_on_second_call(tmp_path, monkeypatch):
    from synth_curator import pipeline as pipeline_mod
    sc = _make(tmp_path)   # _make() already ran fake_extract_features, populating features_dir

    def boom(*a, **kw):
        raise AssertionError("extract_features() should skip real work when its cache already exists")

    monkeypatch.setattr(pipeline_mod.features, "extract_features", boom)
    assert sc.extract_features(sc.workdir / "real_standardized") == sc.features_dir


# ---- fingerprinting: reusing a workdir after changing inputs must recompute, not go stale -----

def test_standardize_recomputes_when_resolution_changes_on_the_same_workdir(tmp_path):
    from PIL import Image
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    workdir = tmp_path / "run"

    sc1 = SynthCurator(workdir, TARGET, model=fake_model(), synthetic_dir=synthetic_dir,
                       real_dir=real_dir, resolution=32, **KW)
    out1 = sc1.standardize()
    assert Image.open(next(out1.iterdir())).size == (32, 32)

    sc2 = SynthCurator(workdir, TARGET, model=fake_model(), synthetic_dir=synthetic_dir,
                       real_dir=real_dir, resolution=64, **KW)
    out2 = sc2.standardize()
    assert Image.open(next(out2.iterdir())).size == (64, 64)   # genuinely recomputed, not stale 32px


def test_standardize_still_skips_when_nothing_changed(tmp_path, monkeypatch):
    from synth_curator import pipeline as pipeline_mod
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    workdir = tmp_path / "run"
    SynthCurator(workdir, TARGET, model=fake_model(), synthetic_dir=synthetic_dir,
                real_dir=real_dir, resolution=32, **KW).standardize()

    def boom(*a, **kw):
        raise AssertionError("standardize() should skip real work when its fingerprint still matches")
    monkeypatch.setattr(pipeline_mod.standardize, "standardize", boom)

    sc2 = SynthCurator(workdir, TARGET, model=fake_model(), synthetic_dir=synthetic_dir,
                       real_dir=real_dir, resolution=32, **KW)
    sc2.standardize()   # same resolution, same dirs - must not call standardize.standardize()


# ---- pool_workdir: sharing steps 1-3 across a target_distribution/model sweep -----------------

def test_pool_workdir_defaults_to_workdir(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    sc = SynthCurator(tmp_path / "run", TARGET, model=fake_model(), synthetic_dir=synthetic_dir,
                      real_dir=real_dir, resolution=32, **KW)
    assert sc.pool_workdir == sc.workdir
    assert sc.features_dir == sc.workdir / "features"


def test_pool_workdir_separates_steps_1_to_3_from_the_rest(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    pool = tmp_path / "shared_pool"
    sc = SynthCurator(tmp_path / "run", TARGET, model=fake_model(), synthetic_dir=synthetic_dir,
                      real_dir=real_dir, resolution=32, pool_workdir=pool, **KW)
    assert sc.features_dir == pool / "features"

    std_dir = sc.standardize()
    assert std_dir == pool / "real_standardized"
    assert not (sc.workdir / "real_standardized").exists()

    fake_extract_features(std_dir, synthetic_dir, sc.features_dir, real_dir)
    metadata_csv = sc.compute_metrics(std_dir, sc.features_dir)
    assert metadata_csv == pool / "metadata.csv"

    metric_sets_out = sc.curate_metric_sets(metadata_csv, sc.features_dir)
    assert all(str(sc.workdir) in str(p) for paths in metric_sets_out.values() for p in paths.values())


def test_pool_workdir_shared_across_a_target_distribution_sweep_is_only_built_once(tmp_path, monkeypatch):
    """The actual sweep scenario: two SynthCurator instances, different workdir + target_distribution,
    same pool_workdir - standardize() and extract_features() must be skipped outright for the
    second. compute_metrics() always calls through (its own per-column resumability handles reuse,
    same as always), but since the underlying features are untouched, its output must come out
    byte-identical either way."""
    from synth_curator import pipeline as pipeline_mod
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    pool = tmp_path / "shared_pool"

    def make(workdir, target_distribution):
        sc = SynthCurator(tmp_path / workdir, target_distribution, model=fake_model(),
                          synthetic_dir=synthetic_dir, real_dir=real_dir, resolution=32,
                          pool_workdir=pool, **KW)
        std_dir = sc.standardize()
        if not (sc.features_dir / "synthetic_dino.pt").exists():
            fake_extract_features(std_dir, synthetic_dir, sc.features_dir, real_dir)
        else:
            sc.extract_features(std_dir)   # should hit the skip path, not the real (network) extractor
        sc.compute_metrics(std_dir, sc.features_dir)
        return sc

    sc1 = make("sweep_a", TARGET)
    metadata_before = pd.read_csv(pool / "metadata.csv")

    def boom(*a, **kw):
        raise AssertionError("pool steps should be skipped/untouched on the second sweep point")
    monkeypatch.setattr(pipeline_mod.standardize, "standardize", boom)
    monkeypatch.setattr(pipeline_mod.features, "extract_features", boom)

    other_target = {"A": 20.0, "B": 30.0, "C": 50.0}   # different target_distribution, same pool
    sc2 = make("sweep_b", other_target)   # must not hit either boom()

    assert sc1.workdir != sc2.workdir
    assert sc1.pool_workdir == sc2.pool_workdir == pool
    pd.testing.assert_frame_equal(pd.read_csv(pool / "metadata.csv"), metadata_before)


def test_from_config_reads_pool_workdir(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    pool = tmp_path / "shared_pool"
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"""
workdir: {tmp_path / 'run'}
target_distribution: {{A: 50, B: 30, C: 20}}
inputs: {{synthetic_dir: {synthetic_dir}, real_dir: {real_dir}, resolution: 32, pool_workdir: {pool}}}
metrics: {{instance: {METRICS}}}
curation: {{num_images: 15, dist_group_by: [], safety_max_similarity_threshold: null}}
""")
    sc = SynthCurator.from_config(cfg, model=fake_model())
    assert sc.pool_workdir == pool


def test_extract_features_recomputes_when_synthetic_dir_changes_on_the_same_workdir(tmp_path, monkeypatch):
    from synth_curator import pipeline as pipeline_mod
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    other_synthetic_dir = tmp_path / "other_synthetic"
    for cls, seed in (("A", 90), ("B", 91), ("C", 92)):
        make_image(other_synthetic_dir / cls / "s0.png", seed)

    calls = []

    def fake_extract(std_dir, syn_dir, out_dir, real_dir_arg):
        calls.append(syn_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "synthetic_dino.pt").write_bytes(b"x")   # the marker the wrapper checks for
        return out_dir

    monkeypatch.setattr(pipeline_mod.features, "extract_features", fake_extract)
    workdir = tmp_path / "run"

    sc1 = SynthCurator(workdir, TARGET, model=fake_model(), synthetic_dir=synthetic_dir,
                       real_dir=real_dir, resolution=32, **KW)
    std_dir = sc1.standardize()
    sc1.extract_features(std_dir)

    sc2 = SynthCurator(workdir, TARGET, model=fake_model(), synthetic_dir=other_synthetic_dir,
                       real_dir=real_dir, resolution=32, **KW)
    sc2.extract_features(std_dir)

    assert calls == [synthetic_dir, other_synthetic_dir]   # both ran - the second wasn't silently skipped


def test_force_retrain_is_forwarded_to_run_tournament(tmp_path, monkeypatch):
    from synth_curator import pipeline as pipeline_mod
    sc = _make(tmp_path, force_retrain=True)
    seen = {}

    def spy(metric_sets, model, out_dir, force_retrain):
        seen["force_retrain"] = force_retrain
        return {}

    monkeypatch.setattr(pipeline_mod.tournament, "run", spy)
    sc.run_tournament({})
    assert seen["force_retrain"] is True


def test_from_config_reads_force_retrain(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"""
workdir: {tmp_path / 'run'}
target_distribution: {{A: 50, B: 30, C: 20}}
inputs: {{synthetic_dir: {synthetic_dir}, real_dir: {real_dir}, resolution: 32}}
model: {{trainer: tests.conftest:cli_fake_trainer}}
metrics: {{instance: {METRICS}}}
tournament: {{force_retrain: true}}
curation: {{num_images: 15, dist_group_by: [], safety_max_similarity_threshold: null}}
""")
    sc = SynthCurator.from_config(cfg)
    assert sc.force_retrain is True


def test_from_config_without_a_model_is_a_clear_error(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"""
workdir: {tmp_path / 'run'}
target_distribution: {{A: 50, B: 30, C: 20}}
inputs: {{synthetic_dir: {synthetic_dir}, real_dir: {real_dir}, resolution: 32}}
curation: {{num_images: 15, dist_group_by: [], safety_max_similarity_threshold: null}}
""")
    with pytest.raises(SynthCuratorError, match="model is required"):
        SynthCurator.from_config(cfg)


# ---- _validate() : fail fast, before any expensive work -----------------------------------

def test_validate_rejects_a_model_without_train_and_evaluate(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    with pytest.raises(SynthCuratorError, match="train_and_evaluate"):
        SynthCurator(tmp_path / "r", TARGET, model=object(), synthetic_dir=synthetic_dir,
                    real_dir=real_dir, resolution=32, **KW)


def test_init_rejects_model_factory_without_gpu_ids(tmp_path):
    from tests.conftest import cli_fake_model_factory
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    with pytest.raises(SynthCuratorError, match="model_factory needs gpu_ids"):
        SynthCurator(tmp_path / "r", TARGET, model_factory=cli_fake_model_factory, synthetic_dir=synthetic_dir,
                    real_dir=real_dir, resolution=32, **KW)


def test_init_rejects_gpu_ids_without_model_factory(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    with pytest.raises(SynthCuratorError, match="gpu_ids needs model_factory"):
        SynthCurator(tmp_path / "r", TARGET, model=fake_model(), gpu_ids=[0, 1], synthetic_dir=synthetic_dir,
                    real_dir=real_dir, resolution=32, **KW)


def test_init_rejects_a_non_callable_model_factory(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    with pytest.raises(SynthCuratorError, match="model_factory must be callable"):
        SynthCurator(tmp_path / "r", TARGET, model_factory="not callable", gpu_ids=[0], synthetic_dir=synthetic_dir,
                    real_dir=real_dir, resolution=32, **KW)


def test_model_factory_and_gpu_ids_accepted_without_a_plain_model(tmp_path):
    from tests.conftest import cli_fake_model_factory
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    sc = SynthCurator(tmp_path / "r", TARGET, model_factory=cli_fake_model_factory, gpu_ids=[0, 1],
                      synthetic_dir=synthetic_dir, real_dir=real_dir, resolution=32, **KW)
    assert sc.model is None
    assert sc.gpu_ids == [0, 1]


def test_run_tournament_dispatches_to_run_parallel_when_model_factory_is_set(tmp_path, monkeypatch):
    from synth_curator import pipeline as pipeline_mod
    from tests.conftest import cli_fake_model_factory
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    sc = SynthCurator(tmp_path / "r", TARGET, model_factory=cli_fake_model_factory, gpu_ids=[0, 1],
                      synthetic_dir=synthetic_dir, real_dir=real_dir, resolution=32, **KW)
    seen = {}

    def spy(metric_sets, model_factory, gpu_ids, out_dir, force_retrain):
        seen["model_factory"], seen["gpu_ids"] = model_factory, gpu_ids
        return {}

    monkeypatch.setattr(pipeline_mod.tournament, "run_parallel", spy)
    sc.run_tournament({})
    assert seen["model_factory"] is cli_fake_model_factory
    assert seen["gpu_ids"] == [0, 1]


def test_validate_rejects_a_missing_synthetic_dir(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    with pytest.raises(SynthCuratorError, match="synthetic_dir"):
        SynthCurator(tmp_path / "r", TARGET, model=fake_model(), synthetic_dir=tmp_path / "nope",
                    real_dir=real_dir, resolution=32, **KW)


def test_validate_rejects_an_empty_synthetic_dir(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SynthCuratorError, match="synthetic_dir"):
        SynthCurator(tmp_path / "r", TARGET, model=fake_model(), synthetic_dir=empty,
                    real_dir=real_dir, resolution=32, **KW)


def test_validate_rejects_a_target_distribution_class_with_no_matching_synthetic_folder(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)   # classes A, B, C
    bad_target = {"A": 50.0, "B": 30.0, "Z": 20.0}   # "Z" has no folder under synthetic_dir
    with pytest.raises(SynthCuratorError, match="Z"):
        SynthCurator(tmp_path / "r", bad_target, model=fake_model(), synthetic_dir=synthetic_dir,
                    real_dir=real_dir, resolution=32, **KW)


def test_validate_rejects_a_distributional_metric_with_no_dist_group_by(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    with pytest.raises(SynthCuratorError, match="kid_dino"):
        SynthCurator(tmp_path / "r", TARGET, model=fake_model(), synthetic_dir=synthetic_dir,
                    real_dir=real_dir, resolution=32,
                    **{**KW, "metrics": [*METRICS, "kid_dino"], "dist_group_by": []})


def test_validate_allows_a_distributional_metric_when_dist_group_by_is_set(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    SynthCurator(tmp_path / "r", TARGET, model=fake_model(), synthetic_dir=synthetic_dir,
                real_dir=real_dir, resolution=32,
                **{**KW, "metrics": [*METRICS, "kid_dino"], "dist_group_by": ["some_col"]})


def test_validate_allows_a_zero_percent_class_absent_from_synthetic_dir(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)   # classes A, B, C
    target = {"A": 50.0, "B": 50.0, "Z": 0.0}   # "Z" is 0% - fine even with no matching folder
    SynthCurator(tmp_path / "r", target, model=fake_model(), synthetic_dir=synthetic_dir,
                real_dir=real_dir, resolution=32, **KW)


def test_validate_rejects_synthetic_images_with_no_class_folder(tmp_path):
    real_dir, _ = build_image_pool(tmp_path)
    flat_synthetic = tmp_path / "flat_synthetic"
    make_image(flat_synthetic / "img0.png", seed=0)   # directly under root, no class folder
    with pytest.raises(SynthCuratorError, match="class folder"):
        SynthCurator(tmp_path / "r", TARGET, model=fake_model(), synthetic_dir=flat_synthetic,
                    real_dir=real_dir, resolution=32, **KW)


# ---- metadata_csv: an alternative to build_metadata()'s folder scan -----------------------

def _write_metadata_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_validate_rejects_a_missing_metadata_csv(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    with pytest.raises(SynthCuratorError, match="metadata_csv"):
        SynthCurator(tmp_path / "r", TARGET, model=fake_model(), synthetic_dir=synthetic_dir,
                    real_dir=real_dir, resolution=32, metadata_csv=tmp_path / "nope.csv", **KW)


def test_validate_rejects_a_metadata_csv_missing_required_columns(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    csv = _write_metadata_csv(tmp_path / "meta.csv", {"filename": ["a.png"]})   # no target_class_label
    with pytest.raises(SynthCuratorError, match="target_class_label"):
        SynthCurator(tmp_path / "r", TARGET, model=fake_model(), synthetic_dir=synthetic_dir,
                    real_dir=real_dir, resolution=32, metadata_csv=csv, **KW)


def test_validate_rejects_a_metadata_csv_that_is_not_valid_csv(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    bad = tmp_path / "meta.csv"
    bad.write_bytes(b"\x00\x01not,a,csv\x02")
    with pytest.raises(SynthCuratorError, match="metadata_csv"):
        SynthCurator(tmp_path / "r", TARGET, model=fake_model(), synthetic_dir=synthetic_dir,
                    real_dir=real_dir, resolution=32, metadata_csv=bad, **KW)


def test_validate_rejects_a_target_distribution_class_missing_from_metadata_csv(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    csv = _write_metadata_csv(tmp_path / "meta.csv", {
        "filename": ["a.png", "b.png"], "target_class_label": ["A", "B"]})   # TARGET also needs "C"
    with pytest.raises(SynthCuratorError, match="C"):
        SynthCurator(tmp_path / "r", TARGET, model=fake_model(), synthetic_dir=synthetic_dir,
                    real_dir=real_dir, resolution=32, metadata_csv=csv, **KW)


def test_validate_with_metadata_csv_does_not_require_synthetic_dir_class_folders(tmp_path):
    """The class-match check should read the CSV, not synthetic_dir's folders, once metadata_csv
    is given - so a flat synthetic_dir (no class subfolders at all) is fine here."""
    real_dir, _ = build_image_pool(tmp_path)
    flat_synthetic = tmp_path / "flat_synthetic"
    make_image(flat_synthetic / "img0.png", seed=0)
    csv = _write_metadata_csv(tmp_path / "meta.csv", {
        "filename": ["a.png", "b.png", "c.png"], "target_class_label": ["A", "B", "C"]})
    SynthCurator(tmp_path / "r", TARGET, model=fake_model(), synthetic_dir=flat_synthetic,
                real_dir=real_dir, resolution=32, metadata_csv=csv, **KW)


def test_from_config_reads_metadata_csv(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    csv = _write_metadata_csv(tmp_path / "meta.csv", {
        "filename": ["a.png", "b.png", "c.png"], "target_class_label": ["A", "B", "C"]})
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"""
workdir: {tmp_path / 'run'}
target_distribution: {{A: 50, B: 30, C: 20}}
inputs: {{synthetic_dir: {synthetic_dir}, real_dir: {real_dir}, resolution: 32, metadata_csv: {csv}}}
metrics: {{instance: {METRICS}}}
curation: {{num_images: 15, dist_group_by: [], safety_max_similarity_threshold: null}}
""")
    sc = SynthCurator.from_config(cfg, model=fake_model())
    assert sc.metadata_csv == csv


def test_compute_metrics_clears_the_cached_csv_when_metadata_csv_changes(tmp_path, monkeypatch):
    """pipeline.compute_metrics() always calls steps.metrics.compute_metrics() - unlike
    standardize()/extract_features() it doesn't skip the call itself, since the real function
    already resumes incrementally per metric column. What the fingerprint controls is whether the
    stale CACHED CSV is cleared first, so a changed metadata_csv can't be masked by stale columns."""
    from synth_curator import pipeline as pipeline_mod
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    csv_a = _write_metadata_csv(tmp_path / "meta_a.csv", {"filename": ["a.png"], "target_class_label": ["A"]})
    csv_b = _write_metadata_csv(tmp_path / "meta_b.csv", {"filename": ["b.png"], "target_class_label": ["B"]})
    workdir = tmp_path / "run"

    seen_out_csv_existed = []

    def fake_compute_metrics(synthetic_dir, real_std_dir, features_dir, out_csv, metrics, base_metadata=None):
        seen_out_csv_existed.append(out_csv.exists())
        out_csv.write_text("filename,target_class_label\nx.png,A\n")
        return out_csv

    monkeypatch.setattr(pipeline_mod.metrics, "compute_metrics", fake_compute_metrics)

    sc1 = SynthCurator(workdir, {"A": 100.0}, model=fake_model(), synthetic_dir=synthetic_dir,
                       real_dir=real_dir, resolution=32, metadata_csv=csv_a, **{**KW, "num_images": 1})
    sc1.compute_metrics(sc1.workdir / "real_standardized", sc1.features_dir)

    sc2 = SynthCurator(workdir, {"B": 100.0}, model=fake_model(), synthetic_dir=synthetic_dir,
                       real_dir=real_dir, resolution=32, metadata_csv=csv_b, **{**KW, "num_images": 1})
    sc2.compute_metrics(sc2.workdir / "real_standardized", sc2.features_dir)

    assert seen_out_csv_existed == [False, False]   # second call: cleared first, since metadata_csv changed


def test_compute_metrics_does_not_clear_the_cached_csv_when_metadata_csv_is_unchanged(tmp_path, monkeypatch):
    from synth_curator import pipeline as pipeline_mod
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    csv = _write_metadata_csv(tmp_path / "meta.csv", {"filename": ["a.png"], "target_class_label": ["A"]})
    workdir = tmp_path / "run"

    seen_out_csv_existed = []

    def fake_compute_metrics(synthetic_dir, real_std_dir, features_dir, out_csv, metrics, base_metadata=None):
        seen_out_csv_existed.append(out_csv.exists())
        out_csv.write_text("filename,target_class_label\na.png,A\n")
        return out_csv

    monkeypatch.setattr(pipeline_mod.metrics, "compute_metrics", fake_compute_metrics)

    def make():
        return SynthCurator(workdir, {"A": 100.0}, model=fake_model(), synthetic_dir=synthetic_dir,
                            real_dir=real_dir, resolution=32, metadata_csv=csv, **{**KW, "num_images": 1})

    make().compute_metrics(workdir / "real_standardized", workdir / "features")
    make().compute_metrics(workdir / "real_standardized", workdir / "features")

    assert seen_out_csv_existed == [False, True]   # second call: out_csv was NOT cleared first - fingerprint matched


# ---- fingerprinting: steps 4 (curate_metric_sets) and 7 (curate) -----------------------------

def test_curate_metric_sets_recomputes_when_num_images_changes_on_the_same_workdir(tmp_path, monkeypatch):
    from synth_curator import pipeline as pipeline_mod
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    workdir = tmp_path / "run"
    meta_csv = tmp_path / "meta.csv"

    seen_out_dir_existed = []

    def fake_curate_metric_sets(metadata_csv, features_dir, out_dir, metrics, target_distribution,
                                num_images, dist_group_by, seeds, image_dir=None):
        seen_out_dir_existed.append(out_dir.exists())
        out_dir.mkdir(parents=True, exist_ok=True)
        return {}

    monkeypatch.setattr(pipeline_mod.metric_sets, "curate_metric_sets", fake_curate_metric_sets)

    sc1 = SynthCurator(workdir, TARGET, model=fake_model(), synthetic_dir=synthetic_dir,
                       real_dir=real_dir, resolution=32, **{**KW, "num_images": 15})
    sc1.curate_metric_sets(meta_csv, sc1.features_dir)

    sc2 = SynthCurator(workdir, TARGET, model=fake_model(), synthetic_dir=synthetic_dir,
                       real_dir=real_dir, resolution=32, **{**KW, "num_images": 30})
    sc2.curate_metric_sets(meta_csv, sc2.features_dir)

    assert seen_out_dir_existed == [False, False]   # second call: cleared first, since num_images changed


def test_curate_metric_sets_does_not_recompute_when_nothing_changed(tmp_path, monkeypatch):
    from synth_curator import pipeline as pipeline_mod
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    workdir = tmp_path / "run"
    meta_csv = tmp_path / "meta.csv"

    seen_out_dir_existed = []

    def fake_curate_metric_sets(metadata_csv, features_dir, out_dir, metrics, target_distribution,
                                num_images, dist_group_by, seeds, image_dir=None):
        seen_out_dir_existed.append(out_dir.exists())
        out_dir.mkdir(parents=True, exist_ok=True)
        return {}

    monkeypatch.setattr(pipeline_mod.metric_sets, "curate_metric_sets", fake_curate_metric_sets)

    def make():
        return SynthCurator(workdir, TARGET, model=fake_model(), synthetic_dir=synthetic_dir,
                            real_dir=real_dir, resolution=32, **KW)

    make().curate_metric_sets(meta_csv, workdir / "features")
    make().curate_metric_sets(meta_csv, workdir / "features")

    assert seen_out_dir_existed == [False, True]   # second call: NOT cleared - fingerprint matched


def test_curate_step_recomputes_when_weights_change(tmp_path, monkeypatch):
    from synth_curator import pipeline as pipeline_mod
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    meta_csv = tmp_path / "meta.csv"

    seen_manifest_existed = []

    def fake_curate(metadata_csv, features_dir, weights, out_dir, **kw):
        manifest = out_dir / f"seed_{kw['seed']}.csv"
        seen_manifest_existed.append(manifest.exists())
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("source_path,target_class,safe_filename\n")
        return manifest

    monkeypatch.setattr(pipeline_mod.curate, "curate", fake_curate)

    sc = SynthCurator(tmp_path / "run", TARGET, model=fake_model(), synthetic_dir=synthetic_dir,
                      real_dir=real_dir, resolution=32, **KW)
    sc.curate(meta_csv, sc.features_dir, {"A": {"m": 1.0}})
    sc.curate(meta_csv, sc.features_dir, {"A": {"m": 0.5}})   # different weights, same workdir

    assert seen_manifest_existed == [False, False]   # second call: cleared first, since weights changed


def test_curate_step_does_not_recompute_when_nothing_changed(tmp_path, monkeypatch):
    from synth_curator import pipeline as pipeline_mod
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    meta_csv = tmp_path / "meta.csv"

    seen_manifest_existed = []

    def fake_curate(metadata_csv, features_dir, weights, out_dir, **kw):
        manifest = out_dir / f"seed_{kw['seed']}.csv"
        seen_manifest_existed.append(manifest.exists())
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("source_path,target_class,safe_filename\n")
        return manifest

    monkeypatch.setattr(pipeline_mod.curate, "curate", fake_curate)

    sc = SynthCurator(tmp_path / "run", TARGET, model=fake_model(), synthetic_dir=synthetic_dir,
                      real_dir=real_dir, resolution=32, **KW)
    same_weights = {"A": {"m": 1.0}}
    sc.curate(meta_csv, sc.features_dir, same_weights)
    sc.curate(meta_csv, sc.features_dir, same_weights)

    assert seen_manifest_existed == [False, True]   # second call: NOT cleared - fingerprint matched


def test_validate_rejects_a_missing_real_dir(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    with pytest.raises(SynthCuratorError, match="real_dir"):
        SynthCurator(tmp_path / "r", TARGET, model=fake_model(), synthetic_dir=synthetic_dir,
                    real_dir=tmp_path / "nope", resolution=32, **KW)


def test_validate_rejects_a_non_positive_resolution(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    with pytest.raises(SynthCuratorError, match="resolution"):
        SynthCurator(tmp_path / "r", TARGET, model=fake_model(), synthetic_dir=synthetic_dir,
                    real_dir=real_dir, resolution=0, **KW)


def test_validate_rejects_a_target_distribution_not_summing_to_100(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    with pytest.raises(SynthCuratorError, match="target_distribution"):
        SynthCurator(tmp_path / "r", {"A": 0.5, "B": 0.3, "C": 0.2}, model=fake_model(),
                    synthetic_dir=synthetic_dir, real_dir=real_dir, resolution=32, **KW)


def test_validate_rejects_a_negative_target_distribution_value(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    with pytest.raises(SynthCuratorError, match="target_distribution"):
        SynthCurator(tmp_path / "r", {"A": 110.0, "B": -10.0, "C": 0.0}, model=fake_model(),
                    synthetic_dir=synthetic_dir, real_dir=real_dir, resolution=32, **KW)


def test_validate_rejects_an_unknown_utility_formula(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    with pytest.raises(SynthCuratorError, match="utility_formula"):
        SynthCurator(tmp_path / "r", TARGET, model=fake_model(), synthetic_dir=synthetic_dir,
                    real_dir=real_dir, resolution=32, utility_formula="not_a_formula", **KW)


def test_validate_rejects_an_unknown_dist_utility_formula(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    with pytest.raises(SynthCuratorError, match="dist_utility_formula"):
        SynthCurator(tmp_path / "r", TARGET, model=fake_model(), synthetic_dir=synthetic_dir,
                    real_dir=real_dir, resolution=32, dist_utility_formula="not_a_formula", **KW)


def test_validate_allows_a_none_dist_utility_formula(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    SynthCurator(tmp_path / "r", TARGET, model=fake_model(), synthetic_dir=synthetic_dir,
                real_dir=real_dir, resolution=32, dist_utility_formula=None, **KW)


def test_validate_rejects_metric_beta_and_scoring_together(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    with pytest.raises(SynthCuratorError, match="metric_beta"):
        SynthCurator(tmp_path / "r", TARGET, model=fake_model(), synthetic_dir=synthetic_dir,
                    real_dir=real_dir, resolution=32, metric_beta=3.0,
                    scoring=lambda t, p, c: {x: 0.0 for x in c}, **KW)


def test_validate_allows_the_default_metric_beta_alongside_scoring(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    SynthCurator(tmp_path / "r", TARGET, model=fake_model(), synthetic_dir=synthetic_dir,
                real_dir=real_dir, resolution=32, scoring=lambda t, p, c: {x: 0.0 for x in c}, **KW)


def test_validate_rejects_a_non_positive_num_images(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    with pytest.raises(SynthCuratorError, match="num_images"):
        SynthCurator(tmp_path / "r", TARGET, model=fake_model(), synthetic_dir=synthetic_dir,
                    real_dir=real_dir, resolution=32, **{**KW, "num_images": 0})


def test_validate_rejects_a_non_positive_temperature(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    with pytest.raises(SynthCuratorError, match="temperature"):
        SynthCurator(tmp_path / "r", TARGET, model=fake_model(), synthetic_dir=synthetic_dir,
                    real_dir=real_dir, resolution=32, temperature=0.0, **KW)


def test_validate_rejects_num_blocks_below_one(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    with pytest.raises(SynthCuratorError, match="num_blocks"):
        SynthCurator(tmp_path / "r", TARGET, model=fake_model(), synthetic_dir=synthetic_dir,
                    real_dir=real_dir, resolution=32, num_blocks=0, **KW)


def test_validate_rejects_an_out_of_range_safety_threshold(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    with pytest.raises(SynthCuratorError, match="safety_max_similarity_threshold"):
        SynthCurator(tmp_path / "r", TARGET, model=fake_model(), synthetic_dir=synthetic_dir,
                    real_dir=real_dir, resolution=32,
                    **{**KW, "safety_max_similarity_threshold": 1.5})


# ---- train_production() ---------------------------------------------------------------------

def test_train_production_rejects_a_model_without_it_before_doing_any_work(tmp_path, monkeypatch):
    sc = _make(tmp_path)

    def boom():
        raise AssertionError("train_production() must validate before calling run()")
    monkeypatch.setattr(sc, "run", boom)

    with pytest.raises(SynthCuratorError, match="train_production"):
        sc.train_production()


def test_train_production_rejects_a_model_factory_only_construction(tmp_path):
    """model_factory-built models aren't kept around after the tournament - train_production()
    needs a plain model=... instance too."""
    from tests.conftest import cli_fake_model_factory
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    sc = SynthCurator(tmp_path / "r", TARGET, model_factory=cli_fake_model_factory, gpu_ids=[0],
                      synthetic_dir=synthetic_dir, real_dir=real_dir, resolution=32, **KW)
    with pytest.raises(SynthCuratorError, match="plain model"):
        sc.train_production()


def test_train_production_curates_then_delegates_and_returns_whatever_the_model_returns(tmp_path):
    class TwoInterfaceModel:
        def __init__(self):
            self.calls = []

        def train_and_evaluate(self, train_manifest, seed):
            return fake_model().train_and_evaluate(train_manifest, seed)

        def train_production(self, train_manifest):
            self.calls.append(train_manifest)
            return {"checkpoint": "not/a/real/path.pt", "trained_on": str(train_manifest)}

    model = TwoInterfaceModel()
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    sc = SynthCurator(tmp_path / "run", TARGET, model=model, synthetic_dir=synthetic_dir,
                      real_dir=real_dir, resolution=32, **KW)
    std_dir = sc.standardize()
    fake_extract_features(std_dir, synthetic_dir, sc.features_dir, real_dir)

    result = sc.train_production()

    assert len(model.calls) == 1
    manifest = model.calls[0]
    assert manifest.name == "seed_0.csv" and len(pd.read_csv(manifest)) == 15
    assert result == {"checkpoint": "not/a/real/path.pt", "trained_on": str(manifest)}


# ---- weights_path / shipped_weights (precomputed weights - skip the tournament) -----------

PRECOMPUTED_KW = dict(dist_group_by=[], num_images=15, safety_max_similarity_threshold=None)   # no metrics= - that's forbidden here


class _ProductionOnlyModel:
    """No train_and_evaluate at all - valid with weights_path/shipped_weights set, since
    train_and_evaluate is never called there; the only thing model=... is for is train_production()."""

    def __init__(self):
        self.calls = []

    def train_production(self, train_manifest):
        self.calls.append(train_manifest)
        return {"trained_on": str(train_manifest)}


def _make_precomputed(tmp_path, bundle_dir, **overrides):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    kw = dict(dist_group_by=[], num_images=15, safety_max_similarity_threshold=None,
             weights_path=bundle_dir, synthetic_dir=synthetic_dir, real_dir=real_dir, resolution=32)
    kw.update(overrides)
    sc = SynthCurator(tmp_path / "run", TARGET, **kw)
    std_dir = sc.standardize()
    fake_extract_features(std_dir, synthetic_dir, sc.features_dir, real_dir)
    return sc


def test_weights_path_and_shipped_weights_are_mutually_exclusive(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    bundle = build_weights_bundle(tmp_path / "b", list(TARGET), METRICS)
    with pytest.raises(SynthCuratorError, match="can't both be set"):
        SynthCurator(tmp_path / "run", TARGET, weights_path=bundle, shipped_weights="isic2018_swin_b",
                    synthetic_dir=synthetic_dir, real_dir=real_dir, resolution=32, **PRECOMPUTED_KW)


def test_model_is_optional_with_weights_path(tmp_path):
    bundle = build_weights_bundle(tmp_path / "b", list(TARGET), METRICS)
    sc = _make_precomputed(tmp_path, bundle)
    assert sc.model is None


def test_unknown_shipped_weights_name_lists_whats_available(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    with pytest.raises(SynthCuratorError, match=r"not found - available: \[.*isic2018_swin_b.*\]"):
        SynthCurator(tmp_path / "run", TARGET, shipped_weights="bogus_preset_name",
                    synthetic_dir=synthetic_dir, real_dir=real_dir, resolution=32, **PRECOMPUTED_KW)


def test_model_factory_forbidden_with_weights_path(tmp_path):
    from tests.conftest import cli_fake_model_factory
    bundle = build_weights_bundle(tmp_path / "b", list(TARGET), METRICS)
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    with pytest.raises(SynthCuratorError, match="model_factory/gpu_ids don't apply"):
        SynthCurator(tmp_path / "run", TARGET, weights_path=bundle, model_factory=cli_fake_model_factory,
                    gpu_ids=[0], synthetic_dir=synthetic_dir, real_dir=real_dir, resolution=32, **PRECOMPUTED_KW)


def test_random_seeds_per_metric_set_forbidden_with_weights_path(tmp_path):
    bundle = build_weights_bundle(tmp_path / "b", list(TARGET), METRICS)
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    with pytest.raises(SynthCuratorError, match="random_seeds_per_metric_set/force_retrain"):
        SynthCurator(tmp_path / "run", TARGET, weights_path=bundle, random_seeds_per_metric_set=2,
                    synthetic_dir=synthetic_dir, real_dir=real_dir, resolution=32, **PRECOMPUTED_KW)


def test_force_retrain_forbidden_with_weights_path(tmp_path):
    bundle = build_weights_bundle(tmp_path / "b", list(TARGET), METRICS)
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    with pytest.raises(SynthCuratorError, match="random_seeds_per_metric_set/force_retrain"):
        SynthCurator(tmp_path / "run", TARGET, weights_path=bundle, force_retrain=True,
                    synthetic_dir=synthetic_dir, real_dir=real_dir, resolution=32, **PRECOMPUTED_KW)


def test_utility_formula_forbidden_with_weights_path(tmp_path):
    bundle = build_weights_bundle(tmp_path / "b", list(TARGET), METRICS)
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    with pytest.raises(SynthCuratorError, match="utility_formula/dist_utility_formula/metric_beta/scoring"):
        SynthCurator(tmp_path / "run", TARGET, weights_path=bundle, utility_formula="mean_anchored",
                    synthetic_dir=synthetic_dir, real_dir=real_dir, resolution=32, **PRECOMPUTED_KW)


def test_metric_beta_forbidden_with_weights_path(tmp_path):
    bundle = build_weights_bundle(tmp_path / "b", list(TARGET), METRICS)
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    with pytest.raises(SynthCuratorError, match="utility_formula/dist_utility_formula/metric_beta/scoring"):
        SynthCurator(tmp_path / "run", TARGET, weights_path=bundle, metric_beta=1.0,
                    synthetic_dir=synthetic_dir, real_dir=real_dir, resolution=32, **PRECOMPUTED_KW)


def test_metrics_forbidden_with_weights_path(tmp_path):
    bundle = build_weights_bundle(tmp_path / "b", list(TARGET), METRICS)
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    with pytest.raises(SynthCuratorError, match="metrics doesn't apply"):
        SynthCurator(tmp_path / "run", TARGET, weights_path=bundle, metrics=METRICS,
                    dist_group_by=[], num_images=15, safety_max_similarity_threshold=None,
                    synthetic_dir=synthetic_dir, real_dir=real_dir, resolution=32)


def test_metrics_defaults_from_the_bundle_with_weights_path(tmp_path):
    bundle = build_weights_bundle(tmp_path / "b", list(TARGET), METRICS)
    sc = _make_precomputed(tmp_path, bundle)
    assert sc.metrics == METRICS


def test_classes_mismatch_between_bundle_and_target_distribution_is_a_clear_error(tmp_path):
    bundle = build_weights_bundle(tmp_path / "b", ["X", "Y", "Z"], METRICS)
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    with pytest.raises(SynthCuratorError, match="classes"):
        SynthCurator(tmp_path / "run", TARGET, weights_path=bundle,
                    synthetic_dir=synthetic_dir, real_dir=real_dir, resolution=32, **PRECOMPUTED_KW)


def test_model_needing_only_train_production_is_valid_with_weights_path(tmp_path):
    bundle = build_weights_bundle(tmp_path / "b", list(TARGET), METRICS)
    sc = _make_precomputed(tmp_path, bundle, model=_ProductionOnlyModel())
    assert isinstance(sc.model, _ProductionOnlyModel)   # didn't raise for lacking train_and_evaluate


def test_model_without_train_production_is_a_clear_error_with_weights_path(tmp_path):
    bundle = build_weights_bundle(tmp_path / "b", list(TARGET), METRICS)
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    with pytest.raises(SynthCuratorError, match="train_production"):
        SynthCurator(tmp_path / "run", TARGET, weights_path=bundle, model=object(),
                    synthetic_dir=synthetic_dir, real_dir=real_dir, resolution=32, **PRECOMPUTED_KW)


def test_resolve_weights_uses_the_bundle_and_skips_the_tournament(tmp_path, monkeypatch):
    weights = {c: {m: float(i) for i, m in enumerate(METRICS)} for c in TARGET}
    bundle = build_weights_bundle(tmp_path / "b", list(TARGET), METRICS, weights=weights)
    sc = _make_precomputed(tmp_path, bundle)

    def boom(*a, **kw):
        raise AssertionError("no tournament should run with weights_path set")
    monkeypatch.setattr(sc, "curate_metric_sets", boom)
    monkeypatch.setattr(sc, "run_tournament", boom)
    monkeypatch.setattr(sc, "derive_weights", boom)

    metadata_csv = sc.compute_metrics(sc.standardize(), sc.features_dir)
    resolved = sc.resolve_weights(metadata_csv, sc.features_dir)
    assert resolved == weights
    assert json.loads((sc.workdir / "weights.json").read_text()) == weights
    assert (sc.workdir / "weights_source.json").is_file()


def test_run_end_to_end_with_weights_path(tmp_path):
    bundle = build_weights_bundle(tmp_path / "b", list(TARGET), METRICS)
    sc = _make_precomputed(tmp_path, bundle)
    manifest = sc.run()
    assert len(pd.read_csv(manifest)) == 15
    assert not (sc.workdir / "metric_sets").exists()   # step 4 never ran
    assert not (sc.workdir / "tournament").exists()    # step 5 never ran


def test_train_production_works_with_weights_path(tmp_path):
    bundle = build_weights_bundle(tmp_path / "b", list(TARGET), METRICS)
    model = _ProductionOnlyModel()
    sc = _make_precomputed(tmp_path, bundle, model=model)
    result = sc.train_production()
    assert len(model.calls) == 1
    assert result == {"trained_on": str(model.calls[0])}


# ---- from_config: weights_path / shipped_weights -------------------------------------------

def _write_precomputed_config(tmp_path, bundle_dir, extra_yaml="", model_yaml=""):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"""
workdir: {tmp_path / 'run'}
target_distribution: {{A: 50, B: 30, C: 20}}
inputs: {{synthetic_dir: {synthetic_dir}, real_dir: {real_dir}, resolution: 32}}
curation: {{num_images: 15, dist_group_by: [], safety_max_similarity_threshold: null}}
weights_path: {bundle_dir}
{model_yaml}
{extra_yaml}
""")
    return cfg


def test_from_config_weights_path_forbids_tournament_block(tmp_path):
    bundle = build_weights_bundle(tmp_path / "b", list(TARGET), METRICS)
    cfg = _write_precomputed_config(tmp_path, bundle, extra_yaml="tournament: {force_retrain: true}")
    with pytest.raises(SynthCuratorError, match="tournament"):
        SynthCurator.from_config(cfg)


def test_from_config_weights_path_forbids_weights_block(tmp_path):
    bundle = build_weights_bundle(tmp_path / "b", list(TARGET), METRICS)
    cfg = _write_precomputed_config(tmp_path, bundle, extra_yaml="weights: {beta: 1.0}")
    with pytest.raises(SynthCuratorError, match="weights"):
        SynthCurator.from_config(cfg)


def test_from_config_weights_path_forbids_metrics_instance(tmp_path):
    bundle = build_weights_bundle(tmp_path / "b", list(TARGET), METRICS)
    cfg = _write_precomputed_config(tmp_path, bundle, extra_yaml=f"metrics: {{instance: {METRICS}}}")
    with pytest.raises(SynthCuratorError, match="metrics.instance"):
        SynthCurator.from_config(cfg)


def test_from_config_weights_path_allows_metrics_dist_min_samples(tmp_path):
    bundle = build_weights_bundle(tmp_path / "b", list(TARGET), METRICS)
    cfg = _write_precomputed_config(tmp_path, bundle, extra_yaml="metrics: {dist_min_samples: {fid: 10}}")
    sc = SynthCurator.from_config(cfg)
    assert sc.dist_min_samples == {"fid": 10}


def test_from_config_weights_path_forbids_model_gpu_ids(tmp_path):
    bundle = build_weights_bundle(tmp_path / "b", list(TARGET), METRICS)
    cfg = _write_precomputed_config(tmp_path, bundle,
                                    model_yaml="model: {trainer: tests.conftest:cli_fake_model_factory, gpu_ids: [0]}")
    with pytest.raises(SynthCuratorError, match="model.gpu_ids"):
        SynthCurator.from_config(cfg)


def test_from_config_weights_path_model_trainer_builds_a_plain_model(tmp_path):
    bundle = build_weights_bundle(tmp_path / "b", list(TARGET), METRICS)
    cfg = _write_precomputed_config(tmp_path, bundle,
                                    model_yaml="model: {trainer: tests.conftest:cli_fake_trainer_with_production}")
    sc = SynthCurator.from_config(cfg)
    assert sc.model is not None
    assert sc.model_factory is None


def test_from_config_weights_path_model_is_optional(tmp_path):
    bundle = build_weights_bundle(tmp_path / "b", list(TARGET), METRICS)
    cfg = _write_precomputed_config(tmp_path, bundle)
    sc = SynthCurator.from_config(cfg)
    assert sc.model is None


def test_from_config_shipped_weights_full_path(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"""
workdir: {tmp_path / 'run'}
target_distribution: {{A: 50, B: 30, C: 20}}
inputs: {{synthetic_dir: {synthetic_dir}, real_dir: {real_dir}, resolution: 32}}
curation: {{num_images: 15, dist_group_by: [], safety_max_similarity_threshold: null}}
shipped_weights: nonexistent_preset_for_this_test
""")
    with pytest.raises(SynthCuratorError, match="not found"):
        SynthCurator.from_config(cfg)


def test_shipped_weights_isic2018_swin_b_loads_for_real(tmp_path):
    """Constructs (not a full run(), too heavy for a unit test) against the real shipped bundle -
    a regression test that the actual packaged asset stays loadable and internally consistent."""
    isic_target = {"AKIEC": 3.27, "BCC": 5.13, "BKL": 10.97, "DF": 1.15, "MEL": 11.11, "NV": 66.95, "VASC": 1.42}
    real_dir, synthetic_dir = tmp_path / "real", tmp_path / "synthetic"
    for i, cls in enumerate(isic_target):
        make_image(real_dir / cls / f"r_{cls}.png", i)
        make_image(synthetic_dir / cls / f"s_{cls}.png", i + 100)
    sc = SynthCurator(tmp_path / "run", isic_target, shipped_weights="isic2018_swin_b",
                      synthetic_dir=synthetic_dir, real_dir=real_dir, resolution=32,
                      dist_group_by=["grp"], num_images=15, safety_max_similarity_threshold=None)
    assert sc.model is None
    assert sc.classes == sorted(isic_target)
    assert set(sc.metrics) == {
        "clip_log_odds_score", "fscs_score", "clip_similarity_cancer", "clip_similarity_artifact",
        "lpips_score", "fsim_score", "brisqe_score", "psnr_score", "ssim_score",
        "diversity_score", "realism_score", "is_contribution_score",
        "fid_inception", "kid_dino", "swd_dino"}
