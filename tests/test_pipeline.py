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

from .conftest import TARGET, build_image_pool, fake_extract_features, fake_model

pytest.importorskip("PIL")

METRICS = ["fscs_score", "realism_score", "diversity_score", "is_contribution_score"]  # feature-based only
KW = dict(metrics=METRICS, group_by=[], num_images=15, safety_max_similarity_threshold=None)


def _make(tmp_path, workdir="run", **overrides):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    sc = SynthCurator(tmp_path / workdir, TARGET, model=fake_model(), synthetic_dir=synthetic_dir,
                      real_dir=real_dir, resolution=32, **{**KW, **overrides})
    std_dir = sc.standardize()
    fake_extract_features(std_dir, synthetic_dir, sc.features_dir, real_dir)
    return sc


def test_full_path_runs_linearly_and_curates(tmp_path):
    sc = _make(tmp_path, seeds_per_metric_set=2)
    manifest = sc.run()

    w = json.loads((tmp_path / "run" / "weights.json").read_text())
    assert set(w) == set(TARGET)
    assert all(abs(sum(cw.values()) - 1) < 1e-9 or sum(cw.values()) == 0 for cw in w.values())

    df = pd.read_csv(manifest)
    assert len(df) == 15 and df.target_class.value_counts().to_dict() == {"A": 8, "B": 4, "C": 3}
    assert list(df.columns) == ["source_path", "target_class", "safe_filename"]

    used = json.loads((manifest.parent / "weights_used_seed0.json").read_text())
    assert used["algorithm"] == "v15"

    # 4 metrics + random control, 2 seeds each
    preds = list((tmp_path / "run" / "tournament").glob("*_predictions.csv"))
    assert len(preds) == 5 * 2


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
    with pytest.raises(TypeError):
        SynthCurator(tmp_path / "r", TARGET, synthetic_dir=synthetic_dir, real_dir=real_dir, resolution=32, **KW)


def test_from_config_full_path(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"""
workdir: {tmp_path / 'run'}
target_distribution: {{A: 50, B: 30, C: 20}}
inputs: {{synthetic_dir: {synthetic_dir}, real_dir: {real_dir}, resolution: 32}}
metrics: {{instance: {METRICS}}}
curation: {{num_images: 15, group_by: [], safety_max_similarity_threshold: null}}
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
curation: {{num_images: 15, group_by: [], safety_max_similarity_threshold: null}}
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


def test_seeds_per_metric_set_must_be_at_least_one(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    with pytest.raises(SynthCuratorError, match="seeds_per_metric_set"):
        SynthCurator(tmp_path / "r", TARGET, model=fake_model(), synthetic_dir=synthetic_dir,
                    real_dir=real_dir, resolution=32, seeds_per_metric_set=0)


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
weights: {{scoring: tests.conftest:cli_fake_scoring}}
curation: {{num_images: 15, group_by: [], safety_max_similarity_threshold: null}}
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
weights: {{beta: 1.5}}
curation: {{num_images: 15, group_by: [], safety_max_similarity_threshold: null}}
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


def test_from_config_without_a_model_is_a_clear_error(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"""
workdir: {tmp_path / 'run'}
target_distribution: {{A: 50, B: 30, C: 20}}
inputs: {{synthetic_dir: {synthetic_dir}, real_dir: {real_dir}, resolution: 32}}
curation: {{num_images: 15, group_by: [], safety_max_similarity_threshold: null}}
""")
    with pytest.raises(SynthCuratorError, match="model is required"):
        SynthCurator.from_config(cfg)


# ---- _validate() : fail fast, before any expensive work -----------------------------------

def test_validate_rejects_a_model_without_train_and_evaluate(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    with pytest.raises(SynthCuratorError, match="train_and_evaluate"):
        SynthCurator(tmp_path / "r", TARGET, model=object(), synthetic_dir=synthetic_dir,
                    real_dir=real_dir, resolution=32, **KW)


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
