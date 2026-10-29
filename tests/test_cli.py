"""The literal entry point: `synth-curator config.yaml`."""
import logging
import runpy
import sys

import pandas as pd
import pytest

from synth_curator import cli
from synth_curator.cli import main
from synth_curator.exceptions import SynthCuratorError

from .conftest import build_image_pool, fake_extract_features

pytest.importorskip("PIL")


def _config(tmp_path):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"""
workdir: {tmp_path / 'run'}
target_distribution: {{A: 50, B: 30, C: 20}}
inputs: {{synthetic_dir: {synthetic_dir}, real_dir: {real_dir}, resolution: 32}}
model: {{trainer: tests.conftest:cli_fake_trainer}}
metrics: {{instance: [fscs_score, realism_score, diversity_score, is_contribution_score]}}
curation: {{num_images: 15, dist_group_by: [], safety_max_similarity_threshold: null}}
""")
    from synth_curator import SynthCurator

    from .conftest import cli_fake_trainer
    sc = SynthCurator.from_config(config_path, model=cli_fake_trainer())
    std_dir = sc.standardize()
    fake_extract_features(std_dir, synthetic_dir, sc.features_dir, real_dir)
    return config_path


def test_cli_runs_full_path_end_to_end(tmp_path, capsys):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"""
workdir: {tmp_path / 'run'}
target_distribution: {{A: 50, B: 30, C: 20}}
inputs: {{synthetic_dir: {synthetic_dir}, real_dir: {real_dir}, resolution: 32}}
model: {{trainer: tests.conftest:cli_fake_trainer}}
metrics: {{instance: [fscs_score, realism_score, diversity_score, is_contribution_score]}}
curation: {{num_images: 15, dist_group_by: [], safety_max_similarity_threshold: null}}
""")

    # pre-seed features: DINOv2/InceptionV3 extraction needs network + heavy deps not available here
    from synth_curator import SynthCurator

    from .conftest import cli_fake_trainer
    sc = SynthCurator.from_config(config_path, model=cli_fake_trainer())
    std_dir = sc.standardize()
    fake_extract_features(std_dir, synthetic_dir, sc.features_dir, real_dir)

    assert main([str(config_path)]) == 0

    manifest_path = capsys.readouterr().out.strip()
    manifest = pd.read_csv(manifest_path)
    assert len(manifest) == 15 and manifest.target_class.value_counts().to_dict() == {"A": 8, "B": 4, "C": 3}


def test_cli_train_production_flag_trains_and_prints_the_result(tmp_path, capsys):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"""
workdir: {tmp_path / 'run'}
target_distribution: {{A: 50, B: 30, C: 20}}
inputs: {{synthetic_dir: {synthetic_dir}, real_dir: {real_dir}, resolution: 32}}
model: {{trainer: tests.conftest:cli_fake_trainer_with_production}}
metrics: {{instance: [fscs_score, realism_score, diversity_score, is_contribution_score]}}
curation: {{num_images: 15, dist_group_by: [], safety_max_similarity_threshold: null}}
""")
    from synth_curator import SynthCurator

    from .conftest import cli_fake_trainer_with_production
    sc = SynthCurator.from_config(config_path, model=cli_fake_trainer_with_production())
    std_dir = sc.standardize()
    fake_extract_features(std_dir, synthetic_dir, sc.features_dir, real_dir)

    assert main([str(config_path), "--train-production"]) == 0

    out = capsys.readouterr().out.strip()
    assert "trained_on" in out
    manifest = pd.read_csv(tmp_path / "run" / "curated" / "seed_0.csv")
    assert len(manifest) == 15   # the full pipeline still ran, same as a plain run()


def test_cli_train_production_flag_requires_the_model_to_implement_it(tmp_path):
    config_path = _config(tmp_path)   # model: tests.conftest:cli_fake_trainer, no train_production
    with pytest.raises(SynthCuratorError, match="train_production"):
        main([str(config_path), "--train-production"])


def test_cli_without_the_flag_only_calls_run(tmp_path, monkeypatch):
    """Regression: --train-production must be opt-in, not the new default."""
    from synth_curator import pipeline as pipeline_mod
    config_path = _config(tmp_path)

    def boom(self):
        raise AssertionError("train_production() must not run unless --train-production is passed")
    monkeypatch.setattr(pipeline_mod.SynthCurator, "train_production", boom)

    assert main([str(config_path)]) == 0


def test_cli_requires_a_config_argument():
    try:
        main([])
    except SystemExit as e:
        assert e.code == 2
    else:
        raise AssertionError("expected SystemExit for a missing config argument")


def test_cli_quiet_flag_requests_warning_level(tmp_path, monkeypatch, capsys):
    config_path = _config(tmp_path)
    captured = {}
    monkeypatch.setattr(cli.logging, "basicConfig", lambda **kw: captured.update(kw))
    assert main([str(config_path), "-q"]) == 0
    assert captured["level"] == logging.WARNING
    capsys.readouterr()


def test_cli_without_quiet_flag_requests_info_level(tmp_path, monkeypatch, capsys):
    config_path = _config(tmp_path)
    captured = {}
    monkeypatch.setattr(cli.logging, "basicConfig", lambda **kw: captured.update(kw))
    assert main([str(config_path)]) == 0
    assert captured["level"] == logging.INFO
    capsys.readouterr()


def test_main_module_runs_the_cli_entry_point(tmp_path, monkeypatch, capsys):
    """python -m synth_curator <config.yaml>"""
    config_path = _config(tmp_path)
    monkeypatch.setattr(sys, "argv", ["synth_curator", str(config_path)])
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("synth_curator.__main__", run_name="__main__")
    assert excinfo.value.code == 0
    manifest = pd.read_csv(capsys.readouterr().out.strip())
    assert len(manifest) == 15


def test_main_module_missing_config_argument_exits_2(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["synth_curator"])
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("synth_curator.__main__", run_name="__main__")
    assert excinfo.value.code == 2


# ---- resample: config section -----------------------------------------------------------

def _config_with(tmp_path, trainer, extra_yaml=""):
    real_dir, synthetic_dir = build_image_pool(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"""
workdir: {tmp_path / 'run'}
target_distribution: {{A: 50, B: 30, C: 20}}
inputs: {{synthetic_dir: {synthetic_dir}, real_dir: {real_dir}, resolution: 32}}
model: {{trainer: {trainer}}}
metrics: {{instance: [fscs_score, realism_score, diversity_score, is_contribution_score]}}
curation: {{num_images: 15, dist_group_by: [], safety_max_similarity_threshold: null}}
{extra_yaml}
""")
    from synth_curator import SynthCurator
    sc = SynthCurator.from_config(config_path)
    std_dir = sc.standardize()
    fake_extract_features(std_dir, synthetic_dir, sc.features_dir, real_dir)
    return config_path


def test_cli_resample_sweeps_seeds_and_curates_each(tmp_path, capsys):
    config_path = _config_with(tmp_path, "tests.conftest:cli_fake_trainer", extra_yaml="resample: {seeds: 3}")

    assert main([str(config_path)]) == 0

    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 3
    for i, line in enumerate(lines):
        assert line.startswith(f"seed={i}: ")
        manifest_path = line.split(": ", 1)[1]
        assert len(pd.read_csv(manifest_path)) == 15
        assert f"resample/seed_{i}" in manifest_path


def test_cli_resample_with_train_production_trains_each_seed(tmp_path, capsys):
    config_path = _config_with(tmp_path, "tests.conftest:cli_fake_trainer_with_production",
                               extra_yaml="resample: {seeds: 2}")

    assert main([str(config_path), "--train-production"]) == 0

    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("seed=0: ") and "trained_on" in lines[0]
    assert lines[1].startswith("seed=1: ") and "trained_on" in lines[1]
    assert lines[0] != lines[1]   # distinct manifests trained on


def test_cli_resample_reuses_weights_without_rerunning_the_tournament(tmp_path, monkeypatch, capsys):
    config_path = _config_with(tmp_path, "tests.conftest:cli_fake_trainer", extra_yaml="resample: {seeds: 3}")

    calls = []
    from synth_curator import pipeline as pipeline_mod
    real_run_tournament = pipeline_mod.SynthCurator.run_tournament

    def spy(self, metric_sets):
        calls.append(self.workdir)
        return real_run_tournament(self, metric_sets)
    monkeypatch.setattr(pipeline_mod.SynthCurator, "run_tournament", spy)

    assert main([str(config_path)]) == 0
    assert len(calls) == 1   # the tournament ran once, on the base config's own workdir - not per seed
