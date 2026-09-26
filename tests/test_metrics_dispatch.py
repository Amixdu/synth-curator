"""steps/metrics/__init__.py dispatcher - metadata-building and error paths."""
import pandas as pd
import pytest

from synth_curator.core import registry
from synth_curator.exceptions import SynthCuratorError
from synth_curator.steps.metrics import build_metadata, compute_metrics


def test_build_metadata_requires_class_subfolder(tmp_path):
    (tmp_path / "loose.png").write_bytes(b"\x89PNG\r\n")   # directly under synthetic_dir, no class folder
    with pytest.raises(SynthCuratorError, match="expected <class_name>"):
        build_metadata(tmp_path)


def test_build_metadata_raises_when_pool_is_empty(tmp_path):
    with pytest.raises(SynthCuratorError, match="no images under"):
        build_metadata(tmp_path)


def test_build_metadata_scans_class_subfolders(tmp_path):
    (tmp_path / "A").mkdir()
    (tmp_path / "A" / "x.png").write_bytes(b"\x89PNG\r\n")
    df = build_metadata(tmp_path)
    assert df.to_dict("records") == [{"filename": "A/x.png", "target_class_label": "A"}]


def test_compute_metrics_requires_filename_and_target_class_label_columns(tmp_path):
    bad = tmp_path / "meta.csv"
    pd.DataFrame({"filename": ["a.png"]}).to_csv(bad, index=False)
    with pytest.raises(SynthCuratorError, match="target_class_label"):
        compute_metrics(tmp_path, tmp_path, tmp_path, tmp_path / "out.csv", [], base_metadata=bad)


def test_compute_metrics_raises_when_a_registered_metric_has_no_calculator(tmp_path, monkeypatch):
    """Defensive check in the dispatcher: a metric registered as 'instance' but with no entry in
    STEPS/STEP_OF_METRIC must fail loudly rather than silently doing nothing."""
    monkeypatch.setitem(registry.METRIC_PROPERTIES, "fake_instance_metric", {"type": "instance", "optimize": "max"})
    base = tmp_path / "base.csv"
    pd.DataFrame({"filename": ["a.png"], "target_class_label": ["A"]}).to_csv(base, index=False)
    with pytest.raises(SynthCuratorError, match="no calculator for metrics"):
        compute_metrics(tmp_path, tmp_path, tmp_path, tmp_path / "out.csv", ["fake_instance_metric"],
                        base_metadata=base)


def test_compute_metrics_hands_back_metadata_untouched_when_no_metrics_requested(tmp_path):
    base = tmp_path / "base.csv"
    pd.DataFrame({"filename": ["a.png"], "target_class_label": ["A"]}).to_csv(base, index=False)
    out = compute_metrics(tmp_path, tmp_path, tmp_path, tmp_path / "out.csv", [], base_metadata=base)
    assert pd.read_csv(out).to_dict("records") == [{"filename": "a.png", "target_class_label": "A"}]
