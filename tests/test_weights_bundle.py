"""weights_bundle.py - loading/validating weights.json + meta.json bundles."""
import json

import pytest

from synth_curator import __version__, weights_bundle as wb
from synth_curator.exceptions import SynthCuratorError
from tests.conftest import build_weights_bundle


def test_load_weights_path_reads_a_bundle(tmp_path):
    bundle = build_weights_bundle(tmp_path / "b", ["A", "B"], ["m1", "m2"])
    weights, meta = wb.load_weights_path(bundle)
    assert weights == {"A": {"m1": 1.0, "m2": 1.0}, "B": {"m1": 1.0, "m2": 1.0}}
    assert meta["classes"] == ["A", "B"]
    assert meta["metrics"] == ["m1", "m2"]


def test_load_weights_path_missing_weights_json_is_a_clear_error(tmp_path):
    bundle = build_weights_bundle(tmp_path / "b", ["A"], ["m1"])
    (bundle / "weights.json").unlink()
    with pytest.raises(SynthCuratorError, match="weights.json not found"):
        wb.load_weights_path(bundle)


def test_load_weights_path_missing_meta_json_is_a_clear_error(tmp_path):
    bundle = build_weights_bundle(tmp_path / "b", ["A"], ["m1"])
    (bundle / "meta.json").unlink()
    with pytest.raises(SynthCuratorError, match="meta.json not found"):
        wb.load_weights_path(bundle)


def test_load_weights_path_rejects_unsupported_schema_version(tmp_path):
    bundle = build_weights_bundle(tmp_path / "b", ["A"], ["m1"], schema_version=99)
    with pytest.raises(SynthCuratorError, match="schema_version"):
        wb.load_weights_path(bundle)


def test_load_weights_path_rejects_classes_mismatch_between_weights_and_meta(tmp_path):
    bundle = build_weights_bundle(tmp_path / "b", ["A", "B"], ["m1"])
    meta = json.loads((bundle / "meta.json").read_text())
    meta["classes"] = ["A", "B", "C"]   # doesn't match weights.json's actual keys anymore
    (bundle / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(SynthCuratorError, match="classes"):
        wb.load_weights_path(bundle)


def test_load_weights_path_rejects_metrics_mismatch_between_weights_and_meta(tmp_path):
    bundle = build_weights_bundle(tmp_path / "b", ["A"], ["m1", "m2"])
    meta = json.loads((bundle / "meta.json").read_text())
    meta["metrics"] = ["m1"]   # doesn't match weights.json's actual per-class keys anymore
    (bundle / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(SynthCuratorError, match="metrics"):
        wb.load_weights_path(bundle)


def test_load_weights_path_rejects_meta_missing_required_keys(tmp_path):
    bundle = build_weights_bundle(tmp_path / "b", ["A"], ["m1"])
    (bundle / "meta.json").write_text(json.dumps({"schema_version": 1}))
    with pytest.raises(SynthCuratorError, match="classes.*metrics|metrics.*classes"):
        wb.load_weights_path(bundle)


def test_load_weights_path_warns_on_synth_curator_version_mismatch(tmp_path, caplog):
    bundle = build_weights_bundle(tmp_path / "b", ["A"], ["m1"],
                                  extra_meta={"provenance": {"synth_curator_version": "0.0.0-not-real"}})
    with caplog.at_level("WARNING"):
        wb.load_weights_path(bundle)
    assert "0.0.0-not-real" in caplog.text
    assert __version__ in caplog.text


def test_load_weights_path_no_warning_when_version_matches(tmp_path, caplog):
    bundle = build_weights_bundle(tmp_path / "b", ["A"], ["m1"],
                                  extra_meta={"provenance": {"synth_curator_version": __version__}})
    with caplog.at_level("WARNING"):
        wb.load_weights_path(bundle)
    assert caplog.text == ""


def test_load_weights_path_no_warning_when_provenance_absent(tmp_path, caplog):
    bundle = build_weights_bundle(tmp_path / "b", ["A"], ["m1"])
    with caplog.at_level("WARNING"):
        wb.load_weights_path(bundle)
    assert caplog.text == ""


def test_list_shipped_weights_includes_isic2018_presets():
    assert {"isic2018_swin_b", "isic2018_resnet50"} <= set(wb.list_shipped_weights())


@pytest.mark.parametrize("name,model_name_prefix", [
    ("isic2018_swin_b", "Swin-B"),
    ("isic2018_resnet50", "ResNet-50"),
])
def test_load_shipped_weights_isic2018_presets(name, model_name_prefix):
    weights, meta = wb.load_shipped_weights(name)
    assert set(weights) == set(meta["classes"]) == {"AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"}
    assert meta["schema_version"] == 1
    assert meta["model"]["name"].startswith(model_name_prefix)


def test_load_shipped_weights_unknown_name_is_a_clear_error():
    with pytest.raises(SynthCuratorError, match="not found"):
        wb.load_shipped_weights("no_such_preset_at_all")
