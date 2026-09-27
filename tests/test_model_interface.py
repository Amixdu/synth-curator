import pandas as pd
import pytest

from synth_curator.exceptions import SynthCuratorError
from synth_curator.model_interface import (
    check_predictions, import_dotted, load_model, load_model_factory, model_from_function)


def test_check_predictions_passes_through_a_valid_frame():
    df = pd.DataFrame({"unique_image_id": ["a"], "true_label": ["A"], "predicted_label": ["A"]})
    assert check_predictions(df) is df


def test_check_predictions_raises_synth_curator_error_not_keyerror_when_columns_missing():
    df = pd.DataFrame({"true_label": ["A"]})
    with pytest.raises(SynthCuratorError) as excinfo:
        check_predictions(df)
    assert not isinstance(excinfo.value, (KeyError, AssertionError))
    assert "unique_image_id" in str(excinfo.value) and "predicted_label" in str(excinfo.value)


def test_check_predictions_reports_every_missing_column():
    df = pd.DataFrame({"unique_image_id": ["a"]})
    with pytest.raises(SynthCuratorError, match=r"predicted_label.*true_label|true_label.*predicted_label"):
        check_predictions(df)


def test_load_model_dotted_path_calls_class_with_params():
    model = load_model("tests.conftest:cli_fake_trainer", params=None)
    assert hasattr(model, "train_and_evaluate")


def test_load_model_rejects_a_path_without_a_colon():
    with pytest.raises(SynthCuratorError, match="must look like"):
        load_model("tests.conftest.cli_fake_trainer")


def test_import_dotted_wraps_a_bad_module_as_synth_curator_error_not_import_error():
    with pytest.raises(SynthCuratorError) as excinfo:
        import_dotted("no_such_module_at_all:something")
    assert not isinstance(excinfo.value, ImportError)
    assert "no_such_module_at_all" in str(excinfo.value)


def test_import_dotted_wraps_a_bad_attribute_as_synth_curator_error_not_attribute_error():
    with pytest.raises(SynthCuratorError) as excinfo:
        import_dotted("tests.conftest:no_such_attribute_at_all")
    assert not isinstance(excinfo.value, AttributeError)
    assert "no_such_attribute_at_all" in str(excinfo.value)


def test_load_model_factory_calls_trainer_with_gpu_id_per_worker():
    factory = load_model_factory("tests.conftest:cli_fake_model_factory", params=None)
    model = factory(3)
    assert model.gpu_id == 3


def test_load_model_factory_forwards_params_alongside_gpu_id():
    factory = load_model_factory("tests.conftest:cli_fake_model_factory", params={"arch": "resnet50"})
    model = factory(1)   # doesn't raise on the extra kwarg, and gpu_id still lands correctly
    assert model.gpu_id == 1


def test_model_from_function_wraps_a_plain_callable():
    calls = []

    def fn(manifest, seed):
        calls.append((manifest, seed))
        return pd.DataFrame({"unique_image_id": [], "true_label": [], "predicted_label": []})

    model = model_from_function(fn)
    model.train_and_evaluate("m.csv", 3)
    assert calls == [("m.csv", 3)]
