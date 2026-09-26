import pandas as pd
import pytest

from synth_curator.exceptions import SynthCuratorError
from synth_curator.model_interface import check_predictions, load_model, model_from_function


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


def test_model_from_function_wraps_a_plain_callable():
    calls = []

    def fn(manifest, seed):
        calls.append((manifest, seed))
        return pd.DataFrame({"unique_image_id": [], "true_label": [], "predicted_label": []})

    model = model_from_function(fn)
    model.train_and_evaluate("m.csv", 3)
    assert calls == [("m.csv", 3)]
