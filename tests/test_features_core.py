"""core/features.py - feature-tensor access and its error paths."""
from pathlib import Path

import pandas as pd
import pytest
import torch

from synth_curator.core.features import (
    feature_path,
    load_features,
    real_class_lookup,
    stack_real,
    stack_synthetic,
)
from synth_curator.exceptions import MissingFeaturesError


def test_feature_path_naming():
    assert feature_path("/x/feat", "real", "dino") == Path("/x/feat/real_dino.pt")


def test_load_features_raises_missing_features_error_when_file_absent(tmp_path):
    with pytest.raises(MissingFeaturesError, match="missing feature file"):
        load_features(tmp_path, "real", "dino")


def test_load_features_round_trips(tmp_path):
    data = {"a.png": torch.tensor([1.0, 2.0])}
    torch.save(data, tmp_path / "synthetic_inception.pt")
    loaded = load_features(tmp_path, "synthetic", "inception")
    assert torch.equal(loaded["a.png"], data["a.png"])


def test_stack_real_preserves_insertion_order():
    real = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}
    stacked = stack_real(real)
    assert stacked.tolist() == [[1.0], [2.0]]


def test_stack_synthetic_raises_missing_features_error_for_absent_filename():
    synth = {"a.png": torch.tensor([1.0])}
    with pytest.raises(MissingFeaturesError, match="in the metadata but not in the feature dict"):
        stack_synthetic(synth, ["a.png", "missing.png"])


def test_real_class_lookup_raises_missing_features_error_when_csv_absent(tmp_path):
    with pytest.raises(MissingFeaturesError, match="real_labels.csv"):
        real_class_lookup(tmp_path)


def test_real_class_lookup_maps_stem_to_class(tmp_path):
    pd.DataFrame({"filename": ["a.png", "sub/b.png"], "class": ["A", "B"]}).to_csv(
        tmp_path / "real_labels.csv", index=False)
    assert real_class_lookup(tmp_path) == {"a": "A", "b": "B"}
