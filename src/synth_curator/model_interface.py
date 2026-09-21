"""The one interface a user implements to plug their own classifier into the pipeline.

The library never trains anything itself. During the tournament (step 5) it hands your model a
curated training manifest and expects raw predictions back. Everything else about the model
(architecture, framework, real training data, augmentation, test split, GPUs) is yours.

    class MyModel:
        name = "my_effnet"

        def fit_predict(self, train_manifest: Path, seed: int) -> pd.DataFrame:
            # train on the manifest (+ whatever real data you use), evaluate on your held-out set
            return pd.DataFrame({"unique_image_id": ..., "true_label": ..., "predicted_label": ...})

A plain function works too: `Model.from_function("my_effnet", fn)`.
"""
from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from .exceptions import SynthCuratorError

PREDICTION_COLUMNS = ("unique_image_id", "true_label", "predicted_label")


@runtime_checkable
class Model(Protocol):
    name: str

    def fit_predict(self, train_manifest: Path, seed: int) -> pd.DataFrame:
        """Train on `train_manifest` (CSV: source_path, target_class, safe_filename), evaluate,
        return one row per evaluated image with PREDICTION_COLUMNS."""
        ...


class _FunctionModel:
    def __init__(self, name: str, fn: Callable[[Path, int], pd.DataFrame]):
        self.name, self._fn = name, fn

    def fit_predict(self, train_manifest: Path, seed: int) -> pd.DataFrame:
        return self._fn(train_manifest, seed)


def model_from_function(name: str, fn: Callable[[Path, int], pd.DataFrame]) -> Model:
    return _FunctionModel(name, fn)


def load_model(name: str, trainer: str, params: dict[str, Any] | None = None) -> Model:
    """Config route: `trainer` is a dotted path 'package.module:ClassOrFactory', called with **params."""
    module, _, attr = trainer.partition(":")
    if not attr:
        raise SynthCuratorError(f"trainer {trainer!r} must look like 'package.module:Name'")
    obj = getattr(importlib.import_module(module), attr)(**(params or {}))
    obj.name = name
    return obj


def check_predictions(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    missing = set(PREDICTION_COLUMNS) - set(df.columns)
    if missing:
        raise SynthCuratorError(f"model {model_name!r} returned predictions missing columns: {sorted(missing)}")
    return df
