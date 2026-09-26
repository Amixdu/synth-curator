"""The one interface a user implements to plug their own classifier into the pipeline.

The library never trains anything itself. During the tournament (step 5) it hands your model a
curated training manifest and expects raw predictions back. Everything else about the model
(architecture, framework, real training data, augmentation, test split, GPUs) is yours.

    class MyModel:
        def train_and_evaluate(self, train_manifest: Path, seed: int) -> pd.DataFrame:
            # train on the manifest (+ whatever real data you use), evaluate on your held-out set
            return pd.DataFrame({"unique_image_id": ..., "true_label": ..., "predicted_label": ...})

A plain function works too: `model_from_function(fn)`. One SynthCurator run uses exactly one
model; if you want to compare classifiers, use a separate workdir per model.
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
    def train_and_evaluate(self, train_manifest: Path, seed: int) -> pd.DataFrame:
        """Train on `train_manifest` (CSV: source_path, target_class, safe_filename), evaluate,
        return one row per evaluated image with PREDICTION_COLUMNS."""
        ...


class _FunctionModel:
    def __init__(self, fn: Callable[[Path, int], pd.DataFrame]):
        self._fn = fn

    def train_and_evaluate(self, train_manifest: Path, seed: int) -> pd.DataFrame:
        return self._fn(train_manifest, seed)


def model_from_function(fn: Callable[[Path, int], pd.DataFrame]) -> Model:
    return _FunctionModel(fn)


def load_model(trainer: str, params: dict[str, Any] | None = None) -> Model:
    """Config route: `trainer` is a dotted path 'package.module:ClassOrFactory', called with **params."""
    module, _, attr = trainer.partition(":")
    if not attr:
        raise SynthCuratorError(f"trainer {trainer!r} must look like 'package.module:Name'")
    return getattr(importlib.import_module(module), attr)(**(params or {}))


def check_predictions(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(PREDICTION_COLUMNS) - set(df.columns)
    if missing:
        raise SynthCuratorError(f"the model's predictions are missing columns: {sorted(missing)}")
    return df
