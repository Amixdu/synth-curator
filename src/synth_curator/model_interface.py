"""The interface(s) a user implements to plug their own classifier into the pipeline.

The library never trains anything itself. There are two, separate methods, for two different
purposes - implement the first always, the second only if you want `train_production()`:

    class MyModel:
        def train_and_evaluate(self, train_manifest: Path, seed: int) -> pd.DataFrame:
            # REQUIRED. Called ~16x per seed during the tournament (step 5), once per metric set,
            # purely to produce a ranking signal - keep this cheap. Train on the manifest (+
            # whatever real data you use), evaluate on your held-out set, return predictions.
            return pd.DataFrame({"unique_image_id": ..., "true_label": ..., "predicted_label": ...})

        def train_production(self, train_manifest: Path) -> Any:
            # OPTIONAL. Called exactly once, by train_production(), on the final curated set - the
            # one training run that's actually worth being expensive. Return whatever you want
            # (a checkpoint path, a wrapped model, anything) - the library never inspects it.
            ...
            return checkpoint_path

Everything about *how* you train - architecture, framework, real training data, augmentation,
your validation split, GPUs - is yours in both methods. A plain function works too for the first
one: `model_from_function(fn)`. One SynthCurator run uses exactly one model; if you want to
compare classifiers, use a separate workdir per model.

If either method picks a checkpoint internally (e.g. best-of-N epochs), pick it by the same
metric the tournament scores predictions with - by default F-beta (`SynthCurator(...,
metric_beta=...)`, default 2.0 = F2), or whatever you passed as `SynthCurator(..., scoring=...)`
if you overrode it. The library only defines that function's shape (`(true_labels, pred_labels,
classes) -> {class_name: score}`), not any particular metric, so this is a small bit of code you
write once yourself. Otherwise the checkpoint you consider "best" during training may not be the
one the tournament actually credits.
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
    """The required interface: enough to run() and derive weights."""

    def train_and_evaluate(self, train_manifest: Path, seed: int) -> pd.DataFrame:
        """Train on `train_manifest` (CSV: source_path, target_class, safe_filename), evaluate,
        return one row per evaluated image with PREDICTION_COLUMNS."""
        ...


@runtime_checkable
class ProductionModel(Model, Protocol):
    """Model, plus the optional method needed for train_production(). Not required for run()."""

    def train_production(self, train_manifest: Path) -> Any:
        """Train the model you'll actually deploy, on the final curated set. Called once, never
        swept across seeds - manage your own reproducibility if you care about it. Return
        anything; the library passes it straight back to the caller without inspecting it."""
        ...


class _FunctionModel:
    def __init__(self, fn: Callable[[Path, int], pd.DataFrame]):
        self._fn = fn

    def train_and_evaluate(self, train_manifest: Path, seed: int) -> pd.DataFrame:
        return self._fn(train_manifest, seed)


def model_from_function(fn: Callable[[Path, int], pd.DataFrame]) -> Model:
    return _FunctionModel(fn)


def import_dotted(path: str) -> Any:
    """'package.module:attr' -> the attribute itself, uncalled. Shared by every config field that
    names Python code by dotted path (a model trainer, a custom weights.scoring function, ...)."""
    module, _, attr = path.partition(":")
    if not attr:
        raise SynthCuratorError(f"{path!r} must look like 'package.module:Name'")
    try:
        imported = importlib.import_module(module)
    except ImportError as e:
        raise SynthCuratorError(f"{path!r}: could not import module {module!r} ({e})") from e
    try:
        return getattr(imported, attr)
    except AttributeError:
        raise SynthCuratorError(f"{path!r}: module {module!r} has no attribute {attr!r}") from None


def load_model(trainer: str, params: dict[str, Any] | None = None) -> Model:
    """Config route: `trainer` is a dotted path 'package.module:ClassOrFactory', called with **params."""
    return import_dotted(trainer)(**(params or {}))


def check_predictions(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(PREDICTION_COLUMNS) - set(df.columns)
    if missing:
        raise SynthCuratorError(f"the model's predictions are missing columns: {sorted(missing)}")
    return df
