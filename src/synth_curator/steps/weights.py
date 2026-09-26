"""Step 6. Tournament predictions -> per-class utility weights {class: {metric: weight}} (each class sums to 1).

Port of v15 calculate_per_class_utilities. Two formulas:
    worst_anchored   u = (F2 - min_m F2) + 1e-4     weights ALL metrics, nothing zeroed
    random_anchored  u = max(0, F2 - F2_random)     the sieve: metrics not beating random get 0
`formula` fixes the instance-vs-distributional balance. `dist_formula`, if different, only
redistributes weight WITHIN the distributional group (computing both over one pool would
collapse the distributional share). Per-class F2 comes from the saved per-image predictions.

Predictions: {metric_set_name: DataFrame[true_label, predicted_label, (seed)]}, where metric_set_name
is a metric name or "random_control". Labels are class names, or ints indexing the sorted classes.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..core.registry import METRIC_PROPERTIES, is_distributional
from ..exceptions import WeightsError

log = logging.getLogger(__name__)

RANDOM_METRIC_SET = "random_control"


def _labels_to_names(labels, classes: list[str]) -> np.ndarray:
    arr = np.asarray(labels)
    if np.issubdtype(arr.dtype, np.integer):
        return np.asarray(classes, dtype=object)[arr]
    return arr.astype(object)


def fbeta_per_class(true_labels, pred_labels, classes: list[str], beta: float = 2.0) -> dict[str, float]:
    """Per-class F-beta from counts (F2 = 5TP / (5TP + 4FN + FP)); an empty class scores 0."""
    t, p = _labels_to_names(true_labels, classes), _labels_to_names(pred_labels, classes)
    b2 = beta ** 2
    out = {}
    for class_name in classes:
        tp = int(np.sum((t == class_name) & (p == class_name)))
        fn = int(np.sum((t == class_name) & (p != class_name)))
        fp = int(np.sum((t != class_name) & (p == class_name)))
        denom = (1 + b2) * tp + b2 * fn + fp
        out[class_name] = ((1 + b2) * tp / denom) if denom > 0 else 0.0
    return out


def _worst_anchored(scores: dict[str, float], _rand: float | None) -> dict[str, float]:
    lo = min(scores.values())
    return {m: (v - lo) + 1e-4 for m, v in scores.items()}


def _random_anchored(scores: dict[str, float], rand: float | None) -> dict[str, float]:
    return {m: max(0.0, v - rand) for m, v in scores.items()}


FORMULAS = {"worst_anchored": _worst_anchored, "random_anchored": _random_anchored}


def derive_weights(predictions: dict[str, pd.DataFrame], classes: list[str],
                   formula: str = "worst_anchored", dist_formula: str | None = "random_anchored",
                   active_metrics: list[str] | None = None) -> dict[str, dict[str, float]]:
    for f in (formula, dist_formula):
        if f is not None and f not in FORMULAS:
            raise WeightsError(f"unknown formula {f!r}; known: {sorted(FORMULAS)}")

    classes = sorted(classes)
    allowed = [m for m in (active_metrics or METRIC_PROPERTIES) if m in METRIC_PROPERTIES]

    # mean per-class F2 across seeds, per metric set
    metric_set_f2: dict[str, dict[str, float]] = {}
    for name, df in predictions.items():
        runs = [d for _, d in df.groupby("seed")] if "seed" in df.columns else [df]
        per_run = [fbeta_per_class(d["true_label"].values, d["predicted_label"].values, classes) for d in runs]
        metric_set_f2[name] = {c: float(np.mean([r[c] for r in per_run])) for c in classes}

    metric_f2 = {m: v for m, v in metric_set_f2.items() if m in allowed}
    if not metric_f2:
        raise WeightsError("no metric sets found in the tournament predictions")
    needs_random = "random_anchored" in (formula, dist_formula)
    if needs_random and RANDOM_METRIC_SET not in metric_set_f2:
        raise WeightsError(f"random_anchored needs the {RANDOM_METRIC_SET!r} metric set")
    rand = metric_set_f2.get(RANDOM_METRIC_SET)

    out: dict[str, dict[str, float]] = {}
    for class_name in classes:
        scores = {m: metric_f2[m][class_name] for m in metric_f2}
        r = rand[class_name] if rand else None
        u = FORMULAS[formula](scores, r)
        total = sum(u.values())
        if total <= 0:
            log.warning("[%s] no metric beats random - this class will sample uniformly", class_name)
            out[class_name] = {m: 0.0 for m in scores}
            continue
        out[class_name] = {m: v / total for m, v in u.items()}

        if dist_formula is not None and dist_formula != formula:
            dist = [m for m in out[class_name] if is_distributional(m)]
            share = sum(out[class_name][m] for m in dist)
            u2 = FORMULAS[dist_formula]({m: scores[m] for m in dist}, r)
            t2 = sum(u2.values())
            if t2 <= 0:
                log.warning("[%s] no distributional metric survives %s - keeping the %s split", class_name, dist_formula, formula)
            else:
                for m in dist:
                    out[class_name][m] = share * u2[m] / t2
    return out
