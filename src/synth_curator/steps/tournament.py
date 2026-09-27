"""Step 5. Train + evaluate your model on each metric set, once per seed that metric set has.

Training is the user's `Model.train_and_evaluate` (see model_interface.py). How many seeds a given
metric set trains is decided entirely by `metric_sets` itself (see steps.metric_sets): instance
metrics carry exactly one seed (deterministic data, nothing to average out), while distributional
metrics and random_control carry one per tournament seed (their curated data is re-randomized each
time, so repeats are a weighting signal, not a statistical claim, at 1 seed - more averages out
sampling + training noise). Per-(metric_set, seed) predictions are cached in out_dir so an
interrupted tournament resumes. Port of final_rerun/scripts/6_tournament_final.py (orchestration
only). `force_retrain` bypasses the cache entirely (e.g. after changing the model); cache reuse is
always logged otherwise.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..model_interface import Model, check_predictions

log = logging.getLogger(__name__)


def run(metric_sets: dict[str, dict[int, Path]], model: Model, out_dir: Path,
       force_retrain: bool = False) -> dict[str, pd.DataFrame]:
    """
    metric_sets: {metric_set_name: {seed: manifest_csv}} — output of steps.metric_sets.curate_metric_sets, e.g.
        {"lpips_score": {0: Path(".../lpips_score.csv")},
         "random_control": {0: Path(".../random_control_seed0.csv"), 1: Path(".../random_control_seed1.csv")}}
        each metric set's own dict of seeds decides how many times it trains here - nothing else does.
    model: Model — your classifier (see model_interface.py)
    out_dir: Path — predictions cache root; written to as out_dir/<metric_set>_seed<seed>_predictions.csv
    force_retrain: ignore any existing cache and retrain everything, overwriting it
    returns: {metric_set_name: DataFrame[unique_image_id, true_label, predicted_label, seed]}
    """
    out_dir = Path(out_dir)
    results: dict[str, pd.DataFrame] = {}
    for name, manifests in metric_sets.items():
        runs = []
        for seed, manifest_csv in manifests.items():
            cache = out_dir / f"{name}_seed{seed}_predictions.csv"
            if cache.exists() and not force_retrain:
                log.info("reusing cached predictions: metric_set=%s seed=%d (%s) - delete this "
                        "file, the workdir, or pass force_retrain=True to force a retrain", name, seed, cache)
                preds = pd.read_csv(cache)
            else:
                log.info("training metric_set=%s seed=%d", name, seed)
                preds = check_predictions(model.train_and_evaluate(manifest_csv, seed))
                cache.parent.mkdir(parents=True, exist_ok=True)
                preds.to_csv(cache, index=False)
            runs.append(preds.assign(seed=seed))
        results[name] = pd.concat(runs, ignore_index=True)
    return results
