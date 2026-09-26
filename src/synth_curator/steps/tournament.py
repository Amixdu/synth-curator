"""Step 5. Train + evaluate your model on each metric set, once per seed.

Training is the user's `Model.train_and_evaluate` (see model_interface.py). One seed per metric set
(default) is a weighting signal, not a statistical claim; more seeds average out training noise.
Per-(metric_set, seed) predictions are cached in out_dir so an interrupted tournament resumes.
Port of final_rerun/scripts/6_tournament_final.py (orchestration only).
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..model_interface import Model, check_predictions

log = logging.getLogger(__name__)


def run(metric_sets: dict[str, dict[int, Path]], model: Model, out_dir: Path,
       seeds: list[int]) -> dict[str, pd.DataFrame]:
    """
    metric_sets: {metric_set_name: {seed: manifest_csv}} — output of steps.metric_sets.curate_metric_sets, e.g.
        {"lpips_score": {0: Path(".../lpips_score.csv")},
         "random_control": {0: Path(".../random_control_seed0.csv"), 1: Path(".../random_control_seed1.csv")}}
    model: Model — your classifier (see model_interface.py)
    out_dir: Path — predictions cache root; written to as out_dir/<metric_set>_seed<seed>_predictions.csv
    seeds: [int, ...] — e.g. [0, 1, 2]; one training run per (metric_set, seed)
    returns: {metric_set_name: DataFrame[unique_image_id, true_label, predicted_label, seed]}
    """
    out_dir = Path(out_dir)
    results: dict[str, pd.DataFrame] = {}
    for name, manifests in metric_sets.items():
        runs = []
        for seed in seeds:
            cache = out_dir / f"{name}_seed{seed}_predictions.csv"
            if cache.exists():
                preds = pd.read_csv(cache)
            else:
                log.info("training metric_set=%s seed=%d", name, seed)
                preds = check_predictions(model.train_and_evaluate(manifests[seed], seed))
                cache.parent.mkdir(parents=True, exist_ok=True)
                preds.to_csv(cache, index=False)
            runs.append(preds.assign(seed=seed))
        results[name] = pd.concat(runs, ignore_index=True)
    return results
