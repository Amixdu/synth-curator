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

`run()` trains every job in one process, one after another. `run_parallel()` does the same jobs,
but spread across GPU worker processes (one per id in gpu_ids) - see model_interface.py's
model_factory docs for the constraints on the factory it's given.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
from pathlib import Path
from typing import Any

import pandas as pd

from ..exceptions import SynthCuratorError
from ..model_interface import Model, ModelFactory, check_predictions

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


def run_parallel(metric_sets: dict[str, dict[int, Path]], model_factory: ModelFactory, gpu_ids: list[int],
                 out_dir: Path, force_retrain: bool = False) -> dict[str, pd.DataFrame]:
    """Same jobs and same cache as run(), but every job not already cached is trained on a GPU
    worker process instead of here: one process per id in gpu_ids, each building its own model via
    model_factory(gpu_id) once and reusing it for every job it picks up."""
    out_dir = Path(out_dir)
    jobs: list[tuple[str, int, Path]] = []
    outputs: list[tuple[str, int, pd.DataFrame]] = []
    for name, manifests in metric_sets.items():
        for seed, manifest_csv in manifests.items():
            cache = out_dir / f"{name}_seed{seed}_predictions.csv"
            if cache.exists() and not force_retrain:
                log.info("reusing cached predictions: metric_set=%s seed=%d (%s) - delete this "
                        "file, the workdir, or pass force_retrain=True to force a retrain", name, seed, cache)
                outputs.append((name, seed, pd.read_csv(cache)))
            else:
                jobs.append((name, seed, manifest_csv))

    if jobs:
        log.info("training %d job(s) across %d GPU worker(s): %s", len(jobs), len(gpu_ids), gpu_ids)
        outputs.extend(_train_on_gpus(jobs, model_factory, gpu_ids, out_dir))

    grouped: dict[str, list[pd.DataFrame]] = {}
    for name, seed, preds in outputs:
        grouped.setdefault(name, []).append(preds.assign(seed=seed))
    return {name: pd.concat(runs, ignore_index=True) for name, runs in grouped.items()}


def _worker_loop(gpu_id: int, model_factory: ModelFactory, job_queue: Any, result_queue: Any) -> None:
    model = model_factory(gpu_id)
    while True:
        job = job_queue.get()
        if job is None:                    # sentinel: this worker has no more jobs
            return
        name, seed, manifest_csv = job
        try:
            preds = check_predictions(model.train_and_evaluate(manifest_csv, seed))
            result_queue.put(("ok", name, seed, preds))
        except Exception as e:  # noqa: BLE001 - marshal as a plain message; not every exception pickles
            result_queue.put(("error", name, seed, f"{type(e).__name__}: {e}"))


def _train_on_gpus(jobs: list[tuple[str, int, Path]], model_factory: ModelFactory, gpu_ids: list[int],
                   out_dir: Path) -> list[tuple[str, int, pd.DataFrame]]:
    ctx = mp.get_context("spawn")          # fork + CUDA is unsafe; each worker starts a clean process
    job_queue, result_queue = ctx.Queue(), ctx.Queue()
    for job in jobs:
        job_queue.put(job)
    for _ in gpu_ids:
        job_queue.put(None)                # one stop sentinel per worker

    # not daemonic: your model may itself use multiprocessing (e.g. DataLoader workers), and
    # daemonic processes aren't allowed to have children.
    workers = [ctx.Process(target=_worker_loop, args=(gpu_id, model_factory, job_queue, result_queue))
              for gpu_id in gpu_ids]
    for w in workers:
        w.start()

    outputs = []
    try:
        for _ in range(len(jobs)):
            status, name, seed, payload = result_queue.get()
            if status == "error":
                raise SynthCuratorError(
                    f"training failed for metric_set={name!r} seed={seed} on a GPU worker: {payload}")
            log.info("finished training metric_set=%s seed=%d", name, seed)
            cache = out_dir / f"{name}_seed{seed}_predictions.csv"
            cache.parent.mkdir(parents=True, exist_ok=True)
            payload.to_csv(cache, index=False)
            outputs.append((name, seed, payload))
    finally:
        for w in workers:
            w.join(timeout=5)
        for w in workers:
            if w.is_alive():
                w.terminate()
    return outputs
