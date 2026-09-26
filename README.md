# synth-curator

The thesis curation framework (v15) as a pipeline: synthetic + real images in, best curated
synthetic training set out.

> **Status: full path only.** `run()` is a fully linear pipeline from raw images to a curated set:
> it always derives weights from a tournament with your model, and `synthetic_dir`/`real_dir`/
> `resolution`/`model` are all required. Bringing your own weights, a shipped profile, or a
> precomputed metadata/features shortcut aren't wired up yet (planned next).

```
src/synth_curator/
  pipeline.py          SynthCurator: chains the steps below; from_config(yaml)
  model_interface.py   Model: the one thing you implement to plug in your classifier
  steps/               one file per pipeline step, in run order
    standardize.py     1. resize real images to the pool's generation resolution
    features.py        2. DINOv2/Inception features, probs, KNN map, real labels
    metrics/           3. instance metrics -> metadata CSV (paired.py, single_image.py, from_features.py)
    metric_sets.py     4. one curated dataset per metric (+ random control)
    tournament.py      5. train/eval each metric set with your model
    weights.py         6. predictions -> per-class utility weights
    curate.py          7. v15 framework -> best curated set
  core/                shared math: scoring, distributional (FID/KID/SWD), sampling, safety, groups,
                       block_search, geometry, features, registry, output
```

`run()` threads each step's output explicitly into the next - standardize -> extract_features ->
compute_metrics -> curate_metric_sets -> run_tournament -> derive_weights -> curate - with no step
reaching upward to pull its own input. Each individual step still skips its own work if its
workdir output already exists (so a crashed run resumes); that resumability doesn't reintroduce
any branching into `run()` itself. One `SynthCurator` (one `workdir`) is scoped to one model -
comparing models means a separate workdir per model. Config: `config.example.yaml`.

```python
class MyModel:
    def train_and_evaluate(self, train_manifest, seed):
        ...  # train on the curated manifest, evaluate, return DataFrame[unique_image_id, true_label, predicted_label]

SynthCurator("run1", target_dist, model=MyModel(), synthetic_dir=..., real_dir=..., resolution=512).run()

# or from a config file (see config.example.yaml); also the `synth-curator config.yaml` CLI entry point
SynthCurator.from_config("config.yaml").run()
```

Tests: `pip install -e '.[dev]' && pytest`. Image-based metrics need `pip install -e '.[full]'`.
Design notes and the porting rationale: `docs/plan.md`, `docs/diagram.md`.
