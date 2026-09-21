# synth-curator

The thesis curation framework (v15) as a pipeline. See `docs/plan.md`, `docs/diagram.md`.

```
src/synth_curator/
  pipeline.py        SynthCurator: chains the steps below
  model_interface.py Model interface: the one thing you implement to plug in your classifier
  steps/             one file per pipeline step, in run order
    standardize.py   1. resize real images to the pool's resolution
    features.py      2. DINOv2/Inception features, probs, KNN map
    metrics.py       3. 13-metric pipeline -> metadata CSV
    arms.py          4. curate one dataset per metric (+ random)
    tournament.py    5. train/eval each arm with your models
    weights.py       6. predictions -> per-class utility weights
    curate.py        7. v15 framework -> best curated set
  core/              shared math: scoring, distributional metrics, sampling, metric registry
  shipped/           optional pre-derived weight profiles (Fast mode, domain-specific)
```

```python
class MyModel:
    name = "my_effnet"
    def fit_predict(self, train_manifest, seed):
        ...  # train on the curated manifest, evaluate, return DataFrame[unique_image_id, true_label, predicted_label]

sc = SynthCurator("run1", target_dist, models=[MyModel()],
                  synthetic_dir=..., real_dir=..., resolution=512)
sc.run("my_effnet")                                   # full mode: derives weights for YOUR model

SynthCurator("run2", target_dist, metadata_csv=..., features_dir=...,
             shipped_profile="isic2018_swin_b").run()  # fast mode
```
