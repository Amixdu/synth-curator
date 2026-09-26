# synth-curator

**Pick the best synthetic images for training a classifier — weighted by which quality metrics
actually improve it, not by which metrics sound reasonable.**

Generative models produce huge pools of synthetic images. The usual way to pick a training subset
is to filter on one convenient metric (FID, usually) and hope it correlates with what actually
helps a downstream classifier. It often doesn't, and different metrics matter for different
classes. `synth-curator` instead runs a small tournament — train your own classifier on a
dataset curated by *each* metric alone, see which ones actually move the needle per class — and
uses the result to build one final, per-class-weighted curated set.

- **No metric is assumed useful.** Every metric earns its weight by measurably beating a random
  baseline on your own classifier and your own data.
- **Weights are per class, not global.** A metric that helps for one lesion type can be useless,
  or actively harmful, for another.
- **Domain-agnostic.** Nothing in the library assumes skin lesions, or any particular image
  domain — you provide the classes, the pool, and the classifier.
- **Two-tier training.** A cheap, repeated signal drives the metric ranking; a separate, optional,
  one-time call trains the model you actually deploy. You decide how cheap or expensive each is.

> **Status:** the full pipeline works end to end (raw images in, a curated set or a trained model
> out), but it always derives weights from a fresh tournament with your model — bringing your own
> precomputed weights isn't wired up yet.

## Install

```bash
pip install synth-curator             # core: enough to run the pipeline with feature-based metrics
pip install synth-curator[full]       # + PSNR/SSIM/FSIM/LPIPS/BRISQUE/CLIP calculators
```

## The idea, in one picture

```
raw images  ->  standardize  ->  extract features  ->  compute per-image metrics
                                                              |
                          curate one dataset per metric (+ a random control)
                                                              |
              train YOUR classifier on each one (cheap, repeated) - train_and_evaluate
                                                              |
              turn "which metric's dataset produced the best classifier"
                    into per-class weights (per class, not global)
                                                              |
        curate the final set: weighted composite score, temperature sampling,
                    optional memorization filter, optional block search
                                                              |
                    a curated manifest CSV  ->  optionally: train the model
                          you'll actually deploy on it (once) - train_production
```

Every step is idempotent — it checks its own output first and skips the work if it's already
there, so an interrupted run just resumes. See `docs/design.md` for the exact call sequence.

## Quickstart

A complete, working example: curating a pool of synthetic cat/dog images against a small real
reference set, then training a deployable ResNet.

### 1. Lay out your data

```
synthetic_pool/
  cats/img0001.png
  cats/img0002.png
  dogs/img0001.png
  ...
real_reference/
  cats/real0001.jpg
  dogs/real0001.jpg
  ...
```

Any two-level `<class>/<image>` layout works for both. `real_reference` is whatever real dataset
you'd train the classifier on anyway — it also doubles as the reference distribution the
distributional metrics (FID/KID/SWD) compare the synthetic pool against.

### 2. Implement your model: two methods, two very different jobs

```python
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms

TRANSFORM = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])


class ImagesFromManifest(Dataset):
    """Tiny Dataset wrapping a synth-curator manifest (source_path, target_class, safe_filename)."""

    def __init__(self, manifest: pd.DataFrame, class_to_idx: dict):
        self.manifest, self.class_to_idx = manifest.reset_index(drop=True), class_to_idx

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, i):
        row = self.manifest.iloc[i]
        image = TRANSFORM(datasets.folder.default_loader(row["source_path"]))
        return image, self.class_to_idx[row["target_class"]]


class ResNetClassifier:
    """Both methods train a ResNet on the manifest they're given - the only difference is how much
    training you're willing to pay for. train_and_evaluate runs ~16x per seed during the
    tournament, purely to rank metrics, so it stays cheap (few epochs). train_production runs
    exactly once, on the final curated set, so it's the one worth doing properly."""

    def __init__(self, val_dir: str, class_to_idx: dict, device: str = "cuda"):
        self.val_dir, self.class_to_idx, self.device = Path(val_dir), class_to_idx, device

    def _train(self, train_manifest: Path, epochs: int, seed: int) -> torch.nn.Module:
        torch.manual_seed(seed)
        manifest = pd.read_csv(train_manifest)
        loader = DataLoader(ImagesFromManifest(manifest, self.class_to_idx), batch_size=32, shuffle=True)
        model = models.resnet18(num_classes=len(self.class_to_idx)).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)

        model.train()
        for _ in range(epochs):
            for x, y in loader:
                opt.zero_grad()
                loss = torch.nn.functional.cross_entropy(model(x.to(self.device)), y.to(self.device))
                loss.backward()
                opt.step()
        return model

    def _evaluate(self, model: torch.nn.Module) -> pd.DataFrame:
        val_ds = datasets.ImageFolder(self.val_dir, transform=TRANSFORM)
        model.eval()
        ids, true, pred = [], [], []
        with torch.no_grad():
            for i, (x, y) in enumerate(DataLoader(val_ds, batch_size=64)):
                p = model(x.to(self.device)).argmax(1).cpu()
                ids += [val_ds.samples[j][0] for j in range(i * 64, i * 64 + len(y))]
                true += y.tolist()
                pred += p.tolist()
        return pd.DataFrame({"unique_image_id": ids, "true_label": true, "predicted_label": pred})

    # REQUIRED: called ~16x per seed during the tournament. Keep this cheap.
    def train_and_evaluate(self, train_manifest: Path, seed: int) -> pd.DataFrame:
        model = self._train(train_manifest, epochs=3, seed=seed)
        return self._evaluate(model)

    # OPTIONAL: called once, on the final curated set. Worth spending real compute on.
    def train_production(self, train_manifest: Path) -> Path:
        model = self._train(train_manifest, epochs=30, seed=0)
        checkpoint_path = Path("checkpoints/resnet18_final.pt")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        return checkpoint_path   # anything you want - a Path, a dict, a wrapped object
```

That's the entire interface. `train_and_evaluate` is required and must return a DataFrame with
columns `unique_image_id`, `true_label`, `predicted_label`. `train_production` is optional, only
needed if you want the last step below, and can return *anything* — the library never inspects it.
Everything about *how* you train — framework, architecture, augmentation, your validation split,
which checkpoint you keep if you train for several epochs — is entirely yours in both.

### 3. Construct the pipeline

```python
from synth_curator import SynthCurator

class_to_idx = {"cats": 0, "dogs": 1}

sc = SynthCurator(
    workdir="runs/cats_vs_dogs",
    target_distribution={"cats": 50.0, "dogs": 50.0},   # class proportions in the curated set (%), sums to 100
    model=ResNetClassifier(val_dir="real_reference_holdout", class_to_idx=class_to_idx),
    synthetic_dir="synthetic_pool",
    real_dir="real_reference",
    resolution=512,               # your generator's output resolution
    num_images=2000,               # images in the final curated set
    group_by=[],                   # no generation-group metadata? leave this empty
)
```

Construction validates everything up front — a missing directory, an empty one, a model without
`train_and_evaluate`, or a `target_distribution` that doesn't sum to ~100 all raise immediately
here, before any standardizing, feature extraction, or training happens.

The tournament scores each metric set's predictions with F-beta by default (`metric_beta=2.0` =
F2, favoring recall). Pass a different `metric_beta` for a different F-beta, or replace the
metric entirely with `scoring=my_function`, a function `(true_labels, pred_labels, classes) ->
{class_name: score}` you write yourself (`metric_beta` is then unused - construction raises if
you set both). Whatever you pick, use the exact same computation in your model's own checkpoint
selection too if it has one, so "best during training" and "best per the tournament" agree.

### 4. Get the curated set and the derived weights

```python
manifest_path = sc.run()
```

`run()` does everything up to a curated dataset: standardizes the real images, extracts
DINOv2/InceptionV3 features, computes the metric portfolio over the pool, curates one dataset per
metric, trains your classifier on each one (`train_and_evaluate`), turns the results into per-class
weights, and curates the final set. It returns the path to the curated manifest:

```python
import pandas as pd
pd.read_csv(manifest_path)
#                                     source_path target_class      safe_filename
# 0   /abs/path/synthetic_pool/cats/img0004.png         cats  cats_img0004.png
# 1   /abs/path/synthetic_pool/dogs/img0011.png         dogs  dogs_img0011.png
# ...
```

Alongside it: `runs/cats_vs_dogs/weights.json` (the derived per-class metric weights),
`weights_used_seed0.json` (everything `curate()` used, for reproducibility), and a per-block
diagnostics log.

### 5. Or, get a deployable model directly

If your model implements `train_production` too, skip step 4 and just call:

```python
checkpoint_path = sc.train_production()
# -> curates the set (steps above, or reuses them if already done), then trains once more
#    on the final curated set via ResNetClassifier.train_production(), and returns
#    whatever that method returns - here, checkpoints/resnet18_final.pt
```

This fails immediately, before curating anything, if your model doesn't implement
`train_production` — so you find out at the start of a run, not after it.

### Or: a config file + the CLI

Everything in steps 3-4 can live in one YAML file instead (see `config.example.yaml` for every
field):

```yaml
workdir: runs/cats_vs_dogs
target_distribution: {cats: 50.0, dogs: 50.0}
model:
  trainer: my_project.training:ResNetClassifier
  params: {val_dir: real_reference_holdout, class_to_idx: {cats: 0, dogs: 1}}
inputs:
  synthetic_dir: synthetic_pool
  real_dir: real_reference
  resolution: 512
curation:
  num_images: 2000
  group_by: []
```

```bash
synth-curator config.yaml
# runs/cats_vs_dogs/curated/seed_0.csv

synth-curator config.yaml --train-production
# trains the production model on that same curated set too, and prints whatever
# ResNetClassifier.train_production() returned instead (here, checkpoints/resnet18_final.pt)
```

`trainer` is a dotted `module:ClassOrFactory` path, called with `params` — so the CLI never needs
your model object in memory, just an import path to it.

## Metrics

The default portfolio (override with `metrics=[...]`):

| Instance (per image) | Distributional (per generation group) |
|---|---|
| `psnr_score`, `ssim_score`, `fsim_score`, `lpips_score` — vs. nearest real image | `fid_inception` |
| `brisqe_score`, `clip_similarity_*`, `clip_log_odds_score` | `kid_dino` |
| `fscs_score`, `diversity_score`, `realism_score`, `is_contribution_score` | `swd_dino` |

`fsim_score`/`lpips_score`/`brisqe_score`/`clip_*` need the `[full]` extra (`phasepack`, `lpips`,
`brisque`, `transformers`). Everything else needs only `torch`/`torchvision`/`scikit-image`.
Add your own by registering it in `synth_curator.core.registry.METRIC_PROPERTIES` and a calculator
in `synth_curator.steps.metrics`.

## Architecture

```
src/synth_curator/
  pipeline.py          SynthCurator: run() and train_production() (see below)
  model_interface.py   Model (required) / ProductionModel (+ train_production, optional)
  cli.py                synth-curator config.yaml
  steps/                one file per pipeline step
    standardize.py      1. resize real images to the pool's generation resolution
    features.py          2. DINOv2/InceptionV3 features, KNN map, real class labels
    metrics/              3. per-image metrics -> metadata CSV
    metric_sets.py       4. one curated dataset per metric (+ random control)
    tournament.py         5. train/evaluate your model on each one
    weights.py            6. per-class utility weights from the results
    curate.py             7. final weighted, temperature-sampled curated set
  core/                 shared math: scoring, FID/KID/SWD, sampling, safety, groups, registry
```

`run()` is fully linear — each step's output is threaded explicitly into the next, with no step
reaching upward to pull its own input. Everything is validated up front, before step 1 ever runs.
`docs/design.md` has the exact, verified call sequence as a Mermaid diagram, including
`train_production()` and which parts are skipped when their output already exists.

## Development

```bash
pip install -e '.[dev]'
pytest                    # core suite: no network, no GPU
pip install -e '.[full]'
pytest                    # also exercises FSIM/LPIPS/BRISQUE/CLIP
```
