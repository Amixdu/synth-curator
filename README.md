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

> **Status:** the full pipeline works end to end (raw images in, curated set out), but it always
> derives weights from a fresh tournament with your model — bringing your own precomputed weights
> isn't wired up yet.

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
                    train YOUR classifier on each one, evaluate it
                                                              |
              turn "which metric's dataset produced the best classifier"
                    into per-class weights (per class, not global)
                                                              |
        curate the final set: weighted composite score, temperature sampling,
                    optional memorization filter, optional block search
                                                              |
                                  a curated manifest CSV
```

Every step is idempotent — it checks its own output first and skips the work if it's already
there, so an interrupted run just resumes. See `docs/design.md` for the exact call sequence.

## Quickstart

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

Any two-level `<class>/<image>` layout works. `real_reference` is whatever real dataset you'd
train the classifier on anyway — it's also the reference distribution the distributional metrics
(FID/KID/SWD) compare the synthetic pool against.

### 2. Implement your classifier

The library never trains anything itself — you do, in one method:

```python
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

class ResNetClassifier:
    """One method: train on the manifest, evaluate on your own held-out set, return predictions."""

    def __init__(self, val_dir: Path, num_classes: int, epochs: int = 3, device: str = "cuda"):
        self.val_dir, self.num_classes, self.epochs, self.device = val_dir, num_classes, epochs, device

    def train_and_evaluate(self, train_manifest: Path, seed: int) -> pd.DataFrame:
        torch.manual_seed(seed)
        tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

        # train_manifest has columns: source_path, target_class, safe_filename
        manifest = pd.read_csv(train_manifest)
        train_ds = ImagesFromManifest(manifest, tf)   # your own tiny Dataset wrapping the CSV
        model = models.resnet18(num_classes=self.num_classes).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)

        model.train()
        for _ in range(self.epochs):
            for x, y in DataLoader(train_ds, batch_size=32, shuffle=True):
                opt.zero_grad()
                loss = torch.nn.functional.cross_entropy(model(x.to(self.device)), y.to(self.device))
                loss.backward()
                opt.step()

        val_ds = datasets.ImageFolder(self.val_dir, transform=tf)
        model.eval()
        ids, true, pred = [], [], []
        with torch.no_grad():
            for i, (x, y) in enumerate(DataLoader(val_ds, batch_size=64)):
                p = model(x.to(self.device)).argmax(1).cpu()
                ids += [val_ds.samples[j][0] for j in range(i * 64, i * 64 + len(y))]
                true += y.tolist()
                pred += p.tolist()

        return pd.DataFrame({"unique_image_id": ids, "true_label": true, "predicted_label": pred})
```

That's the entire contract: a `train_and_evaluate(train_manifest, seed) -> DataFrame` with columns
`unique_image_id`, `true_label`, `predicted_label`. Everything about *how* you train — framework,
architecture, augmentation, your validation split — is entirely yours.

### 3. Run the pipeline

```python
from synth_curator import SynthCurator

sc = SynthCurator(
    workdir="runs/cats_vs_dogs",
    target_distribution={"cats": 50.0, "dogs": 50.0},   # class proportions in the curated set (%)
    model=ResNetClassifier(val_dir="real_reference_holdout", num_classes=2),
    synthetic_dir="synthetic_pool",
    real_dir="real_reference",
    resolution=512,               # your generator's output resolution
    block_size=2000,               # images in the final curated set
    group_by=[],                   # no generation-group metadata? leave this empty
)

manifest_path = sc.run()
```

`run()` does everything: standardizes the real images to your generator's resolution, extracts
DINOv2/InceptionV3 features, computes the metric portfolio over the pool, curates one dataset per
metric, trains your classifier on each one, turns the results into per-class weights, and curates
the final set. The return value is the path to the curated manifest:

```python
import pandas as pd
pd.read_csv(manifest_path)
#                                     source_path target_class      safe_filename
# 0   /abs/path/synthetic_pool/cats/img0004.png         cats  cats_img0004.png
# 1   /abs/path/synthetic_pool/dogs/img0011.png         dogs  dogs_img0011.png
# ...
```

Alongside it: `weights.json` (the derived per-class metric weights), `weights_used_seed0.json`
(everything `curate()` used, for reproducibility), and a per-block diagnostics log.

### Or: a config file + the CLI

Everything above can live in one YAML file instead (see `config.example.yaml` for every field):

```yaml
workdir: runs/cats_vs_dogs
target_distribution: {cats: 50.0, dogs: 50.0}
model:
  trainer: my_project.training:ResNetClassifier
  params: {val_dir: real_reference_holdout, num_classes: 2}
inputs:
  synthetic_dir: synthetic_pool
  real_dir: real_reference
  resolution: 512
curation:
  block_size: 2000
  group_by: []
```

```bash
synth-curator config.yaml
# runs/cats_vs_dogs/curated/seed_0.csv
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
  pipeline.py          SynthCurator: the orchestrator (see below)
  model_interface.py   Model: the one interface you implement
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
reaching upward to pull its own input. `docs/design.md` has the exact, verified call sequence as a
Mermaid diagram, including which parts are skipped when their output already exists.

## Development

```bash
pip install -e '.[dev]'
pytest                    # core suite: no network, no GPU
pip install -e '.[full]'
pytest                    # also exercises FSIM/LPIPS/BRISQUE/CLIP
```
