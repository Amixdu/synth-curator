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

## Install

```bash
pip install synth-curator                    # the pipeline + most of the metric portfolio
pip install synth-curator[full]              # + FSIM/LPIPS/BRISQUE calculators
pip install synth-curator[yaml]              # + YAML config files / the CLI
pip install synth-curator[full,yaml]         # both
```

The Quickstart below uses the default metric portfolio, which needs `[full]`; pass
`metrics=[...]` to use only core-dependency metrics instead.

The feature-extraction step downloads DINOv2 (`facebook/dinov2-base`, via `transformers`) and
torchvision's InceptionV3 weights on first use — make sure you have network access, or those
already cached, before running on an offline machine.

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

("Temperature sampling," the "memorization filter," and "block search" are `temperature`,
`safety_max_similarity_threshold`, and `block_search`/`num_blocks` respectively — see
`config.example.yaml` for all of them.)

Every step is idempotent — it checks its own output first and skips the work if it's already
there, so an interrupted run just resumes. This covers your *inputs* (paths, resolution, config);
it can't cover your *model*, since that's arbitrary code the library doesn't inspect — if you
change your model without changing anything else, pass `force_retrain=True` (or set
`tournament.force_retrain: true` in a config file) to force the tournament to retrain rather than
reuse cached predictions.

## Quickstart

A complete, working example: curating a pool of synthetic cat/dog images against a small real
reference set, then training a deployable ResNet. Needs `pip install synth-curator[full]`.

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

Any two-level `<class>/<image>` layout works for both. `synth-curator` only ever *reads*
`real_reference` — as the comparison distribution for FID/KID/SWD — it never trains on it or
manages a train/validation split for you. If your own classifier needs a held-out validation
slice, carve it out of `real_reference` yourself.

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
    training you're willing to pay for. train_and_evaluate runs once per (metric, seed) pair during
    the tournament - one metric set per metric in your portfolio, plus a random-control baseline -
    purely to rank metrics, so it stays cheap (few epochs). `seed` lets SynthCurator average out
    training noise across repeats (its seeds_per_metric_set, default 1); use it or ignore it.
    train_production runs exactly once, on the final curated set, so it's the one worth doing
    properly."""

    def __init__(self, val_dir: str, class_to_idx: dict, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
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
        idx_to_class = {i: c for c, i in self.class_to_idx.items()}   # the model's own label order
        model.eval()
        ids, true, pred = [], [], []
        with torch.no_grad():
            for i, (x, y) in enumerate(DataLoader(val_ds, batch_size=64)):
                p = model(x.to(self.device)).argmax(1).cpu()
                ids += [val_ds.samples[j][0] for j in range(i * 64, i * 64 + len(y))]
                true += [val_ds.classes[v] for v in y.tolist()]      # ImageFolder's own label order
                pred += [idx_to_class[v] for v in p.tolist()]
        return pd.DataFrame({"unique_image_id": ids, "true_label": true, "predicted_label": pred})

    # REQUIRED: called once per metric set per seed during the tournament. Keep this cheap.
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

That's the entire interface — a plain class, nothing to subclass (`Model`/`ProductionModel` in
`synth_curator.model_interface` are `typing.Protocol`s for optional type-checking, not base
classes). `train_and_evaluate` is required and must return a DataFrame with columns
`unique_image_id`, `true_label`, `predicted_label`. `train_production` is optional, only needed if
you want step 5 below, and can return *anything* — the library never inspects it. Everything about
*how* you train — framework, architecture, augmentation, your validation split, which checkpoint
you keep if you train for several epochs — is entirely yours in both.

Two things worth being precise about, since getting them wrong fails silently rather than loudly:

- **`train_manifest` lists curated *synthetic* images only** — three columns, `source_path`,
  `target_class`, `safe_filename` — real images are never added to it automatically. If you want
  real + synthetic training data together (a reasonable default, and consistent with
  `real_reference` being real training data you already have), load and concatenate your own real
  images inside `train_and_evaluate`/`train_production`; `synth-curator` has no opinion on this,
  same as everything else about training.
- **`true_label`/`predicted_label` can be class-name strings (as above) or integers** — but
  integers are interpreted as indices into `sorted(target_distribution)`, *not* your own
  `class_to_idx`, which the library never sees. The example above translates both back to strings
  in `_evaluate` (using `val_ds.classes` and `class_to_idx` respectively, since those two can
  legitimately disagree on ordering) specifically to avoid this ever mattering. `unique_image_id`
  just needs to be unique within one call; the library never interprets it.

### 3. Construct the pipeline

```python
from synth_curator import SynthCurator

class_to_idx = {"cats": 0, "dogs": 1}

sc = SynthCurator(
    workdir="runs/cats_vs_dogs",
    target_distribution={"cats": 50.0, "dogs": 50.0},   # class proportions in the curated set (%), sums to 100
    model=ResNetClassifier(val_dir="real_reference_holdout", class_to_idx=class_to_idx),  # a slice you carve out of real_reference yourself
    synthetic_dir="synthetic_pool",
    real_dir="real_reference",
    resolution=512,               # your synthetic pool's own resolution; real images are resized to match it
    num_images=2000,               # images in the final curated set - each tournament metric set is this size too
    dist_group_by=[],               # empty disables fid_inception/kid_dino/swd_dino entirely, not just grouping
)
```

Construction validates everything up front — a missing directory, an empty one, a model without
`train_and_evaluate`, or a `target_distribution` that doesn't sum to ~100 all raise immediately
here, before any standardizing, feature extraction, or training happens.

Classes normally come from `synthetic_dir`'s folder names. Instead, you can pass `metadata_csv=`,
a CSV with exactly three kinds of column:

- `filename` (required) — matches a file under `synthetic_dir`
- `target_class_label` (required) — that image's class
- your `dist_group_by` columns (optional) — only needed if you're using distributional metrics
  (`fid_inception`/`kid_dino`/`swd_dino`) and want to group by something folders can't express

Any other column is ignored, and the images still need to exist under `synthetic_dir` either way —
`metadata_csv` only changes where class labels (and group columns) come from. For example, if your
generator was run twice and you want those three metrics scored per run:

```
filename,               target_class_label, generation_batch
cats/img0001.png,       cats,                run_a
cats/img0002.png,       cats,                run_a
dogs/img0001.png,       dogs,                run_b
dogs/img0002.png,       dogs,                run_b
```

Only two lines change from the constructor above: `metadata_csv="metadata.csv"` instead of
scanning folders, and `dist_group_by=["generation_batch"]` instead of `[]`.

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
weights, and curates the final set. It returns the path to one curated manifest — always exactly
one file, for `self.seed` (default 0), regardless of how many tournament seeds you configured:

```python
import pandas as pd
pd.read_csv(manifest_path)
#                                     source_path target_class      safe_filename
# 0   /abs/path/synthetic_pool/cats/img0004.png         cats  cats_img0004.png
# 1   /abs/path/synthetic_pool/dogs/img0011.png         dogs  dogs_img0011.png
# ...
```

`source_path` and `target_class` are what `train_manifest` actually gives your model;
`safe_filename` is the same relative path with `/` replaced, for when you need a flat, unique,
filesystem-safe name (e.g. copying files out by hand).

Alongside it: `runs/cats_vs_dogs/weights.json` (the derived per-class metric weights),
`weights_used_seed0.json` (everything the final curation step used, for reproducibility), and a
per-block diagnostics log.

### 5. Train the deployable model

If your model implements `train_production` too:

```python
checkpoint_path = sc.train_production()
```

`train_production()` does everything `run()` does — curating the set — and reuses it automatically
if you already called `sc.run()` first, so calling both costs no more than calling this one alone.
It then trains once more on the final curated set via `ResNetClassifier.train_production()`, and
returns whatever that method returns — here, `checkpoints/resnet18_final.pt`.

This fails immediately, before curating anything, if your model doesn't implement
`train_production` — so you find out at the start of a run, not after it.

## Alternatives

### Config file + CLI

Everything in steps 3-5 can live in one YAML file instead (see `config.example.yaml` for every
field; needs `pip install synth-curator[yaml]`):

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
  dist_group_by: []
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

The default portfolio (override with `metrics=[...]`, a flat list of names — the YAML config's
nested `metrics: {instance: [...], distributional: [...]}` is config-only sugar for that same
list):

| Instance (per image) | Distributional (per generation group) |
|---|---|
| `psnr_score`, `ssim_score`, `fsim_score`, `lpips_score` — vs. nearest real image | `fid_inception` |
| `brisqe_score`, `clip_similarity_cancer`, `clip_similarity_artifact`, `clip_log_odds_score` | `kid_dino` |
| `fscs_score`, `diversity_score`, `realism_score`, `is_contribution_score` | `swd_dino` |

The distributional metrics need `dist_group_by` pointing at >=2 real generation groups (different
checkpoints, CFG values, whatever you actually varied) to mean anything — with only one group,
they can't compare anything, so `SynthCurator` rejects that combination at construction.

`fsim_score`/`lpips_score`/`brisqe_score` (that's the registry's actual spelling, not a typo) need
the `[full]` extra (`opencv-python`, `phasepack`, `lpips`, `brisque`). Everything else - including
`clip_*` - runs on the core dependencies. `clip_similarity_cancer`/`clip_similarity_artifact` are
dermoscopy-specific by name and origin; outside that domain they're a reasonable first pair to
drop via `metrics=[...]`. Add your own by registering it in
`synth_curator.core.registry.METRIC_PROPERTIES` and a calculator in `synth_curator.steps.metrics`.

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
  core/                 shared math: scoring, FID/KID/SWD, sampling, safety, groups, registry, fingerprint
```

`Model`/`ProductionModel` are `typing.Protocol`s, not base classes — nothing needs to inherit from
them; they exist for optional type-checking and documentation only.

`run()` is fully linear — each step's output is threaded explicitly into the next, with no step
reaching upward to pull its own input. Everything is validated up front, before step 1 ever runs.
`docs/design.md` has the exact, verified call sequence as a Mermaid diagram, including
`train_production()` and which parts are skipped when their output already exists.

## Coming up

- **Bring your own weights.** Every run currently derives weights from a fresh tournament against
  your model. Passing in precomputed weights, skipping the tournament entirely, is coming up.

## Development

```bash
pip install -e '.[dev]'
pytest                    # core suite: no network, no GPU
pip install -e '.[full]'
pytest                    # also exercises FSIM/LPIPS/BRISQUE/CLIP
```
