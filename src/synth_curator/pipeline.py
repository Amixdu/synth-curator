"""SynthCurator: the whole pipeline as one object, one model per instance.

Raw images + a model in, curated set out - or raw images + already-known weights
(`weights_path=`/`shipped_weights=`) in, skipping the tournament (steps 4-6) entirely; see
weights_bundle.py. `run()` is the orchestrator, and it is a fully linear thread either way - each
step's output is passed explicitly into the next, with no step reaching upward to pull its own input:

    std_dir = standardize()
    features_dir = extract_features(std_dir)
    metadata_csv = compute_metrics(std_dir, features_dir)
    weights = resolve_weights(metadata_csv, features_dir)   # steps 4-6, or precomputed weights
    curate(metadata_csv, features_dir, weights, ...)

Every step still checks its own workdir output first and returns immediately if it already
exists, so a crashed run resumes where it stopped (delete the file/dir to force a re-run) - that
resumability is orthogonal to the linear threading above and does not reintroduce any branching
into `run()` itself.

`workdir` is scoped to one model and one curation run. Steps 1-3 don't depend on either, though -
they only care about `synthetic_dir`/`real_dir`/`resolution`/`metadata_csv` - so `pool_workdir`
(defaults to `workdir`) lets you share just those across a sweep of workdirs (different models,
`target_distribution`s, etc.) instead of re-standardizing/re-extracting/re-scoring the same pool
for every one.
"""
from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from . import (
    __version__,  # set at the top of __init__.py, before this module is imported
)
from .core import fingerprint
from .core.registry import DEFAULT_METRICS, is_distributional, is_instance, props
from .exceptions import SynthCuratorError
from .model_interface import (
    Model,
    ModelFactory,
    import_dotted,
    load_model,
    load_model_factory,
)
from .steps import (
    curate,
    features,
    metric_sets,
    metrics,
    standardize,
    tournament,
    weights,
)
from .steps.weights import ScoringFn
from .weights_bundle import SCHEMA_VERSION, load_shipped_weights, load_weights_path

log = logging.getLogger(__name__)


class SynthCurator:
    def __init__(
        self,
        workdir: str | Path,
        target_distribution: dict[str, float],      # class -> percent, sums to ~100
        *,
        model: Model | None = None,                  # your classifier; required unless model_factory is set
        model_factory: ModelFactory | None = None,    # (gpu_id) -> Model, for the tournament's GPU workers - see model_interface.py
        gpu_ids: Sequence[int] | None = None,         # required with model_factory: which GPU each worker uses
        synthetic_dir: str | Path,                   # <class>/**/<image>
        real_dir: str | Path,                        # reference set, <class>/<image>
        resolution: int,                             # synthetic pool's generation resolution
        pool_workdir: str | Path | None = None,       # steps 1-3's cache (standardize/features/metrics);
                                                       # defaults to workdir - share it across a target_distribution sweep
        metrics: Sequence[str] = DEFAULT_METRICS,   # the portfolio the tournament sweeps
        num_images: int = 8012,
        temperature: float = 0.05,
        dist_group_by: Sequence[str] = ("source_checkpoint_kimg", "cfg_scale"),  # distributional metrics only
        metadata_csv: str | Path | None = None,       # CSV (filename, target_class_label, ...) instead of scanning synthetic_dir
        block_search: bool = False,
        num_blocks: int = 1,
        safety_max_similarity_threshold: float | None = 0.97,
        utility_formula: str = "worst_anchored",
        dist_utility_formula: str | None = "random_anchored",
        dist_min_samples: dict[str, int] | None = None,
        seed: int = 0,
        random_seeds_per_metric_set: int = 1,         # repeats for distributional metrics + random_control only - instance metrics always train once
        metric_beta: float = 2.0,                    # the beta in F-beta the tournament scores by; 2.0 = F2
        scoring: ScoringFn | None = None,             # override: (true, pred, classes) -> {class: score};
                                                       # replaces F-beta entirely, and metric_beta is then unused
        force_retrain: bool = False,                  # ignore the tournament's prediction cache, e.g. after changing the model
        weights_path: str | Path | None = None,       # bring your own precomputed weights - a directory with weights.json + meta.json; skips the tournament (steps 4-6) entirely
        shipped_weights: str | None = None,           # same, but a bundle shipped with the library by name - see weights_bundle.list_shipped_weights()
    ):
        if weights_path is not None and shipped_weights is not None:
            raise SynthCuratorError("weights_path and shipped_weights can't both be set - pick one")
        precomputed = weights_path is not None or shipped_weights is not None
        if model is None and model_factory is None and not precomputed:
            raise SynthCuratorError("a model is required: pass model=... (one instance, trained "
                                    "sequentially), model_factory=...+gpu_ids=... (a fresh model per "
                                    "GPU worker, trained in parallel), or weights_path=.../"
                                    "shipped_weights=... to skip the tournament entirely")
        if precomputed:
            if model_factory is not None or gpu_ids:
                raise SynthCuratorError(
                    "model_factory/gpu_ids don't apply with weights_path/shipped_weights set - "
                    "there's no tournament to run in parallel. A plain model=... is still fine, "
                    "for train_production()")
            if random_seeds_per_metric_set != 1 or force_retrain:
                raise SynthCuratorError(
                    "random_seeds_per_metric_set/force_retrain don't apply with weights_path/"
                    "shipped_weights set - there's no tournament to repeat seeds on or a "
                    "prediction cache to bypass")
            if (utility_formula != "worst_anchored" or dist_utility_formula != "random_anchored"
                    or metric_beta != 2.0 or scoring is not None):
                raise SynthCuratorError(
                    "utility_formula/dist_utility_formula/metric_beta/scoring don't apply with "
                    "weights_path/shipped_weights set - weights are supplied directly, not derived")
            if list(metrics) != list(DEFAULT_METRICS):
                raise SynthCuratorError(
                    "metrics doesn't apply with weights_path/shipped_weights set - the metric "
                    "portfolio comes from the weights bundle itself")
        if model_factory is not None and not gpu_ids:
            raise SynthCuratorError("model_factory needs gpu_ids too - which GPU each worker "
                                    "should use, e.g. gpu_ids=[0, 1, 2, 3]")
        if gpu_ids and model_factory is None:
            raise SynthCuratorError("gpu_ids needs model_factory too - a plain model= is one "
                                    "instance, so there's nothing to build a fresh copy from per GPU worker")
        if model_factory is not None and not callable(model_factory):
            raise SynthCuratorError("model_factory must be callable: (gpu_id) -> Model")
        if random_seeds_per_metric_set < 1:
            raise SynthCuratorError("random_seeds_per_metric_set must be >= 1")
        if metric_beta <= 0:
            raise SynthCuratorError(f"metric_beta must be positive, got {metric_beta}")
        if scoring is not None and not callable(scoring):
            raise SynthCuratorError("scoring must be callable: (true_labels, pred_labels, classes) -> {class: score}")
        self.workdir = Path(workdir)
        self.pool_workdir = Path(pool_workdir) if pool_workdir is not None else self.workdir
        self.target_distribution = target_distribution
        self.classes = sorted(target_distribution)
        self.model = model
        self.model_factory = model_factory
        self.gpu_ids = list(gpu_ids) if gpu_ids else None
        self.synthetic_dir = Path(synthetic_dir)
        self.real_dir = Path(real_dir)
        self.resolution = resolution
        self.features_dir = self.pool_workdir / "features"
        self.precomputed_weights, self.precomputed_weights_meta = None, None
        if precomputed:
            source = f"weights_path={weights_path}" if weights_path is not None else f"shipped_weights={shipped_weights!r}"
            w, m = (load_weights_path(weights_path) if weights_path is not None
                   else load_shipped_weights(shipped_weights))
            if set(self.classes) != set(m["classes"]):
                raise SynthCuratorError(
                    f"{source}: classes {sorted(m['classes'])} don't match target_distribution's "
                    f"classes {self.classes}")
            for metric in m["metrics"]:
                props(metric)   # catch an unknown/renamed metric from an old or malformed bundle
            self.precomputed_weights, self.precomputed_weights_meta = w, m
            self.metrics = list(m["metrics"])
        else:
            for metric in metrics:
                props(metric)
            self.metrics = list(metrics)
        self.num_images = num_images
        self.temperature = temperature
        self.dist_group_by = list(dist_group_by)
        self.metadata_csv = Path(metadata_csv) if metadata_csv is not None else None
        self.block_search = block_search
        self.num_blocks = num_blocks
        self.safety_max_similarity_threshold = safety_max_similarity_threshold
        self.utility_formula = utility_formula
        self.dist_utility_formula = dist_utility_formula
        self.dist_min_samples = dist_min_samples
        self.seed = seed
        self.seeds = [seed + i for i in range(random_seeds_per_metric_set)]
        self.metric_beta = metric_beta
        self.scoring = scoring
        self.force_retrain = force_retrain
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.pool_workdir.mkdir(parents=True, exist_ok=True)
        self._validate()

    def _validate(self) -> None:
        """Check everything the pipeline needs is actually present, before any expensive work
        runs - directories with no images or a model missing a method should fail here, not
        after standardizing/embedding/scoring the whole pool."""
        # model_factory's models aren't built until a GPU worker starts, so this can only check
        # the plain model= route upfront - a bad model_factory surfaces at the first training job instead.
        if self.model is not None:
            if self.precomputed_weights is not None:
                # train_and_evaluate is never called with precomputed weights - the only thing
                # model=... can be for here is train_production()
                if not callable(getattr(self.model, "train_production", None)):
                    raise SynthCuratorError(
                        "model must implement train_production(train_manifest) -> Any with "
                        "weights_path/shipped_weights set, since train_and_evaluate is never "
                        "called in this mode")
            elif not callable(getattr(self.model, "train_and_evaluate", None)):
                raise SynthCuratorError("model must implement train_and_evaluate(train_manifest, seed) -> DataFrame")
        synthetic_images = None
        for name, path in (("synthetic_dir", self.synthetic_dir), ("real_dir", self.real_dir)):
            if not path.is_dir():
                raise SynthCuratorError(f"{name} does not exist or is not a directory: {path}")
            images = standardize.list_images(path)
            if not images:
                raise SynthCuratorError(f"{name} has no images (.jpg/.jpeg/.png) anywhere under it: {path}")
            if name == "synthetic_dir":
                synthetic_images = images
        if self.resolution <= 0:
            raise SynthCuratorError(f"resolution must be a positive integer, got {self.resolution}")
        if any(v < 0 for v in self.target_distribution.values()):
            raise SynthCuratorError(f"target_distribution values must be non-negative: {self.target_distribution}")
        total = sum(self.target_distribution.values())
        if abs(total - 100) > 1.0:
            raise SynthCuratorError(f"target_distribution must sum to ~100 (percent), got {total}")
        if self.metadata_csv is not None:
            if not self.metadata_csv.is_file():
                raise SynthCuratorError(f"metadata_csv does not exist or is not a file: {self.metadata_csv}")
            try:
                meta_df = pd.read_csv(self.metadata_csv)
            except Exception as e:
                raise SynthCuratorError(f"metadata_csv could not be read as a CSV: {self.metadata_csv} ({e})") from e
            missing_cols = {"filename", "target_class_label"} - set(meta_df.columns)
            if missing_cols:
                raise SynthCuratorError(
                    f"metadata_csv is missing required column(s) {sorted(missing_cols)}: {self.metadata_csv}")
            present = set(meta_df["target_class_label"].astype(str))
            source = f"metadata_csv ({self.metadata_csv})"
        else:
            flat = [p for p in synthetic_images if len(p.relative_to(self.synthetic_dir).parts) < 2]
            if flat:
                raise SynthCuratorError(
                    f"synthetic_dir must be laid out as <class_name>/<image>, but found images directly "
                    f"under {self.synthetic_dir} with no class folder (e.g. {flat[0].name})")
            present = {p.relative_to(self.synthetic_dir).parts[0] for p in synthetic_images}
            source = f"synthetic_dir folders ({self.synthetic_dir})"
        wanted = {c for c, pct in self.target_distribution.items() if pct > 0}
        missing = sorted(wanted - present)
        if missing:
            raise SynthCuratorError(f"target_distribution has class(es) not found in {source}: {missing}")
        dist_metrics = [m for m in self.metrics if is_distributional(m)]
        if dist_metrics and not self.dist_group_by:
            raise SynthCuratorError(
                f"metrics includes distributional metric(s) {dist_metrics} but dist_group_by is "
                "empty - they need >=2 real generation groups to compare and do nothing without "
                "one. Drop them from metrics, or set dist_group_by.")
        for name, formula in (("utility_formula", self.utility_formula), ("dist_utility_formula", self.dist_utility_formula)):
            if formula is not None and formula not in weights.FORMULAS:
                raise SynthCuratorError(f"{name} must be one of {sorted(weights.FORMULAS)} or None, got {formula!r}")
        if self.scoring is not None and self.metric_beta != 2.0:
            raise SynthCuratorError(
                "metric_beta is ignored once scoring is set - pass one or the other, not both "
                f"(got metric_beta={self.metric_beta} alongside a scoring override)")
        if self.num_images <= 0:
            raise SynthCuratorError(f"num_images must be a positive integer, got {self.num_images}")
        if self.temperature <= 0:
            raise SynthCuratorError(f"temperature must be positive, got {self.temperature}")
        if self.num_blocks < 1:
            raise SynthCuratorError(f"num_blocks must be >= 1, got {self.num_blocks}")
        if self.safety_max_similarity_threshold is not None and not (0.0 <= self.safety_max_similarity_threshold <= 1.0):
            raise SynthCuratorError(
                "safety_max_similarity_threshold must be within [0, 1] (it's an absolute cosine "
                f"similarity) or None, got {self.safety_max_similarity_threshold}")

    @classmethod
    def from_config(cls, path: str | Path, model: Model | None = None) -> SynthCurator:
        """Build from config.example.yaml-style YAML (or config.example.shipped.yaml, for
        weights_path/shipped_weights). `model` (a Python object) is used if the config's `model:`
        block is absent; if both are given, the config's `model:` wins. If the config's
        `model.gpu_ids` is set, `trainer` becomes a GPU-worker factory instead, and a plain
        instance (for train_production()) is also built the same way, bound to
        `model.production_gpu_id` (or gpu_ids[0] if that's not set) - unless `model` was already
        given here, which is kept as-is.

        If the config sets `weights_path`/`shipped_weights`, there's no tournament to run at all:
        `tournament:`, `weights:` and `metrics.instance`/`metrics.distributional` must all be
        absent (they'd do nothing), and `model:` is optional and, if given, only needs
        `trainer`/`params` (no gpu_ids/production_gpu_id) - purely for train_production()."""
        import yaml
        config = yaml.safe_load(Path(path).read_text())
        inputs_cfg = config.get("inputs", {})
        metrics_cfg = config.get("metrics", {})
        curation_cfg = config.get("curation", {})
        model_cfg = config.get("model")
        weights_path, shipped_weights = config.get("weights_path"), config.get("shipped_weights")
        precomputed = weights_path is not None or shipped_weights is not None
        model_factory, gpu_ids = None, None

        if precomputed:
            forbidden = [k for k in ("tournament", "weights") if k in config]
            if metrics_cfg.get("instance") or metrics_cfg.get("distributional"):
                forbidden.append("metrics.instance/metrics.distributional")
            if forbidden:
                raise SynthCuratorError(
                    f"config key(s) {forbidden} don't apply with weights_path/shipped_weights set "
                    "- there's no tournament to run or derive weights from")
            if model_cfg is not None:
                if "trainer" not in model_cfg:
                    raise SynthCuratorError("config's model: block needs a 'trainer' key "
                                            "(a dotted 'package.module:ClassOrFactory' path)")
                if model_cfg.get("gpu_ids") or model_cfg.get("production_gpu_id") is not None:
                    raise SynthCuratorError(
                        "config's model.gpu_ids/production_gpu_id don't apply with weights_path/"
                        "shipped_weights set - there's no tournament to run in parallel. Keep "
                        "just trainer/params, for train_production()")
                if model is None:
                    model = load_model(model_cfg["trainer"], model_cfg.get("params"))
        else:
            if model_cfg is not None:
                if "trainer" not in model_cfg:
                    raise SynthCuratorError("config's model: block needs a 'trainer' key "
                                            "(a dotted 'package.module:ClassOrFactory' path)")
                gpu_ids = model_cfg.get("gpu_ids")
                production_gpu_id = model_cfg.get("production_gpu_id")
                if production_gpu_id is not None and not gpu_ids:
                    raise SynthCuratorError(
                        "model.production_gpu_id needs model.gpu_ids too - there's no parallel "
                        "tournament here for it to pick a GPU apart from")
                if gpu_ids:
                    # trainer(**params, gpu_id=...) is called once per tournament worker, not once
                    # here - your trainer needs to accept that gpu_id keyword.
                    model_factory = load_model_factory(model_cfg["trainer"], model_cfg.get("params"))
                    if model is None:
                        # train_production() needs a plain instance too, and the CLI never passes
                        # one (there's no Python object to pass) - build one the same way a worker
                        # would, on production_gpu_id if you set it (e.g. a bigger GPU, for the one
                        # training run that's worth being expensive), else just the first
                        # tournament GPU. An explicitly-passed model=... here is left alone either way.
                        model = model_factory(production_gpu_id if production_gpu_id is not None else gpu_ids[0])
                else:
                    model = load_model(model_cfg["trainer"], model_cfg.get("params"))
            if model is None and model_factory is None:
                raise SynthCuratorError("a model is required: add a config 'model:' block, or pass model=... here")

        synthetic_dir, real_dir, resolution = (
            inputs_cfg.get("synthetic_dir"), inputs_cfg.get("real_dir"), inputs_cfg.get("resolution"))
        if not (synthetic_dir and real_dir and resolution):
            raise SynthCuratorError("inputs.synthetic_dir, inputs.real_dir and inputs.resolution are all required")
        kw = {
            "model": model, "model_factory": model_factory, "gpu_ids": gpu_ids,
            "weights_path": weights_path, "shipped_weights": shipped_weights,
            "synthetic_dir": synthetic_dir, "real_dir": real_dir, "resolution": resolution,
            "pool_workdir": inputs_cfg.get("pool_workdir"),
            "num_images": curation_cfg.get("num_images", 8012), "temperature": curation_cfg.get("temperature", 0.05),
            "dist_group_by": curation_cfg.get("dist_group_by", ["source_checkpoint_kimg", "cfg_scale"]),
            "metadata_csv": inputs_cfg.get("metadata_csv"),
            "block_search": curation_cfg.get("block_search", False), "num_blocks": curation_cfg.get("num_blocks", 1),
            "safety_max_similarity_threshold": curation_cfg.get("safety_max_similarity_threshold", 0.97),
            "dist_min_samples": metrics_cfg.get("dist_min_samples"),
            "seed": config.get("seed", 0),
        }
        if not precomputed:
            weights_cfg = config.get("weights", {})
            kw.update(
                utility_formula=weights_cfg.get("utility_formula", "worst_anchored"),
                dist_utility_formula=weights_cfg.get("dist_utility_formula", "random_anchored"),
                random_seeds_per_metric_set=config.get("tournament", {}).get("random_seeds_per_metric_set", 1),
                metric_beta=weights_cfg.get("beta", 2.0),
                scoring=import_dotted(weights_cfg["scoring"]) if weights_cfg.get("scoring") else None,
                force_retrain=config.get("tournament", {}).get("force_retrain", False),
            )
            if metrics_cfg.get("instance") or metrics_cfg.get("distributional"):
                kw["metrics"] = [*metrics_cfg.get("instance", []), *metrics_cfg.get("distributional", [])]
        for key in ("workdir", "target_distribution"):
            if key not in config:
                raise SynthCuratorError(f"config is missing the required {key!r} key")
        return cls(config["workdir"], config["target_distribution"], **kw)

    @staticmethod
    def resample_seeds_from_config(path: str | Path) -> list[int] | None:
        """The config's `resample:` section, as a seed list for resample() - None if the section
        is absent (no sweep). `from_config()` itself doesn't read this key (it builds one
        SynthCurator; resampling is what you do with it afterward, not part of building it) - this
        is what the CLI calls, and what you'd call too if you're driving the same config from your
        own script instead of `synth-curator config.yaml`."""
        import yaml
        resample_cfg = yaml.safe_load(Path(path).read_text()).get("resample")
        if resample_cfg is None:
            return None
        seeds = resample_cfg.get("seeds", 1)
        return list(range(seeds)) if isinstance(seeds, int) else list(seeds)

    # ---- steps ------------------------------------------------------------------------------

    def standardize(self) -> Path:
        """1. Resize real images to the pool's generation resolution."""
        out = self.pool_workdir / "real_standardized"
        current = {"real_dir": str(self.real_dir), "resolution": self.resolution}
        stale = fingerprint.exists(out) and not fingerprint.matches(out, current)
        if out.exists() and not stale:
            return out
        if stale:
            log.info("real_dir or resolution changed since %s was last built - recomputing", out)
        result = standardize.standardize(self.real_dir, out, self.resolution)
        fingerprint.write(result, current)
        return result

    def extract_features(self, std_dir: Path) -> Path:
        """2. DINOv2/Inception features, probabilities, KNN map, real labels."""
        current = {"synthetic_dir": str(self.synthetic_dir), "real_dir": str(self.real_dir)}
        stale = fingerprint.exists(self.features_dir) and not fingerprint.matches(self.features_dir, current)
        if (self.features_dir / "synthetic_dino.pt").exists() and not stale:
            return self.features_dir
        if stale:
            log.info("synthetic_dir or real_dir changed since %s was last built - recomputing", self.features_dir)
            shutil.rmtree(self.features_dir)
        result = features.extract_features(std_dir, self.synthetic_dir, self.features_dir, self.real_dir)
        fingerprint.write(result, current)
        return result

    def compute_metrics(self, std_dir: Path, features_dir: Path, wanted: Sequence[str] | None = None) -> Path:
        """3. Instance-metric columns -> metadata CSV. Returns the cached CSV untouched if it already has them all."""
        instance_metrics = [metric for metric in (wanted if wanted is not None else self.metrics)
                            if is_instance(metric)]
        out_csv = self.pool_workdir / "metadata.csv"
        current = {"metadata_csv": str(self.metadata_csv) if self.metadata_csv else None,
                  "synthetic_dir": str(self.synthetic_dir)}
        if out_csv.exists() and fingerprint.exists(out_csv) and not fingerprint.matches(out_csv, current):
            log.info("metadata source changed since %s was last built - recomputing", out_csv)
            out_csv.unlink()
        result = metrics.compute_metrics(
            self.synthetic_dir, std_dir, features_dir, out_csv, instance_metrics,
            base_metadata=self.metadata_csv)
        fingerprint.write(result, current)
        return result

    def curate_metric_sets(self, metadata_csv: Path, features_dir: Path) -> dict[str, dict[int, Path]]:
        """4. One curated dataset per metric + random control."""
        out_dir = self.workdir / "metric_sets"
        current = {"metadata_csv": str(self.metadata_csv) if self.metadata_csv else None,
                  "synthetic_dir": str(self.synthetic_dir), "target_distribution": self.target_distribution,
                  "num_images": self.num_images, "dist_group_by": self.dist_group_by,
                  "metrics": sorted(self.metrics)}
        if out_dir.exists() and fingerprint.exists(out_dir) and not fingerprint.matches(out_dir, current):
            log.info("curation inputs changed since %s was last built - recomputing", out_dir)
            shutil.rmtree(out_dir)
        result = metric_sets.curate_metric_sets(
            metadata_csv, features_dir, out_dir, self.metrics,
            self.target_distribution, self.num_images, self.dist_group_by, self.seeds, self.synthetic_dir)
        fingerprint.write(out_dir, current)
        return result

    def run_tournament(self, metric_sets: dict[str, dict[int, Path]]):
        """5. Train/eval your model on each metric set - across GPU workers if model_factory/gpu_ids
        are set, one process otherwise."""
        if self.model_factory is not None:
            return tournament.run_parallel(metric_sets, self.model_factory, self.gpu_ids,
                                           self.workdir / "tournament", force_retrain=self.force_retrain)
        return tournament.run(metric_sets, self.model, self.workdir / "tournament",
                              force_retrain=self.force_retrain)

    def derive_weights(self, predictions: dict) -> dict[str, dict]:
        """6. Per-class utility weights from the tournament predictions, scored with `scoring`
        if set, else F-`metric_beta`."""
        resolved = weights.derive_weights(
            predictions, self.classes, self.utility_formula, self.dist_utility_formula, self.metrics,
            beta=self.metric_beta, scoring=self.scoring)
        (self.workdir / "weights.json").write_text(json.dumps(resolved, indent=2))
        return resolved

    def curate(self, metadata_csv: Path, features_dir: Path, resolved_weights: dict) -> Path:
        """7. Final weighted, temperature-sampled curated set."""
        out_dir = self.workdir / "curated"
        manifest = out_dir / f"seed_{self.seed}.csv"
        current = {"weights": resolved_weights, "target_distribution": self.target_distribution,
                  "num_images": self.num_images, "temperature": self.temperature,
                  "dist_group_by": self.dist_group_by, "block_search": self.block_search,
                  "num_blocks": self.num_blocks,
                  "safety_max_similarity_threshold": self.safety_max_similarity_threshold}
        if manifest.exists() and fingerprint.exists(manifest) and not fingerprint.matches(manifest, current):
            log.info("curation inputs changed since %s was last built - recomputing", manifest)
            manifest.unlink()
        result = curate.curate(
            metadata_csv, features_dir, resolved_weights, out_dir,
            target_distribution=self.target_distribution, num_images=self.num_images,
            temperature=self.temperature, dist_group_by=self.dist_group_by,
            block_search=self.block_search, num_blocks=self.num_blocks,
            safety_max_similarity_threshold=self.safety_max_similarity_threshold, seed=self.seed,
            image_dir=self.synthetic_dir, min_samples=self.dist_min_samples)
        fingerprint.write(result, current)
        return result

    def _own_weights_meta(self) -> dict:
        """meta.json for weights this instance just derived itself, so this workdir - not just a
        shipped/bring-your-own bundle - is itself a valid weights_path= source afterward. See
        weights_bundle.py for the format; everything past classes/metrics here is informational."""
        return {
            "schema_version": SCHEMA_VERSION,
            "classes": self.classes,
            "metrics": self.metrics,
            "weights_derivation": {
                "utility_formula": self.utility_formula,
                "dist_utility_formula": self.dist_utility_formula,
                "metric_beta": self.metric_beta,
                "scoring": f"{self.scoring.__module__}:{self.scoring.__qualname__}" if self.scoring else None,
            },
            "tournament": {"random_seeds_per_metric_set": len(self.seeds)},
            "curation": {"target_distribution": self.target_distribution, "dist_group_by": self.dist_group_by},
            "provenance": {"synth_curator_version": __version__, "computed_on": date.today().isoformat()},
        }

    def resolve_weights(self, metadata_csv: Path, features_dir: Path) -> dict[str, dict]:
        """4-6: curate metric sets, run the tournament, derive weights - or, with
        weights_path/shipped_weights set, skip straight to those precomputed weights instead.
        Either way, writes workdir/weights.json + meta.json, so this workdir is itself a valid
        weights_path= source afterward (see resample())."""
        if self.precomputed_weights is not None:
            resolved, meta = self.precomputed_weights, self.precomputed_weights_meta
        else:
            metric_sets = self.curate_metric_sets(metadata_csv, features_dir)
            predictions = self.run_tournament(metric_sets)
            resolved = self.derive_weights(predictions)
            meta = self._own_weights_meta()
        (self.workdir / "weights.json").write_text(json.dumps(resolved, indent=2))
        (self.workdir / "meta.json").write_text(json.dumps(meta, indent=2))
        return resolved

    def run(self) -> Path:
        """The whole pipeline -> curated manifest CSV path."""
        std_dir = self.standardize()                              # 1
        features_dir = self.extract_features(std_dir)             # 2
        metadata_csv = self.compute_metrics(std_dir, features_dir)  # 3
        resolved_weights = self.resolve_weights(metadata_csv, features_dir)  # 4-6, or precomputed
        return self.curate(metadata_csv, features_dir, resolved_weights)  # 7

    def train_production(self):
        """Curate the best set (same as run()), then train the model you'll actually deploy on it.

        Needs your model to additionally implement train_production(train_manifest) -> Any -
        checked before any curation work happens. The return value is entirely yours (a checkpoint
        path, a wrapped model, anything); the library never inspects it, just passes it back.

        Needs a plain model=... instance, even if you're using model_factory+gpu_ids for the
        tournament - a factory-built model isn't kept around after the tournament ends, so pass
        both if you want the parallel tournament and train_production().
        """
        if self.model is None:
            raise SynthCuratorError(
                "train_production() needs a plain model=... instance - pass model=... alongside "
                "model_factory=...+gpu_ids=... if you also want the parallel tournament")
        if not callable(getattr(self.model, "train_production", None)):
            raise SynthCuratorError(
                "train_production() needs your model to implement train_production(train_manifest) -> Any, "
                "in addition to train_and_evaluate() - see synth_curator.model_interface.ProductionModel")
        manifest = self.run()
        return self.model.train_production(manifest)

    def resample(self, seeds: Sequence[int], subdir: str = "resample") -> dict[int, SynthCurator]:
        """Model variance locked, data variance swept: one SynthCurator per seed, sharing this
        instance's model/weights/target_distribution/curation settings - only `seed` (and
        `workdir`, so they don't collide) differs. This is the standard way to check a curation
        framework's actual effect (averaged over many draws), not one draw's luck.

        Weights are resolved once, here, on `self` - not re-derived per seed - then reused by every
        seed via weights_path=self.workdir (any workdir is a valid weights_path= source after its
        weights are resolved; see resolve_weights()). pool_workdir is shared too, so steps 1-3
        aren't repeated per seed either. Returns instances, not results - call .run() or
        .train_production() on each yourself, e.g.:

            checkpoints = {seed: sc.train_production() for seed, sc in curator.resample(range(30)).items()}
        """
        std_dir = self.standardize()
        features_dir = self.extract_features(std_dir)
        metadata_csv = self.compute_metrics(std_dir, features_dir)
        self.resolve_weights(metadata_csv, features_dir)
        # precomputed-weights mode (which every seed here uses) only ever calls train_production,
        # never train_and_evaluate - so a model that implements only the latter (fine for the
        # tournament self.model may have just run) isn't usable here; drop it to None instead of
        # forwarding something that would fail _validate(). .run() still works either way -
        # .train_production() will raise its own clear error if you then call it without one.
        model = self.model if callable(getattr(self.model, "train_production", None)) else None
        return {
            seed: SynthCurator(
                workdir=self.workdir / subdir / f"seed_{seed}",
                target_distribution=self.target_distribution, model=model,
                synthetic_dir=self.synthetic_dir, real_dir=self.real_dir, resolution=self.resolution,
                pool_workdir=self.pool_workdir, weights_path=self.workdir,
                metadata_csv=self.metadata_csv, num_images=self.num_images, temperature=self.temperature,
                dist_group_by=self.dist_group_by, block_search=self.block_search, num_blocks=self.num_blocks,
                safety_max_similarity_threshold=self.safety_max_similarity_threshold,
                dist_min_samples=self.dist_min_samples, seed=seed,
            )
            for seed in seeds
        }
