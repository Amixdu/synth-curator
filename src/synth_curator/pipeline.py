"""SynthCurator: the whole pipeline as one object, one model per instance.

This is the full path only, no shortcuts: raw images + a model in, curated set out. `run()` is
the orchestrator, and it is now a fully linear thread - each step's output is passed explicitly
into the next, with no step reaching upward to pull its own input:

    std_dir = standardize()
    features_dir = extract_features(std_dir)
    metadata_csv = compute_metrics(std_dir, features_dir)
    metric_sets = curate_metric_sets(metadata_csv, features_dir)
    predictions = run_tournament(metric_sets)
    weights = derive_weights(predictions)
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
from pathlib import Path

import pandas as pd

from .core import fingerprint
from .core.registry import DEFAULT_METRICS, is_distributional, is_instance, props
from .exceptions import SynthCuratorError
from .model_interface import Model, import_dotted, load_model
from .steps import curate, features, metric_sets, metrics, standardize, tournament, weights
from .steps.weights import ScoringFn

log = logging.getLogger(__name__)


class SynthCurator:
    def __init__(
        self,
        workdir: str | Path,
        target_distribution: dict[str, float],      # class -> percent, sums to ~100
        *,
        model: Model,                                # your classifier
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
    ):
        if random_seeds_per_metric_set < 1:
            raise SynthCuratorError("random_seeds_per_metric_set must be >= 1")
        if metric_beta <= 0:
            raise SynthCuratorError(f"metric_beta must be positive, got {metric_beta}")
        if scoring is not None and not callable(scoring):
            raise SynthCuratorError("scoring must be callable: (true_labels, pred_labels, classes) -> {class: score}")
        for metric in metrics:
            props(metric)
        self.workdir = Path(workdir)
        self.pool_workdir = Path(pool_workdir) if pool_workdir is not None else self.workdir
        self.target_distribution = target_distribution
        self.classes = sorted(target_distribution)
        self.model = model
        self.synthetic_dir = Path(synthetic_dir)
        self.real_dir = Path(real_dir)
        self.resolution = resolution
        self.features_dir = self.pool_workdir / "features"
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
        if not callable(getattr(self.model, "train_and_evaluate", None)):
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
            except Exception as e:  # noqa: BLE001 - any parse failure is a clear config error, not a crash
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
        """Build from config.example.yaml-style YAML. `model` (a Python object) is used if the
        config's `model:` block is absent; if both are given, the config's `model:` wins."""
        import yaml
        config = yaml.safe_load(Path(path).read_text())
        inputs_cfg = config.get("inputs", {})
        metrics_cfg = config.get("metrics", {})
        weights_cfg = config.get("weights", {})
        curation_cfg = config.get("curation", {})
        model_cfg = config.get("model")
        if model_cfg is not None:
            if "trainer" not in model_cfg:
                raise SynthCuratorError("config's model: block needs a 'trainer' key "
                                        "(a dotted 'package.module:ClassOrFactory' path)")
            model = load_model(model_cfg["trainer"], model_cfg.get("params"))
        if model is None:
            raise SynthCuratorError("a model is required: add a config 'model:' block, or pass model=... here")
        synthetic_dir, real_dir, resolution = (
            inputs_cfg.get("synthetic_dir"), inputs_cfg.get("real_dir"), inputs_cfg.get("resolution"))
        if not (synthetic_dir and real_dir and resolution):
            raise SynthCuratorError("inputs.synthetic_dir, inputs.real_dir and inputs.resolution are all required")
        kw = dict(
            model=model, synthetic_dir=synthetic_dir, real_dir=real_dir, resolution=resolution,
            pool_workdir=inputs_cfg.get("pool_workdir"),
            num_images=curation_cfg.get("num_images", 8012), temperature=curation_cfg.get("temperature", 0.05),
            dist_group_by=curation_cfg.get("dist_group_by", ["source_checkpoint_kimg", "cfg_scale"]),
            metadata_csv=inputs_cfg.get("metadata_csv"),
            block_search=curation_cfg.get("block_search", False), num_blocks=curation_cfg.get("num_blocks", 1),
            safety_max_similarity_threshold=curation_cfg.get("safety_max_similarity_threshold", 0.97),
            utility_formula=weights_cfg.get("utility_formula", "worst_anchored"),
            dist_utility_formula=weights_cfg.get("dist_utility_formula", "random_anchored"),
            dist_min_samples=metrics_cfg.get("dist_min_samples"),
            seed=config.get("seed", 0),
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
        """5. Train/eval your model on each metric set."""
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

    def run(self) -> Path:
        """The whole pipeline -> curated manifest CSV path."""
        std_dir = self.standardize()                              # 1
        features_dir = self.extract_features(std_dir)             # 2
        metadata_csv = self.compute_metrics(std_dir, features_dir)  # 3
        metric_sets = self.curate_metric_sets(metadata_csv, features_dir)  # 4
        predictions = self.run_tournament(metric_sets)             # 5
        resolved_weights = self.derive_weights(predictions)        # 6
        return self.curate(metadata_csv, features_dir, resolved_weights)  # 7

    def train_production(self):
        """Curate the best set (same as run()), then train the model you'll actually deploy on it.

        Needs your model to additionally implement train_production(train_manifest) -> Any -
        checked before any curation work happens. The return value is entirely yours (a checkpoint
        path, a wrapped model, anything); the library never inspects it, just passes it back.
        """
        if not callable(getattr(self.model, "train_production", None)):
            raise SynthCuratorError(
                "train_production() needs your model to implement train_production(train_manifest) -> Any, "
                "in addition to train_and_evaluate() - see synth_curator.model_interface.ProductionModel")
        manifest = self.run()
        return self.model.train_production(manifest)
