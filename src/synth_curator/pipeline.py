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

Everything under `workdir` is scoped to one model. If you want to compare models, use a separate
workdir per model - steps 1-4 will genuinely rerun for each one, which is the accepted cost of a
workdir holding exactly one model's state.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

from .core.registry import DEFAULT_METRICS, is_instance, props
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
        metrics: Sequence[str] = DEFAULT_METRICS,   # the portfolio the tournament sweeps
        num_images: int = 8012,
        temperature: float = 0.05,
        group_by: Sequence[str] = ("source_checkpoint_kimg", "cfg_scale"),
        block_search: bool = False,
        num_blocks: int = 1,
        safety_max_similarity_threshold: float | None = 0.97,
        utility_formula: str = "worst_anchored",
        dist_utility_formula: str | None = "random_anchored",
        dist_min_samples: dict[str, int] | None = None,
        seed: int = 0,
        seeds_per_metric_set: int = 1,               # tournament runs per metric set; more = more accurate weights
        metric_beta: float = 2.0,                    # the beta in F-beta the tournament scores by; 2.0 = F2
        scoring: ScoringFn | None = None,             # override: (true, pred, classes) -> {class: score};
                                                       # replaces F-beta entirely, and metric_beta is then unused
    ):
        if seeds_per_metric_set < 1:
            raise SynthCuratorError("seeds_per_metric_set must be >= 1")
        if metric_beta <= 0:
            raise SynthCuratorError(f"metric_beta must be positive, got {metric_beta}")
        if scoring is not None and not callable(scoring):
            raise SynthCuratorError("scoring must be callable: (true_labels, pred_labels, classes) -> {class: score}")
        for metric in metrics:
            props(metric)
        self.workdir = Path(workdir)
        self.target_distribution = target_distribution
        self.classes = sorted(target_distribution)
        self.model = model
        self.synthetic_dir = Path(synthetic_dir)
        self.real_dir = Path(real_dir)
        self.resolution = resolution
        self.features_dir = self.workdir / "features"
        self.metrics = list(metrics)
        self.num_images = num_images
        self.temperature = temperature
        self.group_by = list(group_by)
        self.block_search = block_search
        self.num_blocks = num_blocks
        self.safety_max_similarity_threshold = safety_max_similarity_threshold
        self.utility_formula = utility_formula
        self.dist_utility_formula = dist_utility_formula
        self.dist_min_samples = dist_min_samples
        self.seed = seed
        self.seeds = [seed + i for i in range(seeds_per_metric_set)]
        self.metric_beta = metric_beta
        self.scoring = scoring
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._validate()

    def _validate(self) -> None:
        """Check everything the pipeline needs is actually present, before any expensive work
        runs - directories with no images or a model missing a method should fail here, not
        after standardizing/embedding/scoring the whole pool."""
        if not callable(getattr(self.model, "train_and_evaluate", None)):
            raise SynthCuratorError("model must implement train_and_evaluate(train_manifest, seed) -> DataFrame")
        for name, path in (("synthetic_dir", self.synthetic_dir), ("real_dir", self.real_dir)):
            if not path.is_dir():
                raise SynthCuratorError(f"{name} does not exist or is not a directory: {path}")
            if not standardize.list_images(path):
                raise SynthCuratorError(f"{name} has no images (.jpg/.jpeg/.png) anywhere under it: {path}")
        if self.resolution <= 0:
            raise SynthCuratorError(f"resolution must be a positive integer, got {self.resolution}")
        if any(v < 0 for v in self.target_distribution.values()):
            raise SynthCuratorError(f"target_distribution values must be non-negative: {self.target_distribution}")
        total = sum(self.target_distribution.values())
        if abs(total - 100) > 1.0:
            raise SynthCuratorError(f"target_distribution must sum to ~100 (percent), got {total}")
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
            model = load_model(model_cfg["trainer"], model_cfg.get("params"))
        if model is None:
            raise SynthCuratorError("a model is required: add a config 'model:' block, or pass model=... here")
        synthetic_dir, real_dir, resolution = (
            inputs_cfg.get("synthetic_dir"), inputs_cfg.get("real_dir"), inputs_cfg.get("resolution"))
        if not (synthetic_dir and real_dir and resolution):
            raise SynthCuratorError("inputs.synthetic_dir, inputs.real_dir and inputs.resolution are all required")
        kw = dict(
            model=model, synthetic_dir=synthetic_dir, real_dir=real_dir, resolution=resolution,
            num_images=curation_cfg.get("num_images", 8012), temperature=curation_cfg.get("temperature", 0.05),
            group_by=curation_cfg.get("group_by", ["source_checkpoint_kimg", "cfg_scale"]),
            block_search=curation_cfg.get("block_search", False), num_blocks=curation_cfg.get("num_blocks", 1),
            safety_max_similarity_threshold=curation_cfg.get("safety_max_similarity_threshold", 0.97),
            utility_formula=weights_cfg.get("utility_formula", "worst_anchored"),
            dist_utility_formula=weights_cfg.get("dist_utility_formula", "random_anchored"),
            dist_min_samples=metrics_cfg.get("dist_min_samples"),
            seed=config.get("seed", 0), seeds_per_metric_set=config.get("tournament", {}).get("seeds_per_metric_set", 1),
            metric_beta=weights_cfg.get("beta", 2.0),
            scoring=import_dotted(weights_cfg["scoring"]) if weights_cfg.get("scoring") else None,
        )
        if metrics_cfg.get("instance") or metrics_cfg.get("distributional"):
            kw["metrics"] = [*metrics_cfg.get("instance", []), *metrics_cfg.get("distributional", [])]
        return cls(config["workdir"], config["target_distribution"], **kw)

    # ---- steps ------------------------------------------------------------------------------

    def standardize(self) -> Path:
        """1. Resize real images to the pool's generation resolution."""
        out = self.workdir / "real_standardized"
        if out.exists():
            return out
        return standardize.standardize(self.real_dir, out, self.resolution)

    def extract_features(self, std_dir: Path) -> Path:
        """2. DINOv2/Inception features, probabilities, KNN map, real labels."""
        if (self.features_dir / "synthetic_dino.pt").exists():
            return self.features_dir
        return features.extract_features(std_dir, self.synthetic_dir, self.features_dir, self.real_dir)

    def compute_metrics(self, std_dir: Path, features_dir: Path, wanted: Sequence[str] | None = None) -> Path:
        """3. Instance-metric columns -> metadata CSV. Returns the cached CSV untouched if it already has them all."""
        instance_metrics = [metric for metric in (wanted if wanted is not None else self.metrics)
                            if is_instance(metric)]
        return metrics.compute_metrics(
            self.synthetic_dir, std_dir, features_dir, self.workdir / "metadata.csv", instance_metrics)

    def curate_metric_sets(self, metadata_csv: Path, features_dir: Path) -> dict[str, dict[int, Path]]:
        """4. One curated dataset per metric + random control."""
        return metric_sets.curate_metric_sets(
            metadata_csv, features_dir, self.workdir / "metric_sets", self.metrics,
            self.target_distribution, self.num_images, self.group_by, self.seeds, self.synthetic_dir)

    def run_tournament(self, metric_sets: dict[str, dict[int, Path]]):
        """5. Train/eval your model on each metric set."""
        return tournament.run(metric_sets, self.model, self.workdir / "tournament", self.seeds)

    def derive_weights(self, predictions: dict) -> dict[str, dict]:
        """6. Per-class utility weights from the tournament predictions, scored with `scoring`
        if set, else F-`metric_beta`."""
        resolved = weights.derive_weights(
            predictions, self.classes, self.utility_formula, self.dist_utility_formula, self.metrics,
            beta=self.metric_beta, scoring=self.scoring)
        (self.workdir / "weights.json").write_text(json.dumps(resolved, indent=2))
        return resolved

    def run(self) -> Path:
        """The whole pipeline -> curated manifest CSV path."""
        std_dir = self.standardize()                              # 1
        features_dir = self.extract_features(std_dir)             # 2
        metadata_csv = self.compute_metrics(std_dir, features_dir)  # 3
        metric_sets = self.curate_metric_sets(metadata_csv, features_dir)  # 4
        predictions = self.run_tournament(metric_sets)             # 5
        resolved_weights = self.derive_weights(predictions)        # 6

        return curate.curate(                                      # 7
            metadata_csv, features_dir, resolved_weights, self.workdir / "curated",
            target_distribution=self.target_distribution, num_images=self.num_images,
            temperature=self.temperature, group_by=self.group_by,
            block_search=self.block_search, num_blocks=self.num_blocks,
            safety_max_similarity_threshold=self.safety_max_similarity_threshold, seed=self.seed,
            image_dir=self.synthetic_dir, min_samples=self.dist_min_samples)

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
