"""SynthCurator: the whole pipeline as one object. Each step is a method; `run()` chains them.

    FULL MODE   standardize -> extract_features -> compute_metrics -> curate_arms
                -> run_tournament -> derive_weights -> curate
    FAST MODE   pool already has metrics + features, weights are shipped/supplied -> curate

Every step writes into `workdir` and is skipped if its output already exists, so a crashed run
resumes where it stopped (delete the file/dir to force a re-run).
"""
from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

from .exceptions import SynthCuratorError
from .model_interface import Model
from .steps import arms, curate, features, metrics, standardize, tournament, weights

log = logging.getLogger(__name__)


class SynthCurator:
    def __init__(
        self,
        workdir: str | Path,
        target_distribution: dict[str, float],
        *,
        models: Sequence[Model] = (),         # your classifiers; needed for the tournament (full mode)
        synthetic_dir: str | Path | None = None,
        real_dir: str | Path | None = None,
        resolution: int | None = None,        # synthetic pool's generation resolution
        metadata_csv: str | Path | None = None,   # Fast mode: metrics already computed
        features_dir: str | Path | None = None,   # Fast mode: features already extracted
        weights_json: str | Path | None = None,   # your own weights; None -> shipped_profile
        shipped_profile: str | None = None,       # e.g. "isic2018_swin_b" (domain-specific!)
        block_size: int = 8012,
        temperature: float = 0.05,
        group_by: Sequence[str] = ("source_checkpoint_kimg", "cfg_scale"),
        block_search: bool = False,
        num_blocks: int = 1,
        safety_max_similarity_threshold: float | None = 0.97,
        utility_formula: str = "worst_anchored",
        dist_utility_formula: str = "random_anchored",
        seed: int = 0,
        seeds_per_arm: int = 1,   # tournament runs per (arm, model); more = more accurate weights
    ):
        if seeds_per_arm < 1:
            raise SynthCuratorError("seeds_per_arm must be >= 1")
        names = [m.name for m in models]
        if len(set(names)) != len(names):
            raise SynthCuratorError(f"model names must be unique: {names}")
        self.workdir = Path(workdir)
        self.target_distribution = target_distribution
        self.models = {m.name: m for m in models}
        self.synthetic_dir = Path(synthetic_dir) if synthetic_dir else None
        self.real_dir = Path(real_dir) if real_dir else None
        self.resolution = resolution
        self.metadata_csv = Path(metadata_csv) if metadata_csv else self.workdir / "metadata.csv"
        self.features_dir = Path(features_dir) if features_dir else self.workdir / "features"
        self.weights_json = Path(weights_json) if weights_json else None
        self.shipped_profile = shipped_profile
        self.block_size = block_size
        self.temperature = temperature
        self.group_by = list(group_by)
        self.block_search = block_search
        self.num_blocks = num_blocks
        self.safety_max_similarity_threshold = safety_max_similarity_threshold
        self.utility_formula = utility_formula
        self.dist_utility_formula = dist_utility_formula
        self.seed = seed
        self.seeds_per_arm = seeds_per_arm
        self.workdir.mkdir(parents=True, exist_ok=True)

    # ---- steps ------------------------------------------------------------------------------

    def standardize(self) -> Path:
        """1. Resize real images to the pool's generation resolution."""
        out = self.workdir / "real_standardized"
        if out.exists():
            return out
        if not (self.real_dir and self.resolution):
            raise SynthCuratorError("standardize needs real_dir and resolution")
        return standardize.standardize(self.real_dir, out, self.resolution)

    def extract_features(self) -> Path:
        """2. DINOv2/Inception features + probabilities + KNN map."""
        if self.features_dir.exists():
            return self.features_dir
        if not self.synthetic_dir:
            raise SynthCuratorError("extract_features needs synthetic_dir")
        return features.extract_features(self.standardize(), self.synthetic_dir, self.features_dir)

    def compute_metrics(self) -> Path:
        """3. 13-metric pipeline -> metadata CSV."""
        if self.metadata_csv.exists():
            return self.metadata_csv
        if not (self.synthetic_dir and self.real_dir):
            raise SynthCuratorError("compute_metrics needs synthetic_dir and real_dir")
        return metrics.compute_metrics(
            self.synthetic_dir, self.standardize(), self.extract_features(), self.metadata_csv)

    def curate_arms(self) -> dict[str, Path]:
        """4. One curated dataset per metric + random control."""
        return arms.curate_arms(
            self.compute_metrics(), self.workdir / "arms", self.target_distribution,
            self.block_size, self.seed)

    def run_tournament(self):
        """5. Train/eval each arm with each of your models."""
        if not self.models:
            raise SynthCuratorError("the tournament needs at least one model (models=[...])")
        return tournament.run_tournament(
            self.curate_arms(), list(self.models.values()), self.workdir / "tournament",
            [self.seed + i for i in range(self.seeds_per_arm)])

    def derive_weights(self) -> dict[str, dict]:
        """6. Per-model, per-class utility weights from the tournament predictions."""
        preds = self.run_tournament()
        out = {}
        for name in self.models:
            out[name] = weights.derive_weights(
                preds, name, self.utility_formula, self.dist_utility_formula)
            path = self.workdir / "weights" / f"{name}.json"
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(out[name], indent=2))
        return out

    def get_weights(self, model_name: str | None = None) -> dict:
        """Priority: your weights file > shipped profile > (cached or freshly) derived from the tournament."""
        if self.weights_json:
            return json.loads(self.weights_json.read_text())
        if self.shipped_profile:
            return weights.load_shipped_weights(self.shipped_profile)
        if model_name is None or model_name not in self.models:
            raise SynthCuratorError(
                f"no weights: pass weights_json, shipped_profile, or a known model (have {list(self.models)})")
        cached = self.workdir / "weights" / f"{model_name}.json"
        if cached.exists():
            return json.loads(cached.read_text())
        return self.derive_weights()[model_name]

    def curate(self, model_name: str | None = None) -> Path:
        """7. Framework curation with the weights -> manifest CSV path.

        `model_name` picks which model's weights to use (and names the output dir); it can be
        omitted when weights_json / shipped_profile is given."""
        w = self.get_weights(model_name)
        return curate.curate(
            self.compute_metrics() if self.synthetic_dir else self.metadata_csv,
            self.features_dir, w, self.workdir / "curated" / (model_name or "default"),
            target_distribution=self.target_distribution, block_size=self.block_size,
            temperature=self.temperature, group_by=self.group_by,
            block_search=self.block_search, num_blocks=self.num_blocks,
            safety_max_similarity_threshold=self.safety_max_similarity_threshold, seed=self.seed)

    def run(self, model_name: str | None = None) -> Path:
        """Whole pipeline. Steps 1-6 only run if the weights they produce are actually needed."""
        return self.curate(model_name)
