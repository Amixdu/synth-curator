"""Feature-tensor access. A feature space is two dicts {filename: 1-D tensor}, one for the real
reference set and one for the synthetic pool, stored as `{real,synthetic}_{space}.pt` in
`features_dir` (space = dino | inception). Real class labels (only needed for block_search) live
in `real_labels.csv` (columns: filename, class)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from ..exceptions import MissingFeaturesError


def feature_path(features_dir: Path, role: str, space: str) -> Path:
    return Path(features_dir) / f"{role}_{space}.pt"


def load_features(features_dir: Path, role: str, space: str) -> dict[str, torch.Tensor]:
    path = feature_path(features_dir, role, space)
    if not path.exists():
        raise MissingFeaturesError(f"missing feature file {path}")
    return torch.load(path, map_location="cpu")


def stack_real(real: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack(list(real.values())).float()


def stack_synthetic(synth: dict[str, torch.Tensor], filenames) -> torch.Tensor:
    try:
        return torch.stack([synth[f] for f in filenames]).float()
    except KeyError as e:
        raise MissingFeaturesError(f"image {e} is in the metadata but not in the feature dict") from None


def real_class_lookup(features_dir: Path) -> dict[str, str]:
    """{filename stem: class} for the real reference set (FS-CS and block_search need it)."""
    path = Path(features_dir) / "real_labels.csv"
    if not path.exists():
        raise MissingFeaturesError(f"missing {path} (columns: filename, class); FS-CS and block_search need real class labels")
    df = pd.read_csv(path)
    return {Path(f).stem: c for f, c in zip(df["filename"], df["class"])}
