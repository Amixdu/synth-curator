"""Step 6. Raw predictions -> per-class utility weights (per model).

Port of v15 calculate_per_class_utilities. `formula` sets the instance-vs-distributional balance
(worst_anchored: all metrics weighted, none zeroed); `dist_formula` only reshuffles the fixed
distributional share among distributional metrics (random_anchored).
Also loads a shipped weight profile for Fast mode (domain-specific: only use it if your data matches).
"""
import json
from pathlib import Path

import pandas as pd

SHIPPED = Path(__file__).resolve().parent.parent / "shipped"


def derive_weights(predictions: dict[tuple[str, str], pd.DataFrame], model_name: str,
                   formula: str = "worst_anchored", dist_formula: str = "random_anchored") -> dict:
    raise NotImplementedError


def load_shipped_weights(profile: str) -> dict:
    """profile e.g. 'isic2018_swin_b' -> shipped/<profile>.json"""
    return json.loads((SHIPPED / f"{profile}.json").read_text())
