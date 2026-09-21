"""Step 5. Train + evaluate each user model on each arm, `seeds_per_arm` times.

Training is the user's `Model.fit_predict` (see model_interface.py). Default 1 seed per arm is a weighting
signal, not a statistical claim; more seeds average out training noise.
Returns {(model_name, arm_name): predictions_df}, seeds stacked with an added `seed` column.
Port of final_rerun/scripts/6_tournament_final.py (orchestration only).
"""
from pathlib import Path

import pandas as pd

from ..model_interface import Model


def run_tournament(arms: dict[str, Path], models: list[Model], out_dir: Path,
                   seeds: list[int]) -> dict[tuple[str, str], pd.DataFrame]:
    raise NotImplementedError
