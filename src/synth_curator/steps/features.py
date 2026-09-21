"""Step 2. DINOv2 + InceptionV3 features, Inception class probabilities (real & synthetic), KNN map (real).

Port of utils/feature_extraction/extract_*.py and create_knn_map.py.
Writes into `out_dir`: {real,synthetic}_{dino,inception}.pt, {real,synthetic}_inception_probs.pt, knn_map.pt
"""
from pathlib import Path


def extract_features(real_dir: Path, synthetic_dir: Path, out_dir: Path) -> Path:
    raise NotImplementedError
