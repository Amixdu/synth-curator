"""Step 1. Lanczos-resize real images to the synthetic pool's generation resolution.

Port of utils/standardize_images.py. `resolution` is required (thesis hardcoded 512 for EDM2).
"""
from pathlib import Path


def standardize(real_dir: Path, out_dir: Path, resolution: int) -> Path:
    raise NotImplementedError
