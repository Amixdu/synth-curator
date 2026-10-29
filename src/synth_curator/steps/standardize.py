"""Step 1. Center-crop + Lanczos-resize the real images to the synthetic pool's generation resolution,
so metrics compare real and synthetic at the exact pixel grid the generator produced.

`resolution` is REQUIRED - there is no universal default, since it depends entirely on your
generative model. Output is FLAT: `<flatten_name(path)>.png`, keyed on each image's full relative
path (not just its bare filename), so e.g. per-patient subfolders with same-named files don't
collide. `flatten_name` is also used by steps.features._write_real_labels, so the two always agree
on what a given real image is called - see that function before changing this one.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

import numpy as np

from ..exceptions import SynthCuratorError

log = logging.getLogger(__name__)

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def list_images(root: Path) -> list[Path]:
    images = []
    for dirpath, _, filenames in os.walk(root, followlinks=True):
        for f in filenames:
            if Path(f).suffix.lower() in IMAGE_EXTS:
                images.append(Path(dirpath) / f)
    return sorted(images)


def flatten_name(path: Path, root: Path) -> str:
    """A name for `path` unique within `root`, built from its full relative location rather than
    just its bare filename - so e.g. root/patient_001/x.jpg and root/patient_002/x.jpg don't
    collide once both are flattened into one directory."""
    return "__".join(Path(path).relative_to(root).with_suffix("").parts)


def center_crop_and_resize(img_path: Path, resolution: int):
    from PIL import Image
    img = np.array(Image.open(img_path).convert("RGB"))
    crop = min(img.shape[:2])
    img = img[(img.shape[0] - crop) // 2:(img.shape[0] + crop) // 2,
              (img.shape[1] - crop) // 2:(img.shape[1] + crop) // 2]
    return Image.fromarray(img, "RGB").resize((resolution, resolution), Image.Resampling.LANCZOS)


def standardize(real_dir: Path, out_dir: Path, resolution: int) -> Path:
    from tqdm import tqdm
    files = list_images(real_dir)
    if not files:
        raise SynthCuratorError(f"no images (.jpg/.jpeg/.png) under {real_dir}")
    names = [flatten_name(f, real_dir) for f in files]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise SynthCuratorError(
            f"two source images flatten to the same name (output is flat): {sorted(dupes)[:5]} ...")

    tmp = Path(str(out_dir) + ".partial")      # only rename when complete, so a crash never leaves a half-done dir
    tmp.mkdir(parents=True, exist_ok=True)
    for f, name in tqdm(list(zip(files, names, strict=True)), desc=f"Standardizing to {resolution}px"):
        try:
            center_crop_and_resize(f, resolution).save(tmp / f"{name}.png")
        except Exception as e:  # noqa: BLE001 - skip unreadable images, report at the end
            log.warning("skipping %s: %s", f, e)
    if out_dir.exists():
        # rebuilding into an already-populated out_dir (e.g. resolution changed on a reused
        # workdir) - os.rename() can't replace a non-empty directory, so clear it first. A crash
        # here just leaves out_dir absent, which the caller already treats as "needs building".
        shutil.rmtree(out_dir)
    tmp.rename(out_dir)
    return out_dir
