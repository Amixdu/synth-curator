"""Step 1. Center-crop + Lanczos-resize the real images to the synthetic pool's generation resolution,
so metrics compare real and synthetic at the exact pixel grid the generator produced.

`resolution` is REQUIRED - there is no universal default, since it depends entirely on your
generative model. Output is FLAT: `<stem>.png`; duplicate stems are an error.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..exceptions import SynthCuratorError

log = logging.getLogger(__name__)

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def list_images(root: Path) -> list[Path]:
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMAGE_EXTS)


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
    stems = [f.stem for f in files]
    dupes = {s for s in stems if stems.count(s) > 1}
    if dupes:
        raise SynthCuratorError(f"duplicate real image stems (output is flat): {sorted(dupes)[:5]} ...")

    tmp = Path(str(out_dir) + ".partial")      # only rename when complete, so a crash never leaves a half-done dir
    tmp.mkdir(parents=True, exist_ok=True)
    for f in tqdm(files, desc=f"Standardizing to {resolution}px"):
        try:
            center_crop_and_resize(f, resolution).save(tmp / f"{f.stem}.png")
        except Exception as e:  # noqa: BLE001 - skip unreadable images, report at the end
            log.warning("skipping %s: %s", f, e)
    tmp.rename(out_dir)
    return out_dir
