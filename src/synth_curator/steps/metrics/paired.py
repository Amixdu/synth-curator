"""Full-reference metrics against each synthetic image's nearest real image (features_dir/knn_map.json).

Both images are resized to 256x256. Score = best over the mapped neighbours (max; LPIPS: min, it is a distance).
Port of calculate_{psnr,ssim,fsim,lpips}.py. Needs the [full] extra (scikit-image; opencv + phasepack for FSIM; lpips).
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TARGET_SIZE = (256, 256)


def _load_resized(path) -> np.ndarray:
    from skimage import io, transform
    img = io.imread(path)
    if img.ndim == 2:
        img = np.stack((img,) * 3, axis=-1)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return transform.resize(img, TARGET_SIZE, anti_aliasing=True, preserve_range=True).astype(np.uint8)


def _run_pairs(df: pd.DataFrame, ctx, column: str, pair_fn, best) -> pd.DataFrame:
    from tqdm import tqdm
    knn = json.loads((ctx.features_dir / "knn_map.json").read_text())
    real_cache: dict[str, np.ndarray | None] = {}

    def real(name):
        if name not in real_cache:
            try:
                real_cache[name] = _load_resized(ctx.real_std_dir / name)
            except Exception:  # noqa: BLE001 - broken real image: skip it
                real_cache[name] = None
        return real_cache[name]

    todo = df.index[df[column].isna()]
    for n, idx in enumerate(tqdm(todo, desc=column)):
        name = df.at[idx, "filename"]
        neighbours = knn.get(name)
        if not neighbours:
            continue
        try:
            synth = _load_resized(ctx.synthetic_dir / name)
        except Exception as e:  # noqa: BLE001
            log.warning("cannot load %s: %s", name, e)
            continue
        scores = [pair_fn(synth, r) for r in map(real, neighbours) if r is not None]
        scores = [s for s in scores if s is not None]
        if scores:
            df.at[idx, column] = best(scores)
        if (n + 1) % ctx.checkpoint_every == 0:
            ctx.save(df)
    return df


def psnr_pair(synth: np.ndarray, real: np.ndarray):
    if synth.shape != real.shape:
        return None
    mse = np.mean((synth.astype(np.float64) - real.astype(np.float64)) ** 2)
    return 100.0 if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))


def ssim_pair(synth: np.ndarray, real: np.ndarray):
    from skimage import img_as_float
    from skimage.metrics import structural_similarity
    return structural_similarity(img_as_float(real), img_as_float(synth), channel_axis=-1, data_range=1.0)


def fsim_pair(synth: np.ndarray, real: np.ndarray, alpha: float = 1.0) -> float:
    import cv2
    from phasepack.phasecong import phasecong
    from skimage import img_as_float
    try:
        ref = img_as_float(cv2.cvtColor(real, cv2.COLOR_RGB2GRAY))
        syn = img_as_float(cv2.cvtColor(synth, cv2.COLOR_RGB2GRAY))
        pc_ref, pc_syn = phasecong(ref)[0], phasecong(syn)[0]
    except Exception:  # noqa: BLE001
        return 0.0
    gm_ref = np.sqrt(cv2.Sobel(ref, cv2.CV_64F, 1, 0, ksize=3) ** 2 + cv2.Sobel(ref, cv2.CV_64F, 0, 1, ksize=3) ** 2)
    gm_syn = np.sqrt(cv2.Sobel(syn, cv2.CV_64F, 1, 0, ksize=3) ** 2 + cv2.Sobel(syn, cv2.CV_64F, 0, 1, ksize=3) ** 2)
    t1, t2 = 0.085, 160
    s_pc = (2 * pc_ref * pc_syn + t1) / (pc_ref ** 2 + pc_syn ** 2 + t1)
    s_gm = (2 * gm_ref * gm_syn + t2) / (gm_ref ** 2 + gm_syn ** 2 + t2)
    weight = np.maximum(pc_ref, pc_syn)
    denom = np.sum(weight)
    return 1.0 if denom == 0 else float(np.sum(s_pc * (s_gm ** alpha) * weight) / denom)


def psnr(df, ctx):
    return _run_pairs(df, ctx, "psnr_score", psnr_pair, max)


def ssim(df, ctx):
    return _run_pairs(df, ctx, "ssim_score", ssim_pair, max)


def fsim(df, ctx):
    return _run_pairs(df, ctx, "fsim_score", fsim_pair, max)


def lpips(df, ctx, network: str = "alex", batch_size: int = 64, num_workers: int = 4):
    """LPIPS distance (lower = closer); best neighbour = minimum. Batched on GPU when available."""
    from collections import defaultdict

    import lpips as lpips_lib
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from tqdm import tqdm

    knn = json.loads((ctx.features_dir / "knn_map.json").read_text())
    tf = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(),
                             transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
    tasks = [(idx, df.at[idx, "filename"], r) for idx in df.index[df["lpips_score"].isna()]
             for r in knn.get(df.at[idx, "filename"], [])]

    class Pairs(Dataset):
        def __len__(self):
            return len(tasks)

        def __getitem__(self, i):
            idx, s, r = tasks[i]
            try:
                return idx, tf(Image.open(ctx.synthetic_dir / s).convert("RGB")), \
                    tf(Image.open(ctx.real_std_dir / r).convert("RGB")), True
            except Exception:  # noqa: BLE001
                z = torch.zeros((3, 256, 256))
                return idx, z, z, False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = lpips_lib.LPIPS(net=network).to(device).eval()
    acc = defaultdict(list)
    for idxs, s, r, ok in tqdm(DataLoader(Pairs(), batch_size=batch_size, num_workers=num_workers), desc="lpips_score"):
        if not ok.any():
            continue
        with torch.no_grad():
            scores = model(r[ok].to(device), s[ok].to(device)).flatten().cpu().numpy()
        for i, sc in zip(idxs[ok].tolist(), scores):
            acc[i].append(sc)
    for idx, scores in acc.items():
        df.at[idx, "lpips_score"] = min(scores)
    return df
