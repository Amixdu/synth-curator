"""One-image-at-a-time metrics: BRISQUE and CLIP. Port of calculate_brisque.py / calculate_clip.py.

CLIP writes two prompt similarities (logits) and their log-odds: log(cancer / (artifact + eps)).
The prompts are DOMAIN-SPECIFIC (skin lesions); override `clip_prompts` for another modality.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CLIP_MODEL = "openai/clip-vit-base-patch32"
CLIP_PROMPTS = {
    "clip_similarity_cancer": "A clinical photograph of a skin cancer lesion",
    "clip_similarity_artifact": "A computer generated image, an artificial image, a fake image",
}


def _fill(df: pd.DataFrame, ctx, column: str, compute, check_cols: list[str] | None = None) -> pd.DataFrame:
    from tqdm import tqdm
    cols = check_cols or [column]
    todo = df.index[df[cols].isna().any(axis=1)]
    for n, idx in enumerate(tqdm(todo, desc=column)):
        path = ctx.synthetic_dir / df.at[idx, "filename"]
        if not path.is_file():
            log.warning("image not found: %s", path)
            continue
        result = compute(path)
        if result is not None:
            for k, v in result.items():
                df.at[idx, k] = v
        if (n + 1) % ctx.checkpoint_every == 0:
            ctx.save(df)
    return df


def brisque(df, ctx):
    from brisque import BRISQUE
    from skimage import io
    scorer = BRISQUE(url=False)

    def compute(path):
        try:
            return {"brisqe_score": float(scorer.score(img=io.imread(path)))}
        except Exception:  # noqa: BLE001
            return None
    return _fill(df, ctx, "brisqe_score", compute)


def clip(df, ctx, prompts: dict[str, str] | None = None):
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    prompts = prompts or CLIP_PROMPTS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(CLIP_MODEL, use_safetensors=False).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    cols, texts = list(prompts), list(prompts.values())

    def compute(path):
        try:
            inputs = processor(text=texts, images=Image.open(path).convert("RGB"),
                               return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits_per_image
            return dict(zip(cols, logits.squeeze().cpu().tolist()))
        except Exception as e:  # noqa: BLE001
            log.warning("CLIP failed on %s: %s", path.name, e)
            return None

    df = _fill(df, ctx, "clip", compute, check_cols=cols)
    cancer = pd.to_numeric(df["clip_similarity_cancer"], errors="coerce")
    artifact = pd.to_numeric(df["clip_similarity_artifact"], errors="coerce")
    ok = cancer.notna() & artifact.notna()
    df.loc[ok, "clip_log_odds_score"] = np.log(cancer[ok] / (artifact[ok] + ctx.epsilon))
    return df
