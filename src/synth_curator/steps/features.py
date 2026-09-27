"""Step 2. Feature assets for the metric pipeline and distributional metrics.

Writes into `out_dir`:
    {real,synthetic}_dino.pt              DINOv2-base [CLS] vectors     (FID/KID/SWD-DINO, safety, FS-CS, realism, diversity)
    {real,synthetic}_inception.pt         InceptionV3 Mixed_7c pooled   (FID-Inception)
    {real,synthetic}_inception_probs.pt   InceptionV3 softmax           (Inception Score)
    knn_map.json                          synthetic filename -> [nearest real filename] (paired metrics)
    real_labels.csv                       filename, class (from real_dir's class sub-folders, if any)
Each .pt is {relative image path: 1-D tensor}. Port of utils/feature_extraction/*.
Needs the [full] extra (torch, torchvision, transformers, pillow).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import torch

from ..core.features import feature_path, load_features
from ..core.geometry import knn_cosine
from .standardize import flatten_name, list_images

log = logging.getLogger(__name__)

DINO_MODEL = "facebook/dinov2-base"
BATCH = 32


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _extract(image_dir: Path, embed, batch_size: int = BATCH, desc: str = "") -> tuple[dict, ...]:
    """embed(list[PIL.Image]) -> tuple of (B, D) tensors. Returns one {relative path: vector} dict per output."""
    from PIL import Image
    from tqdm import tqdm
    paths = list_images(image_dir)
    out: tuple[dict, ...] | None = None
    for i in tqdm(range(0, len(paths), batch_size), desc=desc):
        batch = paths[i:i + batch_size]
        imgs, ok = [], []
        for p in batch:
            try:
                imgs.append(Image.open(p).convert("RGB"))
                ok.append(p)
            except Exception as e:  # noqa: BLE001
                log.warning("skipping %s: %s", p, e)
        if not imgs:
            continue
        with torch.no_grad():
            vecs = [v.cpu() for v in embed(imgs)]
        out = out or tuple({} for _ in vecs)
        for j, d in enumerate(out):
            for p, v in zip(ok, vecs[j]):
                d[str(p.relative_to(image_dir))] = v
    return out or ()


def _dino_embedder():
    from transformers import AutoImageProcessor, AutoModel
    device = _device()
    processor = AutoImageProcessor.from_pretrained(DINO_MODEL)
    model = AutoModel.from_pretrained(DINO_MODEL).to(device).eval()

    def embed(imgs):
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        return (model(**inputs).last_hidden_state[:, 0],)
    return embed


def _inception_embedder():
    """One InceptionV3 pass yields both the Mixed_7c pooled features (FID) and softmax probs (IS)."""
    from torchvision import models, transforms
    device = _device()
    model = models.inception_v3(weights="IMAGENET1K_V1").to(device).eval()
    captured = {}
    model.Mixed_7c.register_forward_hook(
        lambda m, i, o: captured.__setitem__("f", torch.nn.functional.adaptive_avg_pool2d(o, (1, 1)).flatten(1)))
    tf = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def embed(imgs):
        logits = model(torch.stack([tf(i) for i in imgs]).to(device))
        return captured["f"], torch.softmax(logits, dim=1)
    return embed


def _write_real_labels(real_dir: Path, out_csv: Path) -> None:
    # filename must match what standardize() actually named this image (flatten_name + ".png"),
    # since that's the key its DINOv2/Inception features end up under - not its original name.
    rows = [(f"{flatten_name(p, real_dir)}.png", p.parent.name)
            for p in list_images(real_dir) if p.parent != real_dir]
    if rows:
        pd.DataFrame(rows, columns=["filename", "class"]).to_csv(out_csv, index=False)
    else:
        log.info("real_dir has no class sub-folders: real_labels.csv not written "
                 "(FS-CS and block_search need it; provide it yourself)")


def build_knn_map(features_dir: Path, k: int = 1) -> Path:
    """Nearest real image (DINOv2 cosine) for every synthetic image."""
    real, synth = load_features(features_dir, "real", "dino"), load_features(features_dir, "synthetic", "dino")
    real_names, synth_names = list(real), list(synth)
    _, idx = knn_cosine(torch.stack(list(synth.values())), torch.stack(list(real.values())), k)
    knn = {s: [real_names[idx[i][k - 1]]] for i, s in enumerate(synth_names)}
    path = Path(features_dir) / "knn_map.json"
    path.write_text(json.dumps(knn))
    return path


def extract_features(real_std_dir: Path, synthetic_dir: Path, out_dir: Path, real_dir: Path | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dino = inception = None
    for role, root in (("real", real_std_dir), ("synthetic", synthetic_dir)):
        if not feature_path(out_dir, role, "dino").exists():
            dino = dino or _dino_embedder()
            (vecs,) = _extract(root, dino, desc=f"DINOv2 ({role})")
            torch.save(vecs, feature_path(out_dir, role, "dino"))
        if not (feature_path(out_dir, role, "inception").exists()
                and feature_path(out_dir, role, "inception_probs").exists()):
            inception = inception or _inception_embedder()
            feats, probs = _extract(root, inception, desc=f"InceptionV3 ({role})")
            torch.save(feats, feature_path(out_dir, role, "inception"))
            torch.save(probs, feature_path(out_dir, role, "inception_probs"))
    if not (out_dir / "knn_map.json").exists():
        build_knn_map(out_dir)
    if real_dir is not None and not (out_dir / "real_labels.csv").exists():
        _write_real_labels(Path(real_dir), out_dir / "real_labels.csv")
    return out_dir
