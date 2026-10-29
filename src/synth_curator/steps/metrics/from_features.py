"""Metrics computed straight from the step-2 feature tensors (fast; no images are read).

    fs-cs             cosine to the class prototype (mean DINOv2 vector of that class's real images)
    realism           squared distance to the k-th nearest REAL image (lower = more realistic)
    diversity         squared distance to the k-th nearest OTHER synthetic image (pool-wide)
    inception_score   per-image KL(p(y|x) || p(y)) from Inception probabilities
Port of calculate_{cosine,realism,diversity,inception_score}.py; FAISS replaced by core.geometry.knn_cosine.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch

from ...core.distributional import calculate_per_image_kl_divergence
from ...core.features import load_features, real_class_lookup
from ...core.geometry import knn_cosine
from ...exceptions import MissingFeaturesError

log = logging.getLogger(__name__)


def fscs(df, ctx):
    real = load_features(ctx.features_dir, "real", "dino")
    synth = load_features(ctx.features_dir, "synthetic", "dino")
    lookup = real_class_lookup(ctx.features_dir)        # needs real_labels.csv
    by_class: dict[str, list[torch.Tensor]] = {}
    for key, vec in real.items():
        cls = lookup.get(Path(key).stem)
        if cls is not None:
            by_class.setdefault(cls, []).append(vec.float())
    protos = {c: torch.stack(v).mean(dim=0) for c, v in by_class.items()}

    def score(row):
        v, p = synth.get(row.filename), protos.get(row.target_class_label)
        if v is None or p is None:
            return float("nan")
        return torch.nn.functional.cosine_similarity(v.float().unsqueeze(0), p.unsqueeze(0), eps=1e-8).item()

    df["fscs_score"] = [score(r) for r in df.itertuples()]
    return df


def realism(df, ctx, k: int = 3):
    real = load_features(ctx.features_dir, "real", "dino")
    synth = load_features(ctx.features_dir, "synthetic", "dino")
    d, _ = knn_cosine(torch.stack(list(synth.values())), torch.stack(list(real.values())), k)
    df["realism_score"] = df["filename"].map(dict(zip(synth, d[:, k - 1].astype(float))))
    return df


def diversity(df, ctx, k: int = 3):
    synth = load_features(ctx.features_dir, "synthetic", "dino")
    feats = torch.stack(list(synth.values()))
    d, _ = knn_cosine(feats, feats, k + 1)              # column 0 is the image itself
    df["diversity_score"] = df["filename"].map(dict(zip(synth, d[:, k].astype(float))))
    return df


def inception_score(df, ctx):
    probs = load_features(ctx.features_dir, "synthetic", "inception_probs")
    valid = [f for f in df["filename"] if f in probs]
    if not valid:
        raise MissingFeaturesError("no pool image found in synthetic_inception_probs.pt")
    kl = calculate_per_image_kl_divergence(torch.stack([probs[f] for f in valid]), eps=ctx.epsilon)
    df["is_contribution_score"] = df["filename"].map({f: s.item() for f, s in zip(valid, kl)})
    return df
