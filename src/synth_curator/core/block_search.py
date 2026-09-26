"""Optional per-class block search (block_search=True): distributionally score each class of each
drawn block against the same class of the real set, then pick the best block.

Port of v14 evaluate_distributional_utility_per_class / select_optimal_block_per_class /
write_class_log. Off by default in v15 (the winning block's U_dist did not track downstream F2).
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .distributional import calculate_fid, calculate_kid, calculate_swd, get_stats
from .features import load_features, real_class_lookup, stack_synthetic
from .registry import DEFAULT_DIST_MIN_SAMPLES, is_distributional, props
from .sampling import class_index
from .scoring import calculate_robust_bounds, calculate_robust_score

log = logging.getLogger(__name__)


def _usable_dist_metrics(dist_metrics, n_real, n_synth, min_samples):
    """Which distributional metrics a class has enough samples to estimate."""
    n = min(n_real, n_synth)
    return [m for m in dist_metrics if n >= min_samples[props(m)["family"]]]


def evaluate_distributional_utility_per_class(blocks, per_class_weights, features_dir: Path,
                                              classes: list[str], min_samples: dict | None = None):
    """Sets per block: U_dist_by_class, U_dist (mean over scored classes), dist_detail, distributional_scores."""
    min_samples = {**DEFAULT_DIST_MIN_SAMPLES, **(min_samples or {})}
    idx = class_index({c: 0 for c in classes})
    all_dist = sorted({m for w in per_class_weights.values() for m in w if is_distributional(m)})
    for b in blocks:
        b["U_dist"], b["U_dist_by_class"], b["dist_detail"], b["distributional_scores"] = 0.0, {}, [], {}
    if not all_dist:
        log.info("no distributional metrics in the weights: U_dist = 0 for all blocks")
        return blocks

    lookup = real_class_lookup(features_dir)
    spaces = sorted({props(m)["space"] for m in all_dist})
    synth = {s: load_features(features_dir, "synthetic", s) for s in spaces}
    real_by_class = {}
    for s in spaces:
        buckets = defaultdict(list)
        for key, vec in load_features(features_dir, "real", s).items():
            class_name = lookup.get(Path(key).stem)
            if class_name in idx:
                buckets[class_name].append(vec)
        real_by_class[s] = {c: torch.stack(v).float() for c, v in buckets.items()}

    real_stats_cache = {}
    for class_name in classes:
        weights = per_class_weights.get(class_name, {})
        cls_dist = [m for m in all_dist if m in weights]
        if not cls_dist:
            continue
        n_real = max(len(real_by_class[props(m)["space"]].get(class_name, [])) for m in cls_dist)
        n_synth = min(int((b["df"]["target_class_label"] == class_name).sum()) for b in blocks)
        usable = _usable_dist_metrics(cls_dist, n_real, n_synth, min_samples)
        if not usable:
            log.info("[%s] n_real=%d n_synth=%d: no estimable metric, class skipped", class_name, n_real, n_synth)
            continue
        share = sum(weights[m] for m in usable)
        if share <= 0:
            continue
        renorm = {m: weights[m] / share for m in usable}

        for m in usable:
            p = props(m)
            real_tensor = real_by_class[p["space"]][class_name]
            raw_scores = []
            for b in blocks:
                names = b["df"].loc[b["df"]["target_class_label"] == class_name, "filename"].tolist()
                st = stack_synthetic(synth[p["space"]], names)
                if p["family"] == "fid":
                    key = (class_name, p["space"])
                    if key not in real_stats_cache:
                        real_stats_cache[key] = get_stats(real_tensor)
                    mu_r, sg_r = real_stats_cache[key]
                    mu_s, sg_s = get_stats(st)
                    score = calculate_fid(mu_s, sg_s, mu_r, sg_r)
                elif p["family"] == "kid":
                    score = calculate_kid(st, real_tensor)
                else:  # swd needs equal row counts: trim the longer side
                    a, r = st, real_tensor
                    if a.shape[0] != r.shape[0]:
                        keep = min(a.shape[0], r.shape[0])
                        g = torch.Generator().manual_seed(b["idx"] * 1000 + idx[class_name])
                        if a.shape[0] > keep:
                            a = a[torch.randperm(a.shape[0], generator=g)[:keep]]
                        if r.shape[0] > keep:
                            r = r[torch.randperm(r.shape[0], generator=g)[:keep]]
                    score = calculate_swd(a, r, generator=torch.Generator().manual_seed(idx[class_name] * 97 + len(m)))
                raw_scores.append(score)

            series = pd.Series(raw_scores)
            vmin, vmax = calculate_robust_bounds(series)
            norm = calculate_robust_score(series, vmin, vmax, p["optimize"])
            for i, b in enumerate(blocks):
                b["U_dist_by_class"][class_name] = b["U_dist_by_class"].get(class_name, 0.0) + renorm[m] * float(norm.iloc[i])
                b["dist_detail"].append((class_name, m, raw_scores[i], float(norm.iloc[i])))
                b["distributional_scores"][f"Raw_{class_name}_{m}"] = raw_scores[i]
                b["distributional_scores"][f"Normalized_{class_name}_{m}"] = float(norm.iloc[i])

    for b in blocks:
        scored = b["U_dist_by_class"]
        b["U_dist"] = float(np.mean(list(scored.values()))) if scored else 0.0
    return blocks


def select_optimal_block_per_class(blocks):
    """Argmax of U_dist. Selection is purely distributional (instance quality already shaped the draw);
    U_inst is recorded for logging only."""
    best, best_score = None, -float("inf")
    for b in blocks:
        b["U_inst"] = b["df"]["S_Composite"].mean()
        b["S_Final"] = b["U_dist"]
        if b["U_dist"] > best_score:
            best, best_score = b, b["U_dist"]
    log.info("winner: block %d (U_dist=%.4f)", best["idx"], best_score)
    return best["df"], blocks


def write_class_log(blocks, path: Path) -> None:
    rows = [{"Block_ID": b["idx"], "target_class_label": class_name, "metric": m, "raw": raw, "normalized": norm,
             "U_dist_class": b["U_dist_by_class"].get(class_name), "U_dist_block": b["U_dist"]}
            for b in blocks for class_name, m, raw, norm in b["dist_detail"]]
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)
