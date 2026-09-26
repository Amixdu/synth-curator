"""Group distributional scores: one score per (generation group, distributional metric),
normalized to [0,1] across groups (1.0 = best), inherited by every image in the group.

Port of v15 compute_group_scores. Computed live from feature tensors so any `group_by`
partition works. Groups are large enough that FID is well estimated (per-class it is not).
Group scores are global (all classes pooled); how much a class trusts them is per class (weights).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
import torch

from .distributional import calculate_fid, calculate_kid, calculate_swd, get_stats
from .features import load_features, stack_real, stack_synthetic
from .registry import props
from .scoring import calculate_robust_bounds, calculate_robust_score, group_key

log = logging.getLogger(__name__)

SWD_MAX_N = 8192   # cap the group-vs-real SWD sample so many groups stay affordable


def compute_group_raw(df: pd.DataFrame, features_dir: Path, group_cols: list[str],
                      dist_metrics: list[str]) -> pd.DataFrame:
    """Raw distributional value per (group, metric); lower is better for all of them."""
    df = df.assign(_group=group_key(df, group_cols))
    groups = sorted(df["_group"].unique())
    log.info("computing group scores over %s: %d groups", group_cols, len(groups))

    spaces = sorted({props(m)["space"] for m in dist_metrics})
    real = {s: stack_real(load_features(features_dir, "real", s)) for s in spaces}
    synth = {s: load_features(features_dir, "synthetic", s) for s in spaces}

    real_stats: dict[str, tuple] = {}
    raw: dict[str, list[float]] = {m: [] for m in dist_metrics}
    for gi, g in enumerate(groups):
        names = df.loc[df["_group"] == g, "filename"].tolist()
        for m in dist_metrics:
            p = props(m)
            rt = real[p["space"]]
            st = stack_synthetic(synth[p["space"]], names)
            if p["family"] == "fid":
                if p["space"] not in real_stats:
                    real_stats[p["space"]] = get_stats(rt)
                mu_r, sg_r = real_stats[p["space"]]
                mu_s, sg_s = get_stats(st)
                raw[m].append(calculate_fid(mu_s, sg_s, mu_r, sg_r))
            elif p["family"] == "kid":
                raw[m].append(calculate_kid(st, rt))
            else:
                gen = torch.Generator().manual_seed(gi)
                n = min(st.shape[0], rt.shape[0], SWD_MAX_N)
                a = st[torch.randperm(st.shape[0], generator=gen)[:n]]
                b = rt[torch.randperm(rt.shape[0], generator=gen)[:n]]
                raw[m].append(calculate_swd(a, b, generator=torch.Generator().manual_seed(gi)))

    return pd.DataFrame({m: pd.Series(raw[m], index=groups) for m in dist_metrics})


def compute_group_scores(df: pd.DataFrame, features_dir: Path, group_cols: list[str],
                         dist_metrics: list[str], cache_path: Path | None = None) -> pd.DataFrame:
    """Raw group values normalized across groups to [0,1], 1.0 = best."""
    if cache_path is not None and Path(cache_path).exists():
        g = pd.read_csv(cache_path, index_col=0)
        if set(dist_metrics).issubset(g.columns):
            log.info("group scores loaded from cache %s", cache_path)
            return g

    raw = compute_group_raw(df, features_dir, group_cols, dist_metrics)
    scores = {}
    for m in dist_metrics:
        vmin, vmax = calculate_robust_bounds(raw[m])
        scores[m] = calculate_robust_score(raw[m], vmin, vmax, props(m)["optimize"])
        log.info("%s best group: %s (raw %.4f)", m, scores[m].idxmax(), raw[m][scores[m].idxmax()])
    out = pd.DataFrame(scores)

    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(cache_path.suffix + f".tmp{os.getpid()}")
        out.to_csv(tmp)
        os.replace(tmp, cache_path)
    return out
