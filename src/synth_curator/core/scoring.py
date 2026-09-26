"""Robust bounds/scores and the per-class composite score.

Port of framework_utils/utils.py (calculate_robust_bounds/_score) and
v15 calculate_per_class_score.
"""
from __future__ import annotations

import logging

import pandas as pd

from .registry import is_distributional, is_instance, props

log = logging.getLogger(__name__)


def calculate_robust_bounds(series: pd.Series) -> tuple[float, float]:
    """1st / 99th percentile: a scoring range that ignores extreme outliers."""
    return series.quantile(0.01), series.quantile(0.99)


def calculate_robust_score(series: pd.Series, vmin: float, vmax: float, optimize_dir: str) -> pd.Series:
    """Clip to [vmin, vmax], min-max normalize to [0,1]; inverted for 'min' so 1.0 is always best."""
    clipped = series.clip(lower=vmin, upper=vmax)
    range_val = vmax - vmin if (vmax - vmin) > 0 else 1e-8
    normalized = (clipped - vmin) / range_val
    return 1.0 - normalized if optimize_dir == "min" else normalized


def group_key(df: pd.DataFrame, group_cols: list[str]) -> pd.Series:
    return df[group_cols].astype(str).agg("|".join, axis=1)


def calculate_per_class_score(df: pd.DataFrame, per_class_weights: dict, group_scores: pd.DataFrame | None,
                              group_cols: list[str], per_class_bounds: bool = True) -> pd.DataFrame:
    """Writes `S_Composite`: per-class-weighted instance terms + group distributional terms.

    Both terms are in [0,1] and weights are renormalized over what is usable within a class, so
    S stays in [0,1] and temperature means the same thing for every class. Robust bounds are
    per class because sampling is a softmax over one class's rows.
    """
    df["S_Composite"] = 0.0
    gkey = group_key(df, group_cols) if group_scores is not None else None

    for class_name, weights in per_class_weights.items():
        mask = df["target_class_label"] == class_name
        n = int(mask.sum())
        if n == 0:
            continue
        sub = df.loc[mask]
        score = pd.Series(0.0, index=sub.index)

        inst = [m for m, w in weights.items() if w > 1e-9 and is_instance(m) and m in df.columns]
        dist = [m for m, w in weights.items()
                if w > 1e-9 and is_distributional(m)
                and group_scores is not None and m in group_scores.columns]

        used = sum(weights[m] for m in inst + dist)
        if used <= 1e-9:
            log.warning("[%s] no usable weight - sampling uniformly for this class", class_name)
            continue
        w = {m: weights[m] / used for m in inst + dist}

        for m in inst:
            vmin, vmax = calculate_robust_bounds(sub[m] if per_class_bounds else df[m])
            norm = calculate_robust_score(sub[m], vmin, vmax, props(m)["optimize"])
            df.loc[mask, f"{m}_norm"] = norm
            score += w[m] * norm
        for m in dist:
            score += w[m] * gkey[mask].map(group_scores[m]).astype(float).fillna(0.0)

        df.loc[mask, "S_Composite"] = score
        i_share = sum(w[m] for m in inst)
        log.info("[%s] n=%d instance %.1f%% group %.1f%%", class_name, n, i_share * 100, (1 - i_share) * 100)
    return df
