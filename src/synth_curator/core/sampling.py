"""Temperature-scaled per-class sampling.

SEED FORMULA IS FROZEN for reproducibility across runs:
    seed_offset * 100_000 + block_idx * 100 + class_idx
where class_idx is the class's position in sorted class order.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def class_index(target_proportions: dict[str, float]) -> dict[str, int]:
    return {c: i for i, c in enumerate(sorted(target_proportions))}


def class_counts(target_proportions: dict[str, float], block_size: int) -> dict[str, int]:
    """Per-class image counts for a block (percent proportions); rounding remainder goes to the largest class."""
    counts, total = {}, 0
    for class_name, pct in target_proportions.items():
        counts[class_name] = max(1, math.floor(pct / 100.0 * block_size))
        total += counts[class_name]
    rem = block_size - total
    if rem > 0:
        counts[max(counts, key=counts.get)] += rem
    return counts


def draw_blocks(df: pd.DataFrame, target_proportions: dict[str, float], block_size: int,
                num_blocks: int, temperature: float, seed_offset: int) -> list[dict]:
    """Draw `num_blocks` blocks: per class, sample without replacement with p ~ exp(S_Composite/T)."""
    counts = class_counts(target_proportions, block_size)
    idx = class_index(target_proportions)
    blocks = []
    for bi in range(num_blocks):
        rows = []
        for class_name, count in counts.items():
            cdf = df[df["target_class_label"] == class_name].copy()
            if len(cdf) < count:
                log.warning("pool exhausted for class %r: requested %d, only %d available; block will be undersized",
                            class_name, count, len(cdf))
            count = min(count, len(cdf))
            s = cdf["S_Composite"] / temperature
            s -= s.max()
            p = np.exp(s)
            p /= p.sum()
            rs = seed_offset * 100_000 + bi * 100 + idx.get(class_name, 0)
            rows.append(cdf.sample(n=count, replace=False, weights=p, random_state=rs).copy())
        bdf = pd.concat(rows)
        bdf["Block_ID"] = bi + 1
        blocks.append({"idx": bi + 1, "df": bdf})
        log.info("block %d/%d: %d images (seed_offset=%d)", bi + 1, num_blocks, len(bdf), seed_offset)
    return blocks
