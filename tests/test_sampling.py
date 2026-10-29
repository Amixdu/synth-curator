import numpy as np
import pandas as pd

from synth_curator.core.sampling import class_counts, class_index, draw_blocks


def test_class_counts_sum_to_num_images_and_remainder_goes_to_largest():
    counts = class_counts({"A": 50, "B": 30, "C": 20}, 101)
    assert sum(counts.values()) == 101 and counts["A"] == 51


def test_class_counts_forces_a_minimum_of_one_even_at_zero_percent():
    counts = class_counts({"A": 100, "B": 0}, 4)
    assert counts["B"] == 1


def test_class_index_is_sorted_order_matching_the_isic_class_mapping():
    isic = {c: 0 for c in ["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]}
    assert class_index(isic) == {"AKIEC": 0, "BCC": 1, "BKL": 2, "DF": 3, "MEL": 4, "NV": 5, "VASC": 6}


def test_draw_uses_frozen_seed_formula():
    """Reproduce the draw by hand: seed = offset*100_000 + block*100 + class_idx."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"target_class_label": rng.choice(["A", "B"], 400), "S_Composite": rng.random(400)})
    target, T, offset = {"A": 60.0, "B": 40.0}, 0.5, 3
    (block,) = draw_blocks(df, target, 100, 1, T, offset)
    for cls, idx, n in (("A", 0, 60), ("B", 1, 40)):
        cdf = df[df.target_class_label == cls]
        s = cdf.S_Composite / T
        p = np.exp(s - s.max())
        expected = cdf.sample(n=n, replace=False, weights=p / p.sum(), random_state=offset * 100_000 + idx)
        assert set(expected.index) == set(block["df"][block["df"].target_class_label == cls].index)


def test_low_temperature_prefers_high_scores():
    df = pd.DataFrame({"target_class_label": ["A"] * 1000, "S_Composite": np.linspace(0, 1, 1000)})
    (block,) = draw_blocks(df, {"A": 100.0}, 100, 1, 0.02, 0)
    assert block["df"].S_Composite.mean() > 0.9


def test_draw_blocks_undersizes_rather_than_raising_when_pool_is_exhausted(caplog):
    df = pd.DataFrame({"target_class_label": ["A"] * 5, "S_Composite": [0.1] * 5})
    (block,) = draw_blocks(df, {"A": 100.0}, num_images=50, num_blocks=1, temperature=0.1, seed_offset=0)
    assert len(block["df"]) == 5              # capped at what the pool actually has, no exception
    assert "pool exhausted" in caplog.text
