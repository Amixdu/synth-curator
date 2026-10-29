import numpy as np
import pandas as pd
import pytest
import torch

from synth_curator.steps.metrics import MetricContext, from_features

from .conftest import CLASSES


@pytest.fixture
def pool(tmp_path):
    """Fake pool + feature tensors, fed straight to the calculators - no images, no SynthCurator."""
    rng = np.random.default_rng(0)
    n = 20
    df = pd.DataFrame({
        "filename": [f"g{i % 4}/img{i}.png" for i in range(n)],
        "target_class_label": rng.choice(CLASSES, n, p=[.5, .3, .2]),
    })
    feats = tmp_path / "feat"
    feats.mkdir()
    real_names = [f"r{i}.png" for i in range(10)]
    for space in ("dino", "inception"):
        torch.save({f: torch.randn(16) for f in real_names}, feats / f"real_{space}.pt")
        torch.save({f: torch.randn(16) for f in df.filename}, feats / f"synthetic_{space}.pt")
    torch.save({f: torch.softmax(torch.randn(10), 0) for f in df.filename}, feats / "synthetic_inception_probs.pt")
    pd.DataFrame({"filename": real_names, "class": rng.choice(CLASSES, 10)}).to_csv(feats / "real_labels.csv", index=False)
    return {"df": df, "features": feats}


def _ctx(pool, tmp_path):
    return MetricContext(tmp_path, tmp_path, pool["features"], tmp_path / "m.csv")


def test_realism_and_diversity_match_bruteforce(pool, tmp_path):
    df = pool["df"][["filename", "target_class_label"]].copy()
    df = from_features.realism(df, _ctx(pool, tmp_path), k=3)
    df = from_features.diversity(df, _ctx(pool, tmp_path), k=3)

    real = torch.stack(list(torch.load(pool["features"] / "real_dino.pt").values()))
    synth_d = torch.load(pool["features"] / "synthetic_dino.pt")
    s = torch.stack(list(synth_d.values()))

    def n(x):
        return x / x.norm(dim=1, keepdim=True)
    d_real = 2 - 2 * (n(s) @ n(real).T)
    assert np.allclose(df["realism_score"].values, d_real.sort(dim=1).values[:, 2].numpy(), atol=1e-4)
    d_self = 2 - 2 * (n(s) @ n(s).T)
    assert np.allclose(df["diversity_score"].values, d_self.sort(dim=1).values[:, 3].numpy(), atol=1e-4)


def test_inception_score_and_fscs_fill_every_row(pool, tmp_path):
    df = pool["df"][["filename", "target_class_label"]].copy()
    df = from_features.inception_score(df, _ctx(pool, tmp_path))
    df = from_features.fscs(df, _ctx(pool, tmp_path))
    assert df["is_contribution_score"].notna().all() and (df["is_contribution_score"] >= -1e-6).all()
    assert df["fscs_score"].between(-1, 1).all()
