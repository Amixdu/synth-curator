"""steps/metric_sets.py - one curated dataset per metric (+ random control)."""
import pandas as pd
import pytest
import torch

from synth_curator.steps import metric_sets as ms


def _build(tmp_path, with_group_col=True):
    features_dir = tmp_path / "feat"
    features_dir.mkdir()
    real = {"r0.png": torch.tensor([0.0, 0.0]), "r1.png": torch.tensor([0.0, 0.0])}
    synth = {}
    rows = []
    for grp, shift in (("close", 0.0), ("far", 500.0)):
        for cls in ("A", "B"):
            for i in range(3):
                fname = f"{grp}/{cls}/{i}.png"
                synth[fname] = torch.tensor([shift, shift])
                row = {"filename": fname, "target_class_label": cls, "lpips_score": float(i)}
                if with_group_col:
                    row["grp"] = grp
                rows.append(row)
    torch.save(real, features_dir / "real_dino.pt")
    torch.save(synth, features_dir / "synthetic_dino.pt")
    metadata_csv = tmp_path / "metadata.csv"
    pd.DataFrame(rows).to_csv(metadata_csv, index=False)
    return metadata_csv, features_dir


def test_distributional_metric_set_picks_the_lowest_raw_group(tmp_path):
    metadata_csv, features_dir = _build(tmp_path)
    out = ms.curate_metric_sets(metadata_csv, features_dir, tmp_path / "sets", ["kid_dino"],
                                {"A": 50, "B": 50}, block_size=4, group_by=["grp"], seeds=[0])
    manifest = pd.read_csv(out["kid_dino"][0])
    assert all("close" in p for p in manifest["source_path"])   # "close" group has ~0 raw kid distance


def test_distributional_metric_set_skipped_without_group_by_columns(tmp_path):
    metadata_csv, features_dir = _build(tmp_path, with_group_col=False)
    out = ms.curate_metric_sets(metadata_csv, features_dir, tmp_path / "sets", ["kid_dino"],
                                {"A": 50, "B": 50}, block_size=4, group_by=["grp"], seeds=[0])
    assert "kid_dino" not in out
    assert ms.RANDOM_METRIC_SET in out                            # random control is unaffected


def test_instance_metric_set_skipped_when_column_absent(tmp_path):
    metadata_csv, features_dir = _build(tmp_path)
    out = ms.curate_metric_sets(metadata_csv, features_dir, tmp_path / "sets", ["psnr_score"],
                                {"A": 50, "B": 50}, block_size=4, group_by=["grp"], seeds=[0])
    assert "psnr_score" not in out


def test_instance_metric_set_is_top_n_per_class(tmp_path):
    metadata_csv, features_dir = _build(tmp_path)
    df = pd.read_csv(metadata_csv)
    out = ms.curate_metric_sets(metadata_csv, features_dir, tmp_path / "sets", ["lpips_score"],
                                {"A": 100, "B": 0}, block_size=2, group_by=["grp"], seeds=[0])
    manifest = pd.read_csv(out["lpips_score"][0])
    counts = manifest["target_class"].value_counts().to_dict()
    assert counts == {"A": 2, "B": 1}          # class_counts: B forced to >= 1 even at 0%

    # lpips_score optimizes "min": every picked row must be among the N lowest lpips_score values
    # in its class (there are ties at i=0 across groups, so compare the *set* of achieved values).
    for cls, n in counts.items():
        picked_ids = set(manifest.loc[manifest.target_class == cls, "safe_filename"])
        pool = df[df.target_class_label == cls].sort_values("lpips_score")
        pool_safe = pool["filename"].str.replace("/", "_")
        assert picked_ids == set(pool_safe.head(n))


def test_random_control_draws_a_fresh_set_per_seed(tmp_path):
    metadata_csv, features_dir = _build(tmp_path)
    out = ms.curate_metric_sets(metadata_csv, features_dir, tmp_path / "sets", [],
                                {"A": 50, "B": 50}, block_size=4, group_by=["grp"], seeds=[0, 1])
    m0 = pd.read_csv(out[ms.RANDOM_METRIC_SET][0])
    m1 = pd.read_csv(out[ms.RANDOM_METRIC_SET][1])
    assert out[ms.RANDOM_METRIC_SET][0] != out[ms.RANDOM_METRIC_SET][1]
    assert len(m0) == 4 and len(m1) == 4


def test_curate_metric_sets_is_resumable(tmp_path, monkeypatch):
    metadata_csv, features_dir = _build(tmp_path)
    calls = []
    real_write = ms.write_manifest

    def spy(selected, path, image_dir=None):
        calls.append(path)
        return real_write(selected, path, image_dir)

    monkeypatch.setattr(ms, "write_manifest", spy)
    kw = dict(target_distribution={"A": 50, "B": 50}, block_size=4, group_by=["grp"], seeds=[0])
    ms.curate_metric_sets(metadata_csv, features_dir, tmp_path / "sets", ["lpips_score", "kid_dino"], **kw)
    first = len(calls)
    ms.curate_metric_sets(metadata_csv, features_dir, tmp_path / "sets", ["lpips_score", "kid_dino"], **kw)
    assert len(calls) == first          # second call wrote nothing new: every path already existed
