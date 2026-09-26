"""steps/curate.py - resumability, error paths, the manifest-before-log write order, and the
group-scored / block_search integrations."""
import pandas as pd
import pytest
import torch

from synth_curator.exceptions import MissingMetricsError, SynthCuratorError
from synth_curator.steps import curate as curate_mod

TARGET = {"A": 50.0, "B": 50.0}
BASE_KW = dict(target_distribution=TARGET, num_images=4, temperature=0.1, group_by=[],
              block_search=False, num_blocks=1, safety_max_similarity_threshold=None, seed=0)


def _metadata(tmp_path, with_group=False):
    rows = []
    for grp in (["g1", "g2"] if with_group else ["g1"]):
        for cls in ("A", "B"):
            for i in range(4):
                row = {"filename": f"{grp}/{cls}/{i}.png", "target_class_label": cls, "psnr_score": float(i)}
                if with_group:
                    row["grp"] = grp
                rows.append(row)
    path = tmp_path / "metadata.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_curate_skips_entirely_when_manifest_already_exists(tmp_path):
    out_dir = tmp_path / "curated"
    out_dir.mkdir()
    manifest = out_dir / "seed_0.csv"
    manifest.write_text("source_path,target_class,safe_filename\nSENTINEL,A,x.png\n")

    metadata_csv = _metadata(tmp_path)
    missing_features_dir = tmp_path / "does_not_exist"     # would blow up if curate() did any real work
    result = curate_mod.curate(metadata_csv, missing_features_dir, {"A": {"psnr_score": 1.0}}, out_dir,
                               **{**BASE_KW, "safety_max_similarity_threshold": 0.9})
    assert result == manifest
    assert pd.read_csv(manifest)["source_path"].tolist() == ["SENTINEL"]   # untouched


def test_missing_metrics_error_for_weighted_column_not_in_metadata(tmp_path):
    metadata_csv = _metadata(tmp_path)
    with pytest.raises(MissingMetricsError, match="lpips_score"):
        curate_mod.curate(metadata_csv, tmp_path / "feat", {"A": {"lpips_score": 1.0}, "B": {"lpips_score": 1.0}},
                          tmp_path / "curated", **BASE_KW)


def test_synth_curator_error_when_metadata_lacks_required_columns(tmp_path):
    path = tmp_path / "bad.csv"
    # has the weighted metric column so it clears the MissingMetricsError check, but no target_class_label
    pd.DataFrame({"filename": ["a.png"], "psnr_score": [1.0]}).to_csv(path, index=False)
    with pytest.raises(SynthCuratorError, match="target_class_label"):
        curate_mod.curate(path, tmp_path / "feat", {"A": {"psnr_score": 1.0}}, tmp_path / "curated", **BASE_KW)


def test_manifest_is_written_before_the_block_log(tmp_path, monkeypatch):
    metadata_csv = _metadata(tmp_path)
    order = []
    real_manifest, real_block_log = curate_mod.write_manifest, curate_mod.write_block_log

    def spy_manifest(*a, **kw):
        order.append("manifest")
        return real_manifest(*a, **kw)

    def spy_block_log(*a, **kw):
        order.append("block_log")
        return real_block_log(*a, **kw)

    monkeypatch.setattr(curate_mod, "write_manifest", spy_manifest)
    monkeypatch.setattr(curate_mod, "write_block_log", spy_block_log)

    manifest = curate_mod.curate(metadata_csv, tmp_path / "feat", {"A": {"psnr_score": 1.0}, "B": {"psnr_score": 1.0}},
                                 tmp_path / "curated", **BASE_KW)
    assert order == ["manifest", "block_log"]

    block_log = manifest.parent / ".block_log_seed0.csv"
    assert manifest.stat().st_mtime_ns <= block_log.stat().st_mtime_ns


def test_curate_with_distributional_weight_uses_group_scores(tmp_path):
    metadata_csv = _metadata(tmp_path, with_group=True)
    features_dir = tmp_path / "feat"
    features_dir.mkdir()
    real = {"r0.png": torch.tensor([0.0, 0.0]), "r1.png": torch.tensor([0.0, 0.0])}
    synth = {}
    for grp, shift in (("g1", 0.0), ("g2", 500.0)):
        for cls in ("A", "B"):
            for i in range(4):
                synth[f"{grp}/{cls}/{i}.png"] = torch.tensor([shift, shift])
    torch.save(real, features_dir / "real_dino.pt")
    torch.save(synth, features_dir / "synthetic_dino.pt")

    out_dir = tmp_path / "curated"
    weights = {"A": {"psnr_score": 0.5, "kid_dino": 0.5}, "B": {"psnr_score": 0.5, "kid_dino": 0.5}}
    manifest = curate_mod.curate(metadata_csv, features_dir, weights, out_dir,
                                 **{**BASE_KW, "group_by": ["grp"]})
    df = pd.read_csv(manifest)
    assert df["target_class"].value_counts().to_dict() == {"A": 2, "B": 2}
    assert (out_dir.parent / ".group_scores.csv").exists()
    group_scores = pd.read_csv(out_dir.parent / ".group_scores.csv", index_col=0)
    assert group_scores.loc["g1", "kid_dino"] > group_scores.loc["g2", "kid_dino"]   # g1 is the close group


def test_curate_with_block_search_writes_class_log(tmp_path):
    metadata_csv = _metadata(tmp_path, with_group=False)
    features_dir = tmp_path / "feat"
    features_dir.mkdir()
    real = {"r0.png": torch.tensor([0.0, 0.0]), "r1.png": torch.tensor([1.0, 1.0])}
    synth = {}
    for cls in ("A", "B"):
        for i in range(4):
            synth[f"g1/{cls}/{i}.png"] = torch.tensor([float(i), float(i)])
    torch.save(real, features_dir / "real_dino.pt")
    torch.save(synth, features_dir / "synthetic_dino.pt")
    pd.DataFrame({"filename": ["r0.png", "r1.png"], "class": ["A", "B"]}).to_csv(
        features_dir / "real_labels.csv", index=False)

    out_dir = tmp_path / "curated"
    weights = {"A": {"psnr_score": 0.5, "kid_dino": 0.5}, "B": {"psnr_score": 0.5, "kid_dino": 0.5}}
    manifest = curate_mod.curate(metadata_csv, features_dir, weights, out_dir,
                                 **{**BASE_KW, "block_search": True, "num_blocks": 3,
                                    "min_samples": {"kid": 1, "fid": 1, "swd": 1}})
    assert pd.read_csv(manifest)["target_class"].value_counts().to_dict() == {"A": 2, "B": 2}
    assert (out_dir / ".class_log_seed0.csv").exists()
