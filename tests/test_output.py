"""core/output.py - manifest / block-log / weights-used writers."""
import json

import pandas as pd
import pytest

from synth_curator import __version__
from synth_curator.core.output import (
    write_block_log,
    write_manifest,
    write_weights_used,
)


def test_write_manifest_columns_and_image_dir_prefix(tmp_path):
    df = pd.DataFrame({"filename": ["A/x.png", "B/y.png"], "target_class_label": ["A", "B"]})
    out = write_manifest(df, tmp_path / "out" / "m.csv", image_dir=tmp_path / "pool")
    written = pd.read_csv(out)
    assert list(written.columns) == ["source_path", "target_class", "safe_filename"]
    assert written["source_path"].tolist() == [str(tmp_path / "pool" / "A/x.png"), str(tmp_path / "pool" / "B/y.png")]
    assert written["safe_filename"].tolist() == ["A_x.png", "B_y.png"]


def test_write_manifest_without_image_dir_uses_filename_verbatim(tmp_path):
    df = pd.DataFrame({"filename": ["A/x.png"], "target_class_label": ["A"]})
    out = write_manifest(df, tmp_path / "m.csv", image_dir=None)
    assert pd.read_csv(out)["source_path"].tolist() == ["A/x.png"]


def test_write_manifest_sanitizes_backslashes_too(tmp_path):
    df = pd.DataFrame({"filename": [r"A\x.png"], "target_class_label": ["A"]})
    out = write_manifest(df, tmp_path / "m.csv")
    assert pd.read_csv(out)["safe_filename"].tolist() == ["A_x.png"]


def _block(idx, filenames, classes, u_inst=0.5, u_dist=0.1, s_final=0.6, extra=None):
    return {
        "idx": idx,
        "df": pd.DataFrame({"filename": filenames, "Block_ID": idx, "target_class_label": classes,
                            "S_Composite": [0.5] * len(filenames)}),
        "U_inst": u_inst, "U_dist": u_dist, "S_Final": s_final,
        "distributional_scores": extra or {},
    }


def test_write_block_log_orders_core_columns_first_and_concatenates_blocks(tmp_path):
    b1 = _block(1, ["a.png"], ["A"], extra={"Raw_A_kid_dino": 1.23})
    b2 = _block(2, ["b.png"], ["B"])
    path = tmp_path / "log.csv"
    write_block_log([b1, b2], path)
    df = pd.read_csv(path)
    assert len(df) == 2
    assert list(df.columns[:4]) == ["filename", "Block_ID", "target_class_label", "S_Composite"]
    assert df.loc[df.filename == "a.png", "Raw_A_kid_dino"].iloc[0] == pytest.approx(1.23)
    assert pd.isna(df.loc[df.filename == "b.png", "Raw_A_kid_dino"]).all()


def test_write_weights_used_contents(tmp_path):
    path = tmp_path / "weights_used.json"
    write_weights_used(path, {"A": {"psnr_score": 1.0}}, {"seed": 3})
    payload = json.loads(path.read_text())
    assert payload["algorithm"] == "v15"
    assert payload["library_version"] == __version__
    assert payload["params"] == {"seed": 3}
    assert payload["weights"] == {"A": {"psnr_score": 1.0}}
