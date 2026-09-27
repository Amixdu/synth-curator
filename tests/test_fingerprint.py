"""core/fingerprint.py - "what inputs built this cached output" markers."""
from synth_curator.core import fingerprint


def test_no_fingerprint_file_means_not_exists_and_never_matches(tmp_path):
    out = tmp_path / "some_output"
    assert not fingerprint.exists(out)
    assert not fingerprint.matches(out, {"a": 1})


def test_write_then_matches_the_same_dict(tmp_path):
    out = tmp_path / "some_output"
    current = {"real_dir": "/a/b", "resolution": 512}
    fingerprint.write(out, current)
    assert fingerprint.exists(out)
    assert fingerprint.matches(out, current)


def test_matches_is_false_for_a_different_dict(tmp_path):
    out = tmp_path / "some_output"
    fingerprint.write(out, {"real_dir": "/a/b", "resolution": 512})
    assert not fingerprint.matches(out, {"real_dir": "/a/b", "resolution": 256})
    assert not fingerprint.matches(out, {"real_dir": "/different", "resolution": 512})


def test_corrupted_fingerprint_file_is_treated_as_no_match_not_a_crash(tmp_path):
    out = tmp_path / "some_output"
    path = tmp_path / "some_output.fingerprint.json"
    path.write_text("not valid json {{{")
    assert fingerprint.exists(out)               # the file is there...
    assert not fingerprint.matches(out, {"a": 1})  # ...but never trusted if unreadable


def test_fingerprint_file_lives_next_to_the_output_not_inside_it(tmp_path):
    out = tmp_path / "some_output"
    fingerprint.write(out, {"a": 1})
    assert not out.exists()   # write() never creates the output dir itself
    assert (tmp_path / "some_output.fingerprint.json").exists()
