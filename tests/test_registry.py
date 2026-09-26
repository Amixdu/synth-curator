import pytest

from synth_curator.core.registry import (
    DEFAULT_METRICS, METRIC_PROPERTIES, is_distributional, is_instance, props,
)
from synth_curator.exceptions import SynthCuratorError


def test_every_metric_has_a_valid_type_and_optimize_direction():
    for name, p in METRIC_PROPERTIES.items():
        assert p["type"] in ("instance", "distributional"), name
        assert p["optimize"] in ("min", "max"), name


def test_distributional_metrics_carry_family_and_space():
    for name, p in METRIC_PROPERTIES.items():
        if p["type"] == "distributional":
            assert p["family"] in ("fid", "kid", "swd"), name
            assert p["space"] in ("dino", "inception"), name
            assert p["optimize"] == "min"        # all distributional metrics: lower raw value is better


def test_instance_metrics_have_no_family_or_space():
    for name, p in METRIC_PROPERTIES.items():
        if p["type"] == "instance":
            assert "family" not in p and "space" not in p, name


def test_is_instance_and_is_distributional_partition_the_registry():
    for name in METRIC_PROPERTIES:
        assert is_instance(name) != is_distributional(name)


def test_props_raises_synth_curator_error_for_unknown_metric():
    with pytest.raises(SynthCuratorError, match="unknown metric"):
        props("not_a_real_metric")


def test_default_metrics_are_all_registered():
    for m in DEFAULT_METRICS:
        props(m)   # must not raise
    assert len(DEFAULT_METRICS) == len(set(DEFAULT_METRICS))
