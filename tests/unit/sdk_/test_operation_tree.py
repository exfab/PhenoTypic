import pytest

from phenotypic import ImagePipeline
from phenotypic.detect import CompositeDetector, ManualPointDetector, OtsuDetector
from phenotypic.enhance import BlurGauss
from phenotypic.sdk_._operation_tree import (
    find_operations,
    get_at_path,
    substitute_at_path,
    walk_operations,
)

CENTERS = [[10.0, 10.0], [10.0, 40.0]]


def _pipeline_with_composite():
    return ImagePipeline(
        ops={
            "BlurGauss": BlurGauss(sigma=2.0),
            "CompositeDetector": CompositeDetector(
                ops=[
                    OtsuDetector(),
                    ManualPointDetector(centers=CENTERS, shape="disk", width=11),
                ],
                mode="overlap",
            ),
        }
    )


def test_walk_yields_list_entries_as_bracket_indexed_strings():
    pipe = _pipeline_with_composite()
    paths = {"/".join(p) for p, _ in walk_operations(pipe)}
    assert "BlurGauss" in paths
    assert "CompositeDetector" in paths
    assert "CompositeDetector/ops[0]" in paths
    assert "CompositeDetector/ops[1]" in paths


def test_every_path_segment_is_a_non_empty_string():
    """pipeline_step_path validation rejects integers and empty strings."""
    pipe = _pipeline_with_composite()
    for path, _ in walk_operations(pipe):
        assert path, "empty path"
        for segment in path:
            assert isinstance(segment, str) and segment


def test_find_operations_locates_a_nested_type():
    pipe = _pipeline_with_composite()
    hits = find_operations(pipe, lambda op: isinstance(op, ManualPointDetector))
    assert len(hits) == 1
    assert hits[0][0] == ("CompositeDetector", "ops[1]")


def test_get_at_path_round_trips_with_walk():
    pipe = _pipeline_with_composite()
    for path, op in walk_operations(pipe):
        assert get_at_path(pipe, path) is op


def test_substitute_replaces_only_the_addressed_node():
    pipe = _pipeline_with_composite()
    replacement = OtsuDetector(ignore_zeros=True)
    out = substitute_at_path(pipe, ("CompositeDetector", "ops[1]"), replacement)

    assert get_at_path(out, ("CompositeDetector", "ops[1]")) is replacement
    # sibling untouched, and the ORIGINAL pipeline is not mutated
    assert isinstance(get_at_path(out, ("CompositeDetector", "ops[0]")), OtsuDetector)
    assert isinstance(
        get_at_path(pipe, ("CompositeDetector", "ops[1]")), ManualPointDetector
    )


def test_substitute_at_depth_two():
    inner = CompositeDetector(ops=[OtsuDetector(), OtsuDetector()], mode="union")
    pipe = ImagePipeline(
        ops={
            "CompositeDetector": CompositeDetector(
                ops=[inner, OtsuDetector()], mode="overlap"
            )
        }
    )
    replacement = OtsuDetector(ignore_zeros=True)
    path = ("CompositeDetector", "ops[0]", "ops[0]")
    out = substitute_at_path(pipe, path, replacement)
    assert get_at_path(out, path) is replacement


def test_get_at_path_raises_on_unknown_path():
    pipe = _pipeline_with_composite()
    with pytest.raises(KeyError):
        get_at_path(pipe, ("CompositeDetector", "ops[9]"))
