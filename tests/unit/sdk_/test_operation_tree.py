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


# ---------------------------------------------------------------------------
# The two mutants that SURVIVED the Phase 0-2 mutation run, M9 and M10, both in
# `substitute_at_path`. See finding F6 of
# docs/superpowers/reports/2026-09-15-nested-gpu-staging/
#   phase-012-implementation-review.md
# ---------------------------------------------------------------------------


def test_substitute_carries_the_slots_a_stage3_pipeline_reads():
    """M9: the rebuild must carry ``_provenance_pipeline`` and ``name``.

    THE OBVIOUS VERSION OF THIS TEST PASSES UNDER THE MUTANT. On a freshly
    constructed pipeline ``_provenance_pipeline`` is ``PrivateAttr(default=None)``
    (``_image_pipeline_core.py:231``), so asserting
    ``out._provenance_pipeline is pipe._provenance_pipeline`` compares
    ``None is None`` and succeeds whether the rebuild carries the slot or drops
    it -- the mutant substitutes ``None``, which is what it already was.

    So the slot is seeded with a distinguishable value first. A test that cannot
    fail is not coverage, and the obvious one could not.
    """
    pipe = _pipeline_with_composite()
    sentinel = {"name": "recorded-elsewhere"}
    pipe._provenance_pipeline = sentinel
    pipe.name = "a-distinctive-name"

    out = substitute_at_path(
        pipe, ("CompositeDetector", "ops[1]"), OtsuDetector(ignore_zeros=True)
    )

    assert out._provenance_pipeline is sentinel
    assert out.name == "a-distinctive-name"


def test_substitute_refuses_a_pipeline_core_it_cannot_rebuild():
    """M10: a non-``ImagePipeline`` ``ImagePipelineCore`` is refused BY NAME.

    ``iter_child_operations`` and ``_child`` both descend any
    ``ImagePipelineCore``, so a walk can hand a path THROUGH one; the rebuild is
    ``ImagePipeline``-specific (it names ``nrows``/``ncols``/``qc``/``plots``).
    Without the guard such a node falls through to the generic pydantic branch,
    where the segment is an ops KEY rather than an attribute, and the caller
    gets ``KeyError('<operation name>')`` -- blaming the operation instead of the
    unsupported container.

    Deliberately does NOT import ``NapariPipelineViewer``, the real second
    subclass. Tying this invariant to an optional extra is exactly why it went
    untested: it lives in the one gate shard that cannot run without the
    ``napari`` extra installed. ``ImagePipelineCore`` is a plain pydantic model
    with no abstract methods (``:143``), so a local subclass pins the type
    RELATION at no cost and without reintroducing that dependency.
    """
    from phenotypic._core._pipeline_parts._image_pipeline_core import (
        ImagePipelineCore,
    )

    class _NotAnImagePipeline(ImagePipelineCore):
        pass

    core = _NotAnImagePipeline(ops={"OtsuDetector": OtsuDetector()})

    with pytest.raises(TypeError, match="_NotAnImagePipeline"):
        substitute_at_path(core, ("OtsuDetector",), OtsuDetector(ignore_zeros=True))
