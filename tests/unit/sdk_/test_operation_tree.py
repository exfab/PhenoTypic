from typing import Any

import pytest

from phenotypic import ImagePipeline
from phenotypic.analysis import (
    LinearLagModel,
    LogGrowthModel,
    TukeyOutlierRemover,
)
from phenotypic.detect import CompositeDetector, ManualPointDetector, OtsuDetector
from phenotypic.enhance import BlurGauss
from phenotypic.measure import MeasureShape, MeasureSymZones
from phenotypic.post import AppendString
from phenotypic.sdk_._operation_tree import (
    _is_operation,
    find_operations,
    get_at_path,
    pipeline_slot_of,
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


def _slots():
    return dict(
        meas={
            "MeasureShape": MeasureShape(),
            "MeasureSymZones": MeasureSymZones(center_detector=OtsuDetector()),
        },
        post={"AppendString": AppendString(column="Temperature", value="C")},
        filters={
            "Tukey": TukeyOutlierRemover(on="Size_Area", groupby=["Metadata_Plate"])
        },
        model=LogGrowthModel(on="Size_Area", groupby=["Metadata_Plate"]),
    )


def _pipeline_with_slots_at_two_depths():
    inner = ImagePipeline(ops={"OtsuDetector": OtsuDetector()}, **_slots())
    return ImagePipeline(
        ops={"BlurGauss": BlurGauss(sigma=2.0), "inner": inner}, **_slots()
    )


@pytest.mark.parametrize(
    "factory", [_pipeline_with_composite, _pipeline_with_slots_at_two_depths]
)
def test_get_at_path_round_trips_with_walk(factory):
    pipe = factory()
    walked = list(walk_operations(pipe))
    assert walked  # premise: a round trip over nothing proves nothing
    for path, op in walked:
        assert get_at_path(pipe, path) is op


def test_walk_descends_every_slot_of_the_root_and_of_a_nested_pipeline():
    pipe = _pipeline_with_slots_at_two_depths()
    paths = {"/".join(p) for p, _ in walk_operations(pipe)}
    for prefix in ("", "inner/"):
        assert f"{prefix}meas:MeasureShape" in paths
        assert f"{prefix}meas:MeasureSymZones" in paths
        assert f"{prefix}meas:MeasureSymZones/center_detector" in paths
        assert f"{prefix}post:AppendString" in paths
        assert f"{prefix}filters:Tukey" in paths
        assert f"{prefix}model:LogGrowthModel" in paths


def test_slot_children_are_yielded_whatever_their_type():
    """``_is_operation`` rejects these three types; the walker must not care.

    Gating slot children on ``_is_operation`` would drop the ``model`` entry,
    and a GpuDetector under it would stop being found.
    """
    pipe = _pipeline_with_slots_at_two_depths()
    slots = _slots()
    rejected = [slots["post"]["AppendString"], slots["filters"]["Tukey"], slots["model"]]
    assert not any(_is_operation(entry) for entry in rejected)  # premise

    types = {type(op) for _, op in walk_operations(pipe)}
    assert {AppendString, TukeyOutlierRemover, LogGrowthModel} <= types


def test_a_model_segment_stops_resolving_once_the_model_is_replaced():
    pipe = ImagePipeline(**_slots())
    path = ("model:LogGrowthModel",)
    assert isinstance(get_at_path(pipe, path), LogGrowthModel)

    pipe.set_model(LinearLagModel(on="Size_Area", groupby=["Metadata_Plate"]))
    with pytest.raises(KeyError):
        get_at_path(pipe, path)


@pytest.mark.parametrize(
    "segment",
    ["meas:NoSuchMeasure", "post:NoSuch", "filters:NoSuch", "model:NoSuchModel",
     "notaslot:MeasureShape", "meas"],
)
def test_an_unresolvable_slot_segment_raises_key_error(segment):
    pipe = ImagePipeline(**_slots())
    with pytest.raises(KeyError):
        get_at_path(pipe, (segment,))


def test_a_segment_that_is_both_an_ops_key_and_a_slot_entry_is_refused():
    pipe = ImagePipeline(
        ops={"meas:MeasureShape": OtsuDetector()},
        meas={"MeasureShape": MeasureShape()},
    )
    with pytest.raises(KeyError, match="ambiguous"):
        get_at_path(pipe, ("meas:MeasureShape",))


def test_pipeline_slot_of_is_decided_on_the_live_node():
    pipe = ImagePipeline(
        ops={"meas:Otsu": OtsuDetector(), "inner": ImagePipeline(**_slots())},
        **_slots(),
    )
    assert pipeline_slot_of(pipe, "meas:MeasureShape") == "meas"
    assert pipeline_slot_of(pipe, "model:LogGrowthModel") == "model"
    assert pipeline_slot_of(pipe, "meas:Otsu") is None  # an ops key
    assert pipeline_slot_of(pipe, "inner") is None
    assert pipeline_slot_of(OtsuDetector(), "meas:MeasureShape") is None
    with pytest.raises(KeyError):
        pipeline_slot_of(pipe, "meas:Absent")


def test_a_bare_pipeline_core_in_an_operation_field_is_walked():
    """``_is_operation`` admits any ``ImagePipelineCore``, not only
    ``ImagePipeline``, so a recorded path through one is also a walked path.
    """
    from phenotypic._core._pipeline_parts._image_pipeline_core import (
        ImagePipelineCore,
    )

    class _NotAnImagePipeline(ImagePipelineCore):
        pass

    class _Holder(OtsuDetector):
        inner: Any = None

    core = _NotAnImagePipeline(ops={"OtsuDetector": OtsuDetector()})
    assert _is_operation(core)
    holder = _Holder(inner=core)
    paths = {p for p, _ in walk_operations(holder)}
    assert ("inner",) in paths
    assert ("inner", "OtsuDetector") in paths


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

    Deliberately does NOT import ``NapariPipelineViewer``, the real one.
    Tying this invariant to an optional extra is exactly why it went
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


@pytest.mark.parametrize(
    "path, slot",
    [
        (("meas:MeasureSymZones", "center_detector"), "meas"),
        (("meas:MeasureShape",), "meas"),
        (("post:AppendString",), "post"),
        (("filters:Tukey",), "filters"),
        (("model:LogGrowthModel",), "model"),
    ],
)
def test_substitute_refuses_a_slot_path_by_name(path, slot):
    """A slot segment gets a named refusal, not a bare ``KeyError``.

    Premise first: each path must really RESOLVE, or the test would pass for the
    wrong reason -- a typo'd segment raises ``KeyError`` whether or not the guard
    exists, and ``pytest.raises(TypeError)`` would then fail, not pass, but a
    reader could not tell which it was guarding. Resolving proves the segment
    names a live slot entry, so the only thing standing between it and a bare
    ``KeyError`` is the guard.
    """
    pipe = ImagePipeline(ops={"OtsuDetector": OtsuDetector()}, **_slots())
    get_at_path(pipe, path)  # premise: the path is real

    with pytest.raises(TypeError, match=rf"does not descend a pipeline's '{slot}'"):
        substitute_at_path(pipe, path, OtsuDetector(ignore_zeros=True))


def test_substitute_refuses_an_ambiguous_segment_as_get_at_path_does():
    """An ops key that is also a slot entry is refused, not silently picked.

    A user may key an op ``"meas:MeasureShape"`` while the ``meas`` slot also
    holds ``MeasureShape``. ``get_at_path`` refuses that segment as ambiguous.
    ``substitute_at_path`` used to check ``head in ops`` first, so it silently
    substituted into the ops entry -- the two functions disagreed about the same
    path, and a substitution driven by a walked path could land on the node the
    walker did not mean.
    """
    pipe = ImagePipeline(
        ops={"meas:MeasureShape": OtsuDetector()},
        meas={"MeasureShape": MeasureShape()},
    )
    with pytest.raises(KeyError, match="ambiguous"):
        get_at_path(pipe, ("meas:MeasureShape",))  # premise: really ambiguous

    with pytest.raises(KeyError, match="ambiguous"):
        substitute_at_path(
            pipe, ("meas:MeasureShape",), OtsuDetector(ignore_zeros=True)
        )


def test_substitute_still_replaces_an_ordinary_ops_entry_beside_filled_slots():
    """Control: the new classification must not break the common case.

    Without this, a guard that refused EVERY head on a pipeline with slots would
    pass both tests above.
    """
    pipe = ImagePipeline(ops={"OtsuDetector": OtsuDetector()}, **_slots())
    replacement = OtsuDetector(ignore_zeros=True)
    out = substitute_at_path(pipe, ("OtsuDetector",), replacement)
    assert get_at_path(out, ("OtsuDetector",)) is replacement
