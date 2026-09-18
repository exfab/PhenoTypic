"""Every ``pipeline_step_path`` in the journal must address something real.

Spec §5.3. The walker's ``gpu_path`` and the recorded ``pipeline_step_path``
are supposed to be one value, not two schemes kept aligned by hand -- but the
invariant that pins that is **resolvability**, not set equality with the
walker.

**Why not set equality.** It demands that every path the *walker* produces also
be *recorded*, and a container's own field is never recorded as itself. Measured
on the ``two-k`` shape below: the walker yields 8 paths, the journal records 14
entries over 6 distinct paths, and ``('TwoK', 'branch_base')`` /
``('TwoK', 'center_detector')`` appear only in the walker. No correct behaviour
makes those two sets equal. Set equality would also pass over composites while
``TwoKFilamentousDetector`` was recording ``['TwoK', 'InoculumDetector']`` -- a
well-formed path ``get_at_path`` cannot resolve at all, which is the defect this
file exists to catch.

(An earlier draft of the plan justified this with the ``None``-slot shape
instead, claiming set equality was "already false today" for it. Measured: it is
not. ``iter_child_operations`` skips the ``None`` but keeps the true index, so
both sides yield ``{('C',), ('C','ops[0]'), ('C','ops[2]')}``. That shape is
kept below as a **control** precisely because it is the case where the two
schemes do agree.)

**Resolvability is deliberately weaker than unique addressing.** An operation
that runs sub-operations without pushing segments -- ``InoculumDetector`` --
leaves its children recording the nearest addressable ancestor. That is an
honest partial address. An unresolvable one is a lie.

---

**What shape of input would make these tests pass on broken code, and why the
fixtures are not that shape.** Three ways, all closed:

1. *Nothing is recorded.* Every recorded path trivially resolves when there are
   none. Closed by ``assert recorded``.
2. *Every recorded path is a single top-level key.* A one-segment path always
   resolves, so plain resolvability is **vacuously true on pre-Task-4 code**,
   which recorded exactly that. Closed by ``min_depth``: each shape declares the
   depth its own nesting must produce, so a run that stopped descending fails in
   the premise rather than passing the invariant.
3. *``get_at_path`` resolves everything.* A degenerate resolver that never
   raised would make all six shapes green. Closed by
   ``test_get_at_path_rejects_a_fabricated_path``, which is the negative control
   for the instrument itself, run against the same pipeline objects.

---

**Before optimising the ``two-k`` case, read this.** It is the slowest test in
the file (~20s) and the only one that catches the defect the file was written
for, so it is the obvious thing to speed up and the worst thing to lose. Three
properties are jointly load-bearing and each can be removed while the test stays
green:

* **The plate is ``load_synth_filamentous_plate()``** -- the image this detector
  is for, and the one its own unit tests use. Measured: it runs in 19.5s and
  finds 60 objects, against 41.3s and 85 objects on the yeast plate. **Both
  work**, so the choice is runtime, not capability; moving it back is allowed,
  moving it to a fixture the detector cannot detect on is not.
* **``TwoKFilamentousDetector()`` takes no arguments.** Its own unit tests pass
  ``center_detector=OtsuDetector(...)``. Copying that here would delete the
  entire defect: the nested ``ImagePipeline`` default is what produced the
  unresolvable ``['TwoK', 'InoculumDetector']``.
* **It is not marked ``slow``.** That marker is excluded from PR runs, which
  would leave this file green and the invariant untested in the only lane that
  gates a merge.
"""

from __future__ import annotations

import pytest

from phenotypic import ImagePipeline
from phenotypic.data import load_synth_filamentous_plate, load_synth_yeast_plate
from phenotypic.detect import (
    CompositeDetector,
    ManualPointDetector,
    OtsuDetector,
    TwoKFilamentousDetector,
)
from phenotypic.enhance import ContrastStretching
from phenotypic.sdk_._operation_tree import get_at_path

CENTERS = [[150.0, 200.0], [300.0, 400.0]]


def _manual() -> ManualPointDetector:
    return ManualPointDetector(centers=CENTERS, shape="disk", width=41)


def _recorded_paths(image) -> list[tuple[str, tuple[str, ...]]]:
    """``(class name, path)`` for every journal entry that carries a path.

    Entries with no path are skipped rather than failed: a measurement's nested
    operation is a private probe and deliberately records none
    (``measure/CLAUDE.md``).
    """
    recorded = []
    for application in image._metadata.provenance_journal.get("applications", []):
        for operation in application.get("operations", []):
            path = operation.get("pipeline_step_path")
            if path:
                recorded.append(
                    (operation["operation_class"].rsplit(".", 1)[-1], tuple(path))
                )
    return recorded


# ---------------------------------------------------------------------------
# The shapes
# ---------------------------------------------------------------------------
#
# `min_depth` is the deepest path the shape's nesting MUST produce. It is the
# guard against vacuity (2) above, and it is measured, not guessed: a probe ran
# every shape and printed its recorded depths before these numbers were written.
#
# `plate` is "yeast" everywhere except two-k. `TwoKFilamentousDetector` is given
# the filamentous plate because that is the image its algorithm is for and the
# one its own unit tests use, and because it halves the runtime (19.5s against
# 41.3s on the yeast plate -- both work). It is deliberately NOT marked `slow`:
# `slow` is excluded from PR runs, and two-k is the single shape that exposed
# the defect this file guards, so hiding it from the default lane would leave
# the file green and the invariant untested.


def _flat() -> ImagePipeline:
    return ImagePipeline(ops={"OtsuDetector": OtsuDetector()})


def _composite() -> ImagePipeline:
    return ImagePipeline(
        ops={"C": CompositeDetector(ops=[OtsuDetector(), _manual()], mode="overlap")}
    )


def _nested_composite() -> ImagePipeline:
    return ImagePipeline(
        ops={
            "C": CompositeDetector(
                ops=[
                    CompositeDetector(ops=[OtsuDetector(), _manual()], mode="union"),
                    _manual(),
                ],
                mode="overlap",
            )
        }
    )


def _branch_pipeline() -> ImagePipeline:
    return ImagePipeline(
        ops={
            "C": CompositeDetector(
                ops=[
                    ImagePipeline(
                        ops={
                            "ContrastStretching": ContrastStretching(
                                input_layer="detect_mat"
                            ),
                            "OtsuDetector": OtsuDetector(),
                        }
                    ),
                    _manual(),
                ],
                mode="overlap",
            )
        }
    )


def _none_slot() -> ImagePipeline:
    """The control: the shape where set equality with the walker DOES hold.

    ``CompositeDetector`` enumerates its whole ``ops`` list including the
    unfilled slot and records ``ops[2]`` for its second real branch;
    ``iter_child_operations`` skips the ``None`` but keeps the true index, so
    both sides agree. A test that passes here and fails on ``two-k`` is
    measuring addressing, not tracking some unrelated difference.
    """
    return ImagePipeline(
        ops={
            "C": CompositeDetector(
                ops=[OtsuDetector(), None, _manual()], mode="overlap"
            )
        }
    )


def _two_k() -> ImagePipeline:
    """The shape that exposed the defect.

    ``center_detector`` defaults to an ``ImagePipeline``, so its children were
    recording ``['TwoK', 'InoculumDetector']`` -- the middle segment dropped --
    until ``_fill_centers`` was routed through ``apply_child``. Constructed with
    NO arguments on purpose: overriding ``center_detector`` (as the detector's
    own unit tests do, to a bare ``OtsuDetector``) removes the nested pipeline
    and with it the entire defect.
    """
    return ImagePipeline(ops={"TwoK": TwoKFilamentousDetector()})


SHAPES = [
    # name, factory, plate, min_depth
    ("flat", _flat, "yeast", 1),
    ("composite", _composite, "yeast", 2),
    ("none-slot", _none_slot, "yeast", 2),
    ("nested-composite", _nested_composite, "yeast", 3),
    ("branch-pipeline", _branch_pipeline, "yeast", 3),
    ("two-k", _two_k, "filamentous", 3),
]


def _load(plate: str):
    return (
        load_synth_yeast_plate() if plate == "yeast" else load_synth_filamentous_plate()
    )


@pytest.mark.parametrize(
    "name,factory,plate,min_depth", SHAPES, ids=[row[0] for row in SHAPES]
)
def test_every_recorded_step_path_resolves(name, factory, plate, min_depth):
    """Every ``pipeline_step_path`` in the journal must address something real.

    NOT set equality with the walker's paths -- see the module docstring. A path
    ``get_at_path`` cannot resolve is not an incomplete journal entry, it is a
    wrong one, and nothing downstream reports it.
    """
    pipeline = factory()
    image = _load(plate)
    pipeline.apply(image, inplace=True)

    recorded = _recorded_paths(image)
    assert recorded, f"{name}: nothing recorded -- the test proves nothing"

    deepest = max(len(path) for _, path in recorded)
    assert deepest >= min_depth, (
        f"{name}: deepest recorded path is {deepest} segments, expected at "
        f"least {min_depth}. The descent this shape exists to exercise did not "
        "happen, so 'every path resolves' below is vacuous -- a one-segment "
        "top-level key always resolves."
    )

    unresolvable = []
    for cls_name, path in recorded:
        try:
            get_at_path(pipeline, path)
        except KeyError as exc:
            unresolvable.append((cls_name, list(path), f"{type(exc).__name__}: {exc}"))

    assert not unresolvable, (
        f"{name}: {len(unresolvable)} recorded path(s) address nothing: "
        f"{unresolvable}"
    )


@pytest.mark.parametrize(
    "name,factory,plate,min_depth", SHAPES, ids=[row[0] for row in SHAPES]
)
def test_get_at_path_rejects_a_fabricated_path(name, factory, plate, min_depth):
    """The negative control for the instrument, not for the pipelines.

    ``test_every_recorded_step_path_resolves`` is an assertion that something
    does NOT raise. A ``get_at_path`` that resolved anything -- or one that had
    been changed to return ``None`` instead of raising -- would make all six
    shapes green while proving nothing at all. So show the resolver rejecting
    paths, on the same objects, before trusting it to accept.

    ``plate``/``min_depth`` are unused here; the parametrisation is shared so
    the control cannot silently cover fewer shapes than the claim.
    """
    pipeline = factory()
    top_level_key = next(iter(pipeline.get_ops()))

    with pytest.raises(KeyError):
        get_at_path(pipeline, (top_level_key, "no_such_field"))
    with pytest.raises(KeyError):
        get_at_path(pipeline, (top_level_key, "ops[99]"))
    with pytest.raises(KeyError):
        get_at_path(pipeline, ("no_such_top_level_key",))


def test_the_gpu_paths_segments_are_the_recorded_segments():
    """The walker's ``gpu_path`` IS the detector's recorded path.

    The narrower claim the staged engine actually depends on: Stage 3
    substitutes at ``gpu_path``, and the journal must say the detector ran
    there.

    **Would this pass on broken code?** It would, on a TOP-LEVEL detector: the
    recorded path and ``gpu_path`` are then both one-element lists holding the
    same key, and any scheme that addresses the detector at all agrees with any
    other. The fixture is a detector nested at ``('C', 'ops[0]')`` and the
    ``len(...) > 1`` assertion below refuses to let that degenerate back.
    """
    from tests._fakes.fake_gpu_detector import FakeGpuDetector

    from phenotypic._cli._cli_pipeline_split import split_pipeline_at_gpu

    detector = CompositeDetector(
        ops=[FakeGpuDetector(input_layer="detect_mat"), _manual()], mode="overlap"
    )
    pipeline = ImagePipeline(ops={"C": detector})
    plan = split_pipeline_at_gpu(pipeline)

    assert len(plan.gpu_path) > 1, (
        "fixture degenerated to a top-level detector; the assertion below "
        "would then hold under any addressing scheme"
    )

    image = load_synth_yeast_plate()
    pipeline.apply(image, inplace=True)

    recorded = [path for cls, path in _recorded_paths(image) if cls == "FakeGpuDetector"]
    assert recorded == [tuple(plan.gpu_path)]
