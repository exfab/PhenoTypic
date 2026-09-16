"""Container operations push a per-branch ``pipeline_step`` as they descend.

Before this change every child of a ``CompositeDetector`` inherited the
container's own path, so N children were indistinguishable in the journal
(spec 5.1). ``apply_child`` fixes that, and makes the walker's path and the
recorded ``pipeline_step_path`` the same value (spec 5.3).
"""

import pytest

from phenotypic import ImagePipeline
from phenotypic._core._provenance import _pipeline_step_path, apply_child
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import CompositeDetector, ManualPointDetector, OtsuDetector
from phenotypic.enhance import BlurGauss, CompositeEnhance, MedianFilter
from phenotypic.measure import MeasureSymZones
from phenotypic.sdk_._operation_tree import walk_operations

CENTERS = [[150.0, 200.0], [300.0, 400.0]]


def _step_paths(image):
    journal = image._metadata.provenance_journal
    return [
        (op["operation_class"].rsplit(".", 1)[-1], op.get("pipeline_step_path"))
        for app in journal.get("applications", [])
        for op in app.get("operations", [])
    ]


def _manual():
    return ManualPointDetector(centers=CENTERS, shape="disk", width=41)


def test_composite_children_record_their_branch_index():
    image = load_synth_yeast_plate()
    ImagePipeline(
        ops={
            "CompositeDetector": CompositeDetector(
                ops=[OtsuDetector(), _manual()], mode="union"
            )
        }
    ).apply(image, inplace=True)

    recorded = dict(_step_paths(image))
    assert recorded["OtsuDetector"] == ["CompositeDetector", "ops[0]"]
    assert recorded["ManualPointDetector"] == ["CompositeDetector", "ops[1]"]


def test_nested_composites_produce_distinct_paths():
    """Before this change all five entries shared ``['CompositeDetector']``."""
    image = load_synth_yeast_plate()
    inner = CompositeDetector(ops=[OtsuDetector(), _manual()], mode="union")
    ImagePipeline(
        ops={
            "CompositeDetector": CompositeDetector(
                ops=[inner, _manual()], mode="overlap"
            )
        }
    ).apply(image, inplace=True)

    paths = [tuple(p) for _, p in _step_paths(image)]
    assert len(paths) == len(set(paths)), f"duplicate step paths: {paths}"


def test_a_measurement_probe_records_no_step_path():
    """A nested op run by a MEASUREMENT is a private probe (measure/CLAUDE.md).

    This exclusion is deliberate. Without this test it is indistinguishable
    from an oversight and will be 'completed' by a later reader.
    """
    image = load_synth_yeast_plate()
    pipe = ImagePipeline(
        ops={"OtsuDetector": OtsuDetector()},
        meas={"MeasureSymZones": MeasureSymZones(center_detector=_manual())},
    )
    pipe.apply(image, inplace=True)
    pipe.measure(image, apply_post=False)

    classes = [cls for cls, _ in _step_paths(image)]
    assert "ManualPointDetector" not in classes, (
        "a measurement's center_detector must not enter the plate journal"
    )


def test_an_empty_slot_does_not_shift_its_siblings_branch_index():
    """The segment is the slot's position in ``ops``, not its position among
    the non-``None`` entries. ``iter_child_operations`` enumerates the whole
    list and skips the ``None`` slots, so the two schemes agree only if the
    container does the same.
    """
    image = load_synth_yeast_plate()
    ImagePipeline(
        ops={
            "CompositeDetector": CompositeDetector(
                ops=[None, OtsuDetector(), _manual()], mode="union"
            )
        }
    ).apply(image, inplace=True)

    recorded = dict(_step_paths(image))
    assert recorded["OtsuDetector"] == ["CompositeDetector", "ops[1]"]
    assert recorded["ManualPointDetector"] == ["CompositeDetector", "ops[2]"]


def test_the_recorded_paths_are_the_walker_paths():
    """The local form of the Task-9 invariant: a journal step path is a path
    the walker also produces for the same pipeline.
    """
    inner = CompositeDetector(ops=[OtsuDetector(), _manual()], mode="union")
    pipeline = ImagePipeline(
        ops={
            "CompositeDetector": CompositeDetector(
                ops=[inner, _manual()], mode="overlap"
            )
        }
    )
    image = load_synth_yeast_plate()
    pipeline.apply(image, inplace=True)

    walker_paths = {path for path, _ in walk_operations(pipeline)}
    recorded_paths = {tuple(p) for _, p in _step_paths(image) if p}
    assert recorded_paths == walker_paths


def test_composite_enhance_children_record_their_branch_index():
    image = load_synth_yeast_plate()
    ImagePipeline(
        ops={
            "CompositeEnhance": CompositeEnhance(
                ops=[BlurGauss(), MedianFilter()], mode="max"
            )
        }
    ).apply(image, inplace=True)

    recorded = dict(_step_paths(image))
    assert recorded["BlurGauss"] == ["CompositeEnhance", "ops[0]"]
    assert recorded["MedianFilter"] == ["CompositeEnhance", "ops[1]"]


# --- apply_child's own contract ------------------------------------------
#
# These pin the two ways the helper can silently throw a container's work
# away: the wrong `inplace`, and a nested pipeline that resets the image its
# parent just enhanced.


@pytest.fixture
def _recorded_pipeline_apply(monkeypatch):
    """Replace ``ImagePipeline.apply`` with a recorder of its call."""
    recorded: dict = {}

    def _record(self, image, inplace=False, reset=None):
        recorded["inplace"] = inplace
        recorded["reset"] = reset
        recorded["step_path"] = _pipeline_step_path.get()
        return image

    monkeypatch.setattr(ImagePipeline, "apply", _record)
    return recorded


def test_apply_child_sends_reset_false_to_a_pipeline_child(_recorded_pipeline_apply):
    """``reset=None`` means ``reset=False``, NOT "the pipeline's own setting".

    ``ImagePipeline.apply`` resolves ``reset=None`` to the pipeline's ``reset``
    field, which defaults to ``False`` -- so for a default-constructed branch
    pipeline this is not a behaviour change. It only bites a pipeline built
    with ``reset=True``, which is the case this pins: a child must not reset
    the image its parent handed it, exactly as ``_run_operations`` forces for
    a nested pipeline.
    """
    pipeline = ImagePipeline(ops={"OtsuDetector": OtsuDetector()}, reset=True)
    apply_child(
        pipeline, load_synth_yeast_plate(), segment="branch_base", inplace=True
    )

    assert _recorded_pipeline_apply == {
        "inplace": True,
        "reset": False,
        "step_path": ("branch_base",),
    }


def test_apply_child_forwards_an_explicit_reset(_recorded_pipeline_apply):
    pipeline = ImagePipeline(ops={"OtsuDetector": OtsuDetector()})
    apply_child(
        pipeline, load_synth_yeast_plate(), segment="ops[3]", reset=True
    )

    assert _recorded_pipeline_apply == {
        "inplace": False,
        "reset": True,
        "step_path": ("ops[3]",),
    }


def test_apply_child_restores_the_step_path_after_the_child_raises():
    # An empty `ops` list is the cheapest real failure inside a child's
    # `_operate`; `_default_ops` only substitutes for an explicit None.
    before = _pipeline_step_path.get()
    with pytest.raises((ValueError, RuntimeError)):
        apply_child(
            CompositeDetector(ops=[]), load_synth_yeast_plate(), segment="ops[0]"
        )
    assert _pipeline_step_path.get() == before
