"""A staged run of a NESTED GpuDetector equals a single-pass run of the same
pipeline -- in pixels, in measurements, and in the journal.

Spec §5.3 closes with a claim nothing else tests:

    "Staged/single-pass journal parity is unaffected: substitution happens at
    the same path, so both runs record the same step path, and the stub's
    identity delegation supplies the same class and parameters."

---

**What shape of input would make these tests pass on broken code?** Four ways,
each of which has actually happened on this change, and each closed below.

1. *Compare only the measurements.* A one-element ``pipeline_step_path`` and a
   three-element one produce identical measurements, so a measurements-only
   check certifies the parity above while the parity is broken. Closed by
   asserting the recorded path.

2. *Assert the path on a TOP-LEVEL detector.* ``[plan.gpu_key]`` and
   ``list(plan.gpu_path)`` are then both one-element lists holding the same key,
   so the corrected assertion passes on the broken code too -- it certifies
   exactly what the measurements-only version did, one level further down. Every
   fixture here is nested, and ``_assert_fixture_is_nested`` refuses a plan
   whose ``gpu_path`` has collapsed to one segment.

3. *Compare the objmap against a plate that already has one.*
   ``load_synth_yeast_plate()`` arrives with a populated ground-truth objmap of
   ~96 objects, and a run that wrote **nothing** would leave both sides holding
   that same ground truth. Closed by ``_assert_the_run_changed_the_objmap``,
   which fails if the reference objmap still equals the pristine plate's, and --
   for the store-driven tests -- by building the reference from
   ``Image.imread(image_path)``, the same bytes Stage 1 read, which carries no
   objmap at all.

4. *Assert the staged and single-pass results are equal without ever showing the
   comparison can fail.* Closed by ``test_a_corrupted_replay_is_detected``.

**Two of these tests are the only thing that can see Task 8.** Stage 3 now
substitutes a ``ReplayDetector`` at ``gpu_path`` inside ``post_pipeline``
instead of writing the objmap by hand. For a top-level detector those two
behaviours are observationally identical, and every fixture in the four existing
staged integration files is top-level -- so the assertions that discriminate are
the ones that drive ``stage2_detect_core`` + ``stage3_merge_measure_core`` over
a **nested** shape: ``test_the_staged_store_matches_a_single_pass_run`` and
``test_stage3_records_the_branchs_own_ops_under_the_enclosing_operation``.

Note what the in-memory tests can and cannot see: they call
``substitute_at_path`` themselves, so their path-parity assertion is correct but
**cannot fail on a broken Stage 3** -- it is testing the algorithm, not the
wiring. It is kept because it localises a failure (algorithm vs. wiring), not
because it proves Task 8.

**Shape D is the reset-guard fixture.** ``_apply_stage2_prefix`` sends
``reset=False`` to any ``ImagePipelineCore`` in the prefix; without that guard a
nested pipeline built with ``reset=True`` calls ``image.reset()`` on the probe
and discards Stage 1's preprocessing before the detector reads it. No existing
fixture puts a *pipeline* in the prefix -- every prefix assertion in the suite
expects a leaf op -- so that mutation survives. Shape D closes it, and shape B
is retained as the control that proves the fixture is load-bearing: with a leaf
prefix the guarded and unguarded paths are indistinguishable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
from pandas.testing import assert_frame_equal

from phenotypic import Image, ImagePipeline
from phenotypic._cli._cli_output_manager import OutputManager
from phenotypic._cli._cli_pipeline_split import StagePlan, split_pipeline_at_gpu
from phenotypic._cli._cli_replay_detector import ReplayDetector
from phenotypic._cli._cli_stage2_token import detector_slot
from phenotypic._cli._cli_staged_workers import (
    _apply_stage2_prefix,
    stage1_preprocess_core,
    stage2_detect_core,
    stage3_merge_measure_core,
)
from phenotypic._cli._cli_types import Dataset
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import CompositeDetector, ManualPointDetector
from phenotypic.enhance import BlurGauss, ContrastStretching, SubtractGaussian
from phenotypic.measure import MeasureShape, MeasureSize
from phenotypic.refine import SmallObjectRemover
from phenotypic.schema import EXPERIMENT
from phenotypic.sdk_ import MEASUREMENT_TABLE_RELATIVE_PATH, zarr_store_path
from phenotypic.sdk_._operation_tree import substitute_at_path
from tests._fakes.fake_gpu_detector import FakeGpuDetector

#: The spike's centres (``spike_nested_gpu.py:28``), a 3x3 grid of CPU-branch
#: detections that overlap the fake detector's threshold mask.
CENTERS = [
    [150.0, 200.0],
    [150.0, 400.0],
    [150.0, 600.0],
    [300.0, 200.0],
    [300.0, 400.0],
    [300.0, 600.0],
    [450.0, 200.0],
    [450.0, 400.0],
    [450.0, 600.0],
]


def _gpu() -> FakeGpuDetector:
    """``input_layer="detect_mat"`` is load-bearing.

    ``GpuDetector`` defaults to ``"rgb"``, which no CPU prefix touches -- a
    prefixed and an unprefixed Stage 2 would then read the same array and shape
    B / shape D would prove nothing about the prefix at all.
    """
    return FakeGpuDetector(input_layer="detect_mat", threshold=0.5)


def _manual() -> ManualPointDetector:
    return ManualPointDetector(centers=CENTERS, shape="disk", width=61)


# ---------------------------------------------------------------------------
# The shapes. A/B/C are the spike's (spike_nested_gpu.py:171,175,184); D is new.
# ---------------------------------------------------------------------------


def shape_a() -> CompositeDetector:
    """GPU leaf directly in ``CompositeDetector.ops``. Prefix: empty."""
    return CompositeDetector(ops=[_gpu(), _manual()], mode="overlap")


def shape_b() -> CompositeDetector:
    """GPU behind a CPU op inside a nested pipeline branch. Prefix: one leaf."""
    branch = ImagePipeline(
        ops={
            "ContrastStretching": ContrastStretching(
                lower_percentile=2, upper_percentile=98, input_layer="detect_mat"
            ),
            "FakeGpuDetector": _gpu(),
        }
    )
    return CompositeDetector(ops=[branch, _manual()], mode="overlap")


def shape_c() -> CompositeDetector:
    """GPU inside a composite inside a composite (depth 2). Prefix: empty."""
    inner = CompositeDetector(ops=[_gpu(), _manual()], mode="union")
    return CompositeDetector(ops=[inner, _manual()], mode="overlap")


def shape_d() -> CompositeDetector:
    """GPU behind a nested PIPELINE inside the branch. Prefix: one pipeline.

    ``_branch_prefix`` yields the raw ``container.get_ops()`` values, so the
    ``"pre"`` entry arrives in ``stage2_prefix`` as an ``ImagePipeline`` object
    rather than as its contents -- which is the only way to reach
    ``_apply_stage2_prefix``'s ``isinstance(operation, ImagePipelineCore)``
    guard. ``reset=True`` is what makes the guard observable: with the guard the
    probe keeps Stage 1's preprocessing, without it ``image.reset()`` throws it
    away and the detector infers on different pixels.

    ``reset=True`` changes nothing about the single-pass run --
    ``_run_operations`` (``_image_pipeline_core.py:886-889``) and ``apply_child``
    both force ``reset=False`` on a nested pipeline unconditionally -- so any
    divergence this fixture produces is staged-only, which is exactly what an
    equivalence test is for.
    """
    inner_pre = ImagePipeline(
        ops={
            "ContrastStretching": ContrastStretching(
                lower_percentile=2, upper_percentile=98, input_layer="detect_mat"
            )
        },
        reset=True,
    )
    branch = ImagePipeline(ops={"pre": inner_pre, "FakeGpuDetector": _gpu()})
    return CompositeDetector(ops=[branch, _manual()], mode="overlap")


SHAPES = {"a": shape_a, "b": shape_b, "c": shape_c, "d": shape_d}


def _make_pipeline(factory) -> ImagePipeline:
    """The spike's surrounding pipeline: CPU ops, the composite, a refiner.

    ``BlurGauss``/``SubtractGaussian`` precede the composite so Stage 1 has real
    work to publish -- without them a prefix that discarded Stage 1's output
    would discard nothing and shape D could not discriminate.
    """
    return ImagePipeline(
        ops={
            "BlurGauss": BlurGauss(sigma=2.0),
            "SubtractGaussian": SubtractGaussian(sigma=50.0),
            "CompositeDetector": factory(),
            "SmallObjectRemover": SmallObjectRemover(min_size=50),
        },
        meas={"MeasureShape": MeasureShape(), "MeasureSize": MeasureSize()},
    )


# ---------------------------------------------------------------------------
# Shared premises and helpers
# ---------------------------------------------------------------------------


def _detector_step_paths(image) -> list[tuple[str, ...]]:
    """Recorded ``pipeline_step_path``s whose entry names the GPU detector.

    A staged run records them through ``ReplayDetector``'s identity delegation,
    so both runs name ``FakeGpuDetector``; that delegation is what makes this a
    parity check rather than a comparison of two different classes.
    """
    return [
        tuple(operation["pipeline_step_path"])
        for application in image._metadata.provenance_journal.get("applications", [])
        for operation in application.get("operations", [])
        if operation.get("pipeline_step_path")
        and operation["operation_class"].rsplit(".", 1)[-1] == "FakeGpuDetector"
    ]


def _assert_fixture_is_nested(plan: StagePlan) -> None:
    """Refuse a plan whose detector sits at the top level.

    Vacuity (2) in the module docstring: every path-parity assertion in this
    file holds under ANY addressing scheme once ``gpu_path`` is one segment.
    """
    assert len(plan.gpu_path) >= 2, (
        f"fixture is not nested: gpu_path={plan.gpu_path}. A one-segment path "
        "makes every step-path assertion in this file pass on broken code."
    )


def _assert_the_run_changed_the_objmap(objmap: np.ndarray) -> None:
    """Refuse a comparison that would be satisfied by the ground truth.

    Vacuity (3): ``load_synth_yeast_plate()`` ships a populated objmap, so two
    runs that both wrote nothing would compare equal at ~96 objects.
    """
    pristine = load_synth_yeast_plate().objmap[:]
    assert not np.array_equal(objmap, pristine), (
        "the pipeline left the plate's ground-truth objmap in place; an "
        "equality check against another such run proves nothing"
    )


def _assert_same_measurements(
    reference: pd.DataFrame, staged: pd.DataFrame, *, check_dtype: bool = True
) -> None:
    """Same columns, same rows, same values -- values compared EXACTLY.

    ``check_exact=True`` rather than ``assert_frame_equal``'s float default
    (``rtol=1e-5``): the claim is that a staged run and a single-pass run
    compute the same numbers, not similar ones, and a 1e-5 window is wide
    enough to hide a real divergence in a measurement expressed in pixels.

    ``check_dtype`` is relaxed only for the store-driven comparison, where a
    parquet round-trip may legitimately re-type an integer column while every
    value stays identical. Dtype is a storage-format property; value equality
    is the scientific claim.
    """
    missing = sorted(set(reference.columns) - set(staged.columns))
    extra = sorted(set(staged.columns) - set(reference.columns))
    assert not missing and not extra, (
        f"measurement columns diverged: staged is missing {missing} and adds "
        f"{extra}"
    )
    assert_frame_equal(
        reference.reset_index(drop=True),
        staged[list(reference.columns)].reset_index(drop=True),
        check_exact=True,
        check_dtype=check_dtype,
    )


# ---------------------------------------------------------------------------
# In-memory: the spike's sequence, driven through the production helpers
# ---------------------------------------------------------------------------


def _run_single_pass(factory):
    image = load_synth_yeast_plate()
    pipeline = _make_pipeline(factory)
    measurements = pipeline.apply_and_measure(image, inplace=True, apply_post=False)
    _assert_the_run_changed_the_objmap(image.objmap[:])
    return image.objmap[:].copy(), measurements, _detector_step_paths(image)


def _run_staged_in_memory(factory, *, corrupt=None):
    """Stage 1 / 2 / 3 in one process, using the real split, prefix and stub.

    The spike reimplemented all three; this does not. ``split_pipeline_at_gpu``,
    ``_apply_stage2_prefix``, ``ReplayDetector`` and ``substitute_at_path`` are
    the production objects. What is NOT production here is the store: the stages
    hand each other an in-memory image, which is why the store-driven tests
    below exist as well.
    """
    image = load_synth_yeast_plate()
    pipeline = _make_pipeline(factory)
    plan = split_pipeline_at_gpu(pipeline)
    _assert_fixture_is_nested(plan)

    plan.pre_pipeline.apply(image, inplace=True)  # Stage 1

    probe = (  # Stage 2
        _apply_stage2_prefix(image, plan.stage2_prefix)
        if plan.stage2_prefix
        else image
    )
    detector = plan.gpu_detector
    array = getattr(probe, detector.input_layer)[:]
    raw = detector._infer_batch(detector._collate([detector._preprocess(array)]))[0]
    if corrupt is not None:
        raw = corrupt(raw)

    stub = ReplayDetector(detector=detector, result=raw)  # Stage 3
    replay_pipeline = substitute_at_path(plan.post_pipeline, plan.gpu_path, stub)
    replay_pipeline.apply(image, inplace=True)
    measurements = replay_pipeline.measure(image, apply_post=False)
    return image.objmap[:].copy(), measurements, _detector_step_paths(image), plan


@pytest.mark.parametrize("name", list(SHAPES), ids=list(SHAPES))
def test_staged_matches_single_pass_in_memory(name):
    """Pixels, measurements and the recorded step path, on four nestings.

    This is the ALGORITHM check. It cannot fail on a broken Stage 3 -- it calls
    ``substitute_at_path`` itself -- so read a failure here as "the split /
    prefix / replay sequence is wrong", and read
    ``test_the_staged_store_matches_a_single_pass_run`` as the wiring check.
    """
    factory = SHAPES[name]
    ref_objmap, ref_meas, ref_paths = _run_single_pass(factory)
    staged_objmap, staged_meas, staged_paths, plan = _run_staged_in_memory(factory)

    assert np.array_equal(ref_objmap, staged_objmap)
    _assert_same_measurements(ref_meas, staged_meas)
    assert ref_paths == [tuple(plan.gpu_path)]
    assert staged_paths == ref_paths


def test_a_corrupted_replay_is_detected():
    """Without this the equivalence assertions above are vacuous.

    The perturbation is applied to the Stage-2 raw array, which is the one thing
    a staged run carries between processes -- so this is the failure mode the
    equivalence test exists to exclude, not an arbitrary one.

    Compare the OBJMAP. ``num_objects`` is deliberately not asserted either way:
    ``split_disconnected_labels`` defaults to ``True`` on ``GpuDetector``, so a
    wrapped roll may or may not change the count, and a count comparison is
    therefore not guaranteed to discriminate. That has not been measured here
    and no claim is made about it.
    """
    ref_objmap, _, _ = _run_single_pass(shape_a)
    clean_objmap, _, _, _ = _run_staged_in_memory(shape_a)
    dirty_objmap, _, _, _ = _run_staged_in_memory(
        shape_a, corrupt=lambda raw: np.roll(raw, 7, axis=0)
    )

    assert np.array_equal(ref_objmap, clean_objmap)
    assert not np.array_equal(ref_objmap, dirty_objmap)


# ---------------------------------------------------------------------------
# The Stage-2 prefix fixtures, stated as premises
# ---------------------------------------------------------------------------


def test_shape_d_puts_a_reset_true_pipeline_in_the_stage2_prefix():
    """The mutation fixture's premise, so a regression lands HERE by name.

    Delete ``_apply_stage2_prefix``'s ``isinstance(operation,
    ImagePipelineCore)`` guard and the equivalence test for shape D goes red;
    this test is what says why, instead of leaving a reader to work out which
    fixture property the failure depended on.
    """
    plan = split_pipeline_at_gpu(_make_pipeline(shape_d))

    assert [type(op).__name__ for op in plan.stage2_prefix] == ["ImagePipeline"]
    assert plan.stage2_prefix[0].reset is True, (
        "the prefix pipeline no longer carries reset=True, so the guard it "
        "exists to exercise is now unobservable"
    )
    assert plan.gpu_path == ("CompositeDetector", "ops[0]", "FakeGpuDetector")


def test_shape_b_cannot_distinguish_the_reset_guard():
    """The control that makes shape D worth having.

    Shape B's prefix is a leaf operation. ``ImageOperation.apply`` takes no
    ``reset``, so the guarded and unguarded paths call it identically and no
    equivalence assertion over shape B can see the guard. This is why "the
    staged tests are green" was not evidence about it.
    """
    plan = split_pipeline_at_gpu(_make_pipeline(shape_b))

    assert [type(op).__name__ for op in plan.stage2_prefix] == ["ContrastStretching"]
    assert not hasattr(plan.stage2_prefix[0], "reset")


# ---------------------------------------------------------------------------
# Store-driven: the production stage cores over a real staged store
# ---------------------------------------------------------------------------


class _StagedStoreRun:
    """One image through ``stage1`` / ``stage2`` / ``stage3``, on disk.

    Follows the established pattern (``tests/integration/cli/conftest.py:264``,
    ``tests/unit/cli/test_staged_stage2_prefix.py:113``) and differs in one
    respect that matters: ``run_stage2`` forwards ``plan.stage2_prefix``. The
    integration harness does not, because its fixtures have no prefix.
    """

    def __init__(self, tmp_path: Path, factory) -> None:
        self.image_path = tmp_path / "img.tiff"
        load_synth_yeast_plate().rgb.imsave(filepath=self.image_path)
        self.output_dir = tmp_path / "out"
        self.output_dir.mkdir(exist_ok=True)
        self.factory = factory
        self.pipeline = _make_pipeline(factory)
        self.pipeline_path = tmp_path / "pipeline.json"
        self.pipeline_path.write_text(self.pipeline.to_json(), encoding="utf-8")
        assert self.pipeline_path.is_file(), "pipeline was not written"
        self.plan = split_pipeline_at_gpu(self.pipeline)
        self.slot = detector_slot(self.plan.gpu_path)
        self.output_manager = OutputManager.from_config(
            self.output_dir, ".tiff", save_overlays=False
        )
        self.output_manager.create_structure(
            [Dataset("ds", [self.image_path], tmp_path, self.output_dir)]
        )

    @property
    def store(self) -> Path:
        return zarr_store_path(self.output_dir, "ds", "img")

    def journal(self) -> dict:
        payload = json.loads((self.store / "zarr.json").read_text(encoding="utf-8"))
        return payload["attributes"]["phenotypic"]["provenance"]

    def run_stage1(self) -> None:
        stage1_preprocess_core(
            self.plan,
            self.image_path,
            "ds",
            "img",
            self.output_dir,
            self.output_manager,
            image_type="Image",
            pipeline_path=self.pipeline_path,
        )

    def run_stage2(self) -> None:
        self.plan.gpu_detector._ensure_model_loaded()
        stage2_detect_core(
            self.plan.gpu_detector,
            self.output_dir,
            "ds",
            "img",
            self.slot,
            "Image",
            stage2_prefix=self.plan.stage2_prefix,
        )

    def run_stage3(self) -> None:
        stage3_merge_measure_core(
            self.plan,
            self.output_dir,
            "ds",
            "img",
            self.output_manager,
            image_type="Image",
        )

    def run_all(self) -> None:
        self.run_stage1()
        self.run_stage2()
        self.run_stage3()

    def embedded_measurements(self) -> pd.DataFrame:
        return pl.read_parquet(
            self.store / MEASUREMENT_TABLE_RELATIVE_PATH
        ).to_pandas()

    def single_pass_reference(self):
        """A single-pass run of the SAME bytes Stage 1 read.

        ``Image.imread(self.image_path)`` rather than
        ``load_synth_yeast_plate()``: the plate ships a populated ground-truth
        objmap, and comparing against it would let a staged run that wrote no
        labels at all look correct.
        """
        image = Image.imread(self.image_path)
        pipeline = _make_pipeline(self.factory)
        measurements = pipeline.apply_and_measure(image, inplace=True, apply_post=False)
        assert image.objmap[:].any(), "the reference run detected nothing"
        return image, measurements


@pytest.mark.parametrize("name", ["a", "d"], ids=["shape-a", "shape-d"])
def test_the_staged_store_matches_a_single_pass_run(tmp_path, name):
    """The WIRING check, and the only thing here that can see Task 8.

    Drives the real ``stage2_detect_core`` + ``stage3_merge_measure_core``
    against a real staged store, on a nested detector. Shape A is the plain
    nested leaf; shape D additionally carries the ``reset=True`` pipeline in its
    Stage-2 prefix, so deleting ``_apply_stage2_prefix``'s guard fails here.

    **The one excluded column, and why the exclusion is legitimate.** The
    store-embedded table carries exactly one column the single-pass
    ``.measure()`` frame does not: ``str(EXPERIMENT.DATASET)``, i.e.
    ``Metadata_Dataset``. It is **not** a staged/single-pass divergence.
    ``OutputManager.save_image_store`` inserts it into the baseline before
    embedding (``_cli_output_manager.py:1895-1903``) whenever
    ``include_dataset_column`` is set, and ``OutputManager.from_config`` defaults
    that to ``True`` (``:1605``). It is a run-configuration property: the dataset
    name is something the CLI knows and a bare ``pipeline.measure(image)`` call
    cannot, because there is no dataset in that world to name. The column
    therefore tracks the ``OutputManager``, not the execution strategy -- which
    is why its presence says nothing about staging either way.

    The exclusion is spelled through the schema rather than as the literal
    ``"Metadata_Dataset"``, because that is the same expression the writer uses
    -- a rename cannot desynchronise the test from the code.

    This is a **named exclusion, not an intersection**. The two assertions below
    are two-sided: the staged table may add that column and nothing else, and it
    may drop nothing at all. Any other column appearing or disappearing --
    including the ``Metadata_Dataset`` column vanishing -- fails here rather than
    being silently skipped, which is what an ``&`` of the two column sets would
    have done.
    """
    run = _StagedStoreRun(tmp_path, SHAPES[name])
    _assert_fixture_is_nested(run.plan)
    run.run_all()

    reference, ref_meas = run.single_pass_reference()
    staged = Image.load_zarr(run.store)

    np.testing.assert_array_equal(reference.objmap[:], staged.objmap[:])

    staged_meas = run.embedded_measurements()
    dataset_column = str(EXPERIMENT.DATASET)
    assert set(staged_meas.columns) - set(ref_meas.columns) == {dataset_column}
    assert not set(ref_meas.columns) - set(staged_meas.columns)
    _assert_same_measurements(
        ref_meas, staged_meas.drop(columns=[dataset_column]), check_dtype=False
    )

    # The journal parity claim of spec 5.3, on a fixture where the two
    # candidate addressing schemes DISAGREE.
    assert _detector_step_paths(reference) == [tuple(run.plan.gpu_path)]
    assert _detector_step_paths(staged) == [tuple(run.plan.gpu_path)]


def test_stage3_records_the_branchs_own_ops_under_the_enclosing_operation(tmp_path):
    """Stage 3 runs the enclosing operation, it does not write the objmap.

    Shape D's branch contains a nested pipeline keyed ``"pre"`` whose
    ``ContrastStretching`` Stage 1 never ran. If Stage 3 wrote the recorded
    objmap directly and then applied ``post_pipeline``, that operation would
    still appear -- so what this pins is the PATH it appears under: the full
    ``CompositeDetector/ops[0]/pre/ContrastStretching``, which only exists
    because the substitution happened inside the enclosing operation's own
    descent.
    """
    run = _StagedStoreRun(tmp_path, shape_d)
    run.run_all()

    journal = run.journal()
    paths = [
        tuple(operation["pipeline_step_path"])
        for application in journal.get("applications", [])
        for operation in application.get("operations", [])
        if operation.get("pipeline_step_path")
    ]
    assert ("CompositeDetector", "ops[0]", "pre", "ContrastStretching") in paths, (
        f"the branch's own prefix op is not recorded inside the enclosing "
        f"operation; recorded paths were {paths}"
    )
    assert ("CompositeDetector", "ops[0]", "FakeGpuDetector") in paths


def test_stage3_runs_at_cli_owner_depth_against_a_staged_store(tmp_path):
    """Stage 3's depth-0 exposure comes from the store, not from the ContextVar.

    ``_application_owner_depth`` is a ``ContextVar`` with ``default=0``
    (``_provenance.py:80-81``), so setting it to 0 outside an enclosing
    ``provenance_application`` is a **no-op** -- an earlier draft of this test
    described that as the mechanism and was wrong. It is set anyway, explicitly,
    so that a future enclosing application cannot silently change what this test
    exercises.

    The real exposure is the trailing ``"staged"`` application Stage 1 leaves in
    the store: ``"staged"`` is outside ``_append_application``'s terminal set,
    so Stage 3 must JOIN that application rather than append a new one. The
    status transition below is the observable consequence.
    """
    from phenotypic._core._provenance import _application_owner_depth

    run = _StagedStoreRun(tmp_path, shape_a)
    run.run_stage1()
    run.run_stage2()
    assert run.journal()["status"] == "staged"

    token = _application_owner_depth.set(0)
    try:
        run.run_stage3()
    finally:
        _application_owner_depth.reset(token)

    completed = run.journal()
    assert completed["status"] == "complete"
    assert len(completed["applications"]) == 1, (
        "Stage 3 appended a second application instead of joining the staged "
        f"one: {[app.get('status') for app in completed['applications']]}"
    )
