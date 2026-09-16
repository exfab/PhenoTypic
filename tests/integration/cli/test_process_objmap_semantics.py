"""``--mode process --layer objmap`` exports the PIPELINE's objmap.

Spec ``2026-09-15-nested-gpu-staging`` §8: the export runs the post-detector op
chain, for a top-level detector and for a nested one alike, so the operations a
user supplies are the operations that shaped the exported map.

**Why these fixtures and not the obvious ones.** Two shapes would make every
assertion below pass on the code this file exists to reject:

* ``load_synth_yeast_plate()`` arrives with a populated 96-object ground-truth
  objmap, so an object-count assertion can be satisfied by data that predates
  the pipeline entirely. Every test here builds its own two-blob image instead
  (:func:`_write_two_blob_image`), whose objmap starts empty and whose expected
  counts are designed rather than discovered.
* A **top-level** detector with no post-detector ops makes "runs the chain" and
  "writes the raw array" observationally identical -- that is precisely why
  ``test_staged_gpu_local.py``'s existing export test cannot catch this change.
  Each test here therefore carries a *control*: the objmap the detector alone
  produces, computed in-process, asserted to differ from what the export wrote.
  The control is what makes the main assertion discriminating; without it,
  "1 object" and "3 objects" are numbers with no witness.
"""

import hashlib

import cv2
import numpy as np
import phenotypic
import pytest

from phenotypic import Image, ImagePipeline
from phenotypic._cli._cli_failure_tracker import work_id_for_image
from phenotypic._cli._cli_output_manager import OutputManager
from phenotypic._cli._cli_pipeline_split import split_pipeline_at_gpu
from phenotypic._cli._cli_process_only import process_only_output_path
from phenotypic._cli._cli_staged_resume import staged_store_matches_work_id
from phenotypic._cli._cli_staged_strategy import StagedGpuStrategy
from phenotypic._cli._cli_staged_workers import (
    stage1_preprocess_core,
    stage2_detect_core,
)
from phenotypic._cli._cli_stage2_token import detector_slot
from phenotypic._cli._cli_types import Dataset, ExecutionConfig
from phenotypic.detect import CompositeDetector, ManualPointDetector
from phenotypic.refine import SmallObjectRemover
from phenotypic.sdk_ import event_log_path, zarr_store_path
from tests._fakes.fake_gpu_detector import FakeGpuDetector

# --------------------------------------------------------------------------
# Fixture geometry. Every number in this file is derived from these four.
# --------------------------------------------------------------------------

#: A bright square the detector finds and every refiner keeps (3600 px).
BLOB = (slice(30, 90), slice(30, 90))
#: A bright speck the detector also finds, below any sane ``min_size`` (9 px).
SPECK = (slice(180, 183), slice(180, 183))
#: Centre of a square stamped by a CPU sibling detector, in a region that is
#: pure background -- so the GPU branch can never produce it and its presence in
#: an exported map proves the composite's merge ran.
MANUAL_CENTER = (180.0, 40.0)
MANUAL_WIDTH = 41


@pytest.fixture(autouse=True)
def _register_fake_gpu_detector(monkeypatch):
    """Make ``FakeGpuDetector`` resolvable by ``ImagePipeline.from_json``.

    The staged strategy loads its pipeline from ``pipeline.json`` and the
    deserializer resolves op classes from the ``phenotypic`` namespace (the
    pattern at ``test_staged_gpu_local.py:99``).
    """
    monkeypatch.setattr(
        phenotypic, "FakeGpuDetector", FakeGpuDetector, raising=False
    )


def _write_two_blob_image(tmp_path):
    """A 240x240 RGB image holding exactly one large blob and one 9-px speck.

    Deliberately NOT ``load_synth_yeast_plate()``: that fixture ships a
    populated objmap, and an object-count assertion made against it can pass
    for reasons that have nothing to do with the export path.
    """
    array = np.zeros((240, 240, 3), dtype=np.uint8)
    array[BLOB] = 255
    array[SPECK] = 255
    path = tmp_path / "img.tiff"
    assert cv2.imwrite(str(path), array), "fixture image was not written"
    assert path.is_file()
    return path


def _detector_only_objmap(image_path):
    """The objmap the OLD semantics exported: the detector's raw output.

    Computed in-process from the same file Stage 1 reads, with the same
    detector, so it is exactly ``gpu_detector._write_object_output(image, raw)``
    -- the call this change replaces. Used as the control in every test below.
    """
    control = Image.imread(image_path)
    FakeGpuDetector(threshold=0.3, output_kind="instance").apply(
        control, inplace=True
    )
    return np.asarray(control.objmap[:])


def _label_count(objmap) -> int:
    """Number of distinct non-zero labels in a raw-label map."""
    return int(np.count_nonzero(np.unique(np.asarray(objmap))))


def _config(out, pipe_path, *, resume=False):
    return ExecutionConfig(
        pipeline_json=pipe_path,
        input_path=out,
        output_dir=out,
        image_type="Image",
        nrows=None,
        ncols=None,
        bit_depth=None,
        n_jobs=1,
        slurm_args={},
        force_local=True,
        wait=False,
        ext=".tiff",
        overlay_alpha=0.5,
        include_dataset_column=False,
        dry_run=False,
        sample=None,
        resume=resume,
        retry_failures=False,
        skip_validation=True,
        save_overlays=False,
        process_only_layer="objmap",
    )


def _write_pipeline(out, pipe):
    """Serialize *pipe* to ``out/pipeline.json`` and assert the file exists.

    ``to_json()`` with no argument RETURNS the JSON; ``to_json(path)`` writes to
    ``ensure_typed_json_suffix(path, ".json.pht-pipe")`` and returns ``None``,
    which leaves the requested path empty and every downstream read failing
    somewhere else. The post-condition keeps that failure here.
    """
    pipe_path = out / "pipeline.json"
    pipe_path.write_text(pipe.to_json(), encoding="utf-8")
    assert pipe_path.is_file() and pipe_path.stat().st_size > 0
    return pipe_path


def _run_process_export(tmp_path, pipe):
    """Run the whole staged process-mode export and return the exported map."""
    image_path = _write_two_blob_image(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    pipe_path = _write_pipeline(out, pipe)
    om = OutputManager.from_config(out, ".tiff", save_overlays=False)
    datasets = [Dataset("ds", [image_path], tmp_path, out)]
    om.create_structure(datasets)

    cfg = _config(out, pipe_path)
    results = StagedGpuStrategy(cfg, om).execute(datasets, out)
    assert results.total_failed == 0, "the staged run did not reach the export"

    exported_path = process_only_output_path(out, image_path, out, "objmap")
    assert exported_path.is_file()
    exported = cv2.imread(str(exported_path), cv2.IMREAD_UNCHANGED)
    assert exported is not None
    return image_path, out, exported


# --------------------------------------------------------------------------
# Top-level detector: the post-detector op chain runs
# --------------------------------------------------------------------------


def test_export_applies_post_detector_ops(tmp_path):
    """``--layer objmap`` exports the PIPELINE's objmap, not the raw output.

    The pipeline is detector -> ``SmallObjectRemover(min_size=100)``. The
    detector finds two objects; the refiner removes one. An export that writes
    the detector's raw array keeps both.

    What would make this pass on the old code: a fixture whose detector output
    already has one object (nothing for the refiner to do), or one whose
    exported map is all zeros (``0 != 1`` catches that, but only if the count is
    an equality and not a ``> 0``). The control below pins the first: the
    detector alone really does produce two.
    """
    pipe = ImagePipeline(
        ops=[
            FakeGpuDetector(output_kind="instance", threshold=0.3),
            SmallObjectRemover(min_size=100),
        ]
    )
    image_path, _out, exported = _run_process_export(tmp_path, pipe)

    control = _detector_only_objmap(image_path)
    assert _label_count(control) == 2, (
        "fixture broken: the detector alone must find the blob AND the speck, "
        "or 'the speck was removed' is not observable"
    )

    assert _label_count(exported) == 1, (
        "the speck survived -- post-detector ops were not applied"
    )
    assert exported[BLOB].min() > 0, "the surviving object is not the blob"
    assert exported[SPECK].max() == 0, "the speck was not removed"


# --------------------------------------------------------------------------
# Nested detector: the composite's merge runs
# --------------------------------------------------------------------------


def test_export_applies_the_composite_merge_for_a_nested_detector(tmp_path):
    """For a nested detector the raw array is one BRANCH, not the objmap.

    ``CompositeDetector(mode="union", ops=[FakeGpuDetector, ManualPoint...])``
    -- the GPU branch finds the blob and the speck; the CPU sibling stamps a
    square over pure background, which the GPU branch cannot produce at any
    threshold. So the merged map has three objects and the raw array has two,
    and the stamped square is present in exactly one of them.

    The shape matters. On a TOP-LEVEL fixture "ran the chain" and "wrote the raw
    array" are the same bytes, so the assertions below would certify nothing;
    the detector here is at ``("CompositeDetector", "ops[0]")``, pinned by the
    slot assertion, and the merge is the only thing that can put a label at
    ``MANUAL_CENTER``.
    """
    manual = ManualPointDetector(
        centers=[MANUAL_CENTER], shape="square", width=MANUAL_WIDTH
    )
    pipe = ImagePipeline(
        ops={
            "CompositeDetector": CompositeDetector(
                ops=[
                    FakeGpuDetector(output_kind="instance", threshold=0.3),
                    manual,
                ],
                mode="union",
            )
        }
    )
    # The detector really is nested: a one-element path would mean this test is
    # exercising the top-level shape under a nested-looking name.
    plan = split_pipeline_at_gpu(pipe)
    assert plan.gpu_path == ("CompositeDetector", "ops[0]")

    image_path, _out, exported = _run_process_export(tmp_path, pipe)

    control = _detector_only_objmap(image_path)
    cy, cx = int(MANUAL_CENTER[0]), int(MANUAL_CENTER[1])
    assert _label_count(control) == 2, (
        "fixture broken: the GPU branch alone must find exactly the blob and "
        "the speck"
    )
    assert control[cy, cx] == 0, (
        "fixture broken: the CPU sibling's square must sit where the GPU "
        "branch finds nothing, or the merge is not observable"
    )

    assert exported[cy, cx] != 0, (
        "the composite's sibling branch is missing from the export -- only "
        "the GPU branch's raw array was written"
    )
    assert _label_count(exported) == 3, (
        "expected blob + speck + the sibling's square after the union merge"
    )


# --------------------------------------------------------------------------
# FLOW-16 / FLOW-30 / FLOW-6: the export never writes into the store
# --------------------------------------------------------------------------


def _store_digest(store) -> str:
    """Content digest of every file in *store*, independent of production code.

    Deliberately not ``_cli_failure_tracker``'s store hasher: this test exists
    to audit a store-write invariant, and an auditor that shares an
    implementation with the thing it audits reports agreement, not correctness.
    """
    digest = hashlib.sha256()
    for member in sorted(store.rglob("*"), key=lambda p: p.as_posix()):
        digest.update(member.relative_to(store).as_posix().encode("utf-8"))
        digest.update(b"/" if member.is_dir() else b"f")
        if member.is_file():
            digest.update(member.read_bytes())
    return digest.hexdigest()


def test_the_store_is_byte_unchanged_by_the_export(tmp_path):
    """The export runs a whole op chain now, and still writes nothing to disk.

    The chain is applied under ``continuing_provenance_application`` with **no**
    ``provenance_success_sink``; the sink is what writes provenance into the
    store, and a store write after ``_publish_local_image_success`` invalidates
    the descriptor the marker just recorded (ledger FLOW-16/FLOW-30/FLOW-6).

    A digest comparison passes trivially when the export never runs -- an
    absent Stage-2 signal makes ``_export_objmap_layer`` record a missing
    prerequisite and move on, leaving the store untouched for the wrong reason.
    The ``completed == 1`` and exported-content assertions are what rule that
    out; they are not decoration.

    It deliberately does NOT assert *which* objmap was exported. The store
    invariant holds under both the old and the new export semantics -- that is
    the point of it -- so this test passes under the old-semantics mutant and
    must not be counted as evidence for the semantics change (that is
    :func:`test_export_applies_post_detector_ops`'s job). An earlier draft
    asserted the chain's result here too, and failed under the mutant while
    naming the store invariant, which reports the wrong cause.

    ``_export_objmap_layer`` is called directly rather than through
    ``execute()`` because the "before" digest has to be taken between Stage 2
    and the export, and ``execute()`` runs both.
    """
    image_path = _write_two_blob_image(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    pipe = ImagePipeline(
        ops=[
            FakeGpuDetector(output_kind="instance", threshold=0.3),
            SmallObjectRemover(min_size=100),
        ]
    )
    pipe_path = _write_pipeline(out, pipe)
    om = OutputManager.from_config(out, ".tiff", save_overlays=False)
    datasets = [Dataset("ds", [image_path], tmp_path, out)]
    om.create_structure(datasets)

    plan = split_pipeline_at_gpu(ImagePipeline.from_json(pipe_path))
    stage1_preprocess_core(
        plan, image_path, "ds", "img", out, om, image_type="Image"
    )
    plan.gpu_detector._ensure_model_loaded()
    stage2_detect_core(
        plan.gpu_detector,
        out,
        "ds",
        "img",
        detector_slot(plan.gpu_path),
        # Keyword, not the sixth positional: `slot` was inserted ahead of
        # `image_type` in this signature, and a positional call there binds
        # "Image" to `slot` without raising -- the signal then lands in a slot
        # named `Image` and every assertion that inspects the real one passes
        # against an empty directory.
        image_type="Image",
        stage2_prefix=plan.stage2_prefix,
    )

    store = zarr_store_path(out, "ds", "img")
    before = _store_digest(store)

    cfg = _config(out, pipe_path)
    results = {"ds": {"total": 1, "completed": 0, "failed": 0}}
    StagedGpuStrategy(cfg, om)._export_objmap_layer(
        plan,
        [(datasets[0], image_path)],
        out,
        event_log_path(out),
        results,
    )

    assert results["ds"] == {"total": 1, "completed": 1, "failed": 0}
    exported = cv2.imread(
        str(process_only_output_path(out, image_path, out, "objmap")),
        cv2.IMREAD_UNCHANGED,
    )
    assert exported is not None and exported.max() > 0, (
        "the export produced nothing, so an unchanged store says nothing"
    )

    assert _store_digest(store) == before, (
        "the export wrote into the store; the success marker's descriptor is "
        "now stale (ledger FLOW-16 / FLOW-30 / FLOW-6)"
    )
    stored = np.asarray(Image.load_zarr(store).objmap[:])
    assert stored.max() == 0, "the store still holds Stage 1's zeros"


# --------------------------------------------------------------------------
# Task 12, end to end: the semantics revision invalidates an old-semantics tree
# --------------------------------------------------------------------------


def test_bumping_the_semantics_revision_invalidates_an_existing_tree(tmp_path):
    """A tree processed under the old semantics is not reused under the new.

    The run is performed three times with ``resume=True`` and **nothing on disk
    is touched between them**. The observable is the staged store's own bytes:
    Stage 1 skips only when the store already carries the run's work id, and a
    re-run rewrites it with a fresh journal (a new ``applied_at_utc``, and the
    new work id itself), so a changed store digest is proof the work was redone.

    All three runs are needed. "The store changed" on its own passes on code
    where resume never skips anything, so the middle run pins that an unchanged
    revision DOES skip -- the digest is identical across it -- and only then
    does the third run's change mean the revision is what changed the answer.

    **Deleting or overwriting the exported PNG cannot be the observable.** An
    earlier draft deleted it between runs and the middle run re-derived the
    image, defeating its own control. The success marker's artifact descriptor
    carries the PNG's size and sha256 (``_cli_completion._artifact_descriptor``)
    and ``valid_image_success`` re-checks both, so touching the output is itself
    an invalidation and the test would have measured its own tampering. Its
    premise assertion is what caught that.

    This test says nothing about which objmap is exported, so it passes under
    the old-export-semantics mutant -- deliberately. Task 11's discrimination
    lives in the three tests above; this one's discrimination is the work-id
    inequality, which fails the moment the revision leaves the digest payload.

    The revision is restored by hand rather than through ``monkeypatch``: the
    autouse fixture above shares this test's ``monkeypatch`` instance, so an
    ``undo()`` here would also unregister ``FakeGpuDetector`` and the final
    ``from_json`` would fail for an unrelated reason.
    """
    from phenotypic._cli import _cli_failure_tracker as tracker

    image_path = _write_two_blob_image(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    pipe = ImagePipeline(
        ops=[
            FakeGpuDetector(output_kind="instance", threshold=0.3),
            SmallObjectRemover(min_size=100),
        ]
    )
    pipe_path = _write_pipeline(out, pipe)
    om = OutputManager.from_config(out, ".tiff", save_overlays=False)
    datasets = [Dataset("ds", [image_path], tmp_path, out)]
    om.create_structure(datasets)
    cfg = _config(out, pipe_path, resume=True)
    exported_path = process_only_output_path(out, image_path, out, "objmap")

    store = zarr_store_path(out, "ds", "img")
    shipped = tracker.PROCESS_LAYER_SEMANTICS_REVISION
    try:
        # ---- the "before the upgrade" run -----------------------------
        tracker.PROCESS_LAYER_SEMANTICS_REVISION = shipped - 1
        old_work_id, _ = work_id_for_image(cfg, "ds", image_path)
        StagedGpuStrategy(cfg, om).execute(datasets, out)
        assert exported_path.is_file()
        assert staged_store_matches_work_id(store, old_work_id)
        after_first = _store_digest(store)

        # ---- resumed at the SAME revision: reused, nothing re-derived -
        StagedGpuStrategy(cfg, om).execute(datasets, out)
        assert _store_digest(store) == after_first, (
            "the run re-derived the image at an unchanged revision -- this "
            "test cannot distinguish reuse from invalidation"
        )
    finally:
        tracker.PROCESS_LAYER_SEMANTICS_REVISION = shipped

    # ---- resumed after the upgrade: invalidated, re-derived -----------
    new_work_id, _ = work_id_for_image(cfg, "ds", image_path)
    assert new_work_id != old_work_id, (
        "the work id ignores the output-semantics revision, so an upgraded "
        "run would accept every old-semantics export as complete"
    )
    StagedGpuStrategy(cfg, om).execute(datasets, out)
    assert _store_digest(store) != after_first, (
        "an old-semantics tree was reused after the semantics revision changed"
    )
    assert staged_store_matches_work_id(store, new_work_id)
    assert not staged_store_matches_work_id(store, old_work_id)
    assert exported_path.is_file()
