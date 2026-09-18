"""Stage 2 applies the GPU branch's CPU prefix, in memory and to nothing else.

Shape B (nested-staging spec §4.2): the detector sits *behind* CPU operations
**inside its own branch**. Stage 1 stops at the detector's top-level ancestor,
so those operations never ran and the published store holds pre-branch pixels.
Stage 2 must therefore apply them itself -- to a discarded in-memory copy,
because Stage 3 re-runs the identical operations inside the enclosing operation
and is the run that records them.

Three separate claims, and they fail independently:

1. the detector reads the **prefixed** array (Task 7);
2. Stage 2 still writes **nothing** -- not the pixels, not the journal (Task 7);
3. both production call sites actually **pass** the prefix (Task 13). The
   parameter defaults to ``None``, so a call site that forgets it is silent:
   shape B simply detects on the wrong pixels and reports success.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import phenotypic
import pytest
from pydantic import PrivateAttr

from phenotypic import Image, ImagePipeline
from phenotypic._cli._cli_output_manager import OutputManager
from phenotypic._cli._cli_pipeline_split import split_pipeline_at_gpu
from phenotypic._cli._cli_stage2_token import detector_slot, load_stage2_raw
from phenotypic._cli._cli_staged_workers import (
    stage1_preprocess_core,
    stage2_detect_core,
)
from phenotypic._cli._cli_types import Dataset, ExecutionConfig
from phenotypic.abc_ import GpuDetector
from phenotypic.data import load_synth_yeast_plate
from phenotypic.enhance import ImageInverter
from phenotypic.measure import MeasureSize
from phenotypic.sdk_ import zarr_store_path
from phenotypic.sdk_.typing_ import GpuInputLayer


class RecordingGpuDetector(GpuDetector):
    """Records the raw ``input_layer`` array it was handed, then labels a box.

    ``_preprocess`` is the first hook that sees the array Stage 2 read off the
    image, before any uint8 scaling, so recording here answers *"which pixels
    did the detector get"* rather than *"which pixels survived scaling"*.
    """

    input_layer: GpuInputLayer = "detect_mat"
    drop_frame_background: bool = False
    split_disconnected_labels: bool = False
    _seen: list = PrivateAttr(default_factory=list)

    def _ensure_model_loaded(self) -> None:
        return None

    def _preprocess(self, array):
        self._seen.append(np.array(array, copy=True))
        return super()._preprocess(array)

    def _infer_one(self, sample):
        labels = np.zeros(sample.shape[:2], dtype=np.uint16)
        labels[20:60, 20:60] = 1
        return labels


@pytest.fixture(autouse=True)
def _register_recording_detector(monkeypatch):
    """``ImagePipeline.from_json`` resolves op classes in the ``phenotypic``
    namespace, and the SLURM Stage-2 site round-trips the pipeline."""
    monkeypatch.setattr(
        phenotypic,
        "RecordingGpuDetector",
        RecordingGpuDetector,
        raising=False,
    )


def _shape_b_pipeline() -> ImagePipeline:
    """A GpuDetector behind ``ImageInverter`` inside a nested branch pipeline.

    The nested pipeline is a ``"sequence"`` container, so ``_branch_prefix``
    contributes the operations preceding the detector *within it* -- which is
    precisely the prefix Stage 1 did not run.
    """
    branch = ImagePipeline(
        ops={"invert": ImageInverter(), "gpu": RecordingGpuDetector()}
    )
    return ImagePipeline(ops={"branch": branch}, meas=[MeasureSize()])


def _store_digest(store: Path) -> str:
    """One digest over every file in the store, paths included."""
    digest = hashlib.sha256()
    for path in sorted(p for p in store.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(store)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _store_journal_operation_names(store: Path) -> list[str]:
    payload = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    journal = payload["attributes"]["phenotypic"]["provenance"]
    return [
        operation["operation_name"]
        for application in journal.get("applications", [])
        for operation in application.get("operations", [])
    ]


class StageOneStore:
    """A published Stage-1 store for one shape-B image, plus its plan."""

    def __init__(self, tmp_path: Path, pipeline: ImagePipeline):
        self.image_path = tmp_path / "img.tiff"
        load_synth_yeast_plate().rgb.imsave(filepath=self.image_path)
        self.output_dir = tmp_path / "out"
        self.output_dir.mkdir(exist_ok=True)
        self.pipeline = pipeline
        self.pipeline_path = tmp_path / "pipeline.json"
        self.pipeline_path.write_text(pipeline.to_json(), encoding="utf-8")
        self.plan = split_pipeline_at_gpu(pipeline)
        self.slot = detector_slot(self.plan.gpu_path)
        self.dataset = Dataset("ds", [self.image_path], tmp_path, self.output_dir)
        self.output_manager = OutputManager.from_config(
            self.output_dir, ".tiff", save_overlays=False
        )
        self.output_manager.create_structure([self.dataset])

    def run_stage1(self) -> None:
        stage1_preprocess_core(
            self.plan,
            self.image_path,
            "ds",
            "img",
            self.output_dir,
            self.output_manager,
            image_type="Image",
        )

    @property
    def store(self) -> Path:
        return zarr_store_path(self.output_dir, "ds", "img")


@pytest.fixture
def staged(tmp_path) -> StageOneStore:
    harness = StageOneStore(tmp_path, _shape_b_pipeline())
    harness.run_stage1()
    return harness


# ---------------------------------------------------------------------------
# The fixture's own premise
# ---------------------------------------------------------------------------


def test_the_fixture_really_is_shape_b(staged):
    """Without a non-empty prefix every assertion below is vacuously true."""
    assert staged.plan.gpu_path == ("branch", "gpu")
    assert [type(op).__name__ for op in staged.plan.stage2_prefix] == [
        "ImageInverter"
    ]
    # Stage 1 ran nothing: the store is the raw image, so the reference
    # computed in the tests below is comparable to it.
    assert staged.plan.pre_pipeline.get_ops() == {}


# ---------------------------------------------------------------------------
# Task 7 -- the detector reads the prefixed array
# ---------------------------------------------------------------------------


def test_the_detector_sees_the_branch_prefixed_array(staged):
    detector = staged.plan.gpu_detector
    stored = Image.load_zarr(staged.store).detect_mat[:]

    reference = Image.imread(staged.image_path)
    np.testing.assert_allclose(
        reference.detect_mat[:],
        stored,
        rtol=0,
        atol=1e-6,
        err_msg="premise broken: the Stage-1 store is not the raw image",
    )
    ImageInverter().apply(reference, inplace=True)
    expected = reference.detect_mat[:]

    stage2_detect_core(
        detector,
        staged.output_dir,
        "ds",
        "img",
        staged.slot,
        "Image",
        stage2_prefix=staged.plan.stage2_prefix,
    )

    assert len(detector._seen) == 1
    np.testing.assert_allclose(detector._seen[0], expected, rtol=0, atol=1e-6)
    # ...and it is not merely equal to what a single-pass apply produces; it
    # differs from what Stage 2 would have read had the prefix been dropped.
    assert not np.allclose(detector._seen[0], stored)


def test_without_a_prefix_the_detector_sees_the_stored_array(staged):
    """The control for the test above.

    A ``stage2_prefix`` that were silently ignored would make that test's
    `expected` comparison the only thing distinguishing the two paths, so pin
    the other side: with no prefix the detector reads the store verbatim.
    """
    detector = staged.plan.gpu_detector
    stored = Image.load_zarr(staged.store).detect_mat[:]

    stage2_detect_core(
        detector, staged.output_dir, "ds", "img", staged.slot, "Image"
    )

    np.testing.assert_allclose(detector._seen[0], stored, rtol=0, atol=1e-6)


def test_the_retained_raw_matches_the_stored_image_shape(staged):
    stage2_detect_core(
        staged.plan.gpu_detector,
        staged.output_dir,
        "ds",
        "img",
        staged.slot,
        "Image",
        stage2_prefix=staged.plan.stage2_prefix,
    )
    raw = load_stage2_raw(staged.output_dir, "ds", "img", staged.slot)
    assert raw.shape == Image.load_zarr(staged.store).detect_mat[:].shape[:2]


# ---------------------------------------------------------------------------
# Task 7 -- and Stage 2 still writes nothing
# ---------------------------------------------------------------------------


def test_the_prefix_is_applied_in_memory_and_never_written(staged):
    before = _store_digest(staged.store)

    stage2_detect_core(
        staged.plan.gpu_detector,
        staged.output_dir,
        "ds",
        "img",
        staged.slot,
        "Image",
        stage2_prefix=staged.plan.stage2_prefix,
    )

    assert _store_digest(staged.store) == before, "Stage 2 wrote into the store"


def test_the_prefix_records_land_in_no_journal(staged):
    """The prefix must reach nothing Stage 3 later reads.

    Byte-identity above already covers the store, but state this claim in its
    own terms: a future Stage 2 that legitimately rewrote some *other* part of
    the store would silence that assertion without silencing this one.
    """
    before = _store_journal_operation_names(staged.store)
    assert "ImageInverter" not in before

    stage2_detect_core(
        staged.plan.gpu_detector,
        staged.output_dir,
        "ds",
        "img",
        staged.slot,
        "Image",
        stage2_prefix=staged.plan.stage2_prefix,
    )

    assert _store_journal_operation_names(staged.store) == before


# ---------------------------------------------------------------------------
# Task 13 -- both production call sites forward the prefix
# ---------------------------------------------------------------------------


def _recording_stage2(module, monkeypatch, real):
    """Patch *module*'s ``stage2_detect_core`` to record kwargs, then call through."""
    received: list[dict] = []

    def _spy(*args, **kwargs):
        received.append(dict(kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "stage2_detect_core", _spy)
    return received


def test_the_slurm_stage2_call_site_forwards_the_prefix(staged, monkeypatch):
    from phenotypic._cli import _cli_staged_slurm_worker as worker

    received = _recording_stage2(worker, monkeypatch, stage2_detect_core)

    worker.run_stage2_shard(
        pipeline_path=staged.pipeline_path,
        output_dir=staged.output_dir,
        image_type="Image",
        manifest=[("ds", "img")],
        shard_index=0,
        n_shards=1,
    )

    assert len(received) == 1, "the SLURM Stage-2 site did not run"
    prefix = received[0].get("stage2_prefix")
    assert prefix, (
        "the SLURM Stage-2 call site dropped stage2_prefix; it defaults to "
        "None, so shape B would detect on unprefixed pixels and report success"
    )
    assert [type(op).__name__ for op in prefix] == ["ImageInverter"]
    # The fence and the write gate must survive the new keyword.
    assert "active_check" in received[0]
    assert "commit_guard" in received[0]


def test_the_local_stage2_call_site_forwards_the_prefix(tmp_path, monkeypatch):
    from phenotypic._cli import _cli_staged_strategy as strategy

    harness = StageOneStore(tmp_path, _shape_b_pipeline())
    received = _recording_stage2(strategy, monkeypatch, stage2_detect_core)

    config = ExecutionConfig(
        pipeline_json=harness.pipeline_path,
        input_path=tmp_path,
        output_dir=harness.output_dir,
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
        resume=False,
        retry_failures=False,
        skip_validation=True,
        save_overlays=False,
        measure_only=False,
        process_only_layer=None,
    )
    strategy.StagedGpuStrategy(config, harness.output_manager).execute(
        [harness.dataset], harness.output_dir
    )

    assert len(received) == 1, "the local Stage-2 site did not run"
    prefix = received[0].get("stage2_prefix")
    assert prefix, (
        "the local Stage-2 call site dropped stage2_prefix; it defaults to "
        "None, so shape B would detect on unprefixed pixels and report success"
    )
    assert [type(op).__name__ for op in prefix] == ["ImageInverter"]
