"""Shared fixtures for CLI integration tests.

Mirrors the synth-plate + pipeline-JSON setup used by
``test_cli_hdf_output.py`` so multiple integration modules can share one
deterministic input dir and serialized pipeline.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from PIL import Image as PILImage

from phenotypic import ImagePipeline
from phenotypic._cli._cli_output_manager import OutputManager
from phenotypic._cli._cli_pipeline_split import split_pipeline_at_gpu
from phenotypic._cli._cli_stage2_token import (
    load_stage2_raw,
    write_stage2_raw,
    write_stage2_token,
)
from phenotypic._cli._cli_staged_resume import remove_stage3_completion_marker
from phenotypic._cli._cli_staged_workers import (
    stage1_preprocess_core,
    stage2_detect_core,
    stage3_merge_measure_core,
)
from phenotypic._cli._cli_types import Dataset
from phenotypic.abc_ import GpuDetector
from phenotypic.data import load_synth_yeast_plate
from phenotypic.measure import MeasureSize
from phenotypic.prefab import RoundPeaksPipeline
from phenotypic.refine import SmallObjectRemover
from phenotypic.sdk_ import dataset_measurements_dir, zarr_store_path


def _write_synth_image(target_path: Path) -> None:
    """Render a synthetic yeast plate as RGB and save to ``target_path``."""
    grid_image = load_synth_yeast_plate()
    pil_img = PILImage.fromarray(grid_image.rgb[:].astype("uint8"))
    pil_img.save(target_path)


@pytest.fixture
def synth_plate_dir(tmp_path: Path) -> Path:
    """Input directory named ``plates`` carrying one synth plate image.

    Named ``plates`` so the CLI's dataset discovery (basename of the input
    dir) yields a predictable ``results/plates/...`` layout.
    """
    input_dir = tmp_path / "plates"
    input_dir.mkdir()
    _write_synth_image(input_dir / "plate_001.png")
    return input_dir


@pytest.fixture
def synth_one_level_input(tmp_path: Path) -> Path:
    """One-level input tree: ``<tmp>/in/day1/plateA.tif`` (one synth plate).

    Returns the input root (``<tmp>/in``) so callers can pass it as
    ``--input`` and assert on the mirrored process-mode output tree.
    """
    root = tmp_path / "in"
    day = root / "day1"
    day.mkdir(parents=True)
    _write_synth_image(day / "plateA.tif")
    return root


@pytest.fixture
def simple_pipeline_json():
    """Write a minimal RoundPeaksPipeline JSON to a temp file."""
    pipeline = RoundPeaksPipeline(
        blur_sigma=3,
        detector_thresh_method="otsu",
        detector_subtract_background=True,
        detector_remove_noise=True,
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as handle:
        handle.write(pipeline.to_json())
        pipeline_path = Path(handle.name)
    try:
        yield pipeline_path
    finally:
        if pipeline_path.exists():
            pipeline_path.unlink()


# ---------------------------------------------------------------------------
# Staged-GPU stage harnesses (Phase 3 Task 3.3)
# ---------------------------------------------------------------------------

class _FixedBlobDetector(GpuDetector):
    """Emit a fixed three-blob instance map, ignoring the pixels entirely.

    Areas are 400, 400 and 25 px, so ``SmallObjectRemover(min_size=100)``
    removes label 3 and nothing else. Relabeling and background dropping are
    off so the emitted label VALUES survive to the store unchanged -- the
    post-refined test compares label sets, and a relabel would renumber them.
    """

    drop_frame_background: bool = False
    split_disconnected_labels: bool = False

    def _ensure_model_loaded(self) -> None:
        return None

    def _infer_one(self, sample):
        gray = sample.mean(axis=-1) if sample.ndim == 3 else sample
        labels = np.zeros(gray.shape[:2], dtype=np.uint16)
        labels[50:70, 50:70] = 1
        labels[50:70, 100:120] = 2
        labels[150:155, 50:55] = 3
        return labels


class _BorderColonyDetector(GpuDetector):
    """A frame-forming background label PLUS a colony touching that frame.

    ``drop_frame_background`` zeroes the label owning the plurality of border
    pixels. On the first pass that is the ring (label 1). If Stage 3 ever
    replays from the already-refined store instead of the retained raw array,
    the ring is gone and the plurality passes to label 2 -- the real colony
    touching the left edge -- which then silently disappears. That is D1.
    """

    drop_frame_background: bool = True
    split_disconnected_labels: bool = False

    def _ensure_model_loaded(self) -> None:
        return None

    def _infer_one(self, sample):
        gray = sample.mean(axis=-1) if sample.ndim == 3 else sample
        return self.frame_labels(gray.shape[:2])

    @staticmethod
    def frame_labels(shape: tuple[int, int]) -> np.ndarray:
        height, width = shape
        labels = np.zeros((height, width), dtype=np.uint16)
        labels[0:2, :] = 1
        labels[-2:, :] = 1
        labels[:, 0:2] = 1
        labels[:, -2:] = 1
        mid = height // 2
        labels[mid - 10 : mid + 10, 0:20] = 2  # touches the left frame
        labels[height // 4 : height // 4 + 20, width // 2 : width // 2 + 20] = 3
        return labels


class StagedStageHarness:
    """Drive the three stage cores over one image, one dataset (``ds``/``img``)."""

    def __init__(self, plan, image_path, output_dir, output_manager, work_id):
        self.plan = plan
        self.image_path = image_path
        self.output_dir = output_dir
        self.output_manager = output_manager
        self.work_id = work_id
        self._raw_snapshot: np.ndarray | None = None

    def store(self, dataset: str = "ds", stem: str = "img") -> Path:
        return zarr_store_path(self.output_dir, dataset, stem)

    def run_stage1(self) -> None:
        stage1_preprocess_core(
            self.plan,
            self.image_path,
            "ds",
            "img",
            self.output_dir,
            self.output_manager,
            image_type="Image",
            work_id=self.work_id,
        )

    def run_stage2(self) -> None:
        self.plan.gpu_detector._ensure_model_loaded()
        stage2_detect_core(self.plan.gpu_detector, self.output_dir, "ds", "img")
        # Snapshot so simulate_timeout_after_promote can rebuild the exact
        # on-disk state of the promote-to-marker window, which a completed
        # Stage 3 has already cleaned up.
        self._raw_snapshot = load_stage2_raw(self.output_dir, "ds", "img")

    def run_stage3(self) -> None:
        stage3_merge_measure_core(
            self.plan,
            self.output_dir,
            "ds",
            "img",
            self.output_manager,
            image_type="Image",
            work_id=self.work_id,
        )

    def simulate_timeout_after_promote(self) -> None:
        """Reproduce ``_cli_staged_workers.py``'s promote-to-marker window.

        At that point the store is promoted and the measurements written, the
        completion marker is not yet there, and the token and raw array are
        both still present.
        """
        assert self._raw_snapshot is not None, "run_stage2 first"
        remove_stage3_completion_marker(self.output_dir, "ds", "img")
        write_stage2_raw(self.output_dir, "ds", "img", self._raw_snapshot)
        write_stage2_token(
            self.output_dir,
            "ds",
            "img",
            objmap_shape=(
                int(self._raw_snapshot.shape[0]),
                int(self._raw_snapshot.shape[1]),
            ),
        )

    def read_measurements(self) -> "pl.DataFrame":
        return pl.read_parquet(
            dataset_measurements_dir(self.output_dir, "ds") / "img.parquet"
        )


def _build_stage_harness(tmp_path: Path, ops: list, *, work_id: str | None):
    image_path = tmp_path / "img.tiff"
    load_synth_yeast_plate().rgb.imsave(filepath=image_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir(exist_ok=True)
    pipeline = ImagePipeline(ops=ops, meas=[MeasureSize()])
    plan = split_pipeline_at_gpu(pipeline)
    output_manager = OutputManager.from_config(
        output_dir, ".tiff", save_overlays=False
    )
    output_manager.create_structure(
        [Dataset("ds", [image_path], tmp_path, output_dir)]
    )
    return StagedStageHarness(
        plan, image_path, output_dir, output_manager, work_id
    )


@pytest.fixture
def staged_run(tmp_path: Path) -> StagedStageHarness:
    """Baseline: detector emits three blobs, no post-ops, no work id."""
    return _build_stage_harness(
        tmp_path, [_FixedBlobDetector()], work_id=None
    )


@pytest.fixture
def staged_run_with_size_filter(tmp_path: Path) -> StagedStageHarness:
    """A post-op that removes exactly one of the detector's three blobs."""
    return _build_stage_harness(
        tmp_path,
        [_FixedBlobDetector(), SmallObjectRemover(min_size=100)],
        work_id=None,
    )


@pytest.fixture
def staged_run_with_work_id(tmp_path: Path) -> StagedStageHarness:
    """Exercises the guarded tail: with a work id Stage 3 publishes nothing."""
    return _build_stage_harness(
        tmp_path, [_FixedBlobDetector()], work_id="w-1"
    )


@pytest.fixture
def staged_run_with_border_colony(tmp_path: Path) -> StagedStageHarness:
    """A real colony provably touches the frame, so a replay has a victim.

    Without one, ``drop_frame_background`` returns early
    (``_objmap_accessor.py:503``) on the second pass and the D1 idempotency
    test would pass even with the defect present.
    """
    harness = _build_stage_harness(
        tmp_path, [_BorderColonyDetector()], work_id=None
    )
    # Assert the fixture's premise: after the first drop_frame_background a
    # non-zero label still reaches the border.
    labels = _BorderColonyDetector.frame_labels((600, 800))
    border = np.concatenate(
        [labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]
    )
    border = border[border > 0]
    values, counts = np.unique(border, return_counts=True)
    background = int(values[counts.argmax()])
    labels[labels == background] = 0
    survivors = np.concatenate(
        [labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]
    )
    assert (survivors > 0).any(), (
        "fixture premise broken: no colony touches the frame after the first "
        "drop_frame_background, so the D1 replay would be a no-op"
    )
    return harness


# ---------------------------------------------------------------------------
# ``--mode migrate`` fixtures (Phase 5)
# ---------------------------------------------------------------------------
#
# Imported rather than redefined: the session-scoped real run and the demotion
# helpers live beside the sdk_ suite that owns them, and promoting six
# migration-specific fixtures to the repo-root conftest would make them global
# to the whole suite.
from tests.unit.sdk_.conftest import (  # noqa: E402,F401
    _completed_run_two,
    finished_legacy_run,
)
