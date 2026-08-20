"""Shared fixtures for CLI unit tests.

Provides lightweight builders used by the process-only strategy / SLURM /
end-to-end tests: a one-level synthetic input tree, a serialized minimal
pipeline with one detector, and ``ExecutionConfig`` / ``OutputManager``
factories with sensible defaults overridable by keyword.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import h5py
import numpy as np
import pytest
from PIL import Image as PILImage

from phenotypic import Image
from phenotypic._cli._cli_completion import (
    SUCCESS_MARKER_VERSION,
    publish_image_success,
)
from phenotypic._cli._cli_output_manager import OutputManager
from phenotypic._cli._cli_stage2_token import (
    write_stage2_raw,
    write_stage2_token,
)
from phenotypic._cli._cli_staged_resume import write_stage3_completion_marker
from phenotypic._cli._cli_types import ExecutionConfig
from phenotypic.data import load_synth_yeast_plate
from phenotypic.prefab import RoundPeaksPipeline
from phenotypic.sdk_ import (
    atomic_write_json,
    dataset_measurements_dir,
    image_completion_marker_path,
    zarr_store_path,
)
from tests._legacy_staged_resume import (
    classify_staged_image as legacy_classify_staged_image,
)
from tests._legacy_staged_resume import legacy_hdf_path, legacy_sidecar_path


@pytest.fixture
def synth_one_level_input(tmp_path: Path) -> Path:
    """One-level input tree: ``<tmp>/in/day1/plateA.tif`` (one synth plate).

    Returns the input root (``<tmp>/in``) so callers can pass it as
    ``--input`` / ``input_path`` and assert on the mirrored output tree.
    """
    root = tmp_path / "in"
    day = root / "day1"
    day.mkdir(parents=True)
    grid_image = load_synth_yeast_plate()
    pil_img = PILImage.fromarray(grid_image.rgb[:].astype("uint8"))
    pil_img.save(day / "plateA.tif")
    return root


@pytest.fixture
def simple_pipeline_json() -> Path:
    """Write a minimal ``RoundPeaksPipeline`` (one detector) JSON to a temp file."""
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


@pytest.fixture
def make_exec_config() -> Callable[..., ExecutionConfig]:
    """Factory: build an ``ExecutionConfig`` with defaults overridable by kwargs."""

    def _build(
        *,
        pipeline_json: Path,
        input_path: Path,
        output_dir: Optional[Path] = None,
        image_type: str = "GridImage",
        nrows: Optional[int] = None,
        ncols: Optional[int] = None,
        bit_depth: Optional[int] = None,
        n_jobs: int = 1,
        slurm_args: Optional[Dict[str, Any]] = None,
        force_local: bool = True,
        wait: bool = False,
        ext: str = ".tiff",
        overlay_alpha: float = 0.3,
        include_dataset_column: bool = True,
        dry_run: bool = False,
        sample: Optional[int] = None,
        resume: bool = False,
        retry_failures: bool = False,
        skip_validation: bool = True,
        detect_mode: str = "gray",
        process_only_layer: Optional[str] = None,
        **overrides: Any,
    ) -> ExecutionConfig:
        return ExecutionConfig(
            pipeline_json=pipeline_json,
            input_path=input_path,
            output_dir=output_dir,
            image_type=image_type,  # type: ignore[arg-type]
            nrows=nrows,
            ncols=ncols,
            bit_depth=bit_depth,
            n_jobs=n_jobs,
            slurm_args=slurm_args if slurm_args is not None else {},
            force_local=force_local,
            wait=wait,
            ext=ext,
            overlay_alpha=overlay_alpha,
            include_dataset_column=include_dataset_column,
            dry_run=dry_run,
            sample=sample,
            resume=resume,
            retry_failures=retry_failures,
            skip_validation=skip_validation,
            detect_mode=detect_mode,
            process_only_layer=process_only_layer,  # type: ignore[arg-type]
            **overrides,
        )

    return _build


@pytest.fixture
def make_output_manager() -> Callable[..., OutputManager]:
    """Factory: build an ``OutputManager`` rooted at a given output dir."""

    def _build(output_dir: Path, **overrides: Any) -> OutputManager:
        return OutputManager.from_config(
            base_dir=output_dir,
            ext=overrides.pop("ext", ".tiff"),
            **overrides,
        )

    return _build


# ---------------------------------------------------------------------------
# Differential resume-parity worlds (Phase 3 Task 3.4)
# ---------------------------------------------------------------------------


class ArtifactWorld:
    """Build one image's durable artifacts, in one of the two storage formats.

    The five booleans are, in order,
    ``(image_state, stage2_signal, parquet, stage3_marker, image_success)`` --
    the axes ``classify_staged_image`` actually distinguishes. Everything
    format-neutral (parquet, Stage-3 marker, success marker) is built
    identically by both worlds, so any divergence the parity test reports is a
    divergence in the ported artifact probes and nothing else.
    """

    DATASET = "ds"
    STEM = "img"

    def __init__(self, base: Path, kind: str) -> None:
        self.base = base
        self.kind = kind
        self.root = base
        self._calls = 0

    def __call__(self, artifacts, *, work_id: str | None) -> Path:
        # A FRESH root per call: artifacts are only ever created, never
        # removed, so reusing one root would let an earlier combination's
        # files leak into a later one and silently turn the enumeration into a
        # monotonically growing superset.
        self.root = self.base / str(self._calls)
        self._calls += 1
        self.root.mkdir(parents=True)
        image_state, stage2_signal, parquet, stage3_marker, success = artifacts
        if image_state:
            self._write_image_state(work_id)
        if stage2_signal:
            self._write_stage2_signal()
        parquet_path = self._parquet_path()
        if parquet:
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            parquet_path.write_bytes(b"measurements")
        if stage3_marker:
            write_stage3_completion_marker(
                self.root, self.DATASET, f"{self.STEM}.tif", self.STEM
            )
        if success:
            self._write_success_marker(work_id, parquet_path)
        return self.root

    # -- format-specific halves --------------------------------------------

    def _write_image_state(self, work_id: str | None) -> None:
        if self.kind == "hdf":
            path = legacy_hdf_path(self.root, self.DATASET, self.STEM)
            path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(path, "w") as handle:
                handle.attrs["schema_version"] = 2
                if work_id is not None:
                    handle.attrs["phenotypic_work_id"] = work_id
                layers = handle.create_group("layers")
                for name in ("gray", "detect_mat", "objmap"):
                    layers.create_dataset(name, data=np.zeros((4, 4)))
            return
        store = zarr_store_path(self.root, self.DATASET, self.STEM)
        store.parent.mkdir(parents=True, exist_ok=True)
        Image(np.zeros((4, 4, 3), dtype=np.uint8)).save2zarr(
            store, work_id=work_id
        )

    def _write_stage2_signal(self) -> None:
        """The sidecar was BOTH the flag and Stage 3's input.

        Its store-world equivalent is therefore the token **and** the retained
        raw array, not the token alone: a token with no raw array is a state
        the HDF world cannot express, so it has no parity counterpart and is
        covered by a dedicated test in ``test_staged_resume.py`` instead.
        """
        if self.kind == "hdf":
            path = legacy_sidecar_path(self.root, self.DATASET, self.STEM)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, np.zeros((4, 4), dtype=np.uint16))
            return
        write_stage2_raw(
            self.root, self.DATASET, self.STEM, np.zeros((4, 4), dtype=np.uint16)
        )
        write_stage2_token(
            self.root, self.DATASET, self.STEM, objmap_shape=(4, 4)
        )

    # -- format-neutral halves ---------------------------------------------

    def _image_artifact_path(self) -> Path:
        """The per-image image-state artifact in this world's format."""
        if self.kind == "hdf":
            return legacy_hdf_path(self.root, self.DATASET, self.STEM)
        return zarr_store_path(self.root, self.DATASET, self.STEM)

    def _parquet_path(self) -> Path:
        return (
            dataset_measurements_dir(self.root, self.DATASET)
            / f"{self.STEM}.parquet"
        )

    def _write_success_marker(
        self, work_id: str | None, parquet_path: Path
    ) -> None:
        """Publish the per-image completion marker branch 1 consults.

        It describes the parquet **and**, when it exists, the per-image image
        artifact -- the ``.h5`` in the HDF world and the store *directory* in
        the zarr world. Task 3.8 gave descriptors a ``kind`` tag, so the two
        are now certifiable in the same marker shape and the comparison stays
        an honest one: if ``valid_image_success`` could not describe a store,
        the two worlds would disagree here and the parity test would fail.
        Before that tag existed this had to describe the parquet alone.
        """
        if parquet_path.is_file():
            artifacts = {"measurements": parquet_path}
            image_artifact = self._image_artifact_path()
            if image_artifact.exists():
                artifacts["image_state"] = image_artifact
            publish_image_success(
                self.root,
                work_id=work_id or "unused",
                dataset=self.DATASET,
                relative_image_path=f"{self.STEM}.tif",
                image_stem=self.STEM,
                mode="full",
                attempt_id="a-1",
                lifecycle_epoch="e-1",
                artifacts=artifacts,
            )
            return
        # No artifact to describe: write the marker anyway, pointing at the
        # parquet that is not there. valid_image_success then returns False --
        # identically in both worlds -- which is the "stale marker" case.
        atomic_write_json(
            image_completion_marker_path(self.root, self.DATASET, self.STEM),
            {
                "version": SUCCESS_MARKER_VERSION,
                "work_id": work_id or "unused",
                "dataset": self.DATASET,
                "relative_image_path": f"{self.STEM}.tif",
                "image_stem": self.STEM,
                "mode": "full",
                "attempt_id": "a-1",
                "lifecycle_epoch": "e-1",
                "artifacts": {
                    "measurements": {
                        "path": parquet_path.relative_to(self.root).as_posix(),
                        "size": 12,
                        "sha256": "0" * 64,
                    }
                },
                "completed_at": "2026-08-19T00:00:00.000+00:00",
            },
        )

    @staticmethod
    def classify(**kwargs) -> str:
        """Classify with the FROZEN pre-port HDF classifier."""
        return legacy_classify_staged_image(**kwargs)


@pytest.fixture
def hdf_world(tmp_path: Path) -> ArtifactWorld:
    """Builds the pre-port artifact set: staged ``.h5`` + ``.npy`` sidecar."""
    return ArtifactWorld(tmp_path / "hdf_out", "hdf")


@pytest.fixture
def zarr_world(tmp_path: Path) -> ArtifactWorld:
    """Builds the ported artifact set: OME-Zarr store + token + raw array."""
    return ArtifactWorld(tmp_path / "zarr_out", "zarr")


# ---------------------------------------------------------------------------
# ``--mode migrate`` fixtures (Phase 5)
# ---------------------------------------------------------------------------
#
# The builders and the session-scoped real run live beside the sdk_ suite that
# defines them; importing the fixtures here makes them visible to the CLI
# tests without promoting six migration-specific fixtures to the repo-root
# conftest, where they would be global to the whole suite.
from tests.unit.sdk_.conftest import (  # noqa: E402,F401
    _completed_run_one,
    _completed_run_two,
    finished_legacy_run,
    half_migrated_run,
    legacy_run,
    markerless_legacy_run,
    migrated_run,
)
from tests.unit.sdk_._migration_fixtures import (  # noqa: E402
    DATASET as _MIGRATION_DATASET,
)


@pytest.fixture
def legacy_format_run(legacy_run: Path) -> Path:  # noqa: F811
    """An output tree whose results are ``.h5`` and whose ``zarr/`` is absent.

    Distinct from ``legacy_headers_run`` below, and the distinction is the
    point (OPEN-QUESTIONS D16): ``recompile`` must **fail** on this one with a
    pointer to ``--mode migrate``, because the forward path cannot read its
    images at all.
    """
    return legacy_run


@pytest.fixture
def legacy_headers_run(
    _completed_run_two: Path,  # noqa: F811
    tmp_path: Path,
) -> Path:
    """A tree already converted to stores, whose metadata headers are legacy.

    ``recompile`` must **succeed** on this one, read those headers, and not
    rewrite them -- flat-metadata decision #3 (permanent stored-data
    compatibility) is untouched by Task 5.4.

    Task 5.5's cut re-based this fixture on a ``.h5``-backed tree, so the
    legacy headers live in the per-dataset measurement parquets rather than in
    a store: ingest canonicalizes on read, so a store cannot hold one.
    """
    import shutil

    from tests.unit.sdk_._migration_fixtures import (
        make_measurement_headers_legacy,
        refresh_marker_descriptors,
        run_stems,
    )

    output_dir = tmp_path / "legacy_headers"
    shutil.copytree(_completed_run_two, output_dir)
    make_measurement_headers_legacy(output_dir, _MIGRATION_DATASET)
    # The parquets the run's markers bound have just changed. A real tree's
    # markers bound the bytes that existed when they were published, so
    # re-fingerprint them -- otherwise recompile fails on a stale digest,
    # which is not what these tests are about.
    for stem in run_stems(output_dir):
        refresh_marker_descriptors(output_dir, _MIGRATION_DATASET, stem)
    return output_dir
