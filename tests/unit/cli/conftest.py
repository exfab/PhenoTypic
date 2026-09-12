"""Shared fixtures for CLI unit tests.

Provides lightweight builders used by the process-only strategy / SLURM /
end-to-end tests: a one-level synthetic input tree, a serialized minimal
pipeline with one detector, and ``ExecutionConfig`` / ``OutputManager``
factories with sensible defaults overridable by keyword.
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import h5py
import numpy as np
import pandas as pd
import polars as pl
import pytest
from PIL import Image as PILImage

from phenotypic import Image
from phenotypic._cli._cli_completion import publish_image_success
from phenotypic._cli._cli_output_manager import OutputManager
from phenotypic._cli._cli_stage2_token import (
    write_stage2_raw,
    write_stage2_token,
)
from phenotypic._cli._cli_staged_resume import write_stage3_completion_marker
from phenotypic._cli._cli_state_management import save_processing_state
from phenotypic._cli._cli_types import (
    DatasetState,
    ExecutionConfig,
    ProcessingState,
)
from phenotypic.data import load_synth_yeast_plate
from phenotypic.prefab import RoundPeaksPipeline
from phenotypic.sdk_ import (
    ARTIFACT_KIND_FILE,
    MEASUREMENT_TABLE_RELATIVE_PATH,
    atomic_write_json,
    dataset_measurements_dir,
    image_record_path,
    master_measurements_parquet_path,
    metadata_csv_deliverable_path,
    zarr_store_path,
)
from phenotypic.sdk_._image_record import (
    RECORD_VERSION,
    read_image_record,
    record_rejection,
)
from tests._legacy_staged_resume import (
    classify_staged_image as legacy_classify_staged_image,
)
from tests._legacy_staged_resume import (
    legacy_hdf_path,
    legacy_sidecar_path,
    legacy_stage3_marker_path,
)


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
    format-neutral is built identically by both worlds, so any divergence the
    parity test reports is a divergence in the ported artifact probes and
    nothing else.

    **The Stage-3 marker left that neutral set at P3 §6.1** and is now a
    format-specific half; see :meth:`_write_stage3_marker`. The parquet and the
    success marker remain neutral, the latter because both classifiers call the
    *live* ``valid_image_success`` -- the frozen module imports it rather than
    freezing it, precisely so a change there moves both worlds together and
    parity keeps meaning what it says.
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
            self._write_stage3_marker()
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

    def _write_stage3_marker(self) -> None:
        """The stage-3 fact, in whichever form this world's classifier reads.

        **This became a format-specific half at P3 §6.1**, and the class
        docstring's "everything format-neutral … is built identically by both
        worlds" no longer covers it. The live writer moved the fact out of
        ``stage3_complete/<ds>/<stem>.json`` and into ``stages.stage3`` of the
        per-image record; the frozen classifier reads the old path and, being
        frozen, always will.

        So calling the live writer for both worlds -- which is what this did --
        would leave the HDF side blind on the ``stage3_marker`` axis and make
        the parity test fail on every cell where that axis changes the frozen
        answer. That failure would look like a defect in the port and would in
        fact be a fixture pointing at a path that moved.
        """
        if self.kind == "hdf":
            path = legacy_stage3_marker_path(
                self.root, self.DATASET, self.STEM
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
            return
        write_stage3_completion_marker(
            self.root, self.DATASET, f"{self.STEM}.tif", self.STEM
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
            self.root,
            self.DATASET,
            self.STEM,
            np.zeros((4, 4), dtype=np.uint16),
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
        if self.kind == "zarr":
            from phenotypic.sdk_ import MEASUREMENT_TABLE_RELATIVE_PATH

            return (
                zarr_store_path(self.root, self.DATASET, self.STEM)
                / MEASUREMENT_TABLE_RELATIVE_PATH
            )
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
            image_artifact_exists = (
                image_artifact.is_file()
                if self.kind == "hdf"
                else (image_artifact / "zarr.json").is_file()
            )
            if image_artifact_exists:
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
        # No artifact to describe: write the RECORD anyway, describing the
        # parquet that is not there. `valid_image_success` then returns False
        # -- identically in both worlds -- which is the "stale record" case.
        #
        # **This branch wrote an `image_complete/` marker until P3's clean
        # break, and the break silently hollowed it out.** `valid_image_success`
        # moved to `images/`, so the legacy marker stopped being read: the
        # verdict stayed False, but because the record was ABSENT rather than
        # stale, and the stale path went unexercised. Nothing reported it,
        # because a parity test comparing False to False passes.
        #
        # The cost was exactly half the suite. `success` and its negation both
        # produced "no record" wherever `parquet=False`, so the axis was inert
        # in 16 of the 32 `ARTIFACTS` combinations -- **192 of the parity
        # suite's 384 cases, at full green**. That is the failure
        # `test_staged_resume_parity.py:26-32` was written to prevent, on this
        # same axis, and a source change three files away undid half of it
        # without touching a line of that test.
        identity = work_id or "unused"
        # **`stages` is merged, never replaced** -- CAN-6 rule 1, which the
        # real `publish_image_record` obeys and which this hand-written path
        # has to obey for the same reason. `__call__` writes the stage-3 entry
        # BEFORE this branch runs, so a replacing write drops it, and the image
        # then looks unprocessed: `stage3_completion_exists` reads the record
        # now, so clobbering `stages` un-marks Stage 3 rather than merely
        # losing a field. That is one file's worth of the exact lost-update
        # `publish_image_record`'s docstring warns about, reproduced in a
        # fixture -- and it diverged only in the zarr world, because the HDF
        # world's stage-3 half writes a separate legacy file this cannot
        # touch.
        existing = read_image_record(self.root, self.DATASET, self.STEM) or {}
        carried = existing.get("stages")
        record = {
            "version": RECORD_VERSION,
            "work_id": identity,
            "dataset": self.DATASET,
            "image_stem": self.STEM,
            "relative_image_path": f"{self.STEM}.tif",
            "mode": "full",
            "stages": dict(carried) if isinstance(carried, dict) else {},
            "artifacts": {
                "measurements": {
                    "path": parquet_path.relative_to(self.root).as_posix(),
                    "kind": ARTIFACT_KIND_FILE,
                    "size": 12,
                    "sha256": "0" * 64,
                }
            },
            "attempt_id": "a-1",
            "lifecycle_epoch": "e-1",
            "completed_at": "2026-08-19T00:00:00.000+00:00",
        }
        # `is None` is the ASSERTION, not an oversight, and it is the one that
        # keeps this branch honest. It says every identity and shape clause
        # passes -- version, dataset, image_stem, work_id, artifacts non-empty
        # -- so the only thing left that can reject this record is the file it
        # names being absent. If a later change makes the shape invalid for
        # some new reason, `record_rejection` returns a sentence and this fails
        # loudly, instead of the branch quietly reverting to "rejected for the
        # wrong reason" and looking fixed.
        #
        # Asserting on a reason *string* is impossible here and that is by
        # design: `record_rejection` checks identity and shape only, leaving
        # artifact contents to `fenced_artifact_path`, whose `None` covers
        # malformed, escapes-the-root and missing-on-disk alike. Neither
        # function names staleness, so the mechanism is isolated by
        # construction instead -- this clause plus the positive control in
        # `test_staged_resume_parity.py`.
        assert (
            record_rejection(
                record,
                work_id=identity,
                dataset=self.DATASET,
                image_stem=self.STEM,
            )
            is None
        )
        atomic_write_json(
            image_record_path(self.root, self.DATASET, self.STEM), record
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
    """Stores plus legacy external tables, requiring explicit migration.

    This is the only legitimate source of legacy headers after embedded tables
    became the current schema: a store without a table descriptor beside the
    external Parquet authority written by older releases.
    """
    import json
    import shutil

    from phenotypic.sdk_ import MEASUREMENT_TABLE_RELATIVE_PATH
    from tests.unit.sdk_._migration_fixtures import (
        make_measurement_headers_legacy,
        refresh_marker_descriptors,
        run_stems,
    )

    output_dir = tmp_path / "legacy_headers"
    shutil.copytree(_completed_run_two, output_dir)
    for stem in run_stems(output_dir):
        store = zarr_store_path(output_dir, _MIGRATION_DATASET, stem)
        embedded = store / MEASUREMENT_TABLE_RELATIVE_PATH
        external = (
            dataset_measurements_dir(output_dir, _MIGRATION_DATASET)
            / f"{stem}.parquet"
        )
        external.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(embedded, external)
        shutil.rmtree(store / "tables")
        root_path = store / "zarr.json"
        root = json.loads(root_path.read_text(encoding="utf-8"))
        root["attributes"]["phenotypic"].pop("tables", None)
        atomic_write_json(root_path, root)

        # The RECORD, on both the read and the write. This fixture degrades a
        # tree the CURRENT publisher wrote -- only the measurement table's
        # location and headers are made legacy -- so the thing binding that
        # table is the record P3 moved to `images/`, not `image_complete/`.
        # `refresh_marker_descriptors` two lines below already reads the
        # record, and a fixture that repoints one file and re-fingerprints a
        # different one is incoherent whichever shape you take it for.
        record_path = image_record_path(output_dir, _MIGRATION_DATASET, stem)
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record["artifacts"]["measurements"]["path"] = external.relative_to(
            output_dir
        ).as_posix()
        atomic_write_json(record_path, record)

    make_measurement_headers_legacy(output_dir, _MIGRATION_DATASET)
    for stem in run_stems(output_dir):
        refresh_marker_descriptors(output_dir, _MIGRATION_DATASET, stem)
    return output_dir


@pytest.fixture
def stub_run_identity():
    """A throwaway ``RunIdentity`` for tests that do not exercise identity.

    ``create_initial_state`` takes the identity as a **required** keyword from
    P2 Task 3 onward, because an optional one would let a caller silently fall
    back to the ``uuid4().hex`` the change exists to remove. Tests about
    manifest digests or state shape still have to supply one, and supplying a
    recognisable stub is clearer than minting a real identity they then ignore
    -- a minted one would also bump a restart epoch as a side effect.
    """
    from phenotypic.sdk_ import RunIdentity

    return RunIdentity(
        processing_generation="stub-generation",
        restart_epoch=0,
        scheduler_epoch=None,
        owner_generation=None,
        inventory_digest="stub-inventory",
        scientific_config_digest="stub-pipeline",
        finalization_input_digest="stub-finalization",
    )


# ---------------------------------------------------------------------------
# Finalization fixture machinery -- real stores, real records, real state
# ---------------------------------------------------------------------------
#
# Promoted verbatim from ``test_finalize_run.py``, which is where they were
# written and where their standing rule comes from:
#
#     Every fixture here builds REAL stores through the forward writer
#     (``OutputManager.save_image_store`` -> ``prepare_image_tables`` ->
#     ``promote_store``), publishes a real per-image record, and aggregates
#     them. Nothing hands ``finalize_run`` a literal DataFrame.
#
# **That rule is why these live here rather than being copied.** P5 splits
# finalization across two suites -- ``test_finalize_run.py`` and
# ``test_finalize_fanout.py`` -- and a duplicated-then-simplified copy is
# exactly how "real store" quietly degrades into "hand-built frame" in one of
# them. The pre-P4 tests that *appeared* to cover post-master finalization
# failed that way: they fed hand-built frames and would have passed on day one
# against code that never aggregated an inverted store.
#
# Imported by name (``from .conftest import _publish_successful_images``),
# following the convention already used across ``tests/unit/gui/builder/``,
# ``tests/integration/gui/builder/`` and ``tests/e2e/gui/``. They are helpers,
# not fixtures: several take arguments and are called mid-test.

DATASET = "plate"

_WELLS = {"a": ["A01", "A02"], "b": ["B01", "B02"], "c": ["C01", "C02"]}

_WELL_INTS = {"A01": 1, "A02": 2, "B01": 3, "B02": 4, "C01": 5, "C02": 6}

def _image(stem: str):
    """One 8x8 RGB image carrying two labelled objects."""
    from phenotypic import Image

    image = Image(np.zeros((8, 8, 3), dtype=np.uint8), name=stem)
    objmap = np.zeros((8, 8), dtype=np.uint16)
    objmap[1:3, 1:3] = 1
    objmap[1:3, 5:7] = 2
    image.objmap[:] = objmap
    return image

def _manager(output_dir: Path):
    from phenotypic._cli._cli_output_manager import OutputManager

    return OutputManager.from_config(
        output_dir,
        ext=".tiff",
        include_dataset_column=True,
        save_overlays=False,
    )

def _measurements(
    stem: str,
    *,
    extra_objects: list[tuple[int, str]] | None = None,
    well_dtype: object = pl.String,
    extra_columns: list[str] | None = None,
    include_well: bool = True,
) -> pd.DataFrame:
    """One image's baseline measurements.

    ``include_well=False`` drops ``Metadata_Well``, leaving a baseline whose
    only metadata is ``IMAGE``-owned -- the shape a forward pipeline actually
    emits. The tests that ask what the master's metadata NAMESPACE contains
    need that shape; the ones that need a per-object join key keep the well.
    """
    labels = [1, 2]
    wells: list[object] = list(_WELLS[stem])
    areas = [4.0, 4.0]
    for label, well in extra_objects or []:
        labels.append(label)
        wells.append(well)
        areas.append(4.0)
    if well_dtype is not pl.String:
        # DISTINCT ints across images. Naively stripping the row letter would
        # map A01/B01 to the same key, making the snapshot's keys duplicates
        # and fanning every measured row out -- a different behaviour from the
        # dtype cast this option exists to exercise.
        wells = [_WELL_INTS[str(well)] for well in wells]
    columns: dict[str, object] = {}
    if include_well:
        columns["Metadata_Well"] = wells
    columns["Shape_Area"] = areas
    columns["Metadata_ImageName"] = [f"{stem}.tiff"] * len(labels)
    columns["Object_Label"] = labels
    frame = pd.DataFrame(columns)
    for column in extra_columns or []:
        frame[column] = list(range(len(labels)))
    return frame

def _install_snapshot(output_dir: Path, text: str | None) -> Path:
    """Write (or remove) ``deliverables/metadata.csv`` -- the run's snapshot."""
    path = metadata_csv_deliverable_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    if text is None:
        path.unlink(missing_ok=True)
    else:
        path.write_text(text, encoding="utf-8")
    return path

def _publish_store(
    output_dir: Path, stem: str, measurements: pd.DataFrame
) -> Path:
    """Promote one real store and publish its per-image record."""
    store = _manager(output_dir).save_image_store(
        _image(stem),
        DATASET,
        stem,
        work_id=f"work-{stem}",
        measurements=measurements,
    )
    assert store is not None, f"the forward writer failed to promote {stem}"
    publish_image_success(
        output_dir,
        work_id=f"work-{stem}",
        dataset=DATASET,
        relative_image_path=f"{stem}.tiff",
        image_stem=stem,
        mode="full",
        attempt_id="attempt-1",
        lifecycle_epoch="epoch-1",
        artifacts={
            "measurements": store / MEASUREMENT_TABLE_RELATIVE_PATH,
            "store": store,
        },
    )
    return store

def _install_overlays(output_dir: Path, stems: list[str]) -> None:
    """Write the overlay PNG each store's record does NOT declare.

    ``_publish_store`` publishes ``measurements`` + ``store`` and no overlay,
    which is a legitimate forward shape (``--no-save-overlays``). But
    ``_handle_recompile`` opens with ``_regenerate_missing_overlays``, and
    that pass refuses -- ``Cannot safely restore marker authority for
    <ds>/<stem>`` -- when an overlay is absent, the marker does not bind one,
    and the store's table exists. It is right to refuse: it cannot re-render
    an overlay and re-certify the image without authority binding the two.

    So a fixture that wants to drive the real recompile has to look like a
    run that HAS its overlays. Writing the file at the canonical path is
    enough: ``discover_missing_overlays`` asks ``is_file()`` and nothing
    else, so the pass reports "all overlays present" and returns before the
    refusal. The bytes are never read.

    Overlays live under ``deliverables/``, so this changes nothing under
    ``results/`` and nothing the master is built from.
    """
    from phenotypic._cli._cli_overlay_rendering import overlay_output_manager

    manager = overlay_output_manager(output_dir, overlay_alpha=0.3)
    for stem in stems:
        overlay = manager.get_output_path(DATASET, "overlays", stem)
        overlay.parent.mkdir(parents=True, exist_ok=True)
        overlay.write_bytes(b"\x89PNG\r\n\x1a\n")


def _install_state(output_dir: Path, stems: list[str]) -> None:
    """Install the processing state that makes the records authoritative."""
    now = datetime.now()
    save_processing_state(
        ProcessingState(
            version="3.0.0",
            pipeline_path=output_dir / "pipeline.json",
            input_path=output_dir / "input",
            output_dir=output_dir,
            timestamp=now,
            execution_mode="local",
            last_updated=now,
            datasets={
                DATASET: DatasetState(
                    initial_images={f"{stem}.tiff" for stem in stems}
                )
            },
            config={
                "success_markers_required": True,
                "work_ids": {
                    DATASET: {
                        f"{stem}.tiff": f"work-{stem}" for stem in stems
                    }
                },
                "processing_generation": "generation",
                "pipeline_sha256": "pipeline",
            },
        ),
        output_dir,
    )

def _publish_successful_images(
    tmp_path: Path,
    *,
    stems: list[str] | None = None,
    snapshot: str | None = None,
    **measurement_kwargs: object,
) -> list[Path]:
    """The workhorse: N real stores, N real records, one processing state."""
    stems = stems or ["a", "b"]
    _install_snapshot(tmp_path, snapshot)
    stores = [
        _publish_store(
            tmp_path,
            stem,
            _measurements(stem, **measurement_kwargs),  # type: ignore[arg-type]
        )
        for stem in stems
    ]
    _install_state(tmp_path, stems)
    return stores

def _remeasure_every_store(output_dir: Path, stems: list[str]) -> None:
    """Drive the ``--mode measure`` shape: rewrite each store, republish.

    ``replace_image_store_measurements`` is the real measure-path writer
    (``_cli_process_single.py`` calls it), and it re-promotes the store, so
    every artifact digest the record certifies changes and the record has to
    be republished exactly as ``_republish_table_marker`` does.
    """
    manager = _manager(output_dir)
    for stem in stems:
        store = zarr_store_path(output_dir, DATASET, stem)
        manager.replace_image_store_measurements(
            store, _measurements(stem), DATASET
        )
        publish_image_success(
            output_dir,
            work_id=f"work-{stem}",
            dataset=DATASET,
            relative_image_path=f"{stem}.tiff",
            image_stem=stem,
            mode="measure",
            attempt_id="attempt-1",
            lifecycle_epoch="epoch-1",
            artifacts={
                "measurements": store / MEASUREMENT_TABLE_RELATIVE_PATH,
                "store": store,
            },
        )

def _run_mode(output_dir: Path, mode: str) -> Path:
    """Build a two-image tree and finalize it the way *mode* does.

    Every mode reaches finalization through ``aggregate_measurements``, which
    is ``finalize_run`` under the publication lock -- that is the property
    under test. What differs is what happened to the stores first: ``full``
    promotes them and ``measure`` rewrites their tables in place.

    **The ``recompile`` arm drives ``_handle_recompile``, not
    ``aggregate_measurements``.** Until 2026-09-11 it re-derived each table
    from the store's own baseline and this helper asserted every store was
    rewritten; recompile stopped rewriting stores outright, and simply
    deleting that line would have left the arm as `full`'s tree finalized a
    second time by the same call -- an equality holding by construction, in
    the one file whose subject is *"there is one FINAL master writer"*. So
    the arm now runs the mode's own entry point end to end: overlay pass,
    authority abort, aggregation, completion evidence, manifest, dashboard.
    What the comparison then asserts is that the mode's whole path still
    lands on the same master bytes as ``full``'s, which is the claim worth
    making and the one a deleted assertion would have quietly stopped making.

    Overlays are installed first because ``_handle_recompile`` begins with
    ``_regenerate_missing_overlays`` -- see :func:`_install_overlays`. They
    are installed for **every** mode, not just this one, so the tree the
    three arms compare is identical; they live under ``deliverables/`` and
    the master is not built from them.

    **No metadata snapshot**, deliberately. Post-D8 the master carries no user
    metadata at all, so a snapshot could only change the mirror; running
    without one keeps the comparison about the thing being compared.
    """
    from phenotypic.phenotypicCLI import _handle_recompile
    from phenotypic._cli._cli_output_manager import aggregate_measurements

    stems = ["a", "b"]
    output_dir.mkdir(parents=True, exist_ok=True)
    _publish_successful_images(output_dir, stems=stems)
    _install_overlays(output_dir, stems)

    if mode == "recompile":
        # The mode's own entry point, `--mode recompile`'s whole local path.
        _handle_recompile(output_dir, None, True, 0.3, 1)
        return output_dir

    if mode == "measure":
        _remeasure_every_store(output_dir, stems)
    elif mode != "full":  # pragma: no cover - guards a typo in a param id
        raise AssertionError(f"unknown mode {mode!r}")

    aggregate_measurements(
        output_dir=output_dir,
        dataset_names=[DATASET],
        include_dataset_column=True,
    )
    return output_dir

def _master_bytes(output_dir: Path) -> bytes:
    return master_measurements_parquet_path(output_dir).read_bytes()
