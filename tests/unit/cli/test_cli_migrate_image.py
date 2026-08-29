"""One explicit manifest image migrates idempotently under publication fences."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
from PIL import Image as PILImage
import pytest

from phenotypic._cli._cli_completion import (
    publish_image_success,
    valid_image_success,
)
from phenotypic._cli._cli_migrate_image import (
    _existing_marker_identity,
    _source_artifact_state,
    migrate_image_task,
)
from phenotypic._cli._cli_migrate_manifest import MigrationImageTask
from phenotypic._cli._cli_slurm_lifecycle import initialize_slurm_lifecycle
from phenotypic._cli._cli_overlay_rendering import overlay_output_manager
from phenotypic._cli._cli_overlay_rendering import valid_migration_overlay
from phenotypic._cli._cli_state_management import save_processing_state
from phenotypic._cli._cli_types import DatasetState, ProcessingState
from phenotypic._cli._embedded_measurement_tables import (
    prepare_embedded_measurement_table,
)
from phenotypic.sdk_ import (
    MEASUREMENT_TABLE_RELATIVE_PATH,
    image_completion_marker_path,
    load_image_from_store,
    replace_embedded_measurement_table,
)
from phenotypic.sdk_._hdf_to_zarr import migrate_hdf_to_zarr
from phenotypic.sdk_.ngff_ import valid_staged_store


_HDF_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "legacy_hdf"
    / "v2_grouped"
    / "img.h5"
)


def _migration_task(
    tmp_path: Path,
    *,
    with_measurements: bool = True,
    zero_objects: bool = False,
) -> tuple[Path, MigrationImageTask]:
    """Build one canonical manifest task without running migration discovery."""
    output_dir = tmp_path / "out"
    hdf_path = output_dir / "results" / "ds" / "hdf" / "img.h5"
    hdf_path.parent.mkdir(parents=True)
    shutil.copy2(_HDF_FIXTURE, hdf_path)
    if zero_objects:
        with h5py.File(hdf_path, mode="a") as handle:
            handle["layers/objmap"][:] = 0

    measurement_path: Path | None = None
    if with_measurements:
        measurement_path = (
            output_dir / "results" / "ds" / "measurements" / "img.parquet"
        )
        measurement_path.parent.mkdir(parents=True)
        pd.DataFrame(
            {
                "Object_Label": [1],
                "Size_Area": [25.0],
                "Metadata_ImageName": ["img"],
            }
        ).to_parquet(measurement_path, index=False)

    return output_dir, MigrationImageTask(
        index=7,
        dataset="ds",
        stem="img",
        hdf_path=hdf_path,
        store_path=output_dir / "results" / "ds" / "zarr" / "img.ome.zarr",
        measurement_path=measurement_path,
        overlay_path=output_dir / "deliverables" / "overlays" / "ds" / "img.png",
        marker_path=image_completion_marker_path(output_dir, "ds", "img"),
    )


def test_source_artifact_state_refuses_directory_as_absence(
    tmp_path: Path,
) -> None:
    """Only a missing source file is authoritative absence."""
    source = tmp_path / "source.h5"
    source.mkdir()

    with pytest.raises(IsADirectoryError):
        _source_artifact_state(source)


def test_migration_marker_identity_uses_active_generation_on_slurm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local migrate inside an allocation publishes with its owned generation."""
    output_dir = tmp_path / "out"
    initialize_slurm_lifecycle(
        output_dir,
        generation="migration-generation",
        mode="migrate",
    )
    monkeypatch.setenv("SLURM_JOB_ID", "12345")

    identity = _existing_marker_identity(
        output_dir,
        "ds",
        "img",
        "work-id",
    )

    assert identity["lifecycle_epoch"] == "migration-generation"


def test_table_failure_preserves_completed_conversion_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A table-stage exception retains the already-completed conversion."""
    import phenotypic._cli._cli_migrate_image as migrate_module

    output_dir, task = _migration_task(tmp_path)
    validity = iter((False, True))
    monkeypatch.setattr(
        migrate_module, "valid_staged_store", lambda _path: next(validity)
    )
    monkeypatch.setattr(
        migrate_module,
        "migrate_hdf_to_zarr",
        lambda *_args, **_kwargs: task.store_path,
    )
    monkeypatch.setattr(
        migrate_module,
        "load_image_from_store",
        lambda _path: SimpleNamespace(num_objects=1),
    )

    def fail_table(*_args: object, **_kwargs: object) -> tuple[bool, bool]:
        raise OSError("table write failed")

    monkeypatch.setattr(migrate_module, "_table_state", fail_table)

    with pytest.raises(migrate_module.MigrationImageStageError) as caught:
        migrate_module.migrate_image_task(
            output_dir,
            task,
            metadata_csv=None,
            overlay_alpha=0.3,
            dry_run=False,
        )

    failure = caught.value.failure
    assert failure.stage == "table"
    assert failure.target == task.measurement_path
    assert failure.partial.converted is True
    assert failure.partial.table_installed is False
    assert failure.partial.overlay_rendered is False


def test_overlay_failure_preserves_conversion_and_table_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An overlay exception retains completed conversion and table evidence."""
    import phenotypic._cli._cli_migrate_image as migrate_module

    output_dir, task = _migration_task(tmp_path)
    validity = iter((False, True))
    monkeypatch.setattr(
        migrate_module, "valid_staged_store", lambda _path: next(validity)
    )
    monkeypatch.setattr(
        migrate_module,
        "migrate_hdf_to_zarr",
        lambda *_args, **_kwargs: task.store_path,
    )
    monkeypatch.setattr(
        migrate_module,
        "load_image_from_store",
        lambda _path: SimpleNamespace(num_objects=1),
    )
    monkeypatch.setattr(
        migrate_module,
        "_table_state",
        lambda *_args, **_kwargs: (True, True),
    )
    monkeypatch.setattr(migrate_module, "_image_plane_shape", lambda _image: (5, 5))
    monkeypatch.setattr(
        migrate_module,
        "valid_migration_overlay",
        lambda *_args, **_kwargs: False,
    )

    class FailingOverlayManager:
        def save_overlay(self, *_args: object, **_kwargs: object) -> Path:
            raise OSError("overlay write failed")

    monkeypatch.setattr(
        migrate_module,
        "overlay_output_manager",
        lambda *_args, **_kwargs: FailingOverlayManager(),
    )

    with pytest.raises(migrate_module.MigrationImageStageError) as caught:
        migrate_module.migrate_image_task(
            output_dir,
            task,
            metadata_csv=None,
            overlay_alpha=0.3,
            dry_run=False,
        )

    failure = caught.value.failure
    assert failure.stage == "overlay"
    assert failure.target == task.overlay_path
    assert failure.partial.converted is True
    assert failure.partial.table_installed is True
    assert failure.partial.overlay_rendered is False


def _install_store(task: MigrationImageTask) -> None:
    """Install only the converted image store for a task."""
    assert task.hdf_path is not None
    migrate_hdf_to_zarr(task.hdf_path, task.store_path)


def _install_table(task: MigrationImageTask) -> None:
    """Install only the real prepared embedded table for a task."""
    assert task.measurement_path is not None
    prepared = prepare_embedded_measurement_table(
        pd.read_parquet(task.measurement_path), None
    )
    replace_embedded_measurement_table(task.store_path, prepared)


def _install_overlay(output_dir: Path, task: MigrationImageTask) -> None:
    """Render only the canonical overlay for a task."""
    manager = overlay_output_manager(output_dir, overlay_alpha=0.3)
    image = load_image_from_store(task.store_path)
    manager.save_overlay(image, task.dataset, task.stem)


def _complete_image(
    output_dir: Path,
    task: MigrationImageTask,
    *,
    metadata_csv: Path | None = None,
):
    """Complete one image through the public task primitive."""
    return migrate_image_task(
        output_dir,
        task,
        metadata_csv=metadata_csv,
        overlay_alpha=0.3,
        dry_run=False,
    )


def _overlay_size(task: MigrationImageTask) -> tuple[int, int]:
    """Return the fully decoded overlay dimensions."""
    with PILImage.open(task.overlay_path) as image:
        image.load()
        return image.size


def _install_fast_migrated_artifacts(
    task: MigrationImageTask,
) -> SimpleNamespace:
    """Install marker-describable artifacts without invoking Zarr's sync bridge."""
    task.store_path.mkdir(parents=True)
    (task.store_path / "zarr.json").write_bytes(b'{"zarr_format": 3}')
    embedded = task.store_path / MEASUREMENT_TABLE_RELATIVE_PATH
    embedded.parent.mkdir(parents=True)
    embedded.write_bytes(b"embedded measurements")
    task.overlay_path.parent.mkdir(parents=True)
    PILImage.new("RGB", (6, 4), "red").save(task.overlay_path, format="PNG")
    return SimpleNamespace(
        gray=np.zeros((4, 6), dtype=np.uint8),
        num_objects=1,
    )


def _save_canonical_work_id(output_dir: Path, work_id: str) -> None:
    """Persist one processing-state-authorized work ID for the fast fixture."""
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
            datasets={"ds": DatasetState(initial_images={"img.tif"})},
            config={
                "success_markers_required": True,
                "work_ids": {"ds": {"img.tif": work_id}},
            },
        ),
        output_dir,
    )


def _patch_fast_migrated_reads(
    monkeypatch: pytest.MonkeyPatch,
    task: MigrationImageTask,
    image: SimpleNamespace,
) -> None:
    """Bypass only Zarr decoding while exercising the real marker state machine."""
    from phenotypic._cli import _cli_migrate_image

    monkeypatch.setattr(
        _cli_migrate_image,
        "valid_staged_store",
        lambda path: path == task.store_path,
    )
    monkeypatch.setattr(
        _cli_migrate_image,
        "_valid_embedded_measurement_contract",
        lambda path: path == task.store_path,
    )
    monkeypatch.setattr(
        _cli_migrate_image,
        "embedded_measurement_table_matches",
        lambda path, _prepared: path == task.store_path,
    )
    monkeypatch.setattr(
        _cli_migrate_image,
        "load_image_from_store",
        lambda path: image if path == task.store_path else None,
    )


@pytest.mark.parametrize(
    ("payload", "expected"),
    [("png", True), ("jpeg", False), ("wrong-size", False), ("corrupt", False)],
)
def test_migration_overlay_requires_verified_full_plane_png(
    tmp_path: Path, payload: str, expected: bool
) -> None:
    """Overlay validity includes format, full decode, nonzero size, and plane size."""
    path = tmp_path / "overlay.png"
    if payload == "png":
        PILImage.new("RGB", (6, 4), "red").save(path, format="PNG")
    elif payload == "jpeg":
        PILImage.new("RGB", (6, 4), "red").save(path, format="JPEG")
    elif payload == "wrong-size":
        PILImage.new("RGB", (1, 2), "red").save(path, format="PNG")
    else:
        path.write_bytes(b"not an image")

    assert valid_migration_overlay(path, (4, 6)) is expected


def test_hdf_only_zero_object_image_completes_without_inventing_a_table(
    tmp_path: Path,
) -> None:
    """A missing external table is valid only for an empty object map."""
    output_dir, task = _migration_task(
        tmp_path, with_measurements=False, zero_objects=True
    )

    result = _complete_image(output_dir, task)

    assert result.index == 7
    assert result.converted is True
    assert result.table_installed is False
    assert result.overlay_rendered is True
    assert result.skipped is False
    assert len(result.marker_digest) == 64
    assert not (task.store_path / MEASUREMENT_TABLE_RELATIVE_PATH).exists()
    marker = json.loads(task.marker_path.read_text(encoding="utf-8"))
    assert set(marker["artifacts"]) == {"overlay", "store"}
    assert valid_image_success(
        output_dir,
        dataset=task.dataset,
        image_stem=task.stem,
        work_id=result.work_id,
    )


def test_legacy_only_marker_is_republished_with_complete_migrated_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Generic legacy evidence cannot certify the completed migrated task."""
    output_dir, task = _migration_task(tmp_path)
    image = _install_fast_migrated_artifacts(task)
    _save_canonical_work_id(output_dir, "canonical-work-id")
    assert task.hdf_path is not None
    publish_image_success(
        output_dir,
        work_id="canonical-work-id",
        dataset=task.dataset,
        relative_image_path="ds/img.tif",
        image_stem=task.stem,
        mode="full",
        attempt_id="legacy-attempt",
        lifecycle_epoch="legacy-epoch",
        artifacts={"hdf": task.hdf_path},
    )
    assert valid_image_success(
        output_dir,
        dataset=task.dataset,
        image_stem=task.stem,
        work_id="canonical-work-id",
    )
    _patch_fast_migrated_reads(monkeypatch, task, image)

    result = migrate_image_task(
        output_dir,
        task,
        metadata_csv=None,
        overlay_alpha=0.3,
        dry_run=False,
    )

    marker = json.loads(task.marker_path.read_text(encoding="utf-8"))
    assert result.skipped is False
    assert set(marker["artifacts"]) == {"store", "overlay", "measurements"}
    assert marker["artifacts"]["store"]["path"] == (
        task.store_path.relative_to(output_dir).as_posix()
    )
    assert marker["artifacts"]["overlay"]["path"] == (
        task.overlay_path.relative_to(output_dir).as_posix()
    )
    assert marker["artifacts"]["measurements"]["path"] == (
        (task.store_path / MEASUREMENT_TABLE_RELATIVE_PATH)
        .relative_to(output_dir)
        .as_posix()
    )


def test_valid_store_with_missing_table_installs_only_the_table_and_later_stages(
    tmp_path: Path,
) -> None:
    """A valid promoted store is not rewritten when its table is absent."""
    output_dir, task = _migration_task(tmp_path)
    _install_store(task)
    root_before = (task.store_path / "zarr.json").read_bytes()

    result = _complete_image(output_dir, task)

    assert result.converted is False
    assert result.table_installed is True
    assert result.overlay_rendered is True
    assert root_before != (task.store_path / "zarr.json").read_bytes()


def test_valid_table_with_missing_overlay_renders_only_overlay_and_marker(
    tmp_path: Path,
) -> None:
    """A missing overlay does not trigger store or table replacement."""
    output_dir, task = _migration_task(tmp_path)
    _install_store(task)
    _install_table(task)
    root_before = (task.store_path / "zarr.json").read_bytes()
    table_before = (task.store_path / MEASUREMENT_TABLE_RELATIVE_PATH).read_bytes()

    result = _complete_image(output_dir, task)

    assert result.converted is False
    assert result.table_installed is False
    assert result.overlay_rendered is True
    assert (task.store_path / "zarr.json").read_bytes() == root_before
    assert (
        task.store_path / MEASUREMENT_TABLE_RELATIVE_PATH
    ).read_bytes() == table_before


@pytest.mark.parametrize("invalid_kind", ["corrupt", "jpeg", "wrong-dimensions"])
def test_invalid_overlay_is_replaced_with_a_verified_full_size_png(
    tmp_path: Path, invalid_kind: str
) -> None:
    """Unreadable, non-PNG, and wrong-plane overlays cannot remain authoritative."""
    output_dir, task = _migration_task(tmp_path)
    first = _complete_image(output_dir, task)
    expected_size = _overlay_size(task)
    if invalid_kind == "corrupt":
        task.overlay_path.write_bytes(b"not an image")
    elif invalid_kind == "jpeg":
        PILImage.new("RGB", expected_size, "red").save(
            task.overlay_path, format="JPEG"
        )
    else:
        PILImage.new("RGB", (1, 2), "red").save(task.overlay_path, format="PNG")

    repaired = _complete_image(output_dir, task)

    assert repaired.work_id == first.work_id
    assert repaired.converted is False
    assert repaired.table_installed is False
    assert repaired.overlay_rendered is True
    assert _overlay_size(task) == expected_size
    with PILImage.open(task.overlay_path) as image:
        assert image.format == "PNG"


@pytest.mark.parametrize("marker_state", ["missing", "invalid"])
def test_complete_artifacts_republish_only_missing_or_invalid_marker(
    tmp_path: Path, marker_state: str
) -> None:
    """Marker repair leaves every already-valid scientific artifact unchanged."""
    output_dir, task = _migration_task(tmp_path)
    first = _complete_image(output_dir, task)
    store_before = (task.store_path / "zarr.json").read_bytes()
    table_before = (task.store_path / MEASUREMENT_TABLE_RELATIVE_PATH).read_bytes()
    overlay_before = task.overlay_path.read_bytes()
    if marker_state == "missing":
        task.marker_path.unlink()
    else:
        marker = json.loads(task.marker_path.read_text(encoding="utf-8"))
        marker["version"] = -1
        task.marker_path.write_text(json.dumps(marker), encoding="utf-8")

    repaired = _complete_image(output_dir, task)

    assert repaired.work_id == first.work_id
    assert repaired.converted is False
    assert repaired.table_installed is False
    assert repaired.overlay_rendered is False
    assert repaired.skipped is False
    assert (task.store_path / "zarr.json").read_bytes() == store_before
    assert (
        task.store_path / MEASUREMENT_TABLE_RELATIVE_PATH
    ).read_bytes() == table_before
    assert task.overlay_path.read_bytes() == overlay_before


def test_retained_external_parquet_bytes_are_provenance_not_migration_targets(
    tmp_path: Path,
) -> None:
    """Embedding canonicalizes in memory and never rewrites Task-1 input."""
    output_dir, task = _migration_task(tmp_path)
    assert task.measurement_path is not None
    legacy = pd.DataFrame(
        {
            "Object_Label": [1],
            "Size_Area": [25.0],
            "MetadataGenetic_Strain": ["BY4741"],
        }
    )
    legacy.to_parquet(task.measurement_path, index=False)
    source_before = task.measurement_path.read_bytes()

    result = _complete_image(output_dir, task)

    embedded = pd.read_parquet(
        task.store_path / MEASUREMENT_TABLE_RELATIVE_PATH
    )
    assert result.table_installed is True
    assert task.measurement_path.read_bytes() == source_before
    assert "Metadata_Strain" in embedded.columns
    assert "MetadataGenetic_Strain" not in embedded.columns
    marker = json.loads(task.marker_path.read_text(encoding="utf-8"))
    assert marker["source_provenance"]["sha256"] == hashlib.sha256(
        source_before
    ).hexdigest()


def test_marker_repair_replaces_structural_but_scientifically_wrong_table(
    tmp_path: Path,
) -> None:
    """Structural readability is insufficient for a newly certified marker."""
    from phenotypic._cli._embedded_measurement_tables import (
        embedded_measurement_table_matches,
    )

    output_dir, task = _migration_task(tmp_path)
    _complete_image(output_dir, task)
    assert task.measurement_path is not None
    wrong_source = pd.read_parquet(task.measurement_path)
    wrong_source["Size_Area"] = [999.0]
    wrong = prepare_embedded_measurement_table(wrong_source, None)
    replace_embedded_measurement_table(task.store_path, wrong)
    task.marker_path.unlink()

    repaired = _complete_image(output_dir, task)

    expected = prepare_embedded_measurement_table(
        pd.read_parquet(task.measurement_path), None
    )
    assert repaired.table_installed is True
    assert embedded_measurement_table_matches(task.store_path, expected)


def test_marker_certification_rechecks_source_digest_and_exact_table_equality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source race after marker publication cannot certify stale science."""
    import phenotypic._cli._cli_migrate_image as migrate_module

    output_dir, task = _migration_task(tmp_path)
    real_publish = migrate_module.publish_image_success

    def publish_then_change_source(*args: object, **kwargs: object) -> Path:
        marker_path = real_publish(*args, **kwargs)
        assert task.measurement_path is not None
        changed = pd.read_parquet(task.measurement_path)
        changed["Size_Area"] = [777.0]
        changed.to_parquet(task.measurement_path, index=False)
        return marker_path

    monkeypatch.setattr(
        migrate_module, "publish_image_success", publish_then_change_source
    )

    with pytest.raises(migrate_module.MigrationImageStageError) as caught:
        _complete_image(output_dir, task)

    assert caught.value.failure.stage == "marker"
    assert "source" in caught.value.failure.reason.lower()


def test_conflicting_marker_work_id_is_repaired_with_processing_state_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale marker identity cannot block canonical work-ID repair."""
    output_dir, task = _migration_task(tmp_path)
    image = _install_fast_migrated_artifacts(task)
    _save_canonical_work_id(output_dir, "canonical-work-id")
    embedded = task.store_path / MEASUREMENT_TABLE_RELATIVE_PATH
    publish_image_success(
        output_dir,
        work_id="stale-work-id",
        dataset=task.dataset,
        relative_image_path="ds/img.tif",
        image_stem=task.stem,
        mode="full",
        attempt_id="old-attempt",
        lifecycle_epoch="old-epoch",
        artifacts={
            "store": task.store_path,
            "overlay": task.overlay_path,
            "measurements": embedded,
        },
    )
    _patch_fast_migrated_reads(monkeypatch, task, image)

    result = migrate_image_task(
        output_dir,
        task,
        metadata_csv=None,
        overlay_alpha=0.3,
        dry_run=False,
    )

    marker = json.loads(task.marker_path.read_text(encoding="utf-8"))
    assert result.work_id == "canonical-work-id"
    assert result.skipped is False
    assert marker["work_id"] == "canonical-work-id"
    assert valid_image_success(
        output_dir,
        dataset=task.dataset,
        image_stem=task.stem,
        work_id="canonical-work-id",
    )


def test_complete_image_is_a_byte_preserving_no_op(tmp_path: Path) -> None:
    """A retry does not republish any canonical artifact after final validation."""
    output_dir, task = _migration_task(tmp_path)
    first = _complete_image(output_dir, task)
    before = {
        path: path.read_bytes()
        for path in (
            task.store_path / "zarr.json",
            task.store_path / MEASUREMENT_TABLE_RELATIVE_PATH,
            task.overlay_path,
            task.marker_path,
        )
    }

    second = _complete_image(output_dir, task)

    assert second == type(first)(
        index=first.index,
        dataset=first.dataset,
        stem=first.stem,
        work_id=first.work_id,
        converted=False,
        table_installed=False,
        overlay_rendered=False,
        marker_digest=first.marker_digest,
        skipped=True,
    )
    assert {path: path.read_bytes() for path in before} == before


def test_invalid_store_is_replaced_from_the_explicit_hdf_source(tmp_path: Path) -> None:
    """A present directory without a valid root is conversion work, not success."""
    output_dir, task = _migration_task(tmp_path)
    task.store_path.mkdir(parents=True)
    (task.store_path / "zarr.json").write_text("{}", encoding="utf-8")

    result = _complete_image(output_dir, task)

    assert result.converted is True
    assert valid_staged_store(task.store_path)


def test_interrupted_temporary_artifacts_do_not_block_retry(tmp_path: Path) -> None:
    """Unique stale preparation siblings never masquerade as canonical output."""
    output_dir, task = _migration_task(tmp_path)
    stale_store = task.store_path.parent / ".img.ome.zarr.interrupted.part"
    stale_store.mkdir(parents=True)
    (stale_store / "zarr.json").write_text("{}", encoding="utf-8")
    task.overlay_path.parent.mkdir(parents=True)
    (task.overlay_path.parent / ".img.png.interrupted.tmp").write_bytes(b"partial")
    assert task.measurement_path is not None
    (task.measurement_path.parent / ".img.parquet.interrupted.tmp").write_bytes(
        b"partial"
    )

    result = _complete_image(output_dir, task)

    assert result.converted is True
    assert result.table_installed is True
    assert result.overlay_rendered is True
    assert valid_staged_store(task.store_path)


def test_complete_primitive_never_discovers_paths_with_glob_or_rglob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One manifest task uses only its explicit canonical paths."""
    output_dir, task = _migration_task(tmp_path)
    _complete_image(output_dir, task)

    def refuse_discovery(*_args: object, **_kwargs: object):
        raise AssertionError("per-image migration performed tree discovery")

    monkeypatch.setattr(Path, "glob", refuse_discovery)
    monkeypatch.setattr(Path, "rglob", refuse_discovery)

    assert _complete_image(output_dir, task).skipped is True


class _RejectingGuard:
    """Publication guard that rejects one selected canonical commit."""

    def __init__(self, reject_entry: int = 1) -> None:
        self.reject_entry = reject_entry
        self.entries = 0
        self.held = False

    @contextmanager
    def __call__(self):
        self.entries += 1
        if self.entries == self.reject_entry:
            raise RuntimeError("migration generation is stale")
        self.held = True
        try:
            yield
        finally:
            self.held = False


@pytest.mark.parametrize("stage", ["store", "table", "overlay", "marker", "result"])
def test_revoked_generation_cannot_publish_or_report_stage_success(
    tmp_path: Path, stage: str
) -> None:
    """Every canonical publisher and the returned success have an independent fence."""
    output_dir, task = _migration_task(tmp_path)
    if stage in {"table", "overlay", "marker"}:
        _install_store(task)
    if stage in {"overlay", "marker"}:
        _install_table(task)
    if stage == "marker":
        _install_overlay(output_dir, task)
    if stage == "result":
        _complete_image(output_dir, task)

    guarded_before = {
        "store": task.store_path / "zarr.json",
        "table": task.store_path / MEASUREMENT_TABLE_RELATIVE_PATH,
        "overlay": task.overlay_path,
        "marker": task.marker_path,
        "result": task.marker_path,
    }[stage]
    before = guarded_before.read_bytes() if guarded_before.is_file() else None

    with pytest.raises(RuntimeError, match="generation is stale"):
        migrate_image_task(
            output_dir,
            task,
            metadata_csv=None,
            overlay_alpha=0.3,
            dry_run=False,
            commit_guard=_RejectingGuard(),
        )

    after = guarded_before.read_bytes() if guarded_before.is_file() else None
    assert after == before


def test_guard_spans_each_final_replace_and_final_success_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lifecycle lock covers the mutation and its last authority check."""
    from phenotypic._cli import _cli_migrate_image

    output_dir, task = _migration_task(tmp_path)
    guard = _RejectingGuard(reject_entry=-1)
    original_replace = os.replace
    guarded_destinations = {
        task.store_path,
        task.overlay_path,
        task.marker_path,
    }
    observed_replaces: list[Path] = []

    def checked_replace(source: str | bytes, destination: str | bytes) -> None:
        destination_path = Path(destination)
        if destination_path in guarded_destinations:
            assert guard.held is True
            observed_replaces.append(destination_path)
        original_replace(source, destination)

    final_validation_lock_states: list[bool] = []
    original_valid = _cli_migrate_image.valid_image_success

    def checked_valid(*args: object, **kwargs: object) -> bool:
        final_validation_lock_states.append(guard.held)
        return original_valid(*args, **kwargs)

    monkeypatch.setattr(os, "replace", checked_replace)
    monkeypatch.setattr(_cli_migrate_image, "valid_image_success", checked_valid)

    result = migrate_image_task(
        output_dir,
        task,
        metadata_csv=None,
        overlay_alpha=0.3,
        dry_run=False,
        commit_guard=guard,
    )

    assert result.skipped is False
    assert guarded_destinations <= set(observed_replaces)
    assert final_validation_lock_states[-1] is True


def test_reclaim_records_exact_prestate_poststate_and_deleted_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clean reclaim result binds every source byte before deleting it."""
    from phenotypic._cli._cli_migrate_image import reclaim_image_sources
    from phenotypic.sdk_ import _hdf_to_zarr

    output_dir, task = _migration_task(tmp_path)
    migrated = _complete_image(output_dir, task)
    original_faithful = _hdf_to_zarr._conversion_is_faithful
    faithful_calls: list[tuple[Path, Path]] = []

    def record_faithful(source: Path, store: Path) -> bool:
        faithful_calls.append((source, store))
        return original_faithful(source, store)

    monkeypatch.setattr(
        _hdf_to_zarr, "_conversion_is_faithful", record_faithful
    )
    result = reclaim_image_sources(
        output_dir,
        task,
        metadata_csv=None,
    )

    assert result.index == task.index
    assert (result.dataset, result.stem, result.work_id) == (
        task.dataset,
        task.stem,
        migrated.work_id,
    )
    assert result.marker_digest == migrated.marker_digest
    assert result.intended_deletions == (task.hdf_path, task.measurement_path)
    assert result.hdf_prestate.exists is True
    assert result.hdf_prestate.size
    assert len(result.hdf_prestate.sha256 or "") == 64
    assert result.parquet_prestate.exists is True
    assert result.parquet_prestate.size
    assert len(result.parquet_prestate.sha256 or "") == 64
    assert [state.exists for state in result.observed_poststate] == [False, False]
    assert result.deleted_paths == (task.hdf_path, task.measurement_path)
    assert result.retained_paths == ()
    assert result.reason is None
    assert faithful_calls == [(task.hdf_path, task.store_path)]


def test_reclaim_rejects_a_generically_valid_legacy_only_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reclaim requires marker authority over migrated, not deletable, artifacts."""
    from phenotypic._cli._cli_migrate_image import reclaim_image_sources

    output_dir, task = _migration_task(
        tmp_path, with_measurements=False, zero_objects=True
    )
    _save_canonical_work_id(output_dir, "canonical-work-id")
    assert task.hdf_path is not None
    publish_image_success(
        output_dir,
        work_id="canonical-work-id",
        dataset=task.dataset,
        relative_image_path="ds/img.tif",
        image_stem=task.stem,
        mode="full",
        attempt_id="legacy-attempt",
        lifecycle_epoch="legacy-epoch",
        artifacts={"hdf": task.hdf_path},
    )
    assert valid_image_success(
        output_dir,
        dataset=task.dataset,
        image_stem=task.stem,
        work_id="canonical-work-id",
    )
    monkeypatch.setattr(
        "phenotypic.sdk_._hdf_to_zarr._conversion_is_faithful",
        lambda _source, _store: True,
    )

    result = reclaim_image_sources(output_dir, task, metadata_csv=None)

    assert task.hdf_path.is_file()
    assert result.deleted_paths == ()
    assert result.retained_paths == (task.hdf_path,)
    assert "marker is not authoritative" in (result.reason or "")


def test_reclaim_retains_hdf_when_full_conversion_comparison_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Structural store validity cannot substitute for full source fidelity."""
    from phenotypic._cli._cli_migrate_image import reclaim_image_sources

    output_dir, task = _migration_task(
        tmp_path, with_measurements=False, zero_objects=True
    )
    _complete_image(output_dir, task)
    monkeypatch.setattr(
        "phenotypic.sdk_._hdf_to_zarr._conversion_is_faithful",
        lambda _source, _store: False,
    )

    result = reclaim_image_sources(output_dir, task, metadata_csv=None)

    assert task.hdf_path is not None and task.hdf_path.is_file()
    assert result.deleted_paths == ()
    assert result.retained_paths == (task.hdf_path,)
    assert "does not match" in (result.reason or "")


def test_reclaim_rechecks_hdf_fidelity_after_guard_entry_mutates_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store changed while waiting for the lifecycle lock retains its HDF."""
    from phenotypic._cli import _cli_migrate_image
    from phenotypic.sdk_ import _hdf_to_zarr

    output_dir, task = _migration_task(
        tmp_path, with_measurements=False, zero_objects=True
    )
    task.store_path.mkdir(parents=True)
    store_authority = task.store_path / "zarr.json"
    store_authority.write_bytes(b"faithful")
    monkeypatch.setattr(
        _cli_migrate_image,
        "_configured_work_id",
        lambda _root, _dataset, _stem: "work-id",
    )
    monkeypatch.setattr(
        _cli_migrate_image,
        "_current_marker_digest",
        lambda _root, _task, _work_id: "a" * 64,
    )
    monkeypatch.setattr(
        _cli_migrate_image,
        "_marker_still_current",
        lambda _root, _task, _work_id, _digest: True,
    )
    monkeypatch.setattr(
        _hdf_to_zarr,
        "_marker_authority_permits_unlink",
        lambda _root, _dataset, _stem: True,
    )
    guard_held = False
    fidelity_guard_states: list[bool] = []

    def faithful_after_latest_store_bytes(_source: Path, _store: Path) -> bool:
        fidelity_guard_states.append(guard_held)
        return store_authority.read_bytes() == b"faithful"

    monkeypatch.setattr(
        _hdf_to_zarr,
        "_conversion_is_faithful",
        faithful_after_latest_store_bytes,
    )

    @contextmanager
    def mutate_store_on_guard_entry():
        nonlocal guard_held
        guard_held = True
        store_authority.write_bytes(b"changed while waiting")
        try:
            yield
        finally:
            guard_held = False

    result = _cli_migrate_image.reclaim_image_sources(
        output_dir,
        task,
        metadata_csv=None,
        commit_guard=lambda: mutate_store_on_guard_entry(),
    )

    assert task.hdf_path is not None and task.hdf_path.is_file()
    assert fidelity_guard_states == [True]
    assert result.deleted_paths == ()
    assert result.retained_paths == (task.hdf_path,)
    assert "does not match" in (result.reason or "")


def test_reclaim_reconstructs_duplicate_metadata_fanout_before_parquet_unlink(
    tmp_path: Path,
) -> None:
    """The real metadata preparation must reproduce every duplicate-key row."""
    from phenotypic._cli._cli_migrate_image import reclaim_image_sources

    output_dir, task = _migration_task(tmp_path)
    metadata_csv = output_dir / "deliverables" / "metadata.csv"
    metadata_csv.parent.mkdir(parents=True)
    metadata_csv.write_text(
        "Metadata_ImageName,Metadata_Strain\nimg,BY4741\nimg,BY4742\n",
        encoding="utf-8",
    )
    _complete_image(output_dir, task, metadata_csv=metadata_csv)
    assert task.hdf_path is not None
    task.hdf_path.unlink()
    parquet_only = replace(task, hdf_path=None)

    result = reclaim_image_sources(
        output_dir,
        parquet_only,
        metadata_csv=metadata_csv,
    )

    assert task.measurement_path is not None
    assert not task.measurement_path.exists()
    assert result.deleted_paths == (task.measurement_path,)
    assert result.retained_paths == ()
    assert result.reason is None


def test_reclaim_retains_structurally_valid_table_with_wrong_duplicate_fanout(
    tmp_path: Path,
) -> None:
    """One-row embedded data cannot certify a two-row metadata fan-out."""
    from phenotypic._cli._cli_migrate_image import reclaim_image_sources

    output_dir, task = _migration_task(tmp_path)
    metadata_csv = output_dir / "deliverables" / "metadata.csv"
    metadata_csv.parent.mkdir(parents=True)
    metadata_csv.write_text(
        "Metadata_ImageName,Metadata_Strain\nimg,BY4741\nimg,BY4742\n",
        encoding="utf-8",
    )
    _complete_image(output_dir, task, metadata_csv=metadata_csv)
    assert task.measurement_path is not None
    wrong = prepare_embedded_measurement_table(
        pd.read_parquet(task.measurement_path), None
    )
    replace_embedded_measurement_table(task.store_path, wrong)
    task.marker_path.unlink()
    _complete_image(output_dir, task, metadata_csv=metadata_csv)
    assert task.hdf_path is not None
    task.hdf_path.unlink()
    parquet_only = replace(task, hdf_path=None)

    result = reclaim_image_sources(
        output_dir,
        parquet_only,
        metadata_csv=metadata_csv,
    )

    assert task.measurement_path.is_file()
    assert result.deleted_paths == ()
    assert result.retained_paths == (task.measurement_path,)
    assert "embedded table does not exactly match" in (result.reason or "")


@pytest.mark.parametrize("source_kind", ["hdf", "parquet"])
def test_revoked_generation_cannot_unlink_a_reclaim_source(
    tmp_path: Path, source_kind: str
) -> None:
    """Both irreversible deletion points acquire the generation guard."""
    from phenotypic._cli._cli_migrate_image import reclaim_image_sources

    output_dir, task = _migration_task(
        tmp_path,
        with_measurements=source_kind == "parquet",
        zero_objects=source_kind == "hdf",
    )
    _complete_image(output_dir, task)
    if source_kind == "parquet":
        assert task.hdf_path is not None
        task.hdf_path.unlink()
        task = replace(task, hdf_path=None)
        source = task.measurement_path
    else:
        source = task.hdf_path
    assert source is not None and source.is_file()

    with pytest.raises(RuntimeError, match="generation is stale"):
        reclaim_image_sources(
            output_dir,
            task,
            metadata_csv=None,
            commit_guard=_RejectingGuard(),
        )

    assert source.is_file()


def test_reclaim_guard_revalidates_parquet_fingerprint_before_unlink(
    tmp_path: Path,
) -> None:
    """A source changed while waiting for the lifecycle lock is retained."""
    from phenotypic._cli._cli_migrate_image import reclaim_image_sources

    output_dir, task = _migration_task(tmp_path)
    _complete_image(output_dir, task)
    assert task.hdf_path is not None
    task.hdf_path.unlink()
    parquet_only = replace(task, hdf_path=None)
    assert task.measurement_path is not None

    @contextmanager
    def mutate_before_commit():
        task.measurement_path.write_bytes(b"changed after preparation")
        yield

    result = reclaim_image_sources(
        output_dir,
        parquet_only,
        metadata_csv=None,
        commit_guard=lambda: mutate_before_commit(),
    )

    assert task.measurement_path.is_file()
    assert result.deleted_paths == ()
    assert result.retained_paths == (task.measurement_path,)
    assert "changed before unlink" in (result.reason or "")


def test_reclaim_reports_unreadable_parquet_as_retained_source_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corrupt source still produces the typed failure result Task 4 publishes."""
    from phenotypic._cli import _cli_migrate_image

    output_dir, task = _migration_task(tmp_path)
    assert task.hdf_path is not None
    task.hdf_path.unlink()
    task = replace(task, hdf_path=None)
    assert task.measurement_path is not None
    task.measurement_path.write_bytes(b"not parquet")
    monkeypatch.setattr(
        _cli_migrate_image,
        "_configured_work_id",
        lambda _root, _dataset, _stem: "work-id",
    )
    monkeypatch.setattr(
        _cli_migrate_image,
        "_current_marker_digest",
        lambda _root, _task, _work_id: "a" * 64,
    )
    monkeypatch.setattr(
        _cli_migrate_image,
        "_marker_still_current",
        lambda _root, _task, _work_id, _digest: True,
    )

    result = _cli_migrate_image.reclaim_image_sources(
        output_dir,
        task,
        metadata_csv=None,
    )

    assert result.deleted_paths == ()
    assert result.retained_paths == (task.measurement_path,)
    assert result.parquet_prestate.exists is True
    assert "external Parquet preparation failed" in (result.reason or "")


def test_reclaim_reports_hdf_unlink_error_as_retained_source_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ordinary unlink failure is evidence, while guard revocation still raises."""
    from phenotypic._cli import _cli_migrate_image
    from phenotypic.sdk_ import _hdf_to_zarr

    output_dir, task = _migration_task(
        tmp_path, with_measurements=False, zero_objects=True
    )
    assert task.hdf_path is not None
    monkeypatch.setattr(
        _cli_migrate_image,
        "_configured_work_id",
        lambda _root, _dataset, _stem: "work-id",
    )
    monkeypatch.setattr(
        _cli_migrate_image,
        "_current_marker_digest",
        lambda _root, _task, _work_id: "a" * 64,
    )
    monkeypatch.setattr(
        _cli_migrate_image,
        "_marker_still_current",
        lambda _root, _task, _work_id, _digest: True,
    )
    monkeypatch.setattr(
        _hdf_to_zarr,
        "_conversion_is_faithful",
        lambda _source, _store: True,
    )
    monkeypatch.setattr(
        _hdf_to_zarr,
        "_marker_authority_permits_unlink",
        lambda _root, _dataset, _stem: True,
    )
    original_unlink = Path.unlink

    def fail_hdf_unlink(path: Path, *args: object, **kwargs: object) -> None:
        if path == task.hdf_path:
            raise OSError("simulated unlink failure")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_hdf_unlink)

    result = _cli_migrate_image.reclaim_image_sources(
        output_dir,
        task,
        metadata_csv=None,
    )

    assert task.hdf_path.is_file()
    assert result.deleted_paths == ()
    assert result.retained_paths == (task.hdf_path,)
    assert "HDF unlink failed: OSError" in (result.reason or "")
