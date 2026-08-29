"""Idempotent migration and source reclamation for one manifest image."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import pandas as pd

from phenotypic._cli._cli_completion import (
    ARTIFACT_KIND_FILE,
    ARTIFACT_KIND_STORE,
    publish_image_success,
    valid_image_success,
)
from phenotypic._cli._cli_migrate_manifest import MigrationImageTask
from phenotypic._cli._cli_overlay_rendering import (
    overlay_output_manager,
    valid_migration_overlay,
)
from phenotypic._cli._embedded_measurement_tables import (
    embedded_measurement_table_matches,
    prepare_embedded_measurement_table,
)
from phenotypic.sdk_ import (
    CommitGuard,
    MEASUREMENT_TABLE_RELATIVE_PATH,
    image_completion_marker_path,
    load_image_from_store,
    publication_commit,
    replace_embedded_measurement_table,
)
from phenotypic.sdk_._hdf_to_zarr import (
    _load_for_migration,
    migrate_hdf_to_zarr,
)
from phenotypic.sdk_._measurement_tables import (
    _valid_embedded_measurement_contract,
)
from phenotypic.sdk_.ngff_ import valid_staged_store


@dataclass(frozen=True)
class MigrationImageResult:
    """Validated result of migrating one explicit manifest image."""

    index: int
    dataset: str
    stem: str
    work_id: str
    converted: bool
    table_installed: bool
    overlay_rendered: bool
    marker_digest: str
    skipped: bool


@dataclass(frozen=True)
class SourceArtifactState:
    """Content fingerprint or explicit absence for one migration source."""

    path: Path | None
    exists: bool
    size: int | None
    sha256: str | None


@dataclass(frozen=True)
class ReclaimResult:
    """Exact before/after evidence for one source-reclamation attempt."""

    index: int
    dataset: str
    stem: str
    work_id: str
    marker_digest: str
    intended_deletions: tuple[Path, ...]
    hdf_prestate: SourceArtifactState
    parquet_prestate: SourceArtifactState
    observed_poststate: tuple[SourceArtifactState, SourceArtifactState]
    deleted_paths: tuple[Path, ...]
    retained_paths: tuple[Path, ...]
    reason: str | None


def _migration_work_id(dataset: str, stem: str) -> str:
    """Return the deterministic work id used to authorize a migrated image."""
    return hashlib.sha256(f"migration:{dataset}/{stem}".encode()).hexdigest()


def _configured_work_id(output_dir: Path, dataset: str, stem: str) -> str:
    """Return the state-authorized work id for a migrated store."""
    from phenotypic._cli._cli_state_management import load_processing_state

    state = load_processing_state(output_dir)
    work_ids = state.config.get("work_ids", {}) if state is not None else {}
    images = work_ids.get(dataset, {}) if isinstance(work_ids, dict) else {}
    if isinstance(images, dict):
        for image_name, value in images.items():
            if Path(str(image_name)).stem == stem and isinstance(value, str):
                return value
    return _migration_work_id(dataset, stem)


def _existing_marker_identity(
    output_dir: Path, dataset: str, stem: str, work_id: str
) -> dict[str, str]:
    """Return preserved identity fields from an existing image marker."""
    defaults = {
        "work_id": work_id,
        "relative_image_path": f"{dataset}/{stem}",
        "mode": "full",
        "attempt_id": "migration",
        "lifecycle_epoch": "migration",
    }
    marker_path = image_completion_marker_path(output_dir, dataset, stem)
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return defaults
    if not isinstance(marker, dict):
        return defaults
    if marker.get("dataset") not in {None, dataset} or marker.get(
        "image_stem"
    ) not in {None, stem}:
        return defaults
    for field in (
        "relative_image_path",
        "mode",
        "attempt_id",
        "lifecycle_epoch",
    ):
        value = marker.get(field)
        if isinstance(value, str) and value:
            defaults[field] = value
    return defaults


def _migration_marker_artifacts(
    task: MigrationImageTask,
    *,
    table_authoritative: bool,
) -> dict[str, Path]:
    """Return the exact durable migrated artifacts a marker must declare."""
    artifacts = {
        "store": task.store_path,
        "overlay": task.overlay_path,
    }
    if table_authoritative:
        artifacts["measurements"] = (
            task.store_path / MEASUREMENT_TABLE_RELATIVE_PATH
        )
    return artifacts


def _valid_migration_marker(
    output_dir: Path,
    task: MigrationImageTask,
    work_id: str,
    *,
    table_authoritative: bool | None = None,
) -> bool:
    """Return whether a marker binds exactly this task's migrated authority."""
    embedded = task.store_path / MEASUREMENT_TABLE_RELATIVE_PATH
    if table_authoritative is None:
        table_authoritative = embedded.is_file()
    expected = _migration_marker_artifacts(
        task,
        table_authoritative=table_authoritative,
    )
    if not valid_image_success(
        output_dir,
        dataset=task.dataset,
        image_stem=task.stem,
        work_id=work_id,
    ):
        return False
    try:
        marker = json.loads(task.marker_path.read_text(encoding="utf-8"))
        declared = marker["artifacts"]
        if not isinstance(declared, dict) or set(declared) != set(expected):
            return False
        output_root = output_dir.resolve()
        for name, path in expected.items():
            descriptor = declared[name]
            if not isinstance(descriptor, dict):
                return False
            relative = path.resolve(strict=True).relative_to(output_root)
            expected_kind = (
                ARTIFACT_KIND_STORE if name == "store" else ARTIFACT_KIND_FILE
            )
            if descriptor.get("path") != relative.as_posix() or descriptor.get(
                "kind"
            ) != expected_kind:
                return False
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def _image_plane_shape(image: object) -> tuple[int, ...]:
    """Return the migrated image plane shape used by overlay validation."""
    gray = getattr(image, "gray")[:]
    return tuple(int(value) for value in gray.shape[:2])


def _table_state(
    task: MigrationImageTask,
    *,
    metadata_csv: Path | None,
    commit_guard: CommitGuard | None,
) -> tuple[bool, bool]:
    """Repair and return ``(valid, installed)`` for one embedded table."""
    valid = _valid_embedded_measurement_contract(task.store_path)
    source = task.measurement_path
    if not valid and source is not None and source.is_file():
        baseline = pd.read_parquet(source)
        prepared = prepare_embedded_measurement_table(baseline, metadata_csv)
        replace_embedded_measurement_table(
            task.store_path,
            prepared,
            commit_guard=commit_guard,
        )
        valid = _valid_embedded_measurement_contract(task.store_path)
        if not valid:
            raise RuntimeError("embedded measurement table validation failed")
        return True, True
    embedded = task.store_path / MEASUREMENT_TABLE_RELATIVE_PATH
    if embedded.exists() and not valid:
        raise RuntimeError("embedded measurement table validation failed")
    return valid, False


def _dry_run_result(
    output_dir: Path,
    task: MigrationImageTask,
    *,
    metadata_csv: Path | None,
    work_id: str,
) -> MigrationImageResult:
    """Inspect one task without publishing scientific bytes."""
    store_valid = valid_staged_store(task.store_path)
    if store_valid:
        image = load_image_from_store(task.store_path)
    else:
        if task.hdf_path is None or not task.hdf_path.is_file():
            raise FileNotFoundError(
                f"No legacy HDF source exists for {task.dataset}/{task.stem}"
            )
        image = _load_for_migration(task.hdf_path)
    table_valid = store_valid and _valid_embedded_measurement_contract(
        task.store_path
    )
    table_needed = (
        not table_valid
        and task.measurement_path is not None
        and task.measurement_path.is_file()
    )
    if not table_valid and not table_needed and getattr(image, "num_objects") != 0:
        raise RuntimeError("nonempty migrated image has no valid measurement table")
    overlay_valid = valid_migration_overlay(
        task.overlay_path, _image_plane_shape(image)
    )
    marker_valid = (
        store_valid
        and (table_valid or getattr(image, "num_objects") == 0)
        and overlay_valid
        and _valid_migration_marker(
            output_dir,
            task,
            work_id,
            table_authoritative=table_valid,
        )
    )
    marker_digest = (
        hashlib.sha256(task.marker_path.read_bytes()).hexdigest()
        if marker_valid
        else ""
    )
    return MigrationImageResult(
        index=task.index,
        dataset=task.dataset,
        stem=task.stem,
        work_id=work_id,
        converted=not store_valid,
        table_installed=table_needed,
        overlay_rendered=not overlay_valid,
        marker_digest=marker_digest,
        skipped=marker_valid,
    )


def migrate_image_task(
    output_dir: Path,
    task: MigrationImageTask,
    *,
    metadata_csv: Path | None,
    overlay_alpha: float,
    dry_run: bool,
    commit_guard: CommitGuard | None = None,
) -> MigrationImageResult:
    """Migrate one explicit image task to complete store-backed authority."""
    output_dir = Path(output_dir)
    work_id = _configured_work_id(output_dir, task.dataset, task.stem)
    if dry_run:
        return _dry_run_result(
            output_dir,
            task,
            metadata_csv=metadata_csv,
            work_id=work_id,
        )

    converted = False
    table_installed = False
    overlay_rendered = False
    if not valid_staged_store(task.store_path):
        if task.hdf_path is None or not task.hdf_path.is_file():
            raise FileNotFoundError(
                f"No legacy HDF source exists for {task.dataset}/{task.stem}"
            )
        migrated = migrate_hdf_to_zarr(
            task.hdf_path,
            task.store_path,
            keep_source=True,
            commit_guard=commit_guard,
        )
        if migrated != task.store_path or not valid_staged_store(task.store_path):
            raise RuntimeError("migrated image store validation failed")
        converted = True

    image = load_image_from_store(task.store_path)
    table_valid, table_installed = _table_state(
        task,
        metadata_csv=metadata_csv,
        commit_guard=commit_guard,
    )
    if not table_valid and image.num_objects != 0:
        raise RuntimeError("nonempty migrated image has no valid measurement table")

    expected_shape = _image_plane_shape(image)
    if not valid_migration_overlay(task.overlay_path, expected_shape):
        manager = overlay_output_manager(output_dir, overlay_alpha=overlay_alpha)
        overlay_path = manager.save_overlay(
            image,
            task.dataset,
            task.stem,
            commit_guard=commit_guard,
        )
        if overlay_path != task.overlay_path or not valid_migration_overlay(
            task.overlay_path, expected_shape
        ):
            raise RuntimeError("migrated overlay validation failed")
        overlay_rendered = True

    marker_valid = _valid_migration_marker(
        output_dir,
        task,
        work_id,
        table_authoritative=table_valid,
    )
    if not marker_valid:
        identity = _existing_marker_identity(
            output_dir, task.dataset, task.stem, work_id
        )
        artifacts = _migration_marker_artifacts(
            task,
            table_authoritative=table_valid,
        )
        marker_path = publish_image_success(
            output_dir,
            work_id=identity["work_id"],
            dataset=task.dataset,
            relative_image_path=identity["relative_image_path"],
            image_stem=task.stem,
            mode=identity["mode"],
            attempt_id=identity["attempt_id"],
            lifecycle_epoch=identity["lifecycle_epoch"],
            artifacts=artifacts,
            commit_guard=commit_guard,
        )
        if marker_path != task.marker_path:
            raise RuntimeError("marker publisher returned a non-canonical path")

    with publication_commit(commit_guard):
        if not valid_staged_store(task.store_path):
            raise RuntimeError("migrated image store validation failed")
        if table_valid != _valid_embedded_measurement_contract(task.store_path):
            raise RuntimeError("embedded measurement table validation failed")
        if not valid_migration_overlay(task.overlay_path, expected_shape):
            raise RuntimeError("migrated overlay validation failed")
        if not _valid_migration_marker(
            output_dir,
            task,
            work_id,
            table_authoritative=table_valid,
        ):
            raise RuntimeError("migrated marker validation failed")
        marker_digest = hashlib.sha256(task.marker_path.read_bytes()).hexdigest()
        return MigrationImageResult(
            index=task.index,
            dataset=task.dataset,
            stem=task.stem,
            work_id=work_id,
            converted=converted,
            table_installed=table_installed,
            overlay_rendered=overlay_rendered,
            marker_digest=marker_digest,
            skipped=not (
                converted
                or table_installed
                or overlay_rendered
                or not marker_valid
            ),
        )


def _source_artifact_state(path: Path | None) -> SourceArtifactState:
    """Fingerprint one source file, preserving explicit absence."""
    if path is None:
        return SourceArtifactState(
            path=None,
            exists=False,
            size=None,
            sha256=None,
        )
    path = Path(path)
    if not path.is_file():
        return SourceArtifactState(
            path=path,
            exists=False,
            size=None,
            sha256=None,
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return SourceArtifactState(
        path=path,
        exists=True,
        size=path.stat().st_size,
        sha256=digest.hexdigest(),
    )


def _current_marker_digest(
    output_dir: Path,
    task: MigrationImageTask,
    work_id: str,
) -> str:
    """Return the digest of the current marker payload, valid or invalid."""
    try:
        return hashlib.sha256(task.marker_path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _marker_still_current(
    output_dir: Path,
    task: MigrationImageTask,
    work_id: str,
    marker_digest: str,
) -> bool:
    """Revalidate exact marker authority at an unlink commit point."""
    return (
        bool(marker_digest)
        and _valid_migration_marker(output_dir, task, work_id)
        and _current_marker_digest(output_dir, task, work_id) == marker_digest
    )


def reclaim_image_sources(
    output_dir: Path,
    task: MigrationImageTask,
    *,
    metadata_csv: Path | None,
    commit_guard: CommitGuard | None = None,
) -> ReclaimResult:
    """Strongly revalidate and reclaim one task's legacy source files."""
    from phenotypic.sdk_ import _hdf_to_zarr

    output_dir = Path(output_dir)
    work_id = _configured_work_id(output_dir, task.dataset, task.stem)
    marker_digest = _current_marker_digest(output_dir, task, work_id)
    marker_authoritative = _marker_still_current(
        output_dir, task, work_id, marker_digest
    )
    hdf_prestate = _source_artifact_state(task.hdf_path)
    parquet_prestate = _source_artifact_state(task.measurement_path)
    intended = tuple(
        path
        for path in (task.hdf_path, task.measurement_path)
        if path is not None
    )
    deleted: list[Path] = []
    reasons: list[str] = []

    if not marker_authoritative:
        reasons.append("the current image marker is not authoritative")
    elif hdf_prestate.exists:
        assert task.hdf_path is not None
        if not _hdf_to_zarr._marker_authority_permits_unlink(
            output_dir, task.dataset, task.stem
        ):
            reasons.append("HDF marker authority does not permit source unlink")
        else:
            with publication_commit(commit_guard):
                if _source_artifact_state(task.hdf_path) != hdf_prestate:
                    reasons.append("HDF source changed before unlink")
                elif not _hdf_to_zarr._marker_authority_permits_unlink(
                    output_dir, task.dataset, task.stem
                ) or not _marker_still_current(
                    output_dir, task, work_id, marker_digest
                ):
                    reasons.append("HDF marker authority changed before unlink")
                elif not _hdf_to_zarr._conversion_is_faithful(
                    task.hdf_path, task.store_path
                ):
                    reasons.append(
                        "HDF re-read of the converted store does not match the source"
                    )
                else:
                    try:
                        task.hdf_path.unlink()
                    except OSError as exc:
                        reasons.append(
                            f"HDF unlink failed: {type(exc).__name__}: {exc}"
                        )
                    else:
                        deleted.append(task.hdf_path)

    if marker_authoritative and parquet_prestate.exists:
        assert task.measurement_path is not None
        try:
            prepared = prepare_embedded_measurement_table(
                pd.read_parquet(task.measurement_path), metadata_csv
            )
        except Exception as exc:  # noqa: BLE001 - preserve typed failure evidence
            reasons.append(
                "external Parquet preparation failed: "
                f"{type(exc).__name__}: {exc}"
            )
        else:
            if not _valid_embedded_measurement_contract(
                task.store_path
            ) or not embedded_measurement_table_matches(
                task.store_path, prepared
            ):
                reasons.append(
                    "external Parquet embedded table does not exactly match the source"
                )
            else:
                with publication_commit(commit_guard):
                    if (
                        _source_artifact_state(task.measurement_path)
                        != parquet_prestate
                    ):
                        reasons.append("external Parquet changed before unlink")
                    elif not _marker_still_current(
                        output_dir, task, work_id, marker_digest
                    ):
                        reasons.append(
                            "external Parquet marker authority changed before unlink"
                        )
                    elif not embedded_measurement_table_matches(
                        task.store_path, prepared
                    ):
                        reasons.append(
                            "embedded table changed before external Parquet unlink"
                        )
                    else:
                        try:
                            task.measurement_path.unlink()
                        except OSError as exc:
                            reasons.append(
                                "external Parquet unlink failed: "
                                f"{type(exc).__name__}: {exc}"
                            )
                        else:
                            deleted.append(task.measurement_path)

    observed = (
        _source_artifact_state(task.hdf_path),
        _source_artifact_state(task.measurement_path),
    )
    retained = tuple(
        path for path in intended if _source_artifact_state(path).exists
    )
    if not intended and not reasons:
        reasons.append("no migration sources were requested for reclamation")
    elif intended and not any(
        state.exists for state in (hdf_prestate, parquet_prestate)
    ) and not reasons:
        reasons.append("intended migration sources were already absent")
    return ReclaimResult(
        index=task.index,
        dataset=task.dataset,
        stem=task.stem,
        work_id=work_id,
        marker_digest=marker_digest,
        intended_deletions=intended,
        hdf_prestate=hdf_prestate,
        parquet_prestate=parquet_prestate,
        observed_poststate=observed,
        deleted_paths=tuple(deleted),
        retained_paths=retained,
        reason="; ".join(reasons) if reasons else None,
    )


__all__ = [
    "MigrationImageResult",
    "ReclaimResult",
    "SourceArtifactState",
    "migrate_image_task",
    "reclaim_image_sources",
]
