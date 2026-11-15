"""Idempotent migration and source reclamation for one manifest image."""

from __future__ import annotations

from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
from typing import Any

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
    PreparedEmbeddedMeasurementTable,
    convert_tables_v1_to_v2,
    declared_measurement_table_path,
    image_completion_marker_path,
    image_record_path,
    load_image_from_store,
    publication_commit,
    replace_embedded_measurement_table,
    source_image_stem,
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
class MigrationImagePartialResult:
    """Durable pass evidence accumulated before one image-stage failure."""

    index: int
    dataset: str
    stem: str
    work_id: str
    converted: bool
    table_installed: bool
    overlay_rendered: bool


@dataclass(frozen=True)
class MigrationImageStageFailure:
    """Typed image-stage failure with all completed legacy-pass evidence."""

    stage: str
    target: Path
    reason: str
    partial: MigrationImagePartialResult


class MigrationImageStageError(RuntimeError):
    """Raised when an image migration stage fails after partial progress."""

    def __init__(self, failure: MigrationImageStageFailure) -> None:
        self.failure = failure
        super().__init__(failure.reason)


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
            if (
                source_image_stem(Path(str(image_name))) == stem
                and isinstance(value, str)
            ):
                return value
    return _migration_work_id(dataset, stem)


def _existing_marker_identity(
    output_dir: Path, dataset: str, stem: str, work_id: str
) -> dict[str, Any]:
    """Return the identity a migrated image's record is (re)published with.

    **The record first** (``images/<ds>/<stem>.json``), and the legacy
    ``image_complete/`` marker only when no record carries an identity (no
    record, or a partial one written by ``record_stage`` with no
    ``work_id``). Reading only the legacy marker gave every record-era tree
    default identity whenever its record stopped verifying -- for example
    after a crash between the ``tables_v2`` conversion and its republish
    (#281 P3 review I2).

    A record is used when it names this dataset and stem and either carries
    the configured *work_id* or is ``"migrated"``, whose ``work_id`` is never
    compared (U-10). Its ``work_id``, ``relative_image_path``, ``mode``,
    ``attempt_id``, ``provenance`` and ``source_provenance`` are kept
    verbatim. A record for another image or another ``work_id`` is not this
    image's identity, and the defaults are used rather than the legacy
    marker's.

    ``lifecycle_epoch`` is the one field that is not kept when a migration
    generation is live: the record is published under that generation, and
    under SLURM ``publish_image_success`` refuses any other epoch.

    Returns:
        ``work_id``, ``relative_image_path``, ``mode``, ``attempt_id``,
        ``lifecycle_epoch``, ``provenance`` and ``source_provenance`` (a
        mapping or ``None``).
    """
    from phenotypic.sdk_._image_record import (
        PROVENANCE_FORWARD,
        PROVENANCE_MIGRATED,
        read_image_record,
        record_provenance,
    )

    active_generation = _active_migration_generation(output_dir)
    defaults: dict[str, Any] = {
        "work_id": work_id,
        "relative_image_path": f"{dataset}/{stem}",
        "mode": "full",
        "attempt_id": "migration",
        "lifecycle_epoch": active_generation or "migration",
        "provenance": PROVENANCE_FORWARD,
        "source_provenance": None,
    }

    def _keep_string_fields(source: dict[str, Any]) -> None:
        for field in (
            "relative_image_path",
            "mode",
            "attempt_id",
            "lifecycle_epoch",
        ):
            value = source.get(field)
            if (
                isinstance(value, str)
                and value
                and not (field == "lifecycle_epoch" and active_generation is not None)
            ):
                defaults[field] = value

    record = read_image_record(output_dir, dataset, stem)
    if record is not None and isinstance(record.get("work_id"), str):
        if record.get("dataset") != dataset or record.get("image_stem") != stem:
            return defaults
        provenance = record_provenance(record)
        if provenance != PROVENANCE_MIGRATED and record["work_id"] != work_id:
            return defaults
        defaults["work_id"] = record["work_id"]
        defaults["provenance"] = provenance
        source_provenance = record.get("source_provenance")
        if isinstance(source_provenance, dict):
            defaults["source_provenance"] = dict(source_provenance)
        _keep_string_fields(record)
        return defaults

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
    _keep_string_fields(marker)
    return defaults


def _certified_run_table(output_dir: Path, task: MigrationImageTask) -> Path | None:
    """The v2 run table the image's record certifies in *task*'s store, or ``None``.

    ``None`` when the image has no record, when its record (or legacy marker)
    certifies the v1 flat path -- a tree migrate has not converted, or one a
    crash left between the conversion and its republish -- or a table that no
    run of this store declares. Only then does migrate fall back to the
    record-less default (review C1, spec D6a).
    """
    from ._cli_recompile_recovery import (
        certified_table_run,
        image_authority_path,
        table_lies_in_store,
    )

    run = certified_table_run(
        output_dir, image_authority_path(output_dir, task.dataset, task.stem)
    )
    if run is None or not table_lies_in_store(run.measurements_path(), task.store_path):
        return None
    return run.measurements_path()


def _import_run_id(store: Path) -> str | None:
    """The run the legacy-Parquet import owns in v2 *store*, or ``None``.

    :func:`~phenotypic.sdk_._table_runs.import_table_run_id`, the rule
    ``replace_embedded_measurement_table`` writes by, so migrate's checks of
    an import judge the run the import writes (review M2). ``None`` for a
    store that is not (yet) v2, whose one table is the flat one, and for an
    unreadable root.
    """
    from phenotypic.sdk_ import ngff_
    from phenotypic.sdk_._table_runs import (
        import_table_run_id,
        tables_schema_of_block,
    )

    try:
        block = ngff_.read_phenotypic_attributes(Path(store))
        if not isinstance(block, dict) or tables_schema_of_block(block) != 2:
            return None
        return import_table_run_id(block)
    except (OSError, KeyError, ValueError):
        return None


def _migration_table_run_id(output_dir: Path, task: MigrationImageTask) -> str | None:
    """The run folder migrate validates and certifies for *task*, or ``None`` (the default).

    In order:

    1. the run the image's record certifies, when it certifies one of this
       store (review C1);
    2. for an image the manifest gave a legacy external Parquet, the run the
       import of it owns (review M2) -- where the import writes, or already wrote, the
       Parquet's rows, so validation, exactness and certification all judge
       one run;
    3. ``None``: the table contract's record-less default, the latest run
       holding ``measurements`` (or a v1 store's flat table).
    """
    table = _certified_run_table(output_dir, task)
    if table is not None:
        return table.parent.parent.name
    # Never stat the source here: a rerun over a valid marker must not touch
    # retained provenance at all. The manifest names a source only when it
    # found one.
    if task.measurement_path is not None:
        return _import_run_id(task.store_path)
    return None


def _embedded_table_path(output_dir: Path, task: MigrationImageTask) -> Path:
    """The measurement table migrate validates and certifies for *task*.

    The table of :func:`_migration_table_run_id`'s run. The record's comes
    first (review C1): a tree measured with A, then B, then A certifies
    folder 1 while folder 2 is the latest, and re-certifying the latest would
    silently move the image to pipeline B. Otherwise the record-less default
    (spec D6a): a v2 store's latest run folder holding ``measurements``, or a
    v1 flat store's one table (migrate reads a tree before converting it;
    spec D10). A store that declares none answers the v1 flat path, which is
    only ever probed for existence or named in an error.
    """
    from phenotypic.sdk_._table_runs import measurement_table_relative_path

    certified = _certified_run_table(output_dir, task)
    if certified is not None:
        return certified
    run_id = _migration_table_run_id(output_dir, task)
    if run_id is not None:
        return Path(task.store_path) / measurement_table_relative_path(run_id)
    declared = declared_measurement_table_path(task.store_path)
    return (
        declared
        if declared is not None
        else Path(task.store_path) / MEASUREMENT_TABLE_RELATIVE_PATH
    )


def _table_contract_holds(output_dir: Path, task: MigrationImageTask) -> bool:
    """Whether the run migrate certifies for *task* passes the table contract."""
    return _valid_embedded_measurement_contract(
        task.store_path, _migration_table_run_id(output_dir, task)
    )


def _source_table_matches(
    task: MigrationImageTask, prepared: PreparedEmbeddedMeasurementTable
) -> bool:
    """Whether the run the import owns holds exactly *prepared* (review M2)."""
    return embedded_measurement_table_matches(
        task.store_path, prepared, run_id=_import_run_id(task.store_path)
    )


def _migration_marker_artifacts(
    output_dir: Path,
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
        artifacts["measurements"] = _embedded_table_path(output_dir, task)
    return artifacts


def _active_migration_generation(output_dir: Path) -> str | None:
    """The live ``--mode migrate`` generation, or ``None`` when there is none."""
    from phenotypic._cli._cli_slurm_lifecycle import load_slurm_lifecycle

    lifecycle = load_slurm_lifecycle(output_dir)
    if (
        lifecycle is not None
        and lifecycle.get("active") is True
        and lifecycle.get("mode") == "migrate"
    ):
        return str(lifecycle["generation"])
    return None


def _convert_store_tables(
    output_dir: Path,
    task: MigrationImageTask,
    *,
    record_was_current: bool,
    commit_guard: CommitGuard | None,
) -> bool:
    """Convert one v1 flat store's tables to a run folder (#281 spec D10).

    The ``tables_v2`` step of the image task. It runs after every step of the
    task that can write the store's tables (the legacy-Parquet import writes
    v2 directly), so it converts only a store a forward release left in the
    v1 layout, and is a no-op on any other.

    Converting rewrites the store root, so the image's record no longer
    verifies. A record that was current just before is republished here,
    exactly as ``--mode measure`` republishes after its rewrite: every
    artifact rehashed, identity kept, and the measurements artifact moved to
    the run folder. A record that was not current is left to the marker
    stage, which publishes one from the migration's own evidence. The run
    proof follows at finalization.

    Returns:
        Whether the store was converted.
    """
    from phenotypic._cli._cli_recompile_tables import _republish_table_marker

    if not _holds_v1_tables(task.store_path):
        return False
    run = convert_tables_v1_to_v2(
        task.store_path, durable=None, commit_guard=commit_guard
    )
    if run is None:
        return False
    record_path = image_record_path(output_dir, task.dataset, task.stem)
    if record_was_current and record_path.is_file():
        _republish_table_marker(
            output_dir,
            record_path,
            commit_guard=commit_guard,
            lifecycle_epoch=_active_migration_generation(output_dir),
            measurements=(
                run.measurements_path() if run.has_measurements else None
            ),
        )
    return True


def _valid_migration_marker(
    output_dir: Path,
    task: MigrationImageTask,
    work_id: str,
    *,
    table_authoritative: bool | None = None,
) -> bool:
    """Return whether the RECORD binds exactly this task's migrated authority.

    **Reads the record, not ``task.marker_path``** -- this is a read-back of
    what ``publish_image_success`` just wrote, and P3's clean break moved that
    to ``images/<ds>/<stem>.json``. It used to work by accident of the two
    being the same file: the publisher overwrote the path it was handed, so
    reading that path back got fresh bytes.

    ``task.marker_path`` still means what it always meant -- **the legacy
    marker this task reads as input** -- and `:163`, the canonicality check
    in `_cli_migrate_manifest`, the `_stage_error` target and the manifest's
    JSON round-trip all keep using it. Nothing was renamed; four read-backs
    stopped borrowing an input path to find an output.
    """
    if table_authoritative is None:
        table_authoritative = _embedded_table_path(output_dir, task).is_file()
    expected = _migration_marker_artifacts(
        output_dir,
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
        record_path = image_record_path(output_dir, task.dataset, task.stem)
        marker = json.loads(record_path.read_text(encoding="utf-8"))
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
    output_dir: Path,
    task: MigrationImageTask,
    *,
    metadata_csv: Path | None,
    commit_guard: CommitGuard | None,
    require_source_equality: bool = False,
) -> tuple[bool, bool]:
    """Repair and return ``(valid, installed)`` for one embedded table."""
    valid = _table_contract_holds(output_dir, task)
    source = task.measurement_path
    prepared = None
    if (
        source is not None
        and (not valid or require_source_equality)
        and source.is_file()
    ):
        baseline = pd.read_parquet(source)
        prepared = prepare_embedded_measurement_table(baseline, metadata_csv)
    exact = prepared is not None and _source_table_matches(task, prepared)
    if prepared is not None and (not valid or (require_source_equality and not exact)):
        written = replace_embedded_measurement_table(
            task.store_path,
            prepared,
            commit_guard=commit_guard,
        )
        # Judge the run the import wrote, never a newer one of another
        # pipeline (review M2): a multi-run store otherwise re-imported on
        # every pass and failed the exactness check each time.
        if not _valid_embedded_measurement_contract(
            task.store_path, written.parent.parent.name
        ) or not _source_table_matches(task, prepared):
            raise RuntimeError("embedded measurement table validation failed")
        return _table_contract_holds(output_dir, task), True
    if _embedded_table_path(output_dir, task).exists() and not valid:
        raise RuntimeError("embedded measurement table validation failed")
    return valid, False


def _certified_source_provenance(
    output_dir: Path,
    task: MigrationImageTask,
    *,
    metadata_csv: Path | None,
) -> dict[str, object] | None:
    """Bind stable source bytes and exact prepared-table equality."""
    source = task.measurement_path
    if source is None:
        return None
    before = _source_artifact_state(source)
    if not before.exists:
        return None
    baseline = pd.read_parquet(source)
    prepared = prepare_embedded_measurement_table(baseline, metadata_csv)
    after = _source_artifact_state(source)
    if after != before:
        raise RuntimeError("measurement source changed during certification")
    if not _source_table_matches(task, prepared):
        raise RuntimeError(
            "embedded table does not exactly match the prepared source"
        )
    assert before.size is not None and before.sha256 is not None
    relative = source.resolve(strict=True).relative_to(output_dir.resolve())
    return {
        "path": relative.as_posix(),
        "size": before.size,
        "sha256": before.sha256,
    }


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
    table_valid = store_valid and _table_contract_holds(output_dir, task)
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
    # Read-back: the digest of what was just published, so a later reclaim
    # can prove nothing changed underneath it. The RECORD, for the same
    # reason as `_valid_migration_marker` -- see its docstring.
    marker_digest = (
        hashlib.sha256(
            image_record_path(
                output_dir, task.dataset, task.stem
            ).read_bytes()
        ).hexdigest()
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
        # A v1 flat store is converted (#281 spec D10) even when its record
        # is current, so it is work, not a skip.
        skipped=marker_valid and not _holds_v1_tables(task.store_path),
    )


def _holds_v1_tables(store: Path) -> bool:
    """Whether *store* still keeps its tables in the v1 flat layout."""
    from phenotypic.sdk_ import store_tables_schema

    try:
        return store_tables_schema(store) == 1
    except (OSError, KeyError, ValueError):
        return False


def _partial_image_result(
    task: MigrationImageTask,
    work_id: str,
    *,
    converted: bool,
    table_installed: bool,
    overlay_rendered: bool,
) -> MigrationImagePartialResult:
    """Return typed completed-pass evidence for an unfinished image."""
    return MigrationImagePartialResult(
        index=task.index,
        dataset=task.dataset,
        stem=task.stem,
        work_id=work_id,
        converted=converted,
        table_installed=table_installed,
        overlay_rendered=overlay_rendered,
    )


def _stage_error(
    task: MigrationImageTask,
    work_id: str,
    *,
    stage: str,
    target: Path,
    cause: Exception,
    converted: bool,
    table_installed: bool,
    overlay_rendered: bool,
) -> MigrationImageStageError:
    """Wrap one exception with its canonical pass category and partial result."""
    return MigrationImageStageError(
        MigrationImageStageFailure(
            stage=stage,
            target=Path(target),
            reason=f"{type(cause).__name__}: {cause}",
            partial=_partial_image_result(
                task,
                work_id,
                converted=converted,
                table_installed=table_installed,
                overlay_rendered=overlay_rendered,
            ),
        )
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
    try:
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
    except Exception as exc:
        raise _stage_error(
            task,
            work_id,
            stage="conversion",
            target=task.hdf_path or task.store_path,
            cause=exc,
            converted=converted,
            table_installed=table_installed,
            overlay_rendered=overlay_rendered,
        ) from exc

    table_was_authoritative = _table_contract_holds(output_dir, task)
    marker_was_valid = _valid_migration_marker(
        output_dir,
        task,
        work_id,
        table_authoritative=table_was_authoritative,
    )

    try:
        table_valid, table_installed = _table_state(
            output_dir,
            task,
            metadata_csv=metadata_csv,
            commit_guard=commit_guard,
            require_source_equality=not marker_was_valid,
        )
        if not table_valid and image.num_objects != 0:
            raise RuntimeError("nonempty migrated image has no valid measurement table")
    except Exception as exc:
        raise _stage_error(
            task,
            work_id,
            stage="table",
            target=(
                task.measurement_path
                or _embedded_table_path(output_dir, task)
            ),
            cause=exc,
            converted=converted,
            table_installed=table_installed,
            overlay_rendered=overlay_rendered,
        ) from exc

    # The `tables_v2` step (#281 spec D10), after every table write above.
    try:
        # The marker stage re-checks the table contract against `table_valid`
        # inside its commit, so a conversion that broke the table fails there.
        tables_converted = _convert_store_tables(
            output_dir,
            task,
            record_was_current=marker_was_valid,
            commit_guard=commit_guard,
        )
    except Exception as exc:
        raise _stage_error(
            task,
            work_id,
            stage="table",
            target=_embedded_table_path(output_dir, task),
            cause=exc,
            converted=converted,
            table_installed=table_installed,
            overlay_rendered=overlay_rendered,
        ) from exc

    try:
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
    except Exception as exc:
        raise _stage_error(
            task,
            work_id,
            stage="overlay",
            target=task.overlay_path,
            cause=exc,
            converted=converted,
            table_installed=table_installed,
            overlay_rendered=overlay_rendered,
        ) from exc

    try:
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
                output_dir,
                task,
                table_authoritative=table_valid,
            )
            with publication_commit(commit_guard):
                source_provenance = _certified_source_provenance(
                    output_dir,
                    task,
                    metadata_csv=metadata_csv,
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
                    provenance=identity["provenance"],
                    # A source this pass certified is the current evidence;
                    # otherwise the record's own bookkeeping is kept.
                    source_provenance=(
                        source_provenance
                        if source_provenance is not None
                        else identity["source_provenance"]
                    ),
                    commit_guard=None,
                )
                # The publisher's canonical path is the RECORD's, not
                # `task.marker_path`. Comparing against the task field made
                # this guard fire unconditionally after P3's clean break --
                # `--mode migrate` raised on the first image of every tree.
                # The guard itself is sound and is kept: it catches a
                # publisher returning somewhere unexpected, which is exactly
                # what it just caught.
                if marker_path != image_record_path(
                    output_dir, task.dataset, task.stem
                ):
                    raise RuntimeError(
                        "record publisher returned a non-canonical path"
                    )
                if not _valid_migration_marker(
                    output_dir,
                    task,
                    work_id,
                    table_authoritative=table_valid,
                ):
                    raise RuntimeError(
                        "measurement source changed before marker certification"
                    )

        with publication_commit(commit_guard):
            if not valid_staged_store(task.store_path):
                raise RuntimeError("migrated image store validation failed")
            if table_valid != _table_contract_holds(output_dir, task):
                raise RuntimeError("embedded measurement table validation failed")
            if not valid_migration_overlay(task.overlay_path, expected_shape):
                raise RuntimeError("migrated overlay validation failed")
            if not marker_valid:
                _certified_source_provenance(
                    output_dir,
                    task,
                    metadata_csv=metadata_csv,
                )
            if not _valid_migration_marker(
                output_dir,
                task,
                work_id,
                table_authoritative=table_valid,
            ):
                raise RuntimeError("migrated marker validation failed")
            # Read-back, and the one that CROSSES THE PROCESS BOUNDARY: this
            # value becomes `MigrationImageResult.marker_digest`, which the
            # controller re-derives and compares at six sites in
            # `_cli_migrate_manifest`. All seven must digest the same file or
            # the seal binds the wrong one -- silently, if they agree on the
            # wrong file, and loudly if they disagree.
            marker_digest = hashlib.sha256(
                image_record_path(
                    output_dir, task.dataset, task.stem
                ).read_bytes()
            ).hexdigest()
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
                    or tables_converted
                    or overlay_rendered
                    or not marker_valid
                ),
            )
    except Exception as exc:
        raise _stage_error(
            task,
            work_id,
            stage="marker",
            target=task.marker_path,
            cause=exc,
            converted=converted,
            table_installed=table_installed,
            overlay_rendered=overlay_rendered,
        ) from exc


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
    digest = hashlib.sha256()
    size = 0
    try:
        handle = path.open("rb")
    except PermissionError as exc:
        # Windows reports opening a directory as EACCES; name it the way
        # POSIX does, so a directory never reads as a permissions problem.
        if path.is_dir():
            raise IsADirectoryError(
                errno.EISDIR, os.strerror(errno.EISDIR), str(path)
            ) from exc
        raise
    except FileNotFoundError:
        return SourceArtifactState(
            path=path,
            exists=False,
            size=None,
            sha256=None,
        )
    with handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return SourceArtifactState(
        path=path,
        exists=True,
        size=size,
        sha256=digest.hexdigest(),
    )


def _current_marker_digest(
    output_dir: Path,
    task: MigrationImageTask,
    work_id: str,
) -> str:
    """Return the digest of the current RECORD payload, valid or invalid.

    **This one guards a deletion.** ``_marker_still_current`` is the check
    immediately before ``--delete-sources`` unlinks a user's ``.h5``, so a
    wrong answer here is the only place in this module where a mistake costs
    data rather than raising.

    It reads the record for the same reason as the other three read-backs,
    and the ordering it restores is worth naming: while the publisher and
    this reader disagreed, ``_valid_migration_marker`` failed first and
    reclaim refused -- correct, but for an unrelated reason. Now that both
    resolve the record, this digest comparison is load-bearing for the first
    time since the break, which is why
    ``test_reclaim_refuses_when_the_record_changed_after_publication``
    asserts the **source file still exists** rather than that a digest
    differs.
    """
    try:
        return hashlib.sha256(
            image_record_path(
                output_dir, task.dataset, task.stem
            ).read_bytes()
        ).hexdigest()
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
                task.store_path, _import_run_id(task.store_path)
            ) or not _source_table_matches(task, prepared):
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
                    elif not _source_table_matches(task, prepared):
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
        # Discovery legitimately yields a store-only task after a previous
        # generation reclaimed its sources.  The current marker has already
        # been revalidated above, so this is authenticated idempotent success.
        pass
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
