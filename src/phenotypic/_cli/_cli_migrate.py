"""``--mode migrate``: convert a legacy output tree, in place.

Migration runs ordered artifact passes followed by strict publication.  The
order is load-bearing:

===== =====================================================================
1     ``migrate_metadata_bundle`` over bundle-durable metadata targets --
      pipeline, named dataset aggregates, and standalone master tables.
2     per-image ``results/*/hdf/*.h5`` -> ``results/*/zarr/*.ome.zarr``.
3     external Parquets -> embedded tables, then missing overlay rendering.
4     image markers, aggregate rebuild and marker, then run completion.
===== =====================================================================

Per-image ``results/<ds>/measurements/<stem>.parquet`` files are Task-1
provenance, not pass-1 metadata targets and not marker-bound deliverables.
Pass 3 reads and canonicalizes them in memory, proves exact equality with the
embedded authoritative table when publishing a new or repaired marker, and
leaves retained source bytes unchanged.

Pass 1 excludes ``.h5`` targets **unconditionally** -- see
:data:`~phenotypic.sdk_._metadata_migration.NON_IMAGE_KINDS` for why that is
correct rather than merely cheaper.

**Local only.** No SLURM controller, no array, no chunking, no
``MaxArraySize`` accounting. Migration is one-time, resumable and restartable,
so it does not justify another scheduler surface.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
from pathlib import Path

import click

from ._cli_migrate_image import (
    MigrationImagePartialResult,
    MigrationImageResult,
    MigrationImageStageError,
    MigrationImageStageFailure,
    ReclaimResult,
    _configured_work_id,
    _existing_marker_identity,
    _migration_work_id,
    _source_artifact_state,
    migrate_image_task,
    reclaim_image_sources,
)
from ._cli_migrate_manifest import (
    MigrationImageSeal,
    MigrationImageTask,
    MigrationReclaimSeal,
    discover_migration_tasks,
    migration_image_seal_path,
    migration_reclaim_seal_path,
    migration_reclaim_status_path,
    migration_task_status_path,
    publish_migration_reclaim_status,
    publish_migration_task_status,
    seal_migration_image_stage,
    seal_migration_reclaim_stage,
    valid_migration_image_seal,
    valid_migration_reclaim_seal,
    write_migration_manifest,
)
from ._cli_completion import (
    publish_run_completion_evidence,
    valid_aggregate_snapshot,
    valid_run_completion,
)
from ._cli_slurm_lifecycle import (
    deactivate_generation,
    generation_publication_guard,
    initialize_slurm_lifecycle,
    mark_generation_failed,
    new_slurm_generation,
)
from ._embedded_measurement_tables import embedded_measurement_table_matches

from phenotypic.sdk_ import (
    BundleLayout,
    CommitGuard,
    MEASUREMENT_TABLE_RELATIVE_PATH,
    STORE_SUFFIX,
    aggregate_publication_marker_path,
    atomic_write_json,
    deliverables_dir,
    image_completion_marker_path,
    load_image_from_store,
    metadata_csv_deliverable_path,
    phenotypic_cache_dir,
    publication_commit,
    replace_embedded_measurement_table,
    run_completion_marker_path,
    store_stem,
    zarr_store_path,
)
from phenotypic.sdk_._hdf_to_zarr import (
    MigrationReport,
    canonical_metadata_view_path,
    emit_canonical_metadata_view,
    republish_aggregate,
)
from phenotypic.sdk_.ngff_ import valid_staged_store
from phenotypic.sdk_._metadata_migration import (
    BUNDLE_DURABLE_TARGET_ROLE,
    NON_IMAGE_KINDS,
    MetadataMigrationAuthority,
    MetadataMigrationReport,
    metadata_migration_authority,
    migrate_preflighted_metadata_bundle,
    preflight_metadata_schema,
    reconcile_metadata_migration_bundle,
)


class MigrateModeError(click.ClickException):
    """Raised when migration cannot proceed or did not finish cleanly."""


_MIGRATION_FAILURE_CATEGORIES = frozenset(
    {
        "metadata",
        "image",
        "image_seal",
        "reclaim_noop",
        "reclaim",
        "aggregate",
        "completion",
    }
)


@dataclass(frozen=True)
class MetadataPassResult:
    """Outcome of pass 1, including stable authority when it wrote."""

    headers_migrated: int
    failures: tuple[tuple[Path, str], ...]
    authority: MetadataMigrationAuthority | None


def migration_terminal_status_path(
    control_root: Path, generation: str
) -> Path:
    """Return the durable typed status path for one migration attempt."""
    return migration_image_seal_path(control_root, generation).with_name(
        "terminal_status.json"
    )


def _migration_report_payload(report: MigrationReport) -> dict[str, object]:
    """Return a JSON-safe, typed summary of one migration report."""
    failure_fields = (
        "failed",
        "header_failures",
        "table_failures",
        "overlay_failures",
        "publication_failures",
    )
    payload: dict[str, object] = {
        "converted": report.converted,
        "skipped": report.skipped,
        "headers_migrated": report.headers_migrated,
        "tables_migrated": report.tables_migrated,
        "tables_skipped": report.tables_skipped,
        "overlays_created": report.overlays_created,
        "overlays_skipped": report.overlays_skipped,
    }
    for field in failure_fields:
        payload[field] = [
            {"path": str(path), "reason": reason}
            for path, reason in getattr(report, field)
        ]
    return payload


def publish_migration_terminal_status(
    output_dir: Path,
    *,
    generation: str,
    succeeded: bool,
    failure_category: str | None,
    reason: str | None,
    report: MigrationReport,
    commit_guard: CommitGuard | None = None,
    control_root: Path | None = None,
) -> Path:
    """Atomically publish typed attempt status before lifecycle closure."""
    if succeeded:
        if failure_category is not None or reason is not None or not report.ok:
            raise ValueError("successful migration status cannot carry failure evidence")
    elif not failure_category or not reason:
        raise ValueError("failed migration status requires a category and reason")
    elif failure_category not in _MIGRATION_FAILURE_CATEGORIES:
        raise ValueError(f"unknown migration failure category: {failure_category}")
    status_root = (
        phenotypic_cache_dir(output_dir)
        if control_root is None
        else Path(control_root).resolve()
    )
    path = migration_terminal_status_path(status_root, generation)
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "generation": generation,
            "status": "succeeded" if succeeded else "failed",
            "failure_category": failure_category,
            "reason": reason,
            "report": _migration_report_payload(report),
            "completed_at": datetime.now(timezone.utc).isoformat(
                timespec="milliseconds"
            ),
        },
        commit_guard=commit_guard,
    )
    return path


def invalidate_migration_terminal_authority(
    output_dir: Path,
    *,
    commit_guard: CommitGuard | None,
) -> None:
    """Invalidate aggregate and run completion together under one fence."""
    with publication_commit(commit_guard):
        aggregate_publication_marker_path(output_dir).unlink(missing_ok=True)
        run_completion_marker_path(output_dir).unlink(missing_ok=True)


def close_migration_generation(
    output_dir: Path,
    *,
    generation: str,
    succeeded: bool,
    reason: str | None,
) -> None:
    """Close one generation only after its typed terminal status is durable."""
    if succeeded:
        if reason is not None:
            raise ValueError("successful migration closure cannot carry a reason")
        deactivate_generation(output_dir, generation)
        return
    if not reason:
        raise ValueError("failed migration closure requires an exact reason")
    mark_generation_failed(output_dir, generation, reason)


def _bundle_layout(output_dir: Path) -> BundleLayout:
    """Return the bundle layout for *output_dir*, constructed DIRECTLY.

    Never ``BundleLayout.detect`` and never a bare ``Path``: both route
    through ``_resolve_bundle``, which raises ``FileNotFoundError`` unless
    ``deliverables/master_measurements.parquet`` exists. A pre-aggregate or
    interrupted legacy run is precisely the migration subject, so that
    resolution would abort pass 1 on the trees migration exists for -- with an
    error message ("Point the viewer at a ``python -m phenotypic`` output
    dir") that names nothing relevant. ``migrate_metadata_schema_for_recompile``
    constructs the layout directly for exactly this reason.

    Args:
        output_dir: Run output root.

    Returns:
        The bundle layout.
    """
    resolved = Path(output_dir).resolve()
    return BundleLayout(
        deliverables_base=deliverables_dir(resolved), output_root=resolved
    )


def _file_sha256(path: Path) -> str | None:
    """Return one file digest, or None when the file is absent."""
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_migration_processing_state(
    output_dir: Path,
    *,
    tasks: Sequence[MigrationImageTask] | None = None,
    commit_guard: CommitGuard | None = None,
) -> None:
    """Reconstruct marker authority for a state-free legacy archive.

    Migration fixtures intentionally omit processing state: copying stale run
    state would make its inventory authoritative over a selected subset. A
    deterministic state synthesized from the converted stores lets the table
    pass publish image markers and the aggregate marker without inventing
    measurement tables for HDF-only stores. Existing state is never replaced.
    """
    from phenotypic._cli._cli_state_management import (
        load_processing_state,
        save_processing_state,
    )
    from phenotypic._cli._cli_types import DatasetState, ProcessingState

    output_dir = Path(output_dir)
    existing = load_processing_state(output_dir)
    raw_existing_work_ids = (
        existing.config.get("work_ids", {}) if existing is not None else {}
    )
    work_ids: dict[str, dict[str, str]] = {
        str(dataset): dict(images)
        for dataset, images in raw_existing_work_ids.items()
        if isinstance(images, dict)
    } if isinstance(raw_existing_work_ids, dict) else {}
    datasets: dict[str, DatasetState] = (
        dict(existing.datasets) if existing is not None else {}
    )
    inventory: dict[str, set[str]] = {}
    if tasks is not None:
        for task in tasks:
            inventory.setdefault(task.dataset, set()).add(task.stem)
    else:
        results = output_dir / "results"
        if not results.is_dir():
            return
        for dataset_dir in sorted(
            path for path in results.iterdir() if path.is_dir()
        ):
            stores = sorted((dataset_dir / "zarr").glob(f"*{STORE_SUFFIX}"))
            stems = {
                store.name[: -len(STORE_SUFFIX)]
                for store in stores
                if (store / "zarr.json").is_file()
            }
            if stems:
                inventory[dataset_dir.name] = stems

    for dataset_name, stems in sorted(inventory.items()):
        if not stems:
            continue
        existing_images = work_ids.get(dataset_name, {})
        if not isinstance(existing_images, dict):
            existing_images = {}
        work_ids[dataset_name] = dict(existing_images)
        for stem in sorted(stems):
            if not any(
                Path(str(image_name)).stem == stem
                for image_name in work_ids[dataset_name]
            ):
                work_ids[dataset_name][stem] = _migration_work_id(
                    dataset_name, stem
                )
        state_names = {
            next(
                (
                    str(image_name)
                    for image_name in work_ids[dataset_name]
                    if Path(str(image_name)).stem == stem
                ),
                stem,
            )
            for stem in stems
        }
        if dataset_name in datasets:
            dataset_state = datasets[dataset_name]
            if tasks is None:
                dataset_state.completed.update(state_names)
            dataset_state.initial_images.update(state_names)
            datasets[dataset_name] = dataset_state
        else:
            datasets[dataset_name] = DatasetState(
                completed=set(state_names) if tasks is None else set(),
                initial_images=set(state_names),
            )
    if not work_ids:
        return

    if existing is not None:
        existing.config["success_markers_required"] = True
        existing.config["work_ids"] = work_ids
        existing.datasets.update(datasets)
        existing.last_updated = datetime.now(timezone.utc)
        with publication_commit(commit_guard):
            save_processing_state(existing, output_dir)
        return

    provenance_candidates = [
        *sorted(output_dir.glob("*.pht-pipe")),
        deliverables_dir(output_dir) / "pipeline.json.pht-pipe",
    ]
    pipeline_path = next(
        (path for path in provenance_candidates if path.is_file()),
        output_dir / "pipeline.pht-pipe",
    )
    inventory = "\n".join(
        f"{dataset}/{stem}:{work_id}"
        for dataset, images in sorted(work_ids.items())
        for stem, work_id in sorted(images.items())
    )
    now = datetime.now(timezone.utc)
    metadata_snapshot = metadata_csv_deliverable_path(output_dir)
    state = ProcessingState(
        version="3.0.0",
        pipeline_path=pipeline_path,
        input_path=output_dir,
        output_dir=output_dir,
        timestamp=now,
        execution_mode="local",
        last_updated=now,
        datasets=datasets,
        config={
            "success_markers_required": True,
            "work_ids": work_ids,
            "processing_generation": hashlib.sha256(
                f"migration\n{inventory}".encode()
            ).hexdigest(),
            "pipeline_sha256": _file_sha256(pipeline_path),
            "metadata_sha256": _file_sha256(metadata_snapshot),
            "include_dataset_column": True,
            "no_qc": True,
            "process_only_layer": None,
        },
    )
    with publication_commit(commit_guard):
        save_processing_state(state, output_dir)


def _pending_header_targets(report: MetadataMigrationReport) -> int:
    """Return how many non-image targets pass 1 would rewrite."""
    return sum(1 for target in report.targets if target.status == "migratable")


def run_metadata_pass(
    output_dir: Path,
    *,
    dry_run: bool,
    commit_guard: CommitGuard | None = None,
) -> MetadataPassResult:
    """Run pass 1 -- the metadata-schema migration over non-image targets.

    ``preflight_metadata_schema`` writes nothing, which is what makes pass 1's
    dry run free; that is the mechanism, not an incidental property.

    Args:
        output_dir: Run output root.
        dry_run: Report what would be rewritten and write nothing.

    Returns:
        Counts, failures, and non-dry terminal metadata authority.

    Raises:
        MigrateModeError: The preflight is ``blocked`` -- conflicts a human
            must resolve -- or the apply did not succeed.
    """
    layout = _bundle_layout(output_dir)
    if not dry_run:
        reconciled = reconcile_metadata_migration_bundle(
            layout,
            kinds=NON_IMAGE_KINDS,
            target_role=BUNDLE_DURABLE_TARGET_ROLE,
            commit_guard=commit_guard,
        )
        if reconciled is not None:
            if reconciled.status not in {"compatible", "applied"}:
                return MetadataPassResult(
                    headers_migrated=0,
                    failures=tuple(
                        (Path(output_dir), conflict)
                        for conflict in (
                            reconciled.conflicts
                            or (
                                f"metadata migration {reconciled.status}",
                            )
                        )
                    ),
                    authority=None,
                )
            return MetadataPassResult(
                headers_migrated=len(reconciled.migrated_targets),
                failures=(),
                authority=metadata_migration_authority(layout),
            )
    report = preflight_metadata_schema(
        layout,
        kinds=NON_IMAGE_KINDS,
        target_role=BUNDLE_DURABLE_TARGET_ROLE,
    )
    if report.status == "blocked":
        raise MigrateModeError(
            "Metadata-schema conflicts must be resolved before migrating: "
            + ("; ".join(report.conflicts) or "no conflict details available")
        )
    if dry_run:
        return MetadataPassResult(
            headers_migrated=_pending_header_targets(report),
            failures=(),
            authority=None,
        )

    result = migrate_preflighted_metadata_bundle(
        layout,
        report=report,
        kinds=NON_IMAGE_KINDS,
        commit_guard=commit_guard,
    )
    # The RESULT's status, not the report's (ledger C11). `_report_from_targets`
    # can only return `blocked`, `migratable` or `compatible`; `"applied"` is
    # exclusively a result status. A legacy bundle is by definition
    # `"migratable"`, so testing the REPORT against {compatible, applied}
    # rejects precisely migration's intended input.
    if result.status not in {"compatible", "applied"}:
        return MetadataPassResult(
            headers_migrated=0,
            failures=tuple(
                (Path(output_dir), conflict)
                for conflict in (
                    result.conflicts
                    or (f"metadata migration {result.status}",)
                )
            ),
            authority=None,
        )
    return MetadataPassResult(
        headers_migrated=len(result.migrated_targets),
        failures=(),
        authority=metadata_migration_authority(layout),
    )


def _migrate_image_result(
    output_dir: Path,
    task: MigrationImageTask,
    *,
    metadata_csv: Path | None,
    overlay_alpha: float,
    dry_run: bool,
    commit_guard: CommitGuard | None,
) -> tuple[MigrationImageResult | None, MigrationImageStageFailure | None]:
    """Run one task while preserving a typed per-image failure."""
    try:
        result = migrate_image_task(
            output_dir,
            task,
            metadata_csv=metadata_csv,
            overlay_alpha=overlay_alpha,
            dry_run=dry_run,
            commit_guard=commit_guard,
        )
    except MigrationImageStageError as exc:
        return None, exc.failure
    except Exception as exc:  # noqa: BLE001 - isolate every manifest image
        target = task.hdf_path or task.store_path
        return None, MigrationImageStageFailure(
            stage="conversion",
            target=target,
            reason=f"{type(exc).__name__}: {exc}",
            partial=MigrationImagePartialResult(
                index=task.index,
                dataset=task.dataset,
                stem=task.stem,
                work_id=_migration_work_id(task.dataset, task.stem),
                converted=False,
                table_installed=False,
                overlay_rendered=False,
            ),
        )
    return result, None


def _execute_migration_tasks(
    output_dir: Path,
    *,
    tasks: Sequence[MigrationImageTask],
    metadata_csv: Path | None,
    overlay_alpha: float,
    dry_run: bool,
    njobs: int,
    commit_guard: CommitGuard | None,
) -> tuple[
    tuple[MigrationImageResult, ...],
    tuple[MigrationImageStageFailure, ...],
]:
    """Execute the one canonical inventory locally, using joblib when asked."""
    if njobs > 1 and len(tasks) > 1:
        from joblib import Parallel, delayed

        outcomes = Parallel(n_jobs=njobs)(
            delayed(_migrate_image_result)(
                output_dir,
                task,
                metadata_csv=metadata_csv,
                overlay_alpha=overlay_alpha,
                dry_run=dry_run,
                commit_guard=commit_guard,
            )
            for task in tasks
        )
    else:
        outcomes = [
            _migrate_image_result(
                output_dir,
                task,
                metadata_csv=metadata_csv,
                overlay_alpha=overlay_alpha,
                dry_run=dry_run,
                commit_guard=commit_guard,
            )
            for task in tasks
        ]
    results = tuple(result for result, _ in outcomes if result is not None)
    failures = tuple(failure for _, failure in outcomes if failure is not None)
    return results, failures


def _report_from_image_results(
    tasks: Sequence[MigrationImageTask],
    results: Sequence[MigrationImageResult],
    failures: Sequence[MigrationImageStageFailure] = (),
) -> MigrationReport:
    """Preserve legacy summary counters from canonical task results."""
    by_index = {result.index: result for result in results}
    partials = tuple(failure.partial for failure in failures)
    conversion_failures = tuple(
        (failure.target, failure.reason)
        for failure in failures
        if failure.stage in {"conversion", "marker"}
    )
    table_failures = tuple(
        (failure.target, failure.reason)
        for failure in failures
        if failure.stage == "table"
    )
    overlay_failures = tuple(
        (failure.target, failure.reason)
        for failure in failures
        if failure.stage == "overlay"
    )
    return MigrationReport(
        converted=(
            sum(result.converted for result in results)
            + sum(result.converted for result in partials)
        ),
        skipped=sum(result.skipped for result in results),
        tables_migrated=(
            sum(result.table_installed for result in results)
            + sum(result.table_installed for result in partials)
        ),
        tables_skipped=sum(
            task.measurement_path is not None
            and task.index in by_index
            and not by_index[task.index].table_installed
            for task in tasks
        ),
        overlays_created=(
            sum(result.overlay_rendered for result in results)
            + sum(result.overlay_rendered for result in partials)
        ),
        overlays_skipped=sum(not result.overlay_rendered for result in results),
        failed=conversion_failures,
        table_failures=table_failures,
        overlay_failures=overlay_failures,
    )


def _retained_reclaim_result(
    output_dir: Path,
    task: MigrationImageTask,
    result: MigrationImageResult | None,
) -> ReclaimResult:
    """Record an exact no-op when the image barrier forbids deletion."""
    try:
        marker_digest = hashlib.sha256(task.marker_path.read_bytes()).hexdigest()
    except OSError:
        marker_digest = ""
    hdf_state = _source_artifact_state(task.hdf_path)
    parquet_state = _source_artifact_state(task.measurement_path)
    intended = tuple(
        path for path in (task.hdf_path, task.measurement_path) if path is not None
    )
    retained = tuple(
        state.path
        for state in (hdf_state, parquet_state)
        if state.exists and state.path is not None
    )
    return ReclaimResult(
        index=task.index,
        dataset=task.dataset,
        stem=task.stem,
        work_id=(
            result.work_id
            if result is not None
            else _configured_work_id(output_dir, task.dataset, task.stem)
        ),
        marker_digest=marker_digest,
        intended_deletions=intended,
        hdf_prestate=hdf_state,
        parquet_prestate=parquet_state,
        observed_poststate=(hdf_state, parquet_state),
        deleted_paths=(),
        retained_paths=retained,
        reason="image seal was not clean; sources retained",
    )


def _publish_migration_aggregate(
    output_dir: Path,
    *,
    commit_guard: CommitGuard | None,
) -> None:
    """Build and validate aggregate authority through existing publishers."""
    from phenotypic._cli._cli_output_manager import aggregate_measurements

    datasets = sorted(
        path.name
        for path in (output_dir / "results").iterdir()
        if path.is_dir()
    ) if (output_dir / "results").is_dir() else []
    snapshot = metadata_csv_deliverable_path(output_dir)
    aggregate_path = aggregate_measurements(
        output_dir,
        datasets,
        metadata_csv=snapshot if snapshot.is_file() else None,
        no_qc=True,
        commit_guard=commit_guard,
    )
    embedded_tables_exist = any(
        (output_dir / "results").glob(
            "*/zarr/*.ome.zarr/tables/measurements/table.parquet"
        )
    )
    if aggregate_path is None and embedded_tables_exist:
        raise RuntimeError("aggregate rebuild produced no measurements")
    if not republish_aggregate(output_dir, commit_guard=commit_guard):
        raise RuntimeError("aggregate marker publication returned false")
    if valid_aggregate_snapshot(output_dir) is None:
        raise RuntimeError("aggregate marker validation failed")


def _append_publication_failure(
    report: MigrationReport,
    target: Path,
    reason: str,
) -> MigrationReport:
    """Append one terminal publication failure without losing prior evidence."""
    return replace(
        report,
        publication_failures=report.publication_failures + ((target, reason),),
    )


def finalize_migration_attempt(
    output_dir: Path,
    *,
    manifest_path: Path,
    expected_scientific_output: Path,
    generation: str,
    metadata_pass: MetadataPassResult,
    image_seal: MigrationImageSeal,
    reclaim_seal: MigrationReclaimSeal | None,
    deletion_requested: bool,
    dry_run: bool,
    report: MigrationReport,
    image_failures: tuple[tuple[Path, str], ...],
    reclaim_failures: tuple[tuple[Path, str], ...],
    commit_guard: CommitGuard | None,
    control_root: Path | None = None,
) -> MigrationReport:
    """Publish terminal migration science and close one owned generation."""
    categorized_image_failures = frozenset(
        report.failed + report.table_failures + report.overlay_failures
    )
    unreported_image_failures = tuple(
        failure
        for failure in image_failures
        if failure not in categorized_image_failures
    )
    if dry_run:
        return replace(
            report,
            headers_migrated=metadata_pass.headers_migrated,
            header_failures=metadata_pass.failures,
            failed=(
                report.failed + unreported_image_failures + reclaim_failures
            ),
        )

    output_dir = Path(output_dir)
    control_root = (
        phenotypic_cache_dir(output_dir)
        if control_root is None
        else Path(control_root).resolve()
    )
    final_report = replace(
        report,
        headers_migrated=metadata_pass.headers_migrated,
        header_failures=metadata_pass.failures,
        failed=report.failed + unreported_image_failures + reclaim_failures,
    )
    failure_category: str | None = None
    reason: str | None = None

    if metadata_pass.failures or metadata_pass.authority is None:
        failure_category = "metadata"
        reason = (
            metadata_pass.failures[0][1]
            if metadata_pass.failures
            else "metadata stage lacks terminal authority"
        )
    else:
        try:
            current_metadata = metadata_migration_authority(
                _bundle_layout(output_dir)
            )
        except Exception as exc:  # noqa: BLE001 - typed terminal failure
            failure_category = "metadata"
            reason = f"metadata authority validation failed: {type(exc).__name__}: {exc}"
            final_report = _append_publication_failure(
                final_report, metadata_pass.authority.status_path, reason
            )
        else:
            if (
                current_metadata.terminal_receipt_digest
                != metadata_pass.authority.terminal_receipt_digest
            ):
                failure_category = "metadata"
                reason = "metadata authority changed before finalization"
                final_report = _append_publication_failure(
                    final_report, metadata_pass.authority.status_path, reason
                )

    if failure_category is None and image_failures:
        failure_category = "image"
        reason = image_failures[0][1]

    if failure_category is None:
        from ._cli_migrate_manifest import _read_manifest

        try:
            _, manifest = _read_manifest(
                manifest_path,
                expected_scientific_output,
                expected_control_root=control_root,
            )
        except Exception as exc:  # noqa: BLE001 - typed terminal failure
            failure_category = "image_seal"
            reason = f"manifest validation failed: {type(exc).__name__}: {exc}"
            final_report = _append_publication_failure(
                final_report,
                migration_image_seal_path(control_root, generation),
                reason,
            )
        else:
            image_seal_valid = (
                manifest.generation == generation
                and image_seal.generation == generation
                and image_seal.manifest_digest == manifest.inventory_digest
                and metadata_pass.authority is not None
                and image_seal.metadata_terminal_digest
                == metadata_pass.authority.terminal_receipt_digest
                and valid_migration_image_seal(
                    control_root,
                    image_seal,
                    manifest_path=manifest_path,
                    expected_scientific_output=expected_scientific_output,
                )
            )
            if not image_seal.clean or not image_seal_valid:
                failure_category = "image_seal"
                reason = "; ".join(image_seal.failures) or (
                    "image seal is not current for this manifest and metadata authority"
                )
                final_report = _append_publication_failure(
                    final_report,
                    migration_image_seal_path(control_root, generation),
                    reason,
                )

    if failure_category is None and deletion_requested:
        if reclaim_failures:
            failure_category = "reclaim"
            reason = reclaim_failures[0][1]
        elif reclaim_seal is None:
            failure_category = "reclaim"
            reason = "source deletion requested without reclaim seal"
        elif not reclaim_seal.clean:
            joined = "; ".join(reclaim_seal.failures)
            failure_category = (
                "reclaim_noop" if "retained" in joined.lower() else "reclaim"
            )
            reason = joined or "reclaim seal is not clean"
        elif not valid_migration_reclaim_seal(
            control_root,
            reclaim_seal,
            manifest_path=manifest_path,
            expected_scientific_output=expected_scientific_output,
        ):
            failure_category = "reclaim"
            reason = "reclaim seal is not current for this manifest"
        if failure_category is not None:
            final_report = _append_publication_failure(
                final_report,
                migration_reclaim_seal_path(control_root, generation),
                reason or "reclaim authority failed",
            )
    elif failure_category is None and reclaim_seal is not None:
        failure_category = "reclaim"
        reason = "reclaim seal exists although source deletion was not requested"
        final_report = _append_publication_failure(
            final_report, reclaim_seal.seal_path, reason
        )

    if failure_category is None:
        try:
            emit_canonical_metadata_view(
                output_dir, commit_guard=commit_guard
            )
            _publish_migration_aggregate(
                output_dir, commit_guard=commit_guard
            )
        except Exception as exc:  # noqa: BLE001 - typed terminal failure
            failure_category = "aggregate"
            reason = (
                "aggregate publication failed: "
                f"{type(exc).__name__}: {exc}"
            )
            final_report = _append_publication_failure(
                final_report,
                aggregate_publication_marker_path(output_dir),
                reason,
            )

    if failure_category is None:
        try:
            publish_run_completion_evidence(
                output_dir,
                execution_epoch=generation,
                commit_guard=commit_guard,
            )
            if valid_run_completion(output_dir) is None:
                raise RuntimeError("run completion marker validation failed")
        except Exception as exc:  # noqa: BLE001 - typed terminal failure
            failure_category = "completion"
            reason = f"completion validation failed: {type(exc).__name__}: {exc}"
            final_report = _append_publication_failure(
                final_report,
                run_completion_marker_path(output_dir),
                reason,
            )

    succeeded = failure_category is None
    if succeeded and not final_report.ok:
        failure_category = "aggregate"
        reason = "migration report contains unclassified terminal failures"
        succeeded = False
    status_durable = False
    publish_migration_terminal_status(
        output_dir,
        generation=generation,
        succeeded=succeeded,
        failure_category=failure_category,
        reason=reason,
        report=final_report,
        commit_guard=commit_guard,
        control_root=control_root,
    )
    status_durable = True
    if status_durable:
        close_migration_generation(
            output_dir,
            generation=generation,
            succeeded=succeeded,
            reason=reason,
        )
    return final_report


def migrate_legacy_measurement_tables(
    output_dir: Path,
    *,
    dry_run: bool,
    delete_sources: bool,
    commit_guard: CommitGuard | None = None,
) -> tuple[int, int, tuple[tuple[Path, str], ...]]:
    """Install legacy per-image Parquets into corresponding image stores."""
    import pandas as pd

    from phenotypic._cli._embedded_measurement_tables import (
        prepare_embedded_measurement_table,
    )

    output_dir = Path(output_dir)
    metadata_snapshot = metadata_csv_deliverable_path(output_dir)
    metadata_csv = metadata_snapshot if metadata_snapshot.is_file() else None
    sources = sorted(
        path
        for path in (output_dir / "results").glob("*/measurements/*.parquet")
        if not path.name.startswith(("_", "."))
    )
    migrated = 0
    skipped = 0
    failures: list[tuple[Path, str]] = []
    for source in sources:
        dataset = source.parent.parent.name
        stem = source.stem
        store = zarr_store_path(output_dir, dataset, stem)
        embedded = store / MEASUREMENT_TABLE_RELATIVE_PATH
        if dry_run:
            if embedded.is_file():
                skipped += 1
            else:
                migrated += 1
            continue
        try:
            from phenotypic.sdk_._measurement_tables import (
                _valid_embedded_measurement_contract,
            )

            if _valid_embedded_measurement_contract(store):
                skipped += 1
            else:
                if not (store / "zarr.json").is_file():
                    raise FileNotFoundError(
                        f"No converted store exists for {source}"
                    )
                baseline = pd.read_parquet(source)
                prepared = prepare_embedded_measurement_table(
                    baseline, metadata_csv
                )
                replace_embedded_measurement_table(
                    store,
                    prepared,
                    commit_guard=commit_guard,
                )
                if not _valid_embedded_measurement_contract(store):
                    raise RuntimeError(
                        "embedded measurement table validation failed"
                    )
                migrated += 1

            if delete_sources:
                baseline = pd.read_parquet(source)
                prepared = prepare_embedded_measurement_table(
                    baseline, metadata_csv
                )
                if not embedded_measurement_table_matches(store, prepared):
                    raise RuntimeError(
                        "embedded table does not exactly match the external "
                        "Parquet source"
                    )
                with publication_commit(commit_guard):
                    if not embedded_measurement_table_matches(store, prepared):
                        raise RuntimeError(
                            "embedded table changed before external Parquet unlink"
                        )
                    source.unlink()
        except Exception as exc:
            failures.append((source, f"{type(exc).__name__}: {exc}"))
    return migrated, skipped, tuple(failures)


def publish_migrated_image_markers(
    output_dir: Path,
) -> tuple[int, tuple[tuple[Path, str], ...]]:
    """Publish one complete marker per valid migrated store.

    A missing measurement table is accepted only when the stored object map
    contains no objects.  Every marker binds the store and overlay, plus the
    embedded table when one exists.
    """
    from phenotypic._cli._cli_completion import (
        publish_image_success,
        valid_image_success,
    )
    from phenotypic._cli._cli_overlay_rendering import overlay_output_manager
    from phenotypic.sdk_._measurement_tables import (
        _valid_embedded_measurement_contract,
    )

    output_dir = Path(output_dir)
    manager = overlay_output_manager(output_dir, overlay_alpha=0.3)
    published = 0
    failures: list[tuple[Path, str]] = []
    for dataset_dir in sorted(
        path
        for path in (output_dir / "results").iterdir()
        if path.is_dir()
    ):
        zarr_dir = dataset_dir / "zarr"
        if not zarr_dir.is_dir():
            continue
        for store in sorted(zarr_dir.glob(f"*{STORE_SUFFIX}")):
            if store.name.startswith(".") or not valid_staged_store(store):
                continue
            stem = store_stem(store)
            marker_path = image_completion_marker_path(
                output_dir, dataset_dir.name, stem
            )
            try:
                overlay = manager.get_output_path(
                    dataset_dir.name, "overlays", stem
                )
                if not overlay.is_file():
                    raise FileNotFoundError(
                        f"Missing migrated overlay: {overlay}"
                    )
                embedded = store / MEASUREMENT_TABLE_RELATIVE_PATH
                table_valid = _valid_embedded_measurement_contract(store)
                if embedded.exists() and not table_valid:
                    raise RuntimeError(
                        "embedded measurement table validation failed"
                    )
                if not table_valid:
                    image = load_image_from_store(store)
                    if image.num_objects != 0:
                        raise RuntimeError(
                            "nonempty migrated image has no valid measurement table"
                        )
                work_id = _configured_work_id(
                    output_dir, dataset_dir.name, stem
                )
                identity = _existing_marker_identity(
                    output_dir, dataset_dir.name, stem, work_id
                )
                artifacts = {"store": store, "overlay": overlay}
                if table_valid:
                    artifacts["measurements"] = embedded
                marker = publish_image_success(
                    output_dir,
                    work_id=identity["work_id"],
                    dataset=dataset_dir.name,
                    relative_image_path=identity["relative_image_path"],
                    image_stem=stem,
                    mode=identity["mode"],
                    attempt_id=identity["attempt_id"],
                    lifecycle_epoch=identity["lifecycle_epoch"],
                    artifacts=artifacts,
                )
                if not marker.is_file() or not valid_image_success(
                    output_dir,
                    dataset=dataset_dir.name,
                    image_stem=stem,
                    work_id=work_id,
                ):
                    raise RuntimeError("migrated marker validation failed")
            except Exception as exc:  # noqa: BLE001 - report every image
                failures.append(
                    (marker_path, f"{type(exc).__name__}: {exc}")
                )
            else:
                published += 1
    return published, tuple(failures)


def render_migration_overlays(
    output_dir: Path,
    *,
    overlay_alpha: float,
    njobs: int,
    dry_run: bool,
) -> tuple[int, int, tuple[tuple[Path, str], ...]]:
    """Render only missing store-backed overlays for migration."""
    from phenotypic._cli._cli_overlay_rendering import (
        discover_missing_overlays,
        overlay_output_manager,
        render_overlay_work,
    )

    manager = overlay_output_manager(
        Path(output_dir), overlay_alpha=overlay_alpha
    )
    if dry_run:
        from phenotypic.sdk_._hdf_to_zarr import iter_legacy_hdfs

        candidates = {
            (dataset, hdf_path.stem)
            for dataset, hdf_path in iter_legacy_hdfs(output_dir)
        }
        results = Path(output_dir) / "results"
        if results.is_dir():
            for store in results.glob(f"*/zarr/*{STORE_SUFFIX}"):
                if store.name.startswith(".") or not store.is_dir():
                    continue
                candidates.add((store.parent.parent.name, store_stem(store)))
        missing = 0
        skipped = 0
        for dataset, stem in sorted(candidates):
            overlay = manager.get_output_path(dataset, "overlays", stem)
            if overlay.is_file():
                skipped += 1
            else:
                missing += 1
        return missing, skipped, ()
    try:
        work, skipped = discover_missing_overlays(output_dir, manager)
    except ValueError as exc:
        return 0, 0, ((Path(output_dir), f"ValueError: {exc}"),)
    report = render_overlay_work(
        work, output_manager=manager, n_jobs=njobs
    )
    return report.rendered, skipped, report.failures


def run_migrate(
    output_dir: Path,
    *,
    njobs: int = 1,
    overlay_alpha: float = 0.3,
    dry_run: bool = False,
    delete_sources: bool = False,
) -> MigrationReport:
    """Run both passes over *output_dir* and return the combined report.

    Args:
        output_dir: Run output root, converted in place.
        njobs: Worker processes for the per-image conversion pass.
        overlay_alpha: Alpha used for newly rendered overlay PNGs.
        dry_run: Report both passes and write nothing.
        delete_sources: Reclaim space by deleting each ``.h5`` whose
            conversion is provably faithful. The only irreversible step here.

    Returns:
        A combined :class:`MigrationReport` covering both passes.

    Raises:
        MigrateModeError: Pass 1 is blocked.
    """
    output_dir = Path(output_dir)
    tasks = tuple(discover_migration_tasks(output_dir))
    metadata_snapshot = metadata_csv_deliverable_path(output_dir)
    metadata_csv = metadata_snapshot if metadata_snapshot.is_file() else None

    if dry_run:
        metadata_pass = run_metadata_pass(output_dir, dry_run=True)
        results, stage_failures = _execute_migration_tasks(
            output_dir,
            tasks=tasks,
            metadata_csv=metadata_csv,
            overlay_alpha=overlay_alpha,
            dry_run=True,
            njobs=njobs,
            commit_guard=None,
        )
        report = _report_from_image_results(tasks, results, stage_failures)
        return replace(
            report,
            headers_migrated=metadata_pass.headers_migrated,
            header_failures=metadata_pass.failures,
        )

    generation = new_slurm_generation()
    initialize_slurm_lifecycle(
        output_dir,
        generation=generation,
        mode="migrate",
    )

    def commit_guard():
        return generation_publication_guard(output_dir, generation)

    scientific_output = deliverables_dir(output_dir)
    manifest_path = phenotypic_cache_dir(output_dir) / "migration_manifest.json"
    try:
        invalidate_migration_terminal_authority(
            output_dir,
            commit_guard=commit_guard,
        )
    except Exception as exc:  # noqa: BLE001 - terminalize owned setup failure
        reason = (
            "terminal authority invalidation failed: "
            f"{type(exc).__name__}: {exc}"
        )
        report = MigrationReport(
            publication_failures=(
                (aggregate_publication_marker_path(output_dir), reason),
            )
        )
        publish_migration_terminal_status(
            output_dir,
            generation=generation,
            succeeded=False,
            failure_category="aggregate",
            reason=reason,
            report=report,
            commit_guard=commit_guard,
        )
        close_migration_generation(
            output_dir,
            generation=generation,
            succeeded=False,
            reason=reason,
        )
        return report
    try:
        with publication_commit(commit_guard):
            manifest = write_migration_manifest(
                output_dir,
                generation=generation,
                scientific_output=scientific_output,
                tasks=tasks,
            )
    except Exception as exc:  # noqa: BLE001 - terminalize owned setup failure
        reason = f"manifest publication failed: {type(exc).__name__}: {exc}"
        report = MigrationReport(
            publication_failures=((manifest_path, reason),)
        )
        publish_migration_terminal_status(
            output_dir,
            generation=generation,
            succeeded=False,
            failure_category="image_seal",
            reason=reason,
            report=report,
            commit_guard=commit_guard,
        )
        close_migration_generation(
            output_dir,
            generation=generation,
            succeeded=False,
            reason=reason,
        )
        return report

    try:
        metadata_pass = run_metadata_pass(
            output_dir,
            dry_run=False,
            commit_guard=commit_guard,
        )
    except Exception as exc:  # noqa: BLE001 - terminalize metadata failure
        metadata_pass = MetadataPassResult(
            headers_migrated=0,
            failures=(
                (
                    output_dir,
                    f"{type(exc).__name__}: {exc}",
                ),
            ),
            authority=None,
        )

    results: tuple[MigrationImageResult, ...] = ()
    stage_failures: tuple[MigrationImageStageFailure, ...] = ()
    image_failures: tuple[tuple[Path, str], ...] = ()
    if not metadata_pass.failures and metadata_pass.authority is not None:
        try:
            _ensure_migration_processing_state(
                output_dir,
                tasks=tasks,
                commit_guard=commit_guard,
            )
        except Exception as exc:  # noqa: BLE001 - terminalize image setup
            image_failures = (
                (
                    output_dir,
                    f"image state preparation failed: {type(exc).__name__}: {exc}",
                ),
            )
        else:
            results, stage_failures = _execute_migration_tasks(
                output_dir,
                tasks=tasks,
                metadata_csv=metadata_csv,
                overlay_alpha=overlay_alpha,
                dry_run=False,
                njobs=njobs,
                commit_guard=commit_guard,
            )
            status_failures = [
                (failure.target, failure.reason) for failure in stage_failures
            ]
            for result in results:
                try:
                    publish_migration_task_status(
                        phenotypic_cache_dir(output_dir),
                        manifest_path=manifest_path,
                        expected_scientific_output=scientific_output,
                        generation=generation,
                        metadata_terminal_digest=(
                            metadata_pass.authority.terminal_receipt_digest
                        ),
                        result=result,
                        commit_guard=commit_guard,
                    )
                except Exception as exc:  # noqa: BLE001 - isolate status failure
                    status_failures.append(
                        (
                            migration_task_status_path(
                                phenotypic_cache_dir(output_dir),
                                generation,
                                result.index,
                            ),
                            f"status publication failed: {type(exc).__name__}: {exc}",
                        )
                    )
            image_failures = tuple(status_failures)

    metadata_digest = (
        metadata_pass.authority.terminal_receipt_digest
        if metadata_pass.authority is not None
        else "missing"
    )
    try:
        image_seal = seal_migration_image_stage(
            phenotypic_cache_dir(output_dir),
            manifest_path=manifest_path,
            expected_scientific_output=scientific_output,
            generation=generation,
            metadata_terminal_digest=metadata_digest,
            commit_guard=commit_guard,
        )
    except Exception as exc:  # noqa: BLE001 - preserve terminal seal evidence
        seal_path = migration_image_seal_path(
            phenotypic_cache_dir(output_dir), generation
        )
        image_seal = MigrationImageSeal(
            generation=generation,
            manifest_digest=manifest.inventory_digest,
            ordered_status_digest=hashlib.sha256(b"").hexdigest(),
            metadata_terminal_digest=metadata_digest,
            clean=False,
            failures=(f"image sealing failed: {type(exc).__name__}: {exc}",),
            seal_path=seal_path,
        )

    reclaim_failures: list[tuple[Path, str]] = []
    reclaim_seal: MigrationReclaimSeal | None = None
    if delete_sources:
        result_by_index = {result.index: result for result in results}
        for task in tasks:
            try:
                reclaim_result = (
                    reclaim_image_sources(
                        output_dir,
                        task,
                        metadata_csv=metadata_csv,
                        commit_guard=commit_guard,
                    )
                    if image_seal.clean
                    else _retained_reclaim_result(
                        output_dir,
                        task,
                        result_by_index.get(task.index),
                    )
                )
            except Exception as exc:  # noqa: BLE001 - retain on reclaim failure
                reclaim_failures.append(
                    (
                        task.hdf_path or task.store_path,
                        f"reclaim failed: {type(exc).__name__}: {exc}",
                    )
                )
                reclaim_result = _retained_reclaim_result(
                    output_dir,
                    task,
                    result_by_index.get(task.index),
                )
            try:
                publish_migration_reclaim_status(
                    phenotypic_cache_dir(output_dir),
                    manifest_path=manifest_path,
                    expected_scientific_output=scientific_output,
                    generation=generation,
                    result=reclaim_result,
                    commit_guard=commit_guard,
                )
            except Exception as exc:  # noqa: BLE001 - seal records missing status
                reclaim_failures.append(
                    (
                        migration_reclaim_status_path(
                            phenotypic_cache_dir(output_dir),
                            generation,
                            task.index,
                        ),
                        f"reclaim status publication failed: "
                        f"{type(exc).__name__}: {exc}",
                    )
                )
        try:
            reclaim_seal = seal_migration_reclaim_stage(
                phenotypic_cache_dir(output_dir),
                manifest_path=manifest_path,
                expected_scientific_output=scientific_output,
                generation=generation,
                deletion_requested=True,
                image_seal=image_seal,
                commit_guard=commit_guard,
            )
        except Exception as exc:  # noqa: BLE001 - terminalize reclaim failure
            reclaim_failures.append(
                (
                    migration_reclaim_seal_path(
                        phenotypic_cache_dir(output_dir), generation
                    ),
                    f"reclaim sealing failed: {type(exc).__name__}: {exc}",
                )
            )

    report = _report_from_image_results(tasks, results, stage_failures)
    return finalize_migration_attempt(
        output_dir,
        manifest_path=manifest_path,
        expected_scientific_output=scientific_output,
        generation=generation,
        metadata_pass=metadata_pass,
        image_seal=image_seal,
        reclaim_seal=reclaim_seal,
        deletion_requested=delete_sources,
        dry_run=False,
        report=report,
        image_failures=image_failures,
        reclaim_failures=tuple(reclaim_failures),
        commit_guard=commit_guard,
    )


def echo_migration_summary(
    output_dir: Path, report: MigrationReport, *, dry_run: bool
) -> None:
    """Print a summary that names all three migration passes.

    A summary reporting only the per-image conversion hides a pass-1 failure
    entirely, and the phase's dry-run criterion is stated per pass.
    """
    verb = "would migrate" if dry_run else "migrated"
    click.echo("")
    click.echo(f"--mode migrate: {output_dir}")
    click.echo(
        f"  Pass 1 (metadata headers, non-image targets): "
        f"{verb} {report.headers_migrated} target(s)"
    )
    click.echo(
        f"  Pass 2 (per-image .h5 -> .ome.zarr): "
        f"{'would convert' if dry_run else 'converted'} "
        f"{report.converted}, skipped {report.skipped}"
    )
    click.echo(
        "  Pass 3 (external Parquet -> embedded table): "
        f"{verb} {report.tables_migrated}, skipped {report.tables_skipped}"
    )
    click.echo(
        "  Pass 4 (store -> overlay PNG): "
        f"{'would render' if dry_run else 'rendered'} "
        f"{report.overlays_created}, preserved {report.overlays_skipped}"
    )
    view = canonical_metadata_view_path(output_dir)
    if view.is_file():
        click.echo(f"  Canonical metadata view: {view}")
    for target, reason in report.header_failures:
        click.echo(f"  Pass 1 FAILED {target}: {reason}", err=True)
    for source, reason in report.failed:
        click.echo(f"  Pass 2 FAILED {source}: {reason}", err=True)
    for source, reason in report.table_failures:
        click.echo(f"  Pass 3 FAILED {source}: {reason}", err=True)
    for overlay, reason in report.overlay_failures:
        click.echo(f"  Pass 4 FAILED {overlay}: {reason}", err=True)
    for target, reason in report.publication_failures:
        click.echo(f"  Publication FAILED {target}: {reason}", err=True)


def handle_migrate_mode(
    output_dir: Path,
    *,
    njobs: int = 1,
    overlay_alpha: float = 0.3,
    dry_run: bool = False,
    delete_sources: bool = False,
) -> int:
    """Run ``--mode migrate`` and return the process exit code.

    Args:
        output_dir: Run output root, converted in place.
        njobs: Worker processes for the per-image conversion pass.
        overlay_alpha: Alpha used for newly rendered overlay PNGs.
        dry_run: Report both passes and write nothing.
        delete_sources: Delete each provably-faithful source after conversion.

    Returns:
        ``0`` when both passes were clean, ``1`` otherwise.
    """
    report = run_migrate(
        output_dir,
        njobs=njobs,
        overlay_alpha=overlay_alpha,
        dry_run=dry_run,
        delete_sources=delete_sources,
    )
    echo_migration_summary(output_dir, report, dry_run=dry_run)
    return 0 if report.ok else 1


__all__ = [
    "MigrateModeError",
    "echo_migration_summary",
    "handle_migrate_mode",
    "run_metadata_pass",
    "migrate_legacy_measurement_tables",
    "run_migrate",
]
