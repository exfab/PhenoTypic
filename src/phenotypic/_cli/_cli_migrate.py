"""``--mode migrate``: convert a legacy output tree, in place.

Migration runs ordered artifact passes followed by strict publication.  The
order is load-bearing:

===== =====================================================================
1     ``migrate_metadata_bundle`` over every **non-image** target -- the
      per-dataset ``measurements/*.parquet`` and the pipeline JSON.
2     per-image ``results/*/hdf/*.h5`` -> ``results/*/zarr/*.ome.zarr``.
3     external Parquets -> embedded tables, then missing overlay rendering.
4     image markers, aggregate rebuild and marker, then run completion.
===== =====================================================================

Images-first is wrong. Pass 1 rewrites ``results/<ds>/measurements/*.parquet``,
and those are **marker-bound**: every per-image completion marker carries that
parquet's ``size`` and ``sha256``. Rewriting them *after* the markers were
republished invalidates every marker just written -- the exact failure the
republication exists to prevent, on the default path. Canonicalizing first and
publishing markers over the **final** bytes needs no bridge, no receipts, and
no ordering hazard (ledger MIG-15, FLOW TRACE-4).

Pass 1 excludes ``.h5`` targets **unconditionally** -- see
:data:`~phenotypic.sdk_._metadata_migration.NON_IMAGE_KINDS` for why that is
correct rather than merely cheaper.

**Local only.** No SLURM controller, no array, no chunking, no
``MaxArraySize`` accounting. Migration is one-time, resumable and restartable,
so it does not justify another scheduler surface.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
from pathlib import Path

import click

from ._cli_migrate_image import (
    _configured_work_id,
    _existing_marker_identity,
    _migration_work_id,
)
from ._embedded_measurement_tables import embedded_measurement_table_matches

from phenotypic.sdk_ import (
    BundleLayout,
    CommitGuard,
    MEASUREMENT_TABLE_RELATIVE_PATH,
    STORE_SUFFIX,
    aggregate_publication_marker_path,
    deliverables_dir,
    image_completion_marker_path,
    load_image_from_store,
    metadata_csv_deliverable_path,
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
    migrate_run_hdf_to_zarr,
    republish_aggregate,
)
from phenotypic.sdk_.ngff_ import valid_staged_store
from phenotypic.sdk_._metadata_migration import (
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


@dataclass(frozen=True)
class MetadataPassResult:
    """Outcome of pass 1, including stable authority when it wrote."""

    headers_migrated: int
    failures: tuple[tuple[Path, str], ...]
    authority: MetadataMigrationAuthority | None


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


def _ensure_migration_processing_state(output_dir: Path) -> None:
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
    results = output_dir / "results"
    if not results.is_dir():
        return
    for dataset_dir in sorted(
        path for path in results.iterdir() if path.is_dir()
    ):
        stores = sorted((dataset_dir / "zarr").glob(f"*{STORE_SUFFIX}"))
        if not stores:
            continue
        stems = {
            store.name[: -len(STORE_SUFFIX)]
            for store in stores
            if (store / "zarr.json").is_file()
        }
        if not stems:
            continue
        existing_images = work_ids.get(dataset_dir.name, {})
        if not isinstance(existing_images, dict):
            existing_images = {}
        work_ids[dataset_dir.name] = dict(existing_images)
        for stem in sorted(stems):
            if not any(
                Path(str(image_name)).stem == stem
                for image_name in work_ids[dataset_dir.name]
            ):
                work_ids[dataset_dir.name][stem] = _migration_work_id(
                    dataset_dir.name, stem
                )
        state_names = {
            next(
                (
                    str(image_name)
                    for image_name in work_ids[dataset_dir.name]
                    if Path(str(image_name)).stem == stem
                ),
                stem,
            )
            for stem in stems
        }
        if dataset_dir.name in datasets:
            dataset_state = datasets[dataset_dir.name]
            dataset_state.completed.update(state_names)
            dataset_state.initial_images.update(state_names)
            datasets[dataset_dir.name] = dataset_state
        else:
            datasets[dataset_dir.name] = DatasetState(
                completed=set(stems),
                initial_images=set(stems),
            )
    if not work_ids:
        return

    if existing is not None:
        existing.config["success_markers_required"] = True
        existing.config["work_ids"] = work_ids
        existing.datasets.update(datasets)
        existing.last_updated = datetime.now(timezone.utc)
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
    save_processing_state(state, output_dir)


def _pending_header_targets(report: MetadataMigrationReport) -> int:
    """Return how many non-image targets pass 1 would rewrite."""
    return sum(1 for target in report.targets if target.status == "migratable")


def run_metadata_pass(output_dir: Path, *, dry_run: bool) -> MetadataPassResult:
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
            layout, kinds=NON_IMAGE_KINDS
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
    report = preflight_metadata_schema(layout, kinds=NON_IMAGE_KINDS)
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
    if not dry_run:
        # Invalidate prior terminal authority before any migration mutation.
        # A failed or empty rebuild must never re-certify stale deliverables.
        aggregate_publication_marker_path(output_dir).unlink(missing_ok=True)
        run_completion_marker_path(output_dir).unlink(missing_ok=True)
    metadata_pass = run_metadata_pass(output_dir, dry_run=dry_run)
    headers_migrated = metadata_pass.headers_migrated
    header_failures = metadata_pass.failures
    report = migrate_run_hdf_to_zarr(
        output_dir,
        keep_source=True,
        njobs=njobs,
        dry_run=dry_run,
        finalize_publication=False,
    )
    if not dry_run:
        _ensure_migration_processing_state(output_dir)
    tables_migrated, tables_skipped, table_failures = (
        migrate_legacy_measurement_tables(
            output_dir,
            dry_run=dry_run,
            delete_sources=delete_sources,
        )
    )
    overlays_created, overlays_skipped, overlay_failures = (
        render_migration_overlays(
            output_dir,
            overlay_alpha=overlay_alpha,
            njobs=njobs,
            dry_run=dry_run,
        )
    )
    hdf_failures = list(report.failed)
    publication_failures: list[tuple[Path, str]] = []
    if not dry_run:
        try:
            # This additive metadata artifact is part of migration output, so
            # finish it before publishing any success authority. Terminal run
            # completion must remain the final write in a clean migration.
            emit_canonical_metadata_view(output_dir)
        except Exception as exc:  # noqa: BLE001 - report publication
            publication_failures.append(
                (
                    canonical_metadata_view_path(output_dir),
                    "canonical metadata view failed: "
                    f"{type(exc).__name__}: {exc}",
                )
            )
        artifact_failures = (
            bool(hdf_failures)
            or bool(header_failures)
            or bool(table_failures)
            or bool(overlay_failures)
        )
        if not artifact_failures and not publication_failures:
            _, marker_failures = publish_migrated_image_markers(output_dir)
            publication_failures.extend(marker_failures)
        if (
            delete_sources
            and not artifact_failures
            and not publication_failures
        ):
            from phenotypic.sdk_._hdf_to_zarr import _reclaim_sources

            hdf_failures.extend(_reclaim_sources(output_dir))
        if not artifact_failures and not publication_failures and not hdf_failures:
            try:
                from phenotypic._cli._cli_output_manager import (
                    aggregate_measurements,
                )

                datasets = sorted(
                    path.name
                    for path in (output_dir / "results").iterdir()
                    if path.is_dir()
                )
                snapshot = metadata_csv_deliverable_path(output_dir)
                aggregate_path = aggregate_measurements(
                    output_dir,
                    datasets,
                    metadata_csv=snapshot if snapshot.is_file() else None,
                    no_qc=True,
                )
                embedded_tables_exist = any(
                    (output_dir / "results").glob(
                        "*/zarr/*.ome.zarr/tables/measurements/table.parquet"
                    )
                )
                if aggregate_path is None and embedded_tables_exist:
                    raise RuntimeError(
                        "aggregate rebuild produced no measurements"
                    )
                if not republish_aggregate(output_dir):
                    raise RuntimeError(
                        "aggregate marker publication returned false"
                    )
            except Exception as exc:  # noqa: BLE001 - report publication
                publication_failures.append(
                    (
                        aggregate_publication_marker_path(output_dir),
                        "aggregate publication failed: "
                        f"{type(exc).__name__}: {exc}",
                    )
                )
        if (
            not artifact_failures
            and not hdf_failures
            and not publication_failures
        ):
            try:
                from phenotypic._cli._cli_completion import (
                    publish_run_completion_evidence,
                    valid_run_completion,
                )

                completion_path = publish_run_completion_evidence(
                    output_dir, execution_epoch="local"
                )
                if valid_run_completion(output_dir) is None:
                    raise RuntimeError(
                        "run completion marker validation failed"
                    )
            except Exception as exc:  # noqa: BLE001 - report publication
                run_completion_marker_path(output_dir).unlink(missing_ok=True)
                publication_failures.append(
                    (
                        run_completion_marker_path(output_dir),
                        f"{type(exc).__name__}: {exc}",
                    )
                )
            else:
                if completion_path != run_completion_marker_path(output_dir):
                    run_completion_marker_path(output_dir).unlink(missing_ok=True)
                    publication_failures.append(
                        (
                            completion_path,
                            "completion publisher returned an unexpected path",
                        )
                    )
    return replace(
        report,
        failed=tuple(hdf_failures),
        headers_migrated=headers_migrated,
        header_failures=header_failures,
        tables_migrated=tables_migrated,
        tables_skipped=tables_skipped,
        table_failures=table_failures,
        overlays_created=overlays_created,
        overlays_skipped=overlays_skipped,
        overlay_failures=overlay_failures,
        publication_failures=tuple(publication_failures),
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
