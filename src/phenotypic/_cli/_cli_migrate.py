"""``--mode migrate``: convert a legacy output tree, in place.

Two passes, in this order, and **the order is load-bearing**:

===== =====================================================================
1     ``migrate_metadata_bundle`` over every **non-image** target -- the
      per-dataset ``measurements/*.parquet`` and the pipeline JSON.
2     per-image ``results/*/hdf/*.h5`` -> ``results/*/zarr/*.ome.zarr``, then
      the marker republication, then the aggregate republish **last**.
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

from dataclasses import replace
from pathlib import Path

import click

from phenotypic.sdk_ import BundleLayout, deliverables_dir
from phenotypic.sdk_._hdf_to_zarr import (
    MigrationReport,
    canonical_metadata_view_path,
    migrate_run_hdf_to_zarr,
)
from phenotypic.sdk_._metadata_migration import (
    NON_IMAGE_KINDS,
    MetadataMigrationReport,
    migrate_metadata_bundle,
    preflight_metadata_schema,
)


class MigrateModeError(click.ClickException):
    """Raised when migration cannot proceed or did not finish cleanly."""


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


def _pending_header_targets(report: MetadataMigrationReport) -> int:
    """Return how many non-image targets pass 1 would rewrite."""
    return sum(1 for target in report.targets if target.status == "migratable")


def run_metadata_pass(output_dir: Path, *, dry_run: bool) -> tuple[int, tuple]:
    """Run pass 1 -- the metadata-schema migration over non-image targets.

    ``preflight_metadata_schema`` writes nothing, which is what makes pass 1's
    dry run free; that is the mechanism, not an incidental property.

    Args:
        output_dir: Run output root.
        dry_run: Report what would be rewritten and write nothing.

    Returns:
        ``(headers_migrated, header_failures)``.

    Raises:
        MigrateModeError: The preflight is ``blocked`` -- conflicts a human
            must resolve -- or the apply did not succeed.
    """
    layout = _bundle_layout(output_dir)
    report = preflight_metadata_schema(layout, kinds=NON_IMAGE_KINDS)
    if report.status == "blocked":
        raise MigrateModeError(
            "Metadata-schema conflicts must be resolved before migrating: "
            + ("; ".join(report.conflicts) or "no conflict details available")
        )
    if dry_run:
        return _pending_header_targets(report), ()

    result = migrate_metadata_bundle(
        layout,
        expected_plan_fingerprint=report.plan_fingerprint,
        kinds=NON_IMAGE_KINDS,
    )
    # The RESULT's status, not the report's (ledger C11). `_report_from_targets`
    # can only return `blocked`, `migratable` or `compatible`; `"applied"` is
    # exclusively a result status. A legacy bundle is by definition
    # `"migratable"`, so testing the REPORT against {compatible, applied}
    # rejects precisely migration's intended input.
    if result.status not in {"compatible", "applied"}:
        return 0, tuple(
            (Path(output_dir), conflict)
            for conflict in (
                result.conflicts or (f"metadata migration {result.status}",)
            )
        )
    return len(result.migrated_targets), ()


def run_migrate(
    output_dir: Path,
    *,
    njobs: int = 1,
    dry_run: bool = False,
    delete_sources: bool = False,
) -> MigrationReport:
    """Run both passes over *output_dir* and return the combined report.

    Args:
        output_dir: Run output root, converted in place.
        njobs: Worker processes for the per-image conversion pass.
        dry_run: Report both passes and write nothing.
        delete_sources: Reclaim space by deleting each ``.h5`` whose
            conversion is provably faithful. The only irreversible step here.

    Returns:
        A combined :class:`MigrationReport` covering both passes.

    Raises:
        MigrateModeError: Pass 1 is blocked.
    """
    output_dir = Path(output_dir)
    headers_migrated, header_failures = run_metadata_pass(
        output_dir, dry_run=dry_run
    )
    report = migrate_run_hdf_to_zarr(
        output_dir,
        keep_source=not delete_sources,
        njobs=njobs,
        dry_run=dry_run,
    )
    return replace(
        report,
        headers_migrated=headers_migrated,
        header_failures=header_failures,
    )


def echo_migration_summary(
    output_dir: Path, report: MigrationReport, *, dry_run: bool
) -> None:
    """Print a summary that names **both** passes.

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
    view = canonical_metadata_view_path(output_dir)
    if view.is_file():
        click.echo(f"  Canonical metadata view: {view}")
    for target, reason in report.header_failures:
        click.echo(f"  Pass 1 FAILED {target}: {reason}", err=True)
    for source, reason in report.failed:
        click.echo(f"  Pass 2 FAILED {source}: {reason}", err=True)


def handle_migrate_mode(
    output_dir: Path,
    *,
    njobs: int = 1,
    dry_run: bool = False,
    delete_sources: bool = False,
) -> int:
    """Run ``--mode migrate`` and return the process exit code.

    Args:
        output_dir: Run output root, converted in place.
        njobs: Worker processes for the per-image conversion pass.
        dry_run: Report both passes and write nothing.
        delete_sources: Delete each provably-faithful source after conversion.

    Returns:
        ``0`` when both passes were clean, ``1`` otherwise.
    """
    report = run_migrate(
        output_dir,
        njobs=njobs,
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
    "run_migrate",
]
