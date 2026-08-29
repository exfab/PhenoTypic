"""Private metadata-schema **inspection** seam for recompile.

``recompile`` stops REWRITING legacy metadata headers; it keeps READING them.
The rewrite moved to ``--mode migrate``, which supersedes flat-metadata
decision #1 ("every recompile migrates automatically... not restricted to a
special command"). Decision #3 -- permanent stored-data compatibility -- is
untouched: the read path canonicalizes legacy headers in memory
(``_cli/_metadata_join``), so no existing output directory breaks. Recompile
simply no longer mutates one as a side effect of an unrelated operation.

What survives here is the read-only preflight, so a recompile can still TELL
the user their bundle is legacy and name the command that converts it.
"""

from __future__ import annotations

from pathlib import Path

from phenotypic.sdk_ import (
    BundleLayout,
    MetadataMigrationReport,
    deliverables_dir,
    preflight_metadata_schema,
)


def report_metadata_schema_for_recompile(
    output_dir: Path,
) -> MetadataMigrationReport:
    """Inspect a recompile's bundle-owned metadata **without changing it**.

    ``preflight_metadata_schema`` writes nothing, so this is safe to run on
    every recompile. The layout is constructed directly because a recoverable
    run may have per-image authority even when an earlier aggregate is absent;
    passing a ``Path`` would route through ``BundleLayout.detect``, which
    raises unless ``deliverables/master_measurements.parquet`` exists.

    External ``--metadata`` inputs are deliberately outside this function and
    are never migration targets.

    Args:
        output_dir: Existing PhenoTypic run-output root.

    Returns:
        The read-only migration plan and compatibility status.
    """
    return preflight_metadata_schema(_metadata_bundle_layout(output_dir))


def legacy_header_target_count(report: MetadataMigrationReport) -> int:
    """Return how many bundle targets still carry legacy metadata headers."""
    return sum(1 for target in report.targets if target.status == "migratable")


def _metadata_bundle_layout(output_dir: Path) -> BundleLayout:
    """Resolve the bundle owned by a recompile without external inputs."""
    resolved_output = Path(output_dir).resolve()
    return BundleLayout(
        deliverables_base=deliverables_dir(resolved_output),
        output_root=resolved_output,
    )


__all__ = [
    "legacy_header_target_count",
    "report_metadata_schema_for_recompile",
]
