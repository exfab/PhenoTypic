"""Private metadata-schema migration seam for recompile.

Keeping this adapter separate from aggregation makes the automatic migration
phase removable without changing measurement publication code.
"""

from __future__ import annotations

from pathlib import Path

from phenotypic.sdk_ import (
    BundleLayout,
    MetadataMigrationResult,
    deliverables_dir,
    migrate_metadata_bundle,
    preflight_metadata_schema,
)


class RecompileMetadataMigrationError(RuntimeError):
    """Raised when metadata migration cannot safely release recompile."""

    def __init__(self, result: MetadataMigrationResult) -> None:
        """Build an actionable error from a blocked or failed result.

        Args:
            result: Durable migration result that prevented recompile.
        """
        self.result = result
        blocked = len(result.blocked_targets)
        migrated = len(result.migrated_targets)
        skipped = len(result.skipped_targets)
        receipt = str(result.receipt_path) if result.receipt_path else "none"
        details = (
            "; ".join(result.conflicts) or "no conflict details available"
        )
        super().__init__(
            f"metadata migration {result.status}: migrated={migrated}, "
            f"skipped={skipped}, blocked={blocked}, receipt={receipt}. "
            f"{details}"
        )


def migrate_metadata_schema_for_recompile(
    output_dir: Path,
) -> MetadataMigrationResult:
    """Preflight and migrate bundle-owned metadata before local recompile.

    The layout is constructed directly because a recoverable run may have
    per-image HDF authority even when an earlier aggregate is absent. External
    ``--metadata`` inputs are deliberately outside this function and are never
    migration targets.

    Args:
        output_dir: Existing PhenoTypic run-output root.

    Returns:
        A compatible no-op result or an applied migration result.

    Raises:
        RecompileMetadataMigrationError: Migration was blocked or failed, so
            aggregation must not start.
    """
    resolved_output = output_dir.resolve()
    layout = _metadata_bundle_layout(resolved_output)
    report = preflight_metadata_schema(layout)
    result = migrate_metadata_bundle(
        layout,
        expected_plan_fingerprint=report.plan_fingerprint,
    )
    if result.status not in {"compatible", "applied"}:
        raise RecompileMetadataMigrationError(result)
    from ._cli_completion import (
        refresh_success_markers_after_metadata_migration,
    )

    refresh_success_markers_after_metadata_migration(
        resolved_output,
        receipt_paths=(
            (result.receipt_path,) if result.receipt_path is not None else ()
        ),
    )
    return result


def _metadata_bundle_layout(output_dir: Path) -> BundleLayout:
    """Resolve the bundle owned by a recompile without external inputs."""
    resolved_output = Path(output_dir).resolve()
    return BundleLayout(
        deliverables_base=deliverables_dir(resolved_output),
        output_root=resolved_output,
    )


__all__ = [
    "RecompileMetadataMigrationError",
    "migrate_metadata_schema_for_recompile",
]
