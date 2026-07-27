"""Pure output compatibility checks and explicit QC recipe migration.

The preflight functions in this module never write to the selected output and
never instantiate through :class:`QcRecipe`, whose warning list is mutable.
Only :func:`migrate_output_recipe` performs writes, after repeating preflight,
verifying the caller's source fingerprint, backing up the exact original
bytes, and validating the complete migrated QC array.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from phenotypic.analysis.qc import ExpectedVsDetectedCount
from phenotypic.schema import METADATA
from phenotypic.sdk_ import (
    BundleLayout,
    atomic_write_bytes,
    atomic_write_json,
    bytes_fingerprint,
    file_fingerprint,
    migration_backup_path,
    migration_receipt_path,
    pipeline_publication_lock,
)
from phenotypic.sdk_._qc_recipe import (
    QcRecipeEntry,
    QcRecipeLoadWarning,
)

CompatibilityStatus = Literal["compatible", "migratable", "blocked"]

_QC_KEY = "qc"
_GRID_OCCUPANCY = "GridOccupancy"
_MIGRATION_RECEIPT_VERSION = 1

# These aliases are deliberate historical contracts, not a general prefix
# guesser. A spelling is rewritten only when the target exists in the
# referenced metadata file.
LEGACY_METADATA_COLUMN_ALIASES: dict[str, str] = {
    "Metadata_ImageName": str(METADATA.IMAGE_NAME),
}


@dataclass(frozen=True)
class CompatibilityIssue:
    """One deterministic compatibility finding."""

    code: str
    status: CompatibilityStatus
    location: str
    message: str
    proposed_change: str | None = None


@dataclass(frozen=True)
class OutputCompatibilityReport:
    """Complete, immutable result of a source-preserving preflight."""

    status: CompatibilityStatus
    source_fingerprint: str
    issues: tuple[CompatibilityIssue, ...]
    migrated_pipeline_payload: dict[str, object] | None = None


@dataclass(frozen=True)
class RecipeMigrationResult:
    """Outcome and rollback evidence for an explicit recipe migration."""

    applied: bool
    report: OutputCompatibilityReport
    pipeline_path: Path
    backup_path: Path | None = None
    receipt_path: Path | None = None
    old_fingerprint: str | None = None
    new_fingerprint: str | None = None


class CompatibilityMigrationError(RuntimeError):
    """Raised when an explicit migration cannot safely publish."""


def preflight_output_compatibility(
    source: Path | BundleLayout,
) -> OutputCompatibilityReport:
    """Classify a pipeline QC recipe without writing or mutating warnings.

    Args:
        source: Pipeline configuration path, output directory, standalone
            deliverables directory, or an already-resolved bundle layout.

    Returns:
        A compatible, migratable, or blocked report.
    """
    pipeline_path = _pipeline_path_for(source)
    try:
        source_bytes = pipeline_path.read_bytes()
    except FileNotFoundError:
        return OutputCompatibilityReport(
            status="compatible",
            source_fingerprint=bytes_fingerprint(b""),
            issues=(),
        )
    except OSError as exc:
        return OutputCompatibilityReport(
            status="blocked",
            source_fingerprint=bytes_fingerprint(b""),
            issues=(
                CompatibilityIssue(
                    code="pipeline.read_error",
                    status="blocked",
                    location=str(pipeline_path),
                    message=f"Pipeline configuration could not be read: {exc}",
                ),
            ),
        )

    fingerprint = bytes_fingerprint(source_bytes)
    try:
        payload = json.loads(source_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return OutputCompatibilityReport(
            status="blocked",
            source_fingerprint=fingerprint,
            issues=(
                CompatibilityIssue(
                    code="pipeline.invalid_json",
                    status="blocked",
                    location=str(pipeline_path),
                    message=f"Pipeline configuration is not valid UTF-8 JSON: {exc}",
                ),
            ),
        )
    return _preflight_payload(
        payload,
        source_fingerprint=fingerprint,
        pipeline_path=pipeline_path,
    )


def migrate_output_recipe(
    source: Path | BundleLayout,
    *,
    expected_source_fingerprint: str,
    now: datetime | None = None,
) -> RecipeMigrationResult:
    """Explicitly migrate a preflight-approved QC recipe.

    Args:
        source: Same source forms accepted by
            :func:`preflight_output_compatibility`.
        expected_source_fingerprint: Fingerprint displayed by the caller's
            preflight. A mismatch refuses the write.
        now: Optional UTC-aware clock override for deterministic tests.

    Returns:
        Migration result. Already-compatible input returns
        ``applied=False`` and creates no backup or receipt.

    Raises:
        CompatibilityMigrationError: If preflight blocks, the fingerprint
            changed, or the proposed payload fails complete validation.
        OSError: If backup, publication, or receipt persistence fails.
    """
    pipeline_path = _pipeline_path_for(source)
    with pipeline_publication_lock(pipeline_path):
        # A canonical typed writer may have published while this migration
        # waited behind a legacy-only output. Re-resolve directory/layout
        # sources under the shared output-level lock before checking the
        # caller's generation.
        pipeline_path = _pipeline_path_for(source)
        return _migrate_output_recipe_locked(
            pipeline_path,
            expected_source_fingerprint=expected_source_fingerprint,
            now=now,
        )


def _migrate_output_recipe_locked(
    pipeline_path: Path,
    *,
    expected_source_fingerprint: str,
    now: datetime | None,
) -> RecipeMigrationResult:
    """Run the migration CAS while holding its canonical interprocess lock."""
    report = preflight_output_compatibility(pipeline_path)
    if report.source_fingerprint != expected_source_fingerprint:
        raise CompatibilityMigrationError(
            "Pipeline changed after compatibility preflight; refresh and retry."
        )
    if report.status == "blocked":
        raise CompatibilityMigrationError(
            "Pipeline compatibility is blocked: "
            + "; ".join(issue.message for issue in report.issues)
        )
    if report.status == "compatible":
        return RecipeMigrationResult(
            applied=False,
            report=report,
            pipeline_path=pipeline_path,
            old_fingerprint=report.source_fingerprint,
            new_fingerprint=report.source_fingerprint,
        )

    migrated_payload = report.migrated_pipeline_payload
    if migrated_payload is None:
        raise CompatibilityMigrationError(
            "Migratable report did not provide a validated pipeline payload."
        )

    original_bytes = pipeline_path.read_bytes()
    original_fingerprint = bytes_fingerprint(original_bytes)
    if original_fingerprint != expected_source_fingerprint:
        raise CompatibilityMigrationError(
            "Pipeline changed while migration was starting; refresh and retry."
        )

    migrated_bytes = (
        json.dumps(
            migrated_payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=False,
        )
        + "\n"
    ).encode("utf-8")
    new_fingerprint = bytes_fingerprint(migrated_bytes)
    validation = _preflight_payload(
        json.loads(migrated_bytes),
        source_fingerprint=new_fingerprint,
        pipeline_path=pipeline_path,
    )
    if validation.status != "compatible":
        raise CompatibilityMigrationError(
            "Migrated pipeline did not pass complete compatibility validation."
        )

    timestamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    timestamp_token = timestamp.strftime("%Y%m%dT%H%M%S.%fZ")
    backup_path = migration_backup_path(
        pipeline_path,
        timestamp=timestamp_token,
        source_fingerprint=original_fingerprint,
    )
    receipt_path = migration_receipt_path(
        pipeline_path,
        resulting_fingerprint=new_fingerprint,
    )
    old_payload = json.loads(original_bytes)
    receipt = {
        "schema_version": _MIGRATION_RECEIPT_VERSION,
        "migration": "qc_recipe",
        "state": "prepared",
        "created_at": timestamp.isoformat(),
        "pipeline_path": str(pipeline_path.resolve(strict=False)),
        "backup_path": str(backup_path.resolve(strict=False)),
        "old_version": (
            old_payload.get("version") if isinstance(old_payload, dict) else None
        ),
        "new_version": migrated_payload.get("version"),
        "old_fingerprint": original_fingerprint,
        "new_fingerprint": new_fingerprint,
    }

    atomic_write_bytes(backup_path, original_bytes)
    # Persist complete rollback evidence before publishing the source change.
    # If publication fails, the receipt remains explicitly "prepared".
    atomic_write_json(receipt_path, receipt, sort_keys=False)
    if file_fingerprint(pipeline_path) != original_fingerprint:
        raise CompatibilityMigrationError(
            "Pipeline changed before atomic publication; backup was retained."
        )
    atomic_write_bytes(pipeline_path, migrated_bytes)
    if file_fingerprint(pipeline_path) != new_fingerprint:
        raise CompatibilityMigrationError(
            "Pipeline changed during atomic publication; the prepared receipt "
            "and exact backup were retained."
        )

    applied_receipt = {**receipt, "state": "applied"}
    try:
        atomic_write_json(receipt_path, applied_receipt, sort_keys=False)
    except Exception as receipt_exc:
        _rollback_failed_receipt(
            pipeline_path,
            original_bytes=original_bytes,
            original_fingerprint=original_fingerprint,
            migrated_fingerprint=new_fingerprint,
            receipt_exc=receipt_exc,
        )

    return RecipeMigrationResult(
        applied=True,
        report=validation,
        pipeline_path=pipeline_path,
        backup_path=backup_path,
        receipt_path=receipt_path,
        old_fingerprint=original_fingerprint,
        new_fingerprint=new_fingerprint,
    )


def _rollback_failed_receipt(
    pipeline_path: Path,
    *,
    original_bytes: bytes,
    original_fingerprint: str,
    migrated_fingerprint: str,
    receipt_exc: Exception,
) -> None:
    """CAS-rollback a migration whose final applied receipt could not persist."""
    current_fingerprint = file_fingerprint(pipeline_path)
    if current_fingerprint == migrated_fingerprint:
        try:
            atomic_write_bytes(pipeline_path, original_bytes)
        except Exception as rollback_exc:
            raise CompatibilityMigrationError(
                "Migration receipt finalization and source rollback both failed. "
                "The prepared receipt and exact backup identify the transaction."
            ) from rollback_exc
        if file_fingerprint(pipeline_path) != original_fingerprint:
            raise CompatibilityMigrationError(
                "Migration receipt finalization failed and rollback verification "
                "did not recover the original source fingerprint."
            ) from receipt_exc
        raise CompatibilityMigrationError(
            "Migration receipt finalization failed; the pipeline was rolled back."
        ) from receipt_exc
    if current_fingerprint == original_fingerprint:
        raise CompatibilityMigrationError(
            "Migration receipt finalization failed; the pipeline remained original."
        ) from receipt_exc
    raise CompatibilityMigrationError(
        "Migration receipt finalization failed and the pipeline changed "
        "concurrently; the prepared receipt and exact backup were retained."
    ) from receipt_exc


def _pipeline_path_for(source: Path | BundleLayout) -> Path:
    """Resolve a pipeline configuration path without creating directories."""
    if isinstance(source, BundleLayout):
        return source.resolved_pipeline_config_path
    path = Path(source)
    if path.is_dir():
        return BundleLayout.detect(path).resolved_pipeline_config_path
    return path


def _preflight_payload(
    payload: object,
    *,
    source_fingerprint: str,
    pipeline_path: Path,
) -> OutputCompatibilityReport:
    """Classify one already-parsed pipeline payload."""
    if not isinstance(payload, dict):
        return OutputCompatibilityReport(
            status="blocked",
            source_fingerprint=source_fingerprint,
            issues=(
                CompatibilityIssue(
                    code="pipeline.invalid_shape",
                    status="blocked",
                    location="$",
                    message="Pipeline JSON must be an object.",
                ),
            ),
        )

    migrated: dict[str, object] = copy.deepcopy(payload)
    raw_qc = payload.get(_QC_KEY, [])
    if raw_qc is None:
        raw_qc = []
    if not isinstance(raw_qc, list):
        return OutputCompatibilityReport(
            status="blocked",
            source_fingerprint=source_fingerprint,
            issues=(
                CompatibilityIssue(
                    code="qc.invalid_shape",
                    status="blocked",
                    location="$.qc",
                    message="Pipeline 'qc' must be an array.",
                ),
            ),
        )

    migrated_entries: list[object] = []
    issues: list[CompatibilityIssue] = []
    seen: set[tuple[str, str, str]] = set()
    changed = False
    blocked = False

    for index, raw_entry in enumerate(raw_qc):
        location = f"$.qc[{index}]"
        if not isinstance(raw_entry, dict):
            migrated_entries.append(copy.deepcopy(raw_entry))
            blocked = True
            _append_issue_once(
                issues,
                seen,
                CompatibilityIssue(
                    code="qc.entry.invalid_shape",
                    status="blocked",
                    location=location,
                    message="QC entry must be an object; raw entry was preserved.",
                ),
            )
            continue

        candidate: dict[str, Any] = copy.deepcopy(raw_entry)
        try:
            parsed = QcRecipeEntry.from_dict(candidate)
        except (TypeError, ValueError):
            migrated_entries.append(candidate)
            blocked = True
            _append_issue_once(
                issues,
                seen,
                CompatibilityIssue(
                    code="qc.entry.invalid_shape",
                    status="blocked",
                    location=location,
                    message="QC entry fields have an invalid shape; raw entry was preserved.",
                ),
            )
            continue
        if isinstance(parsed, QcRecipeLoadWarning):
            migrated_entries.append(candidate)
            blocked = True
            _append_issue_once(
                issues,
                seen,
                CompatibilityIssue(
                    code="qc.entry.unknown_class",
                    status="blocked",
                    location=location,
                    message=(
                        f"QC class {parsed.class_name!r} is unavailable; "
                        "raw entry was preserved."
                    ),
                ),
            )
            continue

        try:
            parsed.instantiate()
        except Exception as original_exc:  # noqa: BLE001 - compatibility diagnosis
            if candidate.get("class") != _GRID_OCCUPANCY:
                migrated_entries.append(candidate)
                blocked = True
                _append_issue_once(
                    issues,
                    seen,
                    CompatibilityIssue(
                        code="qc.entry.invalid_params",
                        status="blocked",
                        location=location,
                        message=(
                            f"{parsed.cls.__name__} cannot be instantiated: "
                            f"{original_exc}"
                        ),
                    ),
                )
                continue

            candidate, migration_issues, entry_changed = (
                _migrate_grid_occupancy_entry(
                    candidate,
                    location=location,
                )
            )
            for issue in migration_issues:
                _append_issue_once(issues, seen, issue)
            if any(issue.status == "blocked" for issue in migration_issues):
                blocked = True
                migrated_entries.append(candidate)
                continue
            if not entry_changed:
                blocked = True
                migrated_entries.append(candidate)
                _append_issue_once(
                    issues,
                    seen,
                    CompatibilityIssue(
                        code="qc.entry.invalid_params",
                        status="blocked",
                        location=location,
                        message=(
                            f"GridOccupancy cannot be instantiated: {original_exc}"
                        ),
                    ),
                )
                continue

            reparsed = QcRecipeEntry.from_dict(candidate)
            if isinstance(reparsed, QcRecipeLoadWarning):
                blocked = True
                migrated_entries.append(candidate)
                continue
            try:
                reparsed.instantiate()
            except Exception as exc:  # noqa: BLE001 - complete proposed validation
                blocked = True
                migrated_entries.append(candidate)
                _append_issue_once(
                    issues,
                    seen,
                    CompatibilityIssue(
                        code="qc.grid_migration.invalid",
                        status="blocked",
                        location=location,
                        message=f"Proposed GridOccupancy migration is invalid: {exc}",
                    ),
                )
                continue

            changed = True
            migrated_entries.append(candidate)
        else:
            migrated_entries.append(candidate)

    migrated[_QC_KEY] = migrated_entries
    if blocked:
        return OutputCompatibilityReport(
            status="blocked",
            source_fingerprint=source_fingerprint,
            issues=tuple(issues),
        )
    if changed:
        return OutputCompatibilityReport(
            status="migratable",
            source_fingerprint=source_fingerprint,
            issues=tuple(issues),
            migrated_pipeline_payload=migrated,
        )
    return OutputCompatibilityReport(
        status="compatible",
        source_fingerprint=source_fingerprint,
        issues=tuple(issues),
    )


def _migrate_grid_occupancy_entry(
    entry: dict[str, Any],
    *,
    location: str,
) -> tuple[dict[str, Any], list[CompatibilityIssue], bool]:
    """Apply the exact historical GridOccupancy field mapping in memory."""
    candidate = copy.deepcopy(entry)
    params_raw = candidate.get("params")
    if not isinstance(params_raw, dict):
        return (
            candidate,
            [
                CompatibilityIssue(
                    code="qc.grid.params.invalid_shape",
                    status="blocked",
                    location=f"{location}.params",
                    message="GridOccupancy params must be an object.",
                )
            ],
            False,
        )
    params: dict[str, Any] = copy.deepcopy(params_raw)
    issues: list[CompatibilityIssue] = []
    changed = False

    metadata = params.get("metadata")
    metadata_source = params.get("metadata_source")
    metadata_text = metadata.strip() if isinstance(metadata, str) else metadata
    source_text = (
        metadata_source.strip()
        if isinstance(metadata_source, str)
        else metadata_source
    )
    if metadata_text and source_text and metadata_text != source_text:
        return (
            candidate,
            [
                CompatibilityIssue(
                    code="qc.grid.metadata.ambiguous",
                    status="blocked",
                    location=f"{location}.params",
                    message=(
                        "GridOccupancy metadata and metadata_source are both "
                        "set to different values."
                    ),
                )
            ],
            False,
        )
    if not metadata_text and source_text:
        params["metadata"] = source_text
        metadata_text = source_text
        changed = True
        issues.append(
            CompatibilityIssue(
                code="qc.grid.metadata_source",
                status="migratable",
                location=f"{location}.params.metadata_source",
                message="GridOccupancy uses the retired metadata_source field.",
                proposed_change="Copy metadata_source to metadata and remove it.",
            )
        )
    if "metadata_source" in params:
        if not any(
            issue.code == "qc.grid.metadata_source" for issue in issues
        ):
            issues.append(
                CompatibilityIssue(
                    code="qc.grid.metadata_source",
                    status="migratable",
                    location=f"{location}.params.metadata_source",
                    message="GridOccupancy uses the retired metadata_source field.",
                    proposed_change="Remove metadata_source; metadata is authoritative.",
                )
            )
        params.pop("metadata_source")
        changed = True
    if params.get("cell_label", object()) is None:
        params.pop("cell_label")
        changed = True
        issues.append(
            CompatibilityIssue(
                code="qc.grid.cell_label_null",
                status="migratable",
                location=f"{location}.params.cell_label",
                message="GridOccupancy has a null retired cell_label value.",
                proposed_change="Omit cell_label so the current default applies.",
            )
        )

    if not isinstance(metadata_text, str) or not metadata_text:
        return (
            candidate,
            [
                *issues,
                CompatibilityIssue(
                    code="qc.grid.metadata.missing",
                    status="blocked",
                    location=f"{location}.params.metadata",
                    message="GridOccupancy migration requires a metadata file path.",
                ),
            ],
            changed,
        )

    try:
        metadata_frame = ExpectedVsDetectedCount._resolve_metadata(metadata_text)
    except Exception as exc:  # noqa: BLE001 - report exact compatibility blocker
        return (
            candidate,
            [
                *issues,
                CompatibilityIssue(
                    code="qc.grid.metadata.unavailable",
                    status="blocked",
                    location=f"{location}.params.metadata",
                    message=f"GridOccupancy metadata is unavailable: {exc}",
                ),
            ],
            changed,
        )

    groupby = params.get("groupby")
    if isinstance(groupby, list):
        mapped_groupby: list[object] = []
        for column in groupby:
            target = (
                LEGACY_METADATA_COLUMN_ALIASES.get(column)
                if isinstance(column, str)
                else None
            )
            if target is not None and target in metadata_frame.columns:
                mapped_groupby.append(target)
                if target != column:
                    changed = True
                    issues.append(
                        CompatibilityIssue(
                            code="qc.grid.groupby_alias",
                            status="migratable",
                            location=f"{location}.params.groupby",
                            message=f"GridOccupancy uses retired column {column!r}.",
                            proposed_change=f"Map {column!r} to {target!r}.",
                        )
                    )
            else:
                mapped_groupby.append(column)
        params["groupby"] = mapped_groupby

    candidate["params"] = params
    return candidate, issues, changed


def _append_issue_once(
    issues: list[CompatibilityIssue],
    seen: set[tuple[str, str, str]],
    issue: CompatibilityIssue,
) -> None:
    """Append an issue once by code and location."""
    key = (issue.code, issue.status, issue.location)
    if key not in seen:
        seen.add(key)
        issues.append(issue)


__all__ = [
    "CompatibilityIssue",
    "CompatibilityMigrationError",
    "CompatibilityStatus",
    "LEGACY_METADATA_COLUMN_ALIASES",
    "OutputCompatibilityReport",
    "RecipeMigrationResult",
    "migrate_output_recipe",
    "preflight_output_compatibility",
]
