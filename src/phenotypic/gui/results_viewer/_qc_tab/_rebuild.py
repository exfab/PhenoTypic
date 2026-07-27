"""Dash-free preflight and atomic publication for explicit QC rebuilds."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

import duckdb
import pandas as pd

from phenotypic._core._image_pipeline import ImagePipeline
from phenotypic.gui.results_viewer._compatibility import (
    preflight_output_compatibility,
)
from phenotypic.gui.shell._runs_registry import run_status_is_nonterminal
from phenotypic.sdk_ import (
    BundleLayout,
    QC_DUCKDB,
    atomic_write_bytes,
    atomic_write_json,
    bytes_fingerprint,
    file_fingerprint,
    generation_staging_path,
    gui_launch_owner_path,
    paths_fingerprint,
)
from phenotypic.sdk_._qc_recipe._runner import (
    qc_publication_lock,
    qc_publication_lock_path,
    run_qc,
)
from phenotypic.sdk_._file_locking import exclusive_path_lock

_REBUILD_RECEIPT_VERSION = 1
_CATALOG_REQUIRED_COLUMNS = {
    "instance_id",
    "class",
    "name",
    "table_name",
    "summary_table",
    "ordinal",
    "groupby_cols",
    "metric_col",
    "status_col",
    "flag_col",
}
_FILE_BACKED_QC_PARAMS: dict[str, tuple[str, ...]] = {
    "ExpectedVsDetectedCount": ("metadata",),
    "GridOccupancy": ("metadata",),
}


@dataclass(frozen=True)
class QcRebuildPreflight:
    """Pure readiness result for an explicit QC database rebuild."""

    ready: bool
    source_fingerprint: str
    target: Path
    blockers: tuple[str, ...]


@dataclass(frozen=True)
class QcRebuildResult:
    """Publication result and durable recovery evidence."""

    applied: bool
    target: Path
    source_fingerprint: str
    database_fingerprint: str
    receipt_path: Path
    backup_path: Path | None = None


class QcRebuildError(RuntimeError):
    """Raised when a QC rebuild cannot be safely published."""


def _layout_for(source: Path | BundleLayout) -> BundleLayout:
    return source if isinstance(source, BundleLayout) else BundleLayout.detect(source)


def _external_qc_input_paths(layout: BundleLayout) -> tuple[Path, ...]:
    """Resolve every supported file-backed input consumed by QC entries."""
    pipeline_path = layout.resolved_pipeline_config_path
    try:
        payload = json.loads(pipeline_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return ()
    if not isinstance(payload, dict) or not isinstance(payload.get("qc"), list):
        return ()

    paths: list[Path] = []
    for raw_entry in payload["qc"]:
        if not isinstance(raw_entry, dict):
            continue
        if raw_entry.get("enabled", True) is False:
            continue
        class_name = raw_entry.get("class")
        params = raw_entry.get("params")
        if not isinstance(class_name, str) or not isinstance(params, dict):
            continue
        for param_name in _FILE_BACKED_QC_PARAMS.get(class_name, ()):
            raw_path = params.get(param_name)
            if isinstance(raw_path, str) and raw_path.strip():
                paths.append(Path(raw_path).expanduser().resolve(strict=False))
    return tuple(paths)


def _has_enabled_qc_entry(layout: BundleLayout) -> bool:
    """Return whether the serialized recipe enables at least one QC entry."""
    try:
        payload = json.loads(
            layout.resolved_pipeline_config_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    raw_qc = payload.get("qc") if isinstance(payload, dict) else None
    return isinstance(raw_qc, list) and any(
        isinstance(entry, dict) and entry.get("enabled", True) is not False
        for entry in raw_qc
    )


def _source_paths(layout: BundleLayout) -> tuple[Path, ...]:
    paths = [
        layout.resolved_pipeline_config_path,
        layout.mirror_parquet,
    ]
    paths.extend(_external_qc_input_paths(layout))
    if layout.output_root is not None:
        paths.append(gui_launch_owner_path(layout.output_root))
    return tuple(paths)


def _owner_blocker(layout: BundleLayout) -> str | None:
    if layout.output_root is None:
        return None
    owner = gui_launch_owner_path(layout.output_root)
    if not owner.exists():
        return None
    try:
        payload = json.loads(owner.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return f"Output owner record is unreadable: {exc}"
    if not isinstance(payload, dict):
        return "Output owner record is not a JSON object."
    status = payload.get("status")
    known_statuses = {
        "queued",
        "submitting",
        "running",
        "reconciling",
        "cancelling",
        "unknown",
        "complete",
        "failed",
        "cancelled",
    }
    if not isinstance(status, str) or status not in known_statuses:
        return "Output owner status is missing or unknown."
    if run_status_is_nonterminal(status):
        return f"Output has a nonterminal owner ({status!s})."
    return None


@contextmanager
def _output_owner_guard(layout: BundleLayout) -> Iterator[None]:
    """Prevent a GUI launch owner from being claimed during rebuild."""
    if layout.output_root is None:
        yield
        return
    owner_lock = gui_launch_owner_path(layout.output_root).with_suffix(".lock")
    with exclusive_path_lock(owner_lock):
        yield


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _writability_blocker(target: Path) -> str | None:
    if target.is_symlink():
        return "QC target must not be a symbolic link."
    anchor = target if target.exists() else _nearest_existing_parent(target.parent)
    if not anchor.exists() or not os.access(anchor, os.W_OK):
        return f"QC target is not writable: {target}"
    return None


def preflight_qc_rebuild(source: Path | BundleLayout) -> QcRebuildPreflight:
    """Check rebuild inputs without writing to the selected output."""
    layout = _layout_for(source)
    target = layout.qc_dir / QC_DUCKDB
    blockers: list[str] = []

    compatibility = preflight_output_compatibility(layout)
    if compatibility.status != "compatible":
        detail = "; ".join(issue.message for issue in compatibility.issues)
        blockers.append(
            "Recipe must be explicitly migrated before rebuild."
            if compatibility.status == "migratable"
            else f"Recipe compatibility is blocked: {detail}"
        )
    pipeline_path = layout.resolved_pipeline_config_path
    if not pipeline_path.is_file():
        blockers.append(f"Pipeline recipe is missing: {pipeline_path}")
    elif compatibility.status == "compatible" and not _has_enabled_qc_entry(
        layout
    ):
        blockers.append("At least one enabled QC recipe entry is required.")

    mirror = layout.mirror_parquet
    if not mirror.is_file() or mirror.stat().st_size == 0:
        blockers.append(
            "The complete measurements.parquet mirror is required for rebuild."
        )
    for dependency in _external_qc_input_paths(layout):
        if not dependency.is_file():
            blockers.append(f"QC recipe input is missing: {dependency}")
        elif not os.access(dependency, os.R_OK):
            blockers.append(f"QC recipe input is unreadable: {dependency}")

    owner_blocker = _owner_blocker(layout)
    if owner_blocker is not None:
        blockers.append(owner_blocker)
    writable_blocker = _writability_blocker(target)
    if writable_blocker is not None:
        blockers.append(writable_blocker)

    try:
        fingerprint = paths_fingerprint(
            _source_paths(layout),
            root=layout.output_root or layout.deliverables_base,
        )
    except OSError as exc:
        blockers.append(f"Could not fingerprint rebuild inputs: {exc}")
        fingerprint = bytes_fingerprint(b"qc-rebuild-unavailable")

    return QcRebuildPreflight(
        ready=not blockers,
        source_fingerprint=fingerprint,
        target=target,
        blockers=tuple(blockers),
    )


def _receipt_path(target: Path, source_fingerprint: str) -> Path:
    digest = source_fingerprint.removeprefix("sha256:")[:16]
    return target.parent / ".rebuild_receipts" / f"{target.name}.{digest}.json"


def _backup_path(
    target: Path,
    *,
    timestamp: datetime,
    source_fingerprint: str,
) -> Path:
    digest = source_fingerprint.removeprefix("sha256:")[:12]
    token = timestamp.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return target.parent / ".rebuild_backups" / (
        f"{target.name}.{token}.{digest}.bak"
    )


def _idempotent_result(
    target: Path,
    receipt_path: Path,
    source_fingerprint: str,
) -> QcRebuildResult | None:
    if not target.is_file() or not receipt_path.is_file():
        return None
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        database_fingerprint = file_fingerprint(target)
    except (OSError, json.JSONDecodeError):
        return None
    if (
        isinstance(receipt, dict)
        and receipt.get("state") == "applied"
        and receipt.get("source_fingerprint") == source_fingerprint
        and receipt.get("database_fingerprint") == database_fingerprint
    ):
        return QcRebuildResult(
            applied=False,
            target=target,
            source_fingerprint=source_fingerprint,
            database_fingerprint=database_fingerprint,
            receipt_path=receipt_path,
        )
    return None


def _validate_database(path: Path, expected_instance_ids: list[str]) -> None:
    if not path.is_file():
        raise QcRebuildError("run_qc did not produce a QC database.")
    con = duckdb.connect(str(path), read_only=True)
    try:
        catalog_columns = {
            row[1] for row in con.execute("PRAGMA table_info('qc_modules')").fetchall()
        }
        if not _CATALOG_REQUIRED_COLUMNS <= catalog_columns:
            raise QcRebuildError("QC catalog schema is incomplete.")
        rows = con.execute(
            "SELECT instance_id, table_name, summary_table "
            "FROM qc_modules ORDER BY ordinal"
        ).fetchall()
        actual_ids = [str(row[0]) for row in rows]
        if actual_ids != expected_instance_ids:
            raise QcRebuildError(
                "QC catalog modules do not match the enabled recipe entries."
            )
        tables = {
            str(row[0])
            for row in con.execute(
                "SELECT table_name FROM information_schema.tables"
            ).fetchall()
        }
        for _instance_id, table_name, summary_table in rows:
            if str(table_name) not in tables or str(summary_table) not in tables:
                raise QcRebuildError("QC module data or summary table is missing.")
    finally:
        con.close()


def rebuild_qc_database(
    source: Path | BundleLayout,
    *,
    expected_source_fingerprint: str,
    now: datetime | None = None,
    runner: Callable[..., object] = run_qc,
    publication_guard: Callable[[], bool] | None = None,
) -> QcRebuildResult:
    """Explicitly rebuild, validate, and atomically publish ``qc.duckdb``."""
    layout = _layout_for(source)
    target = layout.qc_dir / QC_DUCKDB
    _require_rebuild_publication(publication_guard)
    with _output_owner_guard(layout), qc_publication_lock(target):
        _require_rebuild_publication(publication_guard)
        current = preflight_qc_rebuild(layout)
        if current.source_fingerprint != expected_source_fingerprint:
            raise QcRebuildError(
                "Rebuild inputs changed after preflight; refresh and retry."
            )
        if not current.ready:
            raise QcRebuildError("; ".join(current.blockers))

        receipt_path = _receipt_path(target, current.source_fingerprint)
        existing = _idempotent_result(
            target,
            receipt_path,
            current.source_fingerprint,
        )
        if existing is not None:
            return existing
        original_receipt_bytes = (
            receipt_path.read_bytes() if receipt_path.is_file() else None
        )
        receipt_parent_existed = receipt_path.parent.exists()

        pipeline = ImagePipeline.from_json(
            layout.resolved_pipeline_config_path,
            skip_unknown_analyzers=False,
        )
        enabled_ids = [entry.instance_id for entry in pipeline.get_qc() if entry.enabled]
        if not enabled_ids:
            raise QcRebuildError(
                "At least one enabled QC recipe entry is required for rebuild."
            )
        try:
            measurements = pd.read_parquet(layout.mirror_parquet)
        except Exception as exc:
            raise QcRebuildError(
                f"The measurements mirror is incomplete or unreadable: {exc}"
            ) from exc

        generation = uuid.uuid4().hex
        staging_dir = generation_staging_path(target, generation)
        staged_db = staging_dir / QC_DUCKDB
        timestamp = now or datetime.now(timezone.utc)
        original_bytes = target.read_bytes() if target.is_file() else None
        backup_path: Path | None = None
        published = False
        receipt_published = False
        try:
            _require_rebuild_publication(publication_guard)
            if original_bytes is not None:
                backup_path = _backup_path(
                    target,
                    timestamp=timestamp,
                    source_fingerprint=current.source_fingerprint,
                )
                atomic_write_bytes(
                    backup_path,
                    original_bytes,
                    pre_replace=lambda: _require_rebuild_publication(
                        publication_guard
                    ),
                )
            runner(
                measurements,
                pipeline,
                layout.output_root or layout.deliverables_base,
                qc_output_dir=staging_dir,
                publication_guard=publication_guard,
            )
            _validate_database(staged_db, enabled_ids)
            database_bytes = staged_db.read_bytes()
            database_fingerprint = bytes_fingerprint(database_bytes)
            final_source = preflight_qc_rebuild(layout)
            if (
                not final_source.ready
                or final_source.source_fingerprint
                != current.source_fingerprint
            ):
                raise QcRebuildError(
                    "Rebuild inputs changed while QC analysis was running; "
                    "the staged database was discarded."
                )

            _require_rebuild_publication(publication_guard)
            atomic_write_bytes(
                target,
                database_bytes,
                pre_replace=lambda: _require_rebuild_publication(
                    publication_guard
                ),
            )
            published = True
            _validate_database(target, enabled_ids)
            post_publish_source = preflight_qc_rebuild(layout)
            if (
                not post_publish_source.ready
                or post_publish_source.source_fingerprint
                != current.source_fingerprint
            ):
                raise QcRebuildError(
                    "Rebuild inputs changed during database publication."
                )

            receipt = {
                "schema_version": _REBUILD_RECEIPT_VERSION,
                "rebuild": "qc_duckdb",
                "state": "applied",
                "created_at": timestamp.astimezone(timezone.utc).isoformat(),
                "target": str(target.resolve(strict=False)),
                "source_fingerprint": current.source_fingerprint,
                "database_fingerprint": database_fingerprint,
                "backup_path": (
                    str(backup_path.resolve(strict=False))
                    if backup_path is not None
                    else None
                ),
            }
            _require_rebuild_publication(publication_guard)
            atomic_write_json(
                receipt_path,
                receipt,
                sort_keys=False,
                pre_replace=lambda: _require_rebuild_publication(
                    publication_guard
                ),
            )
            receipt_published = True
            receipt_boundary_source = preflight_qc_rebuild(layout)
            if (
                not receipt_boundary_source.ready
                or receipt_boundary_source.source_fingerprint
                != current.source_fingerprint
            ):
                raise QcRebuildError(
                    "Rebuild inputs changed at the generation receipt boundary."
                )
            return QcRebuildResult(
                applied=True,
                target=target,
                source_fingerprint=current.source_fingerprint,
                database_fingerprint=database_fingerprint,
                receipt_path=receipt_path,
                backup_path=backup_path,
            )
        except Exception as exc:
            if receipt_published:
                try:
                    if original_receipt_bytes is not None:
                        atomic_write_bytes(
                            receipt_path,
                            original_receipt_bytes,
                        )
                    else:
                        receipt_path.unlink(missing_ok=True)
                        receipt_path.parent.rmdir()
                except OSError:
                    pass
            elif not receipt_parent_existed:
                try:
                    receipt_path.parent.rmdir()
                except OSError:
                    pass
            if published:
                if original_bytes is not None:
                    atomic_write_bytes(target, original_bytes)
                elif target.exists():
                    staging_dir.mkdir(parents=True, exist_ok=True)
                    os.replace(target, staging_dir / QC_DUCKDB)
            if backup_path is not None:
                try:
                    backup_path.unlink(missing_ok=True)
                    backup_path.parent.rmdir()
                except OSError:
                    pass
            if isinstance(exc, QcRebuildError):
                raise
            raise QcRebuildError(f"QC rebuild failed and was rolled back: {exc}") from exc
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)


def _require_rebuild_publication(
    publication_guard: Callable[[], bool] | None,
) -> None:
    """Fail closed immediately before an explicit rebuild publication."""
    if publication_guard is not None and not publication_guard():
        raise QcRebuildError(
            "QC rebuild publication blocked because its output snapshot changed."
        )


__all__ = [
    "QcRebuildError",
    "QcRebuildPreflight",
    "QcRebuildResult",
    "preflight_qc_rebuild",
    "qc_publication_lock",
    "qc_publication_lock_path",
    "rebuild_qc_database",
]
