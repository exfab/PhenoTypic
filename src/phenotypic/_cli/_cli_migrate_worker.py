"""Internal indexed worker for dispatcher-fed migration SLURM jobs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
from typing import Any, Mapping

import click

from phenotypic.sdk_ import CommitGuard, atomic_write_json, deliverables_dir
from phenotypic.sdk_._hdf_to_zarr import MigrationReport
from phenotypic.sdk_._metadata_migration import MetadataMigrationAuthority

from ._cli_migrate import (
    MetadataPassResult,
    _ensure_migration_processing_state,
    _report_from_image_results,
    _retained_reclaim_result,
    close_migration_generation,
    finalize_migration_attempt,
    invalidate_migration_terminal_authority,
    publish_migration_terminal_status,
    run_metadata_pass,
)
from ._cli_migrate_image import (
    MigrationImageResult,
    migrate_image_task,
    reclaim_image_sources,
)
from ._cli_migrate_manifest import (
    MigrationImageSeal,
    MigrationReclaimSeal,
    _read_manifest,
    migration_image_seal_path,
    migration_reclaim_seal_path,
    publish_migration_reclaim_status,
    publish_migration_task_status,
    read_migration_task,
    seal_migration_image_stage,
    seal_migration_reclaim_stage,
    validate_migration_generation,
    valid_migration_image_seal,
)
from ._cli_migrate_provenance import upgrade_store_provenance
from ._cli_slurm_lifecycle import (
    assert_generation_active,
    generation_publication_guard,
)


@dataclass(frozen=True)
class _WorkerConfig:
    """Validated immutable worker configuration."""

    generation: str
    output_dir: Path
    scientific_output: Path
    control_root: Path
    manifest_path: Path
    inventory_digest: str
    task_count: int
    overlay_alpha: float
    delete_sources: bool
    dry_run: bool

    @property
    def lifecycle_root(self) -> Path:
        """Return the lifecycle owner for scientific or dry control commits."""
        return self.control_root if self.dry_run else self.output_dir


def migration_worker_status_path(
    control_root: Path,
    generation: str,
    stage: str,
    index: int | None = None,
) -> Path:
    """Return one generation-bound typed orchestration status path."""
    generation = validate_migration_generation(generation)
    if stage not in {
        "metadata", "image", "seal", "reclaim", "reclaim-seal", "terminal"
    }:
        raise ValueError("unknown migration worker stage")
    if index is not None and (
        not isinstance(index, int) or isinstance(index, bool) or index < 0
    ):
        raise ValueError("migration worker index must be a non-negative integer")
    suffix = stage if index is None else f"{stage}_{index}"
    root = Path(control_root).resolve()
    path = (root / "worker_status" / generation / f"{suffix}.json").resolve()
    if not path.is_relative_to(root):
        raise ValueError("migration worker status escapes control root")
    return path


def _load_worker_config(path: Path) -> _WorkerConfig:
    """Load a config whose caller-bound location authorizes its control root."""
    config_path = Path(path).absolute()
    if config_path.is_symlink():
        raise ValueError("migration worker config cannot be a symlink")
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid migration worker config") from exc
    fields = {
        "schema_version",
        "generation",
        "output_dir",
        "scientific_output",
        "control_root",
        "manifest_path",
        "inventory_digest",
        "task_count",
        "overlay_alpha",
        "delete_sources",
        "dry_run",
    }
    if not isinstance(raw, Mapping) or set(raw) != fields:
        raise ValueError("invalid migration worker config schema")
    if raw.get("schema_version") != 1:
        raise ValueError("unsupported migration worker config schema")
    generation = validate_migration_generation(raw.get("generation"))
    inventory_digest = raw.get("inventory_digest")
    task_count = raw.get("task_count")
    if (
        not isinstance(inventory_digest, str)
        or len(inventory_digest) != 64
        or not isinstance(task_count, int)
        or isinstance(task_count, bool)
        or task_count < 0
    ):
        raise ValueError("invalid migration worker identity fields")
    path_fields = ("output_dir", "scientific_output", "control_root", "manifest_path")
    if any(not isinstance(raw[field], str) or not raw[field] for field in path_fields):
        raise ValueError("invalid migration worker path fields")
    control_root = Path(raw["control_root"]).resolve()
    if config_path.resolve().parent != control_root:
        raise ValueError("migration worker config has wrong control root")
    output_dir = Path(raw["output_dir"]).resolve()
    scientific_output = Path(raw["scientific_output"]).resolve()
    if scientific_output != deliverables_dir(output_dir):
        raise ValueError("migration worker config has wrong scientific output")
    manifest_path = Path(raw["manifest_path"]).resolve()
    _, manifest = _read_manifest(
        manifest_path,
        scientific_output,
        expected_control_root=control_root,
    )
    if (
        manifest.generation != generation
        or manifest.inventory_digest != inventory_digest
        or manifest.task_count != task_count
    ):
        raise ValueError("migration worker config does not match manifest")
    overlay_alpha = raw["overlay_alpha"]
    delete_sources = raw["delete_sources"]
    dry_run = raw["dry_run"]
    if (
        not isinstance(overlay_alpha, (int, float))
        or isinstance(overlay_alpha, bool)
        or not math.isfinite(overlay_alpha)
        or not 0.0 <= overlay_alpha <= 1.0
        or not isinstance(delete_sources, bool)
        or not isinstance(dry_run, bool)
    ):
        raise ValueError("invalid migration worker option fields")
    return _WorkerConfig(
        generation=generation,
        output_dir=output_dir,
        scientific_output=scientific_output,
        control_root=control_root,
        manifest_path=manifest_path,
        inventory_digest=inventory_digest,
        task_count=task_count,
        overlay_alpha=float(overlay_alpha),
        delete_sources=delete_sources,
        dry_run=dry_run,
    )


def _commit_guard(config: _WorkerConfig) -> CommitGuard:
    """Return the lifecycle-locking publication guard for this generation."""
    return lambda: generation_publication_guard(
        config.lifecycle_root, config.generation
    )


def _publish_worker_status(
    config: _WorkerConfig,
    stage: str,
    *,
    status: str,
    index: int | None = None,
    failure_category: str | None = None,
    reason: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Publish typed orchestration evidence under the generation fence."""
    payload: dict[str, Any] = {
        "schema_version": 1,
        "generation": config.generation,
        "manifest_digest": config.inventory_digest,
        "stage": stage,
        "index": index,
        "status": status,
        "failure_category": failure_category,
        "reason": reason,
    }
    if extra:
        payload.update(extra)
    path = migration_worker_status_path(
        config.control_root, config.generation, stage, index
    )
    atomic_write_json(path, payload, commit_guard=_commit_guard(config))
    return path


def _read_worker_status(
    config: _WorkerConfig, stage: str, index: int | None = None
) -> Mapping[str, Any] | None:
    """Read current generation-bound worker evidence, or return None."""
    path = migration_worker_status_path(
        config.control_root, config.generation, stage, index
    )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(raw, Mapping)
        or raw.get("schema_version") != 1
        or raw.get("generation") != config.generation
        or raw.get("manifest_digest") != config.inventory_digest
        or raw.get("stage") != stage
        or raw.get("index") != index
        or raw.get("status") not in {"complete", "failed", "blocked"}
        or not (
            raw.get("failure_category") is None
            or isinstance(raw.get("failure_category"), str)
        )
        or not (raw.get("reason") is None or isinstance(raw.get("reason"), str))
        or (
            raw.get("status") == "complete"
            and (
                raw.get("failure_category") is not None
                or raw.get("reason") is not None
            )
        )
        or (
            raw.get("status") in {"failed", "blocked"}
            and (
                not isinstance(raw.get("failure_category"), str)
                or not raw.get("failure_category")
                or not isinstance(raw.get("reason"), str)
                or not raw.get("reason")
            )
        )
    ):
        return None
    return raw


def _authority_payload(authority: MetadataMigrationAuthority) -> dict[str, Any]:
    """Serialize stable metadata authority without weakening its digests."""
    return {
        "status_path": str(authority.status_path),
        "terminal_receipt_path": str(authority.terminal_receipt_path),
        "terminal_receipt_digest": authority.terminal_receipt_digest,
        "plan_fingerprint": authority.plan_fingerprint,
        "source_fingerprint": authority.source_fingerprint,
        "resulting_fingerprint": authority.resulting_fingerprint,
        "compatible_noop": authority.compatible_noop,
    }


def _authority_from_payload(value: object) -> MetadataMigrationAuthority | None:
    """Deserialize exact terminal metadata authority from a worker status."""
    if not isinstance(value, Mapping):
        return None
    expected = {
        "status_path",
        "terminal_receipt_path",
        "terminal_receipt_digest",
        "plan_fingerprint",
        "source_fingerprint",
        "resulting_fingerprint",
        "compatible_noop",
    }
    if set(value) != expected:
        return None
    path_fields = ("status_path", "terminal_receipt_path")
    digest_fields = (
        "terminal_receipt_digest", "plan_fingerprint",
        "source_fingerprint", "resulting_fingerprint",
    )

    def _exact_sha256(field: str) -> bool:
        candidate = value[field]
        if not isinstance(candidate, str) or not candidate.startswith("sha256:"):
            return False
        digest = candidate.removeprefix("sha256:")
        if len(digest) != 64:
            return False
        try:
            return bytes.fromhex(digest).hex() == digest
        except ValueError:
            return False

    if (
        any(not isinstance(value[field], str) or not value[field] for field in path_fields)
        or any(not _exact_sha256(field) for field in digest_fields)
        or not isinstance(value["compatible_noop"], bool)
    ):
        return None
    return MetadataMigrationAuthority(
        status_path=Path(value["status_path"]),
        terminal_receipt_path=Path(value["terminal_receipt_path"]),
        terminal_receipt_digest=value["terminal_receipt_digest"],
        plan_fingerprint=value["plan_fingerprint"],
        source_fingerprint=value["source_fingerprint"],
        resulting_fingerprint=value["resulting_fingerprint"],
        compatible_noop=value["compatible_noop"],
    )


def _metadata_prerequisite(
    config: _WorkerConfig, status: Mapping[str, Any] | None
) -> tuple[MetadataMigrationAuthority | None, str | None]:
    """Validate typed metadata completion before any image mutation."""
    if status is None:
        return None, "metadata stage status is missing or corrupt"
    if status.get("status") != "complete":
        reason = status.get("reason")
        return None, (
            reason if isinstance(reason, str) and reason else "metadata stage failed"
        )
    headers = status.get("headers_migrated")
    if not isinstance(headers, int) or isinstance(headers, bool) or headers < 0:
        return None, "metadata stage status has invalid migrated-header count"
    authority = _authority_from_payload(status.get("authority"))
    if config.dry_run:
        if status.get("authority") is not None and authority is None:
            return None, "metadata stage status has invalid authority"
        return authority, None
    if authority is None:
        return None, "metadata stage authority is missing or corrupt"
    return authority, None


def _run_metadata_worker(config: _WorkerConfig) -> int:
    """Run the bundle-wide metadata singleton and publish terminal authority."""
    assert_generation_active(config.lifecycle_root, config.generation)
    try:
        if not config.dry_run:
            invalidate_migration_terminal_authority(
                config.output_dir, commit_guard=_commit_guard(config)
            )
        result = run_metadata_pass(
            config.output_dir,
            dry_run=config.dry_run,
            commit_guard=_commit_guard(config),
        )
        if result.failures or (not config.dry_run and result.authority is None):
            reason = (
                result.failures[0][1]
                if result.failures
                else "metadata stage lacks terminal authority"
            )
            _publish_worker_status(
                config,
                "metadata",
                status="failed",
                failure_category="metadata",
                reason=reason,
                extra={"headers_migrated": result.headers_migrated},
            )
            return 1
        if not config.dry_run:
            tasks = tuple(
                read_migration_task(
                    config.manifest_path,
                    index,
                    expected_scientific_output=config.scientific_output,
                    expected_control_root=config.control_root,
                )
                for index in range(config.task_count)
            )
            _ensure_migration_processing_state(
                config.output_dir,
                tasks=tasks,
                commit_guard=_commit_guard(config),
            )
        extra: dict[str, Any] = {
            "headers_migrated": result.headers_migrated,
            "authority": (
                None
                if result.authority is None
                else _authority_payload(result.authority)
            ),
        }
        _publish_worker_status(config, "metadata", status="complete", extra=extra)
        return 0
    except Exception as exc:  # noqa: BLE001 - preserve typed scheduler failure
        reason = f"{type(exc).__name__}: {exc}"
        try:
            _publish_worker_status(
                config,
                "metadata",
                status="failed",
                failure_category="metadata",
                reason=reason,
            )
        except Exception:  # noqa: BLE001 - stale generations cannot publish
            pass
        return 1


def _run_image_worker(config: _WorkerConfig, index: int) -> int:
    """Run one independent indexed image migration."""
    assert_generation_active(config.lifecycle_root, config.generation)
    metadata = _read_worker_status(config, "metadata")
    authority, prerequisite_failure = _metadata_prerequisite(config, metadata)
    if prerequisite_failure is not None:
        _publish_worker_status(
            config,
            "image",
            index=index,
            status="blocked",
            failure_category="metadata",
            reason=prerequisite_failure,
        )
        return 0
    task = None
    provenance_started = False
    provenance_result = None
    try:
        task = read_migration_task(
            config.manifest_path,
            index,
            expected_scientific_output=config.scientific_output,
            expected_control_root=config.control_root,
        )
        if (task.store_path / "zarr.json").is_file():
            provenance_started = True
            provenance_result = upgrade_store_provenance(
                task.store_path,
                dry_run=config.dry_run,
                commit_guard=_commit_guard(config),
            )
            provenance_started = False
        result = migrate_image_task(
            config.output_dir,
            task,
            metadata_csv=(
                config.scientific_output / "metadata.csv"
                if (config.scientific_output / "metadata.csv").is_file()
                else None
            ),
            overlay_alpha=config.overlay_alpha,
            dry_run=config.dry_run,
            commit_guard=_commit_guard(config),
        )
        if not config.dry_run:
            assert authority is not None
            publish_migration_task_status(
                config.control_root,
                manifest_path=config.manifest_path,
                expected_scientific_output=config.scientific_output,
                generation=config.generation,
                metadata_terminal_digest=authority.terminal_receipt_digest,
                result=result,
                commit_guard=_commit_guard(config),
            )
        _publish_worker_status(
            config,
            "image",
            index=index,
            status="complete",
            extra={
                "result": asdict(result),
                "provenance": (
                    None
                    if provenance_result is None
                    else {
                        "store_path": str(provenance_result.store_path),
                        "schema_before": provenance_result.schema_before,
                        "upgraded": provenance_result.upgraded,
                    }
                ),
            },
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - isolate each array index
        _publish_worker_status(
            config,
            "image",
            index=index,
            status="failed",
            failure_category="provenance" if provenance_started else "image",
            reason=f"{type(exc).__name__}: {exc}",
            extra={
                "target": (
                    str(task.hdf_path or task.store_path)
                    if task is not None
                    else f"manifest index {index}"
                )
            },
        )
        return 1


def _seal_from_path(path: Path) -> MigrationImageSeal | None:
    """Load a typed image seal without treating scheduler status as authority."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    expected = {
        "schema_version", "generation", "manifest_digest",
        "ordered_status_digest", "metadata_terminal_digest", "clean", "failures",
    }
    if (
        not isinstance(raw, Mapping)
        or set(raw) != expected
        or raw.get("schema_version") != 1
        or not all(
            isinstance(raw.get(field), str)
            for field in (
                "generation", "manifest_digest", "ordered_status_digest",
                "metadata_terminal_digest",
            )
        )
        or not isinstance(raw.get("clean"), bool)
        or not isinstance(raw.get("failures"), list)
        or not all(isinstance(value, str) for value in raw["failures"])
    ):
        return None
    try:
        return MigrationImageSeal(
            generation=raw["generation"],
            manifest_digest=raw["manifest_digest"],
            ordered_status_digest=raw["ordered_status_digest"],
            metadata_terminal_digest=raw["metadata_terminal_digest"],
            clean=raw["clean"],
            failures=tuple(raw["failures"]),
            seal_path=path,
        )
    except (KeyError, TypeError):
        return None


def _run_image_seal_worker(config: _WorkerConfig) -> int:
    """Seal exact per-image outcomes after all image chunks are terminal."""
    assert_generation_active(config.lifecycle_root, config.generation)
    metadata = _read_worker_status(config, "metadata")
    authority = (
        None
        if metadata is None
        else _authority_from_payload(metadata.get("authority"))
    )
    if config.dry_run:
        failures = [
            f"image index {index} did not complete"
            for index in range(config.task_count)
            if (_read_worker_status(config, "image", index) or {}).get("status")
            != "complete"
        ]
        _publish_worker_status(
            config,
            "seal",
            status="complete" if not failures else "failed",
            failure_category=None if not failures else "image_seal",
            reason=None if not failures else "; ".join(failures),
            extra={"clean": not failures, "failures": failures},
        )
        return 0 if not failures else 1
    digest = "" if authority is None else authority.terminal_receipt_digest
    try:
        seal = seal_migration_image_stage(
            config.control_root,
            manifest_path=config.manifest_path,
            expected_scientific_output=config.scientific_output,
            generation=config.generation,
            metadata_terminal_digest=digest,
            commit_guard=_commit_guard(config),
        )
        _publish_worker_status(
            config, "seal",
            status="complete" if seal.clean else "failed",
            failure_category=None if seal.clean else "image_seal",
            reason=None if seal.clean else "; ".join(seal.failures),
            extra={"clean": seal.clean, "failures": list(seal.failures)},
        )
        return 0 if seal.clean else 1
    except Exception as exc:  # noqa: BLE001 - preserve stage failure evidence
        _publish_worker_status(
            config, "seal", status="failed", failure_category="image_seal",
            reason=f"{type(exc).__name__}: {exc}",
        )
        return 1


def _run_reclaim_worker(config: _WorkerConfig, index: int) -> int:
    """Run one strong source-reclamation check after the image barrier."""
    assert_generation_active(config.lifecycle_root, config.generation)
    task = None
    seal_path = migration_image_seal_path(config.control_root, config.generation)
    image_seal = _seal_from_path(seal_path)
    metadata = _read_worker_status(config, "metadata")
    authority, metadata_failure = _metadata_prerequisite(config, metadata)
    authorization_failure: str | None = None
    if not config.delete_sources:
        authorization_failure = "source deletion was not requested"
    elif config.dry_run:
        authorization_failure = "dry-run cannot reclaim sources"
    elif metadata_failure is not None or authority is None:
        authorization_failure = metadata_failure or "metadata authority is missing"
    elif image_seal is None or not image_seal.clean:
        authorization_failure = "image seal is missing or not clean"
    elif (
        image_seal.generation != config.generation
        or image_seal.manifest_digest != config.inventory_digest
        or image_seal.metadata_terminal_digest
        != authority.terminal_receipt_digest
    ):
        authorization_failure = "image seal does not bind current migration authority"
    elif not valid_migration_image_seal(
        config.control_root,
        image_seal,
        manifest_path=config.manifest_path,
        expected_scientific_output=config.scientific_output,
    ):
        authorization_failure = "image seal is not current canonical authority"
    try:
        task = read_migration_task(
            config.manifest_path,
            index,
            expected_scientific_output=config.scientific_output,
            expected_control_root=config.control_root,
        )
        result = (
            replace(
                _retained_reclaim_result(config.output_dir, task, None),
                reason=authorization_failure,
            )
            if authorization_failure is not None
            else reclaim_image_sources(
                config.output_dir,
                task,
                metadata_csv=(
                    config.scientific_output / "metadata.csv"
                    if (config.scientific_output / "metadata.csv").is_file()
                    else None
                ),
                commit_guard=_commit_guard(config),
            )
        )
        publish_migration_reclaim_status(
            config.control_root,
            manifest_path=config.manifest_path,
            expected_scientific_output=config.scientific_output,
            generation=config.generation,
            result=result,
            commit_guard=_commit_guard(config),
        )
        _publish_worker_status(
            config,
            "reclaim",
            index=index,
            status="complete" if result.reason is None else "blocked",
            failure_category=None if result.reason is None else "reclaim_noop",
            reason=result.reason,
        )
        return 0 if result.reason is None else 1
    except Exception as exc:  # noqa: BLE001 - preserve exact reclaim failure
        _publish_worker_status(
            config,
            "reclaim",
            index=index,
            status="failed",
            failure_category="reclaim",
            reason=f"{type(exc).__name__}: {exc}",
            extra={
                "target": (
                    f"manifest index {index}"
                    if task is None
                    else str(task.store_path)
                )
            },
        )
        return 1


def _reclaim_seal_from_path(path: Path) -> MigrationReclaimSeal | None:
    """Load typed reclaim seal evidence from its canonical payload."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "schema_version", "generation", "manifest_digest",
            "ordered_reclaim_status_digest", "deletion_requested", "clean", "failures",
        }
        if (
            not isinstance(raw, Mapping)
            or set(raw) != expected
            or raw.get("schema_version") != 1
            or not all(
                isinstance(raw.get(field), str)
                for field in (
                    "generation", "manifest_digest", "ordered_reclaim_status_digest"
                )
            )
            or not isinstance(raw.get("deletion_requested"), bool)
            or not isinstance(raw.get("clean"), bool)
            or not isinstance(raw.get("failures"), list)
            or not all(isinstance(value, str) for value in raw["failures"])
        ):
            return None
        return MigrationReclaimSeal(
            generation=raw["generation"],
            manifest_digest=raw["manifest_digest"],
            ordered_reclaim_status_digest=raw["ordered_reclaim_status_digest"],
            deletion_requested=raw["deletion_requested"],
            clean=raw["clean"],
            failures=tuple(raw["failures"]),
            seal_path=path,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
        return None


def _run_reclaim_seal_worker(config: _WorkerConfig) -> int:
    """Seal exact source transitions after all reclaim chunks are terminal."""
    assert_generation_active(config.lifecycle_root, config.generation)
    image_seal = _seal_from_path(
        migration_image_seal_path(config.control_root, config.generation)
    )
    try:
        seal = seal_migration_reclaim_stage(
            config.control_root,
            manifest_path=config.manifest_path,
            expected_scientific_output=config.scientific_output,
            generation=config.generation,
            deletion_requested=config.delete_sources,
            image_seal=image_seal,
            commit_guard=_commit_guard(config),
        )
        clean = seal is not None and seal.clean
        _publish_worker_status(
            config, "reclaim-seal",
            status="complete" if clean else "failed",
            failure_category=None if clean else "reclaim",
            reason=None if clean else "reclaim seal is not clean",
        )
        return 0 if clean else 1
    except Exception as exc:  # noqa: BLE001 - preserve stage failure evidence
        _publish_worker_status(
            config, "reclaim-seal", status="failed", failure_category="reclaim",
            reason=f"{type(exc).__name__}: {exc}",
        )
        return 1


def _metadata_result_from_status(
    config: _WorkerConfig, status: Mapping[str, Any] | None
) -> MetadataPassResult:
    """Recover typed metadata-stage evidence for the terminal finalizer."""
    authority, reason = _metadata_prerequisite(config, status)
    if reason is not None:
        return MetadataPassResult(0, ((config.output_dir, reason),), None)
    assert status is not None
    return MetadataPassResult(
        status["headers_migrated"], (), authority
    )


def _image_report(config: _WorkerConfig) -> tuple[MigrationReport, tuple[tuple[Path, str], ...]]:
    """Rebuild summary counters and exact failures from indexed statuses."""
    tasks = tuple(
        read_migration_task(
            config.manifest_path,
            index,
            expected_scientific_output=config.scientific_output,
            expected_control_root=config.control_root,
        )
        for index in range(config.task_count)
    )
    results: list[MigrationImageResult] = []
    failures: list[tuple[Path, str]] = []
    provenance_upgraded = 0
    provenance_failures: list[tuple[Path, str]] = []
    for task in tasks:
        status = _read_worker_status(config, "image", task.index)
        if status is None or status.get("status") != "complete":
            reason = (
                "image status is missing"
                if status is None
                else str(status.get("reason") or "image did not complete")
            )
            target = task.hdf_path or task.store_path
            if status is not None and status.get("failure_category") == "provenance":
                provenance_failures.append((task.store_path, reason))
            else:
                failures.append((target, reason))
            continue
        provenance = status.get("provenance")
        if provenance is not None:
            valid_provenance = (
                isinstance(provenance, Mapping)
                and set(provenance)
                == {"store_path", "schema_before", "upgraded"}
                and provenance.get("store_path") == str(task.store_path)
                and isinstance(provenance.get("upgraded"), bool)
                and (
                    provenance.get("schema_before") is None
                    or (
                        isinstance(provenance.get("schema_before"), int)
                        and not isinstance(provenance.get("schema_before"), bool)
                    )
                )
            )
            if not valid_provenance:
                provenance_failures.append(
                    (task.store_path, "provenance result evidence is invalid")
                )
            else:
                provenance_upgraded += int(provenance["upgraded"])
        raw = status.get("result")
        expected_result_fields = {
            "index", "dataset", "stem", "work_id", "converted",
            "table_installed", "overlay_rendered", "marker_digest", "skipped",
        }
        if (
            not isinstance(raw, Mapping)
            or set(raw) != expected_result_fields
            or not isinstance(raw.get("index"), int)
            or isinstance(raw.get("index"), bool)
            or not all(
                isinstance(raw.get(field), str)
                for field in ("dataset", "stem", "work_id", "marker_digest")
            )
            or not all(
                isinstance(raw.get(field), bool)
                for field in (
                    "converted", "table_installed", "overlay_rendered", "skipped"
                )
            )
        ):
            failures.append((task.store_path, "image result evidence is missing"))
            continue
        results.append(
            MigrationImageResult(
                index=raw["index"],
                dataset=raw["dataset"],
                stem=raw["stem"],
                work_id=raw["work_id"],
                converted=raw["converted"],
                table_installed=raw["table_installed"],
                overlay_rendered=raw["overlay_rendered"],
                marker_digest=raw["marker_digest"],
                skipped=raw["skipped"],
            )
        )
    report = replace(
        _report_from_image_results(tasks, results),
        provenance_upgraded=provenance_upgraded,
        provenance_failures=tuple(provenance_failures),
    )
    return report, tuple(failures)


def _placeholder_image_seal(config: _WorkerConfig, reason: str) -> MigrationImageSeal:
    """Represent missing seal evidence without manufacturing authority."""
    return MigrationImageSeal(
        generation=config.generation,
        manifest_digest=config.inventory_digest,
        ordered_status_digest="",
        metadata_terminal_digest="",
        clean=False,
        failures=(reason,),
        seal_path=(
            migration_image_seal_path(config.control_root, config.generation)
        ),
    )


def _publish_terminal_and_close(
    config: _WorkerConfig,
    *,
    succeeded: bool,
    failure_category: str | None,
    reason: str | None,
    report: MigrationReport,
) -> None:
    """Durably publish terminal evidence, then and only then close lifecycle."""
    if config.dry_run:
        publish_migration_terminal_status(
            config.output_dir,
            generation=config.generation,
            succeeded=succeeded,
            failure_category=failure_category,
            reason=reason,
            report=report,
            commit_guard=_commit_guard(config),
            control_root=config.control_root,
        )
        _publish_worker_status(
            config, "terminal",
            status="complete" if succeeded else "failed",
            failure_category=failure_category,
            reason=reason,
        )
    else:
        publish_migration_terminal_status(
            config.output_dir,
            generation=config.generation,
            succeeded=succeeded,
            failure_category=failure_category,
            reason=reason,
            report=report,
            commit_guard=_commit_guard(config),
            control_root=config.control_root,
        )
    close_migration_generation(
        config.lifecycle_root,
        generation=config.generation,
        succeeded=succeeded,
        reason=reason,
    )


def _run_finalizer_worker(config: _WorkerConfig) -> int:
    """Always terminalize and close the generation reached through afterany."""
    assert_generation_active(config.lifecycle_root, config.generation)
    metadata_status = _read_worker_status(config, "metadata")
    metadata_result = _metadata_result_from_status(config, metadata_status)
    try:
        report, image_failures = _image_report(config)
    except Exception as exc:  # noqa: BLE001 - corrupt evidence must still close
        reason = (
            "invalid upstream evidence prevented finalization: "
            f"{type(exc).__name__}: {exc}"
        )
        report = MigrationReport(
            publication_failures=((config.control_root, reason),)
        )
        _publish_terminal_and_close(
            config, succeeded=False, failure_category="completion",
            reason=reason, report=report,
        )
        return 1
    if config.dry_run:
        seal_status = _read_worker_status(config, "seal")
        failure_category: str | None = None
        finalization_reason: str | None = None
        if metadata_result.failures:
            failure_category = "metadata"
            finalization_reason = metadata_result.failures[0][1]
        elif seal_status is None or seal_status.get("status") != "complete":
            failure_category = "image_seal"
            finalization_reason = (
                "image seal status is missing"
                if seal_status is None
                else str(seal_status.get("reason") or "image seal failed")
            )
        final_report = replace(
            report,
            headers_migrated=metadata_result.headers_migrated,
            header_failures=metadata_result.failures,
            failed=report.failed + image_failures,
        )
        _publish_terminal_and_close(
            config, succeeded=failure_category is None,
            failure_category=failure_category,
            reason=finalization_reason,
            report=final_report,
        )
        return 0 if failure_category is None and final_report.ok else 1

    image_seal_path = migration_image_seal_path(config.control_root, config.generation)
    image_seal = _seal_from_path(image_seal_path) or _placeholder_image_seal(
        config, "image seal is missing"
    )
    reclaim_seal = (
        _reclaim_seal_from_path(
            migration_reclaim_seal_path(config.control_root, config.generation)
        )
        if config.delete_sources
        else None
    )
    reclaim_failure_rows: list[tuple[Path, str]] = []
    if config.delete_sources:
        for index in range(config.task_count):
            reclaim_status = _read_worker_status(config, "reclaim", index)
            if (
                reclaim_status is None
                or reclaim_status.get("status") != "complete"
            ):
                reclaim_failure_rows.append(
                    (
                        config.output_dir,
                        (
                            "reclaim did not complete"
                            if reclaim_status is None
                            else str(
                                reclaim_status.get("reason")
                                or "reclaim did not complete"
                            )
                        ),
                    )
                )
    reclaim_failures = tuple(reclaim_failure_rows)
    try:
        final_report = finalize_migration_attempt(
            config.output_dir,
            manifest_path=config.manifest_path,
            expected_scientific_output=config.scientific_output,
            generation=config.generation,
            metadata_pass=metadata_result,
            image_seal=image_seal,
            reclaim_seal=reclaim_seal,
            deletion_requested=config.delete_sources,
            dry_run=False,
            report=report,
            image_failures=image_failures,
            reclaim_failures=reclaim_failures,
            commit_guard=_commit_guard(config),
            control_root=config.control_root,
        )
        return 0 if final_report.ok else 1
    except Exception as exc:  # noqa: BLE001 - finalizer must close its generation
        reason = f"finalizer failed: {type(exc).__name__}: {exc}"
        failed_report = replace(
            report,
            publication_failures=report.publication_failures
            + ((config.output_dir, reason),),
        )
        _publish_terminal_and_close(
            config, succeeded=False, failure_category="completion",
            reason=reason, report=failed_report,
        )
        return 1


@click.group(name="migrate-worker")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
@click.pass_context
def migration_worker_cli(ctx: click.Context, config_path: Path) -> None:
    """Run one private migration stage from a generated immutable config."""
    ctx.obj = _load_worker_config(config_path)


def _exit_with(code: int) -> None:
    """Return an exact worker exit code without rewriting typed status output."""
    if code:
        raise click.exceptions.Exit(code)


@migration_worker_cli.command("metadata")
@click.pass_obj
def _metadata_command(config: _WorkerConfig) -> None:
    """Run the metadata singleton."""
    _exit_with(_run_metadata_worker(config))


@migration_worker_cli.command("image")
@click.option("--index", required=True, type=click.IntRange(min=0))
@click.pass_obj
def _image_command(config: _WorkerConfig, index: int) -> None:
    """Run one indexed image migration."""
    _exit_with(_run_image_worker(config, index))


@migration_worker_cli.command("seal")
@click.pass_obj
def _seal_command(config: _WorkerConfig) -> None:
    """Seal the image stage."""
    _exit_with(_run_image_seal_worker(config))


@migration_worker_cli.command("reclaim")
@click.option("--index", required=True, type=click.IntRange(min=0))
@click.pass_obj
def _reclaim_command(config: _WorkerConfig, index: int) -> None:
    """Run one indexed source reclaim."""
    _exit_with(_run_reclaim_worker(config, index))


@migration_worker_cli.command("reclaim-seal")
@click.pass_obj
def _reclaim_seal_command(config: _WorkerConfig) -> None:
    """Seal the reclaim stage."""
    _exit_with(_run_reclaim_seal_worker(config))


@migration_worker_cli.command("finalize")
@click.pass_obj
def _finalize_command(config: _WorkerConfig) -> None:
    """Publish terminal status and close the lifecycle generation."""
    _exit_with(_run_finalizer_worker(config))


if __name__ == "__main__":
    migration_worker_cli()


__all__ = ["migration_worker_cli", "migration_worker_status_path"]
