"""Internal indexed worker for dispatcher-fed migration SLURM jobs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any, Mapping

import click

from phenotypic.sdk_ import CommitGuard, atomic_write_json, deliverables_dir
from phenotypic.sdk_._hdf_to_zarr import MigrationReport
from phenotypic.sdk_._metadata_migration import MetadataMigrationAuthority

from ._cli_migrate import (
    MetadataPassResult,
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
    publish_migration_reclaim_status,
    publish_migration_task_status,
    read_migration_task,
    seal_migration_image_stage,
    seal_migration_reclaim_stage,
)
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
    suffix = stage if index is None else f"{stage}_{index}"
    return Path(control_root) / "worker_status" / generation / f"{suffix}.json"


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
    generation = raw.get("generation")
    inventory_digest = raw.get("inventory_digest")
    task_count = raw.get("task_count")
    if (
        not isinstance(generation, str)
        or not generation
        or not isinstance(inventory_digest, str)
        or len(inventory_digest) != 64
        or not isinstance(task_count, int)
        or isinstance(task_count, bool)
        or task_count < 0
    ):
        raise ValueError("invalid migration worker identity fields")
    control_root = Path(str(raw["control_root"])).resolve()
    if config_path.resolve().parent != control_root:
        raise ValueError("migration worker config has wrong control root")
    output_dir = Path(str(raw["output_dir"])).resolve()
    scientific_output = Path(str(raw["scientific_output"])).resolve()
    if scientific_output != deliverables_dir(output_dir):
        raise ValueError("migration worker config has wrong scientific output")
    manifest_path = Path(str(raw["manifest_path"])).resolve()
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
    return _WorkerConfig(
        generation=generation,
        output_dir=output_dir,
        scientific_output=scientific_output,
        control_root=control_root,
        manifest_path=manifest_path,
        inventory_digest=inventory_digest,
        task_count=task_count,
        overlay_alpha=float(raw["overlay_alpha"]),
        delete_sources=bool(raw["delete_sources"]),
        dry_run=bool(raw["dry_run"]),
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
        or raw.get("generation") != config.generation
        or raw.get("manifest_digest") != config.inventory_digest
        or raw.get("stage") != stage
        or raw.get("index") != index
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
    return MetadataMigrationAuthority(
        status_path=Path(str(value["status_path"])),
        terminal_receipt_path=Path(str(value["terminal_receipt_path"])),
        terminal_receipt_digest=str(value["terminal_receipt_digest"]),
        plan_fingerprint=str(value["plan_fingerprint"]),
        source_fingerprint=str(value["source_fingerprint"]),
        resulting_fingerprint=str(value["resulting_fingerprint"]),
        compatible_noop=bool(value["compatible_noop"]),
    )


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
            commit_guard=None if config.dry_run else _commit_guard(config),
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
    if metadata is None or metadata.get("status") != "complete":
        reason = (
            "metadata stage status is missing"
            if metadata is None
            else str(metadata.get("reason") or "metadata stage failed")
        )
        _publish_worker_status(
            config,
            "image",
            index=index,
            status="blocked",
            failure_category="metadata",
            reason=reason,
        )
        return 0
    task = read_migration_task(
        config.manifest_path,
        index,
        expected_scientific_output=config.scientific_output,
        expected_control_root=config.control_root,
    )
    try:
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
            commit_guard=None if config.dry_run else _commit_guard(config),
        )
        authority = _authority_from_payload(metadata.get("authority"))
        if not config.dry_run:
            if authority is None:
                raise RuntimeError("metadata stage authority is missing")
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
            extra={"result": asdict(result)},
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - isolate each array index
        _publish_worker_status(
            config,
            "image",
            index=index,
            status="failed",
            failure_category="image",
            reason=f"{type(exc).__name__}: {exc}",
            extra={"target": str(task.hdf_path or task.store_path)},
        )
        return 1


def _seal_from_path(path: Path) -> MigrationImageSeal | None:
    """Load a typed image seal without treating scheduler status as authority."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    try:
        return MigrationImageSeal(
            generation=str(raw["generation"]),
            manifest_digest=str(raw["manifest_digest"]),
            ordered_status_digest=str(raw["ordered_status_digest"]),
            metadata_terminal_digest=str(raw["metadata_terminal_digest"]),
            clean=bool(raw["clean"]),
            failures=tuple(str(value) for value in raw["failures"]),
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
    seal = seal_migration_image_stage(
        config.control_root,
        manifest_path=config.manifest_path,
        expected_scientific_output=config.scientific_output,
        generation=config.generation,
        metadata_terminal_digest=digest,
        commit_guard=_commit_guard(config),
    )
    return 0 if seal.clean else 1


def _run_reclaim_worker(config: _WorkerConfig, index: int) -> int:
    """Run one strong source-reclamation check after the image barrier."""
    assert_generation_active(config.lifecycle_root, config.generation)
    task = read_migration_task(
        config.manifest_path,
        index,
        expected_scientific_output=config.scientific_output,
        expected_control_root=config.control_root,
    )
    seal_path = (
        config.control_root / "migration" / config.generation / "image_seal.json"
    )
    image_seal = _seal_from_path(seal_path)
    try:
        result = (
            _retained_reclaim_result(config.output_dir, task, None)
            if image_seal is None or not image_seal.clean
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
        )
        return 1


def _reclaim_seal_from_path(path: Path) -> MigrationReclaimSeal | None:
    """Load typed reclaim seal evidence from its canonical payload."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return MigrationReclaimSeal(
            generation=str(raw["generation"]),
            manifest_digest=str(raw["manifest_digest"]),
            ordered_reclaim_status_digest=str(
                raw["ordered_reclaim_status_digest"]
            ),
            deletion_requested=bool(raw["deletion_requested"]),
            clean=bool(raw["clean"]),
            failures=tuple(str(value) for value in raw["failures"]),
            seal_path=path,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
        return None


def _run_reclaim_seal_worker(config: _WorkerConfig) -> int:
    """Seal exact source transitions after all reclaim chunks are terminal."""
    assert_generation_active(config.lifecycle_root, config.generation)
    image_seal = _seal_from_path(
        config.control_root / "migration" / config.generation / "image_seal.json"
    )
    seal = seal_migration_reclaim_stage(
        config.control_root,
        manifest_path=config.manifest_path,
        expected_scientific_output=config.scientific_output,
        generation=config.generation,
        deletion_requested=config.delete_sources,
        image_seal=image_seal,
        commit_guard=_commit_guard(config),
    )
    return 0 if seal is not None and seal.clean else 1


def _metadata_result_from_status(
    config: _WorkerConfig, status: Mapping[str, Any] | None
) -> MetadataPassResult:
    """Recover typed metadata-stage evidence for the terminal finalizer."""
    if status is None or status.get("status") != "complete":
        reason = (
            "metadata stage status is missing"
            if status is None
            else str(status.get("reason") or "metadata stage failed")
        )
        return MetadataPassResult(0, ((config.output_dir, reason),), None)
    authority = _authority_from_payload(status.get("authority"))
    if authority is None and not config.dry_run:
        return MetadataPassResult(
            0,
            ((config.output_dir, "metadata stage authority is missing"),),
            None,
        )
    return MetadataPassResult(
        int(status.get("headers_migrated", 0)), (), authority
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
    for task in tasks:
        status = _read_worker_status(config, "image", task.index)
        if status is None or status.get("status") != "complete":
            failures.append(
                (
                    task.hdf_path or task.store_path,
                    "image status is missing"
                    if status is None
                    else str(status.get("reason") or "image did not complete"),
                )
            )
            continue
        raw = status.get("result")
        if not isinstance(raw, Mapping):
            failures.append((task.store_path, "image result evidence is missing"))
            continue
        results.append(
            MigrationImageResult(
                index=int(raw["index"]),
                dataset=str(raw["dataset"]),
                stem=str(raw["stem"]),
                work_id=str(raw["work_id"]),
                converted=bool(raw["converted"]),
                table_installed=bool(raw["table_installed"]),
                overlay_rendered=bool(raw["overlay_rendered"]),
                marker_digest=str(raw["marker_digest"]),
                skipped=bool(raw["skipped"]),
            )
        )
    return _report_from_image_results(tasks, results), tuple(failures)


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
            config.control_root
            / "migration"
            / config.generation
            / "image_seal.json"
        ),
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
        if config.dry_run:
            _publish_worker_status(
                config,
                "terminal",
                status="failed",
                failure_category="completion",
                reason=reason,
            )
        else:
            publish_migration_terminal_status(
                config.output_dir,
                generation=config.generation,
                succeeded=False,
                failure_category="completion",
                reason=reason,
                report=report,
                commit_guard=_commit_guard(config),
            )
        close_migration_generation(
            config.lifecycle_root,
            generation=config.generation,
            succeeded=False,
            reason=reason,
        )
        return 1
    if config.dry_run:
        seal_status = _read_worker_status(config, "seal")
        failure_category: str | None = None
        reason: str | None = None
        if metadata_result.failures:
            failure_category = "metadata"
            reason = metadata_result.failures[0][1]
        elif seal_status is None or seal_status.get("status") != "complete":
            failure_category = "image_seal"
            reason = (
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
        _publish_worker_status(
            config,
            "terminal",
            status="complete" if failure_category is None else "failed",
            failure_category=failure_category,
            reason=reason,
        )
        close_migration_generation(
            config.lifecycle_root,
            generation=config.generation,
            succeeded=failure_category is None,
            reason=reason,
        )
        return 0 if failure_category is None and final_report.ok else 1

    image_seal_path = (
        config.control_root / "migration" / config.generation / "image_seal.json"
    )
    image_seal = _seal_from_path(image_seal_path) or _placeholder_image_seal(
        config, "image seal is missing"
    )
    reclaim_seal = (
        _reclaim_seal_from_path(
            config.control_root
            / "migration"
            / config.generation
            / "reclaim_seal.json"
        )
        if config.delete_sources
        else None
    )
    reclaim_failures = tuple(
        (
            config.output_dir,
            str(status.get("reason") or "reclaim did not complete"),
        )
        for index in range(config.task_count)
        if (status := _read_worker_status(config, "reclaim", index)) is None
        or status.get("status") != "complete"
    ) if config.delete_sources else ()
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
        try:
            publish_migration_terminal_status(
                config.output_dir,
                generation=config.generation,
                succeeded=False,
                failure_category="completion",
                reason=reason,
                report=failed_report,
                commit_guard=_commit_guard(config),
            )
        finally:
            close_migration_generation(
                config.output_dir,
                generation=config.generation,
                succeeded=False,
                reason=reason,
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
