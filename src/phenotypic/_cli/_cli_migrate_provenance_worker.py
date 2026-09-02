"""Internal workers for provenance-only migration arrays."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal, Mapping

import click

from phenotypic.sdk_ import CommitGuard
from phenotypic.sdk_._hdf_to_zarr import MigrationReport

from ._cli_migrate import (
    close_migration_generation,
    publish_migration_terminal_status,
)
from ._cli_migrate_manifest import validate_migration_generation
from ._cli_migrate_provenance import (
    ProvenanceMigrationTarget,
    provenance_migration_lifecycle_root,
    upgrade_store_provenance,
)
from ._cli_migrate_provenance_manifest import (
    _read_manifest,
    provenance_worker_status_path,
    publish_provenance_worker_status,
    read_provenance_migration_seal,
    read_provenance_migration_task,
    seal_provenance_migration,
)
from ._cli_slurm_lifecycle import (
    assert_generation_active,
    generation_publication_guard,
)


@dataclass(frozen=True)
class _ProvenanceWorkerConfig:
    """Validated immutable identity for provenance-only workers."""

    generation: str
    target_kind: Literal["direct_store", "process_tree"]
    target_root: Path
    lifecycle_root: Path
    control_root: Path
    manifest_path: Path
    inventory_digest: str
    task_count: int
    dry_run: bool


def _load_worker_config(path: Path) -> _ProvenanceWorkerConfig:
    """Load a caller-bound versioned provenance-only worker config."""
    config_path = Path(path).absolute()
    if config_path.is_symlink():
        raise ValueError("provenance migration config cannot be a symlink")
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid provenance migration worker config") from exc
    fields = {
        "schema_version", "topology", "generation", "target_kind",
        "target_root", "lifecycle_root", "control_root", "manifest_path",
        "inventory_digest", "task_count", "dry_run",
    }
    if (
        not isinstance(raw, Mapping)
        or set(raw) != fields
        or raw.get("schema_version") != 1
        or raw.get("topology") != "provenance_only"
        or raw.get("target_kind") not in {"direct_store", "process_tree"}
        or not isinstance(raw.get("dry_run"), bool)
    ):
        raise ValueError("invalid provenance migration worker config schema")
    generation = validate_migration_generation(raw.get("generation"))
    path_fields = (
        "target_root", "lifecycle_root", "control_root", "manifest_path",
    )
    if any(
        not isinstance(raw[field], str) or not Path(raw[field]).is_absolute()
        for field in path_fields
    ):
        raise ValueError("provenance migration config paths must be absolute")
    target_root = Path(raw["target_root"]).resolve()
    lifecycle_root = Path(raw["lifecycle_root"]).resolve()
    control_root = Path(raw["control_root"]).resolve()
    manifest_path = Path(raw["manifest_path"]).resolve()
    if config_path.resolve().parent != control_root:
        raise ValueError("provenance migration config has wrong control root")
    digest = raw.get("inventory_digest")
    task_count = raw.get("task_count")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or not isinstance(task_count, int)
        or isinstance(task_count, bool)
        or task_count < 1
    ):
        raise ValueError("invalid provenance migration worker identity")
    target = ProvenanceMigrationTarget(
        raw["target_kind"], target_root, ()
    )
    expected_lifecycle_root = (
        control_root
        if raw["dry_run"]
        else provenance_migration_lifecycle_root(target).resolve()
    )
    if lifecycle_root != expected_lifecycle_root:
        raise ValueError(
            "provenance migration config has wrong lifecycle root"
        )
    manifest = _read_manifest(
        manifest_path,
        expected_target_root=target_root,
        expected_control_root=control_root,
    )
    if (
        manifest.generation != generation
        or manifest.target_kind != raw["target_kind"]
        or manifest.inventory_digest != digest
        or manifest.task_count != task_count
    ):
        raise ValueError("provenance migration config does not match manifest")
    return _ProvenanceWorkerConfig(
        generation=generation,
        target_kind=raw["target_kind"],
        target_root=target_root,
        lifecycle_root=lifecycle_root,
        control_root=control_root,
        manifest_path=manifest_path,
        inventory_digest=digest,
        task_count=task_count,
        dry_run=raw["dry_run"],
    )


def _commit_guard(config: _ProvenanceWorkerConfig) -> CommitGuard:
    """Fence every store, status, seal, and terminal publication."""
    return lambda: generation_publication_guard(
        config.lifecycle_root, config.generation
    )


def _run_store_worker(config: _ProvenanceWorkerConfig, index: int) -> int:
    """Upgrade one store root and publish an isolated typed result."""
    assert_generation_active(config.lifecycle_root, config.generation)
    try:
        task = read_provenance_migration_task(
            config.manifest_path,
            index,
            expected_target_root=config.target_root,
            expected_control_root=config.control_root,
        )
        result = upgrade_store_provenance(
            task.store_path,
            dry_run=config.dry_run,
            commit_guard=_commit_guard(config),
        )
        publish_provenance_worker_status(
            config.manifest_path,
            expected_target_root=config.target_root,
            expected_control_root=config.control_root,
            generation=config.generation,
            index=index,
            result=result,
            commit_guard=_commit_guard(config),
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - isolate array items
        reason = f"{type(exc).__name__}: {exc}"
        try:
            publish_provenance_worker_status(
                config.manifest_path,
                expected_target_root=config.target_root,
                expected_control_root=config.control_root,
                generation=config.generation,
                index=index,
                result=None,
                reason=reason,
                commit_guard=_commit_guard(config),
            )
        except Exception:  # noqa: BLE001 - stale generations cannot publish
            pass
        return 1


def _run_seal_worker(config: _ProvenanceWorkerConfig) -> int:
    """Barrier all exact store statuses into one generation-bound seal."""
    assert_generation_active(config.lifecycle_root, config.generation)
    try:
        seal = seal_provenance_migration(
            config.manifest_path,
            expected_target_root=config.target_root,
            expected_control_root=config.control_root,
            generation=config.generation,
            commit_guard=_commit_guard(config),
        )
    except Exception:  # noqa: BLE001 - finalizer records missing seal authority
        return 1
    return 0 if seal.clean else 1


def _run_finalizer_worker(config: _ProvenanceWorkerConfig) -> int:
    """Publish terminal report authority and close the reached generation."""
    assert_generation_active(config.lifecycle_root, config.generation)
    try:
        seal = read_provenance_migration_seal(
            config.manifest_path,
            expected_target_root=config.target_root,
            expected_control_root=config.control_root,
            generation=config.generation,
        )
        failures = seal.failures
        upgraded = seal.upgraded
        reason = None if seal.clean else failures[0][1]
    except Exception as exc:  # noqa: BLE001 - afterany must always close
        reason = f"provenance seal validation failed: {type(exc).__name__}: {exc}"
        failures = ((config.target_root, reason),)
        upgraded = 0
    report = MigrationReport(
        provenance_upgraded=upgraded,
        provenance_failures=failures,
    )
    succeeded = not failures
    try:
        publish_migration_terminal_status(
            config.target_root,
            generation=config.generation,
            succeeded=succeeded,
            failure_category=None if succeeded else "provenance",
            reason=reason,
            report=report,
            commit_guard=_commit_guard(config),
            control_root=config.control_root,
        )
    except Exception:  # noqa: BLE001 - preserve recoverable active lifecycle
        return 1
    close_migration_generation(
        config.lifecycle_root,
        generation=config.generation,
        succeeded=succeeded,
        reason=reason,
    )
    return 0 if succeeded else 1


@click.group(name="provenance-migrate-worker")
@click.option(
    "--config", "config_path", required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
@click.pass_context
def provenance_migration_worker_cli(
    ctx: click.Context, config_path: Path
) -> None:
    """Run one private provenance-only migration stage."""
    ctx.obj = _load_worker_config(config_path)


def _exit_with(code: int) -> None:
    if code:
        raise click.exceptions.Exit(code)


@provenance_migration_worker_cli.command("store")
@click.option("--index", required=True, type=click.IntRange(min=0))
@click.pass_obj
def _store_command(config: _ProvenanceWorkerConfig, index: int) -> None:
    _exit_with(_run_store_worker(config, index))


@provenance_migration_worker_cli.command("seal")
@click.pass_obj
def _seal_command(config: _ProvenanceWorkerConfig) -> None:
    _exit_with(_run_seal_worker(config))


@provenance_migration_worker_cli.command("finalize")
@click.pass_obj
def _finalize_command(config: _ProvenanceWorkerConfig) -> None:
    _exit_with(_run_finalizer_worker(config))


if __name__ == "__main__":  # pragma: no cover
    provenance_migration_worker_cli()


__all__ = [
    "provenance_migration_worker_cli",
    "provenance_worker_status_path",
]
