"""Dispatcher-fed SLURM planning for ``--mode migrate``."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
from typing import Any

from phenotypic.sdk_ import deliverables_dir, progress_dir
from phenotypic.sdk_.slurm import (
    SlurmArrayScriptSpec,
    calculate_optimal_array_chunks,
    get_slurm_array_limit,
    get_slurm_max_submit_jobs,
    write_slurm_array_script,
)
from phenotypic.sdk_.slurm._dispatcher import SlurmDependencyKind

from ._cli_migrate_manifest import (
    discover_migration_tasks,
    validate_migration_generation,
    write_migration_manifest,
)
from ._cli_slurm_submission import (
    SLURMScriptChainSubmission,
    submit_slurm_script_chain,
)
from ._cli_utils import get_python_command


@dataclass(frozen=True)
class MigrationSlurmPlan:
    """One immutable migration generation and its flat scheduler chain."""

    generation: str
    control_root: Path
    manifest_path: Path
    flat_scripts: tuple[Path, ...]
    finalizer_script: Path
    task_count: int


def _dry_control_root(output_dir: Path, generation: str) -> Path:
    """Return a generation-scoped shared user cache outside scientific data."""
    cache_base = Path(
        os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    ).expanduser()
    run_digest = hashlib.sha256(str(output_dir.resolve()).encode()).hexdigest()[:20]
    control_root = (
        cache_base / "phenotypic" / "migration" / run_digest / generation
    ).resolve()
    scientific_root = output_dir.resolve()
    if control_root == scientific_root or control_root.is_relative_to(scientific_root):
        raise ValueError("dry-run control root must be outside scientific output")
    return control_root


def _migration_control_root(
    output_dir: Path, generation: str, *, dry_run: bool
) -> Path:
    """Return the isolated control root for one migration attempt."""
    if dry_run:
        return _dry_control_root(output_dir, generation)
    return (progress_dir(output_dir) / "migration" / generation).resolve()


def _chunk_limit() -> int:
    """Return the tighter scheduler bound after reserving two submit slots."""
    array_limit = get_slurm_array_limit()
    max_submit = get_slurm_max_submit_jobs()
    if max_submit is not None and max_submit < 3:
        raise ValueError(
            "SLURM MaxSubmitJobs must be at least 3 for migration dispatch "
            "(active dispatcher, next cohort, successor dispatcher/finalizer)."
        )
    submit_capacity = array_limit if max_submit is None else max_submit - 2
    limit = min(array_limit, submit_capacity)
    if limit < 1:
        raise ValueError("SLURM migration has no array capacity after reservation")
    return limit


def _worker_body(config_path: Path, command: str, *, indexed: bool) -> str:
    """Render one internal-worker invocation for a generated array script."""
    python_parts, _ = get_python_command(for_slurm=True)
    prefix = " ".join(shlex.quote(part) for part in python_parts)
    fixed = (
        f"{prefix} -m phenotypic._cli._cli_migrate_worker "
        f"--config {shlex.quote(str(config_path))} {command}"
    )
    return f"{fixed} --index \"$CURRENT_TASK_INDEX\"" if indexed else fixed


def _write_stage_script(
    *,
    script_path: Path,
    log_path: Path,
    job_name: str,
    slurm_args: dict[str, Any],
    config_path: Path,
    command: str,
    indices: list[int],
    indexed: bool,
) -> Path:
    """Write one array or singleton member of the flat chain."""
    return write_slurm_array_script(
        script_path,
        SlurmArrayScriptSpec(
            job_name=job_name,
            slurm_args=slurm_args,
            log_path=log_path,
            error_log_path=log_path.with_name(log_path.name + ".err"),
            task_indices=indices,
            body=_worker_body(config_path, command, indexed=indexed),
            comments=("# Dispatcher-fed migration stage; continuation is afterany.",),
        ),
    )


def generate_migration_slurm_plan(
    output_dir: Path,
    *,
    slurm_args: dict[str, Any],
    overlay_alpha: float = 0.3,
    delete_sources: bool = False,
    dry_run: bool = False,
    generation: str,
) -> MigrationSlurmPlan:
    """Build one bounded flat migration chain without submitting jobs."""
    output_dir = Path(output_dir).resolve()
    generation = validate_migration_generation(generation)
    if not math.isfinite(overlay_alpha) or not 0.0 <= overlay_alpha <= 1.0:
        raise ValueError("migration overlay alpha must be finite and within [0, 1]")
    tasks = tuple(discover_migration_tasks(output_dir))
    limit = _chunk_limit()
    control_root = _migration_control_root(
        output_dir, generation, dry_run=dry_run
    )
    control_root.mkdir(parents=True, exist_ok=True)
    manifest = write_migration_manifest(
        output_dir,
        generation=generation,
        scientific_output=deliverables_dir(output_dir),
        tasks=tasks,
        control_root=control_root,
    )
    manifest_path = control_root / "migration_manifest.json"
    config_path = control_root / "migration_config.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generation": generation,
                "output_dir": str(output_dir),
                "scientific_output": str(deliverables_dir(output_dir)),
                "control_root": str(control_root),
                "manifest_path": str(manifest_path),
                "inventory_digest": manifest.inventory_digest,
                "task_count": len(tasks),
                "overlay_alpha": float(overlay_alpha),
                "delete_sources": bool(delete_sources),
                "dry_run": bool(dry_run),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    script_dir = control_root / "scripts"
    log_dir = control_root / "logs"
    script_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    flat_scripts: list[Path] = []
    flat_scripts.append(
        _write_stage_script(
            script_path=script_dir / "metadata.sh",
            log_path=log_dir / "metadata_%A_%a.log",
            job_name="pht-migrate-meta",
            slurm_args=slurm_args,
            config_path=config_path,
            command="metadata",
            indices=[0],
            indexed=False,
        )
    )
    chunks = calculate_optimal_array_chunks(len(tasks), limit)
    for chunk_index, (start, end) in enumerate(chunks):
        flat_scripts.append(
            _write_stage_script(
                script_path=script_dir / f"image_chunk{chunk_index}.sh",
                log_path=log_dir / f"image_chunk{chunk_index}_%A_%a.log",
                job_name=f"pht-migrate-img-{chunk_index}",
                slurm_args=slurm_args,
                config_path=config_path,
                command="image",
                indices=list(range(start, end)),
                indexed=True,
            )
        )
    flat_scripts.append(
        _write_stage_script(
            script_path=script_dir / "image_seal.sh",
            log_path=log_dir / "image_seal_%A_%a.log",
            job_name="pht-migrate-seal",
            slurm_args=slurm_args,
            config_path=config_path,
            command="seal",
            indices=[0],
            indexed=False,
        )
    )
    if delete_sources:
        for chunk_index, (start, end) in enumerate(chunks):
            flat_scripts.append(
                _write_stage_script(
                    script_path=script_dir / f"reclaim_chunk{chunk_index}.sh",
                    log_path=log_dir / f"reclaim_chunk{chunk_index}_%A_%a.log",
                    job_name=f"pht-migrate-reclaim-{chunk_index}",
                    slurm_args=slurm_args,
                    config_path=config_path,
                    command="reclaim",
                    indices=list(range(start, end)),
                    indexed=True,
                )
            )
        flat_scripts.append(
            _write_stage_script(
                script_path=script_dir / "reclaim_seal.sh",
                log_path=log_dir / "reclaim_seal_%A_%a.log",
                job_name="pht-migrate-reclaim-seal",
                slurm_args=slurm_args,
                config_path=config_path,
                command="reclaim-seal",
                indices=[0],
                indexed=False,
            )
        )
    finalizer_script = _write_stage_script(
        script_path=script_dir / "finalize.sh",
        log_path=log_dir / "finalize_%A_%a.log",
        job_name="pht-migrate-finalize",
        slurm_args=slurm_args,
        config_path=config_path,
        command="finalize",
        indices=[0],
        indexed=False,
    )
    return MigrationSlurmPlan(
        generation=generation,
        control_root=control_root,
        manifest_path=manifest_path,
        flat_scripts=tuple(flat_scripts),
        finalizer_script=finalizer_script,
        task_count=len(tasks),
    )


def submit_migration_slurm_plan(
    plan: MigrationSlurmPlan,
    *,
    slurm_args: dict[str, Any],
    console: Any,
) -> SLURMScriptChainSubmission:
    """Delegate the complete ordered plan to the shared drip-feed dispatcher."""
    config = json.loads(
        (plan.control_root / "migration_config.json").read_text(encoding="utf-8")
    )
    lifecycle_output = (
        plan.control_root
        if config.get("dry_run") is True
        else Path(str(config["output_dir"])).resolve()
    )
    dependencies: tuple[SlurmDependencyKind, ...] = (
        "afterany",
    ) * len(plan.flat_scripts)
    return submit_slurm_script_chain(
        flat_chunk_scripts=plan.flat_scripts,
        output_dir=lifecycle_output,
        control_output_dir=plan.control_root,
        slurm_args=slurm_args,
        console=console,
        finalizer_script=plan.finalizer_script,
        continuation_dependency_kinds=dependencies,
        generation=plan.generation,
    )


__all__ = [
    "MigrationSlurmPlan",
    "generate_migration_slurm_plan",
    "submit_migration_slurm_plan",
]
