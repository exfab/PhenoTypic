"""SLURM planning and scripts for the removable recompile migration phase."""

from __future__ import annotations

import math
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from phenotypic.sdk_ import (
    MetadataMigrationReport,
    MetadataMigrationResult,
    MetadataMigrationTarget,
    atomic_write_json,
    logs_dir,
    preflight_metadata_schema,
    progress_dir,
    recompile_dir,
    slurm_scripts_dir,
)
from phenotypic.sdk_.slurm import (
    SLURM_PYTHONPATH_BOOTSTRAP_BASH,
    SlurmArrayScriptSpec,
    format_sbatch_directives,
    write_slurm_array_script,
)

from ._cli_recompile_metadata_migration import (
    RecompileMetadataMigrationError,
    _metadata_bundle_layout,
)
from ._cli_utils import SLURM_THREAD_PIN_BASH, get_python_command

_MIGRATION_DIRNAME = "metadata_migration"
_MIGRATION_MANIFEST = "migration_plan.json"
_MIGRATION_FINALIZER_STATUS = "finalizer.json"


@dataclass(frozen=True)
class RecompileMetadataSlurmPlan:
    """Read-only metadata migration plan prepared before SLURM submission."""

    output_dir: Path
    report: MetadataMigrationReport
    targets: tuple[MetadataMigrationTarget, ...]
    plan_dir: Path
    manifest_path: Path
    finalizer_status_path: Path


@dataclass(frozen=True)
class RecompileMetadataSlurmScripts:
    """Generated migration shard scripts and their singleton barrier."""

    shard_scripts: tuple[Path, ...]
    finalizer_script: Path


def plan_metadata_schema_for_slurm_recompile(
    output_dir: Path,
    *,
    attempt_id: str | None = None,
) -> RecompileMetadataSlurmPlan:
    """Preflight a recompile bundle without creating any files.

    Args:
        output_dir: Existing PhenoTypic run-output root.
        attempt_id: Optional scheduler-attempt namespace for durable state.

    Returns:
        Immutable plan containing only targets that require migration.

    Raises:
        RecompileMetadataMigrationError: Any target is blocked.
    """
    resolved_output = Path(output_dir).resolve()
    report = preflight_metadata_schema(
        _metadata_bundle_layout(resolved_output)
    )
    if report.status == "blocked":
        raise RecompileMetadataMigrationError(_blocked_result(report))

    digest = report.plan_fingerprint.removeprefix("sha256:")[:16]
    recompile_base = recompile_dir(progress_dir(resolved_output))
    if attempt_id is not None:
        recompile_base = recompile_base / "attempts" / attempt_id
    plan_dir = recompile_base / _MIGRATION_DIRNAME / digest
    return RecompileMetadataSlurmPlan(
        output_dir=resolved_output,
        report=report,
        targets=tuple(
            target
            for target in report.targets
            if target.status == "migratable"
        ),
        plan_dir=plan_dir,
        manifest_path=plan_dir / _MIGRATION_MANIFEST,
        finalizer_status_path=plan_dir / _MIGRATION_FINALIZER_STATUS,
    )


def generate_metadata_migration_slurm_scripts(
    plan: RecompileMetadataSlurmPlan,
    *,
    slurm_args: dict[str, Any],
    array_limit: int,
    attempt_id: str | None = None,
    slurm_generation: str | None = None,
    has_recompile_downstream: bool = True,
) -> RecompileMetadataSlurmScripts | None:
    """Write bounded target arrays and a singleton validation finalizer.

    Args:
        plan: Successful read-only migration preflight.
        slurm_args: SLURM directive arguments.
        array_limit: Maximum number of targets per array script.
        attempt_id: Scheduler-attempt namespace for scripts.
        slurm_generation: Lifecycle generation deactivated by terminal jobs.
        has_recompile_downstream: Whether success must keep the generation
            active for an ``afterok`` recompile continuation.

    Returns:
        Generated scripts, or ``None`` for a canonical no-op plan.

    Raises:
        ValueError: ``array_limit`` is not positive.
    """
    if array_limit <= 0:
        raise ValueError("array_limit must be positive")
    if not plan.targets:
        return None
    if attempt_id is None or not attempt_id:
        raise ValueError("attempt_id is required for SLURM migration scripts")
    if slurm_generation is None or not slurm_generation:
        raise ValueError(
            "slurm_generation is required for SLURM migration scripts"
        )

    atomic_write_json(
        plan.manifest_path,
        {
            "output_dir": str(plan.output_dir),
            "plan_fingerprint": plan.report.plan_fingerprint,
            "source_fingerprint": plan.report.source_fingerprint,
            "attempt_id": attempt_id,
            "slurm_generation": slurm_generation,
            "has_recompile_downstream": has_recompile_downstream,
            "targets": [
                {
                    "path": target.path,
                    "kind": target.kind,
                    "status": target.status,
                    "source_fingerprint": target.source_fingerprint,
                    "proposed_header_map": list(target.proposed_header_map),
                    "needs_metadata_marker": target.needs_metadata_marker,
                    "hdf_snapshot_fingerprint": (
                        target.hdf_snapshot_fingerprint
                    ),
                    "conflicts": list(target.conflicts),
                    "mixed_table": target.mixed_table,
                }
                for target in plan.targets
            ],
        },
        sort_keys=False,
    )

    script_dir = (
        slurm_scripts_dir(plan.output_dir)
        / "recompile"
        / attempt_id
        / "migration"
    )
    log_dir = logs_dir(plan.output_dir) / "slurm" / "recompile" / "migration"
    python_cmd, _ = get_python_command(for_slurm=True)
    python_str = " ".join(shlex.quote(part) for part in python_cmd)
    manifest_arg = shlex.quote(str(plan.manifest_path))
    output_arg = shlex.quote(str(plan.output_dir))
    generation_options = (
        " \\\n    --output-dir "
        + output_arg
        + " \\\n    --slurm-generation "
        + shlex.quote(slurm_generation)
        + " \\\n    --attempt-id "
        + shlex.quote(attempt_id)
    )

    shard_scripts: list[Path] = []
    chunk_count = math.ceil(len(plan.targets) / array_limit)
    for chunk_id in range(chunk_count):
        start = chunk_id * array_limit
        indices = list(
            range(start, min(start + array_limit, len(plan.targets)))
        )
        script_path = script_dir / f"metadata_migration_chunk{chunk_id}.sh"
        body = f"""\
echo "Metadata migration target index: $CURRENT_TASK_INDEX"

{python_str} -m phenotypic._cli._cli_recompile_metadata_migration_worker \\
    --task-manifest {manifest_arg} \\
    --task-index "$CURRENT_TASK_INDEX"{generation_options}
"""
        write_slurm_array_script(
            script_path,
            SlurmArrayScriptSpec(
                job_name=f"pht-meta-migrate-{chunk_id}",
                slurm_args=slurm_args,
                log_path=(
                    log_dir / f"migration_chunk{chunk_id}_%A_%a.log"
                ),
                task_indices=indices,
                body=body,
                prelude=SLURM_THREAD_PIN_BASH,
                comments=(
                    "# Recompile metadata-schema migration shard",
                    f"# Plan: {plan.report.plan_fingerprint}",
                ),
            ),
        )
        shard_scripts.append(script_path)

    finalizer_script = script_dir / "metadata_migration_finalizer.sh"
    finalizer_parts = [
        python_str,
        "-m",
        "phenotypic._cli._cli_recompile_metadata_migration_worker",
        "--task-manifest",
        manifest_arg,
        "--finalize",
    ]
    finalizer_parts.extend(
        [
            "--output-dir",
            output_arg,
            "--slurm-generation",
            shlex.quote(slurm_generation),
            "--attempt-id",
            shlex.quote(attempt_id),
        ]
    )
    finalizer_command = " ".join(finalizer_parts)
    _write_singleton_slurm_script(
        finalizer_script,
        job_name="pht-meta-migrate-finalize",
        slurm_args=slurm_args,
        log_path=log_dir / "migration_finalizer_%j.log",
        command=finalizer_command,
        comments=(
            "# Singleton metadata migration validation barrier",
            f"# Plan: {plan.report.plan_fingerprint}",
        ),
    )
    return RecompileMetadataSlurmScripts(
        shard_scripts=tuple(shard_scripts),
        finalizer_script=finalizer_script,
    )


def metadata_migration_task_status_path(
    manifest_path: Path, task_index: int
) -> Path:
    """Return the plan-scoped atomic target status path."""
    return Path(manifest_path).parent / "status" / f"target_{task_index}.json"


def metadata_migration_finalizer_status_path(manifest_path: Path) -> Path:
    """Return the plan-scoped singleton finalizer status path."""
    return Path(manifest_path).parent / _MIGRATION_FINALIZER_STATUS


def _blocked_result(report: MetadataMigrationReport) -> MetadataMigrationResult:
    """Adapt a blocked preflight report to the shared CLI error payload."""
    return MetadataMigrationResult(
        status="blocked",
        source=report.source,
        source_fingerprint=report.source_fingerprint,
        resulting_fingerprint=None,
        plan_fingerprint=report.plan_fingerprint,
        receipt_path=None,
        blocked_targets=tuple(
            target.path
            for target in report.targets
            if target.status == "blocked"
        ),
        conflicts=report.conflicts,
    )


def _write_singleton_slurm_script(
    path: Path,
    *,
    job_name: str,
    slurm_args: dict[str, Any],
    log_path: Path,
    command: str,
    comments: tuple[str, ...],
) -> Path:
    """Write a non-array SLURM script used as an ``afterok`` barrier."""
    directives = format_sbatch_directives(
        job_name=job_name,
        slurm_args=slurm_args,
        output_log=log_path,
        error_log=log_path,
    )
    comment_block = "\n".join(comments)
    content = f"""#!/bin/bash
{directives}

{comment_block}
set -e
set -u

{SLURM_PYTHONPATH_BOOTSTRAP_BASH}

{SLURM_THREAD_PIN_BASH}

echo "Job ID: ${{SLURM_JOB_ID:-unknown}}"
echo "Node: ${{SLURMD_NODENAME:-$(hostname)}}"
echo "Start Time: $(date)"

set +e
{command}
EXIT_CODE=$?
set -e

echo ""
echo "Exit Code: $EXIT_CODE"
echo "End Time: $(date)"
exit $EXIT_CODE
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)
    return path


__all__ = [
    "RecompileMetadataSlurmPlan",
    "RecompileMetadataSlurmScripts",
    "generate_metadata_migration_slurm_scripts",
    "metadata_migration_finalizer_status_path",
    "metadata_migration_task_status_path",
    "plan_metadata_schema_for_slurm_recompile",
]
