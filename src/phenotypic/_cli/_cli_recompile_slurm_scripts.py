"""SLURM script generation for recompile task arrays."""

from __future__ import annotations

import json
import math
import shlex
from pathlib import Path
from typing import Any, Final

from ._cli_slurm_scripts import generate_slurm_directives
from ._cli_utils import SLURM_THREAD_PIN_BASH, get_python_command
from phenotypic.tools_ import (
    DATASET_AGGREGATED_PARQUET,
    DIR_HDF,
    DIR_LOGS,
    DIR_MEASUREMENTS,
    DIR_OVERLAYS,
    DIR_PROGRESS,
    DIR_RECOMPILE,
    DIR_RESULTS,
    DIR_SLURM_SCRIPTS,
    JobMetadataKey,
    RECOMPILE_TASK_MANIFEST_JSON,
)

TASK_MEASUREMENTS: Final[str] = "measurements"
TASK_OVERLAY: Final[str] = "overlay"
TASK_FINALIZE: Final[str] = "finalize"


def build_recompile_tasks(
    output_dir: Path,
    dataset_names: list[str],
    include_dataset_column: bool,
    overlay_alpha: float,
    shard_size: int,
) -> list[dict[str, Any]]:
    """Discover recompile work under an existing CLI output directory.

    Args:
        output_dir: Existing output directory containing ``results/``.
        dataset_names: Dataset names to scan under ``results/``.
        include_dataset_column: Whether measurement aggregation should add
            ``Metadata_Dataset`` when source files lack it.
        overlay_alpha: Overlay alpha to pass to overlay regeneration tasks.
        shard_size: Number of measurement source files per shard. Values
            below one are treated as one.

    Returns:
        JSON-serializable task dictionaries. When any non-finalizer work is
        found, the final task is exactly one finalizer dictionary.
    """
    output_dir = Path(output_dir)
    shard_size = max(1, int(shard_size))
    tasks: list[dict[str, Any]] = []
    shard_id = 0

    for dataset_name in dataset_names:
        source_files = _measurement_sources_for_dataset(
            output_dir, dataset_name
        )
        for source_shard in _chunk_paths(source_files, shard_size):
            tasks.append(
                {
                    "task_type": TASK_MEASUREMENTS,
                    "shard_id": shard_id,
                    "files": [str(path.absolute()) for path in source_shard],
                    "include_dataset_column": include_dataset_column,
                }
            )
            shard_id += 1

    for dataset_name in dataset_names:
        tasks.extend(
            _overlay_tasks_for_dataset(output_dir, dataset_name, overlay_alpha)
        )

    if tasks:
        tasks.append(
            {
                "task_type": TASK_FINALIZE,
                "dataset_names": list(dataset_names),
                "include_dataset_column": include_dataset_column,
                JobMetadataKey.METADATA_CSV: None,
                "expected_non_finalizer_tasks": len(tasks),
            }
        )

    return tasks


def generate_recompile_slurm_scripts(
    tasks: list[dict[str, Any]],
    output_dir: Path,
    slurm_args: dict[str, Any],
    array_limit: int,
) -> list[Path]:
    """Write a recompile task manifest and SLURM array scripts.

    Args:
        tasks: Recompile task dictionaries.
        output_dir: Existing output directory.
        slurm_args: SLURM directive arguments.
        array_limit: Maximum number of tasks per generated array script.

    Returns:
        Ordered list of generated array script paths.

    Raises:
        ValueError: If ``array_limit`` is not positive.
    """
    if array_limit <= 0:
        raise ValueError("array_limit must be positive")

    output_dir = Path(output_dir)
    recompile_dir = output_dir / DIR_PROGRESS / DIR_RECOMPILE
    recompile_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = recompile_dir / RECOMPILE_TASK_MANIFEST_JSON
    manifest_path.write_text(
        json.dumps({"tasks": tasks}, indent=2) + "\n",
        encoding="utf-8",
    )

    if not tasks:
        return []

    script_dir = output_dir / DIR_SLURM_SCRIPTS / "recompile"
    script_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / DIR_LOGS / "slurm" / "recompile"
    log_dir.mkdir(parents=True, exist_ok=True)

    scripts: list[Path] = []
    task_indices = list(range(len(tasks)))
    chunk_count = math.ceil(len(task_indices) / array_limit)
    for chunk_id in range(chunk_count):
        start = chunk_id * array_limit
        chunk = task_indices[start : start + array_limit]
        script_path = script_dir / f"recompile_array_chunk{chunk_id}.sh"
        script_path.write_text(
            _render_recompile_array_script(
                output_dir=output_dir,
                manifest_path=manifest_path,
                task_indices=chunk,
                chunk_id=chunk_id,
                slurm_args=slurm_args,
                log_dir=log_dir,
            ),
            encoding="utf-8",
        )
        script_path.chmod(0o755)
        scripts.append(script_path)

    return scripts


def _measurement_sources_for_dataset(
    output_dir: Path, dataset_name: str
) -> list[Path]:
    """Return deterministic measurement sources for one dataset."""
    meas_dir = output_dir / DIR_RESULTS / dataset_name / DIR_MEASUREMENTS
    if not meas_dir.is_dir():
        return []

    aggregated = meas_dir / DATASET_AGGREGATED_PARQUET
    if aggregated.exists():
        return [aggregated]

    return sorted(
        path
        for path in meas_dir.glob("*.parquet")
        if not path.name.startswith("_")
    )


def _overlay_tasks_for_dataset(
    output_dir: Path, dataset_name: str, overlay_alpha: float
) -> list[dict[str, Any]]:
    """Return one task for each missing overlay discoverable from HDF."""
    hdf_dir = output_dir / DIR_RESULTS / dataset_name / DIR_HDF
    if not hdf_dir.is_dir():
        return []

    overlay_dir = output_dir / DIR_RESULTS / dataset_name / DIR_OVERLAYS
    tasks: list[dict[str, Any]] = []
    for hdf_path in sorted(hdf_dir.glob("*.h5")):
        overlay_path = overlay_dir / f"{hdf_path.stem}.png"
        if overlay_path.exists():
            continue
        tasks.append(
            {
                "task_type": TASK_OVERLAY,
                "dataset_name": dataset_name,
                "hdf_path": str(hdf_path.absolute()),
                "overlay_alpha": overlay_alpha,
            }
        )
    return tasks


def _chunk_paths(paths: list[Path], shard_size: int) -> list[list[Path]]:
    """Split paths into deterministic non-empty shards."""
    return [
        paths[i : i + shard_size] for i in range(0, len(paths), shard_size)
    ]


def _render_recompile_array_script(
    *,
    output_dir: Path,
    manifest_path: Path,
    task_indices: list[int],
    chunk_id: int,
    slurm_args: dict[str, Any],
    log_dir: Path,
) -> str:
    """Render a single bash array script for recompile task indices."""
    log_path = log_dir / f"recompile_chunk{chunk_id}_%A_%a.log"
    directives = generate_slurm_directives(
        job_name=f"pht-recompile-{chunk_id}",
        slurm_args=slurm_args,
        output_log=log_path,
        error_log=log_path,
    )
    python_cmd, _ = get_python_command(for_slurm=True)
    python_str = " ".join(shlex.quote(part) for part in python_cmd)
    q_output_dir = shlex.quote(str(output_dir.absolute()))
    q_manifest = shlex.quote(str(manifest_path.absolute()))
    index_list = "\n".join(f"    {index}" for index in task_indices)

    return f"""#!/bin/bash
{directives}
#SBATCH --array=0-{len(task_indices) - 1}

# Auto-generated by PhenoTypic CLI recompile SLURM mode
# Chunk: {chunk_id}

set -e
set -u

{SLURM_THREAD_PIN_BASH}

TASK_INDICES=(
{index_list}
)

if [ "${{SLURM_ARRAY_TASK_ID:-}}" = "" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID not set"
    exit 1
fi

if [ "$SLURM_ARRAY_TASK_ID" -ge "${{#TASK_INDICES[@]}}" ]; then
    echo "ERROR: Array task ID $SLURM_ARRAY_TASK_ID exceeds task list size ${{#TASK_INDICES[@]}}"
    exit 1
fi

CURRENT_TASK_INDEX="${{TASK_INDICES[$SLURM_ARRAY_TASK_ID]}}"

echo "Recompile task index: $CURRENT_TASK_INDEX"
echo "Job ID: ${{SLURM_JOB_ID:-unknown}}"
echo "Array Task ID: ${{SLURM_ARRAY_TASK_ID:-unknown}}"
echo "Node: ${{SLURMD_NODENAME:-$(hostname)}}"
echo "Start Time: $(date)"

{python_str} -m phenotypic._cli._cli_recompile_worker \\
    --output-dir {q_output_dir} \\
    --task-manifest {q_manifest} \\
    --task-index "$CURRENT_TASK_INDEX"

EXIT_CODE=$?

echo "Exit Code: $EXIT_CODE"
echo "End Time: $(date)"
exit $EXIT_CODE
"""
