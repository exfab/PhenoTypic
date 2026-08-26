"""SLURM script generation for recompile task arrays."""

from __future__ import annotations

import json
import math
import shlex
from pathlib import Path
from typing import Any, Final

from ._cli_completion import authorized_measurement_sources
from ._measurement_sources import discover_recompile_measurement_sources
from ._cli_utils import SLURM_THREAD_PIN_BASH, get_python_command
from phenotypic.sdk_ import (
    DIR_RESULTS,
    DIR_ZARR,
    MEASUREMENT_TABLE_RELATIVE_PATH,
    STORE_SUFFIX,
    store_stem,
    dataset_overlays_dir,
    image_completion_marker_path,
    JobMetadataKey,
    RECOMPILE_TASK_MANIFEST_JSON,
    logs_dir,
    progress_dir,
    recompile_dir,
    slurm_scripts_dir,
)
from phenotypic.sdk_.slurm import (
    SlurmArrayScriptSpec,
    write_slurm_array_script,
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
    attempt_id: str | None = None,
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
        attempt_id: Optional scheduler-attempt namespace and lifecycle
            generation for task metadata.

    Returns:
        JSON-serializable task dictionaries. When any non-finalizer work is
        found, the final task is exactly one finalizer dictionary.
    """
    output_dir = Path(output_dir)
    shard_size = max(1, int(shard_size))

    authorized_sources = authorized_measurement_sources(output_dir)
    tasks: list[dict[str, Any]] = []
    measurement_sources: list[Path] = []
    shard_id = 0
    overlay_tasks_by_dataset = {
        dataset_name: _overlay_tasks_for_dataset(
            output_dir, dataset_name, overlay_alpha
        )
        for dataset_name in dataset_names
    }

    for dataset_name in dataset_names:
        if authorized_sources is None:
            source_files = [
                source.path
                for source in discover_recompile_measurement_sources(
                    output_dir, [dataset_name]
                )
            ]
        else:
            authorized_for_dataset = {
                path
                for path, dataset in authorized_sources.items()
                if dataset == dataset_name
            }
            recoverable_overlay_tables = {
                Path(str(task["store_path"])) / MEASUREMENT_TABLE_RELATIVE_PATH
                for task in overlay_tasks_by_dataset[dataset_name]
                if task.get("restore_marker_authority")
            }
            source_files = sorted(
                authorized_for_dataset | recoverable_overlay_tables
            )
        measurement_sources.extend(source_files)
        for source_shard in _chunk_paths(source_files, shard_size):
            task = {
                "task_type": TASK_MEASUREMENTS,
                "shard_id": shard_id,
                "files": [str(path.absolute()) for path in source_shard],
                "include_dataset_column": include_dataset_column,
            }
            if attempt_id is not None:
                task["slurm_generation"] = attempt_id
            tasks.append(task)
            shard_id += 1

    for dataset_name in dataset_names:
        overlay_tasks = overlay_tasks_by_dataset[dataset_name]
        if attempt_id is not None:
            for task in overlay_tasks:
                task["slurm_generation"] = attempt_id
        tasks.extend(overlay_tasks)

    if tasks:
        from ._cli_output_manager import _consistent_embedded_join_keys

        metadata_join_keys = (
            _consistent_embedded_join_keys(measurement_sources)
            if measurement_sources
            else None
        )
        finalizer_task = {
            "task_type": TASK_FINALIZE,
            "dataset_names": list(dataset_names),
            "include_dataset_column": include_dataset_column,
            JobMetadataKey.METADATA_CSV: None,
            "metadata_join_keys": list(metadata_join_keys or ()),
            # Re-read provenance after measurement shards finish: a recompile
            # with new metadata rewrites these tables after task submission.
            "measurement_sources": [
                str(path.absolute()) for path in measurement_sources
            ],
            "expected_non_finalizer_tasks": len(tasks),
        }
        if attempt_id is not None:
            finalizer_task["slurm_generation"] = attempt_id
        tasks.append(finalizer_task)

    return tasks


def generate_recompile_slurm_scripts(
    tasks: list[dict[str, Any]],
    output_dir: Path,
    slurm_args: dict[str, Any],
    array_limit: int,
    attempt_id: str | None = None,
) -> list[Path]:
    """Write a recompile task manifest and SLURM array scripts.

    Args:
        tasks: Recompile task dictionaries.
        output_dir: Existing output directory.
        slurm_args: SLURM directive arguments.
        array_limit: Maximum number of tasks per generated array script.
        attempt_id: Optional scheduler-attempt namespace for generated state.

    Returns:
        Ordered list of generated array script paths.

    Raises:
        ValueError: If ``array_limit`` is not positive.
    """
    if array_limit <= 0:
        raise ValueError("array_limit must be positive")
    if attempt_id is None or not attempt_id:
        raise ValueError("attempt_id is required for SLURM recompile scripts")

    output_dir = Path(output_dir)
    rc_dir = recompile_attempt_dir(output_dir, attempt_id)
    rc_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = rc_dir / RECOMPILE_TASK_MANIFEST_JSON
    manifest_tasks = [dict(task) for task in tasks]
    if attempt_id is not None:
        for task in manifest_tasks:
            task_generation = task.get("slurm_generation")
            if task_generation not in {None, attempt_id}:
                raise ValueError(
                    "Recompile task generation conflicts with attempt id"
                )
            task["slurm_generation"] = attempt_id
    manifest_path.write_text(
        json.dumps({"tasks": manifest_tasks}, indent=2) + "\n",
        encoding="utf-8",
    )

    if not tasks:
        return []

    terminal_status_path = recompile_task_status_path(
        manifest_path, len(tasks) - 1
    )

    script_dir = slurm_scripts_dir(output_dir) / "recompile"
    if attempt_id is not None:
        script_dir = script_dir / attempt_id
    script_dir.mkdir(parents=True, exist_ok=True)
    log_dir = logs_dir(output_dir) / "slurm" / "recompile"
    log_dir.mkdir(parents=True, exist_ok=True)

    scripts: list[Path] = []
    task_indices = list(range(len(tasks)))
    chunk_count = math.ceil(len(task_indices) / array_limit)
    for chunk_id in range(chunk_count):
        start = chunk_id * array_limit
        chunk = task_indices[start : start + array_limit]
        script_path = script_dir / f"recompile_array_chunk{chunk_id}.sh"
        write_slurm_array_script(
            script_path,
            _recompile_array_script_spec(
                output_dir=output_dir,
                manifest_path=manifest_path,
                task_indices=chunk,
                chunk_id=chunk_id,
                slurm_args=slurm_args,
                log_dir=log_dir,
                slurm_generation=attempt_id,
                attempt_id=attempt_id,
                terminal_status_path=terminal_status_path,
            ),
        )
        scripts.append(script_path)

    return scripts


def recompile_attempt_dir(output_dir: Path, attempt_id: str | None) -> Path:
    """Return isolated recompile state for one scheduler attempt."""
    base = recompile_dir(progress_dir(Path(output_dir)))
    return base if attempt_id is None else base / "attempts" / attempt_id


def recompile_task_status_path(manifest_path: Path, task_index: int) -> Path:
    """Return an attempt-scoped ordinary recompile task status path."""
    return Path(manifest_path).parent / "status" / f"task_{task_index}.json"


def _overlay_tasks_for_dataset(
    output_dir: Path, dataset_name: str, overlay_alpha: float
) -> list[dict[str, Any]]:
    """Return one task for each overlay missing from a dataset's stores.

    The glob is non-recursive and matches directories: a store is itself a
    directory of files, so an ``is_file()`` filter would find nothing and a
    recursive walk would descend into every chunk.
    """
    zarr_dir = output_dir / DIR_RESULTS / dataset_name / DIR_ZARR
    if not zarr_dir.is_dir():
        return []

    overlay_dir = dataset_overlays_dir(output_dir, dataset_name)
    tasks: list[dict[str, Any]] = []
    for store_path in sorted(zarr_dir.glob(f"*{STORE_SUFFIX}")):
        if not store_path.is_dir() or store_path.name.startswith("."):
            continue
        # ``store_stem``, never ``Path.stem``: `.ome.zarr` is a double
        # suffix, so `.stem` would test for `<stem>.ome.png`, never find
        # it, and queue a redundant overlay task for every image forever.
        stem = store_stem(store_path)
        overlay_path = overlay_dir / f"{stem}.png"
        if overlay_path.exists():
            continue
        task: dict[str, Any] = {
            "task_type": TASK_OVERLAY,
            "dataset_name": dataset_name,
            "store_path": str(store_path.absolute()),
            "overlay_alpha": overlay_alpha,
        }
        if _marker_binds_overlay_and_table(
            output_dir,
            dataset_name,
            stem,
            overlay_path,
            store_path / MEASUREMENT_TABLE_RELATIVE_PATH,
        ):
            task["restore_marker_authority"] = True
        tasks.append(task)
    return tasks


def _marker_binds_overlay_and_table(
    output_dir: Path,
    dataset_name: str,
    stem: str,
    overlay_path: Path,
    table_path: Path,
) -> bool:
    """Return whether an existing marker binds the missing overlay and table."""
    marker_path = image_completion_marker_path(output_dir, dataset_name, stem)
    if not marker_path.is_file() or not table_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        artifacts = marker["artifacts"]
        if not isinstance(artifacts, dict):
            return False
        expected = {
            "overlay": overlay_path.resolve(),
            "measurements": table_path.resolve(),
        }
        output_root = output_dir.resolve()
        for name, expected_path in expected.items():
            descriptor = artifacts.get(name)
            if not isinstance(descriptor, dict):
                return False
            relative = descriptor.get("path")
            if not isinstance(relative, str):
                return False
            if (output_root / relative).resolve() != expected_path:
                return False
    except (KeyError, OSError, TypeError, ValueError):
        return False
    return True


def _chunk_paths(paths: list[Path], shard_size: int) -> list[list[Path]]:
    """Split paths into deterministic non-empty shards."""
    return [
        paths[i : i + shard_size] for i in range(0, len(paths), shard_size)
    ]


def _recompile_array_script_spec(
    *,
    output_dir: Path,
    manifest_path: Path,
    task_indices: list[int],
    chunk_id: int,
    slurm_args: dict[str, Any],
    log_dir: Path,
    slurm_generation: str | None = None,
    attempt_id: str | None = None,
    terminal_status_path: Path | None = None,
) -> SlurmArrayScriptSpec:
    """Build a shared script spec for recompile task indices."""
    log_path = log_dir / f"recompile_chunk{chunk_id}_%A_%a.log"
    python_cmd, _ = get_python_command(for_slurm=True)
    python_str = " ".join(shlex.quote(part) for part in python_cmd)
    q_output_dir = shlex.quote(str(output_dir.absolute()))
    q_manifest = shlex.quote(str(manifest_path.absolute()))
    generation_option = (
        " \\\n    --slurm-generation " + shlex.quote(slurm_generation)
        if slurm_generation is not None
        else ""
    )
    attempt_option = (
        " \\\n    --attempt-id " + shlex.quote(attempt_id)
        if attempt_id is not None
        else ""
    )
    terminal_status_option = (
        " \\\n    --terminal-status-path "
        + shlex.quote(str(terminal_status_path.absolute()))
        if terminal_status_path is not None
        else ""
    )

    body = f"""\
echo "Recompile task index: $CURRENT_TASK_INDEX"

{python_str} -m phenotypic._cli._cli_recompile_worker \\
    --output-dir {q_output_dir} \\
    --task-manifest {q_manifest} \\
    --task-index "$CURRENT_TASK_INDEX"{generation_option}{attempt_option}{terminal_status_option}
"""
    return SlurmArrayScriptSpec(
        job_name=f"pht-recompile-{chunk_id}",
        slurm_args=slurm_args,
        log_path=log_path,
        task_indices=task_indices,
        body=body,
        prelude=SLURM_THREAD_PIN_BASH,
        comments=[
            "# Auto-generated by PhenoTypic CLI recompile SLURM mode",
            f"# Chunk: {chunk_id}",
        ],
        bounds_error_message=(
            "ERROR: Array task ID $SLURM_ARRAY_TASK_ID exceeds task list size "
            "${#TASK_INDICES[@]}"
        ),
    )
