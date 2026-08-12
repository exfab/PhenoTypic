"""Worker CLI for recompile-specific SLURM array tasks."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import click

from ._cli_recompile_slurm_scripts import (
    TASK_FINALIZE,
    TASK_MEASUREMENTS,
    TASK_OVERLAY,
)
from phenotypic.schema import EXPERIMENT_METADATA, METADATA
from phenotypic.sdk_ import (
    DIR_MEASUREMENTS,
    DIR_RECOMPILE_SHARDS,
    DIR_RESULTS,
    JobMetadataKey,
    PARQUET_WRITE_OPTIONS,
    atomic_write_json,
    atomic_write_with_writer,
    load_image_from_hdf,
    master_measurements_csv_path,
    master_measurements_parquet_path,
    task_status_filename,
    task_status_path,
    shard_parquet_filename,
    progress_dir as progress_dir_helper,
    recompile_dir as recompile_dir_helper,
    recompile_status_dir,
)

logger = logging.getLogger(__name__)

_FINALIZER_STATUS_TIMEOUT_SECONDS = 600


@click.command("recompile-worker")
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    required=True,
)
@click.option(
    "--task-manifest",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option("--task-index", type=int, required=True)
def main(output_dir: Path, task_manifest: Path, task_index: int) -> None:
    """Run one recompile task from a JSON task manifest."""
    try:
        run_recompile_task(output_dir, task_manifest, task_index)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


def run_recompile_task(
    output_dir: Path, task_manifest: Path, task_index: int
) -> None:
    """Load and dispatch a single recompile task.

    Args:
        output_dir: Existing CLI output directory.
        task_manifest: JSON manifest written by
            :func:`generate_recompile_slurm_scripts`.
        task_index: Zero-based task index in the manifest.
    """
    output_dir = Path(output_dir)
    task = _load_task(task_manifest, task_index)
    task_type = str(task.get("task_type", ""))

    try:
        if task_type == TASK_MEASUREMENTS:
            _run_measurement_task(output_dir, task)
            _write_status(
                output_dir, task_index, task_type, {"status": "completed"}
            )
        elif task_type == TASK_OVERLAY:
            status = _run_overlay_task(output_dir, task)
            _write_status(output_dir, task_index, task_type, status)
        elif task_type == TASK_FINALIZE:
            _run_finalizer_task(output_dir, task)
            _write_status(
                output_dir, task_index, task_type, {"status": "completed"}
            )
        else:
            raise ValueError(f"Unknown recompile task type: {task_type!r}")
    except Exception as exc:
        _write_status(
            output_dir,
            task_index,
            task_type,
            {"status": "failed", "error": f"{type(exc).__name__}: {exc}"},
        )
        raise


def _load_task(task_manifest: Path, task_index: int) -> dict[str, Any]:
    """Load a task dictionary by index from the manifest."""
    manifest = json.loads(task_manifest.read_text(encoding="utf-8"))
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("Task manifest does not contain a tasks list")
    try:
        task = tasks[task_index]
    except IndexError as exc:
        raise ValueError(f"Task index out of range: {task_index}") from exc
    if not isinstance(task, dict):
        raise ValueError(f"Task {task_index} is not a dictionary")
    return task


def _write_status(
    output_dir: Path, task_index: int, task_type: str, fields: dict[str, Any]
) -> None:
    """Atomically write one recompile task status JSON."""
    status_path = task_status_path(output_dir, task_index)
    payload = {"task_type": task_type, **fields}
    atomic_write_json(status_path, payload, sort_keys=False)


def _run_measurement_task(output_dir: Path, task: dict[str, Any]) -> None:
    """Aggregate one measurement shard and write it under progress."""
    import polars as pl

    from ._cli_parquet_agg import aggregate_parquet_files

    files = [Path(path) for path in task.get("files", [])]
    path_to_dataset = {
        path: _dataset_name_from_measurement_path(output_dir, path)
        for path in files
    }
    shard_df = aggregate_parquet_files(
        file_paths=files,
        path_to_dataset=path_to_dataset,
        include_dataset_column=bool(task.get("include_dataset_column", True)),
        keep_filename=True,
    )
    if shard_df is None:
        raise RuntimeError("No valid measurements found for shard")

    if (
        str(METADATA.IMAGE_NAME) not in shard_df.columns
        and "filename" in shard_df.columns
    ):
        shard_df = shard_df.with_columns(
            pl.col("filename")
            .str.extract(r"([^/\\]+)\.[^.]+$", 1)
            .alias(str(METADATA.IMAGE_NAME))
        )
    if "filename" in shard_df.columns:
        shard_df = shard_df.drop("filename")
    shard_df = _sort_measurement_shard(shard_df)

    shard_id = int(task["shard_id"])
    shard_path = (
        recompile_dir_helper(progress_dir_helper(output_dir))
        / DIR_RECOMPILE_SHARDS
        / shard_parquet_filename(shard_id)
    )
    atomic_write_with_writer(
        shard_path,
        lambda p: shard_df.write_parquet(p, **PARQUET_WRITE_OPTIONS),
    )


def _sort_measurement_shard(shard_df: Any) -> Any:
    """Sort a shard by stable metadata columns when they are available."""
    sort_columns = [
        column
        for column in (
            str(EXPERIMENT_METADATA.DATASET),
            str(METADATA.IMAGE_NAME),
            "Metadata_Well",  # legacy sort key: no WELL schema member (≠ SourceWell); keep literal
            "Object_Label",
        )
        if column in shard_df.columns
    ]
    if not sort_columns:
        return shard_df
    return shard_df.sort(sort_columns)


def _dataset_name_from_measurement_path(output_dir: Path, path: Path) -> str:
    """Derive dataset name from ``results/<dataset>/measurements`` path."""
    try:
        relative = path.resolve().relative_to(output_dir.resolve())
    except ValueError:
        relative = path
    parts = relative.parts
    if (
        len(parts) >= 4
        and parts[0] == DIR_RESULTS
        and parts[2] == DIR_MEASUREMENTS
    ):
        return parts[1]
    if path.parent.name == DIR_MEASUREMENTS:
        return path.parent.parent.name
    raise ValueError(
        f"Cannot derive dataset name from measurement path: {path}"
    )


def _run_overlay_task(
    output_dir: Path, task: dict[str, Any]
) -> dict[str, Any]:
    """Regenerate one overlay, treating per-image failures as nonfatal."""
    try:
        from ._cli_output_manager import OutputManager

        dataset_name = str(task["dataset_name"])
        hdf_path = Path(str(task["hdf_path"]))
        image = load_image_from_hdf(hdf_path)

        output_manager = OutputManager.from_config(
            base_dir=output_dir,
            ext=".png",
            include_dataset_column=False,
            overlay_alpha=float(task.get("overlay_alpha", 0.3)),
            save_overlays=True,
        )
        output_manager.save_overlay(image, dataset_name, hdf_path.stem)
    except Exception as exc:
        logger.warning("Overlay regeneration failed", exc_info=True)
        return {
            "status": "completed",
            "overlay_failed": True,
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {"status": "completed", "overlay_failed": False}


def _run_finalizer_task(output_dir: Path, task: dict[str, Any]) -> None:
    """Finalize recompile outputs after all non-finalizer tasks finish."""
    progress_dir = progress_dir_helper(output_dir)
    status_dir = recompile_status_dir(progress_dir)
    expected = int(task.get("expected_non_finalizer_tasks", 0))

    statuses = _wait_for_non_finalizer_statuses(status_dir, expected)
    failed_measurements = [
        status
        for status in statuses
        if status.get("task_type") == TASK_MEASUREMENTS
        and status.get("status") == "failed"
    ]
    if failed_measurements:
        raise RuntimeError(
            f"{len(failed_measurements)} measurement shard task(s) failed"
        )

    merged_df = _write_master_outputs_from_shards(output_dir)
    _run_post_master_steps(output_dir, progress_dir, task, merged_df)


def _wait_for_non_finalizer_statuses(
    status_dir: Path,
    expected: int,
    timeout: int = _FINALIZER_STATUS_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    """Wait for all non-finalizer status files and return their payloads."""
    deadline = time.monotonic() + timeout
    while True:
        statuses = _read_expected_non_finalizer_statuses(status_dir, expected)
        if len(statuses) >= expected:
            return statuses
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for recompile statuses "
                f"({len(statuses)}/{expected})"
            )
        time.sleep(5)


def _read_expected_non_finalizer_statuses(
    status_dir: Path, expected: int
) -> list[dict[str, Any]]:
    """Read the expected non-finalizer task status files by task index."""
    statuses: list[dict[str, Any]] = []
    for task_index in range(expected):
        status_path = status_dir / task_status_filename(task_index)
        if not status_path.exists():
            continue
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to read status file %s", status_path)
            continue
        if status.get("task_type") == TASK_FINALIZE:
            continue
        statuses.append(status)
    return statuses


def _write_master_outputs_from_shards(output_dir: Path) -> Any | None:
    """Concatenate shard Parquets and write master CSV/Parquet outputs."""
    import polars as pl

    shard_dir = (
        recompile_dir_helper(progress_dir_helper(output_dir))
        / DIR_RECOMPILE_SHARDS
    )
    shard_files = sorted(shard_dir.glob("shard_*.parquet"))
    if not shard_files:
        return None

    frames = [pl.read_parquet(path) for path in shard_files]
    master_df = pl.concat(frames, how="diagonal_relaxed")

    # External metadata join is applied to the mirror in
    # ``finalize_post_master_outputs`` (via ``_run_post_master_steps``), not
    # to the master archive. The master stays a clean, op-free record of
    # what the per-image workers measured.

    try:
        atomic_write_with_writer(
            master_measurements_csv_path(output_dir), master_df.write_csv
        )
    except Exception:
        logger.error("Failed to save master CSV during recompile finalize")
        raise
    try:
        atomic_write_with_writer(
            master_measurements_parquet_path(output_dir),
            lambda p: master_df.write_parquet(p, **PARQUET_WRITE_OPTIONS),
        )
    except Exception:
        logger.warning(
            "Failed to save master Parquet during recompile finalize "
            "(CSV was saved)"
        )

    # Seeding ``measurements.{csv,parquet}``, persisting pipeline.json,
    # emitting configured analysis outputs, and per-feature splits all happen in
    # ``_run_post_master_steps`` via ``finalize_post_master_outputs`` so
    # the post-applied frame seeded into the GUI mirror matches the one
    # fed to the analysis chain.
    return master_df


def _run_post_master_steps(
    output_dir: Path,
    progress_dir: Path,
    task: dict[str, Any],
    merged_df: Any | None,
) -> None:
    """Run final outputs, rebuild the manifest, and generate the dashboard."""
    from ._cli_output_manager import (
        _load_pipeline_from_output_dir,
        finalize_post_master_outputs,
    )
    from ._cli_utils import load_job_metadata
    from ._dashboard import regenerate_dashboard_artifacts

    if merged_df is not None:
        # Single canonical post-master finalize: applies post to a copy of
        # the clean master, joins the external metadata CSV (when given)
        # onto the post-applied frame, seeds ``measurements.{csv,parquet}``,
        # persists ``pipeline.json``, emits configured analysis outputs, and
        # writes per-feature splits, matching the forward CLI path.
        pipeline = _load_pipeline_from_output_dir(output_dir)
        metadata_csv_str = task.get(JobMetadataKey.METADATA_CSV)
        metadata_csv = (
            Path(str(metadata_csv_str)) if metadata_csv_str else None
        )
        no_qc = bool(task.get(JobMetadataKey.NO_QC, False))
        finalize_post_master_outputs(
            output_dir,
            merged_df,
            pipeline,
            metadata_csv=metadata_csv,
            no_qc=no_qc,
        )

    job_meta = load_job_metadata(progress_dir)
    dataset_names = [str(name) for name in task.get("dataset_names", [])]
    datasets_totals = _dataset_totals(output_dir, dataset_names)
    regenerate_dashboard_artifacts(output_dir, job_meta, datasets_totals)


def _dataset_totals(
    output_dir: Path, dataset_names: list[str]
) -> dict[str, int]:
    """Count per-image measurement Parquets for manifest totals."""
    totals: dict[str, int] = {}
    for dataset_name in dataset_names:
        meas_dir = output_dir / DIR_RESULTS / dataset_name / DIR_MEASUREMENTS
        if not meas_dir.is_dir():
            totals[dataset_name] = 0
            continue
        totals[dataset_name] = len(
            [
                path
                for path in meas_dir.glob("*.parquet")
                if not path.name.startswith("_")
            ]
        )
    return totals


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
