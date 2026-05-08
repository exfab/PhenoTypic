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
    from ._cli_output_manager import _atomic_write

    status_path = (
        output_dir
        / "progress"
        / "recompile"
        / "status"
        / f"task_{task_index}.json"
    )
    payload = {"task_type": task_type, **fields}

    def _writer(path: str) -> None:
        Path(path).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )

    _atomic_write(status_path, _writer)


def _run_measurement_task(output_dir: Path, task: dict[str, Any]) -> None:
    """Aggregate one measurement shard and write it under progress."""
    import polars as pl

    from ._cli_duckdb_agg import duckdb_aggregate
    from ._cli_output_manager import _atomic_write

    files = [Path(path) for path in task.get("files", [])]
    path_to_dataset = {
        path: _dataset_name_from_measurement_path(output_dir, path)
        for path in files
    }
    shard_df = duckdb_aggregate(
        file_paths=files,
        path_to_dataset=path_to_dataset,
        include_dataset_column=bool(task.get("include_dataset_column", True)),
        keep_filename=True,
    )
    if shard_df is None:
        raise RuntimeError("No valid measurements found for shard")

    if (
        "Metadata_ImageFile" not in shard_df.columns
        and "filename" in shard_df.columns
    ):
        shard_df = shard_df.with_columns(
            pl.col("filename")
            .str.extract(r"([^/\\]+)\.[^.]+$", 1)
            .alias("Metadata_ImageFile")
        )
    if "filename" in shard_df.columns:
        shard_df = shard_df.drop("filename")
    shard_df = _sort_measurement_shard(shard_df)

    shard_id = int(task["shard_id"])
    shard_path = (
        output_dir
        / "progress"
        / "recompile"
        / "measurement_shards"
        / f"shard_{shard_id}.parquet"
    )
    _atomic_write(
        shard_path,
        lambda p: shard_df.write_parquet(
            p, compression="zstd", compression_level=3
        ),
    )


def _sort_measurement_shard(shard_df: Any) -> Any:
    """Sort a shard by stable metadata columns when they are available."""
    sort_columns = [
        column
        for column in (
            "Metadata_Dataset",
            "Metadata_ImageFile",
            "Metadata_Well",
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
        and parts[0] == "results"
        and parts[2] == "measurements"
    ):
        return parts[1]
    if path.parent.name == "measurements":
        return path.parent.parent.name
    raise ValueError(
        f"Cannot derive dataset name from measurement path: {path}"
    )


def _run_overlay_task(
    output_dir: Path, task: dict[str, Any]
) -> dict[str, Any]:
    """Regenerate one overlay, treating per-image failures as nonfatal."""
    try:
        import h5py  # type: ignore[import-untyped]

        from phenotypic import GridImage, Image

        from ._cli_output_manager import OutputManager

        dataset_name = str(task["dataset_name"])
        hdf_path = Path(str(task["hdf_path"]))
        with h5py.File(hdf_path, "r") as fh:
            cls_attr = fh.attrs.get("phenotypic_class", "Image")
        if isinstance(cls_attr, bytes):
            cls_attr = cls_attr.decode("utf-8", errors="replace")
        image_cls = GridImage if cls_attr == "GridImage" else Image
        image = image_cls.load_hdf5(hdf_path)

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
    progress_dir = output_dir / "progress"
    recompile_dir = progress_dir / "recompile"
    status_dir = recompile_dir / "status"
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

    merged_df = _write_master_outputs_from_shards(output_dir, task)
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
        status_path = status_dir / f"task_{task_index}.json"
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


def _write_master_outputs_from_shards(
    output_dir: Path, task: dict[str, Any]
) -> Any | None:
    """Concatenate shard Parquets and write master CSV/Parquet outputs."""
    import polars as pl

    from ._cli_output_manager import _atomic_write, join_metadata

    shard_dir = output_dir / "progress" / "recompile" / "measurement_shards"
    shard_files = sorted(shard_dir.glob("shard_*.parquet"))
    if not shard_files:
        return None

    frames = [pl.read_parquet(path) for path in shard_files]
    master_df = pl.concat(frames, how="diagonal_relaxed")

    metadata_csv_str = task.get("metadata_csv")
    if metadata_csv_str:
        try:
            master_df = join_metadata(master_df, Path(str(metadata_csv_str)))
        except Exception as exc:
            logger.warning(
                "Failed to join metadata CSV during recompile finalization: %s",
                exc,
            )

    try:
        _atomic_write(output_dir / "master_measurements.csv", master_df.write_csv)
    except Exception:
        logger.error("Failed to save master CSV during recompile finalize")
        raise
    try:
        _atomic_write(
            output_dir / "master_measurements.parquet",
            lambda p: master_df.write_parquet(
                p, compression="zstd", compression_level=3
            ),
        )
    except Exception:
        logger.warning(
            "Failed to save master Parquet during recompile finalize "
            "(CSV was saved)"
        )

    # Seeding ``measurements.{csv,parquet}``, persisting pipeline.json,
    # emitting analysis, and per-feature splits all happen in
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
    """Run split, analysis plugins, manifest rebuild, and dashboard generation."""
    from ._cli_chunk_writer import _run_analysis_plugins
    from ._cli_output_manager import (
        _load_pipeline_from_output_dir,
        finalize_post_master_outputs,
    )
    from ._cli_utils import load_job_metadata
    from ._dashboard._generator import generate_dashboard
    from ._dashboard._manifest_builder import build_manifest

    plugin_df: Any | None = merged_df
    if merged_df is not None:
        # Single canonical post-master finalize: applies post to a copy of
        # the clean master, seeds ``measurements.{csv,parquet}`` with the
        # post-applied frame, persists ``pipeline.json``, emits analysis,
        # and writes per-feature splits — same path the forward CLI takes.
        # Reuse the returned post-applied frame for analysis-plugin
        # dispatch so plugins see the same data the analysis chain and
        # the GUI viewer see.
        pipeline = _load_pipeline_from_output_dir(output_dir)
        plugin_df = finalize_post_master_outputs(output_dir, merged_df, pipeline)

    try:
        _run_analysis_plugins(output_dir, progress_dir, plugin_df)
    except Exception:
        logger.warning(
            "Analysis plugin dispatch failed during recompile", exc_info=True
        )

    job_meta = load_job_metadata(progress_dir)
    dataset_names = [str(name) for name in task.get("dataset_names", [])]
    datasets_totals = _dataset_totals(output_dir, dataset_names)
    build_manifest(
        output_dir=output_dir,
        progress_dir=progress_dir,
        datasets=datasets_totals,
        execution_mode=job_meta.get("execution_mode", "local")
        if job_meta
        else "local",
        start_time=job_meta.get("start_time", "") if job_meta else "",
        slurm_job_ids=job_meta.get("chunk_job_ids") if job_meta else None,
        chunk_scripts=job_meta.get("chunk_scripts") if job_meta else None,
        input_path=job_meta.get("input_path") if job_meta else None,
    )
    generate_dashboard(
        output_dir,
        execution_mode=job_meta.get("execution_mode", "local")
        if job_meta
        else "local",
    )


def _dataset_totals(
    output_dir: Path, dataset_names: list[str]
) -> dict[str, int]:
    """Count per-image measurement Parquets for manifest totals."""
    totals: dict[str, int] = {}
    for dataset_name in dataset_names:
        meas_dir = output_dir / "results" / dataset_name / "measurements"
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
