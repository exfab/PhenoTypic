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
    refresh_overlay_marker_authority,
    repair_overlay_marker_authority,
    recompile_task_status_path,
    recompile_attempt_dir,
)
from ._cli_slurm_lifecycle import (
    SlurmGenerationInactiveError,
    assert_generation_active,
    generation_publication_guard,
)
from phenotypic.schema import EXPERIMENT, IMAGE
from phenotypic.sdk_ import (
    DIR_MEASUREMENTS,
    DIR_RECOMPILE_SHARDS,
    DIR_RESULTS,
    JobMetadataKey,
    PARQUET_WRITE_OPTIONS,
    RECOMPILE_TASK_MANIFEST_JSON,
    atomic_write_json,
    atomic_write_with_writer,
    load_image_from_store,
    store_stem,
    master_measurements_csv_path,
    master_measurements_parquet_path,
    task_status_filename,
    shard_parquet_filename,
    progress_dir as progress_dir_helper,
    recompile_dir as recompile_dir_helper,
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
    type=click.Path(path_type=Path),
    required=True,
)
@click.option("--task-index", type=int, required=True)
@click.option("--slurm-generation", required=True)
@click.option("--attempt-id", required=True)
@click.option("--terminal-status-path", type=click.Path(path_type=Path))
def main(
    output_dir: Path,
    task_manifest: Path,
    task_index: int,
    slurm_generation: str,
    attempt_id: str,
    terminal_status_path: Path | None,
) -> None:
    """Run one recompile task from a JSON task manifest."""
    try:
        run_recompile_task(
            output_dir,
            task_manifest,
            task_index,
            slurm_generation=slurm_generation,
            attempt_id=attempt_id,
            terminal_status_path=terminal_status_path,
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


def run_recompile_task(
    output_dir: Path,
    task_manifest: Path,
    task_index: int,
    *,
    slurm_generation: str,
    attempt_id: str,
    terminal_status_path: Path | None = None,
) -> None:
    """Load and dispatch a single recompile task.

    Args:
        output_dir: Existing CLI output directory.
        task_manifest: JSON manifest written by
            :func:`generate_recompile_slurm_scripts`.
        task_index: Zero-based task index in the manifest.
        slurm_generation: Lifecycle generation supplied independently by the
            scheduler script.
        attempt_id: Attempt namespace supplied independently by the scheduler
            script.
        terminal_status_path: Attempt-scoped finalizer status supplied by the
            scheduler script so a waiter can observe manifest bootstrap
            failures before finalizer task metadata is available.
    """
    output_dir = Path(output_dir).resolve()
    _assert_worker_generation(output_dir, slurm_generation, attempt_id)
    expected_manifest = (
        recompile_attempt_dir(output_dir, attempt_id)
        / RECOMPILE_TASK_MANIFEST_JSON
    ).resolve()
    if task_manifest.resolve() != expected_manifest:
        raise ValueError("Recompile manifest is outside its attempt namespace")
    task: dict[str, Any] | None = None
    task_type = "unknown"
    try:
        task = _load_task(task_manifest, task_index)
        task_type = str(task.get("task_type", ""))
        task_generation = task.get("slurm_generation")
        if task_generation != slurm_generation:
            raise ValueError("Recompile task generation does not match script")
        if task_type == TASK_MEASUREMENTS:
            _run_measurement_task(
                output_dir,
                task_manifest,
                task,
                slurm_generation=slurm_generation,
            )
            _write_status(
                task_manifest,
                task_index,
                task_type,
                {"status": "completed"},
                output_dir=output_dir,
                slurm_generation=slurm_generation,
            )
        elif task_type == TASK_OVERLAY:
            status = _run_overlay_task(
                output_dir, task, slurm_generation=slurm_generation
            )
            _write_status(
                task_manifest,
                task_index,
                task_type,
                status,
                output_dir=output_dir,
                slurm_generation=slurm_generation,
            )
        elif task_type == TASK_FINALIZE:
            _run_finalizer_task(
                output_dir,
                task_manifest,
                task,
                slurm_generation=slurm_generation,
            )
            _write_status(
                task_manifest,
                task_index,
                task_type,
                {"status": "completed"},
                output_dir=output_dir,
                slurm_generation=slurm_generation,
            )
            _deactivate_generation_value(output_dir, slurm_generation)
        else:
            raise ValueError(f"Unknown recompile task type: {task_type!r}")
    except Exception as exc:
        try:
            failure = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            _write_status(
                task_manifest,
                task_index,
                task_type,
                failure,
                output_dir=output_dir,
                slurm_generation=slurm_generation,
            )
            if terminal_status_path is not None:
                current_status_path = recompile_task_status_path(
                    task_manifest, task_index
                )
                if terminal_status_path != current_status_path:
                    with generation_publication_guard(
                        output_dir, slurm_generation
                    ):
                        atomic_write_json(
                            terminal_status_path,
                            {
                                "task_type": TASK_FINALIZE,
                                **failure,
                                "manifest_unreadable": task is None,
                                "worker_terminal_failure": True,
                            },
                            sort_keys=False,
                        )
        finally:
            # A successfully classified non-finalizer worker (measurement or
            # overlay) can fail for a routine, expected reason and must not
            # poison every concurrently-running sibling task. The finalizer
            # decides whether those recorded failures block publication.
            #
            # If bootstrap fails before a valid non-finalizer type can be
            # loaded, however, the worker has already published the terminal
            # finalizer status above and must release the attempt fence. This
            # preserves fail-closed teardown for missing, corrupt, or
            # semantically invalid manifests.
            if (
                not isinstance(exc, SlurmGenerationInactiveError)
                and task_type not in {TASK_MEASUREMENTS, TASK_OVERLAY}
            ):
                _deactivate_generation_value(output_dir, slurm_generation)
        raise


def _assert_worker_generation(
    output_dir: Path, slurm_generation: str, attempt_id: str
) -> None:
    """Validate worker ownership arguments.

    **The two arguments are one value, so there is no equality check.** The
    docstrings on the call path say they are "supplied independently by the
    scheduler script", and at the level of this function that is true -- they
    arrive as two CLI options. It is false of every supplier:

    There are **two** suppliers, and both are in
    ``_cli_recompile_slurm_scripts``. ``_write_recompile_chunk_scripts``
    passes ``slurm_generation=attempt_id, attempt_id=attempt_id`` into the
    script-body builder, which renders ``--slurm-generation`` and
    ``--attempt-id`` from those two parameters; and the manifest writer sets
    ``task["slurm_generation"] = attempt_id`` at four sites. Every path puts
    one variable into both options.

    **``phenotypicCLI.py`` is not a supplier**, and an earlier version of this
    docstring said it was. The line it cited calls
    ``_wait_for_recompile_finalizer_status(..., slurm_generation=attempt_id)``,
    which takes no ``attempt_id`` parameter and never reaches this function --
    whose only caller is ``run_recompile_task``, invoked only by the generated
    sbatch script. Cited by symbol here, not by line: the bad citation had
    also drifted by nine lines, so a reader following it landed on neither the
    right function nor the right place (gate IMPL-F7).

    So ``if slurm_generation != attempt_id: raise`` was **unreachable** --
    a value compared with itself, routed through two options. It read as a
    safety check and enforced nothing, which is worse than no check: a
    reader could reasonably conclude the two were independently verified.

    **Do not reinstate it.** If a future change makes the two genuinely
    distinct, the fix is to give them separate meanings and fence each on its
    own, not to re-add an equality assertion that would then be enforcing an
    invariant nobody stated. Audit §11.1; confirmed against the tree in P2
    Task 4 before removal.

    What remains is real: both values must be present, and the generation
    must still be the active lifecycle fence.

    **Still unpinned, and named so it is not mistaken for covered:** nothing
    tests that every supplier passes one value into both options. The removal
    was correct for today's two suppliers, but a third passing two distinct
    values would be accepted silently. That is the other half of gate
    IMPL-F7, and it is a test this phase did not write.
    """
    if not slurm_generation or not attempt_id:
        raise ValueError("SLURM generation and attempt id are required")
    assert_generation_active(output_dir, slurm_generation)


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
    task_manifest: Path,
    task_index: int,
    task_type: str,
    fields: dict[str, Any],
    *,
    output_dir: Path,
    slurm_generation: str,
) -> None:
    """Atomically write one recompile task status JSON."""
    status_path = recompile_task_status_path(task_manifest, task_index)
    payload = {"task_type": task_type, **fields}
    with generation_publication_guard(output_dir, slurm_generation):
        atomic_write_json(status_path, payload, sort_keys=False)


def _run_measurement_task(
    output_dir: Path,
    task_manifest: Path,
    task: dict[str, Any],
    *,
    slurm_generation: str,
) -> None:
    """Aggregate one measurement shard and write it under progress."""

    from ._cli_parquet_agg import aggregate_parquet_files

    files = [Path(path) for path in task.get("files", [])]
    metadata_csv_raw = task.get(JobMetadataKey.METADATA_CSV)
    metadata_csv = Path(str(metadata_csv_raw)) if metadata_csv_raw else None
    from ._cli_recompile_tables import recompile_embedded_measurement_table

    raw_repairs = task.get("overlay_repairs", [])
    if not isinstance(raw_repairs, list):
        raise ValueError("Measurement task overlay_repairs must be a list")
    repairs_by_table: dict[Path, dict[str, Any]] = {}
    for raw_repair in raw_repairs:
        if not isinstance(raw_repair, dict):
            raise ValueError("Measurement task has invalid overlay repair")
        repair_table = Path(str(raw_repair["table_path"]))
        repairs_by_table[repair_table] = raw_repair

    for table_path in files:
        if tuple(table_path.parts[-3:]) == (
            "tables",
            "measurements",
            "table.parquet",
        ):
            repair = repairs_by_table.get(table_path)
            if repair is not None:
                _repair_measurement_overlay(
                    output_dir,
                    repair,
                    slurm_generation=slurm_generation,
                )
            dataset = _dataset_name_from_measurement_path(
                output_dir, table_path
            )
            recompile_embedded_measurement_table(
                output_dir,
                table_path,
                dataset,
                metadata_csv,
                commit_guard=lambda: generation_publication_guard(
                    output_dir, slurm_generation
                ),
                lifecycle_epoch=slurm_generation,
            )

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

    from ._measurement_sources import (
        add_metadata_image_name_from_filename,
    )

    shard_df = add_metadata_image_name_from_filename(shard_df)
    shard_df = _sort_measurement_shard(shard_df)

    shard_id = int(task["shard_id"])
    shard_path = (
        task_manifest.parent
        / DIR_RECOMPILE_SHARDS
        / shard_parquet_filename(shard_id)
    )
    with generation_publication_guard(output_dir, slurm_generation):
        atomic_write_with_writer(
            shard_path,
            lambda p: shard_df.write_parquet(p, **PARQUET_WRITE_OPTIONS),
        )


def _repair_measurement_overlay(
    output_dir: Path,
    repair: dict[str, Any],
    *,
    slurm_generation: str,
) -> None:
    """Repair a marker-bound overlay before the same task rewrites its table."""
    from ._cli_output_manager import OutputManager

    dataset_name = str(repair["dataset_name"])
    store_path = Path(str(repair["store_path"]))
    expected_table = store_path / "tables" / "measurements" / "table.parquet"
    if Path(str(repair["table_path"])) != expected_table:
        raise ValueError("Overlay repair table does not belong to its store")
    image = load_image_from_store(store_path)
    stem = store_stem(store_path)
    output_manager = OutputManager.from_config(
        base_dir=output_dir,
        ext=".png",
        include_dataset_column=False,
        overlay_alpha=float(repair.get("overlay_alpha", 0.3)),
        save_overlays=True,
    )

    def _render(render_guard: Any) -> object:
        return output_manager.save_overlay(
            image,
            dataset_name,
            stem,
            commit_guard=render_guard,
        )

    if not repair_overlay_marker_authority(
        output_dir,
        dataset_name,
        stem,
        store_path,
        _render,
        lifecycle_epoch=slurm_generation,
        commit_guard=lambda: generation_publication_guard(
            output_dir, slurm_generation
        ),
    ):
        raise RuntimeError(
            "Could not restore marker authority after overlay repair"
        )


def _sort_measurement_shard(shard_df: Any) -> Any:
    """Sort a shard by stable metadata columns when they are available."""
    sort_columns = [
        column
        for column in (
            str(EXPERIMENT.DATASET),
            str(IMAGE.IMAGE_NAME),
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
        and parts[0] == DIR_RESULTS
        and parts[2] == DIR_MEASUREMENTS
    ):
        return parts[1]
    if (
        len(parts) >= 7
        and parts[0] == DIR_RESULTS
        and parts[2] == "zarr"
        and tuple(parts[-3:]) == ("tables", "measurements", "table.parquet")
    ):
        return parts[1]
    if path.parent.name == DIR_MEASUREMENTS:
        return path.parent.parent.name
    raise ValueError(
        f"Cannot derive dataset name from measurement path: {path}"
    )


def _run_overlay_task(
    output_dir: Path,
    task: dict[str, Any],
    *,
    slurm_generation: str,
) -> dict[str, Any]:
    """Regenerate one overlay, treating non-authoritative failures as nonfatal."""
    try:
        from ._cli_output_manager import OutputManager

        dataset_name = str(task["dataset_name"])
        store_path = Path(str(task["store_path"]))
        image = load_image_from_store(store_path)
        stem = store_stem(store_path)
        output_manager = OutputManager.from_config(
            base_dir=output_dir,
            ext=".png",
            include_dataset_column=False,
            overlay_alpha=float(task.get("overlay_alpha", 0.3)),
            save_overlays=True,
        )

        def _render(render_guard: Any) -> object:
            return output_manager.save_overlay(
                image,
                dataset_name,
                stem,
                commit_guard=render_guard,
            )

        if task.get("restore_marker_authority"):
            if not repair_overlay_marker_authority(
                output_dir,
                dataset_name,
                stem,
                store_path,
                _render,
                lifecycle_epoch=slurm_generation,
                commit_guard=lambda: generation_publication_guard(
                    output_dir, slurm_generation
                ),
            ):
                raise RuntimeError(
                    "Could not restore marker authority after overlay repair"
                )
        else:
            _render(
                lambda: generation_publication_guard(
                    output_dir, slurm_generation
                )
            )
    except SlurmGenerationInactiveError:
        raise
    except Exception as exc:
        logger.warning("Overlay regeneration failed", exc_info=True)
        return {
            "status": (
                "failed"
                if task.get("restore_marker_authority")
                else "completed"
            ),
            "overlay_failed": True,
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {"status": "completed", "overlay_failed": False}


def _restore_overlay_marker_authority(
    output_dir: Path,
    task_manifest: Path,
    *,
    slurm_generation: str | None = None,
) -> None:
    """Compare and refresh every repaired overlay marker after array work."""

    manifest = json.loads(task_manifest.read_text(encoding="utf-8"))
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("Task manifest does not contain a tasks list")
    repairs: list[dict[str, Any]] = []
    for item in tasks:
        if not isinstance(item, dict):
            continue
        if item.get("restore_marker_authority"):
            repairs.append(item)
        nested = item.get("overlay_repairs", [])
        if not isinstance(nested, list):
            raise ValueError("Task overlay_repairs must be a list")
        if not all(isinstance(repair, dict) for repair in nested):
            raise ValueError("Task overlay_repairs contains an invalid repair")
        repairs.extend(nested)

    restored: set[tuple[str, Path]] = set()
    for repair in repairs:
        store_path = Path(str(repair["store_path"]))
        dataset_name = str(repair["dataset_name"])
        identity = (dataset_name, store_path.resolve())
        if identity in restored:
            continue
        restored.add(identity)
        if not refresh_overlay_marker_authority(
            output_dir,
            dataset_name,
            store_stem(store_path),
            store_path,
            lifecycle_epoch=slurm_generation,
            commit_guard=(
                (
                    lambda: generation_publication_guard(
                        output_dir, slurm_generation
                    )
                )
                if slurm_generation is not None
                else None
            ),
        ):
            raise RuntimeError(
                "Could not restore marker authority after overlay repair"
            )


def _run_finalizer_task(
    output_dir: Path,
    task_manifest: Path,
    task: dict[str, Any],
    *,
    slurm_generation: str,
) -> None:
    """Finalize recompile outputs after all non-finalizer tasks finish."""
    progress_dir = progress_dir_helper(output_dir)
    attempt_dir = task_manifest.parent
    status_dir = attempt_dir / "status"
    expected = int(task.get("expected_non_finalizer_tasks", 0))

    statuses = _wait_for_non_finalizer_statuses(status_dir, expected)
    failed_statuses = [
        status for status in statuses if status.get("status") == "failed"
    ]
    # A failed TASK_OVERLAY is an independent, non-blocking side artifact: it
    # fails routinely and expectedly for any image with no detected objects
    # (NoObjectsError), which is normal for this dataset (e.g. no-growth
    # timepoints) and unrelated to the measurements aggregate. Every other
    # failure remains blocking, including an unclassified bootstrap failure.
    # Treating every overlay failure as fatal meant recompile could never
    # publish for a dataset with any such images at all.
    blocking_failures = [
        status
        for status in failed_statuses
        if status.get("task_type") != TASK_OVERLAY
    ]
    if blocking_failures:
        raise RuntimeError(
            f"{len(blocking_failures)} blocking non-finalizer recompile "
            "task(s) failed "
            f"(of {len(failed_statuses)} total non-finalizer failures)"
        )
    if failed_statuses:
        logger.warning(
            "%d non-finalizer recompile task(s) failed (all overlay, "
            "non-blocking) — publishing anyway",
            len(failed_statuses),
        )

    from phenotypic.sdk_ import phenotypic_cache_dir
    from phenotypic.sdk_._file_locking import exclusive_path_lock

    publication_lock = (
        phenotypic_cache_dir(output_dir) / ".aggregate_publication.lock"
    )
    with exclusive_path_lock(publication_lock, timeout=60.0):
        # Re-fingerprint co-located overlay repairs once more after every task
        # is terminal so the marker describes the final embedded-table bytes.
        # Each repair holds its store lock before entering the lifecycle fence.
        _restore_overlay_marker_authority(
            output_dir,
            task_manifest,
            slurm_generation=slurm_generation,
        )
        # The aggregate lock excludes competing finalizers. Each canonical
        # mutation also acquires the lifecycle guard independently, allowing a
        # newer generation to fence this worker between publication phases.
        # Marker-last evidence keeps any interrupted mixed snapshot unreadable.
        merged_df = _write_master_outputs_from_shards(
            output_dir,
            attempt_dir,
            slurm_generation=slurm_generation,
        )
        _run_post_master_steps(
            output_dir,
            task,
            merged_df,
            slurm_generation=slurm_generation,
        )
        from ._cli_completion import current_success_counts

        if (
            merged_df is not None
            and current_success_counts(output_dir) is not None
        ):
            from ._cli_completion import (
                current_run_is_complete,
                publish_aggregate_snapshot,
                publish_run_completion_evidence,
            )

            with generation_publication_guard(output_dir, slurm_generation):
                publish_aggregate_snapshot(output_dir)
                if current_run_is_complete(output_dir) is True:
                    publish_run_completion_evidence(
                        output_dir,
                        execution_epoch=slurm_generation,
                    )
        _regenerate_recompile_dashboard(
            output_dir,
            progress_dir,
            task,
            slurm_generation=slurm_generation,
        )


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


def _write_master_outputs_from_shards(
    output_dir: Path,
    attempt_dir: Path | None = None,
    *,
    slurm_generation: str | None = None,
) -> Any | None:
    """Concatenate shard Parquets and write master CSV/Parquet outputs."""
    import polars as pl

    shard_dir = (
        attempt_dir
        if attempt_dir is not None
        else recompile_dir_helper(progress_dir_helper(output_dir))
    ) / DIR_RECOMPILE_SHARDS
    shard_files = sorted(shard_dir.glob("shard_*.parquet"))
    if not shard_files:
        return None

    frames = [pl.read_parquet(path) for path in shard_files]
    master_df = pl.concat(frames, how="diagonal_relaxed")

    # Recompiled embedded tables already carry their publication-time metadata.
    # The master stays their exact, pre-post concatenation; finalization appends
    # only metadata identities absent from all measured tables to the mirror.

    def publish_master_outputs() -> None:
        try:
            atomic_write_with_writer(
                master_measurements_csv_path(output_dir), master_df.write_csv
            )
        except Exception:
            logger.error("Failed to save master CSV during recompile finalize")
            raise
        atomic_write_with_writer(
            master_measurements_parquet_path(output_dir),
            lambda p: master_df.write_parquet(p, **PARQUET_WRITE_OPTIONS),
        )

    if slurm_generation is None:
        publish_master_outputs()
    else:
        with generation_publication_guard(output_dir, slurm_generation):
            publish_master_outputs()

    # Seeding ``measurements.{csv,parquet}``, persisting pipeline.json,
    # emitting configured analysis outputs, and per-feature splits all happen in
    # ``_run_post_master_steps`` via ``finalize_post_master_outputs`` so
    # the post-applied frame seeded into the GUI mirror matches the one
    # fed to the analysis chain.
    return master_df


def _deactivate_generation_value(output_dir: Path, generation: str) -> None:
    """Deactivate a generation supplied independently of task metadata."""
    from ._cli_slurm_lifecycle import deactivate_generation

    deactivate_generation(output_dir, generation)


def _run_post_master_steps(
    output_dir: Path,
    task: dict[str, Any],
    merged_df: Any | None,
    *,
    slurm_generation: str | None = None,
) -> None:
    """Run canonical post-master outputs before marker publication."""
    from ._cli_output_manager import (
        _consistent_embedded_join_keys,
        _load_pipeline_from_output_dir,
        finalize_post_master_outputs,
    )

    measurement_sources = task.get("measurement_sources")
    metadata_join_keys: tuple[str, ...] | None
    if measurement_sources is None:
        metadata_join_keys = tuple(
            str(key) for key in task.get("metadata_join_keys", [])
        )
    else:
        metadata_join_keys = _consistent_embedded_join_keys(
            [Path(str(path)) for path in measurement_sources]
        )

    if merged_df is not None:
        # Single canonical post-master finalize: appends metadata-only
        # identities once, applies post to the joined measured + phantom
        # frame, seeds ``measurements.{csv,parquet}``,
        # persists ``pipeline.json``, emits configured analysis outputs, and
        # writes per-feature splits, matching the forward CLI path.
        pipeline = _load_pipeline_from_output_dir(output_dir)
        metadata_csv_str = task.get(JobMetadataKey.METADATA_CSV)
        metadata_csv = (
            Path(str(metadata_csv_str)) if metadata_csv_str else None
        )
        no_qc = bool(task.get(JobMetadataKey.NO_QC, False))
        if slurm_generation is None:
            finalize_post_master_outputs(
                output_dir,
                merged_df,
                pipeline,
                metadata_csv=metadata_csv,
                metadata_join_keys=metadata_join_keys,
                no_qc=no_qc,
            )
        else:
            with generation_publication_guard(output_dir, slurm_generation):
                finalize_post_master_outputs(
                    output_dir,
                    merged_df,
                    pipeline,
                    metadata_csv=metadata_csv,
                    metadata_join_keys=metadata_join_keys,
                    no_qc=no_qc,
                )


def _regenerate_recompile_dashboard(
    output_dir: Path,
    progress_dir: Path,
    task: dict[str, Any],
    *,
    slurm_generation: str | None = None,
) -> None:
    """Rebuild display caches after marker-last publication succeeds."""
    from ._cli_utils import load_job_metadata
    from ._dashboard import regenerate_dashboard_artifacts

    job_meta = load_job_metadata(progress_dir)
    dataset_names = [str(name) for name in task.get("dataset_names", [])]
    datasets_totals = _dataset_totals(output_dir, dataset_names)
    if slurm_generation is None:
        regenerate_dashboard_artifacts(output_dir, job_meta, datasets_totals)
    else:
        with generation_publication_guard(output_dir, slurm_generation):
            regenerate_dashboard_artifacts(
                output_dir, job_meta, datasets_totals
            )


def _dataset_totals(
    output_dir: Path, dataset_names: list[str]
) -> dict[str, int]:
    """Count per-image measurement Parquets for manifest totals."""
    from ._cli_completion import authorized_measurement_sources

    authorized = authorized_measurement_sources(output_dir)
    if authorized is not None:
        return {
            dataset_name: sum(
                dataset == dataset_name for dataset in authorized.values()
            )
            for dataset_name in dataset_names
        }

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
