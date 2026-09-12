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
    task_status_filename,
    shard_parquet_filename,
    progress_dir as progress_dir_helper,
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
            source_work_ids = _run_measurement_task(
                output_dir,
                task_manifest,
                task,
                slurm_generation=slurm_generation,
            )
            # The merged set travels in the status file, exactly as the
            # forward fan-out's `write_shard_status` records it: the
            # finalizer unions these and publishes the aggregate proof
            # against them, so a store this shard's projection excluded is
            # never certified. Before this it was not reported at all, and
            # the finalizer re-derived the set from the live markers.
            #
            # **The key is absent, not empty, when the shard cannot answer**
            # (legacy external Parquets -- see `_run_measurement_task`).
            # `[]` is a claim that the shard merged nothing; absence is a
            # claim about nothing at all, and the finalizer distinguishes
            # them.
            fields: dict[str, Any] = {"status": "completed"}
            if source_work_ids is not None:
                fields["source_work_ids"] = source_work_ids
            _write_status(
                task_manifest,
                task_index,
                task_type,
                fields,
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
) -> list[str] | None:
    """Aggregate one measurement shard and write it under progress.

    **No store write.** This task used to call
    ``recompile_embedded_measurement_table`` over every embedded table in its
    shard before aggregating them; the rewrite is gone (user ruling,
    2026-09-11) and recompile now reads the stores exactly as the forward
    fan-out does. Only the co-located overlay repair still mutates the tree,
    and it rewrites an overlay PNG and its record, never the store's table.

    Returns:
        The source work identities this shard merged, sorted -- the same
        contract as
        :func:`~phenotypic._cli._cli_finalize_fanout.write_measurement_shard`.
        The finalizer unions these across shards and hands them to
        ``finalize_run`` as ``planned_work_ids``, so a store the projection
        excluded is absent from the set the aggregate proof certifies.

        ``None`` when the shard's sources are **legacy external Parquets**
        rather than embedded tables. Those have no store and so no per-image
        authority to read a ``work_id`` out of, and inventing one would be
        worse than saying nothing: a shard that reports ``None`` makes the
        finalizer fall back to the set it selects itself, which is exactly
        the pre-2026-09-11 behaviour and is correct for a shape that
        ``refuse_mixed_measurement_authority`` is already retiring.
    """

    from ._cli_parquet_agg import aggregate_measurement_sources

    files = [Path(path) for path in task.get("files", [])]

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

    path_to_dataset = {
        path: _dataset_name_from_measurement_path(output_dir, path)
        for path in files
    }
    # Projected like every other read path into the master (P7 Task 4): a
    # migrated store's table is still joined, and reading it raw here would
    # make recompile publish a different master than the forward fan-out.
    shard_df, merged = aggregate_measurement_sources(
        path_to_dataset,
        include_dataset_column=bool(task.get("include_dataset_column", True)),
    )

    shard_id = int(task["shard_id"])
    shard_path = (
        task_manifest.parent
        / DIR_RECOMPILE_SHARDS
        / shard_parquet_filename(shard_id)
    )
    if shard_df is None and merged:
        # Sources existed and were merged, yet no frame came back: the shard
        # cannot be written and the failure is real. Distinct from the branch
        # below, where the projection legitimately excluded every store.
        raise RuntimeError(
            f"Shard {shard_id} found no readable measurements among "
            f"{len(path_to_dataset)} source(s)"
        )
    if shard_df is None:
        # An EMPTY shard is written, deliberately, rather than raising -- the
        # same rule `write_measurement_shard` follows. A shard whose stores
        # the projection excluded (each with its own advisory) has merged
        # nothing legitimately, and raising here failed the whole recompile
        # for a state the forward path treats as ordinary. The finalizer
        # counts shard files, so a skipped file would read as a dead worker.
        import polars as pl

        with generation_publication_guard(output_dir, slurm_generation):
            atomic_write_with_writer(
                shard_path,
                lambda p: pl.DataFrame().write_parquet(
                    p, **PARQUET_WRITE_OPTIONS
                ),
            )
        return []

    from ._measurement_sources import (
        add_metadata_image_name_from_filename,
    )

    shard_df = add_metadata_image_name_from_filename(shard_df)
    shard_df = _sort_measurement_shard(shard_df)

    with generation_publication_guard(output_dir, slurm_generation):
        atomic_write_with_writer(
            shard_path,
            lambda p: shard_df.write_parquet(p, **PARQUET_WRITE_OPTIONS),
        )

    from ._cli_parquet_agg import is_embedded_measurement_table

    if not all(is_embedded_measurement_table(path) for path in merged):
        return None
    return sorted(_merged_source_work_ids(output_dir, merged))


def _merged_source_work_ids(
    output_dir: Path, sources: dict[Path, str]
) -> list[str]:
    """Return the work identity backing each merged source.

    The recompile analogue of
    :func:`~phenotypic._cli._cli_finalize_fanout._work_ids_for_sources`, and
    **deliberately not a call to it**: that one reads
    ``progress/images/<ds>/<stem>.json`` and nothing else, because the forward
    fan-out only ever runs on a tree this build wrote. Recompile still accepts
    a pre-record tree whose authority is a legacy ``image_complete/`` marker
    (:func:`~phenotypic._cli._cli_recompile_recovery._image_authority_shapes`
    carries that arm and its retirement condition), and asking such a tree for
    a record returns ``None`` for every image.

    **A source with no authority raises rather than being skipped**, matching
    the forward rule: a silently shortened planned set means the aggregate
    proof certifies an image the master does not carry.

    Args:
        output_dir: Run output root.
        sources: Merged measurement table -> dataset.

    Returns:
        One work identity per source, in *sources* iteration order.

    Raises:
        RuntimeError: A source has no readable authority payload, or the
            payload carries no ``work_id``.
    """
    from phenotypic.sdk_ import MEASUREMENT_TABLE_RELATIVE_PATH

    from ._cli_recompile_recovery import image_authority_payload

    depth = len(MEASUREMENT_TABLE_RELATIVE_PATH.parts)
    work_ids: list[str] = []
    for source, dataset in sources.items():
        store = Path(source).parents[depth - 1]
        stem = store_stem(store)
        try:
            _path, payload, _version = image_authority_payload(
                Path(output_dir).resolve(), dataset, stem
            )
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise RuntimeError(
                f"No per-image authority for {store}; its work identity "
                "cannot be recorded, and a shard that dropped it would "
                "publish a proof asserting an image the master does not carry"
            ) from exc
        work_id = payload.get("work_id")
        if not isinstance(work_id, str) or not work_id:
            raise RuntimeError(
                f"The per-image authority for {store} carries no work_id"
            )
        work_ids.append(work_id)
    return work_ids


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
        master_path = _run_post_master_steps(
            output_dir,
            task,
            attempt_dir=attempt_dir,
            slurm_generation=slurm_generation,
            statuses=statuses,
        )
        from ._cli_completion import state_requires_success_markers

        # `is not None` asked "is this a schema-3 state?", never a count.
        if master_path is not None and state_requires_success_markers(
            output_dir
        ):
            from ._cli_completion import (
                _all_accepted_images_succeeded,
                publish_run_completion_evidence,
            )

            # No `publish_aggregate_snapshot` here any more: `finalize_run`
            # publishes the aggregate proof itself, on the authorized arm,
            # immediately after the outputs it certifies. Publishing it a
            # second time from out here would be a second writer for one
            # artifact within a single pass.
            with generation_publication_guard(output_dir, slurm_generation):
                # Guarded by the schema-3 check above, so the legacy arm
                # cannot reach here and `== "complete"` is the whole question.
                if _all_accepted_images_succeeded(output_dir) is True:
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


def _deactivate_generation_value(output_dir: Path, generation: str) -> None:
    """Deactivate a generation supplied independently of task metadata."""
    from ._cli_slurm_lifecycle import deactivate_generation

    deactivate_generation(output_dir, generation)


def _recompile_shard_paths(attempt_dir: Path) -> list[Path]:
    """Return this attempt's measurement shards, in merge order."""
    return sorted(
        (attempt_dir / DIR_RECOMPILE_SHARDS).glob("shard_*.parquet")
    )


def _run_post_master_steps(
    output_dir: Path,
    task: dict[str, Any],
    *,
    attempt_dir: Path,
    slurm_generation: str | None = None,
    statuses: list[dict[str, Any]] | None = None,
) -> Path | None:
    """Finalize recompile through the one aggregation + join + publish path.

    **This is where recompile's separate master-merge used to live.**
    ``_write_master_outputs_from_shards`` concatenated the per-shard Parquets,
    wrote its own master CSV and Parquet, and only then handed the frame to
    ``finalize_post_master_outputs`` -- a second implementation of
    finalization that had to be kept in sync with the forward one by hand.
    Spec §7.4 makes recompile *"call finalize_run again"* instead.

    The shards are not dropped: they are handed to :func:`finalize_run` as
    ``shard_paths``, the fan-out hook its signature already declares. That
    keeps the merge exactly where it was -- these workers are why the hook
    exists -- while leaving **one** writer for the master, which is what
    "one writer per artifact, per pass" requires and what two independent
    master writes in a single finalizer would have broken.

    An empty shard directory yields ``None``, as the merge it replaces did.

    ``generation_publication_guard`` still wraps the whole thing under SLURM,
    unchanged -- an ``if slurm_generation is None`` / ``else`` pair around two
    otherwise identical calls.

    Args:
        output_dir: Run output root.
        task: The finalizer task dict. ``include_dataset_column``, the
            metadata snapshot and ``no_qc`` are read back off it -- the
            serialization boundary ``_cli_recompile_slurm_scripts`` writes.
        attempt_dir: This attempt's directory, holding ``measurement_shards/``.
        slurm_generation: Active lifecycle generation, or ``None`` locally.
        statuses: The non-finalizer task statuses this finalizer waited for.
            The union of their ``source_work_ids`` is the **planned** set the
            aggregate proof is published against -- what the shards recorded
            merging, rather than what ``authorized_measurement_sources``
            answers now. ``None``, or a set carrying no measurement task,
            leaves ``planned_work_ids`` unset and ``finalize_run`` falls back
            to the sources it selected itself.

    Returns:
        Path to ``master_measurements.parquet``, or ``None`` when no
        measurement source could be read.
    """
    from ._cli_finalize_run import (
        finalize_run,
        refuse_mixed_measurement_authority,
    )

    # CAN-2: the recorded per-store join keys are gone. `finalize_post_master_
    # outputs` derives its own common columns from the master and the snapshot,
    # so the `measurement_sources` / `metadata_join_keys` split -- which existed
    # only because the two callers arrived with differently-shaped inputs --
    # goes with them. What survives is the mixed-AUTHORITY refusal (H6), which
    # the retired function also carried and which nothing else replaces.
    measurement_sources = task.get("measurement_sources")
    if measurement_sources is not None:
        refuse_mixed_measurement_authority(
            [Path(str(path)) for path in measurement_sources]
        )

    metadata_csv_str = task.get(JobMetadataKey.METADATA_CSV)
    metadata_csv = Path(str(metadata_csv_str)) if metadata_csv_str else None
    kwargs: dict[str, Any] = {
        "dataset_names": [
            str(name) for name in task.get("dataset_names", [])
        ],
        "include_dataset_column": bool(
            task.get("include_dataset_column", True)
        ),
        "metadata_csv": metadata_csv,
        "no_qc": bool(task.get(JobMetadataKey.NO_QC, False)),
        "shard_paths": _recompile_shard_paths(attempt_dir),
    }
    # **Only when every measurement shard answered.** An empty union is a
    # real answer from a shard that merged nothing; an absent one is not the
    # same claim, and passing `[]` for it would publish a proof asserting no
    # image over a master built from every authorized source. So the set is
    # forwarded when there is at least one measurement status and **all** of
    # them carry the key -- a partial union would understate the master just
    # as badly as a live re-derivation overstates it.
    measurement_statuses = [
        status
        for status in (statuses or [])
        if status.get("task_type") == TASK_MEASUREMENTS
    ]
    if measurement_statuses and all(
        isinstance(status.get("source_work_ids"), list)
        for status in measurement_statuses
    ):
        from ._cli_finalize_fanout import planned_work_ids_from_statuses

        kwargs["planned_work_ids"] = planned_work_ids_from_statuses(
            measurement_statuses
        )
    if slurm_generation is None:
        return finalize_run(output_dir, **kwargs)
    with generation_publication_guard(output_dir, slurm_generation):
        return finalize_run(output_dir, **kwargs)


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
