"""Checkpoint handler for manifest rebuilds and final aggregation.

Handles nonterminal ``__PHENOTYPIC_MANIFEST__`` array tasks and the
scheduler-dependent terminal finalizer job. Uses file-lock leader election
so that only one concurrent task performs the work.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import click

from ._cli_file_locking import FileLockTimeout, file_lock
from ._cli_preload import preload_custom_operation_modules
from ._cli_utils import load_job_metadata
from phenotypic.sdk_ import (
    PROCESSING_EVENTS_LOG,
    DashboardManifestKey,
    JobMetadataKey,
    atomic_write_json,
    checkpoint_lock_filename,
    event_log_path,
    processing_report_html_path,
    resolve_manifest_json_path,
    resolve_execution_mode,
    run_completion_marker_path,
    progress_dir as progress_dir_helper,
)
from phenotypic.sdk_.typing_ import CheckpointType
from phenotypic.sdk_._file_locking import exclusive_path_lock

logger = logging.getLogger(__name__)


@click.command("checkpoint-handler")
@click.option(
    "--output-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option(
    "--checkpoint-type",
    type=click.Choice(["manifest", "finalize"]),
    required=True,
)
@click.option(
    "--epoch", default=None, help="Internal staged orchestration epoch."
)
def main(output_dir: Path, checkpoint_type: str, epoch: str | None) -> None:
    """Handle manifest or finalize checkpoint tasks."""
    # Click validated the value via Choice, but it arrives as bare str — narrow
    # to the typed alias before passing into render functions / comparisons.
    checkpoint: CheckpointType = (
        "manifest" if checkpoint_type == "manifest" else "finalize"
    )
    progress_dir = progress_dir_helper(output_dir)
    lock_path = progress_dir / checkpoint_lock_filename(checkpoint)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.touch(exist_ok=True)

    while True:
        try:
            with open(lock_path, "r") as fh:
                with file_lock(
                    fh, timeout=30.0 if epoch is not None else 1.0, shared=False
                ):
                    if epoch is not None:
                        from ._cli_staged_orchestration import (
                            staged_completion_matches,
                        )

                        if staged_completion_matches(output_dir, epoch):
                            return
                    if checkpoint == "manifest":
                        _run_manifest(output_dir, progress_dir)
                    else:
                        _run_finalize(
                            output_dir,
                            progress_dir,
                            epoch=epoch,
                        )
                    return
        except FileLockTimeout:
            if epoch is None:
                logger.info("Another task is handling %s -- skipping", checkpoint)
                return
            from ._cli_staged_orchestration import (
                load_orchestration_state,
                staged_completion_matches,
            )

            if staged_completion_matches(output_dir, epoch):
                return
            state = load_orchestration_state(output_dir)
            if state is None or state.get("epoch") != epoch or state.get(
                "phase"
            ) in {"failed", "cancelled"}:
                raise RuntimeError(
                    "The authoritative staged finalizer ended without a "
                    "matching completion marker"
                )
            logger.info("Waiting for the authoritative staged finalizer")
        except Exception:
            if epoch is not None:
                from ._cli_staged_orchestration import deactivate_orchestration

                deactivate_orchestration(output_dir, "failed")
            raise


# ---------------------------------------------------------------------------
# Manifest checkpoint
# ---------------------------------------------------------------------------


def _run_manifest(output_dir: Path, progress_dir: Path) -> None:
    """Rebuild ``manifest.json`` from event log and sacct.

    Args:
        output_dir: Root output directory.
        progress_dir: Progress directory containing ``job_metadata.json``.
    """
    job_metadata = load_job_metadata(progress_dir)
    if job_metadata is None:
        logger.warning("No job_metadata.json found -- skipping manifest build")
        return

    from ._dashboard._manifest_builder import (
        build_manifest,
        dataset_inventory_from_metadata,
    )

    datasets_raw = job_metadata.get(JobMetadataKey.DATASETS, {}) or {}
    datasets_totals: dict[str, int] = {
        name: (info["total"] if isinstance(info, dict) else int(info))
        for name, info in datasets_raw.items()
    }

    build_manifest(
        output_dir=output_dir,
        progress_dir=progress_dir,
        datasets=datasets_totals,
        execution_mode=resolve_execution_mode(job_metadata),
        start_time=job_metadata.get(JobMetadataKey.START_TIME, ""),
        slurm_job_ids=job_metadata.get(JobMetadataKey.CHUNK_JOB_IDS),
        chunk_scripts=job_metadata.get(JobMetadataKey.CHUNK_SCRIPTS),
        input_path=job_metadata.get(JobMetadataKey.INPUT_PATH),
        dataset_inventory=dataset_inventory_from_metadata(datasets_raw),
        processing_generation=job_metadata.get(
            JobMetadataKey.PROCESSING_GENERATION
        ),
    )

    # Process/export runs use a scheduler-dependent manifest finalizer after
    # the final image array. Publish here only for that mode; forward/measure
    # manifest checkpoints never publish completion.
    slurm_generation = job_metadata.get("slurm_generation")
    if isinstance(slurm_generation, str) and slurm_generation:
        from ._cli_state_management import load_processing_state

        state = load_processing_state(output_dir)
        process_only_layer = (
            state.config.get("process_only_layer") if state is not None else None
        )
        if process_only_layer:
            _publish_run_completion_marker(output_dir, slurm_generation)


# ---------------------------------------------------------------------------
# Finalize checkpoint
# ---------------------------------------------------------------------------


def _run_finalize(
    output_dir: Path, progress_dir: Path, *, epoch: str | None = None
) -> None:
    """Wait for completion, then run final aggregation + manifest + analysis.

    Args:
        output_dir: Root output directory.
        progress_dir: Progress directory containing ``job_metadata.json``.
    """
    job_metadata = load_job_metadata(progress_dir)
    if job_metadata is None:
        if epoch is None:
            logger.warning("No job_metadata.json; cannot finalize")
            return
        raise RuntimeError("No job_metadata.json; cannot finalize")
    slurm_generation_raw = job_metadata.get("slurm_generation")
    slurm_generation = (
        str(slurm_generation_raw)
        if isinstance(slurm_generation_raw, str) and slurm_generation_raw
        else None
    )

    if epoch is not None:
        from ._cli_staged_orchestration import assert_active_epoch

        assert_active_epoch(output_dir, epoch)

    def _check_epoch() -> None:
        if epoch is not None:
            from ._cli_staged_orchestration import assert_active_epoch

            assert_active_epoch(output_dir, epoch)

    datasets_raw = job_metadata.get(JobMetadataKey.DATASETS, {}) or {}
    datasets_totals: dict[str, int] = {
        name: (info["total"] if isinstance(info, dict) else int(info))
        for name, info in datasets_raw.items()
    }
    from ._dashboard._manifest_builder import (
        dataset_inventory_from_metadata,
    )

    dataset_inventory = dataset_inventory_from_metadata(datasets_raw)
    total_expected = sum(datasets_totals.values())

    # Wait for all images to complete (or fail)
    if epoch is None:
        _wait_for_completion(
            progress_dir,
            inventory=dataset_inventory,
            total_expected=total_expected,
            generation=job_metadata.get(
                JobMetadataKey.PROCESSING_GENERATION
            ),
            timeout=600,
        )

    # Final aggregation
    from ._cli_output_manager import aggregate_measurements

    _check_epoch()
    if epoch is not None:
        from ._cli_staged_orchestration import (
            load_orchestration_state,
            quarantine_unchanged_restart_parquets,
        )

        quarantine_unchanged_restart_parquets(output_dir, epoch)
        orchestration = load_orchestration_state(output_dir) or {}
        if bool(orchestration.get("stage3_markers_required", False)):
            from ._cli_staged_resume import reconcile_stage3_publications

            reconcile_stage3_publications(
                output_dir,
                {
                    name: list(info.get("images", []))
                    for name, info in datasets_raw.items()
                    if isinstance(info, dict)
                },
                namespace=epoch,
            )
        _check_epoch()

    metadata_csv_str = job_metadata.get(JobMetadataKey.METADATA_CSV)
    metadata_csv = Path(metadata_csv_str) if metadata_csv_str else None

    aggregate_path = aggregate_measurements(
        output_dir=output_dir,
        dataset_names=list(datasets_totals.keys()),
        include_dataset_column=job_metadata.get(
            JobMetadataKey.INCLUDE_DATASET_COLUMN, True
        ),
        metadata_csv=metadata_csv,
        no_qc=bool(job_metadata.get(JobMetadataKey.NO_QC, False)),
    )
    if aggregate_path is None:
        message = "No current-epoch measurements were available to aggregate"
        if epoch is not None:
            raise RuntimeError(message)
        logger.warning(message)
    _check_epoch()

    # Final manifest
    from ._dashboard._manifest_builder import (
        build_manifest,
        dataset_inventory_from_metadata,
    )

    build_manifest(
        output_dir=output_dir,
        progress_dir=progress_dir,
        datasets=datasets_totals,
        execution_mode=resolve_execution_mode(job_metadata),
        start_time=job_metadata.get(JobMetadataKey.START_TIME, ""),
        slurm_job_ids=job_metadata.get(JobMetadataKey.CHUNK_JOB_IDS),
        chunk_scripts=job_metadata.get(JobMetadataKey.CHUNK_SCRIPTS),
        input_path=job_metadata.get(JobMetadataKey.INPUT_PATH),
        dataset_inventory=dataset_inventory,
        processing_generation=job_metadata.get(
            JobMetadataKey.PROCESSING_GENERATION
        ),
    )
    _check_epoch()

    # Regenerate dashboard
    try:
        from ._dashboard._generator import generate_dashboard

        generate_dashboard(
            output_dir,
            execution_mode=resolve_execution_mode(job_metadata),
        )
        _check_epoch()
    except Exception:
        if epoch is not None or slurm_generation is not None:
            raise
        logger.warning("Dashboard generation failed", exc_info=True)

    if epoch is not None:
        _publish_staged_report_and_readme(output_dir, job_metadata, epoch)
        from ._cli_staged_orchestration import mark_staged_complete

        mark_staged_complete(output_dir, epoch)
    elif slurm_generation is not None:
        _publish_run_completion_marker(output_dir, slurm_generation)

    logger.info("Finalization complete")


def _publish_run_completion_marker(
    output_dir: Path,
    slurm_generation: str,
) -> None:
    """Publish and fence ordinary completion for the exact active generation."""
    from ._cli_slurm_lifecycle import (
        _deactivate_generation_locked,
        lifecycle_lock_path,
        load_slurm_lifecycle,
    )

    marker_path = run_completion_marker_path(output_dir)
    with exclusive_path_lock(lifecycle_lock_path(output_dir), timeout=60.0):
        lifecycle = load_slurm_lifecycle(output_dir)
        if lifecycle is None or lifecycle.get("generation") != slurm_generation:
            raise RuntimeError(
                "Cannot publish completion for a stale SLURM generation"
            )
        if lifecycle.get("active") is not True:
            try:
                existing = json.loads(marker_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                existing = None
            if (
                isinstance(existing, dict)
                and existing.get("generation") == slurm_generation
                and existing.get("status") == "complete"
                and existing.get("finalizer_succeeded") is True
            ):
                return
            raise RuntimeError(
                "Cannot publish completion after the SLURM generation "
                "was cancelled or superseded"
            )
        try:
            manifest = json.loads(
                resolve_manifest_json_path(output_dir).read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                "Cannot publish the SLURM completion marker without "
                "a valid manifest"
            ) from exc
        if not isinstance(manifest, dict):
            raise RuntimeError("Final dashboard manifest is not a JSON object")
        completed = manifest.get(DashboardManifestKey.COMPLETED)
        failed = manifest.get(DashboardManifestKey.FAILED)
        total = manifest.get(DashboardManifestKey.TOTAL_IMAGES)
        complete = (
            manifest.get(DashboardManifestKey.IS_COMPLETE) is True
            and isinstance(completed, int)
            and not isinstance(completed, bool)
            and isinstance(failed, int)
            and not isinstance(failed, bool)
            and isinstance(total, int)
            and not isinstance(total, bool)
            and failed == 0
            and completed == total
        )
        if not complete:
            raise RuntimeError(
                "Cannot publish the SLURM completion marker for an incomplete "
                "or failed manifest"
            )
        atomic_write_json(
            marker_path,
            {
                "schema_version": 1,
                "generation": slurm_generation,
                "status": "complete",
                "finalizer_succeeded": True,
                "completed_at": datetime.now(timezone.utc).isoformat(
                    timespec="milliseconds"
                ),
            },
        )
        if not _deactivate_generation_locked(output_dir, slurm_generation):
            raise RuntimeError(
                "SLURM completion marker was published but the generation "
                "could not be deactivated"
            )


def _publish_staged_report_and_readme(
    output_dir: Path, job_metadata: dict, epoch: str
) -> None:
    """Publish the report and README as part of the sole remote finalizer."""
    preload_custom_operation_modules()

    from phenotypic import ImagePipeline

    from ._cli_readme_generator import READMEGenerator
    from ._cli_report_generator import HTMLReportGenerator
    from ._cli_staged_orchestration import completed_inventory_images
    from ._cli_types import (
        Dataset,
        DatasetResults,
        ExecutionConfig,
        ExecutionResults,
        ImageFailure,
    )
    from ._cli_update_state import aggregate_state_from_events

    datasets_raw = job_metadata.get(JobMetadataKey.DATASETS, {}) or {}
    inventory = {
        name: list(raw.get("images", []))
        for name, raw in datasets_raw.items()
        if isinstance(raw, dict)
    }
    states = aggregate_state_from_events(
        event_log_path(output_dir),
        inventory=inventory,
        generation=job_metadata.get(JobMetadataKey.PROCESSING_GENERATION),
    )
    dataset_results: dict[str, DatasetResults] = {}
    datasets: list[Dataset] = []
    for name, raw in datasets_raw.items():
        images = list(raw.get("images", [])) if isinstance(raw, dict) else []
        state = states.get(name)
        completed = completed_inventory_images(output_dir, name, images)
        event_failed = set() if state is None else state.failed
        failed = (set(images) - completed) | event_failed
        failed -= completed
        errors = {} if state is None else state.errors
        failures = [
            ImageFailure(
                dataset=name,
                image_filename=image,
                error_type="StageFailure",
                error_message=errors.get(image, "Staged processing failed"),
                traceback="",
                timestamp=datetime.now(),
            )
            for image in sorted(failed)
        ]
        dataset_results[name] = DatasetResults(
            name=name,
            total=len(images),
            completed=len(completed),
            failed=len(failed),
            failures=failures,
        )
        datasets.append(
            Dataset(
                name=name,
                images=[Path(image) for image in images],
                input_dir=Path("."),
                output_dir=output_dir,
            )
        )
    start_raw = job_metadata.get(JobMetadataKey.START_TIME, "")
    try:
        start = datetime.fromisoformat(start_raw)
    except (TypeError, ValueError):
        start = datetime.now()
    results = ExecutionResults(
        datasets=dataset_results,
        total_images=sum(result.total for result in dataset_results.values()),
        total_completed=sum(
            result.completed for result in dataset_results.values()
        ),
        total_failed=sum(result.failed for result in dataset_results.values()),
        execution_mode="slurm",
        start_time=start,
        end_time=datetime.now(),
        remote_finalized=True,
    )
    HTMLReportGenerator().generate_report(
        results, processing_report_html_path(output_dir)
    )
    from ._cli_staged_orchestration import assert_active_epoch

    assert_active_epoch(output_dir, epoch)

    pipeline_path = Path(job_metadata[JobMetadataKey.PIPELINE_PATH])
    pipeline = ImagePipeline.from_json(pipeline_path)
    config = cast(
        ExecutionConfig,
        SimpleNamespace(
            pipeline_json=pipeline_path,
            image_type=job_metadata.get(JobMetadataKey.IMAGE_TYPE, "Image"),
            nrows=job_metadata.get(JobMetadataKey.NROWS),
            ncols=job_metadata.get(JobMetadataKey.NCOLS),
        ),
    )
    READMEGenerator(config, pipeline).generate(output_dir, datasets)
    assert_active_epoch(output_dir, epoch)


def _wait_for_completion(
    progress_dir: Path,
    *,
    inventory: dict[str, frozenset[str]] | None,
    total_expected: int,
    generation: str | None,
    timeout: int = 600,
) -> None:
    """Poll event log until all images are done or timeout.

    Args:
        progress_dir: Progress directory (the event log is at
            ``progress_dir.parent / "processing_events.log"``).
        inventory: Authorized current-generation image names by dataset, or
            ``None`` for legacy count-only metadata.
        total_expected: Total number of expected images.
        generation: Durable processing generation.
        timeout: Maximum seconds to wait.
    """
    from ._cli_update_state import aggregate_state_from_events

    event_log = progress_dir.parent / PROCESSING_EVENTS_LOG
    deadline = time.monotonic() + timeout
    done = 0
    while time.monotonic() < deadline:
        dataset_states = aggregate_state_from_events(
            event_log,
            inventory=inventory,
            generation=generation,
        )
        completed = sum(len(ds.completed) for ds in dataset_states.values())
        failed = sum(len(ds.failed) for ds in dataset_states.values())
        done = completed + failed
        if done >= total_expected:
            logger.info(
                "All %d images done (%d completed, %d failed)",
                total_expected,
                completed,
                failed,
            )
            return
        time.sleep(10)
    logger.warning(
        "Timed out waiting for completion (%d/%d done after %ds)",
        done,
        total_expected,
        timeout,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
