"""Checkpoint handler for manifest rebuilds and final aggregation.

Handles ``__PHENOTYPIC_MANIFEST__`` and ``__PHENOTYPIC_FINALIZER__``
sentinel tasks embedded in SLURM array jobs. Uses file-lock leader
election so that only one concurrent task performs the work.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, cast

import click

from ._cli_file_locking import FileLockTimeout, file_lock
from ._cli_utils import load_job_metadata
from phenotypic.sdk_ import (
    PROCESSING_EVENTS_LOG,
    JobMetadataKey,
    checkpoint_lock_filename,
    event_log_path,
    measurements_parquet_path,
    processing_report_html_path,
    resolve_execution_mode,
    progress_dir as progress_dir_helper,
)
from phenotypic.sdk_.typing_ import CheckpointType

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

    from ._dashboard._manifest_builder import build_manifest

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
    )


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
    total_expected = sum(datasets_totals.values())

    # Wait for all images to complete (or fail)
    if epoch is None:
        _wait_for_completion(progress_dir, total_expected, timeout=600)

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

    try:
        from ._dashboard._analysis_data import write_analysis_sidecar

        write_analysis_sidecar(output_dir, metadata_csv=metadata_csv)
        _check_epoch()
    except Exception:
        if epoch is not None:
            raise
        logger.warning("Analysis sidecar write failed", exc_info=True)

    # Final manifest
    from ._dashboard._manifest_builder import build_manifest

    build_manifest(
        output_dir=output_dir,
        progress_dir=progress_dir,
        datasets=datasets_totals,
        execution_mode=resolve_execution_mode(job_metadata),
        start_time=job_metadata.get(JobMetadataKey.START_TIME, ""),
        slurm_job_ids=job_metadata.get(JobMetadataKey.CHUNK_JOB_IDS),
        chunk_scripts=job_metadata.get(JobMetadataKey.CHUNK_SCRIPTS),
        input_path=job_metadata.get(JobMetadataKey.INPUT_PATH),
    )
    _check_epoch()

    # Final analysis plugins
    from ._cli_chunk_writer import _run_analysis_plugins

    import polars as pl

    # Analysis plugins consume the post-applied mirror (which carries the
    # external metadata join) so dashboard sidecars match what the GUI
    # viewer and per-feature splits see. The master archive is intentionally
    # metadata-free and would regress plugin grouping if used here.
    mirror_path = measurements_parquet_path(output_dir)
    merged_df: Optional[pl.DataFrame] = None
    if mirror_path.exists():
        try:
            merged_df = pl.read_parquet(mirror_path)
        except Exception:
            logger.warning(
                "Failed to read measurements mirror for analysis plugins"
            )
    _run_analysis_plugins(output_dir, progress_dir, merged_df)
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
        if epoch is not None:
            raise
        logger.warning("Dashboard generation failed", exc_info=True)

    if epoch is not None:
        _publish_staged_report_and_readme(output_dir, job_metadata, epoch)
        from ._cli_staged_orchestration import mark_staged_complete

        mark_staged_complete(output_dir, epoch)

    logger.info("Finalization complete")


def _publish_staged_report_and_readme(
    output_dir: Path, job_metadata: dict, epoch: str
) -> None:
    """Publish the report and README as part of the sole remote finalizer."""
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
    states = aggregate_state_from_events(event_log_path(output_dir))
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
    progress_dir: Path, total_expected: int, timeout: int = 600
) -> None:
    """Poll event log until all images are done or timeout.

    Args:
        progress_dir: Progress directory (the event log is at
            ``progress_dir.parent / "processing_events.log"``).
        total_expected: Total number of images expected.
        timeout: Maximum seconds to wait.
    """
    from ._cli_update_state import aggregate_state_from_events

    event_log = progress_dir.parent / PROCESSING_EVENTS_LOG
    deadline = time.monotonic() + timeout
    done = 0
    while time.monotonic() < deadline:
        dataset_states = aggregate_state_from_events(event_log)
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
