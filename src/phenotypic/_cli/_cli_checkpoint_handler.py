"""Checkpoint handler for manifest rebuilds and final aggregation.

Handles ``__PHENOTYPIC_MANIFEST__`` and ``__PHENOTYPIC_FINALIZER__``
sentinel tasks embedded in SLURM array jobs. Uses file-lock leader
election so that only one concurrent task performs the work.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import click

from ._cli_file_locking import FileLockTimeout, file_lock
from ._cli_utils import load_job_metadata
from phenotypic.tools_ import (
    PROCESSING_EVENTS_LOG,
    JobMetadataKey,
    checkpoint_lock_filename,
    measurements_parquet_path,
    resolve_execution_mode,
    progress_dir as progress_dir_helper,
)
from phenotypic.tools_.typing_ import CheckpointType

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
def main(output_dir: Path, checkpoint_type: str) -> None:
    """Handle manifest or finalize checkpoint tasks."""
    # Click validated the value via Choice, but it arrives as bare str — narrow
    # to the typed alias before passing into render functions / comparisons.
    checkpoint: CheckpointType = "manifest" if checkpoint_type == "manifest" else "finalize"
    progress_dir = progress_dir_helper(output_dir)
    lock_path = progress_dir / checkpoint_lock_filename(checkpoint)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.touch(exist_ok=True)

    try:
        with open(lock_path, "r") as fh:
            with file_lock(fh, timeout=1.0, shared=False):
                if checkpoint == "manifest":
                    _run_manifest(output_dir, progress_dir)
                else:
                    _run_finalize(output_dir, progress_dir)
    except FileLockTimeout:
        logger.info("Another task is handling %s -- skipping", checkpoint)


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


def _run_finalize(output_dir: Path, progress_dir: Path) -> None:
    """Wait for completion, then run final aggregation + manifest + analysis.

    Args:
        output_dir: Root output directory.
        progress_dir: Progress directory containing ``job_metadata.json``.
    """
    job_metadata = load_job_metadata(progress_dir)
    if job_metadata is None:
        logger.warning("No job_metadata.json -- cannot finalize")
        return

    datasets_raw = job_metadata.get(JobMetadataKey.DATASETS, {}) or {}
    datasets_totals: dict[str, int] = {
        name: (info["total"] if isinstance(info, dict) else int(info))
        for name, info in datasets_raw.items()
    }
    total_expected = sum(datasets_totals.values())

    # Wait for all images to complete (or fail)
    _wait_for_completion(progress_dir, total_expected, timeout=600)

    # Final aggregation
    from ._cli_output_manager import aggregate_measurements

    metadata_csv_str = job_metadata.get(JobMetadataKey.METADATA_CSV)
    metadata_csv = Path(metadata_csv_str) if metadata_csv_str else None

    aggregate_measurements(
        output_dir=output_dir,
        dataset_names=list(datasets_totals.keys()),
        include_dataset_column=job_metadata.get(JobMetadataKey.INCLUDE_DATASET_COLUMN, True),
        metadata_csv=metadata_csv,
    )

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
            logger.warning("Failed to read measurements mirror for analysis plugins")
    _run_analysis_plugins(output_dir, progress_dir, merged_df)

    # Regenerate dashboard
    try:
        from ._dashboard._generator import generate_dashboard

        generate_dashboard(
            output_dir,
            execution_mode=resolve_execution_mode(job_metadata),
        )
    except Exception:
        logger.warning("Dashboard generation failed", exc_info=True)

    logger.info("Finalization complete")


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
