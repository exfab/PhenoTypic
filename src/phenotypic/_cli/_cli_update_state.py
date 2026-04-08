"""
State file updater for the PhenoTypic CLI.

This module handles append-only event logging for tracking image processing
completion status. Uses atomic append operations with file locking for
HPC filesystem safety across parallel workers and distributed SLURM jobs.

Design Note - Race Condition Tradeoff:
    The atomic_read() implementation acquires a shared lock, reads file content,
    releases the lock, then parses the content. This means concurrent writers
    could append events between reading and parsing. We accept this tradeoff
    because:
    1. Minimizing lock hold time is critical for HPC filesystem performance
    2. Progress reporting doesn't need perfect real-time accuracy
    3. Event log state is eventually consistent (next read catches up)
    4. Alternative (parsing under lock) could cause timeout failures with slow parsing
"""

from __future__ import annotations

import logging
import click
from pathlib import Path
from typing import Literal, Dict, Set
from datetime import datetime
from dataclasses import dataclass

from ._cli_types import DatasetState
from ._cli_file_locking import atomic_read, atomic_append, FileLockTimeout

logger = logging.getLogger(__name__)


@dataclass
class ProcessingEvent:
    """Single processing event."""
    timestamp: datetime
    dataset: str
    image: str
    status: Literal["started", "completed", "failed"]
    error_msg: str = ""
    slurm_job_id: str = ""
    slurm_array_task_id: str = ""


def append_event(
    event_log: Path,
    dataset: str,
    image: str,
    status: Literal["started", "completed", "failed"],
    error_msg: str = "",
    slurm_job_id: str = "",
    slurm_array_task_id: str = "",
) -> None:
    """
    Atomically append a processing event to the event log.

    Event format: ``timestamp|dataset|image|status|error_msg|slurm_job_id|slurm_array_task_id``

    Trailing SLURM fields are omitted when empty for backward compatibility.
    Old lines with 4-5 fields still parse correctly.

    This operation uses file locking to ensure thread-safety and process-safety
    across parallel workers (local joblib and distributed SLURM jobs) on HPC
    filesystems (NFS, Lustre).

    Args:
        event_log: Path to processing_events.log file.
        dataset: Dataset name.
        image: Image filename.
        status: ``"started"``, ``"completed"``, or ``"failed"``.
        error_msg: Error message if status is ``"failed"``.
        slurm_job_id: SLURM job ID (from ``$SLURM_JOB_ID``).
        slurm_array_task_id: SLURM array task ID (from ``$SLURM_ARRAY_TASK_ID``).
    """
    event_log.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat(timespec='milliseconds')

    # Escape pipe delimiters in error message
    error_msg_safe = error_msg.replace("|", "\\|").replace("\n", " ")

    # Build event line — only include SLURM fields when non-empty
    parts = [timestamp, dataset, image, status, error_msg_safe]
    if slurm_job_id or slurm_array_task_id:
        parts.append(slurm_job_id)
        parts.append(slurm_array_task_id)
    event_line = "|".join(parts) + "\n"

    try:
        atomic_append(event_log, event_line, timeout=60.0)
    except FileLockTimeout as e:
        logger.error(
            f"Failed to acquire event log lock after 60s: {e}\n"
            f"This may indicate filesystem issues or extremely high contention.\n"
            f"Failed to record: {dataset}/{image} -> {status}"
        )
        raise RuntimeError(
            "Event log lock timeout after 60s. Cannot safely record processing status. "
            "Check filesystem performance or reduce parallel job count."
        ) from e


def append_completion_event(
    event_log: Path,
    dataset: str,
    image: str,
    status: Literal["completed", "failed"],
    error_msg: str = "",
) -> None:
    """
    Atomically append a completion event to the processing log.

    Thin wrapper around :func:`append_event` for backward compatibility.

    Args:
        event_log: Path to processing_events.log file.
        dataset: Dataset name.
        image: Image filename.
        status: ``"completed"`` or ``"failed"``.
        error_msg: Error message if status is ``"failed"``.
    """
    append_event(event_log, dataset, image, status, error_msg=error_msg)


def parse_event_line(line: str) -> ProcessingEvent:
    """
    Parse a single event log line into a ProcessingEvent.

    Supports both old 4-5 field format and new 7-field format with SLURM fields.

    Args:
        line: Raw line from event log.

    Returns:
        ProcessingEvent object.

    Raises:
        ValueError: If line format is invalid.
    """
    line = line.strip()
    if not line:
        raise ValueError("Empty line")

    parts = line.split('|')
    if len(parts) < 4:
        raise ValueError(f"Invalid line format: {line}")

    timestamp_str, dataset, image, status = parts[:4]
    error_msg = parts[4] if len(parts) > 4 else ""
    slurm_job_id = parts[5] if len(parts) > 5 else ""
    slurm_array_task_id = parts[6] if len(parts) > 6 else ""

    # Validate status field
    if status not in ("started", "completed", "failed"):
        raise ValueError(
            f"Invalid status value: '{status}' "
            f"(expected 'started', 'completed', or 'failed')"
        )

    # Unescape error message
    error_msg = error_msg.replace("\\|", "|")

    # Parse timestamp
    try:
        timestamp = datetime.fromisoformat(timestamp_str)
    except ValueError:
        timestamp = datetime.now()

    return ProcessingEvent(
        timestamp=timestamp,
        dataset=dataset,
        image=image,
        status=status,
        error_msg=error_msg,
        slurm_job_id=slurm_job_id,
        slurm_array_task_id=slurm_array_task_id,
    )


def aggregate_state_from_events(event_log: Path) -> Dict[str, DatasetState]:
    """
    Read event log and build complete processing state.

    Uses file locking to ensure consistent reads during parallel execution.
    Processes events in order, allowing retries to override previous failures.
    The most recent status for each image is used.

    Args:
        event_log: Path to processing_events.log file

    Returns:
        Dictionary mapping dataset names to their current state
    """
    def _parse_event_log(content: str) -> Dict[str, DatasetState]:
        """Inner parser function that processes event log content."""
        datasets = {}

        if not content:
            return datasets

        for line_num, line in enumerate(content.splitlines(), 1):
            if not line.strip():
                continue

            try:
                event = parse_event_line(line)
            except ValueError as e:
                logger.debug(
                    f"Skipping malformed line {line_num}: {e}"
                )
                continue

            # Initialize dataset state if needed
            if event.dataset not in datasets:
                datasets[event.dataset] = DatasetState()

            ds = datasets[event.dataset]

            # Update state based on event
            if event.status == "started":
                ds.started.add(event.image)
            elif event.status == "completed":
                ds.completed.add(event.image)
                # Remove from failed if it was previously failed (retry success)
                ds.failed.discard(event.image)
                # Remove error if present
                ds.errors.pop(event.image, None)
            elif event.status == "failed":
                ds.failed.add(event.image)
                # Remove from completed if it was previously completed (shouldn't happen)
                ds.completed.discard(event.image)
                # Store error message
                if event.error_msg:
                    ds.errors[event.image] = event.error_msg

        return datasets

    # Use atomic read with file locking
    try:
        return atomic_read(event_log, _parse_event_log, timeout=60.0)
    except FileLockTimeout as e:
        logger.error(f"Failed to acquire event log lock for reading after 60s: {e}")
        raise RuntimeError(
            "Cannot read event log - file lock timeout after 60s. "
            "Check filesystem performance."
        ) from e


def get_remaining_images(
    all_images: Set[str],
    dataset_state: DatasetState
) -> Set[str]:
    """
    Get set of images that still need processing.
    
    Args:
        all_images: Set of all image filenames in dataset
        dataset_state: Current state of dataset processing
        
    Returns:
        Set of image filenames that haven't been processed
    """
    processed = dataset_state.completed | dataset_state.failed
    return all_images - processed


# CLI interface for use by SLURM jobs
@click.command()
@click.option("--event-log", type=click.Path(path_type=Path), required=True,
              help="Path to processing_events.log file")
@click.option("--dataset", required=True,
              help="Dataset name")
@click.option("--image", required=True,
              help="Image filename")
@click.option("--status", type=click.Choice(["started", "completed", "failed"]), required=True,
              help="Processing status")
@click.option("--error", default="",
              help="Error message if status is failed")
def main(event_log: Path, dataset: str, image: str, status: str, error: str):
    """
    Append completion event to processing log.
    
    This is called by SLURM jobs to record image processing completion.
    """
    try:
        append_completion_event(event_log, dataset, image, status, error)
        click.echo(f"Logged {status} for {dataset}/{image}")
    except Exception as e:
        click.echo(f"Error logging event: {e}", err=True)
        raise


if __name__ == "__main__":
    main()
