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
import os
from collections.abc import Collection, Mapping

import click
from pathlib import Path
from typing import Dict, Set
from datetime import datetime
from dataclasses import dataclass

from ._cli_types import DatasetState
from ._cli_file_locking import atomic_read, atomic_append, FileLockTimeout
from ._stages import STAGED_TERMINAL_STAGE, StageTag, validate_stage_tag
from phenotypic.sdk_ import CommitGuard
from phenotypic.sdk_.typing_ import ProcessingStatus

logger = logging.getLogger(__name__)

PROCESSING_GENERATION_ENV_VAR = "PHENOTYPIC_PROCESSING_GENERATION"
SLURM_GENERATION_ENV_VAR = "PHENOTYPIC_SLURM_GENERATION"
_DIAGNOSTIC_SAMPLE_LIMIT = 20


@dataclass
class ProcessingEvent:
    """Single processing event."""
    timestamp: datetime
    dataset: str
    image: str
    status: ProcessingStatus
    error_msg: str = ""
    slurm_job_id: str = ""
    slurm_array_task_id: str = ""
    stage: StageTag | None = None
    generation: str | None = None


@dataclass(frozen=True)
class EventAggregationDiagnostics:
    """Events excluded from an inventory- and generation-scoped aggregate."""

    malformed_events: int = 0
    unknown_image_events: int = 0
    other_generation_events: int = 0
    samples: tuple[str, ...] = ()


@dataclass(frozen=True)
class EventAggregation:
    """Current dataset state plus bounded diagnostics for ignored events."""

    datasets: Dict[str, DatasetState]
    diagnostics: EventAggregationDiagnostics


def append_event(
    event_log: Path,
    dataset: str,
    image: str,
    status: ProcessingStatus,
    error_msg: str = "",
    slurm_job_id: str = "",
    slurm_array_task_id: str = "",
    stage: str | None = None,
    generation: str | None = None,
    attempt_id: str = "",
    work_id: str = "",
    commit_guard: CommitGuard | None = None,
) -> None:
    """
    Atomically append a processing event to the event log.

    Event format: ``timestamp|dataset|image|status|error_msg|slurm_job_id|slurm_array_task_id|stage|generation``

    Trailing SLURM fields are omitted when empty for backward compatibility.
    Old lines with 4-5 fields still parse correctly. The optional ``stage``
    (``"stage1"``/``"stage2"``/``"stage3"`` for the staged GPU engine) is field 8;
    when present, the SLURM fields 6-7 are always emitted (possibly empty) so the
    positional parser can locate it. The optional durable processing generation
    is field 9. ``status`` stays the closed 3-value set.

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
        generation: Durable processing generation. Defaults to the
            ``PHENOTYPIC_PROCESSING_GENERATION`` environment variable.
    """
    validated_stage = validate_stage_tag(stage)
    event_generation = generation or os.environ.get(
        PROCESSING_GENERATION_ENV_VAR
    )

    event_log.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat(timespec='milliseconds')

    # Escape pipe delimiters in error message
    error_msg_safe = error_msg.replace("|", "\\|").replace("\n", " ")

    # Build event line — include the SLURM fields when non-empty OR when a
    # ``stage`` follows them (stage is field 8; positional parsing needs its
    # placeholder fields 6-7 present). Old 4-5 field lines and 7-field SLURM
    # lines still parse.
    parts = [timestamp, dataset, image, status, error_msg_safe]
    if (
        slurm_job_id
        or slurm_array_task_id
        or validated_stage is not None
        or event_generation
    ):
        parts.append(slurm_job_id)
        parts.append(slurm_array_task_id)
    if validated_stage is not None or event_generation:
        parts.append(validated_stage or "")
    if event_generation:
        parts.append(event_generation)
    if attempt_id or work_id:
        while len(parts) < 9:
            parts.append("")
        parts.extend([attempt_id, work_id])
    event_line = "|".join(parts) + "\n"

    try:
        atomic_append(
            event_log, event_line, timeout=60.0, commit_guard=commit_guard
        )
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
    status: ProcessingStatus,
    error_msg: str = "",
    stage: str | None = None,
    generation: str | None = None,
    commit_guard: CommitGuard | None = None,
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
        stage: Optional staged-engine stage tag (``"stage1"``/``"stage2"``/``"stage3"``).
        generation: Optional durable processing generation.
    """
    append_event(
        event_log,
        dataset,
        image,
        status,
        error_msg=error_msg,
        stage=stage,
        generation=generation,
        commit_guard=commit_guard,
    )


def parse_event_line(line: str) -> ProcessingEvent:
    """
    Parse a single event log line into a ProcessingEvent.

    Supports the old 4-5 field format, the 7-field format with SLURM fields,
    the 8-field staged format (field 8 = ``stage``), and the 9-field
    generation format. Missing trailing fields default to empty / ``None``.

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

    timestamp_str, dataset, image, status_raw = parts[:4]
    error_msg = parts[4] if len(parts) > 4 else ""
    slurm_job_id = parts[5] if len(parts) > 5 else ""
    slurm_array_task_id = parts[6] if len(parts) > 6 else ""
    stage = validate_stage_tag(parts[7] if len(parts) > 7 and parts[7] else None)
    generation = parts[8] if len(parts) > 8 and parts[8] else None

    # Validate + narrow to ProcessingStatus literal
    if status_raw not in ("started", "completed", "failed"):
        raise ValueError(
            f"Invalid status value: '{status_raw}' "
            f"(expected 'started', 'completed', or 'failed')"
        )
    status: ProcessingStatus = status_raw  # type: ignore[assignment]

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
        stage=stage,
        generation=generation,
    )


def aggregate_state_from_events(
    event_log: Path,
    *,
    inventory: Mapping[str, Collection[str]] | None = None,
    generation: str | None = None,
) -> Dict[str, DatasetState]:
    """
    Read event log and build complete processing state.

    Uses file locking to ensure consistent reads during parallel execution.
    Processes events in order, allowing retries to override previous failures.
    The most recent status for each image is used.

    Args:
        event_log: Path to processing_events.log file.
        inventory: Authorized image names for the current generation.
        generation: Current durable processing generation. Tagged events from
            other generations are ignored. Legacy generationless events remain
            eligible and are fenced by ``inventory``.

    Returns:
        Dictionary mapping dataset names to their current state
    """
    return aggregate_state_from_events_with_diagnostics(
        event_log,
        inventory=inventory,
        generation=generation,
    ).datasets


def aggregate_state_from_events_with_diagnostics(
    event_log: Path,
    *,
    inventory: Mapping[str, Collection[str]] | None = None,
    generation: str | None = None,
) -> EventAggregation:
    """Aggregate latest per-image status and report excluded log records."""
    authorized = (
        {dataset: frozenset(images) for dataset, images in inventory.items()}
        if inventory is not None
        else None
    )

    def _parse_event_log(content: str) -> EventAggregation:
        """Inner parser function that processes event log content."""
        datasets: Dict[str, DatasetState] = {}
        malformed_events = 0
        unknown_image_events = 0
        other_generation_events = 0
        samples: list[str] = []

        if not content:
            return EventAggregation(
                datasets=datasets,
                diagnostics=EventAggregationDiagnostics(),
            )

        for line_num, line in enumerate(content.splitlines(), 1):
            if not line.strip():
                continue

            try:
                event = parse_event_line(line)
            except ValueError as e:
                malformed_events += 1
                logger.debug(
                    f"Skipping malformed line {line_num}: {e}"
                )
                if len(samples) < _DIAGNOSTIC_SAMPLE_LIMIT:
                    samples.append(f"malformed line {line_num}")
                continue
            if (
                generation is not None
                and event.generation is not None
                and event.generation != generation
            ):
                other_generation_events += 1
                if len(samples) < _DIAGNOSTIC_SAMPLE_LIMIT:
                    samples.append(
                        f"other generation: {event.dataset}/{event.image}"
                    )
                continue
            if authorized is not None and event.image not in authorized.get(
                event.dataset, ()
            ):
                unknown_image_events += 1
                if len(samples) < _DIAGNOSTIC_SAMPLE_LIMIT:
                    samples.append(
                        f"unknown image: {event.dataset}/{event.image}"
                    )
                continue

            # Initialize dataset state if needed
            if event.dataset not in datasets:
                datasets[event.dataset] = DatasetState()

            ds = datasets[event.dataset]

            # Staged GPU events carry a ``stage`` tag. Overall completion
            # requires the TERMINAL stage (stage3): an intermediate-stage
            # completion means the image is still in progress, not done — so it
            # is counted as ``started`` for the overall view (and cleared from
            # ``failed`` on a retry-success). Legacy events (stage is None)
            # keep the original semantics exactly.
            intermediate_stage = (
                event.stage is not None
                and event.stage != STAGED_TERMINAL_STAGE
            )

            # Update state based on event
            if event.status == "started":
                ds.started.add(event.image)
                ds.completed.discard(event.image)
                ds.failed.discard(event.image)
                ds.errors.pop(event.image, None)
            elif event.status == "completed":
                if intermediate_stage:
                    ds.started.add(event.image)  # progressing, not done
                    ds.completed.discard(event.image)
                else:
                    ds.completed.add(event.image)
                    ds.started.discard(event.image)
                # Remove from failed if it was previously failed (retry success)
                ds.failed.discard(event.image)
                # Remove error if present
                ds.errors.pop(event.image, None)
            elif event.status == "failed":
                ds.failed.add(event.image)
                ds.started.discard(event.image)
                # Remove from completed if it was previously completed (shouldn't happen)
                ds.completed.discard(event.image)
                # Store error message
                if event.error_msg:
                    ds.errors[event.image] = event.error_msg

        return EventAggregation(
            datasets=datasets,
            diagnostics=EventAggregationDiagnostics(
                malformed_events=malformed_events,
                unknown_image_events=unknown_image_events,
                other_generation_events=other_generation_events,
                samples=tuple(samples),
            ),
        )

    # Use atomic read with file locking
    try:
        return atomic_read(event_log, _parse_event_log, timeout=60.0)
    except FileLockTimeout as e:
        logger.error(f"Failed to acquire event log lock for reading after 60s: {e}")
        raise RuntimeError(
            "Cannot read event log - file lock timeout after 60s. "
            "Check filesystem performance."
        ) from e


def aggregate_stage_state_from_events(
    event_log: Path,
) -> Dict[str, Dict[str, DatasetState]]:
    """Per-(dataset, stage) processing state for the staged GPU engine (OQ5).

    Returns ``{dataset: {stage: DatasetState}}`` where *stage* is one of
    ``"stage1"``/``"stage2"``/``"stage3"``. Only stage-tagged events contribute;
    legacy (``stage is None``) events are ignored. This is additive — the
    overall per-dataset view from :func:`aggregate_state_from_events` is
    unchanged — and lets a dashboard show how far each image has progressed
    through the three stages.

    Args:
        event_log: Path to processing_events.log file.

    Returns:
        Mapping of dataset name to a mapping of stage tag to its DatasetState.
    """
    def _parse_stage_event_log(content: str) -> Dict[str, Dict[str, DatasetState]]:
        out: Dict[str, Dict[str, DatasetState]] = {}
        if not content:
            return out

        for line_num, line in enumerate(content.splitlines(), 1):
            if not line.strip():
                continue
            try:
                event = parse_event_line(line)
            except ValueError as e:
                logger.debug(f"Skipping malformed line {line_num}: {e}")
                continue
            if event.stage is None:
                continue

            ds = out.setdefault(event.dataset, {}).setdefault(
                event.stage, DatasetState()
            )
            if event.status == "started":
                ds.started.add(event.image)
            elif event.status == "completed":
                ds.completed.add(event.image)
                ds.failed.discard(event.image)
                ds.errors.pop(event.image, None)
            elif event.status == "failed":
                ds.failed.add(event.image)
                ds.completed.discard(event.image)
                if event.error_msg:
                    ds.errors[event.image] = event.error_msg
        return out

    try:
        return atomic_read(event_log, _parse_stage_event_log, timeout=60.0)
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
def main(event_log: Path, dataset: str, image: str, status: ProcessingStatus, error: str):
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
