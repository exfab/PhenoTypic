"""
State file updater for the PhenoTypic CLI.

This module handles append-only event logging for tracking image processing
completion status. Uses atomic append operations for HPC filesystem safety.
"""

from __future__ import annotations

import click
from pathlib import Path
from typing import Literal, Dict, Set
from datetime import datetime
from dataclasses import dataclass

from ._cli_types import DatasetState


@dataclass
class ProcessingEvent:
    """Single processing completion event."""
    timestamp: datetime
    dataset: str
    image: str
    status: Literal["completed", "failed"]
    error_msg: str = ""


def append_completion_event(
    event_log: Path,
    dataset: str,
    image: str,
    status: Literal["completed", "failed"],
    error_msg: str = ""
) -> None:
    """
    Atomically append completion event to processing log.
    
    Event format: timestamp|dataset|image|status|error_msg
    
    This operation is thread-safe on most HPC filesystems (NFS, Lustre)
    due to using O_APPEND mode which makes small writes atomic.
    
    Args:
        event_log: Path to processing_events.log file
        dataset: Dataset name
        image: Image filename
        status: "completed" or "failed"
        error_msg: Error message if status is "failed"
    """
    # Create parent directory if needed
    event_log.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().isoformat(timespec='milliseconds')
    
    # Escape pipe delimiters in error message
    error_msg_safe = error_msg.replace("|", "\\|").replace("\n", " ")
    
    # Format event line
    event_line = f"{timestamp}|{dataset}|{image}|{status}|{error_msg_safe}\n"
    
    # Append with 'a' mode which uses O_APPEND for atomic writes
    with open(event_log, 'a', encoding='utf-8') as f:
        f.write(event_line)


def parse_event_line(line: str) -> ProcessingEvent:
    """
    Parse a single event log line into a ProcessingEvent.
    
    Args:
        line: Raw line from event log
        
    Returns:
        ProcessingEvent object
        
    Raises:
        ValueError: If line format is invalid
    """
    line = line.strip()
    if not line:
        raise ValueError("Empty line")
    
    parts = line.split('|')
    if len(parts) < 4:
        raise ValueError(f"Invalid line format: {line}")
    
    timestamp_str, dataset, image, status = parts[:4]
    error_msg = parts[4] if len(parts) > 4 else ""
    
    # Unescape error message
    error_msg = error_msg.replace("\\|", "|")
    
    # Parse timestamp
    try:
        timestamp = datetime.fromisoformat(timestamp_str)
    except ValueError:
        # Fallback for older timestamp formats
        timestamp = datetime.now()
    
    return ProcessingEvent(
        timestamp=timestamp,
        dataset=dataset,
        image=image,
        status=status,
        error_msg=error_msg
    )


def aggregate_state_from_events(event_log: Path) -> Dict[str, DatasetState]:
    """
    Read event log and build complete processing state.
    
    Processes events in order, allowing retries to override previous failures.
    The most recent status for each image is used.
    
    Args:
        event_log: Path to processing_events.log file
        
    Returns:
        Dictionary mapping dataset names to their current state
    """
    datasets = {}
    
    if not event_log.exists():
        return datasets
    
    # Read all events
    with open(event_log, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                event = parse_event_line(line)
            except ValueError:
                # Skip malformed lines
                continue
            
            # Initialize dataset state if needed
            if event.dataset not in datasets:
                datasets[event.dataset] = DatasetState()
            
            ds = datasets[event.dataset]
            
            # Update state based on event
            if event.status == "completed":
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
@click.option("--status", type=click.Choice(["completed", "failed"]), required=True,
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
