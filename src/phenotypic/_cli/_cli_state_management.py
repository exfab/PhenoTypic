"""
Processing state management for the PhenoTypic CLI.

This module handles loading, saving, and updating the processing state
for resume capability. Uses append-only event log with periodic aggregation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, List
from datetime import datetime

from ._cli_types import ProcessingState, DatasetState, Dataset, ExecutionConfig
from ._cli_update_state import aggregate_state_from_events
from phenotypic.sdk_ import (
    ProcessingStateKey,
    migrate_legacy_machine_state,
    processing_state_path,
    resolve_event_log_path,
    resolve_processing_state_path,
)


def save_processing_state(
    state: ProcessingState,
    output_dir: Path
) -> Path:
    """
    Save processing state to JSON file.
    
    This aggregates events from the event log and creates a human-readable
    JSON snapshot of the current processing state.
    
    Args:
        state: ProcessingState object to save
        output_dir: Output directory containing processing_state.json
        
    Returns:
        Path to saved state file
    """
    state_file = processing_state_path(output_dir)
    state_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert state to dictionary
    dataset_entries: dict[str, dict[str, Any]] = {}
    state_dict: dict[str, Any] = {
        ProcessingStateKey.VERSION: state.version,
        ProcessingStateKey.PIPELINE_PATH: str(state.pipeline_path),
        ProcessingStateKey.INPUT_PATH: str(state.input_path),
        ProcessingStateKey.OUTPUT_DIR: str(state.output_dir),
        ProcessingStateKey.TIMESTAMP: state.timestamp.isoformat(),
        ProcessingStateKey.EXECUTION_MODE: state.execution_mode,
        ProcessingStateKey.LAST_UPDATED: state.last_updated.isoformat(),
        ProcessingStateKey.DATASETS: dataset_entries,
        ProcessingStateKey.CONFIG: state.config
    }

    # Add dataset states
    for dataset_name, ds_state in state.datasets.items():
        dataset_entries[dataset_name] = {
            ProcessingStateKey.COMPLETED: list(ds_state.completed),
            ProcessingStateKey.FAILED: list(ds_state.failed),
            ProcessingStateKey.ERRORS: ds_state.errors,
            ProcessingStateKey.INITIAL_IMAGES: list(ds_state.initial_images)
        }
    
    # Write atomically (temp file + rename)
    temp_file = state_file.with_suffix('.tmp')
    temp_file.write_text(json.dumps(state_dict, indent=2), encoding="utf-8")
    temp_file.replace(state_file)
    
    return state_file


def load_processing_state(output_dir: Path) -> Optional[ProcessingState]:
    """
    Load processing state from JSON file and aggregate latest events.
    
    Args:
        output_dir: Output directory containing processing_state.json
        
    Returns:
        ProcessingState object, or None if no state file exists
    """
    migrate_legacy_machine_state(output_dir)
    state_file = resolve_processing_state_path(output_dir)

    if not state_file.exists():
        return None

    # Load state JSON
    state_dict = json.loads(state_file.read_text(encoding="utf-8"))

    # Parse timestamps
    timestamp = datetime.fromisoformat(state_dict[ProcessingStateKey.TIMESTAMP])
    last_updated = datetime.fromisoformat(state_dict[ProcessingStateKey.LAST_UPDATED])

    # Aggregate latest events from log (sibling of progress/, per D14)
    event_log = resolve_event_log_path(output_dir)
    if event_log.exists():
        latest_states = aggregate_state_from_events(event_log)
    else:
        latest_states = {}

    # Merge with stored state (prefer event log as source of truth)
    datasets = {}
    for dataset_name in state_dict[ProcessingStateKey.DATASETS].keys():
        ds_dict = state_dict[ProcessingStateKey.DATASETS][dataset_name]
        if dataset_name in latest_states:
            # Use aggregated state from event log, but preserve initial_images from stored state
            event_state = latest_states[dataset_name]
            event_state.initial_images = set(ds_dict.get(ProcessingStateKey.INITIAL_IMAGES, []))
            datasets[dataset_name] = event_state
        else:
            # Fallback to stored state
            ds_dict = state_dict[ProcessingStateKey.DATASETS][dataset_name]
            datasets[dataset_name] = DatasetState(
                completed=set(ds_dict.get(ProcessingStateKey.COMPLETED, [])),
                failed=set(ds_dict.get(ProcessingStateKey.FAILED, [])),
                errors=ds_dict.get(ProcessingStateKey.ERRORS, {}),
                initial_images=set(ds_dict.get(ProcessingStateKey.INITIAL_IMAGES, []))
            )

    # Create ProcessingState object
    state = ProcessingState(
        version=state_dict[ProcessingStateKey.VERSION],
        pipeline_path=Path(state_dict[ProcessingStateKey.PIPELINE_PATH]),
        input_path=Path(state_dict[ProcessingStateKey.INPUT_PATH]),
        output_dir=Path(state_dict[ProcessingStateKey.OUTPUT_DIR]),
        timestamp=timestamp,
        execution_mode=state_dict[ProcessingStateKey.EXECUTION_MODE],
        last_updated=last_updated,
        datasets=datasets,
        config=state_dict[ProcessingStateKey.CONFIG]
    )
    
    return state


def create_initial_state(
    config: ExecutionConfig,
    datasets: List[Dataset],
    output_dir: Path
) -> ProcessingState:
    """
    Create initial processing state for a new run.
    
    Args:
        config: Execution configuration
        datasets: List of datasets to process
        output_dir: Output directory
        
    Returns:
        New ProcessingState object
    """
    # Initialize dataset states with initial image list
    dataset_states = {}
    for dataset in datasets:
        initial_images = {img.name for img in dataset.images}
        dataset_states[dataset.name] = DatasetState(initial_images=initial_images)
    
    # Create state object
    state = ProcessingState(
        version="2.0.0",
        pipeline_path=config.pipeline_json,
        input_path=config.input_path,
        output_dir=output_dir,
        timestamp=datetime.now(),
        execution_mode="slurm" if config.is_slurm_mode() else "local",
        last_updated=datetime.now(),
        datasets=dataset_states,
        config={
            "image_type": config.image_type,
            "nrows": config.nrows,
            "ncols": config.ncols,
            "bit_depth": config.bit_depth,
            "n_jobs": config.n_jobs,
            "slurm_args": config.slurm_args,
            "ext": config.ext,
            "process_only_layer": config.process_only_layer,
        }
    )
    
    return state


def update_state_from_events(state: ProcessingState, output_dir: Path) -> ProcessingState:
    """
    Update processing state by aggregating events from event log.
    
    Args:
        state: Current processing state
        output_dir: Output directory containing event log

    Returns:
        Updated ProcessingState object
    """
    event_log = resolve_event_log_path(output_dir)
    
    if event_log.exists():
        # Aggregate events
        latest_states = aggregate_state_from_events(event_log)
        
        # Update dataset states
        for dataset_name, new_state in latest_states.items():
            state.datasets[dataset_name] = new_state
    
    # Update last_updated timestamp
    state.last_updated = datetime.now()
    
    return state


def validate_resume_compatibility(
    state: ProcessingState,
    config: ExecutionConfig
) -> tuple[bool, Optional[str]]:
    """
    Check if current config is compatible with saved state for resume.
    
    Args:
        state: Saved processing state
        config: Current execution config
        
    Returns:
        Tuple of (is_compatible, error_message)
        If compatible, error_message is None
    """
    # Check pipeline path
    if state.pipeline_path != config.pipeline_json:
        return False, f"Pipeline mismatch: saved={state.pipeline_path}, current={config.pipeline_json}"
    
    # Check input path
    if state.input_path != config.input_path:
        return False, f"Input path mismatch: saved={state.input_path}, current={config.input_path}"
    
    # Check image type
    if state.config.get("image_type") != config.image_type:
        return False, f"Image type mismatch: saved={state.config.get('image_type')}, current={config.image_type}"
    
    # Check grid dimensions for GridImage
    if config.image_type == "GridImage":
        if state.config.get("nrows") != config.nrows:
            return False, f"Grid rows mismatch: saved={state.config.get('nrows')}, current={config.nrows}"
        if state.config.get("ncols") != config.ncols:
            return False, f"Grid cols mismatch: saved={state.config.get('ncols')}, current={config.ncols}"

    # Check process-only layer. Process mode writes one layer to the mirrored
    # image path, so changing layers during resume would mix file semantics.
    saved_process_only_layer = state.config.get("process_only_layer")
    if saved_process_only_layer != config.process_only_layer:
        return (
            False,
            "Process-only layer mismatch: "
            f"saved={saved_process_only_layer}, current={config.process_only_layer}",
        )

    return True, None


def get_remaining_images_for_datasets(
    state: ProcessingState,
    datasets: List[Dataset],
    retry_failures: bool = False
) -> List[Dataset]:
    """
    Get datasets with only remaining (unprocessed) images for resume.
    
    Args:
        state: Current processing state
        datasets: Full list of datasets
        retry_failures: If True, include failed images in remaining
        
    Returns:
        List of Dataset objects with only unprocessed images
    """
    remaining_datasets = []
    
    for dataset in datasets:
        if dataset.name not in state.datasets:
            # New dataset not in state - include all images
            remaining_datasets.append(dataset)
            continue
        
        ds_state = state.datasets[dataset.name]
        
        # Determine which images to process
        all_image_names = {img.name for img in dataset.images}
        processed = ds_state.completed
        
        if not retry_failures:
            # Skip failed images
            processed = processed | ds_state.failed
        
        remaining_names = all_image_names - processed
        
        if not remaining_names:
            # No images remaining in this dataset
            continue
        
        # Create new Dataset with only remaining images
        remaining_images = [
            img for img in dataset.images
            if img.name in remaining_names
        ]
        
        remaining_dataset = Dataset(
            name=dataset.name,
            images=remaining_images,
            input_dir=dataset.input_dir,
            output_dir=dataset.output_dir
        )
        remaining_datasets.append(remaining_dataset)
    
    return remaining_datasets
