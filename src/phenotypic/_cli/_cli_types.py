"""
Type definitions for the PhenoTypic CLI.

This module contains all dataclasses and type definitions used throughout
the CLI implementation for clean type hints and structured data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from datetime import datetime

from phenotypic.sdk_.typing_ import ExecutionMode, ImageTypeName, ProcessOnlyLayer


@dataclass
class Dataset:
    """Represents a collection of images to process (e.g., from a subdirectory)."""
    name: str  # Dataset name ("single_image" for single files, directory name for flat dirs, or subdirectory name)
    images: List[Path]  # List of image file paths
    input_dir: Path  # Source directory for this dataset
    output_dir: Path  # Output directory for this dataset's results


@dataclass
class DatasetState:
    """Processing state for a single dataset."""
    completed: Set[str] = field(default_factory=set)  # Completed image filenames
    failed: Set[str] = field(default_factory=set)  # Failed image filenames
    started: Set[str] = field(default_factory=set)  # Started image filenames
    errors: Dict[str, str] = field(default_factory=dict)  # filename -> error message
    initial_images: Set[str] = field(default_factory=set)  # Initial image set for resume validation

    @property
    def in_progress(self) -> Set[str]:
        """Images that have started but not yet completed or failed."""
        return self.started - self.completed - self.failed

    @property
    def total_processed(self) -> int:
        """Total number of images processed (completed + failed)."""
        return len(self.completed) + len(self.failed)

    @property
    def success_rate(self) -> float:
        """Success rate as a fraction (0.0 to 1.0)."""
        total = self.total_processed
        if total == 0:
            return 0.0
        return len(self.completed) / total


@dataclass
class ProcessingState:
    """Complete processing state for resume capability."""
    version: str  # State file format version
    pipeline_path: Path  # Path to pipeline JSON
    input_path: Path  # Input directory or file
    output_dir: Path  # Output directory
    timestamp: datetime  # Processing start time
    execution_mode: ExecutionMode  # Execution mode
    last_updated: datetime  # Last state update time
    datasets: Dict[str, DatasetState]  # dataset_name -> state
    config: Dict[str, Any]  # Configuration used (for compatibility checking)
    
    def total_images(self) -> int:
        """Total number of images across all datasets."""
        return sum(ds.total_processed for ds in self.datasets.values())
    
    def completed_images(self) -> int:
        """Total completed images across all datasets."""
        return sum(len(ds.completed) for ds in self.datasets.values())
    
    def failed_images(self) -> int:
        """Total failed images across all datasets."""
        return sum(len(ds.failed) for ds in self.datasets.values())


@dataclass
class ExecutionConfig:
    """Complete configuration for CLI execution."""
    # Core paths
    pipeline_json: Path
    input_path: Path
    output_dir: Optional[Path]
    
    # Image configuration. ``nrows``/``ncols`` are optional CLI overrides:
    # ``None`` means "no explicit CLI value, fall back to the pipeline's soft
    # preset or the built-in default at image-load time".
    image_type: ImageTypeName
    nrows: Optional[int]
    ncols: Optional[int]
    bit_depth: Optional[int]

    # Execution mode
    n_jobs: int
    slurm_args: Dict[str, Any]
    force_local: bool
    wait: bool  # Wait for SLURM jobs to complete
    
    # Output options
    ext: str  # Extension for overlay PNG / legacy call sites (no longer the forward-run switch)
    overlay_alpha: float  # Alpha for overlay compositing

    # Processing options
    include_dataset_column: bool
    dry_run: bool
    sample: Optional[int]
    resume: bool
    retry_failures: bool
    skip_validation: bool

    # Metadata join
    metadata_csv: Optional[Path] = None

    # Checkpoint interval for SLURM array jobs
    checkpoint_interval: Optional[int] = None

    # Detection mode (default: gray)
    detect_mode: str = "gray"

    # Overlay PNG output is always-on for forward runs; measure-mode runs
    # never regenerate overlays regardless of this flag.
    save_overlays: bool = True

    # Opt-in saved inspect figures for measurers with .inspect() methods.
    # See OutputManager.save_inspect and the MeasureFeatures ABC docstring
    # for the duck-typed protocol contract.
    save_inspects: bool = False

    # Measure-only mode: reload HDFs and rerun pipeline.measure() without detection
    measure_only: bool = False

    # Process-only mode: run pipeline.apply() and export a single image layer
    # (no measurement / analysis output). None = normal forward/measure run.
    process_only_layer: Optional[ProcessOnlyLayer] = None

    # --- Staged GPU detection (Spec 1 §7/§10) ---------------------------------
    # Model replicas packed per physical GPU (Stage 2 fill for small models).
    gpu_workers_per_gpu: int = 1
    # Parallel Stage-2 GPU tasks (one whole GPU each; SLURM-only, ignored local).
    gpu_shards: int = 1
    # Stage-2 GPU SBATCH resources; inherits/deltas over slurm_args (the CPU
    # profile used by Stages 1 & 3).
    gpu_slurm_args: Dict[str, Any] = field(default_factory=dict)

    def is_slurm_mode(self) -> bool:
        """Check if SLURM mode should be used."""
        if self.force_local:
            return False
        return bool(self.slurm_args)


@dataclass
class ImageFailure:
    """Details of a failed image processing attempt."""
    dataset: str
    image_filename: str
    error_type: str  # Exception class name
    error_message: str  # Short error message
    traceback: str  # Full traceback
    timestamp: datetime


@dataclass
class DatasetResults:
    """Processing results for a single dataset."""
    name: str
    total: int
    completed: int
    failed: int
    failures: List[ImageFailure]
    processing_time: Optional[float] = None  # Total time in seconds


@dataclass
class ExecutionResults:
    """Complete results from CLI execution."""
    datasets: Dict[str, DatasetResults]
    total_images: int
    total_completed: int
    total_failed: int
    execution_mode: ExecutionMode
    start_time: datetime
    end_time: datetime
    
    @property
    def duration(self) -> float:
        """Total execution time in seconds."""
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def success_rate(self) -> float:
        """Overall success rate as a fraction (0.0 to 1.0)."""
        if self.total_images == 0:
            return 0.0
        return self.total_completed / self.total_images
