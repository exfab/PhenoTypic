"""
Pipeline validation for the PhenoTypic CLI.

This module provides validation functions to check pipeline configuration
before running large batch processing jobs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass

from phenotypic import ImagePipeline
from ._cli_types import ExecutionConfig


def validate_pipeline(
    pipeline_path: Path,
    skip_validation: bool = False
) -> Tuple[bool, Optional[str]]:
    """
    Validate that pipeline JSON can be loaded successfully.
    
    Args:
        pipeline_path: Path to pipeline JSON file
        skip_validation: If True, skip validation (for advanced users)
        
    Returns:
        Tuple of (is_valid, error_message)
        If valid, error_message is None
    """
    if skip_validation:
        return True, None
    
    try:
        # Try to load pipeline
        pipeline = ImagePipeline.from_json(pipeline_path)
        
        # Check that pipeline has operations or measurements
        if not pipeline._ops and not pipeline._meas:
            return False, "Pipeline has no operations or measurements"
        
        return True, None
        
    except FileNotFoundError:
        return False, f"Pipeline file not found: {pipeline_path}"
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON in pipeline file: {e}"
    except Exception as e:
        return False, f"Failed to load pipeline: {type(e).__name__}: {e}"


def validate_execution_config(
    config: ExecutionConfig
) -> Tuple[bool, Optional[str]]:
    """
    Validate execution configuration for obvious errors.
    
    Args:
        config: Execution configuration to validate
        
    Returns:
        Tuple of (is_valid, error_message)
        If valid, error_message is None
    """
    # Check pipeline file exists
    if not config.pipeline_json.exists():
        return False, f"Pipeline file not found: {config.pipeline_json}"
    
    # Check input path exists
    if not config.input_path.exists():
        return False, f"Input path not found: {config.input_path}"
    
    # Check grid dimensions for GridImage. ``None`` means "no CLI override —
    # fall back to the pipeline preset / built-in default at resolve time"
    # and is valid here; only explicit non-positive values are rejected.
    if config.image_type == "GridImage":
        if config.nrows is not None and config.nrows <= 0:
            return False, f"Invalid nrows: {config.nrows} (must be positive)"
        if config.ncols is not None and config.ncols <= 0:
            return False, f"Invalid ncols: {config.ncols} (must be positive)"
    
    # Check n_jobs is valid
    if config.n_jobs == 0:
        return False, "n_jobs cannot be 0 (use -1 for all cores or positive integer)"
    
    # Check SLURM parameters if provided
    if config.slurm_args:
        # Warn about common missing parameters
        required_slurm_params = ["slurm_partition"]
        missing = [p for p in required_slurm_params if p not in config.slurm_args]
        if missing:
            # This is a warning, not an error - let SLURM handle it
            pass
    
    return True, None


def full_validation(
    config: ExecutionConfig,
) -> Tuple[bool, list[str]]:
    """
    Validate execution configuration and pipeline loading.

    Args:
        config: Execution configuration.

    Returns:
        Tuple of (is_valid, list_of_errors).
        If valid, list_of_errors is empty.
    """
    errors = []

    # Validate config
    config_valid, config_error = validate_execution_config(config)
    if not config_valid:
        errors.append(config_error)
        return False, errors

    # Validate pipeline can be loaded
    pipeline_valid, pipeline_error = validate_pipeline(
        config.pipeline_json,
        config.skip_validation
    )
    if not pipeline_valid:
        errors.append(pipeline_error)

    return len(errors) == 0, errors


def pipeline_requires_gpu(pipeline_path: Path) -> bool:
    """Check if a pipeline JSON contains any GpuDetector operations.

    Args:
        pipeline_path: Path to pipeline JSON file.

    Returns:
        True if the pipeline contains at least one GpuDetector operation.
    """
    from phenotypic.abc_ import GpuDetector

    pipeline = ImagePipeline.from_json(pipeline_path)
    return any(isinstance(op, GpuDetector) for op in pipeline.get_ops().values())
