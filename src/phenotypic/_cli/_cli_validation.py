"""
Pipeline validation for the PhenoTypic CLI.

This module provides validation functions to check pipeline configuration
before running large batch processing jobs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image

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


def validate_pipeline_on_test_image(
    pipeline_path: Path,
    test_image_path: Path,
    image_cls: type,
    read_kwargs: Dict[str, Any],
    skip_validation: bool = False
) -> Tuple[bool, Optional[str]]:
    """
    Validate pipeline by running it on a single test image.
    
    This catches errors that might not be apparent from just loading
    the pipeline JSON (e.g., missing required operations, incompatible
    parameters, etc.).
    
    Args:
        pipeline_path: Path to pipeline JSON file
        test_image_path: Path to test image
        image_cls: Image class to use (Image or GridImage)
        read_kwargs: Kwargs for imread
        skip_validation: If True, skip validation
        
    Returns:
        Tuple of (is_valid, error_message)
        If valid, error_message is None
    """
    if skip_validation:
        return True, None
    
    try:
        # Load pipeline
        pipeline = ImagePipeline.from_json(pipeline_path)
        
        # Load test image
        image = image_cls.imread(test_image_path, **read_kwargs)
        
        # Try to run pipeline
        measurements = pipeline.apply_and_measure(image, inplace=True)
        
        # Check that measurements were produced
        if measurements is None or len(measurements) == 0:
            return False, "Pipeline produced no measurements"
        
        return True, None
        
    except Exception as e:
        return False, f"Pipeline failed on test image: {type(e).__name__}: {e}"


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
    
    # Check grid dimensions for GridImage
    if config.image_type == "GridImage":
        if config.nrows <= 0:
            return False, f"Invalid nrows: {config.nrows} (must be positive)"
        if config.ncols <= 0:
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
    datasets = None
) -> Tuple[bool, list[str]]:
    """
    Perform comprehensive validation of configuration and pipeline.
    
    Args:
        config: Execution configuration
        datasets: Optional list of Dataset objects for finding test images
            If None, validation will be limited to config and pipeline loading
            
    Returns:
        Tuple of (is_valid, list_of_errors)
        If valid, list_of_errors is empty
    """
    errors = []
    
    # Validate config
    config_valid, config_error = validate_execution_config(config)
    if not config_valid:
        errors.append(config_error)
        return False, errors  # Can't continue without valid config
    
    # Validate pipeline can be loaded
    pipeline_valid, pipeline_error = validate_pipeline(
        config.pipeline_json,
        config.skip_validation
    )
    if not pipeline_valid:
        errors.append(pipeline_error)
        return False, errors  # Can't test on image without valid pipeline
    
    # Test pipeline on image if possible
    test_image_path = None
    if datasets is not None and not config.skip_validation:
        # Try to find a test image from datasets
        try:
            for dataset in datasets:
                if dataset.images:
                    test_image_path = dataset.images[0]
                    break
        except Exception:
            # Can't find test image - skip this validation
            pass
    
    if test_image_path is not None and not config.skip_validation:
        # Determine image class
        from phenotypic import Image, GridImage
        image_cls = GridImage if config.image_type == "GridImage" else Image
        
        # Prepare read kwargs
        read_kwargs = {}
        if config.image_type == "GridImage":
            read_kwargs["nrows"] = config.nrows
            read_kwargs["ncols"] = config.ncols
        if config.bit_depth is not None:
            read_kwargs["bit_depth"] = config.bit_depth
        
        # Validate on test image
        test_valid, test_error = validate_pipeline_on_test_image(
            config.pipeline_json,
            test_image_path,
            image_cls,
            read_kwargs,
            config.skip_validation
        )
        
        if not test_valid:
            errors.append(test_error)

    return len(errors) == 0, errors
