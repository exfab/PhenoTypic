"""Shared utilities for pipeline grid search operations.

This module contains core data structures, validation functions, I/O operations,
backend abstractions, trie algorithms, and HTML viewer generation used by both
PipelineGridSearchBase and MultiPipelineGridSearch.
"""

from __future__ import annotations

import copy
import gc
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import submitit

if TYPE_CHECKING:
    from phenotypic import Image, ImageOperation

logger = logging.getLogger(__name__)

# Batch size constraints
_MIN_BATCH_SIZE = 2  # Minimum pipelines per batch
_MAX_SAFE_BATCH_SIZE = 8  # Conservative default max for memory safety
_IDEAL_BATCH_MULTIPLIER = 2  # Multiply jobs_per_batch by this for ideal size

# Memory safety factors
_MEMORY_SAFETY_FACTOR = 0.90  # Use 75% of available memory for safety

# Memory estimation overhead factor
# Conservative 50% overhead accounts for:
# - Python object allocation metadata (10-15%)
# - Garbage collector structures (5-10%)
# - Temporary arrays during operations (15-20%)
# - NumPy view/slice overhead (5-10%)
# - Safety margin for heavy operations (BM3D, wavelets, morphological pipe_cfgs)
# Using 1.5 factor prevents OOM errors across diverse operation types
_MEMORY_OVERHEAD_FACTOR = 1.5

# Fallback memory estimate when calculation fails (100 MB)
_DEFAULT_FALLBACK_MEMORY_MB = 100

# HTML generation
_THUMBNAIL_SIZE = (200, 200)  # Thumbnail dimensions for trial view
_THUMBNAIL_JPEG_QUALITY = 100  # JPEG quality (0-100) - max quality for scientific imaging


def _get_memory_usage() -> float:
    """Get current process memory usage in MB.

    Returns:
        Memory usage of current process in MB

    Note:
        On macOS, this may trigger a harmless "MallocStackLogging" warning from
        the system malloc library. This warning does not affect functionality and
        can be safely ignored. To suppress it, set: export MallocStackLogging=0
    """
    import psutil

    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def _generate_pipeline_code(index: int) -> str:
    """Generate sequential pipeline code.

    Args:
        index: Zero-based index for the pipeline

    Returns:
        Pipeline code string in format "pipeline_XXX" (e.g., "pipeline_001")

    Example:
        >>> _generate_pipeline_code(0)
        'pipeline_001'
        >>> _generate_pipeline_code(42)
        'pipeline_043'
    """
    return f"pipeline_{index + 1:03d}"


def _create_manifest_json(
        output_dir: Union[str, Path],
        configs_dict: Dict[str, str],
        data_layers: List[str],
) -> Path:
    """Create manifest JSON mapping pipeline codes to configurations.

    Args:
        output_dir: Output directory path
        configs_dict: Dictionary mapping pipeline codes to JSON config strings
        data_layers: List of data layers that were saved

    Returns:
        Path to created manifest.json file

    Raises:
        RuntimeError: If manifest creation fails

    Example:
        >>> configs = {
        ...     "pipeline_001": '{"pipe_cfgs": [{"class": "GaussianBlur", "params": {"sigma": 2.0}}]}',
        ...     "pipeline_002": '{"pipe_cfgs": [{"class": "GaussianBlur", "params": {"sigma": 3.0}}]}'
        ... }
        >>> manifest_path = _create_manifest_json("/tmp/results", configs, ["rgb", "gray"])
    """
    import json
    from datetime import datetime

    manifest_path = Path(output_dir) / "manifest.json"

    try:
        # Build manifest structure
        manifest = {
            "generated_at"   : datetime.now().isoformat(),
            "total_pipelines": len(configs_dict),
            "DataAccessors"  : data_layers,
            "pipelines"      : {}
        }

        # Add each pipeline entry
        for code, json_config_str in configs_dict.items():
            # Parse JSON config to extract metadata
            try:
                config_obj = json.loads(json_config_str)
            except json.JSONDecodeError:
                logger.warning(
                        f"Failed to parse JSON config for {code}, storing as string")
                config_obj = {"raw": json_config_str}

            manifest["pipelines"][code] = {
                "config"   : config_obj,
                "directory": code,
                "layers"   : data_layers
            }

        # Write manifest
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Created manifest: {manifest_path}")
        return manifest_path

    except Exception as e:
        raise RuntimeError(f"Failed to create manifest.json: {e}") from e


def _create_output_directory_structure(
        output_dir: Union[str, Path],
) -> Dict[str, Path]:
    """Create output directory structure for pipeline grid search.

    Creates the following structure:
        output_dir/
        ├── original/       # Original input images
        └── thumbnails/     # Thumbnail images for HTML viewer

    Args:
        output_dir: Base output directory path

    Returns:
        Dictionary with paths to created subdirectories:
        - "base": Main output directory
        - "original": Original images directory
        - "thumbnails": Thumbnails directory

    Raises:
        ValueError: If directory creation fails or permissions are insufficient

    Example:
        >>> paths = _create_output_directory_structure("./results")
        >>> print(paths["original"])
        ./results/original
    """
    try:
        base_path = Path(output_dir)
        base_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        original_dir = base_path / "original"
        original_dir.mkdir(exist_ok=True)

        thumbnails_dir = base_path / "thumbnails"
        thumbnails_dir.mkdir(exist_ok=True)

        # Test write permissions
        test_file = base_path / ".write_test"
        test_file.touch()
        test_file.unlink()

        logger.info(f"Created output directory structure: {base_path}")

        return {
            "base"      : base_path,
            "original"  : original_dir,
            "thumbnails": thumbnails_dir,
        }

    except Exception as e:
        raise ValueError(
                f"Failed to create output directory structure at '{output_dir}': {e}"
        ) from e


def _save_original_images(
        image: "Image",
        original_dir: Union[str, Path],
) -> List[Path]:
    """Save original RGB and grayscale images as TIFF files.

    Args:
        image: Original input image
        original_dir: Directory to save original images

    Returns:
        List of paths to saved TIFF files

    Raises:
        RuntimeError: If saving fails

    Example:
        >>> from phenotypic import Image
        >>> image = Image.imread("test.jpg")
        >>> paths = _save_original_images(image, "./results/original")
        >>> print(paths)
        [PosixPath('./results/original/rgb.tiff'), PosixPath('./results/original/gray.tiff')]
    """
    saved_paths = []

    try:
        # Save RGB if available
        rgb_data = image.rgb[:]
        if rgb_data is not None and rgb_data.size > 0:
            rgb_path = _save_array_as_tiff(rgb_data, original_dir, "rgb")
            saved_paths.append(rgb_path)
            logger.debug(f"Saved original RGB: {rgb_path}")

        # Save grayscale
        gray_data = image.gray[:]
        if gray_data is not None and gray_data.size > 0:
            gray_path = _save_array_as_tiff(gray_data, original_dir, "gray")
            saved_paths.append(gray_path)
            logger.debug(f"Saved original grayscale: {gray_path}")

        logger.info(f"Saved {len(saved_paths)} original images to {original_dir}")
        return saved_paths

    except Exception as e:
        raise RuntimeError(f"Failed to save original images: {e}") from e


def _ops_key(op: "ImageOperation", params: Dict[str, Any]) -> Tuple:
    """Create hashable key for operation tuple comparison.

    Recursively converts unhashable parameter values (lists, dicts, numpy arrays)
    to hashable equivalents (tuples, nested tuples, flattened tuples) to enable
    use as trie node keys.

    Args:
        op: ImageOperation instance
        params: Parameter dictionary for the operation

    Returns:
        Tuple containing (class_name, sorted_params) for hashing

    Raises:
        TypeError: If a parameter value cannot be converted to a hashable type
    """

    def _make_hashable(val):
        """Recursively convert unhashable types to hashable equivalents."""
        if isinstance(val, dict):
            # Convert dict to sorted tuple of (key, value) pairs
            return tuple(
                    sorted((_make_hashable(k), _make_hashable(v)) for k, v in
                           val.items()))
        elif isinstance(val, (list, tuple)):
            # Convert sequences to tuples recursively
            return tuple(_make_hashable(v) for v in val)
        elif isinstance(val, np.ndarray):
            # Convert numpy arrays to flattened tuples
            return tuple(val.flatten().tolist())
        else:
            # Already hashable (int, float, str, bool, etc.)
            return val

    try:
        hashable_params = tuple(
                sorted((k, _make_hashable(v)) for k, v in params.items()))
        return (type(op).__name__, hashable_params)
    except TypeError as e:
        raise TypeError(
                f"Cannot create hashable key for operation {type(op).__name__} with "
                f"parameters {params}. Parameter values must be convertible to hashable types. "
                f"Error: {e}"
        ) from e


@dataclass
class _TrieNode:
    """Node in pipeline execution trie for shared prefix optimization.

    Attributes:
        op: ImageOperation at this node (None for root)
        params: Parameter dict for this operation (None for root)
        children: Child nodes keyed by operation signature
        pipeline_names: Names of pipelines that end at this node
    """

    op: Optional["ImageOperation"] = None
    params: Optional[Dict[str, Any]] = None
    children: Dict[Tuple, "_TrieNode"] = field(default_factory=dict)
    pipeline_names: List[str] = field(default_factory=list)


def _unpack_ops_tuples(
        ops: List[Tuple[ImageOperation, Dict[str, List[Any]]]],
) -> Tuple[List[ImageOperation], List[Dict[str, List[Any]]]]:
    """Unpack list of (operation, params) tuples into separate lists.

    Args:
        ops: List of (operation, params_dict) tuples

    Returns:
        Tuple of (operations_list, parameters_list)
    """
    operations = []
    parameters = []

    for op, params in ops:
        operations.append(op)
        parameters.append(params)

    return operations, parameters


def _validate_inputs(
        ops: List[Tuple[ImageOperation, Dict[str, List[Any]]]],
        data_layers: List[str],
) -> None:
    """Validate all inputs before processing.

    Args:
        ops: List of (operation, params_dict) tuples
        data_layers: List of data layer names to display

    Raises:
        ValueError: If inputs are invalid or malformed
    """
    # Validate pipe_cfgs format
    if not isinstance(ops, list):
        raise ValueError(f"pipe_cfgs must be a list, got {type(ops)}")

    if not ops:
        raise ValueError("pipe_cfgs cannot be empty")

    for idx, item in enumerate(ops):
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(
                    f"pipe_cfgs[{idx}]: Each element must be a tuple (operation, params_dict), "
                    f"got {type(item)}"
            )

        op, params = item
        if not isinstance(params, dict):
            raise ValueError(
                    f"pipe_cfgs[{idx}]: Second element of tuple must be a dict, "
                    f"got {type(params)}"
            )

    # Unpack tuples
    operations, parameters = _unpack_ops_tuples(ops)

    # Verify parameter names exist as operation attributes
    for op_idx, (op, params) in enumerate(zip(operations, parameters)):
        for param_name in params.keys():
            if not hasattr(op, param_name):
                raise ValueError(
                        f"Operation {op_idx} ({op.__class__.__name__}) has no "
                        f"attribute '{param_name}'. Available attributes: "
                        f"{[a for a in dir(op) if not a.startswith('_')]}"
                )

    # Validate DataAccessors
    valid_layers = {"rgb", "gray", "enh_gray", "objmask", "objmap"}
    invalid = set(data_layers) - valid_layers
    if invalid:
        raise ValueError(
                f"Invalid DataAccessors: {invalid}. Must be subset of {valid_layers}"
        )


def _validate_pipeline_configs(
        pipeline_configs: List[Dict[str, Any]],
) -> None:
    """Validate structure of pipeline configuration list.

    Args:
        pipeline_configs: List of pipeline configuration dictionaries

    Raises:
        ValueError: If configs are invalid or malformed
    """
    if not pipeline_configs:
        raise ValueError("pipeline_configs cannot be empty")

    required_keys = {"name", "pipe_cfgs"}

    for idx, config in enumerate(pipeline_configs):
        # Check all required keys present
        missing = required_keys - set(config.keys())
        if missing:
            raise ValueError(
                    f"Pipeline config {idx} missing required keys: {missing}"
            )

        # Validate name is string
        if not isinstance(config["name"], str):
            raise ValueError(f"Pipeline config {idx}: 'name' must be a string")

        # Validate pipe_cfgs is list
        if not isinstance(config["pipe_cfgs"], list):
            raise ValueError(f"Pipeline config {idx}: 'pipe_cfgs' must be a list")

        # Validate each element is a tuple of (operation, params_dict)
        for op_idx, item in enumerate(config["pipe_cfgs"]):
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError(
                        f"Pipeline config {idx}, pipe_cfgs[{op_idx}]: "
                        f"Each element must be a tuple (operation, params_dict)"
                )

            op, params = item
            if not isinstance(params, dict):
                raise ValueError(
                        f"Pipeline config {idx}, pipe_cfgs[{op_idx}]: "
                        f"Second element of tuple must be a dict"
                )

        # Use existing validation for operations
        _validate_inputs(
                ops=config["pipe_cfgs"],
                data_layers=[],
        )


def _generate_param_combinations(
        parameters: List[Dict[str, List[Any]]],
) -> List[Tuple[Dict[str, Any], ...]]:
    """Generate all parameter combinations using itertools.product.

    Args:
        parameters: List of dicts mapping param names to lists of values

    Returns:
        List of tuples, each tuple contains one dict per operation
    """
    param_combinations = []
    for params_dict in parameters:
        if not params_dict:  # Handle empty dict (no params to vary)
            param_combinations.append([{}])
        else:
            keys = list(params_dict.keys())
            values = list(params_dict.values())
            combos = [dict(zip(keys, vals)) for vals in product(*values)]
            param_combinations.append(combos)

    # Generate all pipeline combinations
    return list(product(*param_combinations))


def _create_param_name_string(param_config: Tuple[Dict[str, Any], ...]) -> str:
    """Create descriptive string from parameter configuration.

    Args:
        param_config: Tuple of parameter dicts (one per operation)

    Returns:
        String like "sigma=2.0_threshold=100"
    """
    parts = []
    for params in param_config:
        for key, value in params.items():
            parts.append(f"{key}={value}")

    return "_".join(parts) if parts else "default"
