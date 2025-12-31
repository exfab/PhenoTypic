"""Shared utilities for pipeline grid search operations.

This module contains core data structures, validation functions, I/O operations,
backend abstractions, trie algorithms, and HTML viewer generation used by both
PipelineGridSearch and MultiPipelineGridSearch.
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
_MEMORY_SAFETY_FACTOR = 0.75  # Use 75% of available memory for safety

# Memory estimation overhead factor
# Conservative 50% overhead accounts for:
# - Python object allocation metadata (10-15%)
# - Garbage collector structures (5-10%)
# - Temporary arrays during operations (15-20%)
# - NumPy view/slice overhead (5-10%)
# - Safety margin for heavy operations (BM3D, wavelets, morphological ops)
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
        ...     "pipeline_001": '{"ops": [{"class": "GaussianBlur", "params": {"sigma": 2.0}}]}',
        ...     "pipeline_002": '{"ops": [{"class": "GaussianBlur", "params": {"sigma": 3.0}}]}'
        ... }
        >>> manifest_path = _create_manifest_json("/tmp/results", configs, ["rgb", "gray"])
    """
    import json
    from datetime import datetime

    manifest_path = Path(output_dir) / "manifest.json"

    try:
        # Build manifest structure
        manifest = {
            "generated_at": datetime.now().isoformat(),
            "total_pipelines": len(configs_dict),
            "data_layers": data_layers,
            "pipelines": {}
        }

        # Add each pipeline entry
        for code, json_config_str in configs_dict.items():
            # Parse JSON config to extract metadata
            try:
                config_obj = json.loads(json_config_str)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON config for {code}, storing as string")
                config_obj = {"raw": json_config_str}

            manifest["pipelines"][code] = {
                "config": config_obj,
                "directory": code,
                "layers": data_layers
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
            "base": base_path,
            "original": original_dir,
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
    # Validate ops format
    if not isinstance(ops, list):
        raise ValueError(f"ops must be a list, got {type(ops)}")

    if not ops:
        raise ValueError("ops cannot be empty")

    for idx, item in enumerate(ops):
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(
                    f"ops[{idx}]: Each element must be a tuple (operation, params_dict), "
                    f"got {type(item)}"
            )

        op, params = item
        if not isinstance(params, dict):
            raise ValueError(
                    f"ops[{idx}]: Second element of tuple must be a dict, "
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

    # Validate data_layers
    valid_layers = {"rgb", "gray", "enh_gray", "objmask", "objmap"}
    invalid = set(data_layers) - valid_layers
    if invalid:
        raise ValueError(
                f"Invalid data_layers: {invalid}. Must be subset of {valid_layers}"
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

    required_keys = {"name", "ops"}

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

        # Validate ops is list
        if not isinstance(config["ops"], list):
            raise ValueError(f"Pipeline config {idx}: 'ops' must be a list")

        # Validate each element is a tuple of (operation, params_dict)
        for op_idx, item in enumerate(config["ops"]):
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError(
                        f"Pipeline config {idx}, ops[{op_idx}]: "
                        f"Each element must be a tuple (operation, params_dict)"
                )

            op, params = item
            if not isinstance(params, dict):
                raise ValueError(
                        f"Pipeline config {idx}, ops[{op_idx}]: "
                        f"Second element of tuple must be a dict"
                )

        # Use existing validation for operations
        _validate_inputs(
                ops=config["ops"],
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


def _extract_data_layers(
        result_img: "Image",
        data_layers: List[str],
) -> Dict[str, Any]:
    """Extract only requested data layers as independent array copies.

    Creates explicit copies (not views) of only the requested image data layers.
    This allows the original Image object to be garbage collected immediately,
    reducing memory usage by ~10× compared to keeping full Image objects.

    Args:
        result_img: Image object to extract data from
        data_layers: List of layer names to extract (e.g., [\"rgb\", \"objmask\"])

    Returns:
        Dictionary mapping layer names to numpy array copies. Only includes
        layers that exist and are non-empty in the source image.

    Note:
        All returned arrays are independent copies, not views. This ensures
        the source Image can be garbage collected without affecting the
        extracted data.
    """
    import numpy as np

    extracted = {}

    for layer in data_layers:
        if layer == "rgb":
            rgb_data = result_img.rgb[:]
            if rgb_data is not None and rgb_data.size > 0:
                extracted["rgb"] = rgb_data.copy()
        elif layer == "gray":
            gray_data = result_img.gray[:]
            if gray_data is not None and gray_data.size > 0:
                extracted["gray"] = gray_data.copy()
        elif layer == "enh_gray":
            enh_gray_data = result_img.enh_gray[:]
            if enh_gray_data is not None and enh_gray_data.size > 0:
                extracted["enh_gray"] = enh_gray_data.copy()
        elif layer == "objmask":
            objmask_data = result_img.objmask[:]
            if objmask_data is not None and objmask_data.any():
                extracted["objmask"] = objmask_data.copy()
        elif layer == "objmap":
            objmap_data = result_img.objmap[:]
            if objmap_data is not None and objmap_data.any():
                extracted["objmap"] = objmap_data.copy()

    return extracted


def _save_array_as_tiff(
        array: "np.ndarray",
        save_dir: Union[str, "Path"],
        base_name: str,
) -> "Path":
    """Save numpy array as TIFF file with appropriate format handling.

    Handles RGB, grayscale, boolean masks, and uint16 label maps. Uses PIL
    for saving to ensure maximum compatibility with image viewers.

    Args:
        array: NumPy array to save. Supported shapes:
            - (H, W): Grayscale or labels
            - (H, W, 3): RGB image
        save_dir: Directory to save TIFF file in
        base_name: Base filename without extension (e.g., \"001_sigma=2.0_rgb\")

    Returns:
        Path to saved TIFF file

    Raises:
        ValueError: If array shape is not supported
        RuntimeError: If TIFF saving fails

    Example:
        >>> import numpy as np
        >>> array = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        >>> path = _save_array_as_tiff(array, \"/tmp/results\", \"001_rgb\")
        >>> print(path)
        /tmp/results/001_rgb.tiff
    """
    from pathlib import Path
    from PIL import Image as PIL_Image
    import numpy as np

    try:
        save_path = Path(save_dir) / f"{base_name}.tiff"

        # Handle different array types
        if array.ndim == 2:
            # Grayscale or labels
            if array.dtype == np.bool_:
                # Boolean mask: convert to 0-255
                array_to_save = (array * 255).astype(np.uint8)
                pil_img = PIL_Image.fromarray(array_to_save, mode='L')
            elif array.dtype == np.uint16:
                # Label map: save as 16-bit grayscale
                pil_img = PIL_Image.fromarray(array, mode='I;16')
            elif np.issubdtype(array.dtype, np.floating):
                # Floating point array: normalize [0.0-1.0] to uint8 [0-255]
                # For values outside [0,1], clip to this range for scientific accuracy
                array_normalized = np.clip(array, 0.0, 1.0)
                array_to_save = (array_normalized * 255).astype(np.uint8)
                pil_img = PIL_Image.fromarray(array_to_save, mode='L')
            else:
                # Regular uint8 grayscale (or other integer types)
                pil_img = PIL_Image.fromarray(array, mode='L')

        elif array.ndim == 3 and array.shape[2] == 3:
            # RGB image
            pil_img = PIL_Image.fromarray(array, mode='RGB')

        else:
            raise ValueError(
                    f"Unsupported array shape {array.shape}. Expected (H, W) or (H, W, 3)"
            )

        # Save as TIFF
        pil_img.save(save_path, format='TIFF', compression='lzw')

        logger.debug(f"Saved TIFF: {save_path}")
        return save_path

    except Exception as e:
        logger.error(f"Failed to save TIFF {base_name}: {e}")
        raise RuntimeError(f"TIFF saving failed for {base_name}: {e}") from e


def _validate_output_dir_params(
        output_dir: Optional[str],
        create_viewer: bool,
        backend: str,
) -> None:
    """Validate output directory and backend parameters.

    Args:
        output_dir: Path to output directory
        create_viewer: Whether HTML viewer generation was requested
        backend: Execution backend name

    Raises:
        ValueError: If parameters are invalid or incompatible
        ImportError: If backend requires unavailable package
    """
    from pathlib import Path

    # Validate output_dir is provided
    if output_dir is None:
        raise ValueError(
                "output_dir is required. Please specify a directory for saving results."
        )

    # Validate and create output_dir
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Test write permissions
        test_file = output_path / ".write_test"
        test_file.touch()
        test_file.unlink()

    except Exception as e:
        raise ValueError(
                f"Cannot write to output_dir '{output_dir}': {e}"
        ) from e

    # Validate backend
    if backend not in ["joblib", "submitit"]:
        raise ValueError(
                f"backend must be 'joblib' or 'submitit', got '{backend}'"
        )

    # Check submitit availability
    if backend == "submitit":
        try:
            import submitit  # noqa: F401
        except ImportError as e:
            raise ImportError(
                    "submitit backend requested but submitit is not installed. "
                    "Install with: pip install phenotypic[cluster]"
            ) from e


def _create_submitit_executor(
        slurm_params: Optional[Dict[str, Any]] = None,
) -> "submitit.AutoExecutor":
    """Create and configure submitit executor for SLURM jobs.

    Args:
        slurm_params: Optional configuration dict with keys:
            - folder: Log folder path (default: \"./submitit_logs\")
            - timeout_min: Job timeout in minutes (default: 60)
            - mem_gb: Memory per job in GB (default: 16)
            - cpus_per_task: CPUs per job (default: 1)
            - slurm_partition: SLURM partition name (optional)
            - Any other submitit-supported parameters

    Returns:
        Configured AutoExecutor ready for job submission

    Raises:
        RuntimeError: If executor creation fails
    """
    import submitit

    # Default parameters
    defaults = {
        "folder"       : "./submitit_logs",
        "timeout_min"  : 60,
        "mem_gb"       : 16,
        "cpus_per_task": 1,
    }

    # Merge with user params (user params override defaults)
    params = {**defaults, **(slurm_params or {})}

    # Extract folder/cluster for executor creation
    folder = params.pop("folder")
    cluster = params.pop("cluster", None)

    try:
        # Create executor
        if cluster is not None:
            executor = submitit.AutoExecutor(folder=folder, cluster=cluster)
        else:
            executor = submitit.AutoExecutor(folder=folder)

        # Update with all parameters
        executor.update_parameters(**params)

        logger.info(f"Created submitit executor with params: {params}")
        logger.info(f"Submitit logs will be saved to: {folder}")

        return executor

    except Exception as e:
        raise RuntimeError(
                f"Failed to create submitit executor: {e}. "
                f"Ensure you are on a SLURM cluster and submitit is properly configured."
        ) from e


def _execute_parallel_tasks(
        func: Callable,
        task_args: List[Tuple],
        backend: str = "joblib",
        n_jobs: int = -1,
        slurm_params: Optional[Dict[str, Any]] = None,
        desc: str = "Processing",
) -> List[Any]:
    """Execute tasks in parallel using specified backend.

    Provides unified interface for parallel execution across joblib (local)
    and submitit (SLURM cluster) backends. Handles progress tracking and
    error reporting consistently.

    Args:
        func: Function to execute for each task. Must be picklable for
            submitit backend.
        task_args: List of argument tuples to pass to func. Each tuple
            is unpacked as func(*args).
        backend: Execution backend - \"joblib\" or \"submitit\"
        n_jobs: Number of parallel jobs (joblib only). -1 uses all cores.
        slurm_params: Configuration for submitit backend
        desc: Description for progress bar

    Returns:
        List of results in same order as task_args

    Raises:
        ValueError: If function is not picklable (submitit only)
        RuntimeError: If job execution fails
    """
    if backend == "joblib":
        from joblib import Parallel, delayed
        from tqdm_joblib import tqdm_joblib

        # Wrapper function to collect errors per task
        def _task_with_error_handling(idx, func, args):
            """Execute task and return (result, error, task_idx) tuple."""
            try:
                result = func(*args)
                return result, None, idx
            except Exception as e:
                return None, e, idx

        with tqdm_joblib(desc=desc, total=len(task_args)):
            task_results = Parallel(n_jobs=n_jobs)(
                    delayed(_task_with_error_handling)(idx, func, args)
                    for idx, args in enumerate(task_args)
            )

        # Separate results and errors
        results = []
        failed_tasks = []

        for result, error, task_idx in task_results:
            if error is None:
                results.append(result)
            else:
                logger.error(f"Task {task_idx} failed: {error}")
                failed_tasks.append((task_idx, error))

        # Report failures (match submitit behavior)
        if failed_tasks:
            failure_msg = "\\n".join(
                    f"  Task {idx}: {error}"
                    for idx, error in failed_tasks
            )
            raise RuntimeError(
                    f"{len(failed_tasks)} task(s) failed:\\n{failure_msg}"
            )

        return results

    elif backend == "submitit":
        import submitit
        from tqdm.auto import tqdm

        # Validate function is picklable
        try:
            import pickle

            pickle.dumps(func)
        except Exception as e:
            raise ValueError(
                    f"Function '{func.__name__}' is not picklable and cannot "
                    f"be used with submitit backend. Error: {e}"
            ) from e

        # Create executor
        executor = _create_submitit_executor(slurm_params)

        # Submit all jobs
        logger.info(f"Submitting {len(task_args)} jobs to SLURM...")
        jobs = []
        for args in task_args:
            job = executor.submit(func, *args)
            jobs.append(job)

        logger.info(f"Submitted {len(jobs)} jobs. Waiting for completion...")

        # Wait for results with progress bar
        results = []
        failed_jobs = []

        for idx, job in enumerate(tqdm(jobs, desc=f"{desc} (SLURM)")):
            try:
                result = job.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Job {job.job_id} failed: {e}")
                failed_jobs.append((idx, job.job_id, e))

        # Report failures
        if failed_jobs:
            failure_msg = "\\n".join(
                    f"  Task {idx} (Job {job_id}): {error}"
                    for idx, job_id, error in failed_jobs
            )
            raise RuntimeError(
                    f"{len(failed_jobs)} job(s) failed:\\n{failure_msg}"
            )

        logger.info(f"All {len(jobs)} jobs completed successfully")
        return results

    else:
        # Should never reach here due to validation
        raise ValueError(f"Unknown backend: {backend}")


def _create_single_thumbnail(
        tiff_path: "Path",
        thumbnails_dir: "Path",
        base_name: str,
        layer: str,
) -> Optional[str]:
    """Create a single thumbnail from a TIFF file.

    Args:
        tiff_path: Path to TIFF file
        thumbnails_dir: Directory to save thumbnail JPEG
        base_name: Base name for result
        layer: Data layer name

    Returns:
        Relative path to thumbnail (e.g., \"thumbnails/001_sigma=2.0_rgb_thumb.jpg\")
        or None if thumbnail creation failed
    """
    from PIL import Image as PIL_Image
    import numpy as np

    try:
        img = PIL_Image.open(tiff_path)
        img.thumbnail(_THUMBNAIL_SIZE, PIL_Image.Resampling.LANCZOS)
        thumb_name = f"{base_name}_{layer}_thumb.jpg"
        thumb_path = thumbnails_dir / thumb_name

        # Convert to RGB for JPEG saving if needed
        if img.mode == 'I;16' or img.mode == 'L':
            # Normalize to 0-255 for display
            img_array = np.array(img)
            if img_array.max() > 255:
                img_array = (img_array / img_array.max() * 255).astype(np.uint8)
            else:
                img_array = img_array.astype(np.uint8)
            img = PIL_Image.fromarray(img_array)

        img.save(thumb_path, format='JPEG', quality=_THUMBNAIL_JPEG_QUALITY)
        return f"thumbnails/{thumb_name}"

    except Exception as e:
        logger.warning(f"Failed to create thumbnail for {tiff_path}: {e}")
        return None


def _create_interactive_viewer_html(
        output_dir: Union[str, Path],
        configs_dict: Dict[str, str],
        data_layers: List[str],
) -> Path:
    """Generate interactive HTML viewer with three-panel layout.

    Creates a static HTML viewer with:
    - Left sidebar: Pipeline list for selection
    - Center panel: Main image view with layer selector
    - Right sidebar: Pipeline configuration details
    - Keyboard navigation support

    Args:
        output_dir: Output directory containing pipeline subdirectories
        configs_dict: Dictionary mapping pipeline codes to JSON config strings
        data_layers: List of layer names that were saved

    Returns:
        Path to generated viewer.html file

    Raises:
        RuntimeError: If HTML generation fails

    Example:
        >>> configs = {
        ...     "pipeline_001": '{"ops": [{"class": "GaussianBlur", "params": {"sigma": 2.0}}]}',
        ...     "pipeline_002": '{"ops": [{"class": "GaussianBlur", "params": {"sigma": 3.0}}]}'
        ... }
        >>> html_path = _create_interactive_viewer_html("./results", configs, ["rgb", "gray"])
    """
    import json

    output_path = Path(output_dir)
    html_file = output_path / "viewer.html"
    thumbnails_dir = output_path / "thumbnails"
    thumbnails_dir.mkdir(exist_ok=True)

    logger.info(f"Generating interactive viewer HTML with {len(configs_dict)} pipelines")

    # Create thumbnails for all pipeline layers
    thumbnail_tasks = []
    for pipeline_code in configs_dict.keys():
        pipeline_dir = output_path / pipeline_code
        for layer in data_layers:
            tiff_path = pipeline_dir / f"{layer}.tiff"
            if tiff_path.exists():
                thumbnail_tasks.append((tiff_path, thumbnails_dir, pipeline_code, layer))

    # Parallelize thumbnail creation
    if len(thumbnail_tasks) > 1:
        from joblib import Parallel, delayed
        logger.info(f"Creating {len(thumbnail_tasks)} thumbnails in parallel...")
        Parallel(n_jobs=-1)(
                delayed(_create_single_thumbnail)(*args) for args in thumbnail_tasks
        )
    else:
        for args in thumbnail_tasks:
            _create_single_thumbnail(*args)

    # Build pipeline data structure for JavaScript
    pipelines_data = {}
    for pipeline_code, json_config_str in configs_dict.items():
        try:
            config_obj = json.loads(json_config_str)
        except json.JSONDecodeError:
            config_obj = {"error": "Failed to parse config", "raw": json_config_str}

        pipelines_data[pipeline_code] = {
            "code": pipeline_code,
            "config": config_obj,
            "layers": {layer: f"{pipeline_code}/{layer}.tiff" for layer in data_layers}
        }

    # Serialize pipeline data to JSON for embedding
    pipelines_json = json.dumps(pipelines_data, indent=2)

    # Generate HTML content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pipeline Grid Search Viewer</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            height: 100vh;
            display: flex;
            flex-direction: column;
            background-color: #1e1e1e;
            color: #d4d4d4;
        }}
        
        header {{
            background-color: #252526;
            padding: 12px 20px;
            border-bottom: 1px solid #3e3e42;
        }}
        
        header h1 {{
            font-size: 18px;
            font-weight: 500;
            color: #cccccc;
        }}
        
        .container {{
            display: flex;
            flex: 1;
            overflow: hidden;
        }}
        
        /* Left Sidebar - Pipeline List */
        .sidebar-left {{
            width: 250px;
            background-color: #252526;
            border-right: 1px solid #3e3e42;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}
        
        .sidebar-left h2 {{
            padding: 12px 16px;
            font-size: 13px;
            font-weight: 600;
            color: #999;
            text-transform: uppercase;
            border-bottom: 1px solid #3e3e42;
        }}
        
        .pipeline-list {{
            flex: 1;
            overflow-y: auto;
            padding: 8px 0;
        }}
        
        .pipeline-item {{
            padding: 8px 16px;
            cursor: pointer;
            font-size: 13px;
            font-family: 'Consolas', 'Monaco', monospace;
            color: #cccccc;
            border-left: 3px solid transparent;
            transition: background-color 0.15s;
        }}
        
        .pipeline-item:hover {{
            background-color: #2a2d2e;
        }}
        
        .pipeline-item.active {{
            background-color: #37373d;
            border-left-color: #007acc;
            color: #ffffff;
        }}
        
        /* Center Panel - Image View */
        .main-view {{
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            background-color: #1e1e1e;
        }}
        
        .layer-selector {{
            background-color: #252526;
            padding: 12px 20px;
            border-bottom: 1px solid #3e3e42;
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }}
        
        .layer-button {{
            padding: 6px 14px;
            background-color: #3e3e42;
            color: #cccccc;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 500;
            transition: background-color 0.15s;
        }}
        
        .layer-button:hover {{
            background-color: #505050;
        }}
        
        .layer-button.active {{
            background-color: #007acc;
            color: #ffffff;
        }}
        
        .image-container {{
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: auto;
            padding: 20px;
        }}
        
        .image-container img {{
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            border: 1px solid #3e3e42;
            background-color: #2d2d30;
        }}
        
        .no-image {{
            color: #858585;
            font-size: 14px;
        }}
        
        /* Right Sidebar - Config Panel */
        .sidebar-right {{
            width: 350px;
            background-color: #252526;
            border-left: 1px solid #3e3e42;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}
        
        .sidebar-right h2 {{
            padding: 12px 16px;
            font-size: 13px;
            font-weight: 600;
            color: #999;
            text-transform: uppercase;
            border-bottom: 1px solid #3e3e42;
        }}
        
        .config-content {{
            flex: 1;
            overflow-y: auto;
            padding: 16px;
        }}
        
        .config-section {{
            margin-bottom: 20px;
        }}
        
        .config-section h3 {{
            font-size: 12px;
            color: #999;
            text-transform: uppercase;
            margin-bottom: 8px;
        }}
        
        .config-code {{
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 16px;
            color: #4ec9b0;
            margin-bottom: 16px;
        }}
        
        .operation {{
            background-color: #2d2d30;
            border-left: 3px solid #007acc;
            padding: 10px 12px;
            margin-bottom: 8px;
            border-radius: 3px;
        }}
        
        .operation-name {{
            font-weight: 600;
            color: #dcdcaa;
            font-size: 13px;
            margin-bottom: 6px;
        }}
        
        .param {{
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 12px;
            color: #9cdcfe;
            margin-left: 12px;
        }}
        
        .param-key {{
            color: #9cdcfe;
        }}
        
        .param-value {{
            color: #ce9178;
        }}
        
        /* Scrollbar styling */
        ::-webkit-scrollbar {{
            width: 10px;
            height: 10px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: #1e1e1e;
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: #424242;
            border-radius: 5px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: #4e4e4e;
        }}
    </style>
</head>
<body>
    <header>
        <h1>Pipeline Grid Search Viewer</h1>
    </header>
    
    <div class="container">
        <!-- Left Sidebar: Pipeline List -->
        <div class="sidebar-left">
            <h2>Pipelines</h2>
            <div class="pipeline-list" id="pipelineList"></div>
        </div>
        
        <!-- Center: Image View -->
        <div class="main-view">
            <div class="layer-selector" id="layerSelector"></div>
            <div class="image-container" id="imageContainer">
                <div class="no-image">Select a pipeline to view</div>
            </div>
        </div>
        
        <!-- Right Sidebar: Config Panel -->
        <div class="sidebar-right">
            <h2>Configuration</h2>
            <div class="config-content" id="configPanel">
                <div class="no-image">Select a pipeline to view configuration</div>
            </div>
        </div>
    </div>
    
    <script>
        // Embedded pipeline data
        const pipelinesData = {pipelines_json};
        const dataLayers = {json.dumps(data_layers)};
        
        let currentPipeline = null;
        let currentLayer = dataLayers[0] || 'rgb';
        
        // Initialize viewer
        function initViewer() {{
            renderPipelineList();
            renderLayerSelector();
            
            // Select first pipeline by default
            const firstPipeline = Object.keys(pipelinesData)[0];
            if (firstPipeline) {{
                selectPipeline(firstPipeline);
            }}
            
            // Setup keyboard navigation
            document.addEventListener('keydown', handleKeyboard);
        }}
        
        // Render pipeline list
        function renderPipelineList() {{
            const listEl = document.getElementById('pipelineList');
            listEl.innerHTML = '';
            
            for (const code in pipelinesData) {{
                const item = document.createElement('div');
                item.className = 'pipeline-item';
                item.textContent = code;
                item.onclick = () => selectPipeline(code);
                listEl.appendChild(item);
            }}
        }}
        
        // Render layer selector buttons
        function renderLayerSelector() {{
            const selectorEl = document.getElementById('layerSelector');
            selectorEl.innerHTML = '';
            
            dataLayers.forEach(layer => {{
                const btn = document.createElement('button');
                btn.className = 'layer-button';
                btn.textContent = layer;
                btn.onclick = () => selectLayer(layer);
                if (layer === currentLayer) {{
                    btn.classList.add('active');
                }}
                selectorEl.appendChild(btn);
            }});
        }}
        
        // Select pipeline
        function selectPipeline(code) {{
            currentPipeline = code;
            
            // Update active state
            document.querySelectorAll('.pipeline-item').forEach(item => {{
                item.classList.toggle('active', item.textContent === code);
            }});
            
            // Update displays
            updateImageView();
            updateConfigPanel();
        }}
        
        // Select layer
        function selectLayer(layer) {{
            currentLayer = layer;
            
            // Update active state
            document.querySelectorAll('.layer-button').forEach(btn => {{
                btn.classList.toggle('active', btn.textContent === layer);
            }});
            
            // Update image
            updateImageView();
        }}
        
        // Update image view
        function updateImageView() {{
            const containerEl = document.getElementById('imageContainer');
            
            if (!currentPipeline || !currentLayer) {{
                containerEl.innerHTML = '<div class="no-image">No image available</div>';
                return;
            }}
            
            const pipeline = pipelinesData[currentPipeline];
            const imagePath = pipeline.layers[currentLayer];
            
            if (imagePath) {{
                containerEl.innerHTML = `<img src="${{imagePath}}" alt="${{currentLayer}}" />`;
            }} else {{
                containerEl.innerHTML = '<div class="no-image">Image not available</div>';
            }}
        }}
        
        // Update config panel
        function updateConfigPanel() {{
            const panelEl = document.getElementById('configPanel');
            
            if (!currentPipeline) {{
                panelEl.innerHTML = '<div class="no-image">No pipeline selected</div>';
                return;
            }}
            
            const pipeline = pipelinesData[currentPipeline];
            const config = pipeline.config;
            
            let html = `<div class="config-code">${{pipeline.code}}</div>`;
            
            if (config.ops && Array.isArray(config.ops)) {{
                html += '<div class="config-section"><h3>Operations</h3>';
                
                config.ops.forEach((op, idx) => {{
                    const opClass = op.class || 'Unknown';
                    const params = op.params || {{}};
                    
                    html += `<div class="operation">
                        <div class="operation-name">${{idx + 1}}. ${{opClass}}</div>`;
                    
                    for (const [key, value] of Object.entries(params)) {{
                        const valueStr = typeof value === 'object' ? JSON.stringify(value) : value;
                        html += `<div class="param">
                            <span class="param-key">${{key}}:</span> 
                            <span class="param-value">${{valueStr}}</span>
                        </div>`;
                    }}
                    
                    html += '</div>';
                }});
                
                html += '</div>';
            }} else {{
                html += '<div class="config-section">No operations found</div>';
            }}
            
            panelEl.innerHTML = html;
        }}
        
        // Keyboard navigation
        function handleKeyboard(e) {{
            const codes = Object.keys(pipelinesData);
            const currentIdx = codes.indexOf(currentPipeline);
            
            if (e.key === 'ArrowDown' && currentIdx < codes.length - 1) {{
                e.preventDefault();
                selectPipeline(codes[currentIdx + 1]);
            }} else if (e.key === 'ArrowUp' && currentIdx > 0) {{
                e.preventDefault();
                selectPipeline(codes[currentIdx - 1]);
            }} else if (e.key === 'ArrowRight') {{
                e.preventDefault();
                const layerIdx = dataLayers.indexOf(currentLayer);
                if (layerIdx < dataLayers.length - 1) {{
                    selectLayer(dataLayers[layerIdx + 1]);
                }}
            }} else if (e.key === 'ArrowLeft') {{
                e.preventDefault();
                const layerIdx = dataLayers.indexOf(currentLayer);
                if (layerIdx > 0) {{
                    selectLayer(dataLayers[layerIdx - 1]);
                }}
            }}
        }}
        
        // Initialize on load
        window.addEventListener('DOMContentLoaded', initViewer);
    </script>
</body>
</html>
"""

    # Write HTML file
    try:
        html_file.write_text(html_content, encoding='utf-8')
        logger.info(f"Created interactive viewer: {html_file}")
        return html_file
    except Exception as e:
        raise RuntimeError(f"Failed to create HTML viewer: {e}") from e


def _estimate_pipeline_memory(
        image: "Image",
        num_operations: int,
        data_layers: List[str],
        extract_arrays: bool = True,
) -> int:
    """Estimate peak memory usage for one pipeline execution in bytes.

    Estimates memory based on image size, number of operations, and whether
    array extraction is used. This helps adaptive batching determine how
    many pipelines can run in parallel without exceeding memory limits.

    Args:
        image: Input image to process
        num_operations: Number of operations in the pipeline
        data_layers: List of data layers to extract
        extract_arrays: Whether array extraction optimization is used

    Returns:
        Estimated peak memory usage in bytes for processing one pipeline

    Note:
        This is an estimate and may vary by ±20% depending on operation types
        and actual memory allocation patterns. Conservative estimates help
        prevent OOM errors.
    """
    import sys
    import numpy as np

    # Get base image data sizes without creating copies
    base_size = 0

    # RGB array size (if present) - access shape/dtype without copying
    rgb_accessor = image.rgb
    if rgb_accessor is not None:
        try:
            # Avoid creating full array copy - calculate from shape and dtype
            if hasattr(rgb_accessor, 'shape') and hasattr(rgb_accessor, 'dtype'):
                base_size += np.prod(rgb_accessor.shape) * rgb_accessor.dtype.itemsize
            else:
                # Fallback: minimal peek at data for dtype detection
                rgb_peek = image.rgb[0:1, 0:1] if hasattr(image.rgb,
                                                          'shape') else image.rgb[:]
                if rgb_peek is not None and rgb_peek.size > 0:
                    shape = image.rgb.shape if hasattr(image.rgb,
                                                       'shape') else rgb_peek.shape
                    base_size += np.prod(shape) * rgb_peek.dtype.itemsize
        except Exception:
            pass  # If access fails, skip this layer

    # Gray and enhanced gray (always present) - no copies
    gray_accessor = image.gray
    if gray_accessor is not None:
        try:
            if hasattr(gray_accessor, 'shape') and hasattr(gray_accessor, 'dtype'):
                base_size += np.prod(gray_accessor.shape) * gray_accessor.dtype.itemsize
            else:
                gray_peek = image.gray[0:1, 0:1] if hasattr(image.gray,
                                                            'shape') else image.gray[:]
                if gray_peek is not None:
                    shape = image.gray.shape if hasattr(image.gray,
                                                        'shape') else gray_peek.shape
                    base_size += np.prod(shape) * gray_peek.dtype.itemsize
        except Exception:
            pass

    enh_gray_accessor = image.enh_gray
    if enh_gray_accessor is not None:
        try:
            if hasattr(enh_gray_accessor, 'shape') and hasattr(enh_gray_accessor,
                                                               'dtype'):
                base_size += np.prod(
                        enh_gray_accessor.shape) * enh_gray_accessor.dtype.itemsize
            else:
                enh_peek = image.enh_gray[0:1, 0:1] if hasattr(image.enh_gray,
                                                               'shape') else image.enh_gray[
                    :]
                if enh_peek is not None:
                    shape = image.enh_gray.shape if hasattr(image.enh_gray,
                                                            'shape') else enh_peek.shape
                    base_size += np.prod(shape) * enh_peek.dtype.itemsize
        except Exception:
            pass

    if extract_arrays:
        # With array extraction: 1 full image copy + extracted arrays
        # Estimate extracted array sizes
        extracted_size = 0
        for layer in data_layers:
            if layer == "rgb" and rgb_accessor is not None:
                try:
                    shape = rgb_accessor.shape if hasattr(rgb_accessor,
                                                          'shape') else None
                    dtype = rgb_accessor.dtype if hasattr(rgb_accessor,
                                                          'dtype') else None
                    if shape is not None and dtype is not None:
                        extracted_size += (np.prod(shape) * dtype.itemsize)
                except Exception:
                    pass
            elif layer == "gray" and gray_accessor is not None:
                try:
                    shape = gray_accessor.shape if hasattr(gray_accessor,
                                                           'shape') else None
                    dtype = gray_accessor.dtype if hasattr(gray_accessor,
                                                           'dtype') else None
                    if shape is not None and dtype is not None:
                        extracted_size += (np.prod(shape) * dtype.itemsize)
                except Exception:
                    pass
            elif layer == "enh_gray" and enh_gray_accessor is not None:
                try:
                    shape = enh_gray_accessor.shape if hasattr(enh_gray_accessor,
                                                               'shape') else None
                    dtype = enh_gray_accessor.dtype if hasattr(enh_gray_accessor,
                                                               'dtype') else None
                    if shape is not None and dtype is not None:
                        extracted_size += (np.prod(shape) * dtype.itemsize)
                except Exception:
                    pass
            elif layer in ["objmask", "objmap"]:
                # Estimate label map size (uint16 label maps: 2 bytes per pixel)
                try:
                    shape = gray_accessor.shape if hasattr(gray_accessor,
                                                           'shape') else None
                    if shape is not None:
                        extracted_size += (
                                np.prod(shape) * np.dtype(np.uint16).itemsize)
                except Exception:
                    pass

        # Peak: base image + extracted arrays + overhead
        return int((base_size + extracted_size) * _MEMORY_OVERHEAD_FACTOR)
    else:
        # Without extraction: num_operations intermediate copies
        # Each operation creates a new image
        # Peak occurs when all intermediate copies exist simultaneously
        return int(base_size * (num_operations + 1) * _MEMORY_OVERHEAD_FACTOR)


def _calculate_optimal_batch_size(
        total_pipelines: int,
        memory_per_pipeline: int,
        memory_limit_gb: float = None,
        n_jobs: int = -1,
        max_batch_size: int = None,
) -> Tuple[int, int]:
    """Calculate optimal batch size and parallelism for memory-constrained execution.

    Automatically determines how many pipelines can run in parallel and what
    batch size to use based on available system memory and estimated memory
    per pipeline. Ensures execution stays within memory limits while maximizing
    parallelism.

    Args:
        total_pipelines: Total number of pipelines to execute
        memory_per_pipeline: Estimated memory per pipeline execution (bytes)
        memory_limit_gb: Memory limit in GB. If None, uses 75% of available
            system memory for safety. Default None.
        n_jobs: User-requested parallel jobs. -1 means use all cores. If
            specified, will be respected but limited by memory constraints.
        max_batch_size: Maximum batch size to use. If None, uses default
            _MAX_SAFE_BATCH_SIZE. Allows override for high-memory systems.

    Returns:
        Tuple of (batch_size, jobs_per_batch):
            - batch_size: Number of pipelines to process in each batch
            - jobs_per_batch: Number of parallel jobs to use within each batch

    Note:
        Uses conservative estimates (75% of available memory) to prevent OOM
        errors. Actual memory usage may be lower.

    Example:
        >>> batch_size, jobs = _calculate_optimal_batch_size(
        ...     total_pipelines=200,
        ...     memory_per_pipeline=50_000_000,  # 50 MB
        ...     memory_limit_gb=None,  # Auto-detect
        ...     n_jobs=-1,
        ...     max_batch_size=32  # Allow larger batches on high-memory system
        ... )
        >>> print(f"Process {total_pipelines} in batches of {batch_size}")
    """
    import psutil

    # Determine memory limit in bytes
    if memory_limit_gb is None:
        # Use 75% of available system memory for safety
        available_memory = psutil.virtual_memory().available
        memory_limit = int(available_memory * 0.75)
    else:
        memory_limit = int(memory_limit_gb * 1024 ** 3)

    # Calculate max parallel pipelines that fit in memory
    if memory_per_pipeline <= 0:
        logger.warning(
                f"Memory estimation returned {memory_per_pipeline} bytes (invalid). "
                f"Using conservative fallback estimate of {_DEFAULT_FALLBACK_MEMORY_MB} MB per pipeline."
        )
        memory_per_pipeline = _DEFAULT_FALLBACK_MEMORY_MB * 1024 ** 2  # Convert MB to bytes

    max_parallel = max(1, memory_limit // memory_per_pipeline)

    # Respect user's n_jobs if specified, but limit by memory
    if n_jobs == -1:
        # Use all CPU cores, but limited by memory
        jobs_per_batch = min(max_parallel, psutil.cpu_count())
    else:
        # Use user-specified jobs, but limited by memory
        jobs_per_batch = min(max_parallel, n_jobs)

    # Batch size: process enough to utilize parallel workers efficiently
    # Be conservative to prevent OOM: use multiplier for ideal, but cap at reasonable limit
    max_safe_batch = max_batch_size if max_batch_size is not None else _MAX_SAFE_BATCH_SIZE

    ideal_batch_size = jobs_per_batch * _IDEAL_BATCH_MULTIPLIER
    safe_batch_size = min(max(_MIN_BATCH_SIZE, jobs_per_batch), max_safe_batch)
    batch_size = min(ideal_batch_size, safe_batch_size, total_pipelines)

    return batch_size, jobs_per_batch


def _expand_pipeline_configs_to_concrete(
        pipeline_configs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Expand pipeline configs with parameter lists into concrete pipelines.

    Takes pipeline configs where params are lists of values and generates
    all combinations, creating one concrete pipeline config per combination.

    Args:
        pipeline_configs: List with format:
            [{"name": "Pipeline1",
              "ops": [(op1, {"sigma": [1.0, 2.0]}), (op2, {})]}]

    Returns:
        List of concrete configs with scalar parameter values:
            [{"name": "Pipeline1_sigma=1.0",
              "ops": [(op1, {"sigma": 1.0}), (op2, {})]},
             {"name": "Pipeline1_sigma=2.0",
              "ops": [(op1, {"sigma": 2.0}), (op2, {})]}]
    """
    concrete_configs = []

    for config in pipeline_configs:
        pipeline_name = config["name"]
        ops = config["ops"]

        # Unpack ops tuples
        operations, parameters = _unpack_ops_tuples(ops)

        # Generate parameter combinations
        param_configs = _generate_param_combinations(parameters)

        # Create a concrete pipeline config for each parameter combination
        for param_config in param_configs:
            # Create parameter name string for unique naming
            param_str = _create_param_name_string(param_config)

            # Build unique pipeline name
            if param_str == "default":
                # No parameters varied, use original name
                concrete_name = pipeline_name
            else:
                concrete_name = f"{pipeline_name}_{param_str}"

            # Build concrete ops list with scalar parameter values
            concrete_ops = []
            for op, params in zip(operations, param_config):
                concrete_ops.append((op, params))

            # Add concrete config to list
            concrete_configs.append({
                "name": concrete_name,
                "ops" : concrete_ops,
            })

    return concrete_configs


def _build_pipeline_trie(
        concrete_pipeline_configs: List[Dict[str, Any]],
) -> _TrieNode:
    """Build execution trie from concrete pipeline configurations.

    Groups pipelines with shared prefixes under common nodes. Pipelines are
    compared by operation class name AND parameter values. Parameter values
    must be scalar (not lists), as this function expects concrete configs
    from _expand_pipeline_configs_to_concrete().

    Args:
        concrete_pipeline_configs: List of expanded pipeline configs where
            each param dict contains scalar values (not lists). Each config
            represents one specific parameter combination.

    Returns:
        Root node of the execution trie
    """
    root = _TrieNode()

    for config in concrete_pipeline_configs:
        pipeline_name = config["name"]
        ops = config["ops"]

        current_node = root
        for op, params in ops:
            # Create hashable key for this operation with its specific params
            key = _ops_key(op, params)

            # Create child node if it doesn't exist
            if key not in current_node.children:
                current_node.children[key] = _TrieNode(op=op, params=params)

            # Move to child node
            current_node = current_node.children[key]

        # Mark this node as a pipeline endpoint
        current_node.pipeline_names.append(pipeline_name)

    return root


def _group_pipelines_by_longest_prefix(
        concrete_configs: List[Dict[str, Any]],
) -> List[List[Dict[str, Any]]]:
    """Group pipelines by their longest shared operation prefix.
    
    Builds a temporary trie and groups pipelines that share a common sequence
    of operations. All pipelines sharing a prefix are placed in the SAME trie
    group to enable:
    1. Execute the shared prefix operations only once
    2. Parallelize ALL downstream branches (both parameter sweeps AND structural divergence)
    
    IMPORTANT: This function groups ALL downstream branches (parameter sweeps and
    structural divergence) into a single group at the branch point. Branches are NOT
    split into separate groups - they execute in parallel from the shared prefix.
    
    Example:
        Pipeline A: GaussianBlur(σ=1) → OtsuDetector()
        Pipeline B: GaussianBlur(σ=2) → CannyDetector()
        
        Result: BOTH in same group (branch at GaussianBlur with parameter sweep +
        structural divergence). Execution: run GaussianBlur variants in parallel with
        their respective downstream operations.
    
    Args:
        concrete_configs: List of concrete pipeline configurations
        
    Returns:
        List of pipeline groups sharing longest possible operation sequence.
        All branches (parameter sweeps AND structural divergence) are grouped
        together to maximize parallelization opportunities.
    """
    if not concrete_configs:
        return []
    if len(concrete_configs) == 1:
        return [concrete_configs]

    # Build temporary trie
    temp_trie = _build_pipeline_trie(concrete_configs)

    # Recursively identify groups
    def _collect_groups_from_node(node: _TrieNode) -> List[List[str]]:
        """Collect pipeline groups from a trie node.
        
        Returns list of groups, where each group is a list of pipeline names.
        """
        if not node.children:
            # Leaf node - return pipelines at this endpoint
            if node.pipeline_names:
                return [[name] for name in node.pipeline_names]
            return []

        if len(node.children) == 1:
            # Single child - no branching yet, continue traversal
            child = next(iter(node.children.values()))
            return _collect_groups_from_node(child)

        # Multiple children = branch point
        # Collect ALL downstream pipelines from ALL children and merge into one group
        # This handles both:
        # 1. Parameter sweeps (same op type, different params)
        # 2. Structural divergence (different op types)
        # Both are grouped together to enable parallel execution from this branch point
        all_pipeline_names = []
        for child in node.children.values():
            child_groups = _collect_groups_from_node(child)
            # Flatten: merge all child groups into single group
            for group in child_groups:
                all_pipeline_names.extend(group)
        # Return as single merged group
        return [all_pipeline_names] if all_pipeline_names else []

    # Collect pipeline name groups
    pipeline_name_groups = _collect_groups_from_node(temp_trie)

    # Map pipeline names back to configs
    config_by_name = {cfg["name"]: cfg for cfg in concrete_configs}
    config_groups = [
        [config_by_name[name] for name in group]
        for group in pipeline_name_groups
    ]

    logger.debug(f"Grouped {len(concrete_configs)} pipelines into "
                 f"{len(config_groups)} trie groups by longest shared prefix")
    logger.debug(f"  Group sizes: {[len(g) for g in config_groups]}")

    return config_groups


def _process_trie_groups_sequentially(
        image: "Image",
        trie_groups: List[List[Dict[str, Any]]],
        n_jobs: int,
        data_layers: List[str],
        extract_arrays: bool = True,
        backend: str = "joblib",
        slurm_params: Optional[Dict[str, Any]] = None,
):
    """Process multiple trie groups sequentially.

    Each trie group contains pipelines sharing longest prefix before structural
    divergence. Groups are processed sequentially (not in parallel) to maintain
    memory efficiency and avoid coordination overhead. Within each group, the
    existing shallow traversal is used (serial prefix + parallel branches).

    Args:
        image: Original image to process
        trie_groups: List of pipeline groups (each group is a list of configs)
        n_jobs: Number of parallel jobs for branch execution within each trie
        data_layers: List of data layers to extract
        extract_arrays: Whether to extract arrays vs returning full Images
        backend: Execution backend - "joblib" (local) or "submitit" (SLURM cluster).
            Default "joblib".
        slurm_params: Configuration dict for submitit backend. Only used when
            backend="submitit". Default None.

    Yields:
        Tuple of (pipeline_name, result_data, json_config) for each completed pipeline
    """
    from tqdm.auto import tqdm

    total_groups = len(trie_groups)
    logger.info(f"Processing {total_groups} trie groups sequentially")

    for group_idx, group_configs in enumerate(
            tqdm(
                    trie_groups,
                    desc="Trie groups",
                    unit="group",
                    total=total_groups,
                    disable=(total_groups == 1),
            ),
            start=1,
    ):
        group_size = len(group_configs)
        logger.info(f"Trie group {group_idx}/{total_groups}: {group_size} pipelines")

        # Log first operation for identification
        if group_configs and group_configs[0]["ops"]:
            first_op_name = type(group_configs[0]["ops"][0][0]).__name__
            logger.debug(f"  First operation: {first_op_name}")

        # Build trie for this group
        group_start = time.time()
        group_trie = _build_pipeline_trie(group_configs)

        # Process this trie group using shallow traversal
        group_results_count = 0
        for result in _execute_pipeline_trie(
                group_trie, image, n_jobs, data_layers, extract_arrays,
                backend, slurm_params
        ):
            group_results_count += 1
            yield result

        group_elapsed = time.time() - group_start
        logger.info(f"Trie group {group_idx}/{total_groups} complete: "
                    f"{group_results_count} pipelines in {group_elapsed:.2f}s")


def _analyze_trie_structure(root: _TrieNode) -> Dict[str, Any]:
    """Analyze trie structure for logging and optimization decisions.

    Args:
        root: Root node of the execution trie

    Returns:
        Dictionary with structure statistics including depth, branch points, and path count
    """

    def count_descendants(node: _TrieNode) -> Tuple[int, int, int]:
        """Count depth, branch points, and total leaf paths from node.

        Returns:
            Tuple of (max_depth, branch_point_count, leaf_path_count)
        """
        if not node.children:
            return 0, 0, 1  # Leaf node - counts as 1 path

        max_depth = 0
        total_branches = 0
        total_paths = 0  # Start at 0, then sum all child paths

        for child in node.children.values():
            child_depth, child_branches, child_paths = count_descendants(child)
            max_depth = max(max_depth, child_depth + 1)
            total_branches += child_branches
            total_paths += child_paths  # Sum paths from all children

        # This node is a branch point if it has multiple children
        if len(node.children) > 1:
            total_branches += 1

        return max_depth, total_branches, total_paths

    depth, branch_points, total_paths = count_descendants(root)

    return {
        "max_depth"       : depth,
        "branch_points"   : branch_points,
        "total_leaf_paths": total_paths,
        "total_nodes"     : _count_trie_nodes(root),
    }


def _count_trie_nodes(node: _TrieNode) -> int:
    """Count total nodes in trie (helper for analysis)."""
    count = 1
    for child in node.children.values():
        count += _count_trie_nodes(child)
    return count


def _find_first_branch_point(
        root: _TrieNode,
) -> Tuple[_TrieNode, List["ImageOperation"]]:
    """Find first node with multiple children, return node and ops_stack to that point.

    Traverses trie from root serially until finding a node with multiple children
    (first branch point). Returns that node and the operation stack to reach it.
    If no branch point exists (linear trie), returns the leaf node.

    Args:
        root: Root node of the execution trie

    Returns:
        Tuple of (branch_point_node, operations_stack) where operations_stack
        contains all ImageOperation objects from root to branch point (not including
        branch point's children operations)
    """
    current = root
    ops_stack = []

    while current.children:
        if len(current.children) > 1:
            # Found first branch point
            logger.debug(
                    f"Branch point found with {len(current.children)} children at "
                    f"depth {len(ops_stack)}"
            )
            return current, ops_stack

        # Linear path: single child, continue traversing
        child_node = next(iter(current.children.values()))
        if child_node.op is not None:
            ops_stack.append(child_node.op)
        current = child_node

    # No branch point found - trie is linear
    logger.debug(f"No branch point found - linear trie with depth {len(ops_stack)}")
    return current, ops_stack


def _enumerate_downstream_paths(
        node: _TrieNode,
        ops_stack: List["ImageOperation"],
) -> List[Tuple[str, List["ImageOperation"]]]:
    """Enumerate all pipeline paths from node to leaf nodes.

    Recursively traverses from given node to all leaf endpoints, building complete
    operation stacks for each path. Returns list of (pipeline_name, full_ops_list).

    Args:
        node: Starting node (typically a branch point)
        ops_stack: Operations already applied to reach this node

    Returns:
        List of (pipeline_name, operations_list) tuples, one per pipeline
    """
    pipelines = []

    # If this node is a pipeline endpoint, add to results
    if node.pipeline_names:
        for name in node.pipeline_names:
            pipelines.append((name, ops_stack.copy()))

    # Process children recursively
    for child_node in node.children.values():
        child_ops = ops_stack.copy()
        if child_node.op is not None:
            child_ops.append(child_node.op)

        # Recursively get paths from child
        child_pipelines = _enumerate_downstream_paths(child_node, child_ops)
        pipelines.extend(child_pipelines)

    return pipelines


def _execute_concrete_pipeline_batch(
        image: "Image",
        pipeline_specs: List[Tuple[str, List["ImageOperation"]]],
        shared_prefix_len: int,
        data_layers: List[str],
        extract_arrays: bool = True,
) -> List[Tuple[str, Any, str]]:
    """Execute concrete pipelines that share a common prefix image.

    Takes a batch of pipeline specifications that all operate on the same
    pre-processed image (from shared prefix). Executes only the operations
    AFTER the shared prefix and returns results.

    Args:
        image: Pre-processed image from shared prefix execution
        pipeline_specs: List of (pipeline_name, full_operations_list) tuples
        shared_prefix_len: Number of operations in shared prefix to skip
        data_layers: List of data layers to extract
        extract_arrays: Whether to extract arrays (vs returning full Image objects)

    Returns:
        List of (pipeline_name, result_data, json_config) tuples
        
    Raises:
        ValueError: If shared_prefix_len exceeds the length of any pipeline's operations
    """
    from phenotypic import ImagePipeline

    # Validate shared_prefix_len before processing
    for pipeline_name, all_ops in pipeline_specs:
        if shared_prefix_len > len(all_ops):
            raise ValueError(
                    f"shared_prefix_len ({shared_prefix_len}) exceeds pipeline '{pipeline_name}' "
                    f"length ({len(all_ops)}). This indicates a configuration error in trie traversal."
            )

    results = []

    for pipeline_name, all_ops in pipeline_specs:
        # Get only the operations after the shared prefix
        remaining_ops = all_ops[shared_prefix_len:]

        if not remaining_ops:
            # No operations to apply - result is the input image
            result_image = image.copy()
            # For serialization: use original operations (no remaining ops to copy)
            ops_for_pipeline = copy.deepcopy(all_ops)
        else:
            # Deep copy operations ONCE for both execution and serialization
            # Avoid parameter sharing between executions
            ops_copy = [copy.deepcopy(op) for op in remaining_ops]

            # Apply remaining operations to the already-processed image
            result_image = image.copy()
            for op in ops_copy:
                result_image = op.apply(result_image)

            # Reuse copied operations for serialization (combine prefix + copied ops)
            # Prefix operations are not copied (they already ran)
            ops_for_pipeline = all_ops[:shared_prefix_len] + ops_copy

        # Create pipeline with operations for serialization (no second copy needed)
        pipeline = ImagePipeline(ops=ops_for_pipeline)

        # Extract data or keep full image
        if extract_arrays:
            result_data = _extract_data_layers(result_image, data_layers)
        else:
            result_data = result_image

        # Serialize config
        json_config = pipeline.to_json_str()

        results.append((pipeline_name, result_data, json_config))
        logger.debug(f"Pipeline '{pipeline_name}' completed")

    return results


def _execute_pipeline_trie(
        root: _TrieNode,
        image: "Image",
        n_jobs: int,
        data_layers: List[str] = None,
        extract_arrays: bool = True,
        backend: str = "joblib",
        slurm_params: Optional[Dict[str, Any]] = None,
):
    """Execute pipeline trie using shallow traversal with top-level parallelization.

    Implements hybrid approach:
    1. Serial traversal: Follow shared prefix path until reaching first branch point
    2. Parallel execution: Execute all divergent branches in parallel using specified backend

    This avoids nested parallelization deadlocks while maintaining shared prefix
    optimization and enabling true parallel execution of divergent branches.

    Args:
        root: Root node of the execution trie
        image: Original image to process
        n_jobs: Number of parallel jobs (-1 for all cores)
        data_layers: List of data layers to extract (e.g., ["rgb", "objmask"]).
            Only used if extract_arrays=True. If None, uses default set.
        extract_arrays: If True, yields dict of extracted arrays instead of
            full Image objects. Reduces memory by ~10×.
        backend: Execution backend - "joblib" (local) or "submitit" (SLURM cluster).
            Default "joblib".
        slurm_params: Configuration dict for submitit backend. Only used when
            backend="submitit". Default None.

    Yields:
        Tuple of (pipeline_name, result_data, json_config) where result_data is
        extracted arrays dict (if extract_arrays=True) or Image object otherwise
    """
    from joblib import Parallel, delayed

    # Default data layers if not specified
    if data_layers is None:
        data_layers = ["rgb", "gray", "enh_gray", "objmask", "objmap"]

    mem_start = _get_memory_usage()
    logger.debug(f"Initial memory usage: {mem_start:.1f} MB")

    # Step 1: Analyze trie structure (only if debug logging enabled)
    if logger.isEnabledFor(logging.DEBUG):
        trie_stats = _analyze_trie_structure(root)
        logger.debug(f"Trie structure: depth={trie_stats['max_depth']}, "
                     f"branch_points={trie_stats['branch_points']}, "
                     f"leaf_paths={trie_stats['total_leaf_paths']}")
    else:
        logger.info(
                "Executing pipeline trie (enable DEBUG logging for structure details)")

    # Step 2: Find first branch point and get shared prefix operations
    start_time = time.time()
    branch_node, shared_prefix_ops = _find_first_branch_point(root)
    shared_prefix_len = len(shared_prefix_ops)
    logger.debug(f"Shared prefix: {shared_prefix_len} operations")

    # Step 3: Execute shared prefix operations once
    if shared_prefix_ops:
        logger.info(f"Executing shared prefix ({shared_prefix_len} operations)...")
        prefix_start = time.time()
        mem_before_prefix = _get_memory_usage()
        current_image = image
        for op in shared_prefix_ops:
            current_image = op.apply(current_image.copy())
        prefix_time = time.time() - prefix_start
        mem_after_prefix = _get_memory_usage()
        logger.info(f"Shared prefix execution completed in {prefix_time:.2f}s "
                    f"(memory: {mem_before_prefix:.1f} MB → {mem_after_prefix:.1f} MB)")
    else:
        current_image = image

    # Step 4: Enumerate all downstream paths from branch point
    logger.debug("Enumerating downstream pipeline paths...")
    pipeline_specs = _enumerate_downstream_paths(branch_node, shared_prefix_ops)
    logger.info(f"Enumerated {len(pipeline_specs)} pipeline paths")

    if not pipeline_specs:
        logger.warning("No pipeline paths found from branch point")
        return

    # Step 5: Execute all pipelines in parallel using specified backend (top-level only)
    if len(pipeline_specs) > 1 and n_jobs != 1:
        logger.info(f"Executing {len(pipeline_specs)} pipelines in parallel "
                    f"(backend={backend}, n_jobs={n_jobs})")
        parallel_start = time.time()
        mem_before_parallel = _get_memory_usage()

        # Create task arguments for parallel execution
        task_args = [
            (current_image, [(name, ops)], shared_prefix_len, data_layers,
             extract_arrays)
            for name, ops in pipeline_specs
        ]

        # Execute using specified backend
        batch_results = _execute_parallel_tasks(
                func=_execute_concrete_pipeline_batch,
                task_args=task_args,
                backend=backend,
                n_jobs=n_jobs,
                slurm_params=slurm_params,
                desc="Parallel pipelines"
        )

        parallel_time = time.time() - parallel_start
        mem_after_parallel = _get_memory_usage()
        logger.info(f"Parallel execution completed in {parallel_time:.2f}s "
                    f"(memory: {mem_before_parallel:.1f} MB → {mem_after_parallel:.1f} MB)")

        # Flatten results from batch execution
        for batch in batch_results:
            for result in batch:
                yield result
    else:
        # Serial execution
        if len(pipeline_specs) > 1:
            logger.info(f"Executing {len(pipeline_specs)} pipelines serially "
                        f"(n_jobs=1)")
        serial_start = time.time()
        mem_before_serial = _get_memory_usage()

        results = _execute_concrete_pipeline_batch(
                current_image, pipeline_specs, shared_prefix_len, data_layers,
                extract_arrays
        )

        serial_time = time.time() - serial_start
        mem_after_serial = _get_memory_usage()
        logger.info(f"Serial execution completed in {serial_time:.2f}s "
                    f"(memory: {mem_before_serial:.1f} MB → {mem_after_serial:.1f} MB)")

        for result in results:
            yield result

    total_time = time.time() - start_time
    mem_end = _get_memory_usage()
    logger.info(f"Total trie execution time: {total_time:.2f}s "
                f"(memory: {mem_start:.1f} MB → {mem_end:.1f} MB)")


def _execute_single_pipeline(
        image: Image,
        operations: List[ImageOperation],
        param_config: Tuple[Dict[str, Any], ...],
        inplace: bool = False,
) -> Tuple[Image, Tuple[Dict[str, Any], ...], str]:
    """Execute a single pipeline with given parameter configuration.

    Args:
        image: Original image to process
        operations: Base operations (will be copied and updated)
        param_config: Parameter values for this configuration
        inplace: Whether to apply operations in-place. If True, operations are
            applied in-place to reduce memory usage (6× → 2× for 5-op pipeline).
            If False (default), each operation creates a copy. Only safe to use
            when caller won't reuse the input image. Default False.

    Returns:
        Tuple of (processed_image, param_config, json_config)
    """
    from phenotypic import ImagePipeline

    # Deep copy operations and update parameters
    ops_copy = []
    for op, params in zip(operations, param_config):
        op_copy = copy.deepcopy(op)
        for key, value in params.items():
            setattr(op_copy, key, value)
        ops_copy.append(op_copy)

    # Create and execute pipeline
    pipeline = ImagePipeline(ops=ops_copy)

    if inplace:
        # In-place execution: single copy + all operations in-place
        # ~3× memory reduction for typical 5-operation pipelines
        result_image = image.copy()
        result = pipeline.apply(result_image, inplace=True)
    else:
        # Standard execution: copy for each operation (original behavior)
        result = pipeline.apply(image.copy())

    # Serialize pipeline configuration to JSON
    json_config = pipeline.to_json_str()

    return result, param_config, json_config


def _build_results_dict(
        results: List[Tuple[Image, Tuple[Dict[str, Any], ...]]]
) -> Dict[Tuple, Image]:
    """Build dictionary mapping parameter configs to result images.

    Args:
        results: List of (image, param_config) tuples

    Returns:
        Dictionary with parameter tuples as keys, images as values
    """
    results_dict = {}
    for result_img, param_config in results:
        # Create hashable key from parameters
        key = tuple(tuple(sorted(params.items())) for params in param_config)
        results_dict[key] = result_img

    return results_dict


def _build_configs_dict(
        results: List[Tuple[Image, Tuple[Dict[str, Any], ...], str]],
        pipeline_name: str = None,
) -> Dict[str, str]:
    """Build dictionary mapping base layer names to serialized pipeline configs.

    Args:
        results: List of (image, param_config, json_config) tuples
        pipeline_name: Optional pipeline name prefix for MultiPipelineGridSearch

    Returns:
        Dictionary with base layer names as keys, JSON config strings as values
    """
    configs_dict = {}
    for result_idx, (result_img, param_config, json_config) in enumerate(results):
        param_str = _create_param_name_string(param_config)

        # Build layer name based on whether pipeline_name is provided
        if pipeline_name:
            layer_name = f"{pipeline_name}_{result_idx:03d}_{param_str}"
        else:
            layer_name = f"{result_idx:03d}_{param_str}"

        configs_dict[layer_name] = json_config

    return configs_dict
