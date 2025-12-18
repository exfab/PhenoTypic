"""Grid search utilities for pipeline parameter tuning and architecture comparison.

This module provides functions to perform parameter grid searches on ImagePipelines,
with multiple execution backends (joblib or submitit) and visualization options (napari or TIFF).

## Quick Start

### Interactive Exploration (Napari Mode - Default)
For exploring parameter combinations with visual feedback::

    from phenotypic import Image
    from phenotypic.enhance import GaussianBlur
    from phenotypic.util import PipelineGridSearch

    image = Image.imread('colony_plate.jpg')
    ops = [(GaussianBlur(sigma=1.0), {"sigma": [1.0, 2.0, 3.0]})]

    viewer, configs = PipelineGridSearch(image=image, ops=ops, n_jobs=-1)

### Batch Processing (TIFF Mode - Memory Efficient)
For large grid searches without visualization overhead::

    configs = PipelineGridSearch(
        image=image,
        ops=ops,
        save_tiff_dir="./grid_results",
        create_trial_view=True,
        n_jobs=-1
    )

### Cluster Execution (Submitit Backend)
For submitting to SLURM clusters::

    configs = PipelineGridSearch(
        image=image,
        ops=ops,
        backend="submitit",
        slurm_params={"slurm_partition": "gpu", "mem_gb": 32},
        save_tiff_dir="./cluster_results"
    )

## Key Features

- **Multiple Backends**: Choose between local (joblib) or cluster (submitit) execution
- **Memory Efficient**: TIFF mode achieves 7-13× memory reduction by eliminating napari
- **HTML Reports**: Generate visual quality control pages with thumbnails
- **Shared Prefix Optimization**: Automatically optimize MultiPipelineGridSearch (enabled by default)
- **Automatic Memory Management**: Garbage collection and array cleanup after each batch
- **Progress Tracking**: Terminal and Jupyter-compatible progress bars with ETA

## Progress Tracking

All grid search functions provide automatic progress bars:
- **Terminal mode**: Standard tqdm progress bars with ETA
- **Jupyter mode**: Interactive widget-based progress bars (requires ipywidgets)
- **Multi-level operations**: Nested progress bars show batch → group → pipeline progress

Progress bars are automatically enabled for:
- Batch processing (MultiPipelineGridSearch with adaptive_batching=True)
- Trie group iteration (optimize_shared_prefixes=True)
- Parallel pipeline execution (when n_jobs != 1)

## Memory Optimization for Large Grids

When processing large numbers of pipelines with memory-intensive operations (e.g., BM3D):

1. **Use TIFF mode instead of napari viewer:**
   ```python
   configs = MultiPipelineGridSearch(
       ...,
       save_tiff_dir="./results",
       create_trial_view=True,
       n_jobs=4,  # Reduce parallelism
       memory_limit_gb=8.0  # Set conservative limit
   )
   ```

2. **Reduce parallel workers to fit in memory:**
   - With 4096×4096 images: ~560 MB per pipeline
   - With n_jobs=4: ~2.2 GB peak memory
   - With n_jobs=-1 on 16 cores: ~9 GB peak memory (may OOM on 16 GB systems)

3. **Use submitit backend for cluster execution:**
   - Auto-disables napari viewer (cannot display on clusters)
   - Requires save_tiff_dir parameter
   - Ideal for processing hundreds of pipelines across many jobs

## Notes

- On macOS, you may see a harmless "MallocStackLogging" warning from the system malloc library.
  This warning does not affect functionality and can be safely ignored.
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

if TYPE_CHECKING:
    import napari
    from phenotypic import Image, ImageOperation

logger = logging.getLogger(__name__)

# Memory estimation overhead factor
# Empirical 20% overhead accounts for:
# - Python object allocation metadata
# - Garbage collector bookkeeping structures
# - Temporary arrays created during processing
# - NumPy view/slice overhead
_MEMORY_OVERHEAD_FACTOR = 1.2


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


def _ops_key(op: "ImageOperation", params: Dict[str, Any]) -> Tuple:
    """Create hashable key for operation tuple comparison.

    Args:
        op: ImageOperation instance
        params: Parameter dictionary for the operation

    Returns:
        Tuple containing (class_name, sorted_params) for hashing
    """
    return (type(op).__name__, tuple(sorted(params.items())))


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
        data_layers: List of layer names to extract (e.g., ["rgb", "objmask"])

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
        base_name: Base filename without extension (e.g., "001_sigma=2.0_rgb")

    Returns:
        Path to saved TIFF file

    Raises:
        ValueError: If array shape is not supported
        RuntimeError: If TIFF saving fails

    Example:
        >>> import numpy as np
        >>> array = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        >>> path = _save_array_as_tiff(array, "/tmp/results", "001_rgb")
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
            else:
                # Regular grayscale
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


def _validate_save_tiff_params(
        save_tiff_dir: Optional[str],
        create_trial_view: bool,
        backend: str,
) -> None:
    """Validate TIFF saving and backend parameters.

    Args:
        save_tiff_dir: Path to TIFF save directory (or None)
        create_trial_view: Whether HTML generation was requested
        backend: Execution backend name

    Raises:
        ValueError: If parameters are invalid or incompatible
        ImportError: If backend requires unavailable package
    """
    from pathlib import Path

    # Validate create_trial_view dependency
    if create_trial_view and save_tiff_dir is None:
        raise ValueError(
                "create_trial_view=True requires save_tiff_dir to be specified. "
                "HTML trial view can only be generated when saving TIFFs."
        )

    # Validate and create save_tiff_dir if provided
    if save_tiff_dir is not None:
        try:
            save_path = Path(save_tiff_dir)
            save_path.mkdir(parents=True, exist_ok=True)

            # Test write permissions
            test_file = save_path / ".write_test"
            test_file.touch()
            test_file.unlink()

        except Exception as e:
            raise ValueError(
                    f"Cannot write to save_tiff_dir '{save_tiff_dir}': {e}"
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

        # Submitit backend requires TIFF mode (cluster jobs cannot display napari)
        if save_tiff_dir is None:
            raise ValueError(
                    "save_tiff_dir is required when backend='submitit'. "
                    "Cluster jobs cannot create interactive napari viewers. "
                    "Please specify a directory to save TIFF files."
            )


def _create_submitit_executor(
        slurm_params: Optional[Dict[str, Any]] = None,
) -> "submitit.AutoExecutor":
    """Create and configure submitit executor for SLURM jobs.

    Args:
        slurm_params: Optional configuration dict with keys:
            - folder: Log folder path (default: "./submitit_logs")
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
        backend: Execution backend - "joblib" or "submitit"
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
            failure_msg = "\n".join(
                    f"  Task {idx}: {error}"
                    for idx, error in failed_tasks
            )
            raise RuntimeError(
                    f"{len(failed_tasks)} task(s) failed:\n{failure_msg}"
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
            failure_msg = "\n".join(
                    f"  Task {idx} (Job {job_id}): {error}"
                    for idx, job_id, error in failed_jobs
            )
            raise RuntimeError(
                    f"{len(failed_jobs)} job(s) failed:\n{failure_msg}"
            )

        logger.info(f"All {len(jobs)} jobs completed successfully")
        return results

    else:
        # Should never reach here due to validation
        raise ValueError(f"Unknown backend: {backend}")


def _create_trial_view_html(
        save_dir: Union[str, Path],
        configs_dict: Dict[str, str],
        data_layers: List[str],
) -> Path:
    """Generate HTML overview page with thumbnails of all saved TIFF files.

    Creates an HTML page with:
    - Grid layout showing all results
    - Thumbnails for each data layer
    - Pipeline configuration details
    - Responsive CSS for viewing in browser

    Args:
        save_dir: Directory containing TIFF files
        configs_dict: Dictionary mapping layer base names to JSON configs
        data_layers: List of layer names that were saved

    Returns:
        Path to generated trial_overview.html file

    Raises:
        RuntimeError: If HTML generation fails

    Example:
        >>> configs = {"001_sigma=2.0": "{...json...}", ...}
        >>> html_path = _create_trial_view_html(
        ...     "/tmp/results", configs, ["rgb", "objmask"]
        ... )
    """
    from PIL import Image as PIL_Image

    save_path = Path(save_dir)
    html_file = save_path / "trial_overview.html"
    thumbnails_dir = save_path / "thumbnails"
    thumbnails_dir.mkdir(exist_ok=True)

    logger.info(f"Generating trial view HTML with {len(configs_dict)} results")

    # Group results by base name
    result_groups = {}
    for base_name in configs_dict.keys():
        # Extract pipeline name and params (before layer suffix)
        result_groups[base_name] = {}

        # Find TIFF files for this result
        for layer in data_layers:
            tiff_pattern = f"{base_name}_{layer}.tiff"
            tiff_path = save_path / tiff_pattern

            if tiff_path.exists():
                # Create thumbnail
                try:
                    img = PIL_Image.open(tiff_path)
                    img.thumbnail((200, 200), PIL_Image.Resampling.LANCZOS)
                    thumb_name = f"{base_name}_{layer}_thumb.jpg"
                    thumb_path = thumbnails_dir / thumb_name

                    # Convert to RGB for JPEG saving if needed
                    if img.mode == 'I;16' or img.mode == 'L':
                        # Normalize to 0-255 for display
                        img_array = np.array(img)
                        if img_array.max() > 255:
                            img_array = (img_array / img_array.max() * 255).astype(
                                np.uint8)
                        else:
                            img_array = img_array.astype(np.uint8)
                        img = PIL_Image.fromarray(img_array)

                    img.save(thumb_path, format='JPEG', quality=85)
                    result_groups[base_name][layer] = f"thumbnails/{thumb_name}"

                except Exception as e:
                    logger.warning(f"Failed to create thumbnail for {tiff_path}: {e}")
                    result_groups[base_name][layer] = None

    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pipeline Grid Search - Trial Overview</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
        }}
        .results-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .result-card {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 15px;
        }}
        .result-card h3 {{
            margin-top: 0;
            color: #007bff;
            font-size: 14px;
            word-wrap: break-word;
            font-family: monospace;
        }}
        .layers {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
            gap: 10px;
        }}
        .layer {{
            text-align: center;
        }}
        .layer-name {{
            font-size: 12px;
            font-weight: 600;
            color: #666;
            margin-bottom: 5px;
        }}
        .layer img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .no-thumbnail {{
            width: 180px;
            height: 180px;
            background: #f0f0f0;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #999;
            font-size: 12px;
            border-radius: 4px;
        }}
        .timestamp {{
            text-align: right;
            color: #888;
            font-size: 12px;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <h1>Pipeline Grid Search Results</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p>Total results: {len(result_groups)}</p>

    <div class="results-grid">
"""

    # Add each result card
    for base_name, layers in result_groups.items():
        html_content += f"""
        <div class="result-card">
            <h3>{base_name}</h3>
            <div class="layers">
"""

        for layer in data_layers:
            thumb_path = layers.get(layer)
            if thumb_path:
                html_content += f"""
                <div class="layer">
                    <div class="layer-name">{layer}</div>
                    <img src="{thumb_path}" alt="{layer}" loading="lazy">
                </div>
"""
            else:
                html_content += f"""
                <div class="layer">
                    <div class="layer-name">{layer}</div>
                    <div class="no-thumbnail">No data</div>
                </div>
"""

        html_content += """
            </div>
        </div>
"""

    html_content += """
    </div>

    <div class="timestamp">
        Generated by PhenoTypic Pipeline Grid Search
    </div>
</body>
</html>
"""

    # Write HTML file
    try:
        html_file.write_text(html_content, encoding='utf-8')
        logger.info(f"Created trial view HTML: {html_file}")
        return html_file
    except Exception as e:
        raise RuntimeError(f"Failed to create HTML file: {e}") from e


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

    # Get base image data sizes
    base_size = 0

    # RGB array size (if present)
    rgb_data = image.rgb[:]
    if rgb_data is not None and rgb_data.size > 0:
        base_size += sys.getsizeof(rgb_data)

    # Gray and enhanced gray (always present)
    gray_data = image.gray[:]
    if gray_data is not None:
        base_size += sys.getsizeof(gray_data)

    enh_gray_data = image.enh_gray[:]
    if enh_gray_data is not None:
        base_size += sys.getsizeof(enh_gray_data)

    if extract_arrays:
        # With array extraction: 1 full image copy + extracted arrays
        # Estimate extracted array sizes
        extracted_size = 0
        for layer in data_layers:
            if layer == "rgb" and rgb_data is not None and rgb_data.size > 0:
                extracted_size += (np.prod(rgb_data.shape) * rgb_data.itemsize)
            elif layer == "gray" and gray_data is not None:
                extracted_size += (np.prod(gray_data.shape) * gray_data.itemsize)
            elif layer == "enh_gray" and enh_gray_data is not None:
                extracted_size += (np.prod(enh_gray_data.shape) * enh_gray_data.itemsize)
            elif layer in ["objmask", "objmap"]:
                # Estimate label map size (uint16 label maps: 2 bytes per pixel)
                if gray_data is not None:
                    extracted_size += (np.prod(gray_data.shape) * np.dtype(np.uint16).itemsize)

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
        ...     n_jobs=-1
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
    max_parallel = max(1, memory_limit // memory_per_pipeline)

    # Respect user's n_jobs if specified, but limit by memory
    if n_jobs == -1:
        # Use all CPU cores, but limited by memory
        jobs_per_batch = min(max_parallel, psutil.cpu_count())
    else:
        # Use user-specified jobs, but limited by memory
        jobs_per_batch = min(max_parallel, n_jobs)

    # Batch size: process enough to utilize parallel workers efficiently
    # Be conservative to prevent OOM: use 2× for ideal, but cap at reasonable limit
    ideal_batch_size = jobs_per_batch * 2
    # Conservative minimum: at least 2 jobs per batch, but max 8 pipelines
    safe_batch_size = min(max(2, jobs_per_batch), 8)
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
    
    Builds temporary trie, identifies natural groups where pipelines share
    maximum operation sequence before diverging. Critically distinguishes between
    parameter sweeps (same operation, different params) and structural divergence
    (different operation types).
    
    Args:
        concrete_configs: List of concrete pipeline configurations
        
    Returns:
        List of pipeline groups sharing longest possible operation sequence.
        Parameter sweeps are kept together in one group (enables parallelization).
        Structural divergence creates separate groups.
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

        # Multiple children = potential branch point
        # Check if all children are same operation (parameter sweep) or different (structural divergence)
        child_ops = [child.op for child in node.children.values() if
                     child.op is not None]

        if child_ops:
            # Check if all children are the same operation CLASS (ignore parameter values)
            op_types = set(type(op).__name__ for op in child_ops)

            if len(op_types) == 1:
                # All children are same operation type with different parameters
                # This is a PARAMETER SWEEP, not structural divergence
                # Don't split - merge ALL downstream pipelines into ONE group
                all_pipeline_names = []
                for child in node.children.values():
                    child_groups = _collect_groups_from_node(child)
                    # Flatten: merge all child groups into single group
                    for group in child_groups:
                        all_pipeline_names.extend(group)
                # Return as single merged group
                return [all_pipeline_names] if all_pipeline_names else []

        # Different operation types = structural divergence
        # Split into separate groups (one per child subtree)
        all_groups = []
        for child in node.children.values():
            child_groups = _collect_groups_from_node(child)
            all_groups.extend(child_groups)
        return all_groups

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
            return 0, 0, 1  # Leaf node

        max_depth = 0
        total_branches = 0
        total_paths = 1

        for child in node.children.values():
            child_depth, child_branches, child_paths = count_descendants(child)
            max_depth = max(max_depth, child_depth + 1)
            total_branches += child_branches
            total_paths *= child_paths

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
    """
    from phenotypic import ImagePipeline

    results = []

    for pipeline_name, all_ops in pipeline_specs:
        # Get only the operations after the shared prefix
        remaining_ops = all_ops[shared_prefix_len:]

        if not remaining_ops:
            # No operations to apply - result is the input image
            result_image = image.copy()
            full_ops = all_ops
        else:
            # Deep copy operations to avoid parameter sharing between executions
            ops_copy = []
            for op in remaining_ops:
                ops_copy.append(copy.deepcopy(op))

            # Apply remaining operations to the already-processed image
            result_image = image.copy()
            for op in ops_copy:
                result_image = op.apply(result_image)

            full_ops = all_ops

        # Create pipeline with FULL ops for serialization
        pipeline = ImagePipeline(ops=copy.deepcopy(full_ops))

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

    # Step 1: Analyze trie structure
    trie_stats = _analyze_trie_structure(root)
    logger.info(f"Trie structure: depth={trie_stats['max_depth']}, "
                f"branch_points={trie_stats['branch_points']}, "
                f"leaf_paths={trie_stats['total_leaf_paths']}")

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

    # Step 5: Execute all pipelines in parallel using joblib (top-level only)
    if len(pipeline_specs) > 1 and n_jobs != 1:
        from tqdm_joblib import tqdm_joblib

        logger.info(f"Executing {len(pipeline_specs)} pipelines in parallel "
                    f"(n_jobs={n_jobs})")
        parallel_start = time.time()
        mem_before_parallel = _get_memory_usage()

        # Create joblib delayed tasks
        tasks = [
            delayed(_execute_concrete_pipeline_batch)(
                    current_image,
                    [(name, ops)],
                    shared_prefix_len,
                    data_layers,
                    extract_arrays,
            )
            for name, ops in pipeline_specs
        ]

        # Execute in parallel with progress bar
        with tqdm_joblib(
                desc="Parallel pipelines",
                total=len(pipeline_specs),
                unit="pipeline",
        ):
            batch_results = Parallel(n_jobs=n_jobs)(tasks)

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


def _add_original_layers(viewer: napari.Viewer, image: Image) -> None:
    """Add original RGB and gray reference layers to viewer.

    Args:
        viewer: Napari viewer instance
        image: Original image
    """
    viewer.add_image(image.rgb[:], name="Original_RGB", rgb=True)
    viewer.add_image(image.gray[:], name="Original_Gray", colormap="gray")


def _add_result_layer(
        viewer: napari.Viewer,
        result_data: Union["Image", Any],
        data_layer: str,
        layer_name: str,
) -> None:
    """Add a single data layer to napari viewer.

    Accepts either a full Image object or a numpy array directly. When
    result_data is an Image, extracts the requested layer. When result_data
    is already an array, uses it directly.

    Args:
        viewer: Napari viewer instance
        result_data: Either Image object or numpy array for this layer
        data_layer: Which data to add ("rgb", "gray", etc.)
        layer_name: Name for the layer in napari

    Note:
        For backwards compatibility, this function handles both Image objects
        (legacy behavior) and numpy arrays (memory-optimized behavior).
    """
    import numpy as np

    try:
        # Check if result_data is a numpy array (new behavior)
        if isinstance(result_data, np.ndarray):
            data = result_data
        else:
            # result_data is an Image object (legacy behavior)
            if data_layer == "rgb":
                data = result_data.rgb[:]
            elif data_layer == "gray":
                data = result_data.gray[:]
            elif data_layer == "enh_gray":
                data = result_data.enh_gray[:]
            elif data_layer == "objmask":
                data = result_data.objmask[:]
            elif data_layer == "objmap":
                data = result_data.objmap[:]
            else:
                return  # Unknown layer type

        # Add to viewer based on layer type
        if data_layer == "rgb":
            viewer.add_image(data, name=layer_name, rgb=True)
        elif data_layer in ["gray", "enh_gray"]:
            viewer.add_image(data, name=layer_name, colormap="gray")
        elif data_layer in ["objmask", "objmap"]:
            if data.any():  # Only add if not empty
                viewer.add_labels(data, name=layer_name)

    except Exception as e:
        # Gracefully handle missing data
        print(f"Warning: Could not add layer {layer_name}: {e}")


def _add_result_layers(
        viewer: napari.Viewer,
        result_img: Image,
        param_config: Tuple[Dict[str, Any], ...],
        data_layers: List[str],
        result_idx: int,
) -> None:
    """Add all requested data layers for a single result.

    Args:
        viewer: Napari viewer instance
        result_img: Processed image result
        param_config: Parameter configuration for this result
        data_layers: List of data layers to add
        result_idx: Index of this result in the grid
    """
    param_str = _create_param_name_string(param_config)

    for data_layer in data_layers:
        layer_name = f"{result_idx:03d}_{param_str}_{data_layer}"
        _add_result_layer(viewer, result_img, data_layer, layer_name)


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


def PipelineGridSearch(
        image: Image,
        ops: List[Tuple[ImageOperation, Dict[str, List[Any]]]],
        data_layers: List[str] = ["rgb", "gray", "enh_gray", "objmask", "objmap"],
        n_jobs: int = -1,
        inplace: bool = False,
        viewer_title: str = "Pipeline Grid Search",
        save_tiff_dir: Optional[str] = None,
        create_trial_view: bool = False,
        backend: str = "joblib",
        slurm_params: Optional[Dict[str, Any]] = None,
) -> Union[Dict[str, str], Tuple[napari.Viewer, Dict[str, str]]]:
    """Execute parameter grid search with parallel pipelines and napari visualization.

    Generates all combinations of provided parameters, executes ImagePipeline for each
    combination in parallel. Results are visualized in napari (default) or saved as
    TIFF files for memory-efficient processing.

    Args:
        image: Single Image object to process. All parameter combinations will be
            applied to this image.
        ops: List of (operation, params_dict) tuples. Each tuple contains an
            ImageOperation instance and a dictionary mapping parameter names to
            lists of values to test. Empty dict means no parameters to vary for that
            operation. Example: [(GaussianBlur(sigma=1.0), {"sigma": [1.0, 2.0, 3.0]}),
                                 (OtsuDetector(), {})]
        data_layers: Which image data to display/save. Valid options: "rgb", "gray",
            "enh_gray", "objmask", "objmap". Defaults to all available layers.
        n_jobs: Number of parallel jobs. -1 uses all available cores. Default -1.
        inplace: Whether to apply operations in-place. If True, reduces memory usage
            by ~3× for typical pipelines. Only safe when input image won't be reused.
            Default False.
        viewer_title: Title for the napari viewer window (napari mode only).
        save_tiff_dir: Optional path to directory for saving TIFF files. When provided,
            disables napari viewer creation and saves all result layers as TIFF files
            instead. Enables 7-13× memory reduction. Creates directory if it doesn't
            exist. Default None (napari mode).
        create_trial_view: Generate HTML overview page with thumbnails of all saved
            TIFF files. Only valid when save_tiff_dir is specified. Default False.
        backend: Execution backend - "joblib" (local) or "submitit" (SLURM cluster).
            Default "joblib".
        slurm_params: Configuration dict for submitit backend (keys: folder, timeout_min,
            slurm_partition, mem_gb, cpus_per_task, etc.). Only used when backend="submitit".
            Default None.

    Returns:
        When save_tiff_dir is None (napari mode):
            Tuple[napari.Viewer, Dict[str, str]]: (viewer, configs_dict)
                - viewer: napari.Viewer with all results as layers
                - configs_dict: Maps layer names to JSON pipeline configs

        When save_tiff_dir is provided (TIFF mode):
            Dict[str, str]: configs_dict only
                - Keys: base layer names (e.g., "001_sigma=2.0")
                - Values: JSON pipeline configuration strings
                - TIFF files saved to save_tiff_dir with names like
                  "{idx:03d}_{params}_{layer}.tiff"

    Raises:
        ValueError: If parameters are invalid (create_trial_view without save_tiff_dir,
            invalid backend, save_tiff_dir not writable).
        ImportError: If backend="submitit" but submitit is not installed.
        RuntimeError: If TIFF saving, HTML generation, or job execution fails.

    Example:
        >>> from phenotypic import Image
        >>> from phenotypic.enhance import GaussianBlur
        >>> from phenotypic.detect import OtsuDetector
        >>> from phenotypic.util import PipelineGridSearch
        >>>
        >>> image = Image.imread('colony_plate.jpg')
        >>> ops = [(GaussianBlur(sigma=1.0), {"sigma": [1.0, 2.0, 3.0]}),
        ...        (OtsuDetector(), {})]
        >>>
        >>> # Napari mode (default - interactive exploration)
        >>> viewer, configs = PipelineGridSearch(image=image, ops=ops, n_jobs=-1)
        >>>
        >>> # TIFF mode (memory-efficient - saves results to disk)
        >>> configs = PipelineGridSearch(
        ...     image=image, ops=ops,
        ...     save_tiff_dir="./grid_search_results"
        ... )
        >>>
        >>> # TIFF mode with HTML trial view for visual QC
        >>> configs = PipelineGridSearch(
        ...     image=image, ops=ops,
        ...     save_tiff_dir="./grid_search_results",
        ...     create_trial_view=True
        ... )
        >>>
        >>> # Cluster execution with submitit
        >>> configs = PipelineGridSearch(
        ...     image=image, ops=ops,
        ...     backend="submitit",
        ...     slurm_params={"slurm_partition": "gpu", "mem_gb": 32}
        ... )
    """
    # 1. Validate inputs
    _validate_inputs(ops, data_layers)
    _validate_save_tiff_params(save_tiff_dir, create_trial_view, backend)

    # 2. Unpack ops tuples into separate lists
    operations, parameters = _unpack_ops_tuples(ops)

    # 3. Generate parameter combinations
    all_configs = _generate_param_combinations(parameters)

    # 4. Prepare task arguments
    task_args = [(image, operations, config, inplace) for config in all_configs]

    # 5. Execute pipelines with selected backend
    results = _execute_parallel_tasks(
            func=_execute_single_pipeline,
            task_args=task_args,
            backend=backend,
            n_jobs=n_jobs,
            slurm_params=slurm_params,
            desc="PipelineGridSearch",
    )

    # 6. Process results based on mode
    if save_tiff_dir:
        # ============ TIFF SAVING MODE ============
        configs_dict = {}
        tiff_files = []

        logger.info(f"Saving {len(results)} results as TIFFs to {save_tiff_dir}")

        for idx, (result_img, param_config, json_config) in enumerate(results):
            # Extract arrays
            extracted = _extract_data_layers(result_img, data_layers)

            # Save each layer as TIFF
            param_str = _create_param_name_string(param_config)
            base_name = f"{idx:03d}_{param_str}"

            for layer_name, array_data in extracted.items():
                tiff_path = _save_array_as_tiff(
                        array_data,
                        save_tiff_dir,
                        f"{base_name}_{layer_name}"
                )
                tiff_files.append(tiff_path)

            # Store config
            configs_dict[base_name] = json_config

            # Free memory immediately
            del result_img, extracted
            gc.collect()

        logger.info(f"Saved {len(tiff_files)} TIFF files")

        # Generate HTML view if requested
        if create_trial_view:
            html_path = _create_trial_view_html(
                    save_tiff_dir, configs_dict, data_layers
            )
            logger.info(f"Created trial view: {html_path}")

        # Return only configs (no viewer)
        return configs_dict

    else:
        # ============ NAPARI VIEWER MODE ============
        import napari

        viewer = napari.Viewer(title=viewer_title)
        _add_original_layers(viewer, image)

        configs_dict = {}

        for idx, (result_img, param_config, json_config) in enumerate(results):
            _add_result_layers(viewer, result_img, param_config, data_layers, idx)

            param_str = _create_param_name_string(param_config)
            layer_name = f"{idx:03d}_{param_str}"
            configs_dict[layer_name] = json_config

            del result_img

        # Delete remaining result references for memory cleanup
        del results
        return viewer, configs_dict


def MultiPipelineGridSearch(
        image: Image,
        pipeline_configs: List[Dict[str, Any]],
        data_layers: List[str] = ["rgb", "gray", "enh_gray", "objmask", "objmap"],
        n_jobs: int = -1,
        inplace: bool = False,
        save_tiff_dir: Optional[str] = None,
        viewer_title: str = "Multi-Pipeline Grid Search",
        optimize_shared_prefixes: bool = True,
        memory_limit_gb: float = None,
        adaptive_batching: bool = True,
        create_trial_view: bool = False,
        backend: str = "joblib",
        slurm_params: Optional[Dict[str, Any]] = None,
) -> Union[Dict[str, str], Tuple[napari.Viewer, Dict[str, str]]]:
    """Execute grid search across multiple pipeline configurations.

    Allows comparing different algorithm combinations and architectures. Each pipeline
    configuration is a different set of operations (e.g., GaussianBlur+Otsu vs.
    MedianFilter+Canny). For each pipeline, all parameter combinations are tested
    in parallel. Results are visualized in napari (default) or saved as TIFF files.

    When optimize_shared_prefixes=True (default), pipelines with shared starting
    operations are automatically optimized. Shared operations (same class type and
    parameters) are executed only once, and the intermediate result is reused for
    divergent branches. This can significantly reduce computation time when multiple
    pipelines share common preprocessing steps.

    Args:
        image: Single Image object to process. All pipelines and parameter
            combinations will be applied to this image.
        pipeline_configs: List of pipeline configuration dictionaries. Each dict
            must contain:
            - "name" (str): Descriptive name for this pipeline (e.g., "GaussianBlur_Otsu")
            - "ops" (List[Tuple]): List of (operation, params_dict) tuples
        data_layers: Which image data to display/save. Valid options: "rgb", "gray",
            "enh_gray", "objmask", "objmap". Defaults to all available.
        n_jobs: Number of parallel jobs. -1 uses all available cores. When
            adaptive_batching=True, this value may be reduced automatically to fit
            within memory limits. Default -1.
        inplace: Whether to apply operations in-place. If True, reduces memory usage
            by ~3× for typical pipelines (6× → 2× for 5-op pipeline). Only safe when
            input image won't be reused. Default False.
        optimize_shared_prefixes: If True (default), automatically detect and optimize
            pipelines with shared starting operations. Shared operations are executed
            once and intermediate results are reused. Set to False to use original
            linear execution (useful for debugging or comparison).
        memory_limit_gb: Memory limit in GB for adaptive batching. If None (default),
            uses 75% of available system memory. Only used when adaptive_batching=True.
        adaptive_batching: If True (default), automatically batch pipeline execution
            to stay within memory limits. Calculates optimal batch size based on image
            size and available memory. All results still appear in single napari viewer.
            Set to False to process all pipelines at once (may cause OOM for large grids).
        viewer_title: Title for the napari viewer window (napari mode only).
        save_tiff_dir: Optional path to directory for saving TIFF files. When provided,
            disables napari viewer creation and saves all result layers as TIFF files
            instead. Enables 7-13× memory reduction. Creates directory if it doesn't
            exist. Default None (napari mode).
        create_trial_view: Generate HTML overview page with thumbnails of all saved
            TIFF files. Only valid when save_tiff_dir is specified. Default False.
        backend: Execution backend - "joblib" (local) or "submitit" (SLURM cluster).
            Default "joblib".
        slurm_params: Configuration dict for submitit backend (keys: folder, timeout_min,
            slurm_partition, mem_gb, cpus_per_task, etc.). Only used when backend="submitit".
            Default None.

    Returns:
        When save_tiff_dir is None (napari mode):
            Tuple[napari.Viewer, Dict[str, str]]: (viewer, configs_dict)
                - viewer: napari.Viewer with all results as layers
                - configs_dict: Maps layer names to JSON pipeline configs

        When save_tiff_dir is provided (TIFF mode):
            Dict[str, str]: configs_dict only
                - Keys: base layer names (e.g., "pipeline_001_sigma=2.0")
                - Values: JSON pipeline configuration strings
                - TIFF files saved to save_tiff_dir

    Raises:
        ValueError: If pipeline_configs is empty, contains invalid structures,
            or if operations/parameters/parameters are invalid (same as before) or
            if new parameter combinations are invalid (create_trial_view without
            save_tiff_dir, invalid backend, save_tiff_dir not writable).
        ImportError: If backend="submitit" but submitit is not installed.
        RuntimeError: If TIFF saving, HTML generation, or job execution fails.

    Example:
        >>> from phenotypic import Image
        >>> from phenotypic.enhance import GaussianBlur, MedianFilter
        >>> from phenotypic.detect import OtsuDetector
        >>> from phenotypic.util import MultiPipelineGridSearch
        >>>
        >>> image = Image.imread('colony_plate.jpg')
        >>>
        >>> pipeline_configs = [
        ...     {
        ...         "name": "GaussianBlur_Otsu",
        ...         "ops": [
        ...             (GaussianBlur(sigma=1.0), {"sigma": [1.0, 2.0, 3.0]}),
        ...             (OtsuDetector(), {})
        ...         ]
        ...     },
        ...     {
        ...         "name": "MedianFilter_Otsu",
        ...         "ops": [
        ...             (MedianFilter(size=3), {"size": [3, 5, 7]}),
        ...             (OtsuDetector(), {})
        ...         ]
        ...     }
        ... ]
        >>>
        >>> viewer = MultiPipelineGridSearch(
        ...     image=image,
        ...     pipeline_configs=pipeline_configs,
        ...     n_jobs=-1
        ... )
    """
    # Validate inputs
    _validate_pipeline_configs(pipeline_configs)
    _validate_save_tiff_params(save_tiff_dir, create_trial_view, backend)

    # When using submitit backend, disable trie optimization since jobs are already parallel
    # The trie structure is still available for HTML view generation via all_configs naming
    if backend == "submitit":
        logger.info(
            "Submitit backend detected: disabling trie optimization for execution "
            "(jobs are already parallelized)")
        optimize_shared_prefixes = False

        # Enforce TIFF mode for submitit (cluster jobs cannot display napari viewer)
        if save_tiff_dir is None:
            raise ValueError(
                    "save_tiff_dir is required when backend='submitit'. "
                    "Cluster jobs cannot create interactive napari viewers. "
                    "Please specify a directory to save TIFF files."
            )

    # Storage for combined configs and results
    all_configs = {}
    tiff_files = []

    # Create viewer OR prepare for TIFF mode
    if save_tiff_dir:
        # TIFF mode - no viewer needed
        viewer = None
        logger.info(f"TIFF saving mode: will save results to {save_tiff_dir}")
        # Enable inplace operations for memory efficiency in TIFF mode
        if inplace is False:
            logger.info("TIFF mode: enabling inplace=True for 3× memory reduction")
            inplace = True  # Safe because images are discarded after saving
    else:
        # Napari mode - create viewer
        import napari

        viewer = napari.Viewer(title=viewer_title)
        # Add original reference layers once
        _add_original_layers(viewer, image)

    # Choose execution strategy
    if optimize_shared_prefixes:
        # Step 1: Expand all parameter combinations into concrete pipelines
        concrete_configs = _expand_pipeline_configs_to_concrete(pipeline_configs)
        total_pipelines = len(concrete_configs)
        logger.info(f"Expanded {len(pipeline_configs)} pipeline configs into "
                    f"{total_pipelines} concrete pipelines")

        # Add memory usage warning for large n_jobs
        if total_pipelines > 20 and n_jobs == -1:
            import psutil

            available_gb = psutil.virtual_memory().available / (1024 ** 3)
            logger.warning(
                    f"Processing {total_pipelines} pipelines with n_jobs=-1 (all cores). "
                    f"Available memory: {available_gb:.1f} GB. "
                    f"Consider reducing n_jobs or setting memory_limit_gb if OOM errors occur."
            )

        # Step 2: Determine batch size and parallelism
        if adaptive_batching and total_pipelines > 1:
            # Estimate memory per pipeline
            avg_ops = sum(len(c["ops"]) for c in concrete_configs) / len(
                    concrete_configs)
            memory_per_pipeline = _estimate_pipeline_memory(
                    image, int(avg_ops), data_layers, extract_arrays=True
            )

            # Calculate optimal batching
            batch_size, jobs_per_batch = _calculate_optimal_batch_size(
                    total_pipelines, memory_per_pipeline, memory_limit_gb, n_jobs
            )

            logger.info(f"Adaptive batching: {total_pipelines} pipelines in batches of "
                        f"{batch_size} with {jobs_per_batch} parallel jobs")
            logger.info(
                    f"Estimated memory per pipeline: "
                    f"{memory_per_pipeline / 1024 ** 2:.1f} MB")
        else:
            # Process all at once (original behavior)
            batch_size = total_pipelines
            jobs_per_batch = n_jobs
            logger.info(f"Processing {total_pipelines} pipelines without batching")

        # Step 3: Process in batches
        from tqdm.auto import tqdm

        global_start_time = time.time()
        batch_ranges = list(range(0, total_pipelines, batch_size))
        total_batches = len(batch_ranges)

        for batch_idx, batch_start in enumerate(
                tqdm(
                        batch_ranges,
                        desc="Batches",
                        unit="batch",
                        disable=(total_batches == 1),
                )
        ):
            batch_end = min(batch_start + batch_size, total_pipelines)
            batch_configs = concrete_configs[batch_start:batch_end]
            batch_pipeline_count = batch_end - batch_start

            if adaptive_batching and total_pipelines > batch_size:
                batch_num = (batch_start // batch_size) + 1
                mem_before = _get_memory_usage()
                logger.info(f"Processing batch {batch_num}/{total_batches}: "
                            f"pipelines {batch_start} to {batch_end - 1} "
                            f"({batch_pipeline_count} pipelines, {mem_before:.1f} MB)")

            batch_start_time = time.time()

            # Group batch pipelines by longest shared prefix
            logger.debug(
                f"Grouping {batch_pipeline_count} pipelines by longest shared prefix")
            trie_groups = _group_pipelines_by_longest_prefix(batch_configs)
            logger.info(f"Batch contains {len(trie_groups)} distinct trie groups")

            # Process each trie group sequentially
            logger.debug(f"Starting sequential trie group processing")
            pipeline_results_count = 0
            for pipeline_name, result_data, json_config in _process_trie_groups_sequentially(
                    image, trie_groups, jobs_per_batch,
                    data_layers=data_layers,
                    extract_arrays=True,
                    backend=backend,
                    slurm_params=slurm_params
            ):
                pipeline_results_count += 1
                # result_data is now a dict of {layer_name: np.ndarray}

                if save_tiff_dir:
                    # ============ TIFF SAVING MODE ============
                    # Save each layer as TIFF
                    for layer_name, array_data in result_data.items():
                        tiff_path = _save_array_as_tiff(
                                array_data,
                                save_tiff_dir,
                                f"{pipeline_name}_{layer_name}"
                        )
                        tiff_files.append(tiff_path)

                    # Store config
                    all_configs[pipeline_name] = json_config

                    # Free memory immediately
                    del result_data

                else:
                    # ============ NAPARI VIEWER MODE ============
                    # Add layers with pipeline name
                    for data_layer, array_data in result_data.items():
                        layer_name = f"{pipeline_name}_{data_layer}"
                        _add_result_layer(viewer, array_data, data_layer, layer_name)

                    # Store config always
                    all_configs[pipeline_name] = json_config

                    # Free memory immediately
                    del result_data

            batch_elapsed = time.time() - batch_start_time

            # Explicit garbage collection after each batch
            import gc

            gc.collect()
            mem_after = _get_memory_usage()

            if adaptive_batching and total_pipelines > batch_size:
                logger.info(f"Batch {batch_num}/{total_batches} complete: "
                            f"{pipeline_results_count} pipelines in {batch_elapsed:.2f}s, "
                            f"memory: {mem_before:.1f} MB → {mem_after:.1f} MB")
            else:
                logger.info(f"Batch processing complete: "
                            f"{pipeline_results_count} pipelines in {batch_elapsed:.2f}s, "
                            f"memory: {mem_after:.1f} MB")

        global_elapsed = time.time() - global_start_time
        logger.info(f"All batches completed in {global_elapsed:.2f}s")


    else:
        # Original linear execution path (non-optimized)
        logger.info("Using non-optimized path (optimize_shared_prefixes=False)")

        # Process each pipeline configuration
        for config_idx, config in enumerate(pipeline_configs):
            pipeline_name = config["name"]
            ops = config["ops"]

            # Unpack ops tuples
            operations, parameters = _unpack_ops_tuples(ops)

            # Generate parameter combinations for this pipeline
            param_configs = _generate_param_combinations(parameters)

            # Build task arguments for _execute_parallel_tasks
            # Each task is: (image, operations, param_config, inplace)
            task_args = [
                (image, operations, param_config, inplace)
                for param_config in param_configs
            ]

            logger.info(f"Processing pipeline '{pipeline_name}' with {len(task_args)} parameter configs "
                       f"using {backend} backend")

            # Execute this pipeline's grid search using appropriate backend
            results = _execute_parallel_tasks(
                    func=_execute_single_pipeline,
                    task_args=task_args,
                    backend=backend,
                    n_jobs=n_jobs,
                    slurm_params=slurm_params,
                    desc=f"Pipeline '{pipeline_name}'",
            )

            # Process results
            for result_idx, (result_img, param_config, json_config) in enumerate(
                    results):
                param_str = _create_param_name_string(param_config)
                config_key = f"{pipeline_name}_{result_idx:03d}_{param_str}"

                if save_tiff_dir:
                    # ============ TIFF SAVING MODE ============
                    # Extract arrays
                    extracted = _extract_data_layers(result_img, data_layers)

                    # Save each layer as TIFF
                    for layer_name, array_data in extracted.items():
                        tiff_path = _save_array_as_tiff(
                                array_data,
                                save_tiff_dir,
                                f"{config_key}_{layer_name}"
                        )
                        tiff_files.append(tiff_path)

                    # Store config
                    all_configs[config_key] = json_config

                    # Free memory immediately
                    del result_img, extracted

                else:
                    # ============ NAPARI VIEWER MODE ============
                    # Add layers with pipeline name prefix
                    for data_layer in data_layers:
                        layer_name = f"{config_key}_{data_layer}"
                        _add_result_layer(viewer, result_img, data_layer, layer_name)

                    # Store config always
                    all_configs[config_key] = json_config

                    # Free memory
                    del result_img

    # Generate HTML view if in TIFF mode
    if save_tiff_dir:
        logger.info(f"Saved {len(tiff_files)} TIFF files")

        if create_trial_view:
            html_path = _create_trial_view_html(
                    save_tiff_dir, all_configs, data_layers
            )
            logger.info(f"Created trial view: {html_path}")

        # Return only configs (TIFF mode)
        return all_configs

    else:
        # Return viewer and configs (napari mode)
        return viewer, all_configs


__all__ = ["PipelineGridSearch", "MultiPipelineGridSearch"]
