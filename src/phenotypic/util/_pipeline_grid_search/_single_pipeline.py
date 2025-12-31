"""Single pipeline grid search implementation.

This module provides the PipelineGridSearch function for executing parameter grid
searches on a single ImagePipeline with parallel execution and directory-based output.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from ._shared import (
    _create_interactive_viewer_html,
    _create_manifest_json,
    _create_output_directory_structure,
    _create_param_name_string,
    _execute_parallel_tasks,
    _execute_single_pipeline,
    _extract_data_layers,
    _generate_pipeline_code,
    _save_array_as_tiff,
    _save_original_images,
    _unpack_ops_tuples,
    _validate_inputs,
    _validate_output_dir_params,
)

if TYPE_CHECKING:
    from phenotypic import Image, ImageOperation

logger = logging.getLogger(__name__)


def PipelineGridSearch(
        image: Image,
        ops: List[Tuple[ImageOperation, Dict[str, List[Any]]]],
        output_dir: str,
        data_layers: List[str] = ["rgb", "gray", "enh_gray", "objmask", "objmap"],
        n_jobs: int = -1,
        inplace: bool = False,
        create_viewer: bool = True,
        backend: str = "joblib",
        slurm_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Execute parameter grid search with parallel pipelines and directory-based output.

    Generates all combinations of provided parameters, executes ImagePipeline for each
    combination in parallel, and saves results to organized directory structure with
    an interactive HTML viewer.

    Args:
        image: Single Image object to process. All parameter combinations will be
            applied to this image.
        ops: List of (operation, params_dict) tuples. Each tuple contains an
            ImageOperation instance and a dictionary mapping parameter names to
            lists of values to test. Empty dict means no parameters to vary for that
            operation. Example: [(GaussianBlur(sigma=1.0), {"sigma": [1.0, 2.0, 3.0]}),
                                 (OtsuDetector(), {})]
        output_dir: Directory for saving all results. Will be created if it doesn't exist.
            Results are organized into subdirectories per pipeline configuration.
        data_layers: Which image data to save. Valid options: "rgb", "gray",
            "enh_gray", "objmask", "objmap". Defaults to all available layers.
        n_jobs: Number of parallel jobs. -1 uses all available cores. Default -1.
        inplace: Whether to apply operations in-place. If True, reduces memory usage
            by ~3× for typical pipelines. Only safe when input image won't be reused.
            Default False.
        create_viewer: Whether to generate interactive HTML viewer (viewer.html).
            Default True.
        backend: Execution backend - "joblib" (local) or "submitit" (SLURM cluster).
            Default "joblib".
        slurm_params: Configuration dict for submitit backend (keys: folder, timeout_min,
            slurm_partition, mem_gb, cpus_per_task, etc.). Only used when backend="submitit".
            Default None.

    Returns:
        Dict[str, str]: Dictionary mapping pipeline codes to JSON config strings.
            Keys are pipeline codes (e.g., "pipeline_001", "pipeline_002").
            Values are JSON-serialized pipeline configurations.

    Raises:
        ValueError: If parameters are invalid (output_dir not writable, invalid backend,
            invalid data_layers, etc.).
        ImportError: If backend="submitit" but submitit is not installed.
        RuntimeError: If image saving, HTML generation, or job execution fails.

    Directory Structure:
        output_dir/
        ├── manifest.json           # Maps codes to configs + metadata
        ├── original/
        │   ├── rgb.tiff
        │   └── gray.tiff
        ├── pipeline_001/
        │   ├── rgb.tiff
        │   ├── gray.tiff
        │   ├── enh_gray.tiff
        │   ├── objmask.tiff
        │   └── objmap.tiff
        ├── pipeline_002/
        │   └── ...
        ├── thumbnails/             # Generated for HTML viewer
        │   └── ...
        └── viewer.html             # Interactive HTML viewer

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
        >>> # Standard usage - directory output with HTML viewer
        >>> configs = PipelineGridSearch(
        ...     image=image,
        ...     ops=ops,
        ...     output_dir="./grid_results",
        ...     n_jobs=-1
        ... )
        >>>
        >>> # Cluster execution with submitit
        >>> configs = PipelineGridSearch(
        ...     image=image,
        ...     ops=ops,
        ...     output_dir="./cluster_results",
        ...     backend="submitit",
        ...     slurm_params={"slurm_partition": "gpu", "mem_gb": 32}
        ... )
    """
    # 1. Validate inputs
    _validate_inputs(ops, data_layers)
    _validate_output_dir_params(output_dir, create_viewer, backend)

    # 2. Create output directory structure
    logger.info(f"Creating output directory structure: {output_dir}")
    dir_paths = _create_output_directory_structure(output_dir)

    # 3. Save original images
    logger.info("Saving original images...")
    _save_original_images(image, dir_paths["original"])

    # 4. Unpack ops tuples into separate lists
    operations, parameters = _unpack_ops_tuples(ops)

    # 5. Generate parameter combinations
    from ._shared import _generate_param_combinations

    all_configs = _generate_param_combinations(parameters)
    total_pipelines = len(all_configs)
    logger.info(f"Generated {total_pipelines} parameter combinations")

    # 6. Prepare task arguments
    task_args = [(image, operations, config, inplace) for config in all_configs]

    # 7. Execute pipelines with selected backend
    logger.info(f"Executing {total_pipelines} pipelines using {backend} backend...")
    results = _execute_parallel_tasks(
            func=_execute_single_pipeline,
            task_args=task_args,
            backend=backend,
            n_jobs=n_jobs,
            slurm_params=slurm_params,
            desc="PipelineGridSearch",
    )

    # 8. Save results to individual pipeline directories
    configs_dict = {}

    for idx, (result_img, param_config, json_config) in enumerate(results):
        # Generate pipeline code
        pipeline_code = _generate_pipeline_code(idx)

        # Create pipeline directory
        pipeline_dir = dir_paths["base"] / pipeline_code
        pipeline_dir.mkdir(exist_ok=True)

        # Extract arrays
        extracted = _extract_data_layers(result_img, data_layers)

        # Save each layer as TIFF
        for layer_name, array_data in extracted.items():
            _save_array_as_tiff(
                    array_data,
                    pipeline_dir,
                    layer_name
            )

        # Store config
        configs_dict[pipeline_code] = json_config

        # Free memory immediately
        del result_img, extracted
        gc.collect()

    logger.info(f"Saved {len(configs_dict)} pipeline results")

    # 9. Create manifest JSON
    logger.info("Creating manifest.json...")
    _create_manifest_json(dir_paths["base"], configs_dict, data_layers)

    # 10. Generate HTML viewer if requested
    if create_viewer:
        logger.info("Generating interactive HTML viewer...")
        html_path = _create_interactive_viewer_html(
                dir_paths["base"], configs_dict, data_layers
        )
        logger.info(f"Created viewer: {html_path}")

    logger.info(f"Pipeline grid search complete. Results saved to: {output_dir}")
    return configs_dict
