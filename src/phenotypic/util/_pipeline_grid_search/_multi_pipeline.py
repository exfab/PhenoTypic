"""Multi-pipeline grid search implementation.

This module provides the MultiPipelineGridSearch function for comparing multiple
ImagePipeline configurations with adaptive batching and shared prefix optimization.
"""

from __future__ import annotations

import gc
import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from ._shared import (
    _build_pipeline_trie,
    _calculate_optimal_batch_size,
    _create_interactive_viewer_html,
    _create_manifest_json,
    _create_output_directory_structure,
    _create_param_name_string,
    _estimate_pipeline_memory,
    _execute_parallel_tasks,
    _execute_single_pipeline,
    _expand_pipeline_configs_to_concrete,
    _extract_data_layers,
    _generate_param_combinations,
    _generate_pipeline_code,
    _get_memory_usage,
    _group_pipelines_by_longest_prefix,
    _process_trie_groups_sequentially,
    _save_array_as_tiff,
    _save_original_images,
    _unpack_ops_tuples,
    _validate_output_dir_params,
    _validate_pipeline_configs,
)

if TYPE_CHECKING:
    from phenotypic import Image

logger = logging.getLogger(__name__)


def MultiPipelineGridSearch(
        image: Image,
        pipeline_configs: List[Dict[str, Any]],
        output_dir: str,
        data_layers: List[str] = ["rgb", "gray", "enh_gray", "objmask", "objmap"],
        n_jobs: int = -1,
        inplace: bool = False,
        create_viewer: bool = True,
        optimize_shared_prefixes: bool = True,
        memory_limit_gb: float = None,
        adaptive_batching: bool = True,
        backend: str = "joblib",
        slurm_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Execute grid search across multiple pipeline configurations.

    Allows comparing different algorithm combinations and architectures. Each pipeline
    configuration is a different set of operations (e.g., GaussianBlur+Otsu vs.
    MedianFilter+Canny). For each pipeline, all parameter combinations are tested
    in parallel. Results are saved to organized directory structure with an interactive
    HTML viewer.

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
        output_dir: Directory for saving all results. Will be created if it doesn't exist.
            Results are organized into subdirectories per pipeline configuration.
        data_layers: Which image data to save. Valid options: "rgb", "gray",
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
            size and available memory. Set to False to process all pipelines at once
            (may cause OOM for large grids).
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
        ValueError: If pipeline_configs is empty, contains invalid structures,
            or if operations/parameters are invalid, or if output_dir is not writable.
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
        >>> configs = MultiPipelineGridSearch(
        ...     image=image,
        ...     pipeline_configs=pipeline_configs,
        ...     output_dir="./multi_results",
        ...     n_jobs=-1
        ... )
    """
    # Validate inputs
    _validate_pipeline_configs(pipeline_configs)
    _validate_output_dir_params(output_dir, create_viewer, backend)

    # Create output directory structure
    logger.info(f"Creating output directory structure: {output_dir}")
    dir_paths = _create_output_directory_structure(output_dir)

    # Save original images
    logger.info("Saving original images...")
    _save_original_images(image, dir_paths["original"])

    # When using submitit backend, disable trie optimization since jobs are already parallel
    if backend == "submitit":
        if optimize_shared_prefixes:
            logger.warning(
                "optimize_shared_prefixes=True is incompatible with submitit backend. "
                "Disabling trie optimization (SLURM jobs are already parallelized). "
                "To suppress this warning, set optimize_shared_prefixes=False explicitly."
            )
        else:
            logger.info("Submitit backend: trie optimization disabled (jobs parallelized)")
        optimize_shared_prefixes = False

    # Storage for combined configs
    all_configs = {}
    pipeline_counter = 0  # Track global pipeline index across all batches

    # Enable inplace operations for memory efficiency
    if inplace is False:
        logger.info("Enabling inplace=True for 3× memory reduction")
        inplace = True  # Safe because images are discarded after saving

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

                # Generate pipeline code
                pipeline_code = _generate_pipeline_code(pipeline_counter)
                pipeline_counter += 1

                # Create pipeline directory
                pipeline_dir = dir_paths["base"] / pipeline_code
                pipeline_dir.mkdir(exist_ok=True)

                # Save each layer as TIFF
                for layer_name, array_data in result_data.items():
                    _save_array_as_tiff(
                            array_data,
                            pipeline_dir,
                            layer_name
                    )

                # Store config
                all_configs[pipeline_code] = json_config

                # Free memory immediately
                del result_data

            batch_elapsed = time.time() - batch_start_time

            # Explicit garbage collection after each batch
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
            for result_idx, (result_img, param_config, json_config) in enumerate(results):
                # Generate pipeline code
                pipeline_code = _generate_pipeline_code(pipeline_counter)
                pipeline_counter += 1

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
                all_configs[pipeline_code] = json_config

                # Free memory immediately
                del result_img, extracted
                gc.collect()

    logger.info(f"Saved {len(all_configs)} pipeline results")

    # Create manifest JSON
    logger.info("Creating manifest.json...")
    _create_manifest_json(dir_paths["base"], all_configs, data_layers)

    # Generate HTML viewer if requested
    if create_viewer:
        logger.info("Generating interactive HTML viewer...")
        html_path = _create_interactive_viewer_html(
                dir_paths["base"], all_configs, data_layers
        )
        logger.info(f"Created viewer: {html_path}")

    logger.info(f"Multi-pipeline grid search complete. Results saved to: {output_dir}")
    return all_configs

