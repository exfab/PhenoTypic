"""Single pipeline grid search implementation.

This module provides the PipelineGridSearch function for executing parameter grid
searches on a single ImagePipeline with parallel execution and visualization options.
"""

from __future__ import annotations

import gc
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from ._shared import (
    _add_original_layers,
    _add_result_layers,
    _create_param_name_string,
    _create_trial_view_html,
    _execute_parallel_tasks,
    _execute_single_pipeline,
    _extract_data_layers,
    _save_array_as_tiff,
    _unpack_ops_tuples,
    _validate_inputs,
    _validate_save_tiff_params,
)

if TYPE_CHECKING:
    import napari
    from phenotypic import Image, ImageOperation

logger = logging.getLogger(__name__)


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
    from ._shared import _generate_param_combinations
    all_configs = _generate_param_combinations(parameters)

    # 4. Prepare task arguments
    # Note: We pass image reference (not a copy) to each parallel task. This is intentional
    # and safe because _execute_single_pipeline makes an explicit copy before processing:
    # - Line 1874: `result_image = image.copy()` (inplace=True path)
    # - Line 1878: `result = pipeline.apply(image.copy())` (inplace=False path)
    # This avoids unnecessary image copying when only serial execution occurs, while
    # remaining safe for parallel execution where each worker gets an immutable reference.
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

            # Clean up Image object immediately after use
            del result_img

        # Delete remaining result references for memory cleanup
        del results
        gc.collect()  # Explicit garbage collection to match TIFF mode cleanup
        return viewer, configs_dict
