"""Grid search utilities for pipeline parameter tuning and architecture comparison.

This module provides functions to perform parameter grid searches on ImagePipelines,
with parallel execution via joblib and visualization in napari.
"""

from __future__ import annotations

import copy
from itertools import product
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union

if TYPE_CHECKING:
    import napari
    from phenotypic import Image, ImageOperation


def _validate_inputs(
    image: Image,
    operations: List[ImageOperation],
    parameters: List[Dict[str, List[Any]]],
    data_layers: List[str],
) -> None:
    """Validate all inputs before processing.

    Args:
        image: Image to process
        operations: List of ImageOperation instances
        parameters: List of parameter dictionaries
        data_layers: List of data layer names to display

    Raises:
        ValueError: If inputs are invalid or malformed
    """
    # Check operations and parameters have same length
    if len(operations) != len(parameters):
        raise ValueError(
            f"Length mismatch: {len(operations)} operations but "
            f"{len(parameters)} parameter dictionaries. Must be equal."
        )

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

    required_keys = {"name", "operations", "parameters"}

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

        # Validate operations is list
        if not isinstance(config["operations"], list):
            raise ValueError(f"Pipeline config {idx}: 'operations' must be a list")

        # Validate parameters is list
        if not isinstance(config["parameters"], list):
            raise ValueError(f"Pipeline config {idx}: 'parameters' must be a list")

        # Use existing validation for operations and parameters
        _validate_inputs(
            image=None,  # type: ignore
            operations=config["operations"],
            parameters=config["parameters"],
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


def _execute_single_pipeline(
    image: Image,
    operations: List[ImageOperation],
    param_config: Tuple[Dict[str, Any], ...],
) -> Tuple[Image, Tuple[Dict[str, Any], ...]]:
    """Execute a single pipeline with given parameter configuration.

    Args:
        image: Original image to process
        operations: Base operations (will be copied and updated)
        param_config: Parameter values for this configuration

    Returns:
        Tuple of (processed_image, param_config)
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
    result = pipeline.operate([image.copy()])[0]

    return result, param_config


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
    result_img: Image,
    data_layer: str,
    layer_name: str,
) -> None:
    """Add a single data layer from result image to viewer.

    Args:
        viewer: Napari viewer instance
        result_img: Processed image result
        data_layer: Which data to add ("rgb", "gray", etc.)
        layer_name: Name for the layer in napari
    """
    try:
        if data_layer == "rgb":
            data = result_img.rgb[:]
            viewer.add_image(data, name=layer_name, rgb=True)
        elif data_layer == "gray":
            data = result_img.gray[:]
            viewer.add_image(data, name=layer_name, colormap="gray")
        elif data_layer == "enh_gray":
            data = result_img.enh_gray[:]
            viewer.add_image(data, name=layer_name, colormap="gray")
        elif data_layer == "objmask":
            data = result_img.objmask[:]
            if data.any():  # Only add if not empty
                viewer.add_labels(data, name=layer_name)
        elif data_layer == "objmap":
            data = result_img.objmap[:]
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


def PipelineGridSearch(
    image: Image,
    operations: List[ImageOperation],
    parameters: List[Dict[str, List[Any]]],
    data_layers: List[str] = ["rgb", "gray", "enh_gray", "objmask", "objmap"],
    n_jobs: int = -1,
    return_results: bool = False,
    viewer_title: str = "Pipeline Grid Search",
) -> Union[napari.Viewer, Tuple[napari.Viewer, Dict]]:
    """Execute parameter grid search with parallel pipelines and napari visualization.

    Generates all combinations of provided parameters, executes ImagePipeline for each
    combination in parallel, and visualizes results in napari viewer with organized layer
    naming. This is useful for parameter tuning and sensitivity analysis.

    Args:
        image: Single Image object to process. All parameter combinations will be
            applied to this image.
        operations: List of ImageOperation instances forming the pipeline. Parameters
            will be updated from the `parameters` argument for each combination.
        parameters: List of dictionaries mapping parameter names to lists of values.
            One dict per operation. Empty dict means no parameters to vary for that
            operation. Example: [{"sigma": [1.0, 2.0, 3.0]}, {}]
        data_layers: Which image data to display in napari viewer. Always adds
            original RGB and gray first. Valid options: "rgb", "gray", "enh_gray",
            "objmask", "objmap". Defaults to all available layers.
        n_jobs: Number of parallel jobs for joblib. -1 uses all available cores.
            Default -1.
        return_results: If True, returns tuple of (viewer, results_dict). If False
            (default), returns only viewer.
        viewer_title: Title for the napari viewer window.

    Returns:
        napari.Viewer: If return_results=False (default)
        Tuple[napari.Viewer, Dict]: If return_results=True, where Dict maps
            parameter tuples to processed Image objects. Dictionary keys are tuples
            of (param_name, param_value) pairs.

    Raises:
        ValueError: If operations/parameters length mismatch, parameter names don't
            match operation attributes, or data_layers contains invalid names.

    Example:
        >>> from phenotypic import Image
        >>> from phenotypic.enhance import GaussianBlur
        >>> from phenotypic.detect import OtsuDetector
        >>> from phenotypic.util import PipelineGridSearch
        >>>
        >>> image = Image.imread('colony_plate.jpg')
        >>> operations = [GaussianBlur(sigma=1.0), OtsuDetector()]
        >>> parameters = [{"sigma": [1.0, 2.0, 3.0]}, {}]
        >>>
        >>> # Get viewer only
        >>> viewer = PipelineGridSearch(
        ...     image=image,
        ...     operations=operations,
        ...     parameters=parameters,
        ...     n_jobs=-1
        ... )
        >>>
        >>> # Or get results dictionary too
        >>> viewer, results = PipelineGridSearch(
        ...     image=image,
        ...     operations=operations,
        ...     parameters=parameters,
        ...     return_results=True
        ... )
    """
    import napari
    from joblib import Parallel, delayed

    # 1. Validate inputs
    _validate_inputs(image, operations, parameters, data_layers)

    # 2. Generate parameter combinations
    all_configs = _generate_param_combinations(parameters)

    # 3. Execute pipelines in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(_execute_single_pipeline)(image, operations, config)
        for config in all_configs
    )

    # 4. Create napari viewer and add layers
    viewer = napari.Viewer(title=viewer_title)

    # Add original reference layers
    _add_original_layers(viewer, image)

    # Add result layers for each parameter combination
    for idx, (result_img, param_config) in enumerate(results):
        _add_result_layers(viewer, result_img, param_config, data_layers, idx)
        
        # Delete result image if not returning results (memory optimization)
        if not return_results:
            del result_img

    # 5. Return viewer and optionally results dictionary
    if return_results:
        results_dict = _build_results_dict(results)
        return viewer, results_dict
    else:
        # Delete remaining result references for memory cleanup
        del results
        return viewer


def MultiPipelineGridSearch(
    image: Image,
    pipeline_configs: List[Dict[str, Any]],
    data_layers: List[str] = ["rgb", "gray", "enh_gray", "objmask", "objmap"],
    n_jobs: int = -1,
    return_results: bool = False,
    viewer_title: str = "Multi-Pipeline Grid Search",
) -> Union[napari.Viewer, Tuple[napari.Viewer, Dict]]:
    """Execute grid search across multiple pipeline configurations.

    Allows comparing different algorithm combinations and architectures. Each pipeline
    configuration is a different set of operations (e.g., GaussianBlur+Otsu vs.
    MedianFilter+Canny). For each pipeline, all parameter combinations are tested
    in parallel. All results are visualized in a single napari viewer with pipeline
    names in the layer labels for easy comparison.

    Args:
        image: Single Image object to process. All pipelines and parameter
            combinations will be applied to this image.
        pipeline_configs: List of pipeline configuration dictionaries. Each dict
            must contain:
            - "name" (str): Descriptive name for this pipeline (e.g., "GaussianBlur_Otsu")
            - "operations" (List[ImageOperation]): Operations for this pipeline
            - "parameters" (List[Dict[str, List[Any]]]): Parameter grids for each operation
        data_layers: Which image data to display in napari viewer. Valid options:
            "rgb", "gray", "enh_gray", "objmask", "objmap". Defaults to all available.
        n_jobs: Number of parallel jobs for joblib. -1 uses all available cores.
        return_results: If True, returns tuple of (viewer, results_dict). If False
            (default), returns only viewer.
        viewer_title: Title for the napari viewer window.

    Returns:
        napari.Viewer: If return_results=False (default)
        Tuple[napari.Viewer, Dict]: If return_results=True, where Dict maps
            (pipeline_name, param_tuple) to processed Image objects.

    Raises:
        ValueError: If pipeline_configs is empty, contains invalid structures,
            or if operations/parameters are invalid.

    Example:
        >>> from phenotypic import Image
        >>> from phenotypic.enhance import GaussianBlur, MedianFilter
        >>> from phenotypic.detect import OtsuDetector, CannyDetector
        >>> from phenotypic.util import MultiPipelineGridSearch
        >>>
        >>> image = Image.imread('colony_plate.jpg')
        >>>
        >>> pipeline_configs = [
        ...     {
        ...         "name": "GaussianBlur_Otsu",
        ...         "operations": [GaussianBlur(sigma=1.0), OtsuDetector()],
        ...         "parameters": [{"sigma": [1.0, 2.0, 3.0]}, {}]
        ...     },
        ...     {
        ...         "name": "MedianFilter_Otsu",
        ...         "operations": [MedianFilter(size=3), OtsuDetector()],
        ...         "parameters": [{"size": [3, 5, 7]}, {}]
        ...     }
        ... ]
        >>>
        >>> viewer = MultiPipelineGridSearch(
        ...     image=image,
        ...     pipeline_configs=pipeline_configs,
        ...     n_jobs=-1
        ... )
    """
    import napari
    from joblib import Parallel, delayed

    # Validate pipeline_configs structure
    _validate_pipeline_configs(pipeline_configs)

    # Create single viewer for all results
    viewer = napari.Viewer(title=viewer_title)

    # Add original reference layers once
    _add_original_layers(viewer, image)

    # Storage for combined results if needed
    all_results = {} if return_results else None

    # Process each pipeline configuration
    for config_idx, config in enumerate(pipeline_configs):
        pipeline_name = config["name"]
        operations = config["operations"]
        parameters = config["parameters"]

        # Generate parameter combinations for this pipeline
        param_configs = _generate_param_combinations(parameters)

        # Execute this pipeline's grid search (reuse existing helper)
        results = Parallel(n_jobs=n_jobs)(
            delayed(_execute_single_pipeline)(image, operations, param_config)
            for param_config in param_configs
        )

        # Add results to viewer with pipeline name prefix
        for result_idx, (result_img, param_config) in enumerate(results):
            param_str = _create_param_name_string(param_config)

            # Add layers with pipeline name prefix
            # Format: "PipelineName_idx_params_layer"
            for data_layer in data_layers:
                layer_name = (
                    f"{pipeline_name}_{result_idx:03d}_{param_str}_{data_layer}"
                )
                _add_result_layer(viewer, result_img, data_layer, layer_name)

            # Store in results dict if requested, otherwise delete for memory optimization
            if return_results:
                key = (
                    pipeline_name,
                    tuple(tuple(sorted(p.items())) for p in param_config),
                )
                all_results[key] = result_img
            else:
                del result_img

    # Return viewer and optionally results
    if return_results:
        return viewer, all_results
    else:
        # Delete results list for memory cleanup
        del results
        return viewer


__all__ = ["PipelineGridSearch", "MultiPipelineGridSearch"]
