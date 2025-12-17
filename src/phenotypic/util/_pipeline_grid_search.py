"""Grid search utilities for pipeline parameter tuning and architecture comparison.

This module provides functions to perform parameter grid searches on ImagePipelines,
with parallel execution via joblib and visualization in napari.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from itertools import product
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    import napari
    from phenotypic import Image, ImageOperation


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
        image: Image,
        ops: List[Tuple[ImageOperation, Dict[str, List[Any]]]],
        data_layers: List[str],
) -> None:
    """Validate all inputs before processing.

    Args:
        image: Image to process
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
                image=None,  # type: ignore
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
                "ops": concrete_ops,
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


def _execute_pipeline_trie(
        root: _TrieNode,
        image: "Image",
        n_jobs: int,
):
    """Execute pipeline trie depth-first with shared prefix optimization.

    Executes shared prefix operations once and reuses intermediate results
    for divergent branches. Uses parallel execution for independent branches
    at each level. Yields results as they complete to minimize memory usage.

    Args:
        root: Root node of the execution trie
        image: Original image to process
        n_jobs: Number of parallel jobs (-1 for all cores)

    Yields:
        Tuple of (pipeline_name, result_image, json_config) for each completed pipeline
    """
    from joblib import Parallel, delayed
    from phenotypic import ImagePipeline

    def _process_node(
            node: _TrieNode,
            current_image: "Image",
            ops_stack: List["ImageOperation"],
    ):
        """Recursively process trie node and yield results.

        Args:
            node: Current trie node
            current_image: Image state at this node
            ops_stack: List of operations applied so far

        Yields:
            Tuple of (pipeline_name, result_image, json_config)
        """
        # If this node is a pipeline endpoint, yield result
        if node.pipeline_names:
            for idx, pipeline_name in enumerate(node.pipeline_names):
                # Create pipeline from ops_stack for serialization
                pipeline = ImagePipeline(ops=ops_stack)
                json_config = pipeline.to_json_str()
                
                # Only copy if multiple pipelines share this endpoint
                if len(node.pipeline_names) > 1 and idx < len(node.pipeline_names) - 1:
                    yield (pipeline_name, current_image.copy(), json_config)
                else:
                    # Last/only pipeline can use image directly
                    yield (pipeline_name, current_image, json_config)

        # Process children
        if node.children:
            # Prepare child processing tasks
            child_tasks = []
            for child_node in node.children.values():
                # Deep copy the operation and apply parameters from trie node
                op_copy = copy.deepcopy(child_node.op)
                for key, value in child_node.params.items():
                    setattr(op_copy, key, value)

                child_tasks.append((child_node, op_copy))

            # Execute child branches in parallel if multiple branches
            if len(child_tasks) > 1 and n_jobs != 1:
                # Parallel execution for independent branches

                def _process_branch(child_node, op_copy):
                    # Apply operation - returns a copy, no need to copy first
                    branch_image = op_copy.apply(current_image)
                    # Recursively process this branch and collect results
                    new_stack = ops_stack + [op_copy]
                    return list(_process_node(child_node, branch_image, new_stack))

                branch_results = Parallel(n_jobs=n_jobs)(
                        delayed(_process_branch)(child_node, op_copy)
                        for child_node, op_copy in child_tasks
                )

                # Yield results from all branches
                for branch_result_list in branch_results:
                    for result in branch_result_list:
                        yield result

            else:
                # Serial execution for single branch or when n_jobs=1
                for child_node, op_copy in child_tasks:
                    # Apply operation - returns a copy, no need to copy first
                    branch_image = op_copy.apply(current_image)
                    # Recursively process and yield results
                    new_stack = ops_stack + [op_copy]
                    for result in _process_node(child_node, branch_image, new_stack):
                        yield result

    # Start recursive processing from root and yield results
    for result in _process_node(root, image, []):
        yield result


def _execute_single_pipeline(
        image: Image,
        operations: List[ImageOperation],
        param_config: Tuple[Dict[str, Any], ...],
) -> Tuple[Image, Tuple[Dict[str, Any], ...], str]:
    """Execute a single pipeline with given parameter configuration.

    Args:
        image: Original image to process
        operations: Base operations (will be copied and updated)
        param_config: Parameter values for this configuration

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
        return_results: bool = False,
        viewer_title: str = "Pipeline Grid Search",
) -> Union[Tuple[napari.Viewer, Dict], Tuple[napari.Viewer, Dict, Dict]]:
    """Execute parameter grid search with parallel pipelines and napari visualization.

    Generates all combinations of provided parameters, executes ImagePipeline for each
    combination in parallel, and visualizes results in napari viewer with organized layer
    naming. This is useful for parameter tuning and sensitivity analysis.

    Args:
        image: Single Image object to process. All parameter combinations will be
            applied to this image.
        ops: List of (operation, params_dict) tuples. Each tuple contains an
            ImageOperation instance and a dictionary mapping parameter names to
            lists of values to test. Empty dict means no parameters to vary for that
            operation. Example: [(GaussianBlur(sigma=1.0), {"sigma": [1.0, 2.0, 3.0]}),
                                 (OtsuDetector(), {})]
        data_layers: Which image data to display in napari viewer. Always adds
            original RGB and gray first. Valid options: "rgb", "gray", "enh_gray",
            "objmask", "objmap". Defaults to all available layers.
        n_jobs: Number of parallel jobs for joblib. -1 uses all available cores.
            Default -1.
        return_results: If True, also returns dict of processed Image objects. If False
            (default), only image results are not included.
        viewer_title: Title for the napari viewer window.

    Returns:
        Tuple[napari.Viewer, Dict]: Always returns (viewer, configs_dict). Configs dict maps
            base layer names to serialized pipeline configuration JSON strings.
        Tuple[napari.Viewer, Dict, Dict]: If return_results=True, returns (viewer, results_dict,
            configs_dict) where results_dict maps parameter tuples to processed Image objects.

    Raises:
        ValueError: If ops format is invalid, parameter names don't match operation
            attributes, or data_layers contains invalid names.

    Example:
        >>> from phenotypic import Image
        >>> from phenotypic.enhance import GaussianBlur
        >>> from phenotypic.detect import OtsuDetector
        >>> from phenotypic.util import PipelineGridSearch
        >>>
        >>> image = Image.imread('colony_plate.jpg')
        >>> ops = [
        ...     (GaussianBlur(sigma=1.0), {"sigma": [1.0, 2.0, 3.0]}),
        ...     (OtsuDetector(), {})
        ... ]
        >>>
        >>> # Always get viewer and configs
        >>> viewer, configs = PipelineGridSearch(image=image, ops=ops, n_jobs=-1)
        >>>
        >>> # Or also get image results
        >>> viewer, results, configs = PipelineGridSearch(
        ...     image=image, ops=ops, return_results=True
        ... )
    """
    import napari
    from joblib import Parallel, delayed
    from tqdm_joblib import tqdm_joblib

    # 1. Validate inputs
    _validate_inputs(image, ops, data_layers)

    # 2. Unpack ops tuples into separate lists
    operations, parameters = _unpack_ops_tuples(ops)

    # 3. Generate parameter combinations
    all_configs = _generate_param_combinations(parameters)

    # 4. Execute pipelines in parallel
    with tqdm_joblib(desc="PipelineGridSearch", total=len(all_configs)):
        results = Parallel(n_jobs=n_jobs)(
                delayed(_execute_single_pipeline)(image, operations, config)
                for config in all_configs
        )

    # 5. Create napari viewer and add layers
    viewer = napari.Viewer(title=viewer_title)

    # Add original reference layers
    _add_original_layers(viewer, image)

    # Storage for results and configs
    results_dict = {} if return_results else None
    configs_dict = {}

    # Add result layers for each parameter combination
    for idx, (result_img, param_config, json_config) in enumerate(results):
        _add_result_layers(viewer, result_img, param_config, data_layers, idx)

        # Store config always
        param_str = _create_param_name_string(param_config)
        layer_name = f"{idx:03d}_{param_str}"
        configs_dict[layer_name] = json_config

        # Store result if requested
        if return_results:
            key = tuple(tuple(sorted(params.items())) for params in param_config)
            results_dict[key] = result_img
        else:
            del result_img

    # 6. Return viewer, results (optional), and configs (always)
    if return_results:
        return viewer, configs_dict, results_dict
    else:
        # Delete remaining result references for memory cleanup
        del results
        return viewer, configs_dict


def MultiPipelineGridSearch(
        image: Image,
        pipeline_configs: List[Dict[str, Any]],
        data_layers: List[str] = ["rgb", "gray", "enh_gray", "objmask", "objmap"],
        n_jobs: int = -1,
        return_results: bool = False,
        optimize_shared_prefixes: bool = True,
        viewer_title: str = "Multi-Pipeline Grid Search",
) -> Union[Tuple[napari.Viewer, Dict], Tuple[napari.Viewer, Dict, Dict]]:
    """Execute grid search across multiple pipeline configurations.

    Allows comparing different algorithm combinations and architectures. Each pipeline
    configuration is a different set of operations (e.g., GaussianBlur+Otsu vs.
    MedianFilter+Canny). For each pipeline, all parameter combinations are tested
    in parallel. All results are visualized in a single napari viewer with pipeline
    names in the layer labels for easy comparison.

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
        data_layers: Which image data to display in napari viewer. Valid options:
            "rgb", "gray", "enh_gray", "objmask", "objmap". Defaults to all available.
        n_jobs: Number of parallel jobs for joblib. -1 uses all available cores.
        return_results: If True, also returns dict of processed Image objects. If False
            (default), only image results are not included.
        optimize_shared_prefixes: If True (default), automatically detect and optimize
            pipelines with shared starting operations. Shared operations are executed
            once and intermediate results are reused. Set to False to use original
            linear execution (useful for debugging or comparison).
        viewer_title: Title for the napari viewer window.

    Returns:
        Tuple[napari.Viewer, Dict]: Always returns (viewer, configs_dict). Configs dict maps
            base layer names (with pipeline prefix) to serialized pipeline configuration JSON
            strings.
        Tuple[napari.Viewer, Dict, Dict]: If return_results=True, returns (viewer, results_dict,
            configs_dict) where results_dict maps (pipeline_name, param_tuple) to processed Image
            objects.

    Raises:
        ValueError: If pipeline_configs is empty, contains invalid structures,
            or if operations/parameters are invalid.

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
    import napari
    from joblib import Parallel, delayed

    # Validate pipeline_configs structure
    _validate_pipeline_configs(pipeline_configs)

    # Create single viewer for all results
    viewer = napari.Viewer(title=viewer_title)

    # Add original reference layers once
    _add_original_layers(viewer, image)

    # Storage for combined results and configs
    all_results = {} if return_results else None
    all_configs = {}

    # Choose execution strategy
    if optimize_shared_prefixes:
        # Step 1: Expand all parameter combinations into concrete pipelines
        concrete_configs = _expand_pipeline_configs_to_concrete(pipeline_configs)
        
        # Step 2: Build trie from concrete pipelines (optimizes shared op+param combos)
        trie = _build_pipeline_trie(concrete_configs)
        
        # Step 3: Execute trie with shared prefix optimization
        # Process results as they're generated (streaming) to minimize memory
        for pipeline_name, result_img, json_config in _execute_pipeline_trie(trie, image, n_jobs):
            # Add layers with pipeline name
            for data_layer in data_layers:
                layer_name = f"{pipeline_name}_{data_layer}"
                _add_result_layer(viewer, result_img, data_layer, layer_name)

            # Store config always
            all_configs[pipeline_name] = json_config

            # Store result if requested, otherwise free memory immediately
            if return_results:
                all_results[pipeline_name] = result_img
            else:
                # Explicitly delete to free memory as soon as possible
                del result_img

    else:
        # Original linear execution path (non-optimized)
        # Process each pipeline configuration
        for config_idx, config in enumerate(pipeline_configs):
            pipeline_name = config["name"]
            ops = config["ops"]

            # Unpack ops tuples
            operations, parameters = _unpack_ops_tuples(ops)

            # Generate parameter combinations for this pipeline
            param_configs = _generate_param_combinations(parameters)

            # Execute this pipeline's grid search (reuse existing helper)
            results = Parallel(n_jobs=n_jobs)(
                    delayed(_execute_single_pipeline)(image, operations, param_config)
                    for param_config in param_configs
            )

            # Add results to viewer with pipeline name prefix
            for result_idx, (result_img, param_config, json_config) in enumerate(results):
                param_str = _create_param_name_string(param_config)

                # Add layers with pipeline name prefix
                # Format: "PipelineName_idx_params_layer"
                for data_layer in data_layers:
                    layer_name = (
                        f"{pipeline_name}_{result_idx:03d}_{param_str}_{data_layer}"
                    )
                    _add_result_layer(viewer, result_img, data_layer, layer_name)

                # Store config always
                config_key = f"{pipeline_name}_{result_idx:03d}_{param_str}"
                all_configs[config_key] = json_config

                # Store result if requested
                if return_results:
                    key = (
                        pipeline_name,
                        tuple(tuple(sorted(p.items())) for p in param_config),
                    )
                    all_results[key] = result_img
                else:
                    del result_img

    # Return viewer, results (optional), and configs (always)
    if return_results:
        return viewer, all_configs, all_results
    else:
        # Memory cleanup: results already deleted in loop for optimized path
        # Only delete for non-optimized path
        if not optimize_shared_prefixes:
            del results
        return viewer, all_configs


__all__ = ["PipelineGridSearch", "MultiPipelineGridSearch"]
