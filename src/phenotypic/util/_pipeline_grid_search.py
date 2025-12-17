"""Grid search utilities for pipeline parameter tuning and architecture comparison.

This module provides functions to perform parameter grid searches on ImagePipelines,
with parallel execution via joblib and visualization in napari.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from itertools import product
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    import napari
    from phenotypic import Image, ImageOperation

logger = logging.getLogger(__name__)


def _get_memory_usage() -> float:
    """Get current process memory usage in MB.

    Returns:
        Memory usage of current process in MB
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
                extracted_size += (np.prod(rgb_data.shape, dtype=rgb_data.dtype)
                                   * rgb_data.itemsize)
            elif layer == "gray" and gray_data is not None:
                extracted_size += (np.prod(gray_data.shape, dtype=gray_data.dtype)
                                   * gray_data.itemsize)
            elif layer == "enh_gray" and enh_gray_data is not None:
                extracted_size += (np.prod(enh_gray_data.shape,
                                           dtype=enh_gray_data.dtype)
                                   * enh_gray_data.itemsize)
            elif layer in ["objmask", "objmap"]:
                # Estimate label map size (conservative: same as gray)
                if gray_data is not None:
                    extracted_size += (np.prod(enh_gray_data.shape, dtype=np.uint16)
                                       * rgb_data.itemsize)

        # Peak: base image + extracted arrays + overhead (20%)
        return int((base_size + extracted_size) * 1.2)
    else:
        # Without extraction: num_operations intermediate copies
        # Each operation creates a new image
        # Peak occurs when all intermediate copies exist simultaneously
        return int(base_size * (num_operations + 1) * 1.2)


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
    # Use 2× parallelism as batch size for good utilization
    batch_size = min(jobs_per_batch * 2, total_pipelines)

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
        child_ops = [child.op for child in node.children.values() if child.op is not None]
        
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
        
    Yields:
        Tuple of (pipeline_name, result_data, json_config) for each completed pipeline
    """
    total_groups = len(trie_groups)
    logger.info(f"Processing {total_groups} trie groups sequentially")
    
    for group_idx, group_configs in enumerate(trie_groups, start=1):
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
            group_trie, image, n_jobs, data_layers, extract_arrays
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
        "max_depth": depth,
        "branch_points": branch_points,
        "total_leaf_paths": total_paths,
        "total_nodes": _count_trie_nodes(root),
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
):
    """Execute pipeline trie using shallow traversal with top-level parallelization.

    Implements hybrid approach:
    1. Serial traversal: Follow shared prefix path until reaching first branch point
    2. Parallel execution: Execute all divergent branches in parallel using joblib

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

        # Execute in parallel
        batch_results = Parallel(n_jobs=n_jobs, verbose=10)(tasks)

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
            current_image, pipeline_specs, shared_prefix_len, data_layers, extract_arrays
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
        inplace: Whether to apply operations in-place. If True, reduces memory usage
            by ~3× for typical pipelines (6× → 2× for 5-op pipeline). Only safe when
            input image won't be reused. Default False.
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
                delayed(_execute_single_pipeline)(image, operations, config, inplace)
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
        inplace: bool = False,
        return_results: bool = False,
        optimize_shared_prefixes: bool = True,
        memory_limit_gb: float = None,
        adaptive_batching: bool = True,
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
            Note: When adaptive_batching=True, this value may be reduced automatically
            to fit within memory limits.
        inplace: Whether to apply operations in-place. If True, reduces memory usage
            by ~3× for typical pipelines (6× → 2× for 5-op pipeline). Only safe when
            input image won't be reused. Default False.
        return_results: If True, also returns dict of extracted arrays instead of
            full Image objects (memory-optimized). If False (default), results are
            not returned.
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
        viewer_title: Title for the napari viewer window.

    Returns:
        Tuple[napari.Viewer, Dict]: Always returns (viewer, configs_dict). Configs dict maps
            base layer names (with pipeline prefix) to serialized pipeline configuration JSON
            strings.
        Tuple[napari.Viewer, Dict, Dict]: If return_results=True, returns (viewer, configs_dict,
            results_dict) where results_dict maps pipeline_name to dict of extracted numpy arrays
            {layer_name: np.ndarray}. This is a BREAKING CHANGE from previous versions that
            returned Image objects.

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
        total_pipelines = len(concrete_configs)
        logger.info(f"Expanded {len(pipeline_configs)} pipeline configs into "
                    f"{total_pipelines} concrete pipelines")

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
        global_start_time = time.time()
        for batch_idx, batch_start in enumerate(range(0, total_pipelines, batch_size)):
            batch_end = min(batch_start + batch_size, total_pipelines)
            batch_configs = concrete_configs[batch_start:batch_end]
            batch_pipeline_count = batch_end - batch_start

            if adaptive_batching and total_pipelines > batch_size:
                batch_num = (batch_start // batch_size) + 1
                total_batches = (total_pipelines + batch_size - 1) // batch_size
                mem_before = _get_memory_usage()
                logger.info(f"Processing batch {batch_num}/{total_batches}: "
                            f"pipelines {batch_start} to {batch_end - 1} "
                            f"({batch_pipeline_count} pipelines, {mem_before:.1f} MB)")

            batch_start_time = time.time()

            # Group batch pipelines by longest shared prefix
            logger.debug(f"Grouping {batch_pipeline_count} pipelines by longest shared prefix")
            trie_groups = _group_pipelines_by_longest_prefix(batch_configs)
            logger.info(f"Batch contains {len(trie_groups)} distinct trie groups")

            # Process each trie group sequentially
            logger.debug(f"Starting sequential trie group processing")
            pipeline_results_count = 0
            for pipeline_name, result_data, json_config in _process_trie_groups_sequentially(
                    image, trie_groups, jobs_per_batch,
                    data_layers=data_layers,
                    extract_arrays=True
            ):
                pipeline_results_count += 1
                # result_data is now a dict of {layer_name: np.ndarray}
                # Add layers with pipeline name
                for data_layer, array_data in result_data.items():
                    layer_name = f"{pipeline_name}_{data_layer}"
                    _add_result_layer(viewer, array_data, data_layer, layer_name)

                # Store config always
                all_configs[pipeline_name] = json_config

                # Store result if requested, otherwise free memory immediately
                if return_results:
                    all_results[pipeline_name] = result_data
                else:
                    # Explicitly delete to free memory as soon as possible
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
                    delayed(_execute_single_pipeline)(image, operations, param_config,
                                                      inplace)
                    for param_config in param_configs
            )

            # Add results to viewer with pipeline name prefix
            for result_idx, (result_img, param_config, json_config) in enumerate(
                    results):
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
