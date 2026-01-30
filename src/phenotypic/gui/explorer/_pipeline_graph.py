"""Pipeline exploration graph for enumerating pipeline variants.

The graph is a configuration tool for exploring different pipeline configurations.
Each path through the graph generates a separate linear ImagePipeline.
Combined with parameter sweeps, this enables systematic exploration of
pipeline configurations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type
import importlib
import itertools
import json
import uuid

import networkx as nx

from phenotypic import ImagePipeline
from phenotypic.abc_ import ImageOperation

from ._sweep_spec import SweepSpec, expand_sweep_combinations, count_sweep_combinations


@dataclass
class GraphNode:
    """Single node in the exploration graph.

    Args:
        id: Unique identifier (UUID string).
        operation_class: Fully qualified class path
            (e.g., 'phenotypic.enhance.GaussianBlur').
        operation_params: Current parameter values for the operation.
        position: (x, y) position for UI layout.
    """

    id: str
    operation_class: str
    operation_params: Dict[str, Any] = field(default_factory=dict)
    position: Tuple[float, float] = (0, 0)

    @property
    def is_output(self) -> bool:
        """Check if this is an output/sink node."""
        return self.operation_class == "__output__"

    @property
    def class_name(self) -> str:
        """Get just the class name without module path."""
        if self.is_output:
            return "Output"
        return self.operation_class.rsplit(".", 1)[-1]

    def get_operation_class(self) -> Type[ImageOperation]:
        """Import and return the operation class.

        Returns:
            The ImageOperation subclass.

        Raises:
            ImportError: If the class cannot be imported.
        """
        if self.is_output:
            raise ValueError("Output nodes do not have an operation class")

        module_path, class_name = self.operation_class.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def instantiate(self, param_overrides: Optional[Dict[str, Any]] = None) -> ImageOperation:
        """Create an operation instance with optional parameter overrides.

        Args:
            param_overrides: Parameters to override from stored values.

        Returns:
            Instantiated ImageOperation.
        """
        if self.is_output:
            raise ValueError("Output nodes cannot be instantiated")

        params = {**self.operation_params}
        if param_overrides:
            params.update(param_overrides)

        op_class = self.get_operation_class()
        return op_class(**params)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "operation_class": self.operation_class,
            "operation_params": self.operation_params,
            "position": list(self.position),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphNode":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            operation_class=data["operation_class"],
            operation_params=data.get("operation_params", {}),
            position=tuple(data.get("position", (0, 0))),
        )


class PipelineGraph:
    """Graph for exploring pipeline variants.

    Each path through the graph generates one or more linear ImagePipelines.
    Parameter sweeps multiply the variants combinatorially.

    The graph supports:
    - Adding operation nodes with parameters
    - Connecting nodes to define data flow
    - Branching (one node connects to multiple downstream nodes)
    - Merging (multiple nodes connect to same downstream node)
    - Parameter sweeps on any node

    Examples:
        Build a simple linear graph:

        >>> from phenotypic.enhance import GaussianBlur
        >>> from phenotypic.detect import OtsuDetector
        >>>
        >>> graph = PipelineGraph()
        >>> gauss = graph.add_operation(GaussianBlur, sigma=1.5)
        >>> otsu = graph.add_operation(OtsuDetector)
        >>> output = graph.add_output()
        >>> graph.connect(gauss, otsu).connect(otsu, output)

        Build a branching graph:

        >>> from phenotypic.detect import CannyDetector
        >>>
        >>> graph = PipelineGraph()
        >>> gauss = graph.add_operation(GaussianBlur, sigma=1.5)
        >>> otsu = graph.add_operation(OtsuDetector)
        >>> canny = graph.add_operation(CannyDetector)
        >>> output = graph.add_output()
        >>> graph.connect(gauss, otsu)
        >>> graph.connect(gauss, canny)  # Branch
        >>> graph.connect(otsu, output)
        >>> graph.connect(canny, output)  # Merge
        >>> graph.path_count
        2

        Add parameter sweep:

        >>> graph.add_sweep(gauss, SweepSpec.from_range('sigma', 1.0, 3.0, 0.5))
        >>> graph.variant_count  # 2 paths × 5 sigma values
        10
    """

    def __init__(self):
        self._graph = nx.DiGraph()
        self._sweeps: Dict[str, List[SweepSpec]] = {}  # node_id -> sweeps
        self._output_ids: List[str] = []

    # =========================================================================
    # Node Management
    # =========================================================================

    def add_operation(
        self,
        op_class: Type[ImageOperation],
        position: Optional[Tuple[float, float]] = None,
        **params,
    ) -> str:
        """Add an operation node to the graph.

        Args:
            op_class: The ImageOperation class to add.
            position: Optional (x, y) position for UI layout.
            **params: Parameters to pass to the operation constructor.

        Returns:
            Node ID (UUID string) for connecting edges.

        Examples:
            >>> graph = PipelineGraph()
            >>> node_id = graph.add_operation(GaussianBlur, sigma=1.5)
        """
        node_id = str(uuid.uuid4())
        position = position or self._next_position()

        node = GraphNode(
            id=node_id,
            operation_class=f"{op_class.__module__}.{op_class.__name__}",
            operation_params=params,
            position=position,
        )

        self._graph.add_node(node_id, data=node)
        return node_id

    def add_output(self, position: Optional[Tuple[float, float]] = None) -> str:
        """Add an output/sink node to mark pipeline endpoints.

        Args:
            position: Optional (x, y) position for UI layout.

        Returns:
            Node ID for connecting edges.

        Examples:
            >>> graph = PipelineGraph()
            >>> output = graph.add_output()
        """
        node_id = str(uuid.uuid4())
        position = position or self._next_position()

        node = GraphNode(
            id=node_id,
            operation_class="__output__",
            operation_params={},
            position=position,
        )

        self._graph.add_node(node_id, data=node)
        self._output_ids.append(node_id)
        return node_id

    def remove_node(self, node_id: str) -> None:
        """Remove a node and its connected edges.

        Args:
            node_id: ID of node to remove.
        """
        if node_id in self._output_ids:
            self._output_ids.remove(node_id)
        if node_id in self._sweeps:
            del self._sweeps[node_id]
        self._graph.remove_node(node_id)

    def get_node(self, node_id: str) -> GraphNode:
        """Get node data by ID.

        Args:
            node_id: Node identifier.

        Returns:
            GraphNode data object.
        """
        return self._graph.nodes[node_id]["data"]

    def update_node_params(self, node_id: str, **params) -> None:
        """Update operation parameters for a node.

        Args:
            node_id: Node to update.
            **params: Parameters to set/update.
        """
        node = self.get_node(node_id)
        node.operation_params.update(params)

    def update_node_position(self, node_id: str, position: Tuple[float, float]) -> None:
        """Update UI position for a node.

        Args:
            node_id: Node to update.
            position: New (x, y) position.
        """
        node = self.get_node(node_id)
        node.position = position

    # =========================================================================
    # Edge Management
    # =========================================================================

    def connect(self, source_id: str, target_id: str) -> "PipelineGraph":
        """Connect two nodes with a directed edge.

        Args:
            source_id: ID of source node.
            target_id: ID of target node.

        Returns:
            Self for method chaining.

        Examples:
            >>> graph.connect(gauss, otsu).connect(otsu, output)
        """
        self._graph.add_edge(source_id, target_id)
        return self

    def disconnect(self, source_id: str, target_id: str) -> "PipelineGraph":
        """Remove edge between two nodes.

        Args:
            source_id: ID of source node.
            target_id: ID of target node.

        Returns:
            Self for method chaining.
        """
        self._graph.remove_edge(source_id, target_id)
        return self

    # =========================================================================
    # Sweep Configuration
    # =========================================================================

    def add_sweep(self, node_id: str, sweep: SweepSpec) -> "PipelineGraph":
        """Add a parameter sweep to a node.

        Args:
            node_id: Node to add sweep to.
            sweep: SweepSpec defining the parameter and values.

        Returns:
            Self for method chaining.

        Examples:
            >>> graph.add_sweep(gauss, SweepSpec.from_range('sigma', 1.0, 3.0, 0.5))
            >>> graph.add_sweep(gauss, SweepSpec('mode', ['reflect', 'constant']))
        """
        if node_id not in self._sweeps:
            self._sweeps[node_id] = []
        self._sweeps[node_id].append(sweep)
        return self

    def remove_sweeps(self, node_id: str) -> None:
        """Remove all sweeps from a node.

        Args:
            node_id: Node to clear sweeps from.
        """
        if node_id in self._sweeps:
            del self._sweeps[node_id]

    def get_sweeps(self, node_id: str) -> List[SweepSpec]:
        """Get sweeps configured for a node.

        Args:
            node_id: Node to query.

        Returns:
            List of SweepSpec objects (empty if none).
        """
        return self._sweeps.get(node_id, [])

    # =========================================================================
    # Graph Properties
    # =========================================================================

    @property
    def nodes(self) -> List[GraphNode]:
        """List of all nodes in the graph."""
        return [self._graph.nodes[n]["data"] for n in self._graph.nodes]

    @property
    def node_ids(self) -> List[str]:
        """List of all node IDs."""
        return list(self._graph.nodes)

    @property
    def edges(self) -> List[Tuple[str, str]]:
        """List of all edges as (source, target) tuples."""
        return list(self._graph.edges)

    @property
    def source_ids(self) -> List[str]:
        """IDs of source nodes (no incoming edges)."""
        return [n for n in self._graph.nodes if self._graph.in_degree(n) == 0]

    @property
    def output_ids(self) -> List[str]:
        """IDs of output/sink nodes."""
        return self._output_ids.copy()

    @property
    def path_count(self) -> int:
        """Number of unique paths from sources to outputs."""
        return len(self.enumerate_paths())

    @property
    def variant_count(self) -> int:
        """Total number of pipeline variants (paths × sweep combinations)."""
        total = 0
        for path in self.enumerate_paths():
            # Get sweeps for nodes in this path
            path_sweeps = []
            for node_id in path:
                path_sweeps.extend(self.get_sweeps(node_id))
            total += count_sweep_combinations(path_sweeps)
        return total if total > 0 else self.path_count

    # =========================================================================
    # Pipeline Enumeration
    # =========================================================================

    def enumerate_paths(self) -> List[List[str]]:
        """Enumerate all paths from source nodes to output nodes.

        Returns:
            List of paths, where each path is a list of node IDs.
        """
        all_paths = []

        for source_id in self.source_ids:
            for output_id in self._output_ids:
                try:
                    paths = list(nx.all_simple_paths(
                        self._graph, source_id, output_id
                    ))
                    all_paths.extend(paths)
                except nx.NetworkXNoPath:
                    pass

        return all_paths

    def enumerate_pipelines(
        self,
    ) -> Iterator[Tuple[str, ImagePipeline, Dict[str, Any]]]:
        """Generate all pipeline variants from the graph.

        Yields:
            Tuples of (variant_id, pipeline, config_dict) where:
            - variant_id: Unique string identifier (e.g., 'path0_combo3')
            - pipeline: Instantiated ImagePipeline
            - config_dict: Dictionary of {node_id: {param: value}} overrides

        Examples:
            >>> for variant_id, pipeline, config in graph.enumerate_pipelines():
            ...     print(f"{variant_id}: {len(pipeline.operations)} ops")
        """
        for path_idx, path in enumerate(self.enumerate_paths()):
            # Get all sweeps for this path
            path_sweeps = []
            sweep_node_map = {}  # sweep index -> node_id
            for node_id in path:
                for sweep in self.get_sweeps(node_id):
                    sweep_node_map[len(path_sweeps)] = node_id
                    path_sweeps.append(sweep)

            # Generate sweep combinations
            if path_sweeps:
                combinations = expand_sweep_combinations(path_sweeps)
            else:
                combinations = [{}]

            for combo_idx, combo in enumerate(combinations):
                variant_id = f"path{path_idx}_combo{combo_idx}"

                # Build config dict mapping node_id -> param overrides
                config = {}
                for param_name, value in combo.items():
                    # Find which node this param belongs to
                    for sweep_idx, sweep in enumerate(path_sweeps):
                        if sweep.param == param_name:
                            node_id = sweep_node_map[sweep_idx]
                            if node_id not in config:
                                config[node_id] = {}
                            config[node_id][param_name] = value
                            break

                # Build pipeline
                operations = []
                for node_id in path:
                    node = self.get_node(node_id)
                    if node.is_output:
                        continue  # Skip output nodes

                    # Check for operation sweep
                    node_config = config.get(node_id, {})
                    if "__operation__" in node_config:
                        # Use the swapped operation directly
                        operations.append(node_config["__operation__"])
                    else:
                        # Instantiate with param overrides
                        operations.append(node.instantiate(node_config))

                pipeline = ImagePipeline(operations)
                yield variant_id, pipeline, config

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary for serialization."""
        return {
            "version": "1.0",
            "nodes": [self.get_node(n).to_dict() for n in self._graph.nodes],
            "edges": [{"source": s, "target": t} for s, t in self._graph.edges],
            "sweeps": {
                node_id: [s.to_dict() for s in sweeps]
                for node_id, sweeps in self._sweeps.items()
            },
            "output_ids": self._output_ids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineGraph":
        """Create graph from dictionary."""
        graph = cls()

        # Add nodes
        for node_data in data.get("nodes", []):
            node = GraphNode.from_dict(node_data)
            graph._graph.add_node(node.id, data=node)
            if node.is_output:
                graph._output_ids.append(node.id)

        # Add edges
        for edge in data.get("edges", []):
            graph._graph.add_edge(edge["source"], edge["target"])

        # Add sweeps
        for node_id, sweeps_data in data.get("sweeps", {}).items():
            for sweep_data in sweeps_data:
                graph._sweeps.setdefault(node_id, []).append(
                    SweepSpec.from_dict(sweep_data)
                )

        # Override output_ids if explicitly provided
        if "output_ids" in data:
            graph._output_ids = data["output_ids"]

        return graph

    def to_json(self, path: Path) -> None:
        """Save graph to JSON file.

        Args:
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "PipelineGraph":
        """Load graph from JSON file.

        Args:
            path: Input file path.

        Returns:
            Loaded PipelineGraph.
        """
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)

    # =========================================================================
    # Convenience Constructors
    # =========================================================================

    @classmethod
    def linear(cls, *operations: ImageOperation) -> "PipelineGraph":
        """Create a linear graph from a sequence of operations.

        Args:
            *operations: ImageOperation instances in sequence.

        Returns:
            PipelineGraph with linear structure.

        Examples:
            >>> graph = PipelineGraph.linear(
            ...     GaussianBlur(sigma=1.5),
            ...     OtsuDetector(),
            ... )
        """
        graph = cls()

        prev_id = None
        for op in operations:
            # Extract class and params
            op_class = type(op)
            params = {}
            for key, value in vars(op).items():
                if not key.startswith("_"):
                    try:
                        json.dumps(value)  # Check if serializable
                        params[key] = value
                    except (TypeError, ValueError):
                        pass

            node_id = graph.add_operation(op_class, **params)
            if prev_id:
                graph.connect(prev_id, node_id)
            prev_id = node_id

        # Add output
        output_id = graph.add_output()
        if prev_id:
            graph.connect(prev_id, output_id)

        return graph

    @classmethod
    def from_pipeline(cls, pipeline: ImagePipeline) -> "PipelineGraph":
        """Convert an existing ImagePipeline to an exploration graph.

        Args:
            pipeline: ImagePipeline to convert.

        Returns:
            Equivalent PipelineGraph.

        Examples:
            >>> pipeline = ImagePipeline([GaussianBlur(), OtsuDetector()])
            >>> graph = PipelineGraph.from_pipeline(pipeline)
        """
        # ImagePipeline stores ops as a dict {name: operation}
        ops = list(pipeline.get_ops().values())
        return cls.linear(*ops)

    # =========================================================================
    # Validation
    # =========================================================================

    def validate(self) -> List[str]:
        """Check graph validity and return issues.

        Returns:
            List of warning/error messages (empty if valid).
        """
        issues = []

        # Check for cycles
        if not nx.is_directed_acyclic_graph(self._graph):
            issues.append("Graph contains cycles")

        # Check for output nodes
        if not self._output_ids:
            issues.append("No output nodes defined")

        # Check for source nodes
        if not self.source_ids:
            issues.append("No source nodes (all nodes have incoming edges)")

        # Check that each source has a path to at least one output
        for source_id in self.source_ids:
            has_path_to_output = False
            for output_id in self._output_ids:
                if nx.has_path(self._graph, source_id, output_id):
                    has_path_to_output = True
                    break
            if not has_path_to_output:
                node = self.get_node(source_id)
                issues.append(
                    f"Source node '{node.class_name}' ({source_id[:8]}) "
                    "has no path to any output"
                )

        # Check for paths to outputs from at least one source
        for output_id in self._output_ids:
            has_path = False
            for source_id in self.source_ids:
                if nx.has_path(self._graph, source_id, output_id):
                    has_path = True
                    break
            if not has_path:
                issues.append(f"Output {output_id[:8]} has no path from any source")

        return issues

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _next_position(self) -> Tuple[float, float]:
        """Calculate position for next node (auto-layout)."""
        if not self._graph.nodes:
            return (100, 100)

        # Find rightmost node and place new one to the right
        max_x = max(self.get_node(n).position[0] for n in self._graph.nodes)
        return (max_x + 200, 100)

    def __repr__(self) -> str:
        return (
            f"PipelineGraph({len(self._graph.nodes)} nodes, "
            f"{len(self._graph.edges)} edges, "
            f"{self.path_count} paths, "
            f"{self.variant_count} variants)"
        )
