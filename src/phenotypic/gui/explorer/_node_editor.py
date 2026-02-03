"""ReactFlow-based node editor for pipeline graph visualization.

Provides an interactive graph editor using ReactFlow embedded via Panel JSComponent.
Supports drag-and-drop node placement, edge creation, and bi-directional state sync.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import uuid

try:
    import param
    import panel as pn
    from panel.custom import JSComponent

    PANEL_AVAILABLE = True
except ImportError:
    PANEL_AVAILABLE = False
    param = None  # type: ignore
    pn = None  # type: ignore
    JSComponent = object  # Fallback for type hints

from ._pipeline_graph import PipelineGraph, GraphNode
from ._sweep_spec import SweepSpec

if TYPE_CHECKING:
    from phenotypic.abc_ import ImageOperation


# =============================================================================
# Helper Functions
# =============================================================================


def get_operation_type(op_class_path: str) -> str:
    """Determine the operation type category from class path.

    Args:
        op_class_path: Full class path (e.g., 'phenotypic.enhance.GaussianBlur').

    Returns:
        Category name for UI display.
    """
    path_lower = op_class_path.lower()

    if "enhance" in path_lower:
        return "Enhancer"
    elif "detect" in path_lower:
        return "Detector"
    elif "refine" in path_lower:
        return "Refiner"
    elif "correct" in path_lower:
        return "Corrector"
    elif "measure" in path_lower:
        return "Measure"
    elif "__output__" in path_lower:
        return "Output"
    else:
        return "Operation"


def graph_node_to_reactflow_node(
    node: GraphNode,
    sweeps: List[SweepSpec],
) -> Dict[str, Any]:
    """Convert a GraphNode to ReactFlow node format.

    Args:
        node: GraphNode to convert.
        sweeps: List of sweeps configured for this node.

    Returns:
        Dictionary in ReactFlow node format.
    """
    sweep_params = {}
    for sweep in sweeps:
        sweep_params[sweep.param] = {
            "values": sweep.values,
            "count": sweep.count,
        }

    return {
        "id": node.id,
        "type": "operation",
        "position": {"x": node.position[0], "y": node.position[1]},
        "data": {
            "label": node.class_name,
            "opType": get_operation_type(node.operation_class),
            "opClass": node.operation_class,
            "params": node.operation_params,
            "sweepParams": sweep_params,
            "hasSweep": len(sweeps) > 0,
        },
    }


def reactflow_node_to_graph_node(rf_node: Dict[str, Any]) -> GraphNode:
    """Convert a ReactFlow node to GraphNode format.

    Args:
        rf_node: ReactFlow node dictionary.

    Returns:
        Equivalent GraphNode.
    """
    position = rf_node.get("position", {"x": 0, "y": 0})

    return GraphNode(
        id=rf_node["id"],
        operation_class=rf_node["data"]["opClass"],
        operation_params=rf_node["data"].get("params", {}),
        position=(position["x"], position["y"]),
    )


def edges_to_tuples(edges: List[Dict[str, str]]) -> List[Tuple[str, str]]:
    """Convert ReactFlow edges to tuple format.

    Args:
        edges: List of ReactFlow edge dictionaries.

    Returns:
        List of (source, target) tuples.
    """
    return [(e["source"], e["target"]) for e in edges]


def tuples_to_edges(edge_tuples: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    """Convert edge tuples to ReactFlow format.

    Args:
        edge_tuples: List of (source, target) tuples.

    Returns:
        List of ReactFlow edge dictionaries.
    """
    return [
        {"id": f"{src}-{tgt}", "source": src, "target": tgt}
        for src, tgt in edge_tuples
    ]


# =============================================================================
# PipelineNodeEditor JSComponent
# =============================================================================


if PANEL_AVAILABLE:

    class PipelineNodeEditor(JSComponent):
        """ReactFlow-based node editor for visual pipeline graph editing.

        This component provides:
        - Drag-and-drop node placement
        - Edge creation via click-and-drag
        - Node selection for parameter editing
        - Bi-directional sync with Python PipelineGraph

        The component uses ReactFlow (via ESM) for the visual editor and
        syncs state with Python through param watchers.

        Args:
            graph: Optional initial PipelineGraph to display.
            height: Editor height in pixels.

        Examples:
            Basic usage:

            >>> editor = PipelineNodeEditor()
            >>> editor.panel()

            With initial graph:

            >>> from phenotypic.gui.explorer import PipelineGraph
            >>> from phenotypic.enhance import GaussianBlur
            >>> graph = PipelineGraph()
            >>> gauss = graph.add_operation(GaussianBlur, sigma=1.5)
            >>> editor = PipelineNodeEditor(graph=graph)
        """

        # === Synced State (Python <-> React) ===
        nodes = param.List(
            default=[],
            doc="""
            Node data for ReactFlow. Each node dict contains:
            - id: Unique identifier
            - type: 'operation' (custom node type)
            - position: {x, y} coordinates
            - data: {label, opType, opClass, params, sweepParams, hasSweep}
            """,
        )

        edges = param.List(
            default=[],
            doc="""
            Edge data for ReactFlow. Each edge dict contains:
            - id: Unique identifier (source-target)
            - source: Source node ID
            - target: Target node ID
            """,
        )

        selected_node_id = param.String(
            default=None,
            allow_None=True,
            doc="ID of currently selected node",
        )

        height = param.Integer(
            default=500,
            bounds=(200, 2000),
            doc="Editor height in pixels",
        )

        # === Internal State ===
        _pending_node_class = param.String(
            default=None,
            allow_None=True,
            doc="Class path of node waiting to be placed",
        )

        _pending_node_params = param.Dict(
            default={},
            doc="Parameters for pending node",
        )

        # === ESM Module Definition ===
        _esm = """
        import React, { useCallback, useState, useEffect, memo } from 'react';
        import {
            ReactFlow,
            Background,
            Controls,
            MiniMap,
            addEdge,
            applyNodeChanges,
            applyEdgeChanges,
            Handle,
            Position,
        } from '@xyflow/react';

        // Type colors for different operation categories
        const typeColors = {
            'Enhancer': '#4CAF50',
            'Detector': '#2196F3',
            'Refiner': '#FF9800',
            'Corrector': '#9C27B0',
            'Measure': '#607D8B',
            'Input': '#333333',
            'Output': '#333333',
            'Operation': '#9E9E9E',
        };

        // Custom Operation Node Component
        const OperationNode = memo(({ data, selected }) => {
            const bgColor = typeColors[data.opType] || typeColors['Operation'];

            return (
                <div style={{
                    padding: '10px 15px',
                    borderRadius: '8px',
                    border: selected ? '2px solid #1976D2' : '1px solid #ccc',
                    background: 'white',
                    minWidth: '120px',
                    boxShadow: selected
                        ? '0 4px 12px rgba(25, 118, 210, 0.3)'
                        : '0 2px 4px rgba(0,0,0,0.1)',
                    transition: 'all 0.15s ease',
                }}>
                    <Handle
                        type="target"
                        position={Position.Top}
                        style={{ background: '#555' }}
                    />

                    <div style={{
                        fontWeight: 'bold',
                        marginBottom: '4px',
                        fontSize: '13px',
                    }}>
                        {data.label}
                    </div>

                    <div style={{
                        fontSize: '11px',
                        color: 'white',
                        background: bgColor,
                        padding: '2px 8px',
                        borderRadius: '3px',
                        display: 'inline-block',
                    }}>
                        {data.opType}
                    </div>

                    {data.hasSweep && (
                        <div style={{
                            fontSize: '10px',
                            color: '#FF5722',
                            marginTop: '6px',
                            fontWeight: '500',
                        }}>
                            ⟳ Sweep configured
                        </div>
                    )}

                    {data.opType !== 'Output' && (
                        <Handle
                            type="source"
                            position={Position.Bottom}
                            style={{ background: '#555' }}
                        />
                    )}
                </div>
            );
        });

        const nodeTypes = { operation: OperationNode };

        export function render({ model }) {
            const [nodes, setNodes] = useState(model.nodes || []);
            const [edges, setEdges] = useState(model.edges || []);

            // Sync from Python to React
            useEffect(() => {
                setNodes(model.nodes || []);
            }, [model.nodes]);

            useEffect(() => {
                setEdges(model.edges || []);
            }, [model.edges]);

            // Handle node changes (position, selection)
            const onNodesChange = useCallback((changes) => {
                setNodes((nds) => {
                    const updated = applyNodeChanges(changes, nds);
                    // Sync back to Python
                    model.nodes = updated;
                    return updated;
                });
            }, []);

            // Handle edge changes
            const onEdgesChange = useCallback((changes) => {
                setEdges((eds) => {
                    const updated = applyEdgeChanges(changes, eds);
                    model.edges = updated;
                    return updated;
                });
            }, []);

            // Handle new edge connections
            const onConnect = useCallback((connection) => {
                setEdges((eds) => {
                    const newEdge = {
                        ...connection,
                        id: `${connection.source}-${connection.target}`,
                    };
                    const updated = addEdge(newEdge, eds);
                    model.edges = updated;
                    return updated;
                });
            }, []);

            // Handle node selection
            const onNodeClick = useCallback((event, node) => {
                model.selected_node_id = node.id;
            }, []);

            // Handle pane click (deselect)
            const onPaneClick = useCallback(() => {
                model.selected_node_id = null;
            }, []);

            // Handle edge deletion
            const onEdgesDelete = useCallback((deletedEdges) => {
                setEdges((eds) => {
                    const deletedIds = new Set(deletedEdges.map(e => e.id));
                    const updated = eds.filter(e => !deletedIds.has(e.id));
                    model.edges = updated;
                    return updated;
                });
            }, []);

            return (
                <div style={{
                    height: model.height + 'px',
                    border: '1px solid #ddd',
                    borderRadius: '4px',
                    overflow: 'hidden',
                }}>
                    <ReactFlow
                        nodes={nodes}
                        edges={edges}
                        onNodesChange={onNodesChange}
                        onEdgesChange={onEdgesChange}
                        onConnect={onConnect}
                        onNodeClick={onNodeClick}
                        onPaneClick={onPaneClick}
                        onEdgesDelete={onEdgesDelete}
                        nodeTypes={nodeTypes}
                        fitView
                        snapToGrid
                        snapGrid={[15, 15]}
                        deleteKeyCode="Backspace"
                        defaultEdgeOptions={{
                            style: { stroke: '#555', strokeWidth: 2 },
                            type: 'smoothstep',
                        }}
                    >
                        <Background variant="dots" gap={15} size={1} color="#ccc" />
                        <Controls />
                        <MiniMap
                            nodeStrokeWidth={3}
                            style={{ background: '#f5f5f5' }}
                        />
                    </ReactFlow>
                </div>
            );
        }
        """

        _importmap = {
            "imports": {
                "react": "https://esm.sh/react@18",
                "react-dom": "https://esm.sh/react-dom@18",
                "@xyflow/react": "https://esm.sh/@xyflow/react@12",
            }
        }

        _stylesheets = [
            "https://cdn.jsdelivr.net/npm/@xyflow/react@12/dist/style.css"
        ]

        def __init__(self, graph: Optional[PipelineGraph] = None, **params):
            """Initialize the node editor.

            Args:
                graph: Optional initial graph to display.
                **params: Additional param parameters.
            """
            super().__init__(**params)
            self._sweep_data: Dict[str, List[SweepSpec]] = {}

            if graph is not None:
                self.load_graph(graph)

        # =====================================================================
        # Public Methods
        # =====================================================================

        def load_graph(self, graph: PipelineGraph) -> None:
            """Load a PipelineGraph into the editor.

            Args:
                graph: Graph to display.
            """
            # Convert nodes
            rf_nodes = []
            for node in graph.nodes:
                sweeps = graph.get_sweeps(node.id)
                self._sweep_data[node.id] = sweeps
                rf_nodes.append(graph_node_to_reactflow_node(node, sweeps))

            # Convert edges
            rf_edges = tuples_to_edges(graph.edges)

            # Update state
            self.nodes = rf_nodes
            self.edges = rf_edges

        def to_pipeline_graph(self) -> PipelineGraph:
            """Convert current editor state to PipelineGraph.

            Returns:
                New PipelineGraph reflecting editor state.
            """
            graph = PipelineGraph()

            # Add nodes using public API
            for rf_node in self.nodes:
                graph_node = reactflow_node_to_graph_node(rf_node)
                graph.add_node(graph_node)

            # Add edges
            for rf_edge in self.edges:
                graph.connect(rf_edge["source"], rf_edge["target"])

            # Add sweeps
            for node_id, sweeps in self._sweep_data.items():
                for sweep in sweeps:
                    graph.add_sweep(node_id, sweep)

            return graph

        def add_operation_node(
            self,
            op_class: type,
            position: Optional[Tuple[float, float]] = None,
            **params,
        ) -> str:
            """Add an operation node to the editor.

            Args:
                op_class: ImageOperation class to add.
                position: Optional (x, y) position. Auto-positioned if None.
                **params: Parameters for the operation.

            Returns:
                Node ID of the added node.
            """
            node_id = str(uuid.uuid4())
            position = position or self._next_position()

            op_class_path = f"{op_class.__module__}.{op_class.__name__}"

            new_node = {
                "id": node_id,
                "type": "operation",
                "position": {"x": position[0], "y": position[1]},
                "data": {
                    "label": op_class.__name__,
                    "opType": get_operation_type(op_class_path),
                    "opClass": op_class_path,
                    "params": params,
                    "sweepParams": {},
                    "hasSweep": False,
                },
            }

            self.nodes = self.nodes + [new_node]
            self._sweep_data[node_id] = []

            return node_id

        def add_output_node(
            self,
            position: Optional[Tuple[float, float]] = None,
        ) -> str:
            """Add an output node to the editor.

            Args:
                position: Optional (x, y) position.

            Returns:
                Node ID of the output node.
            """
            node_id = str(uuid.uuid4())
            position = position or self._next_position()

            new_node = {
                "id": node_id,
                "type": "operation",
                "position": {"x": position[0], "y": position[1]},
                "data": {
                    "label": "Output",
                    "opType": "Output",
                    "opClass": "__output__",
                    "params": {},
                    "sweepParams": {},
                    "hasSweep": False,
                },
            }

            self.nodes = self.nodes + [new_node]
            return node_id

        def remove_node(self, node_id: str) -> None:
            """Remove a node from the editor.

            Args:
                node_id: ID of node to remove.
            """
            self.nodes = [n for n in self.nodes if n["id"] != node_id]
            self.edges = [
                e for e in self.edges
                if e["source"] != node_id and e["target"] != node_id
            ]
            self._sweep_data.pop(node_id, None)

        def connect_nodes(self, source_id: str, target_id: str) -> None:
            """Connect two nodes with an edge.

            Args:
                source_id: Source node ID.
                target_id: Target node ID.
            """
            edge_id = f"{source_id}-{target_id}"
            new_edge = {
                "id": edge_id,
                "source": source_id,
                "target": target_id,
            }
            self.edges = self.edges + [new_edge]

        def get_selected_node_data(self) -> Optional[Dict[str, Any]]:
            """Get data for the currently selected node.

            Returns:
                Node data dictionary or None if no selection.
            """
            if not self.selected_node_id:
                return None

            for node in self.nodes:
                if node["id"] == self.selected_node_id:
                    return node["data"]

            return None

        def update_node_params(
            self,
            node_id: str,
            params: Dict[str, Any],
        ) -> None:
            """Update parameters for a node.

            Args:
                node_id: Node to update.
                params: New parameter values.
            """
            updated_nodes = []
            for node in self.nodes:
                if node["id"] == node_id:
                    node = dict(node)
                    node["data"] = dict(node["data"])
                    node["data"]["params"] = {
                        **node["data"]["params"],
                        **params,
                    }
                updated_nodes.append(node)

            self.nodes = updated_nodes

        def update_node_sweep(
            self,
            node_id: str,
            sweep: Optional[SweepSpec],
            replace: bool = True,
        ) -> None:
            """Update sweep configuration for a node.

            Args:
                node_id: Node to update.
                sweep: SweepSpec to add, or None to clear sweeps.
                replace: If True, replace existing sweeps; otherwise append.
            """
            if sweep is None:
                self._sweep_data[node_id] = []
            elif replace:
                self._sweep_data[node_id] = [sweep]
            else:
                self._sweep_data.setdefault(node_id, []).append(sweep)

            # Update visual indicator
            updated_nodes = []
            for node in self.nodes:
                if node["id"] == node_id:
                    node = dict(node)
                    node["data"] = dict(node["data"])
                    sweeps = self._sweep_data.get(node_id, [])
                    node["data"]["hasSweep"] = len(sweeps) > 0
                    node["data"]["sweepParams"] = {
                        s.param: {"values": s.values, "count": s.count}
                        for s in sweeps
                    }
                updated_nodes.append(node)

            self.nodes = updated_nodes

        def clear(self) -> None:
            """Clear all nodes and edges from the editor."""
            self.nodes = []
            self.edges = []
            self._sweep_data = {}
            self.selected_node_id = None

        # =====================================================================
        # Internal Helpers
        # =====================================================================

        def _next_position(self) -> Tuple[float, float]:
            """Calculate position for next node.

            Returns:
                (x, y) position tuple.
            """
            if not self.nodes:
                return (100.0, 100.0)

            # Place to the right of rightmost node
            max_x = max(n["position"]["x"] for n in self.nodes)
            return (max_x + 200.0, 100.0)

else:
    # Stub class when Panel is not available
    class PipelineNodeEditor:
        """Placeholder when Panel is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PipelineNodeEditor requires Panel. "
                "Install with: pip install phenotypic[gui]"
            )
