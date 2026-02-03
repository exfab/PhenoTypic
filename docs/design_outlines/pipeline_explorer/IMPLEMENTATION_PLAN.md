# Pipeline Variant Explorer - Technical Specification

## Overview

Build a **Pipeline Variant Explorer** widget using Panel + ReactFlow that allows users to:
1. Visually construct pipeline variants via drag-and-drop node graph
2. Configure parameter sweeps on any operation
3. Execute all variants and compare results
4. View results in both interactive widget and exported HTML

**Key insight**: The graph is a **configuration tool** for enumerating linear `ImagePipeline` variants, not a non-linear execution engine.

---

## Architecture

```
src/phenotypic/gui/
├── __init__.py                     # Add new exports
├── _instance_manager.py            # ✅ Exists
├── _operation_registry.py          # ✅ Exists
├── _pipeline_builder.py            # ✅ Exists (linear builder, kept separate)
├── _global_session.py              # ✅ Exists
│
├── components/                     # ✅ Existing + extensions
│   ├── _param_editor.py            # → Add sweep mode
│   └── ...
│
├── explorer/                       # 🆕 Pipeline Variant Explorer
│   ├── __init__.py                 # ✅ DONE - Exports: PipelineGraph, SweepSpec, SweepExecutor
│   ├── _pipeline_graph.py          # ✅ DONE - Graph data model (networkx) + programmatic API
│   ├── _sweep_spec.py              # ✅ DONE - SweepSpec for parameter sweeps
│   ├── _sweep_executor.py          # ✅ DONE - Batch execution engine
│   ├── _sweep_results.py           # ✅ DONE - Results data structure
│   ├── _node_editor.py             # 🔲 TODO - ReactFlow JSComponent (GUI only)
│   └── _explorer_widget.py         # 🔲 TODO - Panel widget (GUI only)
│
└── viewer/                         # 🔲 TODO - Results Comparison
    ├── __init__.py
    ├── _comparison_widget.py       # Interactive Panel viewer
    ├── _html_exporter.py           # Static HTML generation
    └── templates/
        └── sweep_viewer.html       # Jinja2 template
```

---

## Module 1: Pipeline Graph (`explorer/_pipeline_graph.py`) ✅ DONE

### Data Structures

```python
@dataclass
class GraphNode:
    """Single node in the exploration graph."""
    id: str                                    # Unique identifier (uuid)
    operation_class: str                       # e.g., "phenotypic.enhance.GaussianBlur"
    operation_params: Dict[str, Any]           # Current parameter values
    position: Tuple[float, float] = (0, 0)     # UI position for ReactFlow

    def instantiate(self, param_overrides: Dict[str, Any] = None) -> ImageOperation:
        """Create operation instance with optional parameter overrides."""

    @property
    def is_output(self) -> bool:
        """Check if this is an output node."""

    @property
    def class_name(self) -> str:
        """Get short class name."""


class PipelineGraph:
    """Graph for exploring pipeline variants.

    Each path through the graph generates one or more linear ImagePipelines.
    Parameter sweeps multiply the variants combinatorially.
    """

    # === Node/Edge Management ===
    def add_operation(self, op_class: type, **params) -> str: ...
    def add_output(self) -> str: ...
    def remove_node(self, node_id: str) -> None: ...
    def update_node_params(self, node_id: str, **params) -> None: ...
    def connect(self, source_id: str, target_id: str) -> 'PipelineGraph': ...  # chainable
    def disconnect(self, source_id: str, target_id: str) -> None: ...

    # === Sweep Configuration ===
    def add_sweep(self, node_id: str, sweep: SweepSpec) -> 'PipelineGraph': ...  # chainable
    def get_sweeps(self, node_id: str) -> List[SweepSpec]: ...
    def remove_sweeps(self, node_id: str) -> None: ...

    # === Pipeline Enumeration ===
    def enumerate_paths(self) -> List[List[str]]: ...
    def enumerate_pipelines(self) -> Iterator[Tuple[str, ImagePipeline, Dict]]: ...

    @property
    def variant_count(self) -> int: ...
    @property
    def path_count(self) -> int: ...
    @property
    def source_ids(self) -> List[str]: ...
    @property
    def output_ids(self) -> List[str]: ...

    # === Serialization ===
    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineGraph': ...
    def to_json(self, path: Path) -> None: ...
    @classmethod
    def from_json(cls, path: Path) -> 'PipelineGraph': ...

    # === Convenience Constructors ===
    @classmethod
    def linear(cls, *operations) -> 'PipelineGraph': ...
    @classmethod
    def from_pipeline(cls, pipeline: ImagePipeline) -> 'PipelineGraph': ...

    # === Validation ===
    def validate(self) -> List[str]: ...
```

---

## Module 2: Sweep Spec (`explorer/_sweep_spec.py`) ✅ DONE

```python
@dataclass
class SweepSpec:
    """Parameter sweep specification.

    Examples:
        # Numeric range
        SweepSpec.from_range('sigma', 1.0, 3.0, 0.5)  # [1.0, 1.5, 2.0, 2.5, 3.0]

        # Linear spacing
        SweepSpec.from_linspace('sigma', 1.0, 3.0, 5)  # 5 evenly spaced values

        # Log spacing
        SweepSpec.from_logspace('threshold', 1, 3, 3)  # [10, 100, 1000]

        # Categorical list
        SweepSpec(param='shape', values=['disk', 'square', 'diamond'])
    """
    param: str
    values: List[Any]

    @classmethod
    def from_range(cls, param: str, start: float, stop: float, step: float): ...
    @classmethod
    def from_linspace(cls, param: str, start: float, stop: float, num: int): ...
    @classmethod
    def from_logspace(cls, param: str, start: float, stop: float, num: int): ...

    @property
    def count(self) -> int: ...
    @property
    def is_operation_sweep(self) -> bool: ...

    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SweepSpec': ...


# Utility functions
def expand_sweep_combinations(sweeps: List[SweepSpec]) -> List[Dict[str, Any]]: ...
def count_sweep_combinations(sweeps: List[SweepSpec]) -> int: ...
```

---

## Module 3: Sweep Executor (`explorer/_sweep_executor.py`) ✅ DONE

```python
class SweepExecutor:
    """Execute pipeline variants with parallel processing."""

    def __init__(
        self,
        graph: PipelineGraph,
        output_dir: Union[str, Path],
        data2save: Optional[Set[str]] = None,  # {'overlay', 'objmask', 'objmap', ...}
        njobs: int = -1,
        ground_truth_dir: Optional[Union[str, Path]] = None,
    ): ...

    def run(
        self,
        images: Union[Iterable[Union[str, Path]], Path, str],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> SweepResults: ...
```

---

## Module 4: Sweep Results (`explorer/_sweep_results.py`) ✅ DONE

```python
@dataclass
class SweepResult:
    """Result from a single pipeline variant execution."""
    variant_id: str
    pipeline_config: Dict[str, Any]
    image_name: str
    success: bool
    outputs: Dict[str, Path] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time: float = 0.0


@dataclass
class SweepResults:
    """Aggregated results from a sweep."""
    sweep_dir: Path
    results: List[SweepResult]
    created: datetime
    graph_config: Dict[str, Any]

    @property
    def successful(self) -> List[SweepResult]: ...
    @property
    def failed(self) -> List[SweepResult]: ...

    def to_dataframe(self) -> pd.DataFrame: ...
    def best_by_metric(self, metric: str, minimize: bool = False) -> SweepResult: ...
    def filter_by_metric(self, metric: str, min_val=None, max_val=None) -> List[SweepResult]: ...
    def save_manifest(self) -> Path: ...
    @classmethod
    def load_manifest(cls, manifest_path: Path) -> 'SweepResults': ...
```

---

## Module 5: ReactFlow Node Editor (`explorer/_node_editor.py`) 🔲 TODO

### JSComponent Implementation

```python
import param
import panel as pn
from panel.custom import JSComponent

class PipelineNodeEditor(JSComponent):
    """ReactFlow-based node editor embedded in Panel.

    Handles:
    - Drag-and-drop node placement
    - Edge creation via click-and-drag
    - Node selection for parameter editing
    - Bi-directional sync with Python
    """

    # === Synced State (Python ↔ React) ===
    nodes = param.List(default=[], doc="""
        Node data for ReactFlow. Each node dict:
        {
            'id': str,
            'type': 'operation',  # Custom node type
            'position': {'x': float, 'y': float},
            'data': {
                'label': str,           # Operation name
                'opType': str,          # 'Enhancer', 'Detector', etc.
                'opClass': str,         # Full class path
                'params': dict,         # Current parameters
                'sweepParams': dict,    # Sweep configurations
                'hasSweep': bool,       # Visual indicator
            }
        }
    """)

    edges = param.List(default=[], doc="""
        Edge data for ReactFlow. Each edge dict:
        {
            'id': str,
            'source': str,  # Source node ID
            'target': str,  # Target node ID
        }
    """)

    selected_node_id = param.String(default=None, doc="Currently selected node")

    height = param.Integer(default=500, doc="Editor height in pixels")

    _esm = """
    import React, { useCallback, useState, useEffect } from 'react';
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
    import '@xyflow/react/dist/style.css';

    // Custom Operation Node Component
    const OperationNode = ({ data, selected }) => {
        const typeColors = {
            'Enhancer': '#4CAF50',
            'Detector': '#2196F3',
            'Refiner': '#FF9800',
            'Corrector': '#9C27B0',
            'Measure': '#607D8B',
            'Input': '#333',
            'Output': '#333',
        };

        return (
            <div style={{
                padding: '10px 15px',
                borderRadius: '8px',
                border: selected ? '2px solid #1976D2' : '1px solid #ccc',
                background: 'white',
                minWidth: '120px',
                boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
            }}>
                <Handle type="target" position={Position.Top} />

                <div style={{ fontWeight: 'bold', marginBottom: '4px' }}>
                    {data.label}
                </div>
                <div style={{
                    fontSize: '11px',
                    color: 'white',
                    background: typeColors[data.opType] || '#999',
                    padding: '2px 6px',
                    borderRadius: '3px',
                    display: 'inline-block',
                }}>
                    {data.opType}
                </div>
                {data.hasSweep && (
                    <div style={{
                        fontSize: '10px',
                        color: '#FF5722',
                        marginTop: '4px',
                    }}>
                        ⟳ Sweep configured
                    </div>
                )}

                <Handle type="source" position={Position.Bottom} />
            </div>
        );
    };

    const nodeTypes = { operation: OperationNode };

    export function render({ model }) {
        const [nodes, setNodes] = useState(model.nodes || []);
        const [edges, setEdges] = useState(model.edges || []);

        // Sync from Python to React
        useEffect(() => { setNodes(model.nodes || []); }, [model.nodes]);
        useEffect(() => { setEdges(model.edges || []); }, [model.edges]);

        const onNodesChange = useCallback((changes) => {
            setNodes((nds) => {
                const updated = applyNodeChanges(changes, nds);
                model.nodes = updated;
                return updated;
            });
        }, []);

        const onEdgesChange = useCallback((changes) => {
            setEdges((eds) => {
                const updated = applyEdgeChanges(changes, eds);
                model.edges = updated;
                return updated;
            });
        }, []);

        const onConnect = useCallback((connection) => {
            setEdges((eds) => {
                const updated = addEdge(connection, eds);
                model.edges = updated;
                return updated;
            });
        }, []);

        const onNodeClick = useCallback((event, node) => {
            model.selected_node_id = node.id;
        }, []);

        const onPaneClick = useCallback(() => {
            model.selected_node_id = null;
        }, []);

        return (
            <div style={{ height: model.height + 'px', border: '1px solid #ddd' }}>
                <ReactFlow
                    nodes={nodes}
                    edges={edges}
                    onNodesChange={onNodesChange}
                    onEdgesChange={onEdgesChange}
                    onConnect={onConnect}
                    onNodeClick={onNodeClick}
                    onPaneClick={onPaneClick}
                    nodeTypes={nodeTypes}
                    fitView
                    snapToGrid
                    snapGrid={[15, 15]}
                >
                    <Background variant="dots" gap={15} size={1} />
                    <Controls />
                    <MiniMap nodeStrokeWidth={3} />
                </ReactFlow>
            </div>
        );
    }
    """

    _stylesheets = [
        'https://cdn.jsdelivr.net/npm/@xyflow/react@12/dist/style.css'
    ]

    # === Python Methods ===
    def add_operation_node(self, op_class: type, position: Tuple[float, float] = None): ...
    def get_selected_node_data(self) -> Optional[Dict]: ...
    def update_node_params(self, node_id: str, params: Dict[str, Any]): ...
    def update_node_sweep(self, node_id: str, sweep_params: Dict[str, SweepRange]): ...
    def to_pipeline_graph(self) -> PipelineGraph: ...
```

---

## Module 6: Explorer Widget (`explorer/_explorer_widget.py`) 🔲 TODO

### Main Widget Layout

```python
import param
import panel as pn

class PipelineExplorer(param.Parameterized):
    """Main widget for exploring pipeline variants.

    Layout:
    ┌─────────────────────────────────────────────────────────────────────┐
    │ Pipeline Variant Explorer                           [Run Sweep]     │
    ├─────────────────────────────────────────────────────────────────────┤
    │ Operations    │  Graph Editor (ReactFlow)        │ Node Parameters │
    │ [Enhance ▼]   │                                  │                 │
    │ ├─ GaussBlur  │   ┌──────┐    ┌──────┐          │ GaussianBlur    │
    │ ├─ CLAHE      │   │Gauss │───▶│ Otsu │───┐     │ ────────────    │
    │ [Detect ▼]    │   └──────┘    └──────┘   │     │ sigma: [1.5]    │
    │ ├─ Otsu       │       │                  │     │                 │
    │ ├─ Canny      │       │       ┌──────┐   │     │ [x] Sweep       │
    │               │       └──────▶│Canny │───┤     │ start: [1.0]    │
    │ [Click to     │               └──────┘   │     │ stop:  [3.0]    │
    │  select, then │                          ▼     │ step:  [0.5]    │
    │  click canvas │               ┌──────────┐     │ → 5 values      │
    │  to place]    │               │  Output  │     │                 │
    │               │               └──────────┘     │                 │
    ├─────────────────────────────────────────────────────────────────────┤
    │ Input: [./images/*.tif]  Summary: 2 paths × 5 sigma = 10 variants  │
    └─────────────────────────────────────────────────────────────────────┘
    """

    preview_image = param.ClassSelector(class_=Image, default=None)
    output_dir = param.Path(default=None)
    njobs = param.Integer(default=-1, bounds=(-1, None))

    # Data to save
    save_overlay = param.Boolean(default=True)
    save_objmask = param.Boolean(default=True)
    save_detect_mat = param.Boolean(default=False)
    save_rgb = param.Boolean(default=False)

    def __init__(self, image: Image = None, manager: InstanceManager = None, **params): ...
    def panel(self) -> pn.viewable.Viewable: ...
    def _build_operations_sidebar(self) -> pn.viewable.Viewable: ...
    def _build_params_sidebar(self) -> pn.viewable.Viewable: ...
    def _build_footer(self) -> pn.viewable.Viewable: ...
    def _on_node_selected(self, event): ...
    def _add_operation(self, op_class: type): ...
    def _run_sweep(self, event): ...
```

---

## Module 7: Results Viewer (`viewer/`) 🔲 TODO

### Interactive Comparison Widget

```python
class SweepComparisonWidget(param.Parameterized):
    """Interactive comparison of sweep results in Jupyter."""

    results = param.ClassSelector(class_=SweepResults)

    variant_a = param.Selector(doc="First variant to compare")
    variant_b = param.Selector(doc="Second variant to compare")
    view = param.Selector(
        objects=['overlay', 'objmask', 'detect_mat', 'rgb'],
        default='overlay',
    )

    show_diff = param.Boolean(default=False, doc="Highlight differences")
    show_metrics = param.Boolean(default=True)

    def __init__(self, results: SweepResults, **params): ...
    def panel(self) -> pn.viewable.Viewable: ...
```

### HTML Exporter

```python
class SweepHTMLExporter:
    """Generate static HTML viewer for sweep results."""

    def __init__(self, results: SweepResults): ...

    def export(self, output_path: Path = None) -> Path:
        """Generate HTML viewer.

        Features:
        - Keyboard navigation (← → to switch variants)
        - Grid view of all variants
        - Click to select for comparison
        - Metrics table with sorting
        - Filter by parameter values
        """
```

---

## Dependencies

### pyproject.toml

```toml
[project.optional-dependencies]
gui = [
    "panel>=1.3",
    "param>=2.0",
    "bokeh>=3.3",
    "jinja2>=3.1",
    "jupyter-bokeh>=4.0.5",
    "networkx>=3.0",        # ✅ Added
]
```

ReactFlow loaded via ESM in JSComponent (no Python package needed).

---

## Design Decisions

1. **Adding operations**: Click in sidebar to select, then click on canvas to place (two-click pattern)
2. **Validation**: No validation on connections - allow any connections (full user flexibility)
3. **Metrics**:
   - Always: object count
   - If pipeline has MeasureFeatures: include those measurements
   - Optional: ground truth comparison (IoU, accuracy) if GT directory provided
4. **Progress**: Background thread with Panel progress indicator (non-blocking execution)
5. **I/O Nodes**: Hybrid - first node auto-becomes input, user adds explicit Output node(s)
6. **Persistence**: Save both graph config (.graph.json) AND export enumerated pipelines
7. **PipelineBuilder**: Keep both widgets - Explorer for sweep/optimization, PipelineBuilder for simple linear building
8. **Sweep UI**: Right sidebar (reuse existing parameter panel)
9. **Output nodes**: Multiple endpoints allowed - each is the end of a pipeline path
10. **Image input**: No preview - select directory or single image path, process all
11. **Undo/Redo**: Not needed initially - keep simple
12. **Output naming**: By variant ID (`path0_combo0/overlay.png`) - manifest maps to config
13. **Ground truth format**: Labeled PNG masks (unique integer per object) for IoU

---

## Verification Plan

### Unit Tests (`tests/unit/gui/explorer/`)

1. **test_pipeline_graph.py** ✅ DONE (30 tests)
   - Graph node/edge CRUD operations
   - Path enumeration with branching
   - Sweep combination generation
   - JSON serialization roundtrip
   - `PipelineGraph.linear()` helper
   - `PipelineGraph.from_pipeline()` conversion

2. **test_sweep_spec.py** ✅ DONE (18 tests)
   - Numeric range generation
   - Categorical lists
   - Combinatorial expansion

3. **test_sweep_executor.py** 🔲 TODO
   - Single pipeline execution
   - Parallel execution with mock pipelines
   - Error handling (failed variants)
   - Manifest generation
   - Ground truth IoU calculation

4. **test_node_editor.py** 🔲 TODO (requires Panel)
   - Node addition/removal
   - Edge connection
   - Python ↔ React state sync

### Integration Test

Manual test in Jupyter notebook to verify full workflow.
