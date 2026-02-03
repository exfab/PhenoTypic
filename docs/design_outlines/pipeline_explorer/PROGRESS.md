# Pipeline Variant Explorer - Implementation Progress

## Summary

The **programmatic Python API** and **GUI components** are now complete and tested. This includes the ReactFlow node editor, Panel widgets, comparison viewer, and HTML exporter.

---

## Completed ✅

### 1. Explorer Module Structure
- Created `src/phenotypic/gui/explorer/` module
- Set up `__init__.py` with proper exports

### 2. SweepSpec Class (`_sweep_spec.py`)
Parameter sweep specification with multiple creation methods:
- `SweepSpec(param, values)` - explicit values
- `SweepSpec.from_range(param, start, stop, step)` - numeric range
- `SweepSpec.from_linspace(param, start, stop, num)` - linear spacing
- `SweepSpec.from_logspace(param, start, stop, num)` - logarithmic spacing
- Categorical values (strings, etc.)
- Serialization roundtrip (to_dict/from_dict)
- `expand_sweep_combinations()` for Cartesian product
- `count_sweep_combinations()` for counting without expansion

**Tests:** 18 tests in `tests/unit/gui/explorer/test_sweep_spec.py` - ALL PASSING

### 3. SweepResults Class (`_sweep_results.py`)
Results data structures:
- `SweepResult` - single variant execution result
- `SweepResults` - aggregated results container
- `to_dataframe()` for pandas analysis
- `best_by_metric()` and `filter_by_metric()` helpers
- Manifest save/load for persistence

### 4. PipelineGraph Class (`_pipeline_graph.py`)
Graph-based pipeline configuration:
- Node management: `add_operation()`, `add_output()`, `remove_node()`, `update_node_params()`
- Edge management: `connect()` (chainable), `disconnect()`
- Sweep configuration: `add_sweep()`, `get_sweeps()`, `remove_sweeps()`
- Path enumeration: `enumerate_paths()`, `enumerate_pipelines()`
- Properties: `variant_count`, `path_count`, `source_ids`, `output_ids`
- Serialization: `to_dict()`, `from_dict()`, `to_json()`, `from_json()`
- Convenience: `PipelineGraph.linear(*ops)`, `PipelineGraph.from_pipeline(pipeline)`
- Validation: `validate()` returns list of issues

**Tests:** 30 tests in `tests/unit/gui/explorer/test_pipeline_graph.py` - ALL PASSING

### 5. SweepExecutor Class (`_sweep_executor.py`)
Batch execution engine:
- Parallel execution with joblib
- Image loading from file paths (using `Image.imread()`)
- Multiple output views: overlay, objmask, objmap, detect_mat, rgb, gray
- Overlay generation consistent with CLI (uses `image.rgb.save_overlay()`)
- Metrics computation: object count, optional ground truth IoU/precision/recall
- Manifest generation with full configuration
- Pipeline JSON export for each variant

**Tests:** 32 tests in `tests/unit/gui/explorer/test_sweep_executor.py` - ALL PASSING

### 6. PipelineNodeEditor (`_node_editor.py`)
ReactFlow-based JSComponent for visual graph editing:
- Custom operation node rendering with type-specific colors
- Drag-and-drop node placement
- Edge creation via click-and-drag
- Node selection for parameter editing
- Bi-directional state sync with Python
- Helper functions for converting between ReactFlow and PipelineGraph formats

### 7. PipelineExplorer Widget (`_explorer_widget.py`)
Main Panel widget combining all components:
- Operations sidebar (categorized list from OperationRegistry)
- Central graph editor (PipelineNodeEditor)
- Parameters sidebar with sweep configuration
- Footer with image input, output directory, and run button
- Summary showing path and variant counts
- Progress indicator during sweep execution

### 8. Viewer Module (`viewer/`)
Results comparison and export:
- `SweepComparisonWidget` - interactive side-by-side comparison
  - Variant selection dropdowns
  - View type selection (overlay, objmask, etc.)
  - Image selection for multi-image sweeps
  - Difference visualization mode
  - Metrics display
- `SweepHTMLExporter` - static HTML with keyboard navigation
  - Grid view of all variants
  - Side-by-side comparison mode
  - Keyboard navigation (arrow keys, number keys)
  - Sortable metrics table
  - Image/view filtering
  - Optional base64 image embedding

### 9. Dependencies
- Added `networkx>=3.0` to gui optional dependencies in `pyproject.toml`

### 10. Module Exports
- Updated `src/phenotypic/gui/__init__.py` with lazy imports for:
  - `PipelineGraph`
  - `SweepSpec`
  - `SweepExecutor`
  - `SweepResult`
  - `SweepResults`
  - `GraphNode`
  - `PipelineNodeEditor`
  - `PipelineExplorer`
  - `SweepComparisonWidget`
  - `SweepHTMLExporter`

---

## File Locations

### Implementation Files
- `src/phenotypic/gui/explorer/__init__.py` ✅
- `src/phenotypic/gui/explorer/_sweep_spec.py` ✅
- `src/phenotypic/gui/explorer/_sweep_results.py` ✅
- `src/phenotypic/gui/explorer/_pipeline_graph.py` ✅
- `src/phenotypic/gui/explorer/_sweep_executor.py` ✅
- `src/phenotypic/gui/explorer/_node_editor.py` ✅
- `src/phenotypic/gui/explorer/_explorer_widget.py` ✅
- `src/phenotypic/gui/viewer/__init__.py` ✅
- `src/phenotypic/gui/viewer/_comparison_widget.py` ✅
- `src/phenotypic/gui/viewer/_html_exporter.py` ✅

### Test Files
- `tests/unit/gui/explorer/__init__.py` ✅
- `tests/unit/gui/explorer/test_sweep_spec.py` ✅
- `tests/unit/gui/explorer/test_pipeline_graph.py` ✅
- `tests/unit/gui/explorer/test_sweep_executor.py` ✅

### Updated Files
- `src/phenotypic/gui/__init__.py` ✅ (added lazy imports)
- `pyproject.toml` ✅ (added networkx dependency)

---

## Usage Examples

### Programmatic API

```python
from phenotypic.gui.explorer import PipelineGraph, SweepSpec, SweepExecutor
from phenotypic.enhance import GaussianBlur, CLAHE
from phenotypic.detect import OtsuDetector, CannyDetector

# Build exploration graph
graph = PipelineGraph()

# Add nodes
gauss = graph.add_operation(GaussianBlur, sigma=1.5)
otsu = graph.add_operation(OtsuDetector)
canny = graph.add_operation(CannyDetector, low_threshold=50)
output = graph.add_output()

# Connect nodes (chainable)
graph.connect(gauss, otsu).connect(gauss, canny)  # Branch
graph.connect(otsu, output).connect(canny, output)  # Merge

# Configure sweeps
graph.add_sweep(gauss, SweepSpec.from_range('sigma', 1.0, 3.0, 0.5))
graph.add_sweep(canny, SweepSpec('low_threshold', [30, 50, 70, 100]))

# Check variant count
print(f"Total variants: {graph.variant_count}")
# Output: Total variants: 40 (2 paths × 5 sigma × 4 thresholds)

# Execute sweep
executor = SweepExecutor(
    graph=graph,
    output_dir='./sweep_results',
    data2save={'overlay', 'objmask'},
    njobs=-1,  # Use all CPUs
)
results = executor.run(images='./plates/')

# Analyze results
df = results.to_dataframe()
print(df.sort_values('object_count', ascending=False).head(10))

# Find best configuration
best = results.best_by_metric('object_count')
print(f"Best config: {best.variant_id}")

# Save/load graph
graph.to_json('./my_exploration.graph.json')
loaded = PipelineGraph.from_json('./my_exploration.graph.json')
```

### GUI Usage

```python
from phenotypic.gui import PipelineExplorer

# Launch the interactive explorer
explorer = PipelineExplorer()
explorer.panel()  # Display in Jupyter notebook
```

### Results Comparison

```python
from phenotypic.gui.viewer import SweepComparisonWidget, SweepHTMLExporter
from phenotypic.gui.explorer import SweepResults

# Load results
results = SweepResults.load_manifest('./sweep_results/manifest.json')

# Interactive comparison
widget = SweepComparisonWidget(results=results)
widget.panel()

# Export to static HTML
exporter = SweepHTMLExporter(results)
exporter.export('./sweep_results/viewer.html')
```

### Convenience Constructors

```python
# Create linear graph from operations
graph = PipelineGraph.linear(
    GaussianBlur(sigma=1.5),
    OtsuDetector(),
)

# Convert existing pipeline
from phenotypic import ImagePipeline
pipeline = ImagePipeline([GaussianBlur(sigma=1.5), OtsuDetector()])
graph = PipelineGraph.from_pipeline(pipeline)
```

---

## Test Commands

```bash
# Run all explorer tests
uv run pytest tests/unit/gui/explorer/ -v

# Run specific test file
uv run pytest tests/unit/gui/explorer/test_pipeline_graph.py -v

# Run with coverage
uv run pytest tests/unit/gui/explorer/ --cov=phenotypic.gui.explorer
```

---

## Architecture Summary

```
src/phenotypic/gui/
├── __init__.py                     # Lazy imports for all components
├── explorer/                       # Pipeline Variant Explorer
│   ├── __init__.py                 # ✅ Exports all components
│   ├── _pipeline_graph.py          # ✅ Graph data model (networkx)
│   ├── _sweep_spec.py              # ✅ Parameter sweep specification
│   ├── _sweep_executor.py          # ✅ Batch execution engine
│   ├── _sweep_results.py           # ✅ Results data structure
│   ├── _node_editor.py             # ✅ ReactFlow JSComponent
│   └── _explorer_widget.py         # ✅ Main Panel widget
│
└── viewer/                         # Results Comparison
    ├── __init__.py                 # ✅ Exports viewer components
    ├── _comparison_widget.py       # ✅ Interactive Panel viewer
    └── _html_exporter.py           # ✅ Static HTML generation
```
