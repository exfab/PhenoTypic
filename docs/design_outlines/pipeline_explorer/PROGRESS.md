# Pipeline Variant Explorer - Implementation Progress

## Summary

The **programmatic Python API** is complete and fully tested. The **GUI components** (ReactFlow node editor, Panel widgets, HTML viewer) are not yet implemented.

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
- Multiple output views: overlay, objmask, objmap, enh_gray, rgb, gray
- Overlay generation consistent with CLI (uses `image.rgb.save_overlay()`)
- Metrics computation: object count, optional ground truth IoU/precision/recall
- Manifest generation with full configuration
- Pipeline JSON export for each variant

**Manual test:** Successfully ran sweep with 2 variants, saved overlays and objmasks

### 6. Dependencies
- Added `networkx>=3.0` to gui optional dependencies in `pyproject.toml`

### 7. Module Exports
- Updated `src/phenotypic/gui/__init__.py` with lazy imports for:
  - `PipelineGraph`
  - `SweepSpec`
  - `SweepExecutor`
  - `SweepResult`
  - `SweepResults`
  - `GraphNode`

---

## Remaining 🔲

### 1. PipelineNodeEditor (`_node_editor.py`)
ReactFlow-based JSComponent for visual graph editing:
- Custom operation node rendering with type colors
- Drag-and-drop node placement
- Edge creation via click-and-drag
- Node selection for parameter editing
- Bi-directional state sync with Python

**Complexity:** Medium-High (requires React/JSComponent expertise)

### 2. PipelineExplorer Widget (`_explorer_widget.py`)
Main Panel widget combining all components:
- Operations sidebar (categorized list from OperationRegistry)
- Central graph editor (PipelineNodeEditor)
- Parameters sidebar (reuse existing param editor with sweep mode)
- Footer with summary and run button
- Progress indicator during sweep execution

**Complexity:** Medium

### 3. Viewer Module (`viewer/`)
Results comparison and export:
- `SweepComparisonWidget` - interactive side-by-side comparison
- `SweepHTMLExporter` - static HTML with keyboard navigation
- Jinja2 template for HTML viewer

**Complexity:** Medium

### 4. Additional Tests
- `test_sweep_executor.py` - unit tests for executor
- `test_node_editor.py` - tests for ReactFlow component (requires Panel)
- Integration tests in Jupyter notebook

---

## Usage Examples

### Programmatic API (Working Now)

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

## File Locations

### Implementation Files
- `src/phenotypic/gui/explorer/__init__.py` ✅
- `src/phenotypic/gui/explorer/_sweep_spec.py` ✅
- `src/phenotypic/gui/explorer/_sweep_results.py` ✅
- `src/phenotypic/gui/explorer/_pipeline_graph.py` ✅
- `src/phenotypic/gui/explorer/_sweep_executor.py` ✅
- `src/phenotypic/gui/explorer/_node_editor.py` 🔲 TODO
- `src/phenotypic/gui/explorer/_explorer_widget.py` 🔲 TODO
- `src/phenotypic/gui/viewer/` 🔲 TODO (entire module)

### Test Files
- `tests/unit/gui/explorer/__init__.py` ✅
- `tests/unit/gui/explorer/test_sweep_spec.py` ✅
- `tests/unit/gui/explorer/test_pipeline_graph.py` ✅
- `tests/unit/gui/explorer/test_sweep_executor.py` 🔲 TODO

### Updated Files
- `src/phenotypic/gui/__init__.py` ✅ (added lazy imports)
- `pyproject.toml` ✅ (added networkx dependency)

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

## Notes for Continuation

1. **ReactFlow Integration**: The JSComponent approach is outlined in the plan. Key challenge is bi-directional state sync between React and Python via param.

2. **Sweep Mode for Parameters**: The existing `_param_editor.py` component can be extended with a checkbox to enable sweep mode, which shows start/stop/step fields.

3. **Progress Indicator**: Use Panel's `pn.indicators.Progress` or `pn.widgets.Progress` with a background thread for non-blocking execution.

4. **HTML Viewer**: Use Jinja2 templating. The template should include embedded JavaScript for keyboard navigation and filtering.

5. **Ground Truth Comparison**: Already implemented in `SweepExecutor._compute_gt_metrics()`. Just needs GT directory with labeled PNG masks.
