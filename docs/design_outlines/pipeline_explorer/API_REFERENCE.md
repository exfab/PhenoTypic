# Pipeline Explorer API Reference

Quick reference for the implemented programmatic API.

---

## Imports

```python
from phenotypic.gui.explorer import (
    PipelineGraph,
    SweepSpec,
    SweepExecutor,
    SweepResult,
    SweepResults,
    GraphNode,
)
```

---

## SweepSpec

### Creating Sweeps

```python
# Explicit values
sweep = SweepSpec('sigma', [1.0, 1.5, 2.0, 2.5, 3.0])

# Numeric range (start, stop, step)
sweep = SweepSpec.from_range('sigma', 1.0, 3.0, 0.5)
# → [1.0, 1.5, 2.0, 2.5, 3.0]

# Linear spacing (start, stop, num_points)
sweep = SweepSpec.from_linspace('sigma', 1.0, 3.0, 5)
# → [1.0, 1.5, 2.0, 2.5, 3.0]

# Logarithmic spacing (start_exp, stop_exp, num_points)
sweep = SweepSpec.from_logspace('threshold', 1, 3, 3)
# → [10.0, 100.0, 1000.0]

# Categorical values
sweep = SweepSpec('shape', ['disk', 'square', 'diamond'])
```

### Properties

```python
sweep.param      # Parameter name (str)
sweep.values     # List of values
sweep.count      # Number of values
sweep.is_operation_sweep  # True if param == '__operation__'
```

### Serialization

```python
data = sweep.to_dict()
sweep = SweepSpec.from_dict(data)
```

---

## PipelineGraph

### Building Graphs

```python
graph = PipelineGraph()

# Add operation nodes (returns node_id)
gauss_id = graph.add_operation(GaussianBlur, sigma=1.5)
otsu_id = graph.add_operation(OtsuDetector, offset=0)
output_id = graph.add_output()

# Connect nodes (chainable)
graph.connect(gauss_id, otsu_id).connect(otsu_id, output_id)

# Update parameters
graph.update_node_params(gauss_id, sigma=2.0, mode='reflect')

# Remove nodes (also removes connected edges)
graph.remove_node(otsu_id)

# Disconnect specific edge
graph.disconnect(gauss_id, otsu_id)
```

### Configuring Sweeps

```python
# Add sweep to node (chainable)
graph.add_sweep(gauss_id, SweepSpec.from_range('sigma', 1.0, 3.0, 0.5))
graph.add_sweep(otsu_id, SweepSpec('offset', [-5, 0, 5]))

# Get sweeps for a node
sweeps = graph.get_sweeps(gauss_id)  # List[SweepSpec]

# Remove all sweeps from node
graph.remove_sweeps(gauss_id)
```

### Querying the Graph

```python
graph.nodes          # Dict[str, GraphNode]
graph.edges          # List[Tuple[str, str]]
graph.source_ids     # List[str] - nodes with no incoming edges
graph.output_ids     # List[str] - output nodes
graph.path_count     # Number of paths through graph
graph.variant_count  # Total variants (paths × sweep combinations)

# Get specific node
node = graph.get_node(node_id)  # GraphNode

# Enumerate all paths (list of node_id lists)
paths = graph.enumerate_paths()

# Enumerate all pipeline variants
for variant_id, pipeline, config in graph.enumerate_pipelines():
    # variant_id: "path0_combo0", "path0_combo1", etc.
    # pipeline: ImagePipeline instance
    # config: {node_id: {param: value, ...}, ...}
    pass
```

### Validation

```python
issues = graph.validate()  # List[str]
# Returns list of validation issues:
# - "Graph contains cycles"
# - "No output nodes defined"
# - "Source node 'X' has no path to any output"
```

### Serialization

```python
# To/from dict
data = graph.to_dict()
graph = PipelineGraph.from_dict(data)

# To/from JSON file
graph.to_json(Path('./exploration.graph.json'))
graph = PipelineGraph.from_json(Path('./exploration.graph.json'))
```

### Convenience Constructors

```python
# Create linear graph from operation instances
graph = PipelineGraph.linear(
    GaussianBlur(sigma=1.5),
    CLAHE(clip_limit=2.0),
    OtsuDetector(),
)

# Create from existing ImagePipeline
pipeline = ImagePipeline([GaussianBlur(sigma=1.5), OtsuDetector()])
graph = PipelineGraph.from_pipeline(pipeline)
```

---

## GraphNode

```python
node = graph.get_node(node_id)

node.id                # Unique identifier
node.operation_class   # Full class path (e.g., "phenotypic.enhance.GaussianBlur")
node.operation_params  # Dict of parameter values
node.position          # (x, y) tuple for UI
node.is_output         # True if output node
node.class_name        # Short name (e.g., "GaussianBlur")

# Create operation instance
op = node.instantiate()                    # Uses stored params
op = node.instantiate({'sigma': 2.5})     # With overrides
```

---

## SweepExecutor

### Creating Executor

```python
executor = SweepExecutor(
    graph=graph,                           # PipelineGraph
    output_dir='./results',                # Output directory
    data2save={'overlay', 'objmask'},      # Views to save
    njobs=-1,                              # Parallel jobs (-1 = all CPUs)
    ground_truth_dir='./gt',               # Optional GT for IoU
)
```

### Available Views (data2save)

- `'overlay'` - RGB overlay with colored detection labels
- `'objmask'` - Binary object mask
- `'objmap'` - Labeled object map (normalized for visualization)
- `'enh_gray'` - Enhanced grayscale
- `'rgb'` - Original RGB
- `'gray'` - Grayscale

### Running Sweeps

```python
# Single image
results = executor.run(images='./plate001.tif')

# Directory (all images)
results = executor.run(images='./plates/')

# Glob pattern
results = executor.run(images='./plates/*.tif')

# Multiple specific images
results = executor.run(images=['./p1.tif', './p2.tif'])

# With progress callback
def progress(current, total, message):
    print(f"{current}/{total}: {message}")

results = executor.run(images='./plates/', progress_callback=progress)
```

---

## SweepResults

### Properties

```python
results.sweep_dir      # Output directory path
results.results        # List[SweepResult]
results.successful     # List[SweepResult] - only successful
results.failed         # List[SweepResult] - only failed
results.created        # datetime
results.graph_config   # Original graph configuration dict
```

### Analysis

```python
# Convert to pandas DataFrame
df = results.to_dataframe()
# Columns: variant_id, image, success, time, object_count,
#          {sweep_params}, {metrics}

# Find best result by metric
best = results.best_by_metric('object_count')            # Maximize
best = results.best_by_metric('execution_time', minimize=True)

# Filter by metric range
filtered = results.filter_by_metric('object_count', min_val=50, max_val=200)
```

### Persistence

```python
# Save manifest (automatic during executor.run())
manifest_path = results.save_manifest()

# Load from manifest
results = SweepResults.load_manifest(Path('./results/manifest.json'))
```

---

## SweepResult

Single execution result:

```python
result.variant_id       # "path0_combo0"
result.pipeline_config  # Flattened config dict
result.image_name       # Source image filename
result.success          # True/False
result.outputs          # {'overlay': Path, 'objmask': Path, ...}
result.metrics          # {'object_count': 121, 'iou': 0.85, ...}
result.error            # Error message if failed
result.execution_time   # Seconds
```

---

## Output Directory Structure

```
results/
├── manifest.json           # Full sweep metadata
├── images/
│   ├── path0_combo0/
│   │   └── image_stem/
│   │       ├── overlay.png
│   │       └── objmask.png
│   ├── path0_combo1/
│   │   └── image_stem/
│   │       ├── overlay.png
│   │       └── objmask.png
│   └── ...
└── pipelines/
    ├── path0_combo0.json   # Exported ImagePipeline
    ├── path0_combo1.json
    └── ...
```

---

## Complete Example

```python
from phenotypic.gui.explorer import PipelineGraph, SweepSpec, SweepExecutor
from phenotypic.enhance import GaussianBlur, CLAHE
from phenotypic.detect import OtsuDetector, CannyDetector

# Build graph with branching
graph = PipelineGraph()
gauss = graph.add_operation(GaussianBlur, sigma=1.5)
clahe = graph.add_operation(CLAHE, clip_limit=2.0)
otsu = graph.add_operation(OtsuDetector)
canny = graph.add_operation(CannyDetector, low_threshold=50)
output = graph.add_output()

# Linear path with branch
graph.connect(gauss, clahe)
graph.connect(clahe, otsu).connect(clahe, canny)  # Branch
graph.connect(otsu, output).connect(canny, output)  # Merge

# Configure sweeps
graph.add_sweep(gauss, SweepSpec.from_range('sigma', 1.0, 2.0, 0.5))  # 3 values
graph.add_sweep(canny, SweepSpec('low_threshold', [30, 50, 100]))     # 3 values

print(f"Paths: {graph.path_count}")        # 2 (otsu path, canny path)
print(f"Variants: {graph.variant_count}")  # 12 (2 paths × 3 sigma × [1 or 3 threshold])

# Validate
issues = graph.validate()
if issues:
    print("Warnings:", issues)

# Save configuration
graph.to_json('./my_sweep.graph.json')

# Execute
executor = SweepExecutor(
    graph=graph,
    output_dir='./sweep_results',
    data2save={'overlay', 'objmask'},
    njobs=-1,
)
results = executor.run(images='./plates/')

# Analyze
df = results.to_dataframe()
print(df.groupby('variant_id')['object_count'].mean())

best = results.best_by_metric('object_count')
print(f"Best: {best.variant_id} with {best.metrics['object_count']} objects")
```
