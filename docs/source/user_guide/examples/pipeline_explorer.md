# Pipeline Explorer Tutorial

Explore pipeline variants through visual graph editing and parameter sweeps.

The `PipelineExplorer` widget provides an advanced interface for experimenting with
multiple pipeline configurations simultaneously. Instead of building a single linear
pipeline, you can create branching graphs and parameter sweeps to find optimal
settings for your images.

```{figure} /_static/screenshots/pipeline_explorer_main.png
:alt: PipelineExplorer main interface
:width: 100%
:class: screenshot-placeholder

The PipelineExplorer interface with operations sidebar, graph editor, and parameter panel.
```

## Prerequisites

Install GUI extras to use PipelineExplorer:

```bash
pip install phenotypic[gui]
```

Or with uv:

```bash
uv pip install phenotypic[gui]
```

## Quick Start

```python
from phenotypic.gui.explorer import PipelineExplorer

# Create the explorer
explorer = PipelineExplorer()

# Display in Jupyter
explorer.panel()
```

## Installation & Setup

PipelineExplorer uses [Panel](https://panel.holoviz.org/) for widgets and a visual
graph editor. Panel is automatically initialized when you call `panel()`.

```python
from phenotypic.gui.explorer import PipelineExplorer

explorer = PipelineExplorer()
explorer.panel()  # Initializes Panel if needed
```

## Interface Overview

The interface has a three-panel layout:

```
┌─────────────────────────────────────────────────────────────────────┐
│ Pipeline Variant Explorer                                           │
├────────────┬────────────────────────────────┬──────────────────────┤
│ Operations │      Graph Editor              │  Node Parameters     │
│            │                                │                      │
│ [Enhance▼] │  ┌──────┐    ┌──────┐         │  GaussianBlur        │
│ ├─GaussBl  │  │Gauss │───▶│ Otsu │───┐     │  ────────────        │
│ ├─CLAHE    │  └──────┘    └──────┘   │     │  sigma: [1.5]        │
│            │                          │     │                      │
│ [Detect▼]  │              ┌──────┐   │     │  [x] Sweep           │
│ ├─Otsu     │              │Canny │───┤     │  start: [1.0]        │
│ ├─Canny    │              └──────┘   │     │  stop:  [3.0]        │
│            │                         ▼     │  step:  [0.5]        │
│ [Output]   │              ┌──────────┐     │                      │
│            │              │  Output  │     │                      │
│            │              └──────────┘     │                      │
├────────────┴────────────────────────────────┴──────────────────────┤
│ Input: [./images/]   Paths: 2 | Variants: 10      [Run Sweep]      │
└─────────────────────────────────────────────────────────────────────┘
```

### Three Panels

1. **Operations Sidebar** (left) - Accordion with operation categories
2. **Graph Editor** (center) - Visual node graph with connections
3. **Node Parameters** (right) - Edit parameters and configure sweeps

### Footer Controls

- **Image source** - Path to images or directory to process
- **Output directory** - Where to save sweep results
- **Output checkboxes** - Select which data to save
- **Run Sweep** button - Execute all variants

## Operations Sidebar

The sidebar organizes available operations into categories:

| Category | Description |
|----------|-------------|
| **Enhance** | Preprocessing (GaussianBlur, CLAHE, etc.) |
| **Detect** | Detection algorithms (OtsuDetector, CannyDetector, etc.) |
| **Refine** | Post-processing (MorphologicalClose, RemoveSmallObjects) |
| **Correct** | Image corrections (RotationCorrector, etc.) |
| **Measure** | Feature extraction operations |
| **Output** | Terminal node to mark pipeline endpoints |

Click any operation to add it as a node in the graph editor.

## Graph Editor

The graph editor is the central workspace where you build pipeline variants.

```{figure} /_static/screenshots/pipeline_explorer_graph.png
:alt: Graph editor with connected nodes
:width: 80%
:class: screenshot-placeholder

A pipeline graph showing branching detection paths converging to a single output.
```

### Adding Nodes

1. Click an operation in the sidebar
2. A new node appears in the graph
3. Position nodes by dragging

### Connecting Nodes

1. Click and drag from a node's output handle
2. Connect to another node's input handle
3. Connections define the data flow

### The Output Node

Every graph needs at least one **Output** node. This marks the endpoint(s) of your
pipeline. Each path from operations to an Output node represents one pipeline
variant.

```{tip}
You can have multiple Output nodes to test completely different pipeline
configurations in the same sweep.
```

### Graph Validation

The graph validates automatically. Common warnings:
- "No output nodes" - Add an Output node
- "Disconnected nodes" - Connect all nodes to the graph
- "Cycles detected" - Remove circular connections

## Node Parameters

When you select a node in the graph, the right panel shows its parameters.

### Editing Parameters

Parameters display with appropriate widgets based on their type:

| Type | Widget |
|------|--------|
| `float` | FloatInput |
| `int` | IntInput |
| `bool` | Checkbox |
| `str` | TextInput |

Changes are applied immediately to the node.

## Parameter Sweeps

The key feature of PipelineExplorer is **parameter sweeps** - testing multiple
values for a parameter automatically.

### Enabling Sweeps

1. Select a node in the graph
2. Check **Enable Sweep** in the right panel
3. Select the parameter to sweep
4. Configure the sweep type and values
5. Click **Apply Sweep**

### Sweep Types

| Type | Description | Example |
|------|-------------|---------|
| **range** | Start, stop, step | sigma: 1.0 to 3.0, step 0.5 |
| **linspace** | Start, stop, num points | sigma: 1.0 to 3.0, 5 points |
| **logspace** | Logarithmic spacing | threshold: 10¹ to 10³, 5 points |
| **values** | Explicit comma-separated list | sigma: 1.0, 1.5, 2.0, 3.0 |

### Sweep Configuration

```python
# Range sweep: 1.0, 1.5, 2.0, 2.5, 3.0
SweepSpec.from_range('sigma', start=1.0, stop=3.0, step=0.5)

# Linspace: 5 evenly spaced values from 1.0 to 3.0
SweepSpec.from_linspace('sigma', start=1.0, stop=3.0, num=5)

# Logspace: 10, 100, 1000
SweepSpec.from_logspace('threshold', start=1, stop=3, num=3)

# Explicit values
SweepSpec('shape', ['disk', 'square', 'diamond'])
```

### Variant Count

The footer shows the total number of variants:

```
Paths: 2 | Variants: 10
```

- **Paths** = Number of distinct routes through the graph
- **Variants** = Paths × all sweep combinations

Multiple sweeps multiply: 5 sigma values × 3 threshold values = 15 variants per path.

### Clearing Sweeps

Click **Clear Sweep** to remove the sweep configuration from the selected node.

## Running Sweeps

### Configuration

Before running, configure:

1. **Images** - Path to input images (glob patterns supported: `./plates/*.tif`)
2. **Output Directory** - Where results are saved
3. **Output Options** - What to save (checkboxes)

### Output Options

| Checkbox | Saves |
|----------|-------|
| **Overlay** | Detection overlay on original image |
| **ObjMask** | Binary mask of detected objects |
| **ObjMap** | Labeled object map (each object has unique ID) |
| **EnhGray** | Enhanced grayscale (detection matrix) |

### Execution

1. Click **Run Sweep**
2. Progress bar shows completion
3. Results are saved to the output directory

Sweeps run in parallel using `njobs` workers (default: all CPUs).

## Output Structure

Sweep results are organized by variant:

```
sweep_results/
├── manifest.json           # Full results manifest
├── path0_combo0/
│   ├── image1_overlay.png
│   ├── image1_objmask.png
│   └── image1_objmap.png
├── path0_combo1/
│   └── ...
└── path1_combo0/
    └── ...
```

## Working with Results

After a sweep completes, access results via the `results` attribute:

```python
# Get results from the explorer
results = explorer.results

# Summary
print(results.summary())

# Convert to DataFrame for analysis
df = results.to_dataframe()
print(df.head())

# Find best variant by a metric
best = results.best_by_metric('object_count')
print(f"Best variant: {best.variant_id}")
print(f"Config: {best.pipeline_config}")

# Filter by metric range
high_count = results.filter_by_metric('object_count', min_value=50)
```

### SweepResults API

| Property/Method | Description |
|-----------------|-------------|
| `results.successful` | List of successful SweepResult objects |
| `results.failed` | List of failed SweepResult objects |
| `results.success_rate` | Fraction of successful executions |
| `results.to_dataframe()` | Convert to pandas DataFrame |
| `results.best_by_metric(metric, minimize=False)` | Find best variant |
| `results.filter_by_metric(metric, min, max)` | Filter by metric range |
| `results.summary()` | Human-readable summary string |

## Programmatic Usage

For scripting and automation, use the underlying classes directly:

### PipelineGraph

```python
from phenotypic.gui.explorer import PipelineGraph
from phenotypic.enhance import GaussianBlur
from phenotypic.detect import OtsuDetector

# Create a linear pipeline graph
graph = PipelineGraph()
n1 = graph.add_operation(GaussianBlur(sigma=1.5))
n2 = graph.add_operation(OtsuDetector())
out = graph.add_output()

graph.connect(n1, n2)
graph.connect(n2, out)

# Validate
issues = graph.validate()
print(f"Validation issues: {issues}")
```

### SweepSpec

```python
from phenotypic.gui.explorer import SweepSpec

# Create sweep specifications
sigma_sweep = SweepSpec.from_range('sigma', 1.0, 3.0, 0.5)
print(f"Values: {sigma_sweep.values}")  # [1.0, 1.5, 2.0, 2.5, 3.0]
print(f"Count: {sigma_sweep.count}")    # 5
```

### SweepExecutor

```python
from phenotypic.gui.explorer import SweepExecutor

executor = SweepExecutor(
    graph=graph,
    output_dir="./sweep_results",
    data2save={"overlay", "objmask"},
    njobs=-1,  # Use all CPUs
)

results = executor.run(images="./plates/*.tif")
print(results.summary())
```

## Complete Example

Here's a full workflow for parameter optimization:

```python
from phenotypic.gui.explorer import PipelineExplorer, PipelineGraph, SweepSpec
from phenotypic.enhance import GaussianBlur
from phenotypic.detect import OtsuDetector

# 1. Create explorer and display
explorer = PipelineExplorer()
explorer.panel()  # Interactive building in Jupyter

# 2. Or build graph programmatically
graph = PipelineGraph()
blur_node = graph.add_operation(GaussianBlur(sigma=1.5))
detect_node = graph.add_operation(OtsuDetector())
output_node = graph.add_output()

graph.connect(blur_node, detect_node)
graph.connect(detect_node, output_node)

# 3. Add parameter sweep
sigma_sweep = SweepSpec.from_linspace('sigma', 0.5, 3.0, 6)
graph.set_sweep(blur_node, sigma_sweep)

# 4. Load into explorer
explorer.set_graph(graph)

# 5. Run sweep (after configuring image source in UI)
# Or programmatically:
from phenotypic.gui.explorer import SweepExecutor

executor = SweepExecutor(
    graph=graph,
    output_dir="./optimization_results",
    data2save={"overlay", "objmask"},
)
results = executor.run(images="./test_images/*.tif")

# 6. Analyze results
df = results.to_dataframe()
print(df.sort_values('object_count', ascending=False).head(10))

best = results.best_by_metric('object_count')
print(f"\nBest configuration:")
print(f"  Variant: {best.variant_id}")
print(f"  Object count: {best.metrics['object_count']}")
print(f"  Parameters: {best.pipeline_config}")
```

## API Reference

### PipelineExplorer

```python
PipelineExplorer(
    image=None,   # Optional preview image
    graph=None,   # Optional initial PipelineGraph
)
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `panel()` | Panel layout | Get the interactive widget |
| `get_graph()` | PipelineGraph | Export current graph |
| `set_graph(graph)` | None | Load a PipelineGraph |

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `results` | SweepResults | Results from last sweep execution |
| `output_dir` | str | Directory for sweep outputs |
| `njobs` | int | Number of parallel workers |

### Related Classes

- {py:class}`phenotypic.gui.explorer.PipelineGraph` - Visual pipeline graph
- {py:class}`phenotypic.gui.explorer.SweepSpec` - Parameter sweep specification
- {py:class}`phenotypic.gui.explorer.SweepExecutor` - Sweep execution engine
- {py:class}`phenotypic.gui.explorer.SweepResults` - Aggregated sweep results
