# Pipeline Builder Tutorial

Interactive pipeline construction for image processing workflows.

The `PipelineBuilder` widget provides a visual interface for creating and testing
`ImagePipeline` objects. It supports adding operations from all categories (Enhance,
Detect, Refine, Correct) plus measurements, with real-time parameter editing and
preview capabilities.

```{figure} /_static/screenshots/pipeline_builder_main.png
:alt: PipelineBuilder main interface
:width: 100%
:class: screenshot-placeholder

The PipelineBuilder interface showing operations list, parameter editor, and preview panel.
```

## Prerequisites

Install GUI extras to use PipelineBuilder:

```bash
pip install phenotypic[gui]
```

Or with uv:

```bash
uv pip install phenotypic[gui]
```

## Quick Start

```python
from phenotypic.gui import PipelineBuilder
from phenotypic.data import load_synth_yeast_plate

# Load a sample image
image = load_synth_yeast_plate()

# Create the builder
builder = PipelineBuilder(image=image)

# Display in Jupyter (Panel extension initialized automatically)
builder.panel()
```

## Installation & Setup

PipelineBuilder uses [Panel](https://panel.holoviz.org/) for its interactive widgets.
When you create a `PipelineBuilder` instance, Panel is automatically initialized for
Jupyter environments.

For explicit initialization (e.g., in scripts or custom applications):

```python
import panel as pn
pn.extension()

from phenotypic.gui import PipelineBuilder
builder = PipelineBuilder()
builder.panel()
```

## Loading Sample Data

PhenoTypic provides sample images in the `phenotypic.data` module:

```python
from phenotypic.data import load_synth_yeast_plate

# Load synthetic yeast plate image (GridImage with 8x12 wells)
image = load_synth_yeast_plate()

# The image has pre-detected colonies for testing
print(f"Image shape: {image.rgb.shape}")
print(f"Grid: {image.nrows}x{image.ncols} wells")
```

## Interface Overview

The PipelineBuilder interface is organized into several sections:

### Main Layout

1. **Pipeline Header** - Save/Load controls with pipeline name input
2. **Operations List** - Ordered list of pipeline operations
3. **Control Buttons** - Move Up, Move Down, Delete for reordering
4. **Add Operation Menu** - Categorized dropdown to add new operations
5. **Edit Operation Section** - Parameter editor for selected operation
6. **Measurements Section** - Separate list for measurement operations
7. **Preview Panel** - Test the pipeline on a sample image

## Adding Operations

Use the **Add Operation Menu** to add operations to your pipeline. Operations are
organized by category:

| Category | Description | Examples |
|----------|-------------|----------|
| **Enhance** | Preprocessing operations | GaussianBlur, CLAHE, MedianFilter |
| **Detect** | Object detection algorithms | OtsuDetector, CannyDetector, HysteresisDetector |
| **Refine** | Post-detection refinement | MorphologicalClose, RemoveSmallObjects |
| **Correct** | Image corrections | RotationCorrector, VignetteCorrector |

```python
# Operations are added in order and executed sequentially
# Example sequence:
# 1. GaussianBlur (denoise)
# 2. CLAHE (enhance contrast)
# 3. OtsuDetector (detect colonies)
# 4. RemoveSmallObjects (clean up noise)
```

When you click an operation in the menu, it's added to the end of the operations list
and automatically selected for editing.

## Editing Parameters

When you select an operation in the list, the **Edit Operation** section displays
its parameters. Parameter types are automatically detected:

| Parameter Type | Widget | Example |
|----------------|--------|---------|
| `float` | FloatInput | sigma: 1.5 |
| `int` | IntInput | kernel_size: 3 |
| `bool` | Checkbox | invert: False |
| `str` | TextInput | shape: "disk" |
| Selection | Select | mode: ["light", "dark"] |

```{note}
Changes to parameters take effect immediately. The operation is re-instantiated
with new values to ensure derived state is properly computed.
```

Each operation also shows a collapsible **Help** section with its docstring for
quick reference.

## Adding Measurements

Measurements are special operations that extract quantitative features from detected
objects. They appear in a separate section below the main operations.

Click the **Measurements** toggle to expand the section, then use the measurement
menu to add:

- **MeasureArea** - Object area in pixels
- **MeasurePerimeter** - Object perimeter length
- **MeasureIntensity** - Mean/min/max intensity values
- **MeasureMorphology** - Circularity, eccentricity, solidity

Measurements are executed after all detection and refinement operations.

## Managing Operations

Use the control buttons to organize your pipeline:

| Button | Action |
|--------|--------|
| **Move Up** | Move selected operation earlier in the sequence |
| **Move Down** | Move selected operation later in the sequence |
| **Delete** | Remove selected operation from the pipeline |

The operations execute in list order from top to bottom.

## Collapsible Sections

Two sections can be collapsed to save space:

- **Edit Operation** (▼) - Collapse when you're done editing parameters
- **Measurements** (▼) - Collapse if not using measurements

Click the toggle buttons to expand/collapse each section. Widgets remain mounted
(not destroyed) when collapsed for stability.

## Saving & Loading Pipelines

PipelineBuilder integrates with `InstanceManager` for pipeline persistence.

### Saving

1. Enter a name in the text field (e.g., "my_colony_pipeline")
2. Click **Save**
3. Pipeline is saved to the workspace directory

### Loading

1. Select a pipeline from the **Load** dropdown
2. The pipeline is loaded and replaces current operations

```python
# Programmatic save/load with InstanceManager
from phenotypic.gui import InstanceManager

manager = InstanceManager(workspace="./my_pipelines")
builder = PipelineBuilder(manager=manager, image=image)

# After building interactively, pipelines are saved to ./my_pipelines/
```

By default, PipelineBuilder uses a global manager. You can provide a custom manager
for project-specific workspaces.

## Preview Panel

The preview panel at the bottom allows you to test your pipeline:

1. The pipeline is applied to the loaded image
2. Results are displayed as an overlay (detected objects on original image)
3. Update the preview after making changes to verify the pipeline works

```{tip}
Always test your pipeline on a representative image before batch processing.
The preview helps catch parameter issues early.
```

## Getting the Pipeline

After building your pipeline interactively, export it for programmatic use:

```python
# Get the current pipeline as an ImagePipeline object
pipeline = builder.get_pipeline()

# Use it programmatically
from phenotypic.data import load_synth_yeast_plate
image = load_synth_yeast_plate()
result = pipeline.apply(image)

# Save to JSON for batch processing
pipeline.to_json("my_pipeline.json")
```

The returned `ImagePipeline` contains all operations and measurements you configured.

## Complete Example

Here's a full workflow from start to export:

```python
from phenotypic.gui import PipelineBuilder
from phenotypic.data import load_synth_yeast_plate

# 1. Load sample image
image = load_synth_yeast_plate()

# 2. Create and display builder
builder = PipelineBuilder(image=image)
builder.panel()  # Display in Jupyter

# 3. Interactively:
#    - Add GaussianBlur (sigma=1.5)
#    - Add CLAHE (clip_limit=2.0)
#    - Add OtsuDetector
#    - Add MeasureArea
#    - Test with Preview

# 4. Export the pipeline
pipeline = builder.get_pipeline()

# 5. Save for batch processing
pipeline.to_json("colony_detection.json")

# 6. Run batch processing via CLI
# uv run python -m phenotypic colony_detection.json ./plates/ -o ./results/
```

## API Reference

### PipelineBuilder

```python
PipelineBuilder(
    pipeline=None,    # Initial ImagePipeline to load
    manager=None,     # InstanceManager for save/load (uses global if None)
    image=None,       # Preview Image or GridImage
)
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `panel()` | Panel layout | Get the interactive widget for display |
| `get_pipeline()` | ImagePipeline | Export current operations as a pipeline |

### Related Classes

- {py:class}`phenotypic.ImagePipeline` - The pipeline class built by PipelineBuilder
- {py:class}`phenotypic.gui.InstanceManager` - Manages saved pipelines and sessions
- {py:mod}`phenotypic.data` - Sample images for testing
