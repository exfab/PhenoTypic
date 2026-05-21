# PhenoTypic Interactive UI System - Full Design Plan (Reference)

> **Note**: This is the complete design plan including SweepExecutor. The current implementation focuses on PipelineBuilder first. See the active plan at `.claude/plans/iridescent-baking-ember.md` for the narrowed scope.

## Overview

Design a Jupyter-based interactive UI system for PhenoTypic with three core components:
1. **InstanceManager** - Session/workspace management with persistent storage
2. **PipelineBuilder** - Panel-based widget for creating/editing pipelines with dynamic parameter generation
3. **ParameterSweep** - Grid search with dual interface (CLI like PipeGridSearch + Panel GUI)

---

## Architecture: Panel + param

**Selected approach**: Use **Panel** (HoloViz) with **param** for declarative, reactive widgets.

**Why Panel:**
- **Dynamic parameter generation**: Convert operation `__init__` signatures to `param.Parameter` types at runtime - no custom widgets per operation
- **Declarative reactivity**: Parameters automatically sync between Python and UI via `param.Parameterized`
- **`pn.Param()` auto-widgets**: Single call renders all parameter controls automatically
- Works in Jupyter, JupyterLab, VS Code, standalone server
- Mature ecosystem with excellent documentation
- Integrates with HoloViz stack (hvPlot, HoloViews) for advanced visualization

**Key pattern - Dynamic Parameter Wrapping:**
```python
import param
import inspect
import typing
from typing import get_origin, get_args, Literal

def wrap_operation(op_class):
    """Convert ImageOperation to Parameterized for auto-widget generation."""
    params = {}
    sig = inspect.signature(op_class.__init__)
    hints = typing.get_type_hints(op_class.__init__)

    for name, p in sig.parameters.items():
        if name == 'self':
            continue
        hint = hints.get(name)
        default = p.default if p.default is not inspect.Parameter.empty else None

        if hint == float or isinstance(default, float):
            params[name] = param.Number(default=default)
        elif hint == int or isinstance(default, int):
            params[name] = param.Integer(default=default)
        elif hint == bool or isinstance(default, bool):
            params[name] = param.Boolean(default=default)
        elif get_origin(hint) is Literal:
            params[name] = param.Selector(objects=list(get_args(hint)), default=default)
        elif isinstance(default, str):
            params[name] = param.String(default=default)

    return type(f'{op_class.__name__}Params', (param.Parameterized,), params)

# Usage - Panel auto-generates all widgets
GaussianBlurParams = wrap_operation(GaussianBlur)
instance = GaussianBlurParams(sigma=1.5)
pn.Param(instance)  # <-- Auto-renders all parameter widgets
```

---

## Component 1: InstanceManager

### Purpose
Manage working directories, persist user artifacts (pipelines, outputs), and handle cleanup.

### Architecture

```python
class InstanceManager:
    """Manages workspace sessions for PhenoTypic UI.

    Provides:
    - Persistent folder management (user-specified or temp)
    - Pipeline storage and retrieval
    - Output organization
    - Automatic cleanup for temp sessions
    """

    def __init__(
        self,
        workspace: Optional[Path] = None,  # None = temp folder
        auto_cleanup: bool = True,          # Cleanup temp on close
    ):
        self._workspace = workspace or self._create_temp_workspace()
        self._is_temp = workspace is None
        self._auto_cleanup = auto_cleanup and self._is_temp

        # Create standard subdirectories
        self._pipelines_dir = self._workspace / "pipelines"
        self._outputs_dir = self._workspace / "outputs"
        self._sweeps_dir = self._workspace / "sweeps"

    # Core methods
    def save_pipeline(self, pipeline: ImagePipeline, name: str) -> Path
    def load_pipeline(self, name: str) -> ImagePipeline
    def list_pipelines(self) -> list[str]
    def delete_pipeline(self, name: str) -> None

    # Output management
    def get_output_dir(self, sweep_name: str) -> Path
    def list_sweeps(self) -> list[str]

    # Cleanup
    def close(self) -> None  # Manual cleanup
    def __enter__ / __exit__  # Context manager
    def __del__  # Destructor fallback
```

### Storage Structure
```
workspace/
├── pipelines/
│   ├── my_detection_pipeline.json
│   └── colony_preprocessing.json
├── outputs/
│   └── sweep_2024_01_28_143022/
│       ├── manifest.json
│       ├── viewer.html
│       └── images/
└── sweeps/
    └── (sweep configuration files)
```

---

## Component 2: PipelineBuilder (Panel)

### Purpose
Interactive pipeline construction with dynamic parameter widgets, operation ordering, and live preview.

### Architecture: Panel + param

```python
import param
import panel as pn


class PipelineBuilder(param.Parameterized):
    """Interactive pipeline construction widget using Panel.

    Features:
    - Add/remove/reorder operations
    - Dynamic parameter widgets via param.Parameterized wrapping
    - Embed saved pipelines or create inline
    - Live preview with configurable views
    """

    # Reactive parameters
    operations = param.List(default=[], doc="List of (name, operation) tuples")
    selected_view = param.Selector(
        objects=['rgb', 'gray', 'detect_mat', 'objmask', 'objmap', 'overlay'],
        default='overlay'
    )

    def __init__(
        self,
        pipeline: Optional[ImagePipeline] = None,
        instance_manager: Optional[InstanceManager] = None,
        image: Optional[Image] = None,
        **params
    ):
        super().__init__(**params)
        self._pipeline = pipeline or ImagePipeline(pipe_cfgs=[])
        self._manager = instance_manager
        self._image = image

        # Initialize operations from pipeline
        if pipeline is not None:
            self.operations = [(name, op) for name, op in pipeline._ops.items()]

    def panel(self) -> pn.viewable.Viewable:
        """Build and return the Panel layout."""
        return pn.Row(
            self._build_operations_panel(),
            self._build_preview_panel(),
            sizing_mode='stretch_both'
        )
```

### Dynamic Operation Card with Panel

```python
class OperationCard(param.Parameterized):
    """Card widget for a single operation with auto-generated parameter widgets."""

    collapsed = param.Boolean(default=True, doc="Whether parameters are collapsed")

    def __init__(self, operation, index: int, on_move=None, on_delete=None, **params):
        super().__init__(**params)
        self.operation = operation
        self.index = index
        self._on_move = on_move
        self._on_delete = on_delete

        # Wrap operation for parameter widgets
        self._param_wrapper = self._wrap_operation_instance(operation)

    def _wrap_operation_instance(self, op):
        """Create param.Parameterized wrapper mirroring operation's current values."""
        params = {}
        sig = inspect.signature(op.__init__)
        hints = typing.get_type_hints(op.__init__)

        for name, p in sig.parameters.items():
            if name == 'self':
                continue
            current_value = getattr(op, name, p.default)
            hint = hints.get(name)

            if isinstance(current_value, bool):
                params[name] = param.Boolean(default=current_value)
            elif isinstance(current_value, int):
                params[name] = param.Integer(default=current_value)
            elif isinstance(current_value, float):
                params[name] = param.Number(default=current_value)
            elif get_origin(hint) is Literal:
                params[name] = param.Selector(
                    objects=list(get_args(hint)),
                    default=current_value
                )
            elif isinstance(current_value, str):
                params[name] = param.String(default=current_value)

        # Create class and instance
        WrapperClass = type(f'{op.__class__.__name__}Params', (param.Parameterized,), params)
        wrapper = WrapperClass()

        # Sync changes back to operation
        def sync_param(event):
            setattr(op, event.name, event.new)
        wrapper.param.watch(sync_param, list(params.keys()))

        return wrapper

    def panel(self) -> pn.viewable.Viewable:
        """Build the card widget."""
        op_name = self.operation.__class__.__name__
        op_type = self._get_operation_type()

        # Header with controls
        header = pn.Row(
            pn.pane.HTML(f"<b>{op_name}</b>"),
            pn.pane.HTML(f"<span style='background:#e0e0e0;padding:2px 6px;"
                        f"border-radius:3px;font-size:0.8em;'>{op_type}</span>"),
            pn.Spacer(),
            pn.widgets.Button(name='▲', width=30, on_click=lambda e: self._on_move(self.index, -1)),
            pn.widgets.Button(name='▼', width=30, on_click=lambda e: self._on_move(self.index, 1)),
            pn.widgets.Button(name='×', width=30, button_type='danger',
                             on_click=lambda e: self._on_delete(self.index)),
            sizing_mode='stretch_width'
        )

        # Parameters panel using pn.Param - THE KEY FEATURE
        params_panel = pn.Param(
            self._param_wrapper,
            show_name=False,
            widgets={
                # Can customize specific widgets if needed
            }
        )

        # Collapsible card
        return pn.Card(
            params_panel,
            header=header,
            collapsed=self.collapsed,
            sizing_mode='stretch_width'
        )
```

### UI Layout
```
┌─────────────────────────────────────────────────────────────────┐
│  Pipeline Builder                                     [Save] [Load]│
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────┐  ┌──────────────────────────────────┐ │
│  │ OPERATIONS           │  │ PREVIEW                          │ │
│  │ ┌──────────────────┐ │  │                                  │ │
│  │ │ ▶ GaussianBlur   │ │  │   [View: Overlay ▼]             │ │
│  │ │   [▲][▼][×]      │ │  │                                  │ │
│  │ ├──────────────────┤ │  │   ┌──────────────────────────┐  │ │
│  │ │  sigma: [1.5   ] │ │  │   │                          │  │ │
│  │ │  (auto-expanded) │ │  │   │    [Image Preview]       │  │ │
│  │ └──────────────────┘ │  │   │                          │  │ │
│  │ ┌──────────────────┐ │  │   └──────────────────────────┘  │ │
│  │ │ ▶ OtsuDetector   │ │  │                                  │ │
│  │ │   [▲][▼][×]      │ │  │   [Update Preview]              │ │
│  │ └──────────────────┘ │  │                                  │ │
│  │                      │  └──────────────────────────────────┘ │
│  │ [+ Add Operation ▼]  │                                       │
│  │ [+ Add Pipeline ▼]   │                                       │
│  └──────────────────────┘                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component 3: ParameterSweep (Dual Interface)

### Purpose
Systematic parameter exploration with **both CLI/programmatic interface** (like PipeGridSearch) and **Panel GUI** for configuration.

### Architecture: Following PipeGridSearch Pattern

```python
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple, Dict, Any, Optional, Set, Union

import numpy as np
from joblib import Parallel, delayed

if TYPE_CHECKING:
    from phenotypic import Image, ImagePipeline
    from phenotypic.abc_ import ImageOperation


@dataclass
class SweepResult:
    """Result from a single (pipeline, image) execution."""
    pipeline_id: str
    pipeline_name: str
    image_name: str
    success: bool
    outputs: Dict[str, Path] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class SweepResults:
    """Aggregated results from a sweep execution."""
    sweep_dir: Path
    results: List[SweepResult]
    manifest_path: Path
    viewer_path: Optional[Path] = None

    @property
    def successful(self) -> List[SweepResult]:
        return [r for r in self.results if r.success]

    @property
    def failed(self) -> List[SweepResult]:
        return [r for r in self.results if not r.success]


class ParameterSweep:
    """Grid search over pipeline parameters with organized outputs.

    Dual interface following PipeGridSearch pattern:
    - **CLI/Programmatic**: `process(image, njobs=-1)` for local parallel execution
    - **Panel GUI**: `configurator()` returns interactive setup panel

    Args:
        pipe_cfgs: Dictionary mapping pipeline configuration names to operation lists.
            Each value is a list of (operation_instance, parameter_dict) tuples where:
            - operation_instance: An ImageOperation subclass instance
            - parameter_dict: Maps parameter names to lists of values to test
        output_dir: Directory where results will be saved. If None, uses temp directory.
        data2save: Image layers to persist to disk. Valid options:
            - "rgb": Original color image (if available)
            - "gray": Grayscale luminance
            - "detect_mat": Enhanced grayscale for processing
            - "objmask": Binary detection mask
            - "objmap": Labeled object map
            - "overlay": Matplotlib overlay figure
        ground_truth_dir: Optional directory with ground truth masks for metrics.

    Examples:
        **Programmatic Usage (like PipeGridSearch):**

        >>> from phenotypic.ui.sweep import ParameterSweep
        >>> from phenotypic.enhance import GaussianBlur, CLAHE
        >>> from phenotypic.detect import OtsuDetector
        >>> from phenotypic import GridImage
        >>>
        >>> # Define parameter grid (4×3×2 = 24 combinations)
        >>> pipe_cfgs = {
        ...     "DetectionPipeline": [
        ...         (GaussianBlur(), {"sigma": [1.0, 1.5, 2.0, 2.5]}),
        ...         (CLAHE(), {"clip_limit": [1.5, 2.0, 2.5]}),
        ...         (OtsuDetector(), {"ignore_zeros": [True, False]}),
        ...     ]
        ... }
        >>>
        >>> # Create sweep
        >>> sweep = ParameterSweep(
        ...     pipe_cfgs=pipe_cfgs,
        ...     output_dir="./sweep_results",
        ...     data2save={"detect_mat", "objmask", "overlay"}
        ... )
        >>>
        >>> # Load image
        >>> image = GridImage.imread("plate_001.jpg", nrows=8, ncols=12)
        >>>
        >>> # Execute with auto memory-aware job scaling
        >>> results = sweep.process(image, njobs=-1)

        **Panel GUI Usage:**

        >>> # Get interactive configurator
        >>> configurator = sweep.configurator()
        >>> configurator.servable()  # Or display in Jupyter
    """

    DEFAULT_VIEWS = {'rgb', 'gray', 'detect_mat', 'objmask', 'objmap', 'overlay'}

    def __init__(
        self,
        pipe_cfgs: Dict[str, List[Tuple[ImageOperation, Dict[str, List[Any]]]]],
        output_dir: Optional[Union[Path, str]] = None,
        data2save: Optional[Set[str]] = None,
        ground_truth_dir: Optional[Union[Path, str]] = None,
    ):
        self.pipe_cfgs = pipe_cfgs
        self.output_dir = Path(output_dir) if output_dir else self._create_temp_dir()
        self.data2save = data2save or self.DEFAULT_VIEWS
        self.ground_truth_dir = Path(ground_truth_dir) if ground_truth_dir else None

    def process(
        self,
        image: Union[Image, Path, List[Path]],
        njobs: int = -1
    ) -> SweepResults:
        """Execute parameter sweep with local parallel processing."""
        # Implementation details...
        pass

    def configurator(self) -> 'SweepConfigurator':
        """Get interactive Panel configurator for this sweep."""
        return SweepConfigurator(self)
```

### SweepConfigurator Panel GUI

```python
import param
import panel as pn


class SweepConfigurator(param.Parameterized):
    """Interactive Panel GUI for configuring and running parameter sweeps.

    Provides:
    - Visual pipeline configuration editing
    - Parameter range specification with editable tables
    - Data output selection
    - Run controls with progress feedback
    """

    # Configuration parameters
    output_dir = param.Path(doc="Output directory for sweep results")
    njobs = param.Integer(default=-1, bounds=(-1, None), doc="Number of parallel jobs (-1=auto)")

    # Data to save
    save_rgb = param.Boolean(default=True, doc="Save RGB images")
    save_gray = param.Boolean(default=True, doc="Save grayscale images")
    save_detect_mat = param.Boolean(default=True, doc="Save detection matrix")
    save_objmask = param.Boolean(default=True, doc="Save object masks")
    save_objmap = param.Boolean(default=True, doc="Save labeled object maps")
    save_overlay = param.Boolean(default=True, doc="Save overlay figures")

    # Ground truth
    ground_truth_dir = param.Path(default=None, doc="Ground truth directory (optional)")

    # Run state
    running = param.Boolean(default=False, doc="Whether sweep is running")
    progress = param.Number(default=0, bounds=(0, 100), doc="Progress percentage")
    status_message = param.String(default="Ready", doc="Current status")

    def __init__(self, sweep: ParameterSweep, **params):
        super().__init__(**params)
        self._sweep = sweep
        # ... initialization

    def panel(self) -> pn.viewable.Viewable:
        """Build the configurator panel."""
        # ... build UI
        pass
```

### UI Layout for SweepConfigurator
```
┌─────────────────────────────────────────────────────────────────┐
│  # Parameter Sweep Configurator                                 │
├─────────────────────────────────────────────────────────────────┤
│  ### Sweep Summary                                              │
│  - Pipeline configurations: 1                                   │
│  - Total combinations: 24                                       │
│  - Output directory: ./sweep_results                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────┐  ┌────────────────────────────┐ │
│  │ ### Pipeline Configurations│  │ ### Output Settings        │ │
│  │ ┌────────────────────────┐ │  │ Output Directory: [______] │ │
│  │ │ ▶ DetectionPipeline    │ │  │ Parallel Jobs:    [-1   ] │ │
│  │ │   - GaussianBlur:      │ │  │ Ground Truth Dir: [______] │ │
│  │ │     sigma: [1,1.5,2]   │ │  │                            │ │
│  │ │   - CLAHE:             │ │  │ ### Data to Save           │ │
│  │ │     clip_limit: [1.5,2]│ │  │ [x] RGB                    │ │
│  │ │   - OtsuDetector:      │ │  │ [x] Grayscale              │ │
│  │ │     ignore_zeros: T/F  │ │  │ [x] Detection Matrix     │ │
│  │ └────────────────────────┘ │  │ [x] Object Mask            │ │
│  └────────────────────────────┘  │ [x] Object Map             │ │
│                                  │ [x] Overlay                │ │
│                                  └────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  ### Execution                                                  │
│  [Run Sweep]                                                    │
│  [████████████████░░░░░░░░░░] 60%                              │
│  Processing pipeline 15/24...                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## HTML Viewer

The HTML viewer is generated via Jinja2 templates with keyboard navigation for comparing sweep results.

### Manifest Structure

```json
{
  "version": "1.0",
  "created": "2024-01-28T14:30:22",
  "phenotypic_version": "0.13.0",
  "source_images": [
    {"name": "plate_001.tif", "path": "original/plate_001.tif"}
  ],
  "pipelines": [
    {
      "id": "DetectionPipeline_000",
      "name": "DetectionPipeline #0",
      "config_file": "pipelines/DetectionPipeline_000.json",
      "parameters": {
        "GaussianBlur.sigma": 1.0,
        "CLAHE.clip_limit": 1.5,
        "OtsuDetector.ignore_zeros": true
      }
    }
  ],
  "ground_truth": {
    "enabled": false,
    "source": null
  },
  "results": [
    {
      "pipeline_id": "DetectionPipeline_000",
      "image_name": "plate_001.tif",
      "success": true,
      "outputs": {
        "rgb": "images/DetectionPipeline_000/plate_001/rgb.png",
        "objmask": "images/DetectionPipeline_000/plate_001/objmask.png"
      },
      "metrics": {
        "accuracy": 0.94,
        "iou": 0.87,
        "num_objects": 384
      }
    }
  ]
}
```

---

## Dependencies: [gui] Optional Group

```toml
# pyproject.toml additions
[project.optional-dependencies]
gui = [
    # Panel ecosystem
    "panel>=1.3",               # Main widget framework
    "param>=2.0",               # Parameterized classes for reactive widgets
    "bokeh>=3.3",               # Underlying visualization (Panel dependency)

    # HoloViz complementary packages
    "hvplot>=0.9",              # High-level plotting API
    "holoviews>=1.18",          # Composable visualizations

    # HTML viewer generation
    "jinja2>=3.1",              # Template engine for sweep viewer HTML
]
```

---

## File Structure (Full)

```
src/phenotypic/ui/
├── __init__.py                     # Public API exports
├── _instance_manager.py            # Workspace/session management
├── _operation_registry.py          # Operation discovery and categorization
├── _param_wrapper.py               # Dynamic param.Parameterized wrapping
│
├── widgets/                        # Panel widget components
│   ├── __init__.py
│   ├── _operation_card.py          # Single operation card with param widgets
│   ├── _add_operation_menu.py      # Categorized operation selector
│   ├── _preview_panel.py           # Image preview panel
│   └── _pipeline_builder.py        # Main builder container
│
├── sweep/                          # Parameter sweep system
│   ├── __init__.py
│   ├── _parameter_sweep.py         # Main sweep class (CLI + GUI)
│   ├── _sweep_configurator.py      # Panel GUI for sweep setup
│   └── _ground_truth.py            # GT comparison utilities
│
└── viewer/                         # HTML viewer generation
    ├── __init__.py
    ├── _sweep_viewer.py            # Viewer generator class
    └── templates/
        ├── sweep_viewer.html       # Main Jinja2 template
        ├── _styles.css             # Embedded CSS
        └── _scripts.js             # Embedded JavaScript

tests/unit/ui/
├── __init__.py
├── test_instance_manager.py
├── test_operation_registry.py
├── test_param_wrapper.py
├── test_pipeline_builder.py
├── test_parameter_sweep.py
└── test_sweep_viewer.py
```

---

## Implementation Phases (Full)

| Phase | Component | Effort | Dependencies |
|-------|-----------|--------|--------------|
| 1 | InstanceManager | Small | None |
| 2a | Operation Registry | Small | Phase 1 |
| 2b | Param Wrapper Utilities | Small | Phase 2a |
| 2c | OperationCard (Panel) | Medium | Phase 2b |
| 2d | PipelineBuilder (Panel) | Medium | Phase 2c |
| 3a | ParameterSweep (CLI interface) | Medium | Phase 1 |
| 3b | SweepConfigurator (Panel GUI) | Medium | Phase 3a |
| 4a | HTML Viewer Templates | Medium | None |
| 4b | SweepViewer Integration | Small | Phase 3a, 4a |

**Total estimated: ~2,200 LOC** (excluding tests)

---

## Design Decisions

1. **Framework**: Panel + param (declarative reactivity, dynamic widget generation)
2. **Operation Reordering**: Up/down arrows (simple, reliable)
3. **Embedded Pipelines**: Editable in-place
4. **Sweep Scale**: Small (1-50 images), sync execution with parallel processing via joblib
5. **Comparison View**: Single view with keyboard navigation between pipelines
6. **Ground Truth Format**: Binary PNG (black/white masks)
7. **Parameter Display**: pn.Card (collapsed by default, click to expand)
8. **Sweep Interface**: Dual - CLI/programmatic (like PipeGridSearch) + Panel GUI
