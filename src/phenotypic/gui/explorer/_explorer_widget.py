"""Main Pipeline Explorer widget combining graph editor and parameter controls.

Provides an interactive Panel-based interface for exploring pipeline variants
through visual graph editing and parameter sweeps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
import logging

try:
    import param
    import panel as pn

    PANEL_AVAILABLE = True
except ImportError:
    PANEL_AVAILABLE = False
    param = None
    pn = None

from ._pipeline_graph import PipelineGraph
from ._sweep_spec import SweepSpec
from ._sweep_executor import SweepExecutor
from ._sweep_results import SweepResults

if TYPE_CHECKING:
    from phenotypic import Image
    from phenotypic.abc_ import ImageOperation

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================


def get_operation_categories() -> Dict[str, List[type]]:
    """Get available operations organized by category.

    Returns:
        Dictionary mapping category names to lists of operation classes.
    """
    from phenotypic.gui._operation_registry import OperationRegistry

    registry = OperationRegistry()
    operations = registry.get_all()

    categories = {}
    for op_class in operations:
        # Determine category from module path
        module = op_class.__module__
        if "enhance" in module:
            category = "Enhance"
        elif "detect" in module:
            category = "Detect"
        elif "refine" in module:
            category = "Refine"
        elif "correct" in module:
            category = "Correct"
        elif "measure" in module:
            category = "Measure"
        else:
            category = "Other"

        categories.setdefault(category, []).append(op_class)

    return categories


def build_sweep_spec_from_inputs(
    param_name: str,
    sweep_type: str,
    start: float,
    stop: float,
    step_or_num: float,
    values_str: str,
) -> Optional[SweepSpec]:
    """Build a SweepSpec from UI input values.

    Args:
        param_name: Parameter name to sweep.
        sweep_type: One of 'range', 'linspace', 'logspace', 'values'.
        start: Start value for numeric sweeps.
        stop: Stop value for numeric sweeps.
        step_or_num: Step size or number of points.
        values_str: Comma-separated values for explicit list.

    Returns:
        SweepSpec or None if inputs are invalid.
    """
    try:
        if sweep_type == "range":
            return SweepSpec.from_range(param_name, start, stop, step_or_num)
        elif sweep_type == "linspace":
            return SweepSpec.from_linspace(param_name, start, stop, int(step_or_num))
        elif sweep_type == "logspace":
            return SweepSpec.from_logspace(param_name, start, stop, int(step_or_num))
        elif sweep_type == "values":
            # Parse comma-separated values
            values = []
            for v in values_str.split(","):
                v = v.strip()
                if not v:
                    continue
                # Try to parse as number
                try:
                    if "." in v:
                        values.append(float(v))
                    else:
                        values.append(int(v))
                except ValueError:
                    # Keep as string
                    values.append(v)
            if values:
                return SweepSpec(param_name, values)
        return None
    except Exception as e:
        logger.warning(f"Failed to build sweep spec: {e}")
        return None


def format_variant_count(count: int) -> str:
    """Format variant count for display.

    Args:
        count: Number of variants.

    Returns:
        Formatted string.
    """
    if count == 0:
        return "No variants"
    elif count == 1:
        return "1 variant"
    else:
        return f"{count:,} variants"


# =============================================================================
# PipelineExplorer Widget
# =============================================================================


if PANEL_AVAILABLE:

    class PipelineExplorer(param.Parameterized):
        """Main widget for exploring pipeline variants.

        Provides a visual interface for:
        - Building pipeline graphs with drag-and-drop
        - Configuring parameter sweeps on operations
        - Executing all variants in parallel
        - Viewing and comparing results

        Layout:
        ```
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
        │               │               └──────┘   │     │ stop:  [3.0]    │
        │               │                          ▼     │ step:  [0.5]    │
        │               │               ┌──────────┐     │ → 5 values      │
        │               │               │  Output  │     │                 │
        │               │               └──────────┘     │                 │
        ├─────────────────────────────────────────────────────────────────────┤
        │ Input: [./images/*.tif]  Summary: 2 paths × 5 sigma = 10 variants  │
        └─────────────────────────────────────────────────────────────────────┘
        ```

        Args:
            image: Optional preview image for testing operations.
            graph: Optional initial PipelineGraph.

        Examples:
            Basic usage:

            >>> explorer = PipelineExplorer()
            >>> explorer.panel()

            With initial graph:

            >>> from phenotypic.gui.explorer import PipelineGraph
            >>> from phenotypic.enhance import GaussianBlur
            >>> graph = PipelineGraph.linear(GaussianBlur(sigma=1.5))
            >>> explorer = PipelineExplorer(graph=graph)
            >>> explorer.panel()
        """

        # === Configuration Parameters ===
        output_dir = param.String(
            default="./sweep_results",
            doc="Directory to save sweep outputs",
        )

        image_source = param.String(
            default="",
            doc="Path to image(s) or directory to process",
        )

        njobs = param.Integer(
            default=-1,
            bounds=(-1, None),
            doc="Number of parallel jobs (-1 = all CPUs)",
        )

        # === Output Selection ===
        save_overlay = param.Boolean(default=True, doc="Save overlay images")
        save_objmask = param.Boolean(default=True, doc="Save object masks")
        save_objmap = param.Boolean(default=False, doc="Save labeled object maps")
        save_enh_gray = param.Boolean(default=False, doc="Save enhanced grayscale")

        # === State ===
        is_running = param.Boolean(default=False, doc="Whether sweep is running")
        progress_value = param.Number(default=0, bounds=(0, 100))
        progress_message = param.String(default="")
        results = param.ClassSelector(
            class_=SweepResults,
            default=None,
            allow_None=True,
            doc="Results from last sweep",
        )

        def __init__(
            self,
            image: Optional[Image] = None,
            graph: Optional[PipelineGraph] = None,
            **params,
        ):
            """Initialize the Pipeline Explorer.

            Args:
                image: Optional preview image.
                graph: Optional initial graph.
                **params: Additional parameters.
            """
            super().__init__(**params)

            self._preview_image = image
            self._graph = graph or PipelineGraph()
            self._selected_op_class: Optional[type] = None

            # Initialize widgets
            self._init_widgets()

        def _init_widgets(self) -> None:
            """Initialize all UI widgets."""
            # Import node editor here to avoid circular imports
            from ._node_editor import PipelineNodeEditor

            # Graph editor
            self._editor = PipelineNodeEditor(graph=self._graph, height=500)

            # Operations sidebar
            self._ops_accordion = self._build_operations_accordion()

            # Parameter editor panel
            self._param_panel = pn.Column(
                pn.pane.Markdown("### Node Parameters"),
                pn.pane.Markdown("*Select a node to edit parameters*"),
                sizing_mode="stretch_width",
            )

            # Sweep configuration panel
            self._sweep_panel = self._build_sweep_panel()

            # Footer controls
            self._footer = self._build_footer()

            # Progress indicator
            self._progress = pn.indicators.Progress(
                value=0,
                max=100,
                sizing_mode="stretch_width",
                visible=False,
            )
            self._progress_text = pn.pane.Markdown("")

            # Watch for node selection
            self._editor.param.watch(
                self._on_node_selected,
                ["selected_node_id"],
            )
            # Keep summary in sync with graph changes
            self._editor.param.watch(
                self._on_graph_changed,
                ["nodes", "edges"],
            )

        def _build_operations_accordion(self) -> pn.Accordion:
            """Build the operations sidebar accordion.

            Returns:
                Accordion widget with operation buttons.
            """
            categories = get_operation_categories()

            accordion = pn.Accordion(
                sizing_mode="stretch_width",
                active=[0],  # First category expanded
            )

            for category, ops in sorted(categories.items()):
                buttons = []
                for op_class in sorted(ops, key=lambda x: x.__name__):
                    btn = pn.widgets.Button(
                        name=op_class.__name__,
                        button_type="light",
                        sizing_mode="stretch_width",
                    )
                    # Capture op_class in closure
                    btn.on_click(self._make_add_operation_callback(op_class))
                    buttons.append(btn)

                accordion.append((category, pn.Column(*buttons)))

            # Add output button separately
            output_btn = pn.widgets.Button(
                name="Add Output Node",
                button_type="primary",
                sizing_mode="stretch_width",
            )
            output_btn.on_click(self._add_output_node)
            accordion.append(("Output", pn.Column(output_btn)))

            return accordion

        def _make_add_operation_callback(self, op_class: type):
            """Create callback for adding an operation.

            Args:
                op_class: Operation class to add.

            Returns:
                Callback function.
            """
            def callback(event):
                self._add_operation(op_class)

            return callback

        def _build_sweep_panel(self) -> pn.Column:
            """Build the sweep configuration panel.

            Returns:
                Column containing sweep controls.
            """
            self._sweep_enabled = pn.widgets.Checkbox(
                name="Enable Sweep",
                value=False,
            )
            self._sweep_param = pn.widgets.Select(
                name="Parameter",
                options=[],
                disabled=True,
            )
            self._sweep_type = pn.widgets.Select(
                name="Type",
                options=["range", "linspace", "logspace", "values"],
                value="range",
                disabled=True,
            )
            self._sweep_start = pn.widgets.FloatInput(
                name="Start",
                value=0,
                disabled=True,
            )
            self._sweep_stop = pn.widgets.FloatInput(
                name="Stop",
                value=1,
                disabled=True,
            )
            self._sweep_step = pn.widgets.FloatInput(
                name="Step/Num",
                value=0.1,
                disabled=True,
            )
            self._sweep_values = pn.widgets.TextInput(
                name="Values (comma-separated)",
                value="",
                disabled=True,
                visible=False,
            )
            self._sweep_apply = pn.widgets.Button(
                name="Apply Sweep",
                button_type="primary",
                disabled=True,
            )
            self._sweep_clear = pn.widgets.Button(
                name="Clear Sweep",
                button_type="warning",
                disabled=True,
            )

            # Wire up callbacks
            self._sweep_enabled.param.watch(self._on_sweep_enabled_change, ["value"])
            self._sweep_type.param.watch(self._on_sweep_type_change, ["value"])
            self._sweep_apply.on_click(self._apply_sweep)
            self._sweep_clear.on_click(self._clear_sweep)

            return pn.Column(
                pn.pane.Markdown("### Sweep Configuration"),
                self._sweep_enabled,
                self._sweep_param,
                self._sweep_type,
                pn.Row(self._sweep_start, self._sweep_stop),
                self._sweep_step,
                self._sweep_values,
                pn.Row(self._sweep_apply, self._sweep_clear),
                sizing_mode="stretch_width",
            )

        def _build_footer(self) -> pn.Row:
            """Build the footer controls.

            Returns:
                Row containing footer widgets.
            """
            self._image_input = pn.widgets.TextInput(
                name="Images",
                placeholder="Path to images or directory...",
                value=self.image_source,
                sizing_mode="stretch_width",
            )
            self._output_input = pn.widgets.TextInput(
                name="Output Directory",
                value=self.output_dir,
                width=200,
            )
            self._summary_text = pn.pane.Markdown(
                self._get_summary_text(),
                sizing_mode="stretch_width",
            )
            self._run_button = pn.widgets.Button(
                name="Run Sweep",
                button_type="success",
                width=120,
            )
            self._run_button.on_click(self._run_sweep)

            # Output checkboxes
            self._save_overlay_checkbox = pn.widgets.Checkbox(
                name="Overlay",
                value=self.save_overlay,
            )
            self._save_objmask_checkbox = pn.widgets.Checkbox(
                name="ObjMask",
                value=self.save_objmask,
            )
            self._save_objmap_checkbox = pn.widgets.Checkbox(
                name="ObjMap",
                value=self.save_objmap,
            )
            self._save_enh_gray_checkbox = pn.widgets.Checkbox(
                name="EnhGray",
                value=self.save_enh_gray,
            )
            self._bind_checkbox(self._save_overlay_checkbox, "save_overlay")
            self._bind_checkbox(self._save_objmask_checkbox, "save_objmask")
            self._bind_checkbox(self._save_objmap_checkbox, "save_objmap")
            self._bind_checkbox(self._save_enh_gray_checkbox, "save_enh_gray")

            output_checks = pn.Row(
                self._save_overlay_checkbox,
                self._save_objmask_checkbox,
                self._save_objmap_checkbox,
                self._save_enh_gray_checkbox,
            )

            return pn.Column(
                pn.Row(
                    self._image_input,
                    self._output_input,
                    self._run_button,
                    sizing_mode="stretch_width",
                ),
                pn.Row(
                    self._summary_text,
                    output_checks,
                    sizing_mode="stretch_width",
                ),
                sizing_mode="stretch_width",
            )

        # =====================================================================
        # Event Handlers
        # =====================================================================

        def _add_operation(self, op_class: type) -> None:
            """Add an operation node to the graph.

            Args:
                op_class: Operation class to add.
            """
            node_id = self._editor.add_operation_node(op_class)
            self._update_summary()
            logger.debug(f"Added operation: {op_class.__name__} ({node_id})")

        def _add_output_node(self, event=None) -> None:
            """Add an output node to the graph."""
            node_id = self._editor.add_output_node()
            self._update_summary()
            logger.debug(f"Added output node: {node_id}")

        def _on_node_selected(self, event) -> None:
            """Handle node selection change.

            Args:
                event: Param event.
            """
            node_id = event.new
            if node_id is None:
                self._param_panel.objects = [
                    pn.pane.Markdown("### Node Parameters"),
                    pn.pane.Markdown("*Select a node to edit parameters*"),
                ]
                self._disable_sweep_controls()
                return

            node_data = self._editor.get_selected_node_data()
            if node_data is None:
                return

            # Build parameter editor
            param_widgets = [
                pn.pane.Markdown(f"### {node_data['label']}"),
                pn.pane.Markdown(f"Type: **{node_data['opType']}**"),
            ]

            if node_data["opType"] != "Output":
                # Add parameter inputs
                params = node_data.get("params", {})
                param_names = list(params.keys())

                for param_name, value in params.items():
                    if isinstance(value, bool):
                        widget = pn.widgets.Checkbox(
                            name=param_name,
                            value=value,
                        )
                    elif isinstance(value, int):
                        widget = pn.widgets.IntInput(
                            name=param_name,
                            value=value,
                        )
                    elif isinstance(value, float):
                        widget = pn.widgets.FloatInput(
                            name=param_name,
                            value=value,
                        )
                    elif isinstance(value, str):
                        widget = pn.widgets.TextInput(
                            name=param_name,
                            value=value,
                        )
                    else:
                        widget = pn.pane.Markdown(f"**{param_name}**: {value}")

                    if hasattr(widget, "param") and "value" in widget.param:
                        widget.param.watch(
                            self._make_param_update_callback(
                                node_id,
                                param_name,
                            ),
                            ["value"],
                        )

                    param_widgets.append(widget)

                # Enable sweep controls
                self._enable_sweep_controls(param_names)

            self._param_panel.objects = param_widgets

        def _make_param_update_callback(
            self,
            node_id: str,
            param_name: str,
        ):
            """Create callback for updating node parameters."""
            def callback(event):
                self._editor.update_node_params(
                    node_id,
                    {param_name: event.new},
                )

            return callback

        def _bind_checkbox(
            self,
            checkbox: pn.widgets.Checkbox,
            param_name: str,
        ) -> None:
            """Bind a checkbox to a Parameterized boolean param."""
            def _from_widget(event):
                if getattr(self, param_name) != event.new:
                    setattr(self, param_name, event.new)

            def _from_param(event):
                if checkbox.value != event.new:
                    checkbox.value = event.new

            checkbox.param.watch(_from_widget, ["value"])
            self.param.watch(_from_param, param_name)

        def _on_graph_changed(self, event) -> None:
            """Handle graph edits to keep summary in sync."""
            self._update_summary()

        def _enable_sweep_controls(self, param_names: List[str]) -> None:
            """Enable sweep controls for available parameters.

            Args:
                param_names: List of parameter names.
            """
            self._sweep_enabled.disabled = False
            self._sweep_param.options = param_names
            if param_names:
                self._sweep_param.value = param_names[0]

        def _disable_sweep_controls(self) -> None:
            """Disable all sweep controls."""
            self._sweep_enabled.value = False
            self._sweep_enabled.disabled = True
            self._sweep_param.options = []
            self._sweep_param.disabled = True
            self._sweep_type.disabled = True
            self._sweep_start.disabled = True
            self._sweep_stop.disabled = True
            self._sweep_step.disabled = True
            self._sweep_values.disabled = True
            self._sweep_apply.disabled = True
            self._sweep_clear.disabled = True

        def _on_sweep_enabled_change(self, event) -> None:
            """Handle sweep enabled checkbox change."""
            enabled = event.new
            self._sweep_param.disabled = not enabled
            self._sweep_type.disabled = not enabled
            self._sweep_start.disabled = not enabled
            self._sweep_stop.disabled = not enabled
            self._sweep_step.disabled = not enabled
            self._sweep_apply.disabled = not enabled
            self._sweep_clear.disabled = not enabled

            if enabled:
                self._on_sweep_type_change(None)

        def _on_sweep_type_change(self, event) -> None:
            """Handle sweep type change."""
            sweep_type = self._sweep_type.value
            is_values = sweep_type == "values"

            self._sweep_start.visible = not is_values
            self._sweep_stop.visible = not is_values
            self._sweep_step.visible = not is_values
            self._sweep_values.visible = is_values
            self._sweep_values.disabled = not is_values

            # Update step label
            if sweep_type == "range":
                self._sweep_step.name = "Step"
            else:
                self._sweep_step.name = "Num Points"

        def _apply_sweep(self, event=None) -> None:
            """Apply sweep configuration to selected node."""
            node_id = self._editor.selected_node_id
            if not node_id:
                return

            sweep = build_sweep_spec_from_inputs(
                param_name=self._sweep_param.value,
                sweep_type=self._sweep_type.value,
                start=self._sweep_start.value,
                stop=self._sweep_stop.value,
                step_or_num=self._sweep_step.value,
                values_str=self._sweep_values.value,
            )

            if sweep:
                self._editor.update_node_sweep(node_id, sweep, replace=True)
                self._update_summary()
                logger.info(f"Applied sweep to {node_id}: {sweep}")

        def _clear_sweep(self, event=None) -> None:
            """Clear sweep from selected node."""
            node_id = self._editor.selected_node_id
            if not node_id:
                return

            self._editor.update_node_sweep(node_id, None)
            self._update_summary()

        def _run_sweep(self, event=None) -> None:
            """Execute the parameter sweep."""
            if self.is_running:
                return

            # Get image source
            image_source = self._image_input.value.strip()
            if not image_source:
                pn.state.notifications.error("Please specify image source")
                return

            # Get output directory
            output_dir = self._output_input.value.strip()
            if not output_dir:
                pn.state.notifications.error("Please specify output directory")
                return

            # Get graph
            graph = self._editor.to_pipeline_graph()
            issues = graph.validate()
            if issues:
                pn.state.notifications.warning(
                    f"Graph validation warnings: {', '.join(issues)}"
                )

            # Get data to save
            data2save = set()
            if self.save_overlay:
                data2save.add("overlay")
            if self.save_objmask:
                data2save.add("objmask")
            if self.save_objmap:
                data2save.add("objmap")
            if self.save_enh_gray:
                data2save.add("enh_gray")

            # Run sweep
            self.is_running = True
            self._run_button.disabled = True
            self._progress.visible = True

            try:
                executor = SweepExecutor(
                    graph=graph,
                    output_dir=output_dir,
                    data2save=data2save,
                    njobs=self.njobs,
                )

                def progress_callback(current: int, total: int, message: str):
                    self.progress_value = (current / total) * 100 if total > 0 else 0
                    self.progress_message = message
                    self._progress.value = int(self.progress_value)
                    self._progress_text.object = message

                self.results = executor.run(
                    images=image_source,
                    progress_callback=progress_callback,
                )

                pn.state.notifications.success(
                    f"Sweep complete: {len(self.results.successful)}/{len(self.results.results)} successful"
                )

            except Exception as e:
                logger.exception("Sweep failed")
                pn.state.notifications.error(f"Sweep failed: {e}")

            finally:
                self.is_running = False
                self._run_button.disabled = False
                self._progress.visible = False

        def _update_summary(self) -> None:
            """Update the summary text."""
            self._summary_text.object = self._get_summary_text()

        def _get_summary_text(self) -> str:
            """Get the summary text for current graph state.

            Returns:
                Markdown summary string.
            """
            try:
                graph = self._editor.to_pipeline_graph()
                path_count = graph.path_count
                variant_count = graph.variant_count

                return (
                    f"**Paths:** {path_count} | "
                    f"**Variants:** {format_variant_count(variant_count)}"
                )
            except Exception as e:
                logger.debug(f"Failed to compute graph summary: {e}")
                return "**Paths:** 0 | **Variants:** 0"

        # =====================================================================
        # Public Interface
        # =====================================================================

        def panel(self) -> pn.viewable.Viewable:
            """Get the Panel layout for display.

            Returns:
                Complete Panel layout.
            """
            # Ensure Panel is initialized
            if not pn.state._extensions_loaded:
                pn.extension()

            # Build layout
            sidebar = pn.Column(
                pn.pane.Markdown("## Operations"),
                self._ops_accordion,
                width=200,
                sizing_mode="stretch_height",
            )

            right_panel = pn.Column(
                self._param_panel,
                pn.layout.Divider(),
                self._sweep_panel,
                width=250,
                sizing_mode="stretch_height",
            )

            main_content = pn.Column(
                self._editor,
                sizing_mode="stretch_both",
            )

            body = pn.Row(
                sidebar,
                pn.layout.Divider(margin=(0, 10)),
                main_content,
                pn.layout.Divider(margin=(0, 10)),
                right_panel,
                sizing_mode="stretch_both",
            )

            footer = pn.Column(
                pn.layout.Divider(),
                self._footer,
                self._progress,
                self._progress_text,
                sizing_mode="stretch_width",
            )

            layout = pn.Column(
                pn.pane.Markdown("# Pipeline Variant Explorer"),
                body,
                footer,
                sizing_mode="stretch_both",
                min_height=700,
            )

            return layout

        def get_graph(self) -> PipelineGraph:
            """Get the current PipelineGraph.

            Returns:
                Current graph state.
            """
            return self._editor.to_pipeline_graph()

        def set_graph(self, graph: PipelineGraph) -> None:
            """Set the current graph.

            Args:
                graph: Graph to display.
            """
            self._editor.load_graph(graph)
            self._update_summary()


else:
    # Stub class when Panel is not available
    class PipelineExplorer:
        """Placeholder when Panel is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PipelineExplorer requires Panel. "
                "Install with: pip install phenotypic[gui]"
            )
