"""Embedded pipeline editor component for PhenoTypic GUI.

Provides a compact inline pipeline editor for operation parameters
that accept ImagePipeline instances. Uses stable widget pattern to
avoid Bokeh reference errors.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple


class EmbeddedPipelineEditor:
    """Compact inline pipeline editor for embedding in operation parameters.

    Used when an operation parameter accepts ImagePipeline (e.g., FilamentousFungiDetector).
    Shows a simplified version of the full PipelineBuilder.

    Uses stable widget pattern: widgets are created once and only their
    content/options are updated, avoiding Bokeh UnknownReferenceError.

    Features:
    - Nesting depth tracking (operations within this editor inherit depth + 1)
    - Compact layout for nested contexts
    - Select-based operation list (stable widget)
    - Widget caching to prevent Bokeh reference errors
    """

    def __init__(
        self,
        pipeline=None,  # ImagePipeline
        manager=None,  # InstanceManager
        on_change: Optional[Callable[[Any], None]] = None,
        nesting_depth: int = 0,
        **params,
    ):
        """Initialize EmbeddedPipelineEditor.

        Args:
            pipeline: Initial ImagePipeline (creates empty if None)
            manager: InstanceManager for loading saved pipelines
            on_change: Callback when pipeline changes
            nesting_depth: Current nesting level
            **params: Additional parameters
        """
        from phenotypic import ImagePipeline

        self._pipeline = pipeline or ImagePipeline([])
        self._manager = manager
        self._on_change = on_change
        self._nesting_depth = nesting_depth

        # Internal state: list of (name, operation) tuples
        self._operations: List[Tuple[str, Any]] = []
        # Measurements preserved but not editable in embedded context
        self._measurements: List[Tuple[str, Any]] = []

        if pipeline:
            # Use public accessors if available, fallback to private attrs
            ops = (
                pipeline.get_ops()
                if hasattr(pipeline, "get_ops")
                else dict(pipeline._ops)
            )
            meas = (
                pipeline.get_meas()
                if hasattr(pipeline, "get_meas")
                else dict(pipeline._meas)
            )
            for name, op in ops.items():
                self._operations.append((name, op))
            for name, m in meas.items():
                self._measurements.append((name, m))

        # Selection tracking - auto-select first if available
        self._selected_index: int = 0 if self._operations else -1

        # Stable widget references (set in panel())
        self._ops_list_widget: Optional[Any] = None
        self._detail_panel: Optional[Any] = None
        self._up_btn: Optional[Any] = None
        self._down_btn: Optional[Any] = None
        self._delete_btn: Optional[Any] = None

        # Widget cache for stable widget pattern (prevents Bokeh UnknownReferenceError)
        self._op_widget_cache: Dict[int, Any] = {}
        self._placeholder_widget: Optional[Any] = None

    def panel(self):
        """Build the embedded editor with stable widgets.

        Returns:
            Panel widget for embedded pipeline editing
        """
        import panel as pn

        # === STABLE: Operations List Widget ===
        self._ops_list_widget = pn.widgets.Select(
            name="Operations",
            options={},  # Populated by _refresh_display()
            size=5,  # Smaller for embedded context
            sizing_mode="stretch_width",
        )
        self._ops_list_widget.param.watch(self._on_selection_change, "value")

        # === STABLE: Control Buttons (compact) ===
        self._up_btn = pn.widgets.Button(name="▲", width=40, disabled=True)
        self._down_btn = pn.widgets.Button(name="▼", width=40, disabled=True)
        self._delete_btn = pn.widgets.Button(
            name="×", width=40, button_type="danger", disabled=True
        )

        self._up_btn.on_click(lambda e: self._move_selected(-1))
        self._down_btn.on_click(lambda e: self._move_selected(1))
        self._delete_btn.on_click(lambda e: self._delete_selected())

        controls = pn.Row(
            self._up_btn, self._down_btn, self._delete_btn, sizing_mode="stretch_width"
        )

        # === STABLE: Detail Panel with placeholder ===
        self._placeholder_widget = pn.pane.Markdown("*Select an operation to edit*")
        self._detail_panel = pn.Column(
            self._placeholder_widget,
            sizing_mode="stretch_width",
        )

        # === Add Operation Selector (excludes Measure for embedded) ===
        from ._add_operation_menu import AddOperationMenu

        add_menu = AddOperationMenu(
            on_select=self._add_operation, exclude_categories=["Measure"]
        )

        # === Load from saved (optional) ===
        load_section = pn.Column()
        if self._manager:
            pipelines = self._manager.list_pipelines()
            if pipelines:
                load_select = pn.widgets.Select(
                    name="Load Saved Pipeline",
                    options=[""] + pipelines,
                    value="",
                )
                load_select.param.watch(self._load_pipeline, "value")
                load_section.append(load_select)

        # Initialize display
        self._refresh_display()

        return pn.Column(
            self._ops_list_widget,
            controls,
            self._detail_panel,
            add_menu.panel(),
            load_section,
            css_classes=["embedded-pipeline-editor"],
            sizing_mode="stretch_width",
        )

    def _refresh_display(self):
        """Update widget content without recreating widgets."""
        if self._ops_list_widget is None:
            return

        # Build options dict: display_name -> index
        options = {}
        for i, (name, op) in enumerate(self._operations):
            op_type = self._get_operation_type(op)
            display_name = f"{i+1}. {name} [{op_type}]"
            options[display_name] = i

        # Update list widget options
        self._ops_list_widget.options = options

        # Validate selection
        if self._selected_index >= len(self._operations):
            self._selected_index = len(self._operations) - 1

        # Update selection in widget
        if self._selected_index >= 0 and options:
            self._ops_list_widget.value = self._selected_index
        else:
            self._ops_list_widget.value = None

        # Update button states
        self._update_button_states()

        # Update detail panel (using stable widget pattern)
        self._update_detail_panel()

        # Notify parent of change
        self._notify_change()

    def _update_button_states(self):
        """Update control button enabled/disabled states."""
        has_selection = self._selected_index >= 0
        self._up_btn.disabled = not has_selection or self._selected_index <= 0
        self._down_btn.disabled = (
            not has_selection or self._selected_index >= len(self._operations) - 1
        )
        self._delete_btn.disabled = not has_selection

    def _on_selection_change(self, event):
        """Handle selection change in operations list."""
        new_index = event.new if event.new is not None else -1

        # Only update if index actually changed (prevents double rendering)
        if new_index == self._selected_index:
            return

        self._selected_index = new_index
        self._update_button_states()
        self._update_detail_panel()

    def _update_detail_panel(self):
        """Update detail panel using cached widgets (stable widget pattern).

        This prevents Bokeh UnknownReferenceError by caching widgets and
        toggling visibility instead of recreating them.
        """
        if self._detail_panel is None:
            return

        # Hide all cached operation widgets
        for widget in self._op_widget_cache.values():
            widget.visible = False

        # Show placeholder if no selection
        if self._selected_index < 0 or self._selected_index >= len(self._operations):
            self._placeholder_widget.visible = True
            return

        self._placeholder_widget.visible = False
        name, op = self._operations[self._selected_index]
        op_id = id(op)

        # Get or create cached widget for this operation
        if op_id not in self._op_widget_cache:
            widget = self._build_operation_panel(name, op)
            self._op_widget_cache[op_id] = widget
            self._detail_panel.append(widget)

        self._op_widget_cache[op_id].visible = True

    def _build_operation_panel(self, name: str, op: Any) -> Any:
        """Build a complete panel for an operation (cached for stability)."""
        import panel as pn

        op_type = self._get_operation_type(op)

        # Compact header
        header_html = pn.pane.HTML(
            f"<b>{name}</b> "
            f"<span style='background:#e0e0e0;padding:1px 4px;"
            f"border-radius:3px;font-size:0.75em;'>{op_type}</span>"
        )

        # Parameter widgets
        param_widgets = self._build_param_widgets(op)

        components = [header_html]
        if param_widgets.objects:
            components.append(param_widgets)
        else:
            components.append(pn.pane.Markdown("*No parameters*"))

        return pn.Column(*components, sizing_mode="stretch_width")

    def _build_param_widgets(self, operation) -> Any:
        """Build parameter editor widgets for an operation."""
        import panel as pn
        from phenotypic.gui._operation_registry import get_registry

        from ._param_editor import ParamEditor

        registry = get_registry()
        info = registry.get(operation.__class__.__name__)

        widgets = []
        if info:
            for param_name, param_info in info.parameters.items():
                current_value = getattr(operation, param_name, param_info.default)

                # Create callback that re-instantiates the operation
                def make_callback(op, pname):
                    def callback(value):
                        self._update_operation_param(op, pname, value)

                    return callback

                editor = ParamEditor(
                    param_info=param_info,
                    initial_value=current_value,
                    manager=self._manager,
                    on_change=make_callback(operation, param_name),
                    nesting_depth=self._nesting_depth + 1,
                )
                widgets.append(editor.panel())

        return pn.Column(*widgets, sizing_mode="stretch_width")

    def _update_operation_param(self, operation: Any, param_name: str, new_value: Any):
        """Update an operation parameter by re-instantiating.

        This handles operations that may compute derived state in __init__.
        Falls back to setattr if re-instantiation fails.
        """
        from phenotypic.gui._operation_registry import get_registry

        try:
            current_value = getattr(operation, param_name)
        except Exception:
            current_value = None
        if current_value is new_value:
            return
        if isinstance(new_value, (bool, int, float, str, type(None))):
            if current_value == new_value:
                return

        # Find the operation in the list
        op_index = -1
        for i, (name, op) in enumerate(self._operations):
            if op is operation:
                op_index = i
                break

        if op_index < 0:
            # Fallback to setattr if operation not found
            setattr(operation, param_name, new_value)
            self._notify_change()
            return

        registry = get_registry()
        info = registry.get(operation.__class__.__name__)

        if info is None:
            setattr(operation, param_name, new_value)
            self._notify_change()
            return

        try:
            # Collect current parameter values
            params = {}
            for pname, pinfo in info.parameters.items():
                if pname == param_name:
                    params[pname] = new_value
                else:
                    params[pname] = getattr(operation, pname, pinfo.default)

            # Re-instantiate with updated params
            new_operation = info.cls(**params)

            # Get old op_id for cache cleanup
            old_op_id = id(operation)

            # Replace in list
            name = self._operations[op_index][0]
            self._operations[op_index] = (name, new_operation)

            # Update widget cache - remove old, create new on next access
            if old_op_id in self._op_widget_cache:
                old_widget = self._op_widget_cache.pop(old_op_id)
                old_widget.visible = False

            # Force detail panel update to create new cached widget
            self._update_detail_panel()

        except Exception:
            # Fallback to setattr if re-instantiation fails
            setattr(operation, param_name, new_value)

        self._notify_change()

    def _add_operation(self, op_name: str):
        """Add operation by name."""
        from phenotypic.gui._operation_registry import get_registry

        registry = get_registry()
        info = registry.get(op_name)
        if info:
            op = info.cls()
            unique_name = self._make_unique_name(op_name)
            self._operations.append((unique_name, op))

            # Select the newly added operation
            self._selected_index = len(self._operations) - 1
            self._refresh_display()

    def _move_selected(self, direction: int):
        """Move selected operation up (-1) or down (+1)."""
        if self._selected_index < 0:
            return

        new_index = self._selected_index + direction
        if 0 <= new_index < len(self._operations):
            self._operations[self._selected_index], self._operations[new_index] = (
                self._operations[new_index],
                self._operations[self._selected_index],
            )
            self._selected_index = new_index
            self._refresh_display()

    def _delete_selected(self):
        """Delete currently selected operation."""
        if self._selected_index < 0 or self._selected_index >= len(self._operations):
            return

        _name, removed_op = self._operations.pop(self._selected_index)

        # Clean up cached widget
        op_id = id(removed_op)
        if op_id in self._op_widget_cache:
            self._op_widget_cache[op_id].visible = False

        if self._selected_index >= len(self._operations):
            self._selected_index = len(self._operations) - 1

        self._refresh_display()

    def _load_pipeline(self, event):
        """Load a saved pipeline."""
        if event.new and self._manager:
            pipeline = self._manager.load_pipeline(event.new)

            # Clear existing widget cache
            for widget in self._op_widget_cache.values():
                widget.visible = False
            self._op_widget_cache.clear()

            # Use public accessors if available
            ops = (
                pipeline.get_ops()
                if hasattr(pipeline, "get_ops")
                else dict(pipeline._ops)
            )
            meas = (
                pipeline.get_meas()
                if hasattr(pipeline, "get_meas")
                else dict(pipeline._meas)
            )

            self._operations = [(n, op) for n, op in ops.items()]
            self._measurements = [(n, m) for n, m in meas.items()]
            self._selected_index = 0 if self._operations else -1
            self._refresh_display()

    def _make_unique_name(self, base: str) -> str:
        """Generate unique operation name."""
        existing = {name for name, _ in self._operations}
        if base not in existing:
            return base
        i = 1
        while f"{base}_{i}" in existing:
            i += 1
        return f"{base}_{i}"

    def _get_operation_type(self, op) -> str:
        """Get human-readable operation type."""
        from phenotypic.abc_ import (
            GridOperation,
            ImageCorrector,
            ImageEnhancer,
            MeasureFeatures,
            ObjectDetector,
            ObjectRefiner,
        )

        if isinstance(op, ImageEnhancer):
            return "Enhancer"
        if isinstance(op, ObjectDetector):
            return "Detector"
        if isinstance(op, ObjectRefiner):
            return "Refiner"
        if isinstance(op, ImageCorrector):
            return "Corrector"
        if isinstance(op, MeasureFeatures):
            return "Measure"
        if isinstance(op, GridOperation):
            return "Grid"
        return "Operation"

    def _notify_change(self):
        """Notify parent of pipeline change."""
        if self._on_change:
            pipeline = self._get_pipeline()
            self._on_change(pipeline)

    def _get_pipeline(self):
        """Build ImagePipeline from current operations and preserved measurements."""
        from phenotypic import ImagePipeline

        pipeline = ImagePipeline({name: op for name, op in self._operations})
        if self._measurements:
            pipeline.set_meas({name: meas for name, meas in self._measurements})
        return pipeline
