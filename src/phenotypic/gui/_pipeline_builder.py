"""Main PipelineBuilder GUI for PhenoTypic.

Top-level interactive interface for building and testing ImagePipelines
using Panel components with stable widget pattern to avoid Bokeh reference errors.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class PipelineBuilder:
    """Interactive pipeline construction widget using Panel.

    Uses stable widget pattern: widgets are created once in panel() and only
    their content/options are updated, avoiding Bokeh UnknownReferenceError.

    Composes reusable components:
    - AddOperationMenu for selecting operations
    - PreviewPanel for image preview
    - ParamEditor for parameter editing

    Supports embedded operations/pipelines for complex detectors
    like FilamentousFungiDetector.

    Examples:
        Basic usage (auto-initialization in Jupyter):

        >>> from phenotypic.gui import PipelineBuilder
        >>> from phenotypic.data import load_synth_yeast_plate
        >>>
        >>> # Load image and build GUI
        >>> image = load_synth_yeast_plate()
        >>> builder = PipelineBuilder(image=image)  # Auto-uses global manager
        >>>
        >>> # Display in Jupyter (pn.extension() called automatically)
        >>> builder.panel()
        >>>
        >>> # Get current pipeline
        >>> pipeline = builder.get_pipeline()

        Advanced usage with custom manager:

        >>> from phenotypic.gui import PipelineBuilder, InstanceManager
        >>> manager = InstanceManager(workspace="./my_pipelines")
        >>> builder = PipelineBuilder(manager=manager, image=image)
        >>> builder.panel()
    """

    def __init__(
        self,
        pipeline=None,  # ImagePipeline
        manager=None,  # InstanceManager
        image=None,  # Image or GridImage
        **params,
    ):
        """Initialize PipelineBuilder.

        Args:
            pipeline: Initial ImagePipeline to load (creates empty if None)
            manager: InstanceManager for saving/loading pipelines.
                If None, uses the global manager (similar to napari's global viewer).
            image: Preview Image or GridImage
            **params: Additional parameters
        """
        # Auto-initialize Panel in Jupyter
        from ._global_session import _ensure_panel_initialized, get_global_manager

        _ensure_panel_initialized()

        # Use global manager if none provided
        if manager is None:
            manager = get_global_manager()

        self._manager = manager
        self._image = image

        # Operations list: [(name, operation), ...] - source of truth
        self._operations: List[Tuple[str, Any]] = []
        # Measurements list: [(name, measurement), ...] - source of truth
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
        self._selected_meas_index: int = 0 if self._measurements else -1

        # Stable widget references (set in panel())
        self._ops_list_widget: Optional[Any] = None
        self._meas_list_widget: Optional[Any] = None
        self._detail_panel: Optional[Any] = None
        self._meas_detail_panel: Optional[Any] = None
        self._up_btn: Optional[Any] = None
        self._down_btn: Optional[Any] = None
        self._delete_btn: Optional[Any] = None
        self._meas_up_btn: Optional[Any] = None
        self._meas_down_btn: Optional[Any] = None
        self._meas_delete_btn: Optional[Any] = None
        self._load_select_widget: Optional[Any] = None

        # Widget cache for stable widget pattern (prevents Bokeh UnknownReferenceError)
        # Maps operation id() -> cached widget panel
        self._op_widget_cache: Dict[int, Any] = {}
        self._meas_widget_cache: Dict[int, Any] = {}
        self._placeholder_widget: Optional[Any] = None
        self._meas_placeholder_widget: Optional[Any] = None

        # Toast notifications
        from ._toast import ToastNotification

        self._toast = ToastNotification(app=None)

    def panel(self):
        """Build the main layout with stable widgets.

        Returns:
            Panel Column widget with operations panel above preview panel
        """
        import panel as pn
        from .components import AddOperationMenu, PreviewPanel

        # === STABLE: Operations List Widget ===
        self._ops_list_widget = pn.widgets.Select(
            name="Operations",
            options={},  # Populated by _refresh_display()
            size=8,
            sizing_mode="stretch_width",
        )
        self._ops_list_widget.param.watch(self._on_selection_change, "value")

        # === STABLE: Control Buttons ===
        self._up_btn = pn.widgets.Button(name="Move Up", width=100, disabled=True)
        self._down_btn = pn.widgets.Button(name="Move Down", width=100, disabled=True)
        self._delete_btn = pn.widgets.Button(
            name="Delete", width=80, button_type="danger", disabled=True
        )

        self._up_btn.on_click(lambda e: self._move_selected(-1))
        self._down_btn.on_click(lambda e: self._move_selected(1))
        self._delete_btn.on_click(lambda e: self._delete_selected())

        controls = pn.Row(
            self._up_btn, self._down_btn, self._delete_btn,
            sizing_mode="stretch_width",
        )

        # === STABLE: Detail Panel (container stable, content cached) ===
        self._placeholder_widget = pn.pane.Markdown(
            "*Select an operation to edit parameters*"
        )
        self._detail_panel = pn.Column(
            self._placeholder_widget,
            sizing_mode="stretch_width",
            min_height=150,
        )

        # === Add Operation Menu (excludes Measure category) ===
        add_menu = AddOperationMenu(
            on_select=self._add_operation, exclude_categories=["Measure"]
        )

        # === MEASUREMENTS SECTION ===
        self._meas_list_widget = pn.widgets.Select(
            name="Measurements",
            options={},
            size=4,
            sizing_mode="stretch_width",
        )
        self._meas_list_widget.param.watch(self._on_meas_selection_change, "value")

        # Measurement control buttons
        self._meas_up_btn = pn.widgets.Button(name="Move Up", width=100, disabled=True)
        self._meas_down_btn = pn.widgets.Button(
            name="Move Down", width=100, disabled=True
        )
        self._meas_delete_btn = pn.widgets.Button(
            name="Delete", width=80, button_type="danger", disabled=True
        )

        self._meas_up_btn.on_click(lambda e: self._move_selected_meas(-1))
        self._meas_down_btn.on_click(lambda e: self._move_selected_meas(1))
        self._meas_delete_btn.on_click(lambda e: self._delete_selected_meas())

        meas_controls = pn.Row(
            self._meas_up_btn,
            self._meas_down_btn,
            self._meas_delete_btn,
            sizing_mode="stretch_width",
        )

        # Measurement detail panel
        self._meas_placeholder_widget = pn.pane.Markdown(
            "*Select a measurement to edit parameters*"
        )
        self._meas_detail_panel = pn.Column(
            self._meas_placeholder_widget,
            sizing_mode="stretch_width",
            min_height=100,
        )

        # Add Measurement Menu (only Measure category)
        add_meas_menu = AddOperationMenu(
            on_select=self._add_measurement, categories=["Measure"]
        )

        # === Save/Load controls ===
        name_input = pn.widgets.TextInput(placeholder="Pipeline name", width=150)
        save_btn = pn.widgets.Button(name="Save", button_type="success", width=80)
        save_btn.on_click(lambda e: self._save_pipeline(name_input.value))

        load_options = [""] + (
            self._manager.list_pipelines() if self._manager else []
        )
        self._load_select_widget = pn.widgets.Select(
            name="Load", options=load_options, value="", width=200
        )
        self._load_select_widget.param.watch(self._load_pipeline, "value")

        save_load_controls = pn.Row(
            name_input, save_btn, self._load_select_widget, sizing_mode="stretch_width"
        )

        # === Operations Panel Layout ===
        operations_panel = pn.Column(
            "## Pipeline Operations",
            self._ops_list_widget,
            controls,
            pn.layout.Divider(),
            "### Edit Operation",
            self._detail_panel,
            pn.layout.Divider(),
            add_menu.panel(),
            pn.layout.Divider(),
            "## Measurements",
            self._meas_list_widget,
            meas_controls,
            pn.layout.Divider(),
            "### Edit Measurement",
            self._meas_detail_panel,
            pn.layout.Divider(),
            add_meas_menu.panel(),
            sizing_mode="stretch_width",
        )

        # === BOTTOM: Preview Panel (Full Width) ===
        preview = PreviewPanel(
            image=self._image,
            get_pipeline=self.get_pipeline,
        )

        preview_panel = pn.Column(
            pn.layout.Divider(),
            "## Preview",
            preview.panel(),
            sizing_mode="stretch_width",
            min_height=600,
        )

        # Initialize display
        self._refresh_display()
        self._refresh_meas_display()

        return pn.Column(
            "## Pipeline",
            save_load_controls,
            pn.layout.Divider(),
            operations_panel,
            preview_panel,
            sizing_mode="stretch_both",
        )

    def _refresh_display(self):
        """Update widget content without recreating widgets."""
        if self._ops_list_widget is None:
            return  # panel() not called yet

        # Build options dict: display_name -> index
        options = {}
        for i, (name, op) in enumerate(self._operations):
            op_type = self._get_operation_type(op)
            display_name = f"{i+1}. {name} [{op_type}]"
            options[display_name] = i

        # Update list widget options (NOT recreate)
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

    def _refresh_meas_display(self):
        """Update measurement widget content without recreating widgets."""
        if self._meas_list_widget is None:
            return

        options = {}
        for i, (name, meas) in enumerate(self._measurements):
            display_name = f"{i+1}. {name}"
            options[display_name] = i

        self._meas_list_widget.options = options

        if self._selected_meas_index >= len(self._measurements):
            self._selected_meas_index = len(self._measurements) - 1

        if self._selected_meas_index >= 0 and options:
            self._meas_list_widget.value = self._selected_meas_index
        else:
            self._meas_list_widget.value = None

        self._update_meas_button_states()
        self._update_meas_detail_panel()

    def _update_button_states(self):
        """Update control button enabled/disabled states."""
        has_selection = self._selected_index >= 0
        self._up_btn.disabled = not has_selection or self._selected_index <= 0
        self._down_btn.disabled = (
            not has_selection or self._selected_index >= len(self._operations) - 1
        )
        self._delete_btn.disabled = not has_selection

    def _update_meas_button_states(self):
        """Update measurement control button states."""
        has_selection = self._selected_meas_index >= 0
        self._meas_up_btn.disabled = not has_selection or self._selected_meas_index <= 0
        self._meas_down_btn.disabled = (
            not has_selection
            or self._selected_meas_index >= len(self._measurements) - 1
        )
        self._meas_delete_btn.disabled = not has_selection

    def _on_selection_change(self, event):
        """Handle selection change in operations list."""
        new_index = event.new if event.new is not None else -1

        # Only update if index actually changed (prevents double rendering)
        if new_index == self._selected_index:
            return

        self._selected_index = new_index
        self._update_button_states()
        self._update_detail_panel()  # Show parameters for selected operation

    def _on_meas_selection_change(self, event):
        """Handle selection change in measurements list."""
        new_index = event.new if event.new is not None else -1

        if new_index == self._selected_meas_index:
            return

        self._selected_meas_index = new_index
        self._update_meas_button_states()
        self._update_meas_detail_panel()  # Show parameters for selected measurement

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

    def _update_meas_detail_panel(self):
        """Update measurement detail panel using cached widgets."""
        if self._meas_detail_panel is None:
            return

        for widget in self._meas_widget_cache.values():
            widget.visible = False

        if (
            self._selected_meas_index < 0
            or self._selected_meas_index >= len(self._measurements)
        ):
            self._meas_placeholder_widget.visible = True
            return

        self._meas_placeholder_widget.visible = False
        name, meas = self._measurements[self._selected_meas_index]
        meas_id = id(meas)

        if meas_id not in self._meas_widget_cache:
            widget = self._build_measurement_panel(name, meas)
            self._meas_widget_cache[meas_id] = widget
            self._meas_detail_panel.append(widget)

        self._meas_widget_cache[meas_id].visible = True

    def _build_operation_panel(self, name: str, op: Any) -> Any:
        """Build a complete panel for an operation (cached for stability)."""
        import panel as pn

        op_type = self._get_operation_type(op)

        # Header with name and type badge
        header_html = pn.pane.HTML(
            f"<h4 style='margin-bottom: 5px;'>{name}</h4>"
            f"<span style='background:#e0e0e0;padding:2px 6px;"
            f"border-radius:3px;font-size:0.8em;'>{op_type}</span>"
        )

        # Help accordion (collapsed by default)
        docstring = op.__class__.__doc__ or "*No documentation available*"
        help_card = pn.Card(
            pn.pane.Markdown(
                docstring,
                sizing_mode="stretch_width",
                styles={
                    "max-height": "150px",
                    "overflow-y": "auto",
                    "font-size": "0.85em",
                    "word-wrap": "break-word",
                    "white-space": "pre-wrap",
                },
            ),
            header="Help",
            collapsed=True,
            sizing_mode="stretch_width",
            styles={"background": "#f8f8f8"},
        )

        # Parameter widgets
        param_widgets = self._build_param_widgets(op, is_measurement=False)

        components = [header_html, help_card]
        if param_widgets.objects:
            components.append(param_widgets)
        else:
            components.append(pn.pane.Markdown("*No parameters*"))

        return pn.Column(*components, sizing_mode="stretch_width")

    def _build_measurement_panel(self, name: str, meas: Any) -> Any:
        """Build a complete panel for a measurement (cached for stability)."""
        import panel as pn

        header_html = pn.pane.HTML(
            f"<h4 style='margin-bottom: 5px;'>{name}</h4>"
            f"<span style='background:#d4edda;padding:2px 6px;"
            f"border-radius:3px;font-size:0.8em;'>Measure</span>"
        )

        docstring = meas.__class__.__doc__ or "*No documentation available*"
        help_card = pn.Card(
            pn.pane.Markdown(
                docstring,
                sizing_mode="stretch_width",
                styles={
                    "max-height": "150px",
                    "overflow-y": "auto",
                    "font-size": "0.85em",
                    "word-wrap": "break-word",
                    "white-space": "pre-wrap",
                },
            ),
            header="Help",
            collapsed=True,
            sizing_mode="stretch_width",
            styles={"background": "#f8f8f8"},
        )

        param_widgets = self._build_param_widgets(meas, is_measurement=True)

        components = [header_html, help_card]
        if param_widgets.objects:
            components.append(param_widgets)
        else:
            components.append(pn.pane.Markdown("*No parameters*"))

        return pn.Column(*components, sizing_mode="stretch_width")

    def _build_param_widgets(self, operation, is_measurement: bool = False) -> Any:
        """Build parameter editor widgets for an operation or measurement."""
        import panel as pn
        from ._operation_registry import get_registry
        from .components._param_editor import ParamEditor

        registry = get_registry()
        info = registry.get(operation.__class__.__name__)

        widgets = []
        if info:
            for param_name, param_info in info.parameters.items():
                current_value = getattr(operation, param_name, param_info.default)

                # Create callback that re-instantiates the operation
                def make_callback(op, pname, is_meas):
                    def callback(value):
                        self._update_operation_param(op, pname, value, is_meas)

                    return callback

                editor = ParamEditor(
                    param_info=param_info,
                    initial_value=current_value,
                    manager=self._manager,
                    on_change=make_callback(operation, param_name, is_measurement),
                    nesting_depth=0,
                )
                widgets.append(editor.panel())

        return pn.Column(*widgets, sizing_mode="stretch_width")

    def _update_operation_param(
        self, operation: Any, param_name: str, new_value: Any, is_measurement: bool
    ):
        """Update an operation parameter by re-instantiating.

        This handles operations that may compute derived state in __init__.
        Falls back to setattr if re-instantiation fails.
        """
        from ._operation_registry import get_registry

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
        op_list = self._measurements if is_measurement else self._operations
        op_index = -1
        for i, (name, op) in enumerate(op_list):
            if op is operation:
                op_index = i
                break

        if op_index < 0:
            # Fallback to setattr if operation not found
            setattr(operation, param_name, new_value)
            return

        registry = get_registry()
        info = registry.get(operation.__class__.__name__)

        if info is None:
            setattr(operation, param_name, new_value)
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
            name = op_list[op_index][0]
            op_list[op_index] = (name, new_operation)

            # Update widget cache - remove old, create new on next access
            cache = self._meas_widget_cache if is_measurement else self._op_widget_cache
            if old_op_id in cache:
                old_widget = cache.pop(old_op_id)
                old_widget.visible = False

            # Refresh detail panel to show updated operation
            if is_measurement:
                self._update_meas_detail_panel()
            else:
                self._update_detail_panel()

        except Exception:
            # Fallback to setattr if re-instantiation fails
            setattr(operation, param_name, new_value)

    def _add_operation(self, op_name: str):
        """Add operation by name."""
        from ._operation_registry import get_registry

        registry = get_registry()
        info = registry.get(op_name)
        if info:
            try:
                op = info.cls()
                unique_name = self._make_unique_name(op_name)
                self._operations.append((unique_name, op))

                # Select the newly added operation
                self._selected_index = len(self._operations) - 1
                self._refresh_display()

                self._toast.success(f"Added {op_name}")
            except Exception as e:
                self._toast.error(f"Failed to add {op_name}: {str(e)}")

    def _add_measurement(self, meas_name: str):
        """Add measurement by name."""
        from ._operation_registry import get_registry

        registry = get_registry()
        info = registry.get(meas_name)
        if info:
            try:
                meas = info.cls()
                unique_name = self._make_unique_name(
                    meas_name, is_measurement=True
                )
                self._measurements.append((unique_name, meas))

                self._selected_meas_index = len(self._measurements) - 1
                self._refresh_meas_display()

                self._toast.success(f"Added {meas_name}")
            except Exception as e:
                self._toast.error(f"Failed to add {meas_name}: {str(e)}")

    def _move_selected(self, direction: int):
        """Move selected operation up (-1) or down (+1)."""
        if self._selected_index < 0:
            return

        new_index = self._selected_index + direction
        if 0 <= new_index < len(self._operations):
            # Swap operations in data
            self._operations[self._selected_index], self._operations[new_index] = (
                self._operations[new_index],
                self._operations[self._selected_index],
            )
            # Update selection to follow the moved item
            self._selected_index = new_index
            self._refresh_display()

    def _move_selected_meas(self, direction: int):
        """Move selected measurement up (-1) or down (+1)."""
        if self._selected_meas_index < 0:
            return

        new_index = self._selected_meas_index + direction
        if 0 <= new_index < len(self._measurements):
            self._measurements[self._selected_meas_index], self._measurements[
                new_index
            ] = (
                self._measurements[new_index],
                self._measurements[self._selected_meas_index],
            )
            self._selected_meas_index = new_index
            self._refresh_meas_display()

    def _delete_selected(self):
        """Delete currently selected operation."""
        if self._selected_index < 0 or self._selected_index >= len(self._operations):
            return

        name, op = self._operations.pop(self._selected_index)

        # Clean up cached widget
        op_id = id(op)
        if op_id in self._op_widget_cache:
            self._op_widget_cache[op_id].visible = False

        # Adjust selection
        if self._selected_index >= len(self._operations):
            self._selected_index = len(self._operations) - 1

        self._refresh_display()
        self._toast.info(f"Removed {name}")

    def _delete_selected_meas(self):
        """Delete currently selected measurement."""
        if (
            self._selected_meas_index < 0
            or self._selected_meas_index >= len(self._measurements)
        ):
            return

        name, removed_meas = self._measurements.pop(self._selected_meas_index)

        meas_id = id(removed_meas)
        if meas_id in self._meas_widget_cache:
            self._meas_widget_cache[meas_id].visible = False

        if self._selected_meas_index >= len(self._measurements):
            self._selected_meas_index = len(self._measurements) - 1

        self._refresh_meas_display()
        self._toast.info(f"Removed {name}")

    def _save_pipeline(self, name: str):
        """Save pipeline with toast feedback."""
        if not name or not self._manager:
            if not name:
                self._toast.warning("Please enter a pipeline name")
            if not self._manager:
                self._toast.error("No InstanceManager configured")
            return

        try:
            pipeline = self.get_pipeline()
            self._manager.save_pipeline(pipeline, name.strip(), overwrite=True)

            # Refresh load dropdown options
            if self._load_select_widget is not None:
                self._load_select_widget.options = [""] + self._manager.list_pipelines()

            self._toast.success(f"Pipeline '{name}' saved successfully")
        except Exception as e:
            self._toast.error(f"Failed to save pipeline: {str(e)}")

    def _load_pipeline(self, event):
        """Load pipeline with toast feedback."""
        if event.new and self._manager:
            try:
                pipeline = self._manager.load_pipeline(event.new)

                # Clear existing widget caches
                for widget in self._op_widget_cache.values():
                    widget.visible = False
                self._op_widget_cache.clear()

                for widget in self._meas_widget_cache.values():
                    widget.visible = False
                self._meas_widget_cache.clear()

                # Load operations using public accessor if available
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

                # Auto-select first items
                self._selected_index = 0 if self._operations else -1
                self._selected_meas_index = 0 if self._measurements else -1

                self._refresh_display()
                self._refresh_meas_display()

                self._toast.success(f"Loaded pipeline '{event.new}'")

                # Reset dropdown to allow reloading same pipeline
                if self._load_select_widget is not None:
                    self._load_select_widget.value = ""

            except Exception as e:
                self._toast.error(f"Failed to load pipeline: {str(e)}")

    def _make_unique_name(self, base: str, is_measurement: bool = False) -> str:
        """Generate unique operation/measurement name."""
        existing = (
            {name for name, _ in self._measurements}
            if is_measurement
            else {name for name, _ in self._operations}
        )
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

    def get_pipeline(self):
        """Get current pipeline.

        Returns:
            ImagePipeline with current operations and measurements
        """
        from phenotypic import ImagePipeline

        pipeline = ImagePipeline({name: op for name, op in self._operations})
        if self._measurements:
            pipeline.set_meas({name: meas for name, meas in self._measurements})
        return pipeline
