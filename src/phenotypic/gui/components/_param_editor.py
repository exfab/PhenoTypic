"""Core parameter editor component for PhenoTypic GUI.

Renders appropriate widgets for any parameter type including basic types,
operations, and pipelines with nesting depth tracking.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, get_args, get_origin
from typing_extensions import Literal, Union

# Maximum nesting depth for editable operations/pipelines
MAX_NESTING_DEPTH = 3


class ParamEditor:
    """Widget for editing a single operation parameter.

    Handles basic types, operations, and pipelines with appropriate UI.
    This is the core reusable component for parameter editing.

    Features:
    - Type hint validation with warnings (free-form input allowed)
    - Nesting depth tracking for embedded operations/pipelines
    - Loaded pipelines shown as non-editable summary cards beyond depth limit
    """

    def __init__(
        self,
        param_info,  # ParamInfo from OperationRegistry
        initial_value: Any = None,
        manager=None,  # InstanceManager
        on_change: Optional[Callable[[Any], None]] = None,
        nesting_depth: int = 0,
    ):
        """Initialize ParamEditor.

        Args:
            param_info: ParamInfo describing this parameter
            initial_value: Initial value (uses param_info.default if None)
            manager: InstanceManager for loading/saving pipelines
            on_change: Callback when value changes
            nesting_depth: Current nesting level (for depth limiting)
        """
        import param

        self.param_info = param_info
        self._manager = manager
        self._on_change = on_change
        self._nesting_depth = nesting_depth

        # Set initial value
        self.value = initial_value if initial_value is not None else param_info.default
        self.warning = ""  # Validation warning message

    def panel(self):
        """Build the parameter editor widget.

        Returns:
            Panel widget for this parameter
        """
        if self.param_info.is_operation or self.param_info.is_pipeline:
            return self._build_operation_editor()
        else:
            return self._build_basic_editor()

    def _build_basic_editor(self):
        """Build widget for basic types with type hint validation."""
        import panel as pn

        hint = self.param_info.type_hint
        value = self.value

        # Check for Union types with multiple basic types (show type selector)
        origin = get_origin(hint)
        if origin is Union:
            args = get_args(hint)
            # Filter to basic types (exclude None and operation/pipeline types)
            basic_types = [
                a for a in args
                if a is not type(None) and not self._is_operation_type(a)
            ]
            # If multiple basic types, show type selector dropdown
            if len(basic_types) > 1:
                return self._build_union_editor(basic_types)

        # Warning display (shown when type validation fails)
        warning_pane = pn.pane.HTML(
            "",
            css_classes=["param-warning"],
            styles={"color": "orange", "font-size": "0.85em"},
        )

        def validate_and_warn(new_value):
            """Validate value against type hint, show warning if mismatch."""
            warning_msg = self._validate_type(new_value, hint)
            warning_pane.object = f"⚠️ {warning_msg}" if warning_msg else ""
            self.warning = warning_msg
            self._handle_change(new_value)

        # Handle Literal types (dropdown)
        if origin is Literal:
            options = list(get_args(hint))
            widget = pn.widgets.Select(
                name=self.param_info.name,
                options=options,
                value=value if value in options else options[0],
            )
        elif isinstance(value, bool) or hint is bool:
            widget = pn.widgets.Checkbox(name=self.param_info.name, value=bool(value))
        elif isinstance(value, int) or hint is int:
            widget = pn.widgets.IntInput(
                name=self.param_info.name, value=int(value) if value is not None else 0
            )
        elif isinstance(value, float) or hint is float:
            widget = pn.widgets.FloatInput(
                name=self.param_info.name,
                value=float(value) if value is not None else 0.0,
            )
        elif isinstance(value, str) or hint is str:
            widget = pn.widgets.TextInput(
                name=self.param_info.name, value=str(value) if value else ""
            )
        else:
            # Fallback: text representation (free-form)
            widget = pn.widgets.TextInput(
                name=self.param_info.name,
                value=str(value) if value is not None else "",
                placeholder="<complex type>",
            )

        # Bind change handler with validation
        widget.param.watch(lambda e: validate_and_warn(e.new), "value")

        # Add description if available
        items = [widget]
        if self.param_info.description:
            desc_html = pn.pane.HTML(
                f"<span style='color: #666; font-size: 0.85em; font-style: italic; margin-left: 2px;'>"
                f"{self.param_info.description}</span>",
                sizing_mode="stretch_width",
            )
            items.append(desc_html)
        items.append(warning_pane)

        return pn.Column(*items, sizing_mode="stretch_width")

    def _validate_type(self, value: Any, hint: Any) -> str:
        """Validate value against type hint. Returns warning message or empty string."""
        if hint is None or value is None:
            return ""
        try:
            origin = get_origin(hint)
            if origin is Union:
                # Check if value matches any type in Union
                for arg in get_args(hint):
                    if arg is type(None) and value is None:
                        return ""
                    if isinstance(arg, type) and isinstance(value, arg):
                        return ""
                return f"Expected one of {get_args(hint)}, got {type(value).__name__}"
            elif isinstance(hint, type):
                if not isinstance(value, hint):
                    return f"Expected {hint.__name__}, got {type(value).__name__}"
        except Exception:
            pass  # Type checking failed, allow free-form
        return ""

    def _build_operation_editor(self):
        """Build widget for operation/pipeline parameters with nesting depth control."""
        import panel as pn
        from phenotypic.gui._operation_registry import get_registry

        # Check if we've exceeded nesting depth
        at_depth_limit = self._nesting_depth >= MAX_NESTING_DEPTH

        # Mode selector: None / Select Operation / Create Inline / Load Pipeline
        mode_options = []
        if self.param_info.is_optional:
            mode_options.append("None")
        mode_options.append("Select Operation")
        if not at_depth_limit:  # Only allow inline creation within depth limit
            mode_options.append("Create Inline")
        if self.param_info.is_pipeline:
            mode_options.append("Load Pipeline")

        mode_select = pn.widgets.Select(
            name=f"{self.param_info.name} Mode",
            options=mode_options,
            value=self._infer_current_mode(),
        )

        # Content area (changes based on mode)
        content = pn.Column()

        def update_content(event):
            mode = event.new if event else mode_select.value
            content.clear()

            if mode == "None":
                self.value = None
                content.append(pn.pane.Markdown("*No operation selected*"))

            elif mode == "Select Operation":
                # Dropdown of available operations (filtered by type hint)
                available_ops = self._get_compatible_operations()
                op_select = pn.widgets.Select(
                    name="Operation",
                    options=[""] + available_ops,
                    value="",
                )

                def on_op_select(e):
                    if e.new:
                        registry = get_registry()
                        info = registry.get(e.new)
                        if info:
                            op_instance = info.cls()
                            self.value = op_instance
                            # Clear previous operation cards (keep dropdown at index 0)
                            while len(content.objects) > 1:
                                content.pop(-1)
                            # Show nested editor with incremented depth
                            from ._operation_card import OperationCard

                            nested = OperationCard(
                                operation=op_instance,
                                show_controls=False,
                                nesting_depth=self._nesting_depth + 1,
                            )
                            content.append(nested.panel())
                            self._handle_change(self.value)

                op_select.param.watch(on_op_select, "value")
                content.append(op_select)

                # If we already have an operation value, show its card immediately
                if self.value is not None:
                    from phenotypic.abc_ import ImageOperation

                    if isinstance(self.value, ImageOperation):
                        # Pre-select the current operation in dropdown
                        # Use param.update() to avoid triggering on_op_select callback
                        op_name = type(self.value).__name__
                        if op_name in available_ops:
                            op_select.param.update(value=op_name)
                        # Show nested editor
                        from ._operation_card import OperationCard

                        nested = OperationCard(
                            operation=self.value,
                            show_controls=False,
                            nesting_depth=self._nesting_depth + 1,
                        )
                        content.append(nested.panel())

            elif mode == "Create Inline":
                # Show mini pipeline builder (only if within depth limit)
                from ._embedded_pipeline import EmbeddedPipelineEditor

                inline_editor = EmbeddedPipelineEditor(
                    manager=self._manager,
                    on_change=lambda p: setattr(self, "value", p),
                    nesting_depth=self._nesting_depth + 1,
                )

                # Wrap in distinctive Card for visual distinction
                embedded_card = pn.Card(
                    inline_editor.panel(),
                    header="📦 Embedded Pipeline",
                    collapsed=False,
                    sizing_mode="stretch_width",
                    styles={
                        "border": "2px solid #4a90d9",
                        "border-radius": "8px",
                        "background": "#f0f7ff",
                    },
                )
                content.append(embedded_card)

            elif mode == "Load Pipeline":
                # Dropdown of saved pipelines
                if self._manager:
                    pipelines = self._manager.list_pipelines()
                    pipe_select = pn.widgets.Select(
                        name="Pipeline",
                        options=[""] + pipelines,
                        value="",
                    )

                    def on_pipe_select(e):
                        if e.new:
                            loaded = self._manager.load_pipeline(e.new)
                            self.value = loaded
                            # Clear previous summary cards (keep dropdown at index 0)
                            while len(content.objects) > 1:
                                content.pop(-1)
                            # Show as non-editable summary card
                            from phenotypic.gui._pipeline_summary_card import (
                                PipelineSummaryCard,
                            )

                            summary = PipelineSummaryCard(pipeline=loaded, name=e.new)
                            content.append(summary.panel())
                            self._handle_change(self.value)

                    pipe_select.param.watch(on_pipe_select, "value")
                    content.append(pipe_select)
                else:
                    content.append(pn.pane.Markdown("*No InstanceManager configured*"))

            if event is not None:
                self._handle_change(self.value)

        mode_select.param.watch(update_content, "value")
        update_content(None)  # Initial render

        return pn.Column(
            mode_select,
            content,
            css_classes=["param-editor-operation"],
        )

    def _get_compatible_operations(self) -> List[str]:
        """Get operations compatible with this parameter's type hint."""
        from phenotypic.abc_ import (
            ImageOperation,
            ImageEnhancer,
            ObjectDetector,
            ObjectRefiner,
            ImageCorrector,
        )
        from phenotypic.gui._operation_registry import get_registry

        registry = get_registry()
        hint = self.param_info.type_hint

        # Extract allowed base classes from Union (handle forward refs)
        allowed_bases = []
        origin = get_origin(hint)
        if origin is Union:
            for arg in get_args(hint):
                if arg is type(None):
                    continue
                # Handle resolved types
                if isinstance(arg, type):
                    allowed_bases.append(arg)
                # Handle string forward references
                elif isinstance(arg, str):
                    if "ImageEnhancer" in arg or "Enhancer" in arg:
                        allowed_bases.append(ImageEnhancer)
                    elif "ObjectDetector" in arg or "Detector" in arg:
                        allowed_bases.append(ObjectDetector)
                    elif "ObjectRefiner" in arg or "Refiner" in arg:
                        allowed_bases.append(ObjectRefiner)
                    elif "ImageCorrector" in arg or "Corrector" in arg:
                        allowed_bases.append(ImageCorrector)
                    elif "ImageOperation" in arg or "Operation" in arg:
                        allowed_bases.append(ImageOperation)
                # Handle ForwardRef objects
                elif hasattr(arg, "__forward_arg__"):
                    ref_name = arg.__forward_arg__
                    if "ImageEnhancer" in ref_name or "Enhancer" in ref_name:
                        allowed_bases.append(ImageEnhancer)
                    elif "ObjectDetector" in ref_name or "Detector" in ref_name:
                        allowed_bases.append(ObjectDetector)
                    elif "ObjectRefiner" in ref_name or "Refiner" in ref_name:
                        allowed_bases.append(ObjectRefiner)
                    elif "ImageCorrector" in ref_name or "Corrector" in ref_name:
                        allowed_bases.append(ImageCorrector)
                    elif "ImageOperation" in ref_name or "Operation" in ref_name:
                        allowed_bases.append(ImageOperation)
        elif isinstance(hint, type):
            allowed_bases.append(hint)
        # Handle bare string annotations
        elif isinstance(hint, str):
            if "ImageEnhancer" in hint or "Enhancer" in hint:
                allowed_bases.append(ImageEnhancer)
            elif "ObjectDetector" in hint or "Detector" in hint:
                allowed_bases.append(ObjectDetector)
            elif "ImageCorrector" in hint or "Corrector" in hint:
                allowed_bases.append(ImageCorrector)
            elif "ImageOperation" in hint or "Operation" in hint:
                allowed_bases.append(ImageOperation)

        # If no bases found but param is flagged as operation, allow all operations
        if not allowed_bases and self.param_info.is_operation:
            allowed_bases.append(ImageOperation)

        # Filter operations by compatibility
        compatible = []
        for name, info in registry.get_all().items():
            for base in allowed_bases:
                try:
                    if issubclass(info.cls, base):
                        compatible.append(name)
                        break
                except TypeError:
                    # Not a class, skip
                    pass

        return sorted(compatible)

    def _infer_current_mode(self) -> str:
        """Infer current mode from value."""
        if self.value is None:
            return "None" if self.param_info.is_optional else "Select Operation"
        from phenotypic import ImagePipeline

        if isinstance(self.value, ImagePipeline):
            return "Create Inline"  # or "Load Pipeline" if we track source
        return "Select Operation"

    def _handle_change(self, new_value):
        """Handle value change."""
        self.value = new_value
        if self._on_change:
            self._on_change(new_value)

    def _is_operation_type(self, type_cls: Any) -> bool:
        """Check if type is an operation or pipeline type."""
        from phenotypic.abc_ import ImageOperation
        from phenotypic import ImagePipeline

        if not isinstance(type_cls, type):
            return False
        try:
            return issubclass(type_cls, (ImageOperation, ImagePipeline))
        except TypeError:
            return False

    def _build_union_editor(self, basic_types: List[type]):
        """Build widget for Union types with type selector dropdown.

        Args:
            basic_types: List of basic types from the Union (excluding None)
        """
        import panel as pn

        # Build type map: display_name -> type
        type_map: Dict[str, type] = {}
        for t in basic_types:
            if hasattr(t, "__name__"):
                type_map[t.__name__] = t
            else:
                type_map[str(t)] = t

        # Check if None is allowed
        hint = self.param_info.type_hint
        args = get_args(hint)
        if type(None) in args:
            type_map["None"] = type(None)

        if not type_map:
            return self._build_basic_editor()  # Fallback

        # Infer initial type from value
        initial_type_name = self._infer_type_name(self.value, type_map)

        # Type selector dropdown
        type_select = pn.widgets.Select(
            name=f"{self.param_info.name} type",
            options=list(type_map.keys()),
            value=initial_type_name,
            width=150,
        )

        # Container for type-specific widget (stable container, dynamic content)
        inner_container = pn.Column(sizing_mode="stretch_width")

        def update_inner_widget(event=None):
            """Recreate inner widget for selected type."""
            inner_container.clear()
            selected_type = type_map.get(type_select.value)

            if selected_type is type(None):
                self.value = None
                self._handle_change(None)
                inner_container.append(pn.pane.Markdown("*Value: None*"))
                return

            # Create appropriate widget for this type
            widget = self._create_widget_for_type(selected_type, self.value)
            inner_container.append(widget)

        type_select.param.watch(update_inner_widget, "value")
        update_inner_widget()  # Initial render

        # Add description if available
        items = [pn.Row(type_select, sizing_mode="stretch_width"), inner_container]
        if self.param_info.description:
            desc_html = pn.pane.HTML(
                f"<span style='color: #666; font-size: 0.85em; font-style: italic; margin-left: 2px;'>"
                f"{self.param_info.description}</span>",
                sizing_mode="stretch_width",
            )
            items.append(desc_html)

        return pn.Column(*items, sizing_mode="stretch_width")

    def _infer_type_name(self, value: Any, type_map: Dict[str, type]) -> str:
        """Infer type name from current value."""
        if value is None and "None" in type_map:
            return "None"

        # Try exact type match
        for name, t in type_map.items():
            if t is type(None):
                continue
            if type(value) == t:
                return name

        # Default to first non-None type
        for name in type_map:
            if name != "None":
                return name
        return list(type_map.keys())[0]

    def _create_widget_for_type(self, type_cls: type, value: Any):
        """Create input widget for specific type."""
        import panel as pn

        # Handle Literal types
        origin = get_origin(type_cls)
        if origin is Literal:
            options = list(get_args(type_cls))
            widget = pn.widgets.Select(
                name=self.param_info.name,
                options=options,
                value=value if value in options else options[0],
            )
        elif type_cls is bool:
            widget = pn.widgets.Checkbox(
                name=self.param_info.name,
                value=bool(value) if value is not None else False,
            )
        elif type_cls is int:
            try:
                int_val = int(value) if value is not None else 0
            except (ValueError, TypeError):
                int_val = 0
            widget = pn.widgets.IntInput(
                name=self.param_info.name,
                value=int_val,
            )
        elif type_cls is float:
            try:
                float_val = float(value) if value is not None else 0.0
            except (ValueError, TypeError):
                float_val = 0.0
            widget = pn.widgets.FloatInput(
                name=self.param_info.name,
                value=float_val,
            )
        elif type_cls is str:
            widget = pn.widgets.TextInput(
                name=self.param_info.name,
                value=str(value) if value is not None else "",
            )
        else:
            # Fallback: text input
            type_name = type_cls.__name__ if hasattr(type_cls, "__name__") else "value"
            widget = pn.widgets.TextInput(
                name=self.param_info.name,
                value=str(value) if value is not None else "",
                placeholder=f"<{type_name}>",
            )

        # Bind change handler
        widget.param.watch(lambda e: self._handle_change(e.new), "value")
        return widget
