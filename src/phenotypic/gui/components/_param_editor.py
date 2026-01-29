"""Core parameter editor component for PhenoTypic GUI.

Renders appropriate widgets for any parameter type including basic types,
operations, and pipelines with nesting depth tracking.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, get_args, get_origin
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
        origin = get_origin(hint)
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
                            # Show nested editor with incremented depth
                            from ._operation_card import OperationCard

                            nested = OperationCard(
                                operation=op_instance,
                                show_controls=False,
                                nesting_depth=self._nesting_depth + 1,
                            )
                            content.append(nested.panel())

                op_select.param.watch(on_op_select, "value")
                content.append(op_select)

            elif mode == "Create Inline":
                # Show mini pipeline builder (only if within depth limit)
                from ._embedded_pipeline import EmbeddedPipelineEditor

                inline_editor = EmbeddedPipelineEditor(
                    manager=self._manager,
                    on_change=lambda p: setattr(self, "value", p),
                    nesting_depth=self._nesting_depth + 1,
                )
                content.append(inline_editor.panel())

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
                            # Show as non-editable summary card
                            from phenotypic.gui._pipeline_summary_card import (
                                PipelineSummaryCard,
                            )

                            summary = PipelineSummaryCard(pipeline=loaded, name=e.new)
                            content.append(summary.panel())

                    pipe_select.param.watch(on_pipe_select, "value")
                    content.append(pipe_select)
                else:
                    content.append(pn.pane.Markdown("*No InstanceManager configured*"))

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
        from phenotypic.abc_ import ImageOperation
        from phenotypic.gui._operation_registry import get_registry

        registry = get_registry()
        hint = self.param_info.type_hint

        # Extract allowed base classes from Union
        allowed_bases = []
        origin = get_origin(hint)
        if origin is Union:
            for arg in get_args(hint):
                if isinstance(arg, type) and arg is not type(None):
                    allowed_bases.append(arg)
        elif isinstance(hint, type):
            allowed_bases.append(hint)

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
