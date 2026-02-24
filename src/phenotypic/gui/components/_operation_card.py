"""Operation card component for PhenoTypic GUI.

Displays a single operation with its parameters in a collapsible card,
including an expandable help section showing the operation's docstring.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional


class OperationCard:
    """Card widget for a single pipeline operation.

    Reusable component that can be used in:
    - PipelineBuilder (with reorder/delete controls)
    - Nested operation editors (without controls)
    - SweepExecutor config (with range inputs instead of single values)

    Features:
    - Expandable "Help" section showing operation docstring
    - Nesting depth tracking for embedded operations
    - State designed for future undo/redo support
    """

    def __init__(
        self,
        operation,  # ImageOperation
        index: int = 0,
        show_controls: bool = True,
        on_move: Optional[Callable[[int, int], None]] = None,
        on_delete: Optional[Callable[[int], None]] = None,
        on_param_change: Optional[Callable[[str, Any], None]] = None,
        manager=None,  # InstanceManager
        nesting_depth: int = 0,
        **params,
    ):
        """Initialize OperationCard.

        Args:
            operation: ImageOperation instance to display
            index: Position in pipeline (for move operations)
            show_controls: Show reorder/delete buttons
            on_move: Callback for move operation (index, direction)
            on_delete: Callback for delete operation (index)
            on_param_change: Callback for parameter changes (name, value)
            manager: InstanceManager for nested pipeline loading
            nesting_depth: Current nesting level
            **params: Additional parameters
        """
        self.operation = operation
        self.index = index
        self._show_controls = show_controls
        self._on_move = on_move
        self._on_delete = on_delete
        self._on_param_change = on_param_change
        self._manager = manager
        self._nesting_depth = nesting_depth

        # State
        self.collapsed = False  # Expanded by default to show parameters
        self.help_expanded = False

        # Get parameter info from registry
        self._param_info = self._get_param_info()

        # Create ParamEditor for each parameter (with nesting depth)
        self._param_editors: Dict[str, Any] = {}
        for name, info in self._param_info.items():
            from ._param_editor import ParamEditor

            current_value = getattr(operation, name, info.default)
            self._param_editors[name] = ParamEditor(
                param_info=info,
                initial_value=current_value,
                manager=manager,
                on_change=lambda v, n=name: self._sync_param(n, v),
                nesting_depth=nesting_depth,
            )

    def _get_param_info(self) -> Dict[str, Any]:
        """Get parameter info for this operation's class."""
        from phenotypic.gui._operation_registry import get_registry

        registry = get_registry()
        info = registry.get(self.operation.__class__.__name__)
        if info:
            return info.parameters
        # Fallback: extract directly
        return registry._extract_parameters(self.operation.__class__)

    def _sync_param(self, name: str, value: Any):
        """Sync parameter value back to operation."""
        setattr(self.operation, name, value)
        if self._on_param_change:
            self._on_param_change(name, value)

    def panel(self):
        """Build the card widget with expandable help section.

        Returns:
            Panel Card widget
        """
        import panel as pn

        op_name = self.operation.__class__.__name__
        op_type = self._get_operation_type()

        # Header
        header_items = [
            pn.pane.HTML(f"<b>{op_name}</b>"),
            pn.pane.HTML(
                f"<span style='background:#e0e0e0;padding:2px 6px;"
                f"border-radius:3px;font-size:0.8em;'>{op_type}</span>"
            ),
            pn.Spacer(),
        ]

        if self._show_controls:
            header_items.extend(
                [
                    pn.widgets.Button(
                        name="▲",
                        width=30,
                        on_click=lambda e: (
                            self._on_move(self.index, -1) if self._on_move else None
                        ),
                    ),
                    pn.widgets.Button(
                        name="▼",
                        width=30,
                        on_click=lambda e: (
                            self._on_move(self.index, 1) if self._on_move else None
                        ),
                    ),
                    pn.widgets.Button(
                        name="×",
                        width=30,
                        button_type="danger",
                        on_click=lambda e: (
                            self._on_delete(self.index) if self._on_delete else None
                        ),
                    ),
                ]
            )

        header = pn.Row(*header_items, sizing_mode="stretch_width")

        # Compact help section (placed directly below operation name)
        docstring = self.operation.__class__.__doc__ or "*No documentation available*"
        help_accordion = pn.Card(
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
            collapsed=True,  # Collapsed by default for compact display
            sizing_mode="stretch_width",
            styles={"background": "#f8f8f8"},
        )

        # Parameters panel
        param_widgets = [editor.panel() for editor in self._param_editors.values()]
        params_panel = (
            pn.Column(*param_widgets)
            if param_widgets
            else pn.pane.Markdown("*No parameters*")
        )

        return pn.Card(
            pn.Column(
                help_accordion,  # Help directly below operation name
                params_panel,  # Parameters below help
            ),
            header=header,
            collapsed=self.collapsed,
            sizing_mode="stretch_width",
        )

    def _get_operation_type(self) -> str:
        """Get human-readable operation type."""
        from phenotypic.abc_ import (
            ImageEnhancer,
            ObjectDetector,
            ObjectRefiner,
            ImageCorrector,
            MeasureFeatures,
            GridOperation,
        )

        op = self.operation
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
