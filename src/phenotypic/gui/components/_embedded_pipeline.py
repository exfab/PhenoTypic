"""Embedded pipeline editor component for PhenoTypic GUI.

Provides a compact inline pipeline editor for operation parameters
that accept ImagePipeline instances.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple


class EmbeddedPipelineEditor:
    """Compact inline pipeline editor for embedding in operation parameters.

    Used when an operation parameter accepts ImagePipeline (e.g., FilamentousFungiDetector).
    Shows a simplified version of the full PipelineBuilder.

    Features:
    - Nesting depth tracking (operations within this editor inherit depth + 1)
    - Compact layout for nested contexts
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
        if pipeline:
            for name, op in pipeline._ops.items():
                self._operations.append((name, op))

    def panel(self):
        """Build the embedded editor.

        Returns:
            Panel widget for embedded pipeline editing
        """
        import panel as pn

        # Operations list (simplified cards)
        self._ops_column = pn.Column(sizing_mode="stretch_width")
        self._rebuild_cards()

        # Add operation selector
        from ._add_operation_menu import AddOperationMenu

        add_select = AddOperationMenu(on_select=self._add_operation)

        # Load from saved (optional)
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

        return pn.Column(
            pn.pane.Markdown("**Embedded Pipeline**"),
            self._ops_column,
            add_select.panel(),
            load_section,
            css_classes=["embedded-pipeline-editor"],
            sizing_mode="stretch_width",
        )

    def _rebuild_cards(self):
        """Rebuild operation cards with nesting depth tracking."""
        self._ops_column.clear()
        from ._operation_card import OperationCard

        for i, (name, op) in enumerate(self._operations):
            card = OperationCard(
                operation=op,
                index=i,
                show_controls=True,
                on_move=self._move_operation,
                on_delete=self._delete_operation,
                manager=self._manager,
                collapsed=True,  # Keep collapsed by default in embedded mode
                nesting_depth=self._nesting_depth,  # Pass nesting depth
            )
            self._ops_column.append(card.panel())
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
            self._rebuild_cards()

    def _move_operation(self, index: int, direction: int):
        """Move operation up/down."""
        new_index = index + direction
        if 0 <= new_index < len(self._operations):
            self._operations[index], self._operations[new_index] = (
                self._operations[new_index],
                self._operations[index],
            )
            self._rebuild_cards()

    def _delete_operation(self, index: int):
        """Delete operation."""
        if 0 <= index < len(self._operations):
            self._operations.pop(index)
            self._rebuild_cards()

    def _load_pipeline(self, event):
        """Load a saved pipeline."""
        if event.new and self._manager:
            pipeline = self._manager.load_pipeline(event.new)
            self._operations = [(n, op) for n, op in pipeline._ops.items()]
            self._rebuild_cards()

    def _make_unique_name(self, base: str) -> str:
        """Generate unique operation name."""
        existing = {name for name, _ in self._operations}
        if base not in existing:
            return base
        i = 1
        while f"{base}_{i}" in existing:
            i += 1
        return f"{base}_{i}"

    def _notify_change(self):
        """Notify parent of pipeline change."""
        if self._on_change:
            pipeline = self._get_pipeline()
            self._on_change(pipeline)

    def _get_pipeline(self):
        """Build ImagePipeline from current operations."""
        from phenotypic import ImagePipeline

        return ImagePipeline({name: op for name, op in self._operations})
