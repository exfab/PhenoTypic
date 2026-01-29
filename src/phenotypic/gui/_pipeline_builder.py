"""Main PipelineBuilder GUI for PhenoTypic.

Top-level interactive interface for building and testing ImagePipelines
using Panel components.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple


class PipelineBuilder:
    """Interactive pipeline construction widget using Panel.

    Composes reusable components:
    - OperationCard (with ParamEditor)
    - AddOperationMenu
    - PreviewPanel

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

        # Operations list: [(name, operation), ...]
        self._operations: List[Tuple[str, Any]] = []
        if pipeline:
            for name, op in pipeline._ops.items():
                self._operations.append((name, op))

        # Toast notifications
        from ._toast import ToastNotification

        self._toast = ToastNotification(app=None)

    def panel(self):
        """Build the main layout.

        Returns:
            Panel Column widget with operations panel above preview panel
        """
        import panel as pn
        from .components import AddOperationMenu, PreviewPanel, OperationCard

        # === TOP: Operations Panel ===
        self._ops_column = pn.Column(
            sizing_mode="stretch_width",
            scroll=True,
            max_height=250,
        )
        self._rebuild_cards()

        add_menu = AddOperationMenu(on_select=self._add_operation)

        # Save/Load controls
        name_input = pn.widgets.TextInput(placeholder="Pipeline name", width=150)
        save_btn = pn.widgets.Button(name="Save", button_type="success", width=80)
        save_btn.on_click(lambda e: self._save_pipeline(name_input.value))

        load_options = (
            [""] + (self._manager.list_pipelines() if self._manager else [])
        )
        load_select = pn.widgets.Select(
            name="Load", options=load_options, value="", width=200
        )
        load_select.param.watch(self._load_pipeline, "value")

        controls = pn.Row(name_input, save_btn, load_select, sizing_mode="fixed")

        operations_panel = pn.Column(
            "## Pipeline Operations",
            self._ops_column,
            add_menu.panel(),
            pn.layout.Divider(),
            "### Save/Load",
            controls,
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
            sizing_mode="stretch_both",
            min_height=600,
        )

        return pn.Column(
            operations_panel,
            preview_panel,
            sizing_mode="stretch_both",
        )

    def _rebuild_cards(self):
        """Rebuild all operation cards."""
        # Only rebuild if panel has been created
        if not hasattr(self, "_ops_column"):
            return

        from .components import OperationCard

        # Build all cards first, then assign at once to avoid reference issues
        new_cards = []
        for i, (name, op) in enumerate(self._operations):
            card = OperationCard(
                operation=op,
                index=i,
                show_controls=True,
                on_move=self._move_operation,
                on_delete=self._delete_operation,
                manager=self._manager,
            )
            new_cards.append(card.panel())

        # Replace all children at once (avoids Bokeh reference issues)
        self._ops_column.objects = new_cards

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
                self._rebuild_cards()
                self._toast.success(f"Added {op_name}")
            except Exception as e:
                self._toast.error(f"Failed to add {op_name}: {str(e)}")

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
            name, _ = self._operations.pop(index)
            self._rebuild_cards()
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
            self._toast.success(f"Pipeline '{name}' saved successfully")
        except Exception as e:
            self._toast.error(f"Failed to save pipeline: {str(e)}")

    def _load_pipeline(self, event):
        """Load pipeline with toast feedback."""
        if event.new and self._manager:
            try:
                pipeline = self._manager.load_pipeline(event.new)
                self._operations = [(n, op) for n, op in pipeline._ops.items()]
                self._rebuild_cards()
                self._toast.success(f"Loaded pipeline '{event.new}'")
            except Exception as e:
                self._toast.error(f"Failed to load pipeline: {str(e)}")

    def _make_unique_name(self, base: str) -> str:
        """Generate unique operation name."""
        existing = {name for name, _ in self._operations}
        if base not in existing:
            return base
        i = 1
        while f"{base}_{i}" in existing:
            i += 1
        return f"{base}_{i}"

    def get_pipeline(self):
        """Get current pipeline.

        Returns:
            ImagePipeline with current operations
        """
        from phenotypic import ImagePipeline

        return ImagePipeline({name: op for name, op in self._operations})
