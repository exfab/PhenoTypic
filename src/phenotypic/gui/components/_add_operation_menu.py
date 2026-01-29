"""Add operation menu component for PhenoTypic GUI.

Provides a categorized dropdown menu for selecting operations to add
to a pipeline.
"""

from __future__ import annotations

from typing import Callable, Optional, Type


class AddOperationMenu:
    """Categorized operation selector menu.

    Provides a dropdown menu organized by operation category
    (Enhancer, Detector, etc.) for adding operations to a pipeline.
    """

    def __init__(
        self,
        on_select: Callable[[str], None],
        filter_base: Optional[Type] = None,
        **params,
    ):
        """Initialize AddOperationMenu.

        Args:
            on_select: Callback when operation is selected (receives operation name)
            filter_base: Filter to subclasses of this type (None = show all)
            **params: Additional parameters
        """
        self._on_select = on_select
        self._filter_base = filter_base

    def panel(self):
        """Build the add operation menu.

        Returns:
            Panel Row with Select widget and Add button
        """
        import panel as pn
        from phenotypic.gui._operation_registry import get_registry

        registry = get_registry()

        # Build grouped options by category
        groups = {}
        for category in registry.get_categories():
            ops = registry.get_by_category(category)
            # Apply filter if specified
            if self._filter_base:
                ops = [o for o in ops if issubclass(o.cls, self._filter_base)]
            if ops:
                groups[category] = [o.name for o in ops]

        select = pn.widgets.Select(
            name="Add Operation",
            groups=groups,
            value=None,
            width=300,
        )

        add_btn = pn.widgets.Button(
            name="Add",
            button_type="primary",
            width=80,
        )

        def on_add(event):
            if select.value:
                self._on_select(select.value)
                select.value = None  # Reset after adding

        add_btn.on_click(on_add)

        return pn.Row(select, add_btn, sizing_mode="fixed")
