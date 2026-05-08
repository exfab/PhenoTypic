"""Flat component-type toggle widget for the sweep viewer."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, get_args

from phenotypic.gui._config import ChannelName
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

_COMPONENT_ORDER = get_args(ChannelName)


def _component_sort_key(component: str) -> tuple[int, str]:
    """Sort key: known components by _COMPONENT_ORDER, unknowns last."""
    try:
        return (_COMPONENT_ORDER.index(component), component)
    except ValueError:
        return (len(_COMPONENT_ORDER), component)


class GroupedLayerWidget(QWidget):
    """Flat checkbox widget that toggles visibility by component type.

    Each component type (rgb, gray, detect_mat, objmap) gets a single
    checkbox.  Toggling it sets ``.visible`` for every napari layer
    whose name contains ``/{component}/`` — covering both
    ``main/Pipeline_0/rgb/plate_001`` and
    ``split/Pipeline_1/rgb/plate_001`` in one click.

    Args:
        viewer: napari ``Viewer`` instance, or ``None`` for headless
            testing (visibility toggles become no-ops).
        parent: Optional parent widget.
    """

    layer_clicked = Signal(str)

    def __init__(
        self,
        viewer=None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._viewer = viewer

        self._active_components: Set[str] = set()
        self._visibility: Dict[str, bool] = {}
        self._checkboxes: Dict[str, QCheckBox] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._header = QLabel("Visible Layers:")
        layout.addWidget(self._header)

        # Container for checkbox rows — rebuilt on set_layers / add_layers.
        self._cb_container = QWidget()
        self._cb_layout = QVBoxLayout(self._cb_container)
        self._cb_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._cb_container)

        layout.addStretch()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_layers(self, entries: List[dict]) -> None:
        """Clear checkboxes and rebuild from *entries*.

        Args:
            entries: List of dicts with ``pipeline``, ``component``,
                and ``image_stem`` keys.
        """
        self.clear()
        self._rebuild_checkboxes(entries)
        self._apply_visibility_to_viewer()

    def add_layers(self, entries: List[dict]) -> None:
        """Merge new component types into existing checkboxes.

        Args:
            entries: Same format as :meth:`set_layers`.
        """
        self._rebuild_checkboxes(entries)
        self._apply_visibility_to_viewer()

    def clear(self) -> None:
        """Remove all checkboxes but preserve visibility state."""
        self._active_components.clear()
        self._remove_all_checkboxes()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _rebuild_checkboxes(self, entries: List[dict]) -> None:
        """Discover component types from *entries* and create checkboxes."""
        new_components: Set[str] = set()
        for entry in entries:
            comp = entry["component"]
            new_components.add(comp)

        added = new_components - self._active_components
        self._active_components |= new_components

        # Assign default visibility for newly-seen components.
        for comp in added:
            self._visibility.setdefault(comp, True)

        # Only rebuild UI when there are genuinely new components.
        if added:
            self._sync_checkbox_ui()

    def _sync_checkbox_ui(self) -> None:
        """Recreate checkbox widgets to match ``_active_components``."""
        self._remove_all_checkboxes()

        ordered = sorted(self._active_components, key=_component_sort_key)
        for comp in ordered:
            cb = QCheckBox(comp)
            cb.setChecked(self._visibility.get(comp, True))
            cb.toggled.connect(
                lambda checked, c=comp: self._on_checkbox_toggled(c, checked),
            )
            self._cb_layout.addWidget(cb)
            self._checkboxes[comp] = cb

    def _remove_all_checkboxes(self) -> None:
        """Remove all checkbox widgets from the container layout."""
        for cb in self._checkboxes.values():
            self._cb_layout.removeWidget(cb)
            cb.deleteLater()
        self._checkboxes.clear()

    def _on_checkbox_toggled(
        self, component: str, checked: bool,
    ) -> None:
        """Handle a component checkbox toggle.

        Updates ``_visibility`` and sets the ``.visible`` flag on every
        napari layer whose name contains ``/{component}/``.
        """
        self._visibility[component] = checked
        self._set_component_visibility(component, checked)

    def _set_component_visibility(
        self, component: str, visible: bool,
    ) -> None:
        """Set visibility on all viewer layers matching *component*."""
        if self._viewer is None:
            return
        token = f"/{component}/"
        for layer in self._viewer.layers:
            if token in layer.name:
                layer.visible = visible

    def _apply_visibility_to_viewer(self) -> None:
        """Apply current visibility state for all active components."""
        for comp in self._active_components:
            self._set_component_visibility(
                comp, self._visibility.get(comp, True),
            )
