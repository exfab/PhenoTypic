"""Grouped layer list widget for the sweep viewer."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

_COMPONENT_ORDER = ("rgb", "gray", "detect_mat", "objmap")


def _component_sort_key(component: str) -> tuple[int, str]:
    """Sort key: known components by _COMPONENT_ORDER, unknowns last."""
    try:
        return (_COMPONENT_ORDER.index(component), component)
    except ValueError:
        return (len(_COMPONENT_ORDER), component)


class GroupedLayerWidget(QWidget):
    """Tree widget that groups loaded layers by pipeline.

    Provides visibility toggling per-pipeline (parent checkbox
    propagates to children) and click-to-select (sets the active
    layer in napari).

    Args:
        viewer: napari ``Viewer`` instance, or ``None`` for headless
            testing (visibility/selection become no-ops).
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

        self._layer_items: Dict[str, QTreeWidgetItem] = {}
        self._pipeline_items: Dict[str, QTreeWidgetItem] = {}
        self._updating: bool = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.itemChanged.connect(self._on_item_changed)
        self._tree.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self._tree)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_layers(self, entries: List[dict]) -> None:
        """Clear the tree and rebuild from *entries*.

        Args:
            entries: List of dicts with ``pipeline``, ``component``,
                and ``image_stem`` keys.
        """
        self.clear()
        self._add_entries(entries)

    def add_layers(self, entries: List[dict]) -> None:
        """Append *entries* to the existing tree (reuse pipeline nodes).

        Args:
            entries: Same format as :meth:`set_layers`.
        """
        self._add_entries(entries)

    def clear(self) -> None:
        """Remove all items from the tree."""
        self._tree.clear()
        self._layer_items.clear()
        self._pipeline_items.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _layer_name_for_entry(entry: dict) -> str:
        """Return ``pipeline/component/stem`` for *entry*."""
        return (
            f"{entry['pipeline']}/{entry['component']}"
            f"/{entry['image_stem']}"
        )

    def _add_entries(self, entries: List[dict]) -> None:
        """Group *entries* by pipeline and populate the tree."""
        groups: Dict[str, List[dict]] = {}
        for entry in entries:
            groups.setdefault(entry["pipeline"], []).append(entry)

        self._updating = True
        try:
            for pipe_name in sorted(groups):
                parent = self._pipeline_items.get(pipe_name)
                if parent is None:
                    parent = QTreeWidgetItem([pipe_name])
                    parent.setFlags(
                        parent.flags() | Qt.ItemIsUserCheckable,
                    )
                    parent.setCheckState(0, Qt.Checked)
                    self._tree.addTopLevelItem(parent)
                    self._pipeline_items[pipe_name] = parent

                for entry in sorted(
                    groups[pipe_name],
                    key=lambda e: _component_sort_key(e["component"]),
                ):
                    layer_name = self._layer_name_for_entry(entry)
                    if layer_name in self._layer_items:
                        continue
                    child = QTreeWidgetItem([entry["component"]])
                    child.setFlags(
                        child.flags() | Qt.ItemIsUserCheckable,
                    )
                    child.setCheckState(0, Qt.Checked)
                    child.setData(0, Qt.UserRole, layer_name)
                    parent.addChild(child)
                    self._layer_items[layer_name] = child

                parent.setExpanded(True)
        finally:
            self._updating = False

    def _on_item_changed(
        self, item: QTreeWidgetItem, column: int,
    ) -> None:
        """Handle checkbox state changes."""
        if self._updating:
            return

        self._updating = True
        try:
            parent = item.parent()
            if parent is None:
                # Top-level (pipeline) node — propagate to children.
                state = item.checkState(column)
                for i in range(item.childCount()):
                    child = item.child(i)
                    child.setCheckState(0, state)
                    name = child.data(0, Qt.UserRole)
                    if name:
                        self._set_layer_visibility(
                            name, state == Qt.Checked,
                        )
            else:
                # Child node — toggle layer visibility.
                name = item.data(0, Qt.UserRole)
                if name:
                    self._set_layer_visibility(
                        name, item.checkState(0) == Qt.Checked,
                    )
                self._update_parent_check_state(parent)
        finally:
            self._updating = False

    def _on_item_clicked(
        self, item: QTreeWidgetItem, column: int,
    ) -> None:
        """Handle click-to-select for child (component) nodes."""
        if item.parent() is None:
            # Pipeline node — no single layer to select.
            return
        name = item.data(0, Qt.UserRole)
        if name:
            self._select_layer(name)
            self.layer_clicked.emit(name)

    def _set_layer_visibility(self, name: str, visible: bool) -> None:
        """Set napari layer visibility (no-op when headless)."""
        if self._viewer is None:
            return
        try:
            self._viewer.layers[name].visible = visible
        except KeyError:
            logger.debug("Layer not found in viewer: %s", name)

    def _select_layer(self, name: str) -> None:
        """Set the active layer in napari (no-op when headless)."""
        if self._viewer is None:
            return
        try:
            self._viewer.layers.selection.active = (
                self._viewer.layers[name]
            )
        except KeyError:
            logger.debug("Layer not found for selection: %s", name)

    def _update_parent_check_state(
        self, parent: QTreeWidgetItem,
    ) -> None:
        """Set parent check state based on children."""
        checked = 0
        total = parent.childCount()
        for i in range(total):
            if parent.child(i).checkState(0) == Qt.Checked:
                checked += 1

        if checked == 0:
            parent.setCheckState(0, Qt.Unchecked)
        elif checked == total:
            parent.setCheckState(0, Qt.Checked)
        else:
            parent.setCheckState(0, Qt.PartiallyChecked)
