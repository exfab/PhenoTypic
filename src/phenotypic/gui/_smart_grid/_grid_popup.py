"""Overlay checkbox in napari's grid popup."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def patch_grid_popup(
    viewer, get_enabled, set_enabled, get_labels_enabled, set_labels_enabled,
) -> None:
    """Add overlay and label checkboxes to the grid button's right-click popup."""
    try:
        viewer_buttons = viewer.window._qt_viewer.viewerButtons
    except AttributeError:
        return

    gvb = viewer_buttons.gridViewButton
    _orig_open = viewer_buttons._open_grid_popup

    def _open_with_overlay():
        _orig_open()  # Creates and shows popup (parent=viewer_buttons)

        # Find popup among viewer_buttons' children
        from napari._qt.dialogs.qt_modal import QtPopup

        popup = None
        for child in viewer_buttons.children():
            if isinstance(child, QtPopup) and child.isVisible():
                popup = child
                break

        if popup is None:
            return

        from qtpy.QtWidgets import QCheckBox, QLabel

        layout = popup.frame.layout()
        row = layout.rowCount()
        layout.addWidget(QLabel("Overlay:"), row, 0)
        cb = QCheckBox("Labels / Points / Shapes")
        cb.setChecked(get_enabled())
        cb.toggled.connect(set_enabled)
        layout.addWidget(cb, row, 1)

        row = layout.rowCount()
        layout.addWidget(QLabel("Labels:"), row, 0)
        cb_labels = QCheckBox("Layer Names")
        cb_labels.setChecked(get_labels_enabled())
        cb_labels.toggled.connect(set_labels_enabled)
        layout.addWidget(cb_labels, row, 1)

    # Disconnect original slot, connect replacement
    gvb.customContextMenuRequested.disconnect(viewer_buttons._open_grid_popup)
    gvb.customContextMenuRequested.connect(lambda pos: _open_with_overlay())
