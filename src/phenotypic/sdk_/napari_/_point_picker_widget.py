"""Blocking napari-based point picker widget."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import napari  # noqa: F401


class PointPickerWidget:
    """Interactive point picker that opens a napari viewer for coordinate selection.

    Args:
        max_points: Maximum number of points allowed. None means unlimited.
    """

    def __init__(self, max_points: int | None = None) -> None:
        self._max_points = max_points

    def run(self, image) -> np.ndarray:
        """Open a napari viewer for interactive point selection and block until closed.

        Args:
            image: A PhenoTypic ``Image`` instance whose layers will be displayed.

        Returns:
            An ``(N, 2)`` array of confirmed ``(y, x)`` coordinates. Returns an
            empty ``(0, 2)`` array if the viewer is closed without confirming.

        Raises:
            ImportError: If napari is not installed.
        """
        from phenotypic._core._image_parts.accessor_abstracts._image_accessor_base import (
            _HAS_NAPARI,
        )

        if not _HAS_NAPARI:
            raise ImportError(
                "napari is required for interactive visualization. "
                "Install with: pip install phenotypic[napari]"
            )
        import napari

        from ._layers import add_image_layer

        viewer = napari.Viewer(title="Point Picker")

        # Non-stretched contrast (integer → full dtype range, normalized float →
        # (0, 1)) so reference layers render at true brightness.
        if not image.rgb.isempty():
            add_image_layer(viewer, image.rgb[:], name="rgb")
        add_image_layer(viewer, image.gray[:], name="gray")
        add_image_layer(viewer, image.detect_mat[:], name="detect_mat")

        points_layer = viewer.add_points(
            np.empty((0, 2)),
            name="picks",
            size=12,
            face_color="red",
            edge_color="white",
        )
        points_layer.mode = "add"

        panel = _make_point_picker_panel(
            viewer, points_layer, max_points=self._max_points
        )
        viewer.window.add_dock_widget(panel, name="Point Picker", area="right")

        napari.run()

        return panel.confirmed_points


class _PointPickerPanelLogic:
    """Qt-independent point-list/confirmation logic for the point picker dock.

    Kept separate from the ``QWidget`` subclass (built lazily in
    :func:`_make_point_picker_panel`) so the behaviour is importable and
    unit-testable without a Qt event loop. The concrete dock widget mixes this
    into ``QWidget`` and supplies ``_viewer``, ``_points_layer``,
    ``_max_points``, ``_updating``, ``_list_widget``, and ``confirmed_points``.
    """

    _viewer: Any
    _points_layer: Any
    _list_widget: Any
    _max_points: int | None
    _updating: bool
    confirmed_points: np.ndarray

    def _on_data_changed(self, event=None) -> None:  # noqa: ARG002
        """Rebuild list widget from the current points layer data."""
        if self._updating:
            return

        data = self._points_layer.data

        if self._max_points is not None and len(data) > self._max_points:
            self._updating = True
            self._points_layer.data = data[-self._max_points :]
            self._updating = False
            data = self._points_layer.data

        self._list_widget.clear()
        for row in data:
            self._list_widget.addItem(f"({row[0]:.1f}, {row[1]:.1f})")

    def _delete_selected(self) -> None:
        """Remove the currently selected point from the layer."""
        row = self._list_widget.currentRow()
        if row < 0:
            return
        self._updating = True
        self._points_layer.data = np.delete(self._points_layer.data, row, axis=0)
        self._updating = False
        self._on_data_changed()

    def _clear_all(self) -> None:
        """Remove all points from the layer."""
        self._updating = True
        self._points_layer.data = np.empty((0, 2))
        self._updating = False
        self._on_data_changed()

    def _confirm(self) -> None:
        """Store the current points and close the viewer."""
        self.confirmed_points = self._points_layer.data.copy()
        self._viewer.close()


def _make_point_picker_panel(viewer, points_layer, *, max_points: int | None = None):
    """Build the napari dock widget with point-list and confirmation controls.

    The ``QWidget`` subclass is defined here (lazily, on first call) rather than
    at module import so ``qtpy`` stays an optional dependency. Defining it with
    ``QWidget`` as a real base at class-creation time also avoids mutating
    ``__bases__`` after the fact, which CPython forbids when the base's
    deallocator differs (PyQt6's sip-wrapped ``QWidget`` vs ``object``).

    Args:
        viewer: The napari viewer instance.
        points_layer: The napari Points layer to manage.
        max_points: Maximum number of points allowed. None means unlimited.

    Returns:
        A ``QWidget`` dock panel wired to the :class:`_PointPickerPanelLogic`
        callbacks.
    """
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import (
        QHBoxLayout,
        QLabel,
        QListWidget,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )

    class _PointPickerPanel(QWidget, _PointPickerPanelLogic):
        def __init__(self) -> None:
            QWidget.__init__(self)

            self._viewer = viewer
            self._points_layer = points_layer
            self._max_points = max_points
            self._updating = False
            self.confirmed_points = np.empty((0, 2))

            layout = QVBoxLayout(self)
            layout.setContentsMargins(4, 4, 4, 4)

            header = QLabel("Selected Points (y, x)")
            header.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(header)

            self._list_widget = QListWidget()
            layout.addWidget(self._list_widget)

            btn_row = QHBoxLayout()
            self._delete_btn = QPushButton("Delete Selected")
            self._clear_btn = QPushButton("Clear All")
            btn_row.addWidget(self._delete_btn)
            btn_row.addWidget(self._clear_btn)
            layout.addLayout(btn_row)

            self._confirm_btn = QPushButton("Confirm")
            layout.addWidget(self._confirm_btn)

            self._delete_btn.clicked.connect(self._delete_selected)
            self._clear_btn.clicked.connect(self._clear_all)
            self._confirm_btn.clicked.connect(self._confirm)

            self._points_layer.events.data.connect(self._on_data_changed)

    return _PointPickerPanel()
