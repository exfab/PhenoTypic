"""Tests for PointPickerWidget and _PointPickerPanel logic."""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# PointPickerWidget public API
# ---------------------------------------------------------------------------


class TestPointPickerWidget:
    """Tests for the public PointPickerWidget interface."""

    def test_init_stores_max_points(self):
        from phenotypic.sdk_.napari_ import PointPickerWidget

        w = PointPickerWidget(max_points=5)
        assert w._max_points == 5

    def test_init_default_max_points_is_none(self):
        from phenotypic.sdk_.napari_ import PointPickerWidget

        w = PointPickerWidget()
        assert w._max_points is None

    def test_run_raises_import_error_without_napari(self):
        from phenotypic.sdk_.napari_ import PointPickerWidget

        w = PointPickerWidget()
        with patch(
            "phenotypic._core._image_parts.accessor_abstracts._image_accessor_base._HAS_NAPARI",
            False,
        ):
            with pytest.raises(ImportError, match="napari is required"):
                w.run(MagicMock())


# ---------------------------------------------------------------------------
# _PointPickerPanel logic (tested without real Qt)
# ---------------------------------------------------------------------------


def _make_mock_panel(
    *,
    max_points: int | None = None,
    initial_data: np.ndarray | None = None,
) -> MagicMock:
    """Build a mock that mimics a ``_PointPickerPanel`` for logic testing.

    Instead of instantiating the real class (which needs Qt), we create a
    ``MagicMock`` and wire up the same attributes and bound methods so we can
    exercise ``_on_data_changed``, ``_delete_selected``, ``_clear_all``, and
    ``_confirm`` directly.
    """
    from phenotypic.sdk_.napari_._point_picker_widget import _PointPickerPanelLogic

    panel = MagicMock()

    # --- attributes identical to a real panel ---
    panel._max_points = max_points
    panel._updating = False

    # points layer mock: .data is a plain attribute we can read/write
    points_layer = MagicMock()
    points_layer.data = (
        initial_data.copy() if initial_data is not None else np.empty((0, 2))
    )
    panel._points_layer = points_layer

    # list widget mock
    list_widget = MagicMock()
    list_widget.currentRow.return_value = -1  # nothing selected by default
    panel._list_widget = list_widget

    # viewer mock
    viewer = MagicMock()
    panel._viewer = viewer

    # confirmed_points default
    panel.confirmed_points = np.empty((0, 2))

    # Bind real methods from the logic class to the mock instance
    panel._on_data_changed = lambda event=None: _PointPickerPanelLogic._on_data_changed(
        panel, event
    )
    panel._delete_selected = lambda: _PointPickerPanelLogic._delete_selected(panel)
    panel._clear_all = lambda: _PointPickerPanelLogic._clear_all(panel)
    panel._confirm = lambda: _PointPickerPanelLogic._confirm(panel)

    return panel


class TestPointPickerPanelDefaults:
    """Test default state of the panel."""

    def test_confirmed_points_default_shape(self):
        panel = _make_mock_panel()
        assert panel.confirmed_points.shape == (0, 2)


class TestOnDataChanged:
    """Tests for ``_PointPickerPanel._on_data_changed``."""

    def test_rebuilds_list_from_data(self):
        data = np.array([[10.0, 20.0], [30.0, 40.0]])
        panel = _make_mock_panel(initial_data=data)

        panel._on_data_changed()

        panel._list_widget.clear.assert_called_once()
        assert panel._list_widget.addItem.call_count == 2
        panel._list_widget.addItem.assert_any_call("(10.0, 20.0)")
        panel._list_widget.addItem.assert_any_call("(30.0, 40.0)")

    def test_skips_when_updating_flag_set(self):
        panel = _make_mock_panel(initial_data=np.array([[1.0, 2.0]]))
        panel._updating = True

        panel._on_data_changed()

        panel._list_widget.clear.assert_not_called()

    def test_trims_to_max_points_fifo(self):
        """When data exceeds max_points, only the *newest* points are kept."""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        panel = _make_mock_panel(max_points=2, initial_data=data)

        panel._on_data_changed()

        trimmed = panel._points_layer.data
        assert trimmed.shape == (2, 2)
        np.testing.assert_array_equal(trimmed, np.array([[5.0, 6.0], [7.0, 8.0]]))

    def test_no_trim_when_within_max_points(self):
        data = np.array([[1.0, 2.0]])
        panel = _make_mock_panel(max_points=3, initial_data=data)

        panel._on_data_changed()

        np.testing.assert_array_equal(panel._points_layer.data, data)

    def test_no_trim_when_max_points_is_none(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        panel = _make_mock_panel(max_points=None, initial_data=data)

        panel._on_data_changed()

        assert panel._points_layer.data.shape[0] == 3


class TestDeleteSelected:
    """Tests for ``_PointPickerPanel._delete_selected``."""

    def test_removes_selected_row(self):
        data = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
        panel = _make_mock_panel(initial_data=data)
        panel._list_widget.currentRow.return_value = 1  # select middle row

        panel._delete_selected()

        remaining = panel._points_layer.data
        assert remaining.shape == (2, 2)
        np.testing.assert_array_equal(
            remaining, np.array([[10.0, 20.0], [50.0, 60.0]])
        )

    def test_no_op_when_nothing_selected(self):
        data = np.array([[10.0, 20.0]])
        panel = _make_mock_panel(initial_data=data)
        panel._list_widget.currentRow.return_value = -1

        panel._delete_selected()

        # data unchanged
        np.testing.assert_array_equal(panel._points_layer.data, data)

    def test_delete_triggers_list_rebuild(self):
        data = np.array([[10.0, 20.0], [30.0, 40.0]])
        panel = _make_mock_panel(initial_data=data)
        panel._list_widget.currentRow.return_value = 0

        panel._delete_selected()

        # _on_data_changed is called at the end, which calls list_widget.clear
        panel._list_widget.clear.assert_called()


class TestClearAll:
    """Tests for ``_PointPickerPanel._clear_all``."""

    def test_sets_empty_data(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        panel = _make_mock_panel(initial_data=data)

        panel._clear_all()

        assert panel._points_layer.data.shape == (0, 2)

    def test_triggers_list_rebuild(self):
        panel = _make_mock_panel(initial_data=np.array([[1.0, 2.0]]))

        panel._clear_all()

        panel._list_widget.clear.assert_called()
        # No items should be added since data is empty
        panel._list_widget.addItem.assert_not_called()


class TestConfirm:
    """Tests for ``_PointPickerPanel._confirm``."""

    def test_stores_copy_of_points(self):
        data = np.array([[10.0, 20.0], [30.0, 40.0]])
        panel = _make_mock_panel(initial_data=data)

        panel._confirm()

        np.testing.assert_array_equal(panel.confirmed_points, data)
        # Must be a copy, not the same object
        assert panel.confirmed_points is not panel._points_layer.data

    def test_closes_viewer(self):
        panel = _make_mock_panel()

        panel._confirm()

        panel._viewer.close.assert_called_once()

    def test_empty_confirm_returns_empty(self):
        panel = _make_mock_panel()  # default empty data

        panel._confirm()

        assert panel.confirmed_points.shape == (0, 2)


class TestRealPanelConstruction:
    """Build the actual Qt dock widget (regression for the __bases__ bug).

    The MagicMock-based tests never instantiate a real QWidget, so they could
    not catch ``TypeError: __bases__ assignment: 'QWidget' deallocator differs
    from 'object'`` raised by the old ``__new__`` trick under PyQt6. Requires a
    live Qt binding (qt-test group, offscreen platform).
    """

    def test_factory_builds_qwidget_and_confirm_writes_points(self, qtbot):
        from qtpy.QtWidgets import QWidget

        from phenotypic.sdk_.napari_._point_picker_widget import (
            _make_point_picker_panel,
        )

        points_layer = MagicMock()
        points_layer.data = np.array([[1.0, 2.0], [3.0, 4.0]])
        viewer = MagicMock()

        panel = _make_point_picker_panel(viewer, points_layer, max_points=None)
        qtbot.addWidget(panel)

        assert isinstance(panel, QWidget)

        panel._confirm()

        np.testing.assert_array_equal(panel.confirmed_points, points_layer.data)
        viewer.close.assert_called_once()
