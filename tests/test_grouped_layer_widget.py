"""Tests for the GroupedLayerWidget (headless, viewer=None)."""

from __future__ import annotations

import pytest

pytest_qt = pytest.importorskip("pytestqt")


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _make_entries(pipeline, components, stem="plate_001"):
    """Build a list of entry dicts for a single pipeline."""
    return [
        {
            "pipeline": pipeline,
            "component": comp,
            "image_stem": stem,
        }
        for comp in components
    ]


class _FakeLayer:
    """Minimal stand-in for a napari layer."""

    def __init__(self, name: str, visible: bool = True):
        self.name = name
        self.visible = visible


class _FakeLayerList(list):
    """Minimal list-like container that supports iteration."""


class _FakeViewer:
    """Minimal viewer stand-in with a layer list."""

    def __init__(self, layers=None):
        self.layers = _FakeLayerList(layers or [])


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------


@pytest.fixture()
def widget(qtbot):
    """Create a headless GroupedLayerWidget (viewer=None)."""
    from phenotypic.gui.sweep._grouped_layer_widget import (
        GroupedLayerWidget,
    )

    w = GroupedLayerWidget(viewer=None)
    qtbot.addWidget(w)
    return w


@pytest.fixture()
def viewer_widget(qtbot):
    """Create a GroupedLayerWidget backed by a fake viewer."""
    from phenotypic.gui.sweep._grouped_layer_widget import (
        GroupedLayerWidget,
    )

    layers = [
        _FakeLayer("main/Pipeline_0/rgb/plate_001"),
        _FakeLayer("main/Pipeline_0/detect_mat/plate_001"),
        _FakeLayer("main/Pipeline_0/objmap/plate_001"),
        _FakeLayer("split/Pipeline_1/rgb/plate_001"),
        _FakeLayer("split/Pipeline_1/detect_mat/plate_001"),
        _FakeLayer("split/Pipeline_1/objmap/plate_001"),
    ]
    viewer = _FakeViewer(layers)
    w = GroupedLayerWidget(viewer=viewer)
    qtbot.addWidget(w)
    return w, viewer


# -------------------------------------------------------------------
# Structure tests
# -------------------------------------------------------------------


class TestCheckboxStructure:
    """Verify checkbox creation, ordering, and cleanup."""

    def test_set_layers_creates_checkboxes(self, widget):
        entries = _make_entries(
            "Pipeline_0", ["rgb", "detect_mat", "objmap"],
        )
        widget.set_layers(entries)

        assert set(widget._checkboxes.keys()) == {
            "rgb", "detect_mat", "objmap",
        }

    def test_component_order(self, widget):
        entries = _make_entries(
            "Pipeline_0", ["objmap", "rgb", "detect_mat", "gray"],
        )
        widget.set_layers(entries)

        ordered_names = [
            widget._cb_layout.itemAt(i).widget().text()
            for i in range(widget._cb_layout.count())
        ]
        assert ordered_names == ["rgb", "gray", "detect_mat", "objmap"]

    def test_set_layers_replaces_previous(self, widget):
        entries_a = _make_entries("Pipeline_0", ["rgb"])
        entries_b = _make_entries("Pipeline_1", ["detect_mat", "objmap"])

        widget.set_layers(entries_a)
        assert set(widget._checkboxes.keys()) == {"rgb"}

        widget.set_layers(entries_b)
        assert set(widget._checkboxes.keys()) == {"detect_mat", "objmap"}

    def test_add_layers_merges_components(self, widget):
        entries_a = _make_entries("Pipeline_0", ["rgb"])
        entries_b = _make_entries("Pipeline_1", ["rgb", "objmap"])

        widget.set_layers(entries_a)
        widget.add_layers(entries_b)

        assert set(widget._checkboxes.keys()) == {"rgb", "objmap"}

    def test_add_layers_no_rebuild_when_same_components(self, widget):
        entries = _make_entries("Pipeline_0", ["rgb", "detect_mat"])
        widget.set_layers(entries)

        cb_rgb_id = id(widget._checkboxes["rgb"])

        # Same component types from a different pipeline — no rebuild.
        entries_b = _make_entries("Pipeline_1", ["rgb", "detect_mat"])
        widget.add_layers(entries_b)

        assert id(widget._checkboxes["rgb"]) == cb_rgb_id

    def test_clear_removes_checkboxes(self, widget):
        widget.set_layers(
            _make_entries("Pipeline_0", ["rgb", "objmap"]),
        )
        widget.clear()

        assert widget._checkboxes == {}
        assert widget._active_components == set()

    def test_clear_preserves_visibility_state(self, widget):
        widget.set_layers(
            _make_entries("Pipeline_0", ["rgb", "objmap"]),
        )
        widget._checkboxes["rgb"].setChecked(False)

        widget.clear()

        assert widget._visibility["rgb"] is False

    def test_duplicate_entries_no_extra_checkboxes(self, widget):
        entries = _make_entries("Pipeline_0", ["rgb", "objmap"])
        widget.set_layers(entries)
        widget.add_layers(entries)  # same entries again

        assert len(widget._checkboxes) == 2

    def test_unknown_component_sorted_last(self, widget):
        entries = (
            _make_entries("Pipeline_0", ["custom_layer"])
            + _make_entries("Pipeline_0", ["rgb"])
        )
        widget.set_layers(entries)

        ordered_names = [
            widget._cb_layout.itemAt(i).widget().text()
            for i in range(widget._cb_layout.count())
        ]
        assert ordered_names == ["rgb", "custom_layer"]


# -------------------------------------------------------------------
# Checkbox defaults
# -------------------------------------------------------------------


class TestCheckboxDefaults:
    """Verify all checkboxes are checked by default."""

    def test_all_checked_by_default(self, widget):
        widget.set_layers(
            _make_entries("Pipeline_0", ["rgb", "objmap"]),
        )
        for cb in widget._checkboxes.values():
            assert cb.isChecked()


# -------------------------------------------------------------------
# Visibility tests (with fake viewer)
# -------------------------------------------------------------------


class TestVisibilityToggle:
    """Verify that toggling a checkbox affects viewer layers."""

    def test_uncheck_hides_all_matching_layers(self, viewer_widget):
        w, viewer = viewer_widget
        entries = (
            _make_entries("Pipeline_0", ["rgb", "detect_mat", "objmap"])
            + _make_entries("Pipeline_1", ["rgb", "detect_mat", "objmap"])
        )
        w.set_layers(entries)

        w._checkboxes["rgb"].setChecked(False)

        for layer in viewer.layers:
            if "/rgb/" in layer.name:
                assert layer.visible is False
            else:
                assert layer.visible is True

    def test_recheck_shows_all_matching_layers(self, viewer_widget):
        w, viewer = viewer_widget
        entries = (
            _make_entries("Pipeline_0", ["rgb", "detect_mat", "objmap"])
            + _make_entries("Pipeline_1", ["rgb", "detect_mat", "objmap"])
        )
        w.set_layers(entries)

        w._checkboxes["rgb"].setChecked(False)
        w._checkboxes["rgb"].setChecked(True)

        for layer in viewer.layers:
            assert layer.visible is True

    def test_visibility_persists_across_clear_and_rebuild(
        self, viewer_widget,
    ):
        w, viewer = viewer_widget
        entries = (
            _make_entries("Pipeline_0", ["rgb", "detect_mat", "objmap"])
            + _make_entries("Pipeline_1", ["rgb", "detect_mat", "objmap"])
        )
        w.set_layers(entries)
        w._checkboxes["rgb"].setChecked(False)

        # Clear and re-set with the same entries.
        w.set_layers(entries)

        assert w._checkboxes["rgb"].isChecked() is False
        for layer in viewer.layers:
            if "/rgb/" in layer.name:
                assert layer.visible is False

    def test_set_layers_applies_visibility(self, viewer_widget):
        w, viewer = viewer_widget
        entries = (
            _make_entries("Pipeline_0", ["rgb", "detect_mat", "objmap"])
            + _make_entries("Pipeline_1", ["rgb", "detect_mat", "objmap"])
        )
        w.set_layers(entries)

        # All layers should be visible by default.
        for layer in viewer.layers:
            assert layer.visible is True


# -------------------------------------------------------------------
# Signal backward compatibility
# -------------------------------------------------------------------


class TestSignalBackwardCompat:
    """Verify that layer_clicked signal still exists."""

    def test_signal_exists(self, widget):
        assert hasattr(widget, "layer_clicked")
