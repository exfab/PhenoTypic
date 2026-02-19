"""Tests for the GroupedLayerWidget (headless, viewer=None)."""

from __future__ import annotations

import pytest

pytest_qt = pytest.importorskip("pytestqt")

from qtpy.QtCore import Qt  # noqa: E402


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


# -------------------------------------------------------------------
# Structure tests
# -------------------------------------------------------------------


class TestTreeStructure:
    """Verify tree population, replacement, accumulation, and clearing."""

    def test_set_layers_single_pipeline(self, widget):
        entries = _make_entries("Pipeline_0", ["rgb", "detect_mat", "objmask"])
        widget.set_layers(entries)

        tree = widget._tree
        assert tree.topLevelItemCount() == 1

        parent = tree.topLevelItem(0)
        assert parent.text(0) == "Pipeline_0"
        # Children sorted by domain-meaningful component order
        children = [
            parent.child(i).text(0)
            for i in range(parent.childCount())
        ]
        assert children == ["rgb", "detect_mat", "objmask"]

    def test_set_layers_replaces_previous(self, widget):
        entries_a = _make_entries("Pipeline_0", ["rgb"])
        entries_b = _make_entries("Pipeline_1", ["detect_mat", "objmask"])

        widget.set_layers(entries_a)
        assert widget._tree.topLevelItemCount() == 1

        widget.set_layers(entries_b)
        assert widget._tree.topLevelItemCount() == 1
        assert widget._tree.topLevelItem(0).text(0) == "Pipeline_1"
        assert len(widget._layer_items) == 2

    def test_add_layers_accumulates(self, widget):
        entries_a = _make_entries("Pipeline_0", ["rgb"])
        entries_b = _make_entries("Pipeline_1", ["rgb", "objmask"])

        widget.set_layers(entries_a)
        widget.add_layers(entries_b)

        assert widget._tree.topLevelItemCount() == 2

    def test_clear_empties_tree(self, widget):
        widget.set_layers(
            _make_entries("Pipeline_0", ["rgb", "objmask"]),
        )
        widget.clear()

        assert widget._tree.topLevelItemCount() == 0
        assert widget._layer_items == {}
        assert widget._pipeline_items == {}

    def test_duplicate_entries_ignored(self, widget):
        entries = _make_entries("Pipeline_0", ["rgb", "objmask"])
        widget.set_layers(entries)
        widget.add_layers(entries)  # same entries again

        parent = widget._tree.topLevelItem(0)
        assert parent.childCount() == 2


# -------------------------------------------------------------------
# Checkbox tests
# -------------------------------------------------------------------


class TestCheckboxBehavior:
    """Verify parent/child checkbox propagation."""

    def test_all_checked_by_default(self, widget):
        widget.set_layers(
            _make_entries("Pipeline_0", ["rgb", "objmask"]),
        )
        parent = widget._tree.topLevelItem(0)
        assert parent.checkState(0) == Qt.Checked
        for i in range(parent.childCount()):
            assert parent.child(i).checkState(0) == Qt.Checked

    def test_uncheck_child_updates_parent_to_partial(self, widget):
        widget.set_layers(
            _make_entries("Pipeline_0", ["rgb", "objmask"]),
        )
        parent = widget._tree.topLevelItem(0)
        child = parent.child(0)

        child.setCheckState(0, Qt.Unchecked)

        assert parent.checkState(0) == Qt.PartiallyChecked

    def test_uncheck_all_children_unchecks_parent(self, widget):
        widget.set_layers(
            _make_entries("Pipeline_0", ["rgb", "objmask"]),
        )
        parent = widget._tree.topLevelItem(0)

        for i in range(parent.childCount()):
            parent.child(i).setCheckState(0, Qt.Unchecked)

        assert parent.checkState(0) == Qt.Unchecked

    def test_uncheck_parent_unchecks_all_children(self, widget):
        widget.set_layers(
            _make_entries("Pipeline_0", ["rgb", "objmask"]),
        )
        parent = widget._tree.topLevelItem(0)

        parent.setCheckState(0, Qt.Unchecked)

        for i in range(parent.childCount()):
            assert parent.child(i).checkState(0) == Qt.Unchecked


# -------------------------------------------------------------------
# Signal tests
# -------------------------------------------------------------------


class TestSignals:
    """Verify layer_clicked signal emission."""

    def test_click_component_emits_layer_clicked(self, qtbot, widget):
        entries = _make_entries("Pipeline_0", ["rgb"])
        widget.set_layers(entries)

        parent = widget._tree.topLevelItem(0)
        child = parent.child(0)

        with qtbot.waitSignal(
            widget.layer_clicked, timeout=1000,
        ) as blocker:
            widget._on_item_clicked(child, 0)

        assert blocker.args == ["Pipeline_0/rgb/plate_001"]

    def test_click_pipeline_does_not_emit(self, qtbot, widget):
        entries = _make_entries("Pipeline_0", ["rgb"])
        widget.set_layers(entries)

        parent = widget._tree.topLevelItem(0)

        with qtbot.assertNotEmitted(widget.layer_clicked):
            widget._on_item_clicked(parent, 0)


# -------------------------------------------------------------------
# Layer name test
# -------------------------------------------------------------------


class TestLayerName:
    """Verify layer name construction from entry dicts."""

    def test_layer_name_for_entry(self):
        from phenotypic.gui.sweep._grouped_layer_widget import (
            GroupedLayerWidget,
        )

        entry = {
            "pipeline": "Pipeline_0",
            "component": "rgb",
            "image_stem": "plate_001",
        }
        assert (
            GroupedLayerWidget._layer_name_for_entry(entry)
            == "Pipeline_0/rgb/plate_001"
        )
