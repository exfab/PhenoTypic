"""Tests for PipelineExplorer widget behavior (requires Panel)."""

import importlib.util

import pytest

PANEL_AVAILABLE = importlib.util.find_spec("panel") is not None

if PANEL_AVAILABLE:
    import panel as pn

pytestmark = pytest.mark.skipif(
    not PANEL_AVAILABLE, reason="Panel not installed (optional dependency)"
)


@pytest.fixture(autouse=True)
def panel_extension():
    """Initialize Panel extension for tests."""
    if PANEL_AVAILABLE:
        pn.extension()


class _Event:
    def __init__(self, new):
        self.new = new


class TestPipelineExplorerWidget:
    """Tests for PipelineExplorer widget interactions."""

    def test_param_widget_updates_node_params(self):
        """Changing a param widget updates node params in the editor."""
        from phenotypic.enhance import GaussianBlur
        from phenotypic.gui.explorer import PipelineExplorer, PipelineGraph

        graph = PipelineGraph()
        node_id = graph.add_operation(GaussianBlur, sigma=1.0)
        output_id = graph.add_output()
        graph.connect(node_id, output_id)

        explorer = PipelineExplorer(graph=graph)
        explorer._editor.selected_node_id = node_id
        explorer._on_node_selected(_Event(node_id))

        sigma_widget = next(
            w for w in explorer._param_panel.objects
            if getattr(w, "name", None) == "sigma"
        )
        sigma_widget.value = 2.5

        node_data = explorer._editor.get_selected_node_data()
        assert node_data["params"]["sigma"] == 2.5

    def test_output_checkbox_binding(self):
        """Output checkboxes are bound to save_* params."""
        from phenotypic.gui.explorer import PipelineExplorer

        explorer = PipelineExplorer()

        explorer._save_overlay_checkbox.value = False
        assert explorer.save_overlay is False

        explorer.save_overlay = True
        assert explorer._save_overlay_checkbox.value is True

    def test_summary_updates_on_graph_changes(self):
        """Summary text updates when graph nodes/edges change."""
        from phenotypic.enhance import GaussianBlur
        from phenotypic.gui.explorer import PipelineExplorer

        explorer = PipelineExplorer()
        initial_summary = explorer._summary_text.object

        node_id = explorer._editor.add_operation_node(GaussianBlur, sigma=1.0)
        output_id = explorer._editor.add_output_node()
        explorer._editor.connect_nodes(node_id, output_id)

        updated_summary = explorer._summary_text.object
        assert updated_summary != initial_summary
        assert "**Paths:** 1" in updated_summary


class TestPipelineNodeEditorSweeps:
    """Tests for PipelineNodeEditor sweep replacement behavior."""

    def test_update_node_sweep_replaces_by_default(self):
        """update_node_sweep(replace=True) replaces existing sweeps."""
        from phenotypic.enhance import GaussianBlur
        from phenotypic.gui.explorer import PipelineNodeEditor, SweepSpec

        editor = PipelineNodeEditor()
        node_id = editor.add_operation_node(GaussianBlur, sigma=1.0)

        sweep_one = SweepSpec("sigma", [1.0, 2.0])
        editor.update_node_sweep(node_id, sweep_one, replace=True)
        assert editor._sweep_data[node_id] == [sweep_one]

        sweep_two = SweepSpec("sigma", [3.0])
        editor.update_node_sweep(node_id, sweep_two, replace=True)
        assert editor._sweep_data[node_id] == [sweep_two]
