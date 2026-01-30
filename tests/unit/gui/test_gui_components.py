"""Unit tests for GUI components (requires Panel).

These tests are skipped if Panel is not installed.
"""

import importlib.util
import pytest

# Check if Panel is available
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


class TestPipelineBuilder:
    """Test PipelineBuilder GUI component."""

    def test_import(self):
        """Test that PipelineBuilder can be imported."""
        from phenotypic.gui import PipelineBuilder

        assert PipelineBuilder is not None

    def test_create_empty_builder(self):
        """Test creating empty PipelineBuilder."""
        from phenotypic.gui import PipelineBuilder

        builder = PipelineBuilder()
        assert builder is not None
        assert len(builder._operations) == 0

    def test_create_builder_with_manager(self, tmp_path):
        """Test creating PipelineBuilder with InstanceManager."""
        from phenotypic.gui import PipelineBuilder, InstanceManager

        manager = InstanceManager(workspace=tmp_path)
        builder = PipelineBuilder(manager=manager)
        assert builder._manager is manager

    def test_create_builder_with_image(self):
        """Test creating PipelineBuilder with preview image."""
        from phenotypic.gui import PipelineBuilder
        from phenotypic.data import load_synth_yeast_plate

        image = load_synth_yeast_plate()
        builder = PipelineBuilder(image=image)
        assert builder._image is image

    def test_create_builder_with_pipeline(self):
        """Test creating PipelineBuilder with existing pipeline."""
        from phenotypic.gui import PipelineBuilder
        from phenotypic import ImagePipeline
        from phenotypic.enhance import GaussianBlur

        pipeline = ImagePipeline([GaussianBlur(sigma=2.0)])
        builder = PipelineBuilder(pipeline=pipeline)
        assert len(builder._operations) == 1

    def test_get_pipeline(self):
        """Test getting current pipeline from builder."""
        from phenotypic.gui import PipelineBuilder
        from phenotypic import ImagePipeline
        from phenotypic.enhance import GaussianBlur

        pipeline = ImagePipeline([GaussianBlur(sigma=2.0)])
        builder = PipelineBuilder(pipeline=pipeline)

        current = builder.get_pipeline()
        assert isinstance(current, ImagePipeline)
        assert len(current._ops) == 1

    def test_panel_method(self):
        """Test that panel() returns a Panel widget."""
        from phenotypic.gui import PipelineBuilder

        builder = PipelineBuilder()
        panel = builder.panel()
        assert panel is not None
        assert isinstance(panel, pn.viewable.Viewable)

    def test_add_operation(self):
        """Test adding operation to pipeline."""
        from phenotypic.gui import PipelineBuilder

        builder = PipelineBuilder()
        assert len(builder._operations) == 0

        # Add operation
        builder._add_operation("GaussianBlur")
        assert len(builder._operations) == 1
        assert builder._operations[0][0] == "GaussianBlur"

    def test_delete_operation(self):
        """Test deleting operation from pipeline."""
        from phenotypic.gui import PipelineBuilder
        from phenotypic import ImagePipeline
        from phenotypic.enhance import GaussianBlur

        pipeline = ImagePipeline([GaussianBlur(sigma=2.0)])
        builder = PipelineBuilder(pipeline=pipeline)
        assert len(builder._operations) == 1

        # Select and delete operation using new API
        builder._selected_index = 0
        builder._delete_selected()
        assert len(builder._operations) == 0

    def test_move_operation(self):
        """Test moving operation up/down."""
        from phenotypic.gui import PipelineBuilder
        from phenotypic import ImagePipeline
        from phenotypic.enhance import GaussianBlur, CLAHE

        pipeline = ImagePipeline([GaussianBlur(), CLAHE()])
        builder = PipelineBuilder(pipeline=pipeline)

        # Initial order
        assert builder._operations[0][1].__class__.__name__ == "GaussianBlur"
        assert builder._operations[1][1].__class__.__name__ == "CLAHE"

        # Move first down using new API (direction +1 = down)
        builder._selected_index = 0
        builder._move_selected(1)
        assert builder._operations[0][1].__class__.__name__ == "CLAHE"
        assert builder._operations[1][1].__class__.__name__ == "GaussianBlur"

    def test_save_load_pipeline(self, tmp_path):
        """Test saving and loading pipeline through builder."""
        from phenotypic.gui import PipelineBuilder, InstanceManager
        from phenotypic import ImagePipeline
        from phenotypic.enhance import GaussianBlur

        manager = InstanceManager(workspace=tmp_path)
        pipeline = ImagePipeline([GaussianBlur(sigma=2.0)])
        builder = PipelineBuilder(pipeline=pipeline, manager=manager)

        # Save
        builder._save_pipeline("test_pipeline")
        assert "test_pipeline" in manager.list_pipelines()

        # Create new builder and load
        builder2 = PipelineBuilder(manager=manager)
        assert len(builder2._operations) == 0

        # Simulate load event
        class Event:
            new = "test_pipeline"

        builder2._load_pipeline(Event())
        assert len(builder2._operations) == 1


class TestParamEditor:
    """Test ParamEditor component."""

    def test_import(self):
        """Test that ParamEditor can be imported."""
        from phenotypic.gui.components import ParamEditor

        assert ParamEditor is not None

    def test_create_basic_param_editor(self):
        """Test creating ParamEditor for basic type."""
        from phenotypic.gui.components._param_editor import ParamEditor
        from phenotypic.gui._operation_registry import ParamInfo

        param_info = ParamInfo(
            name="sigma",
            type_hint=float,
            default=1.0,
            has_default=True,
            is_operation=False,
            is_pipeline=False,
            is_optional=False,
        )

        editor = ParamEditor(param_info=param_info)
        assert editor.value == 1.0
        assert editor.param_info.name == "sigma"

    def test_param_editor_panel_basic(self):
        """Test that ParamEditor.panel() returns widget for basic type."""
        from phenotypic.gui.components._param_editor import ParamEditor
        from phenotypic.gui._operation_registry import ParamInfo

        param_info = ParamInfo(
            name="sigma",
            type_hint=float,
            default=1.0,
            has_default=True,
            is_operation=False,
            is_pipeline=False,
            is_optional=False,
        )

        editor = ParamEditor(param_info=param_info)
        widget = editor.panel()
        assert widget is not None


class TestOperationCard:
    """Test OperationCard component."""

    def test_import(self):
        """Test that OperationCard can be imported."""
        from phenotypic.gui.components import OperationCard

        assert OperationCard is not None

    def test_create_operation_card(self):
        """Test creating OperationCard."""
        from phenotypic.gui.components._operation_card import OperationCard
        from phenotypic.enhance import GaussianBlur

        op = GaussianBlur(sigma=2.0)
        card = OperationCard(operation=op)
        assert card.operation is op

    def test_operation_card_panel(self):
        """Test that OperationCard.panel() returns widget."""
        from phenotypic.gui.components._operation_card import OperationCard
        from phenotypic.enhance import GaussianBlur

        op = GaussianBlur(sigma=2.0)
        card = OperationCard(operation=op)
        widget = card.panel()
        assert widget is not None


class TestAddOperationMenu:
    """Test AddOperationMenu component."""

    def test_import(self):
        """Test that AddOperationMenu can be imported."""
        from phenotypic.gui.components import AddOperationMenu

        assert AddOperationMenu is not None

    def test_create_menu(self):
        """Test creating AddOperationMenu."""
        from phenotypic.gui.components._add_operation_menu import AddOperationMenu

        def on_select(name):
            pass

        menu = AddOperationMenu(on_select=on_select)
        assert menu._on_select is on_select

    def test_menu_panel(self):
        """Test that menu.panel() returns widget."""
        from phenotypic.gui.components._add_operation_menu import AddOperationMenu

        def on_select(name):
            pass

        menu = AddOperationMenu(on_select=on_select)
        widget = menu.panel()
        assert widget is not None


class TestPreviewPanel:
    """Test PreviewPanel component."""

    def test_import(self):
        """Test that PreviewPanel can be imported."""
        from phenotypic.gui.components import PreviewPanel

        assert PreviewPanel is not None

    def test_create_preview_panel(self):
        """Test creating PreviewPanel."""
        from phenotypic.gui.components._preview_panel import PreviewPanel

        panel = PreviewPanel()
        assert panel is not None

    def test_create_preview_panel_with_image(self):
        """Test creating PreviewPanel with image."""
        from phenotypic.gui.components._preview_panel import PreviewPanel
        from phenotypic.data import load_synth_yeast_plate

        image = load_synth_yeast_plate()
        panel = PreviewPanel(image=image)
        assert panel._image is image

    def test_preview_panel_widget(self):
        """Test that panel() returns widget."""
        from phenotypic.gui.components._preview_panel import PreviewPanel

        panel = PreviewPanel()
        widget = panel.panel()
        assert widget is not None


class TestToastNotification:
    """Test ToastNotification helper."""

    def test_import(self):
        """Test that ToastNotification can be imported."""
        from phenotypic.gui._toast import ToastNotification

        assert ToastNotification is not None

    def test_create_toast(self):
        """Test creating ToastNotification."""
        from phenotypic.gui._toast import ToastNotification

        toast = ToastNotification()
        assert toast is not None

    # Note: Testing actual toast display requires a running Panel server
    # and is better suited for integration tests


class TestPipelineSummaryCard:
    """Test PipelineSummaryCard component."""

    def test_import(self):
        """Test that PipelineSummaryCard can be imported."""
        from phenotypic.gui._pipeline_summary_card import PipelineSummaryCard

        assert PipelineSummaryCard is not None

    def test_create_summary_card(self):
        """Test creating PipelineSummaryCard."""
        from phenotypic.gui._pipeline_summary_card import PipelineSummaryCard
        from phenotypic import ImagePipeline
        from phenotypic.enhance import GaussianBlur

        pipeline = ImagePipeline([GaussianBlur(sigma=2.0)])
        card = PipelineSummaryCard(pipeline=pipeline, name="Test Pipeline")
        assert card._pipeline is pipeline
        assert card._name == "Test Pipeline"

    def test_summary_card_panel(self):
        """Test that panel() returns widget."""
        from phenotypic.gui._pipeline_summary_card import PipelineSummaryCard
        from phenotypic import ImagePipeline
        from phenotypic.enhance import GaussianBlur

        pipeline = ImagePipeline([GaussianBlur(sigma=2.0)])
        card = PipelineSummaryCard(pipeline=pipeline)
        widget = card.panel()
        assert widget is not None
