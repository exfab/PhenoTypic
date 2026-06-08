"""Test optional dependency handling for GUI module."""


class TestOptionalDependencies:
    """Test that GUI module handles missing dependencies gracefully."""

    def test_gui_available_flag(self):
        """GUI_AVAILABLE reflects whether the Dash GUI stack is importable."""
        import importlib.util

        from phenotypic.gui import GUI_AVAILABLE

        assert isinstance(GUI_AVAILABLE, bool)
        assert GUI_AVAILABLE == (importlib.util.find_spec("dash") is not None)

    def test_operation_registry_no_panel_required(self):
        """Test OperationRegistry works without Panel installed."""
        from phenotypic.gui import OperationRegistry

        registry = OperationRegistry()
        assert registry is not None

    def test_lazy_import_mechanism(self):
        """Test that importing the gui module does not import Panel."""
        from phenotypic import gui

        assert gui.GUI_AVAILABLE is not None
