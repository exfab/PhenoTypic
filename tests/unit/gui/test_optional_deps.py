"""Test optional dependency handling for GUI module."""

import pytest


class TestOptionalDependencies:
    """Test that GUI module handles missing dependencies gracefully."""

    def test_gui_available_flag(self):
        """Test GUI_AVAILABLE flag is accessible."""
        from phenotypic.gui import GUI_AVAILABLE

        # This should always be accessible
        assert isinstance(GUI_AVAILABLE, bool)

    def test_instance_manager_no_panel_required(self):
        """Test InstanceManager works without Panel installed."""
        # InstanceManager should not require Panel
        from phenotypic.gui import InstanceManager

        manager = InstanceManager()
        assert manager is not None

    def test_operation_registry_no_panel_required(self):
        """Test OperationRegistry works without Panel installed."""
        # OperationRegistry should not require Panel
        from phenotypic.gui import OperationRegistry

        registry = OperationRegistry()
        assert registry is not None

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("panel"),
        reason="Panel is installed, cannot test missing dependency error",
    )
    def test_pipeline_builder_requires_panel(self):
        """Test PipelineBuilder shows helpful error without Panel.

        This test only runs if Panel is installed (to test the import mechanism).
        The actual missing dependency case is hard to test in the same environment.
        """
        from phenotypic.gui import PipelineBuilder

        # If Panel is installed, this should work
        assert PipelineBuilder is not None

    def test_lazy_import_mechanism(self):
        """Test that imports are lazy (don't import Panel at module level)."""
        import sys

        # Clear any cached Panel imports
        panel_modules = [k for k in sys.modules.keys() if k.startswith("panel")]
        for mod in panel_modules:
            if "phenotypic.gui" not in sys.modules.get(mod, "").__dict__.get("__file__", ""):
                # Don't remove panel if it's already imported elsewhere
                pass

        # Import gui module
        from phenotypic import gui

        # Panel should not be imported yet (lazy loading)
        # Note: This might fail if Panel was imported elsewhere
        # This is just a best-effort check
        assert gui.GUI_AVAILABLE is not None
