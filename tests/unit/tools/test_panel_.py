"""Tests for the shared Panel toolkit."""

from __future__ import annotations

import pytest

from phenotypic.tools_.panel_ import (
    PANEL_AVAILABLE,
    PANEL_IMPORT_ERROR,
    display_or_return,
    ensure_panel_extension,
    in_ipython,
    in_jupyter,
    require_panel,
)


class TestPanelAvailability:
    """Tests for PANEL_AVAILABLE and require_panel."""

    def test_panel_available_is_bool(self):
        assert isinstance(PANEL_AVAILABLE, bool)

    def test_panel_import_error_is_str(self):
        assert isinstance(PANEL_IMPORT_ERROR, str)
        assert "panel" in PANEL_IMPORT_ERROR.lower()

    @pytest.mark.skipif(not PANEL_AVAILABLE, reason="Panel not installed")
    def test_require_panel_does_not_raise_when_installed(self):
        require_panel()  # should not raise

    def test_require_panel_message_is_actionable(self):
        """The error message should mention how to install."""
        assert "pip install" in PANEL_IMPORT_ERROR or "uv" in PANEL_IMPORT_ERROR


class TestEnvironmentDetection:
    """Tests for in_ipython and in_jupyter."""

    def test_in_ipython_returns_bool(self):
        result = in_ipython()
        assert isinstance(result, bool)

    def test_in_jupyter_returns_bool(self):
        result = in_jupyter()
        assert isinstance(result, bool)

    def test_in_jupyter_false_in_pytest(self):
        """pytest is not a Jupyter notebook."""
        assert in_jupyter() is False


class TestEnsurePanelExtension:
    """Tests for ensure_panel_extension."""

    def test_ensure_panel_extension_noop_outside_ipython(self):
        """Should not raise or crash when not in IPython."""
        ensure_panel_extension()

    def test_ensure_panel_extension_idempotent(self):
        """Calling twice should not raise."""
        ensure_panel_extension()
        ensure_panel_extension()


class TestDisplayOrReturn:
    """Tests for display_or_return."""

    def test_returns_layout_when_show_false(self):
        sentinel = object()
        result = display_or_return(sentinel, show=False)
        assert result is sentinel

    def test_returns_layout_when_not_jupyter(self):
        """Outside Jupyter, layout is returned even with show=True."""
        sentinel = object()
        result = display_or_return(sentinel, show=True)
        assert result is sentinel
