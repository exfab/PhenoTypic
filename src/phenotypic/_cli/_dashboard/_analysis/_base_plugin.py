"""Abstract base class for dashboard analysis plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseAnalysisPlugin(ABC):
    """Base class for analysis tab plugins.

    Subclasses must define class attributes and implement the ``css``,
    ``html``, and ``js`` methods to contribute content to the dashboard
    analysis tab.

    Attributes:
        call_name: Short identifier used for registration and HTML IDs.
        display_name: Human-readable name shown in the sub-tab button.
        sort_order: Numeric order for tab arrangement (lower = leftmost).
        needs_measurements: Whether the plugin requires measurement data.
        needs_overlay_manifest: Whether the plugin requires overlay paths.
    """

    call_name: str
    display_name: str
    sort_order: int = 0
    needs_measurements: bool = True
    needs_overlay_manifest: bool = False

    @abstractmethod
    def css(self) -> str:
        """Return CSS scoped with the plugin's call_name prefix."""

    @abstractmethod
    def html(self) -> str:
        """Return HTML for the sub-tab body."""

    @abstractmethod
    def js(self) -> str:
        """Return JS including an ``initAnalysis_{call_name}()`` function."""
