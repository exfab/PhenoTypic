"""Panel accessor using registry-based dispatch."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from phenotypic._core._image_parts.accessor_abstracts._image_accessor_base import (
    ImageAccessorBase,
)
from phenotypic.tools_.register import available_dashboards, get_dashboard

# Import panel_accessor package to trigger @register_dashboard decorators
import phenotypic._core._image_parts.panel_accessor  # noqa: F401

if TYPE_CHECKING:
    from phenotypic import Image


class PanelAccessor(ImageAccessorBase):
    """Provides interactive Panel dashboards for image exploration.

    Dashboards are registered via the ``@register_dashboard`` decorator and
    accessed dynamically by method name (e.g., ``image.panel.detect_modes()``
    dispatches to ``DetectModeDashboard``).

    Examples:
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> image = load_synth_yeast_plate()
        >>> dashboard = image.panel.detect_modes()

        List available dashboards:

        >>> from phenotypic._core._image_parts.panel_accessor import available_dashboards
        >>> print(available_dashboards())
        ('detect_modes',)
    """

    def __init__(self, root_image: Image) -> None:
        """Initialize PanelAccessor with a reference to the parent Image.

        Args:
            root_image: The parent Image instance.
        """
        super().__init__(root_image)
        self._instances: dict[str, Any] = {}

    @property
    def _accessor_property_name(self) -> str:
        """Name of the Image property that surfaces this accessor."""
        return "panel"

    def __getattr__(self, name: str) -> Any:
        """Dispatch attribute access to registered dashboard methods.

        Args:
            name: Name of the dashboard method to access.

        Returns:
            The bound method from the registered dashboard instance.

        Raises:
            AttributeError: If *name* is not a registered dashboard.
        """
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

        # Auto-initialize Panel extension in Jupyter
        from phenotypic.gui._global_session import _ensure_panel_initialized

        _ensure_panel_initialized()
        try:
            dashboard_cls = get_dashboard(name)
        except ValueError:
            raise AttributeError(
                f"'{type(self).__name__}' has no attribute '{name}'. "
                f"Available: {', '.join(available_dashboards())}"
            ) from None

        # Lazy instantiate and cache dashboard
        if name not in self._instances:
            self._instances[name] = dashboard_cls(self._root_image)

        return getattr(self._instances[name], name)

    def __dir__(self) -> list[str]:
        """Return list of available attributes including registered dashboards."""
        return sorted(set(super().__dir__()) | set(available_dashboards()))


__all__ = ("PanelAccessor",)
