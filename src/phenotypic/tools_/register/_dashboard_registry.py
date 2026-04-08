"""Registry for dashboard classes."""

from __future__ import annotations

from typing import ClassVar

from ._base_registry import BaseRegistry


class DashboardRegistry(BaseRegistry):
    """Registry for dashboard classes (duck-typed, no strict base).

    Dashboards are registered by their ``call_name`` class attribute, which must
    match the method name used for dispatch (e.g., ``call_name = "detect_modes"``
    corresponds to ``image.panel.detect_modes()``).

    Dashboards may have optional dependencies (e.g., Panel). Classes should
    only be registered when their dependencies are available.
    """

    _REGISTRY: ClassVar[dict[str, type]] = {}
    _registry_name: ClassVar[str] = "dashboard"


def register_dashboard(cls: type) -> type:
    """Decorator to register a dashboard class.

    The class must have a ``call_name`` class attribute that matches the method
    name for dispatch.

    Args:
        cls: Dashboard class to register.

    Returns:
        The registered class (unchanged).

    Examples:
        >>> @register_dashboard
        ... class MyDashboard(BaseDashboard):
        ...     call_name = "my_dashboard"
        ...
        ...     def my_dashboard(self, **kwargs):
        ...         ...
    """
    return DashboardRegistry.register(cls)


def get_dashboard(name: str) -> type:
    """Look up registered dashboard by name.

    Args:
        name: The registered name of the dashboard.

    Returns:
        The registered dashboard class.

    Raises:
        ValueError: If *name* is not registered.
    """
    return DashboardRegistry.get(name)


def available_dashboards() -> tuple[str, ...]:
    """Return names of all registered dashboards.

    Returns:
        Tuple of registered dashboard names, sorted alphabetically.
    """
    return DashboardRegistry.available()
