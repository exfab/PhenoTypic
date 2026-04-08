"""Registry for analysis plugin classes."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from ._base_registry import BaseRegistry

if TYPE_CHECKING:
    from phenotypic._cli._dashboard._analysis._base_plugin import BaseAnalysisPlugin


class AnalysisPluginRegistry(BaseRegistry["BaseAnalysisPlugin"]):
    """Registry for dashboard analysis plugins.

    Plugins are registered by their ``call_name`` class attribute, which is
    used as the HTML element ID prefix and sub-tab identifier.
    """

    _REGISTRY: ClassVar[dict[str, type["BaseAnalysisPlugin"]]] = {}
    _registry_name: ClassVar[str] = "analysis plugin"


def register_analysis(cls: type["BaseAnalysisPlugin"]) -> type["BaseAnalysisPlugin"]:
    """Decorator to register an analysis plugin class.

    The class must have a ``call_name`` class attribute used as the
    registration key and HTML ID prefix.

    Args:
        cls: Analysis plugin class to register.

    Returns:
        The registered class (unchanged).

    Examples:
        >>> @register_analysis
        ... class MyPlugin(BaseAnalysisPlugin):
        ...     call_name = "my_analysis"
        ...     display_name = "My Analysis"
        ...
        ...     def css(self): ...
        ...     def html(self): ...
        ...     def js(self): ...
    """
    return AnalysisPluginRegistry.register(cls)


def get_analysis_plugin(name: str) -> type["BaseAnalysisPlugin"]:
    """Look up registered analysis plugin by name.

    Args:
        name: The registered name of the plugin.

    Returns:
        The registered analysis plugin class.

    Raises:
        ValueError: If *name* is not registered.
    """
    return AnalysisPluginRegistry.get(name)


def available_analysis_plugins() -> tuple[str, ...]:
    """Return names of all registered analysis plugins.

    Returns:
        Tuple of registered plugin names, sorted alphabetically.
    """
    return AnalysisPluginRegistry.available()
