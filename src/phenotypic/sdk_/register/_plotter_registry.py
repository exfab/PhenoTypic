"""Registry for plotter classes."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from ._base_registry import BaseRegistry

if TYPE_CHECKING:
    from phenotypic._core._image_parts.plot_accessor._base_plotter import BasePlotter


class PlotterRegistry(BaseRegistry["BasePlotter"]):
    """Registry for plotter classes.

    Plotters are registered by their ``call_name`` class attribute, which must
    match the method name used for dispatch (e.g., ``call_name = "overlay"``
    corresponds to ``image.plot.size_distribution()``).
    """

    _REGISTRY: ClassVar[dict[str, type["BasePlotter"]]] = {}
    _registry_name: ClassVar[str] = "plotter"


def register_plotter(cls: type["BasePlotter"]) -> type["BasePlotter"]:
    """Decorator to register a plotter class.

    The class must have a ``call_name`` class attribute that matches the method
    name for dispatch.

    Args:
        cls: Plotter class to register.

    Returns:
        The registered class (unchanged).

    Examples:
        >>> @register_plotter
        ... class MyPlotter(BasePlotter):
        ...     call_name = "my_plot"
        ...
        ...     def my_plot(self, **kwargs):
        ...         ...
    """
    return PlotterRegistry.register(cls)


def get_plotter(name: str) -> type["BasePlotter"]:
    """Look up registered plotter by name.

    Args:
        name: The registered name of the plotter.

    Returns:
        The registered plotter class.

    Raises:
        ValueError: If *name* is not registered.
    """
    return PlotterRegistry.get(name)


def available_plotters() -> tuple[str, ...]:
    """Return names of all registered plotters.

    Returns:
        Tuple of registered plotter names, sorted alphabetically.
    """
    return PlotterRegistry.available()
