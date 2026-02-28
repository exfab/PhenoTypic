"""Plot accessor using registry-based dispatch."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from phenotypic._core._image_parts.accessor_abstracts._image_accessor_base import (
    ImageAccessorBase,
)
from phenotypic.tools_.register import available_plotters, get_plotter

# Import plot_accessor package to trigger @register_plotter decorators
import phenotypic._core._image_parts.plot_accessor  # noqa: F401

if TYPE_CHECKING:
    from phenotypic._core._image import Image


class PlotAccessor(ImageAccessorBase):
    """Provides quality-of-life plots for developing image processing pipelines.

    This accessor offers sophisticated visualization methods to help understand how
    morphological operations, size filtering, and spatial patterns affect colony
    detection in arrayed microbial cultures on solid agar media. These plots are
    designed for pipeline development and parameter tuning rather than publication.

    All methods support flexible data requirements, automatically detecting whether
    labeled objects (objmap) or binary masks (objmask) are available, and adapting
    their analysis accordingly.

    Plotters are registered via the ``@register_plotter`` decorator and accessed
    dynamically by method name (e.g., ``image.plot.overlay()`` dispatches to
    ``OverlayPlotter``).

    Note:
        For large images (>3000x3000 pixels), memory usage can be significant.
        Caller is responsible for closing returned figures with ``plt.close(fig)``
        after saving to free memory and prevent accumulation of matplotlib figure
        objects in memory.

    Examples:
        Access plot methods through an Image instance:

        >>> from phenotypic import Image
        >>> from phenotypic.detect import OtsuDetector
        >>> # Load and detect colonies
        >>> image = Image.imread('plate.jpg')
        >>> detector = OtsuDetector()
        >>> detected = detector.apply(image)
        >>> # Access plot methods
        >>> fig, axes = detected.plot.morph_progression()
        >>> plt.savefig('morph.png')
        >>> plt.close(fig)  # Important: free memory
        >>> fig, ax = detected.plot.size_distribution()
        >>> plt.savefig('size.png')
        >>> plt.close(fig)

        List available plotters:

        >>> from phenotypic._core._image_parts.plot_accessor import available_plotters
        >>> print(available_plotters())
        ('all', 'diagnostics', 'morph_progression', 'overlay', ...)
    """

    def __init__(self, root_image: Image) -> None:
        """Initialize PlotAccessor with a reference to the parent Image.

        Args:
            root_image: The parent Image instance containing detection results
                and image data.
        """
        super().__init__(root_image)
        self._instances: dict[str, Any] = {}

    @property
    def _accessor_property_name(self) -> str:
        """Name of the Image property that surfaces this accessor."""
        return "plot"

    def __getattr__(self, name: str) -> Any:
        """Dispatch attribute access to registered plotter methods.

        First tries to find a plotter whose ``name`` attribute matches exactly.
        If not found, searches all registered plotters for a method with the
        requested name. This allows multi-method plotters to expose all their
        methods (e.g., ``MorphologyPlotter`` exposes ``morph_progression``,
        ``structural_response_curve``, and ``boundary_displacement``).

        Args:
            name: Name of the plotter method to access.

        Returns:
            The bound method from the registered plotter instance.

        Raises:
            AttributeError: If *name* is not found on any registered plotter.
        """
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

        # First, try direct registry lookup (plotter.name matches method name)
        try:
            plotter_cls = get_plotter(name)
            # Lazy instantiate and cache plotter by class name
            cls_name = plotter_cls.__name__
            if cls_name not in self._instances:
                self._instances[cls_name] = plotter_cls(self._root_image)
            return getattr(self._instances[cls_name], name)
        except ValueError:
            pass  # Not a primary method name, search all plotters

        # Search all registered plotters for a method with this name
        for plotter_name in available_plotters():
            plotter_cls = get_plotter(plotter_name)
            cls_name = plotter_cls.__name__

            # Check if the plotter class has this method
            if hasattr(plotter_cls, name) and callable(getattr(plotter_cls, name, None)):
                # Lazy instantiate and cache plotter
                if cls_name not in self._instances:
                    self._instances[cls_name] = plotter_cls(self._root_image)
                return getattr(self._instances[cls_name], name)

        # Not found in any plotter
        raise AttributeError(
            f"'{type(self).__name__}' has no attribute '{name}'. "
            f"Available plotters: {', '.join(available_plotters())}"
        )

    def __dir__(self) -> list[str]:
        """Return list of available attributes including all plotter methods."""
        methods = set(super().__dir__())
        methods |= set(available_plotters())

        # Include all public methods from each registered plotter
        for plotter_name in available_plotters():
            plotter_cls = get_plotter(plotter_name)
            for attr_name in dir(plotter_cls):
                if not attr_name.startswith("_") and callable(
                    getattr(plotter_cls, attr_name, None)
                ):
                    methods.add(attr_name)

        return sorted(methods)


__all__ = ("PlotAccessor",)
