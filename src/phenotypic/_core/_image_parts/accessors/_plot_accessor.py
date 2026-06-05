"""Plot accessor with explicit methods for built-in plotters.

Built-in plot methods are defined as explicit methods for IDE autocomplete.
User-registered plotters are still accessible via ``__getattr__`` fallback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from phenotypic._core._image_parts.accessor_abstracts._image_accessor_base import (
    ImageAccessorBase,
)
from phenotypic._core._image_parts.plot_accessor._all_data_plotter import (
    AllDataPlotter,
)
from phenotypic._core._image_parts.plot_accessor._detect_modes_plotter import (
    DetectModesPlotter,
)
from phenotypic._core._image_parts.plot_accessor._diagnostics_plotter import (
    DiagnosticsPlotter,
)
from phenotypic._core._image_parts.plot_accessor._morphology_plotter import (
    MorphologyPlotter,
)
from phenotypic._core._image_parts.plot_accessor._size_distribution_plotter import (
    SizeDistributionPlotter,
)
from phenotypic._core._image_parts.plot_accessor._spatial_plotter import (
    SpatialPlotter,
)
from phenotypic._core._image_parts.plot_accessor._threshold_plotter import (
    ThresholdPlotter,
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

    Built-in plot methods (``all``, ``morph_progression``, etc.) are defined as
    explicit methods for IDE autocomplete. User-registered plotters added via
    ``@register_plotter`` are still accessible through dynamic dispatch.

    For overlay visualization, use ``image.show(overlay=True)`` or
    ``image.dash(overlay=True)`` instead.

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
        ('all', 'diagnostics', 'morph_progression', ...)
    """

    def __init__(self, root_image: Image) -> None:
        """Initialize PlotAccessor with a reference to the parent Image.

        Args:
            root_image: The parent Image instance containing detection results
                and image data.
        """
        super().__init__(root_image)
        self._instances: dict[str, Any] = {}
        self._dash_accessor: Any = None

    @property
    def _accessor_property_name(self) -> str:
        """Name of the Image property that surfaces this accessor."""
        return "plot"

    @property
    def dash(self) -> Any:
        """Interactive Plotly/ipywidgets views: ``image.plot.dash.<name>()``.

        Dispatches to a registered :class:`FigureProvider` plotter's ``.dash()``
        — a composed ``go.Figure`` for control-free providers, or an ipywidgets
        dashboard when the figures declare ``Control``s. Mirrors the
        ``@register_plotter`` registry used by ``image.plot.<name>()``.

        Examples:
            >>> from phenotypic.data import load_synth_yeast_plate
            >>> image = load_synth_yeast_plate()
            >>> dashboard = image.plot.dash.diagnostics()  # ipywidgets dashboard
        """
        if self._dash_accessor is None:
            from ._dash_plot_accessor import DashPlotAccessor

            self._dash_accessor = DashPlotAccessor(self)
        return self._dash_accessor

    def _get_or_create(self, cls: type) -> Any:
        """Lazily instantiate and cache a plotter by its class.

        Args:
            cls: The plotter class to instantiate.

        Returns:
            A cached plotter instance bound to this accessor's image.
        """
        cls_name = cls.__name__
        if cls_name not in self._instances:
            self._instances[cls_name] = cls(self._root_image)
        return self._instances[cls_name]

    # -- Built-in plotter methods (explicit for IDE autocomplete) --

    def all(self, *args: Any, **kwargs: Any) -> Any:
        """Plot all available image data layers side by side.

        See :meth:`~phenotypic._core._image_parts.plot_accessor.AllDataPlotter.all`.
        """
        return self._get_or_create(AllDataPlotter).all(*args, **kwargs)

    def morph_progression(self, *args: Any, **kwargs: Any) -> Any:
        """Show effects of morphological operations at increasing kernel sizes.

        See :meth:`~phenotypic._core._image_parts.plot_accessor.MorphologyPlotter.morph_progression`.
        """
        return self._get_or_create(MorphologyPlotter).morph_progression(
            *args, **kwargs
        )

    def structural_response_curve(self, *args: Any, **kwargs: Any) -> Any:
        """Plot structural response metrics across kernel sizes.

        See :meth:`~phenotypic._core._image_parts.plot_accessor.MorphologyPlotter.structural_response_curve`.
        """
        return self._get_or_create(MorphologyPlotter).structural_response_curve(
            *args, **kwargs
        )

    def boundary_displacement(self, *args: Any, **kwargs: Any) -> Any:
        """Visualize boundary displacement from morphological operations.

        See :meth:`~phenotypic._core._image_parts.plot_accessor.MorphologyPlotter.boundary_displacement`.
        """
        return self._get_or_create(MorphologyPlotter).boundary_displacement(
            *args, **kwargs
        )

    def size_distribution(self, *args: Any, **kwargs: Any) -> Any:
        """Plot object size distribution with optional threshold lines.

        See :meth:`~phenotypic._core._image_parts.plot_accessor.SizeDistributionPlotter.size_distribution`.
        """
        return self._get_or_create(SizeDistributionPlotter).size_distribution(
            *args, **kwargs
        )

    def size_viewer(self, *args: Any, **kwargs: Any) -> Any:
        """Interactive size distribution viewer with threshold selection.

        See :meth:`~phenotypic._core._image_parts.plot_accessor.SizeDistributionPlotter.size_viewer`.
        """
        return self._get_or_create(SizeDistributionPlotter).size_viewer(
            *args, **kwargs
        )

    def spatial_size_map(self, *args: Any, **kwargs: Any) -> Any:
        """Heatmap of object sizes across spatial positions.

        See :meth:`~phenotypic._core._image_parts.plot_accessor.SpatialPlotter.spatial_size_map`.
        """
        return self._get_or_create(SpatialPlotter).spatial_size_map(*args, **kwargs)

    def size_scatter(self, *args: Any, **kwargs: Any) -> Any:
        """Scatter plot of object sizes colored by a secondary metric.

        See :meth:`~phenotypic._core._image_parts.plot_accessor.SpatialPlotter.size_scatter`.
        """
        return self._get_or_create(SpatialPlotter).size_scatter(*args, **kwargs)

    def try_thresh(self, *args: Any, **kwargs: Any) -> Any:
        """Compare multiple thresholding techniques side by side.

        See :meth:`~phenotypic._core._image_parts.plot_accessor.ThresholdPlotter.try_thresh`.
        """
        return self._get_or_create(ThresholdPlotter).try_thresh(*args, **kwargs)

    def diagnostics(self, *args: Any, **kwargs: Any) -> Any:
        """Comprehensive image quality diagnostics dashboard.

        See :meth:`~phenotypic._core._image_parts.plot_accessor.DiagnosticsPlotter.diagnostics`.
        """
        return self._get_or_create(DiagnosticsPlotter).diagnostics(*args, **kwargs)

    def detect_modes(self, *args: Any, **kwargs: Any) -> Any:
        """Faceted comparison of every registered detection mode.

        See :meth:`~phenotypic._core._image_parts.plot_accessor.DetectModesPlotter.detect_modes`.
        """
        return self._get_or_create(DetectModesPlotter).detect_modes(*args, **kwargs)

    # -- Dynamic dispatch for user-registered plotters --

    def __getattr__(self, name: str) -> Any:
        """Dispatch attribute access to user-registered plotter methods.

        Built-in plotters are resolved as explicit methods above. This fallback
        handles plotters added at runtime via ``@register_plotter``.

        Args:
            name: Name of the plotter method to access.

        Returns:
            The bound method from the registered plotter instance.

        Raises:
            AttributeError: If *name* is not found on any registered plotter.
        """
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

        # Try direct registry lookup (plotter.name matches method name)
        try:
            plotter_cls = get_plotter(name)
            instance = self._get_or_create(plotter_cls)
            return getattr(instance, name)
        except ValueError:
            pass  # Not a primary method name, search all plotters

        # Search all registered plotters for a method with this name
        for plotter_name in available_plotters():
            plotter_cls = get_plotter(plotter_name)

            # Check if the plotter class has this method
            if hasattr(plotter_cls, name) and callable(
                getattr(plotter_cls, name, None)
            ):
                instance = self._get_or_create(plotter_cls)
                return getattr(instance, name)

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
