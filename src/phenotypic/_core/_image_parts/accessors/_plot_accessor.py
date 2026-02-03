from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image

from phenotypic._core._image_parts.plot_accessor import (
    AllDataPlotter,
    DiagnosticsPlotter,
    MorphologyPlotter,
    OverlayPlotter,
    SizeDistributionPlotter,
    SpatialPlotter,
    ThresholdPlotter,
)


class PlotAccessor(AllDataPlotter, DiagnosticsPlotter, MorphologyPlotter, OverlayPlotter, SizeDistributionPlotter, SpatialPlotter, ThresholdPlotter):
    """Provides quality-of-life plots for developing image processing pipelines.

    This accessor offers sophisticated visualization methods to help understand how
    morphological operations, size filtering, and spatial patterns affect colony
    detection in arrayed microbial cultures on solid agar media. These plots are
    designed for pipeline development and parameter tuning rather than publication.

    All methods support flexible data requirements, automatically detecting whether
    labeled objects (objmap) or binary masks (objmask) are available, and adapting
    their analysis accordingly.

    Note:
        For large images (>3000×3000 pixels), memory usage can be significant.
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
    """

    def __init__(self, root_image: Image) -> None:
        """Initialize PlotAccessor with a reference to the parent Image.

        Args:
            root_image: The parent Image instance containing detection results
                and image data.
        """
        # Initialize all parent classes
        AllDataPlotter.__init__(self, root_image)
        DiagnosticsPlotter.__init__(self, root_image)
        MorphologyPlotter.__init__(self, root_image)
        OverlayPlotter.__init__(self, root_image)
        SizeDistributionPlotter.__init__(self, root_image)
        SpatialPlotter.__init__(self, root_image)
        ThresholdPlotter.__init__(self, root_image)


__all__ = "PlotAccessor",
