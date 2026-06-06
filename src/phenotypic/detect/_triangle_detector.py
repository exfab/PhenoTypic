from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image
from skimage.filters import threshold_triangle

from ..abc_ import ThresholdDetector


class TriangleDetector(ThresholdDetector):
    """Detect colonies by triangle thresholding on skewed, background-dominant plate histograms.

    Compute a threshold at the base of the triangle formed between the
    histogram peak, the minimum, and the maximum. This method excels when
    colonies occupy a small fraction of the plate so that the intensity
    histogram is strongly skewed toward background, capturing faint colonies
    that variance-based methods may miss. For a full comparison of detection
    strategies see :doc:`/explanation/detection_strategies_compared`.

    Best For:
        - Plates where colonies are sparse and the agar background dominates
          the intensity histogram.
        - Faintly pigmented or translucent colonies that produce a small
          foreground peak relative to the background tail.
        - Early time-point images where colony growth is minimal and most
          pixels represent agar background.
        - Drop-out screens with many empty grid positions and few visible
          colonies.

    Consider Also:
        - :class:`OtsuDetector` when colonies and background occupy roughly
          equal histogram areas, giving a balanced bimodal distribution.
        - :class:`HysteresisDetector` when colony brightness varies across
          the plate and a single threshold under-segments faint regions.
        - :class:`UserThreshold` when an empirically determined threshold is
          known to outperform automatic methods for your plate type.

    Returns:
        Image: Input image with ``objmask`` set to the binary colony mask and
        ``objmap`` set to labeled connected components.

    Raises:
        ValueError: If threshold computation fails due to a degenerate
            histogram with insufficient intensity variation.

    References:
        [1] G. W. Zack, W. E. Rogers, and S. A. Latt, "Automatic
        measurement of sister chromatid exchange frequency," *J. Histochem.
        Cytochem.*, vol. 25, no. 7, pp. 741--753, 1977.

    See Also:
        :doc:`/tutorials/notebooks/02_detecting_colonies` for a step-by-step
        tutorial on basic colony detection.
        :doc:`/how_to/notebooks/choose_detection_algorithm` for guidance on
        selecting the right detector for your plate images.
        :doc:`/explanation/detection_strategies_compared` for an in-depth
        comparison of all thresholding strategies.
    """

    def _operate(self, image: Image) -> Image:
        """
        Applies a thresholding operation on the detection matrix of an image using
        the triangle method.

        Thresholding is performed by comparing each element in the detection matrix
        to the computed triangular threshold, setting the corresponding other_image in
        the output mask (`omask`) to True if the condition is satisfied.

        Args:
            image (Image): The arr image object containing a detection matrix
                (`detect_mat`) which will be processed to generate an output mask.

        Returns:
            Image: The modified image object with an updated output mask (`omask`).
        """
        nbins = 2 ** image.bit_depth
        image.objmask[:] = image.detect_mat[:] >= threshold_triangle(
                image.detect_mat[:], nbins=nbins
        )
        return image


# Set the docstring so that it appears in the sphinx documentation
TriangleDetector.apply.__doc__ = TriangleDetector._operate.__doc__
