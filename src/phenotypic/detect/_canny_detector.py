from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._grid_image import GridImage
    from phenotypic._core._image import Image

from skimage import feature, morphology
from scipy import ndimage

from phenotypic.abc_ import ThresholdDetector


class CannyDetector(ThresholdDetector):
    """Detect colonies by tracing edges and labelling connected regions.

    Applies multi-stage Canny edge detection — Gaussian smoothing, gradient
    estimation, non-maximum suppression, and dual-threshold hysteresis — to
    produce thin edge pixels, then labels connected components of either the
    inverted edge map or the edge map itself. This does not explicitly close
    contours or fill interiors; colony-sized regions are recovered only when
    edges form useful barriers and size filtering removes background regions.
    Because detection relies on boundary contrast rather than absolute
    intensity, it can remain useful on plates with uneven illumination or
    translucent colonies. For a full comparison of detection strategies, see
    :doc:`/explanation/detection_strategies_compared`.

    Best For:
        - Well-separated colonies on solid media where colony edges are sharper
          than the intensity difference relative to background.
        - Translucent or lightly pigmented colonies that lack sufficient
          intensity contrast for threshold-based methods.
        - Plates with heterogeneous colony texture or pigmentation that cause
          intensity-based methods to fragment objects.
        - Images with moderate vignetting where edge contrast is preserved even
          though absolute intensity varies across the plate.

    Consider Also:
        - :class:`OtsuDetector` when colonies differ from background primarily
          in brightness rather than edge contrast.
        - :class:`WatershedDetector` when touching colonies must be separated
          by region-growing from interior distance-transform seeds.
        - :class:`HysteresisDetector` when dual-threshold intensity
          segmentation is preferred over edge-based detection.

    Args:
        sigma: Standard deviation of the Gaussian pre-smoothing kernel in
            pixels. Larger values suppress high-frequency noise but broaden
            edge profiles and may merge boundaries of closely spaced colonies.
            Typical range: 0.5--3.0. Default: 1.0. A reasonable starting point
            for noisy CCD images or plates with fine colony texture is 2.0--3.0.
        low_threshold: Lower hysteresis gate. When ``use_quantiles`` is
            ``True``, interpreted as a fractional rank of gradient magnitudes
            (0.1 retains edges stronger than 10 % of all measured gradients).
            When ``False``, an absolute gradient value requiring per-setup
            calibration. Raise to prune weak noise-driven edge fragments;
            lower to recover faint boundaries on low-contrast colonies.
            Typical quantile range: 0.05--0.2. Default: 0.1.
        high_threshold: Upper hysteresis gate that seeds new edge chains. Must
            exceed ``low_threshold``. A higher value seeds fewer but more
            confident edge segments; a lower value seeds more chains at the
            risk of noise-seeded spurious regions. A 2:1 to 3:1 high-to-low
            ratio is effective for moderately noisy images. Typical quantile
            range: 0.1--0.4. Default: 0.2.
        use_quantiles: When ``True`` (default), thresholds are interpreted as
            gradient-magnitude percentile ranks, adapting automatically to
            image contrast across batches with variable scanner gain or
            illumination. When ``False``, thresholds are absolute gradient
            values for tightly controlled imaging conditions. Default: True.
        min_size: Minimum connected-region area in pixels. Regions smaller than
            this are discarded as dust, debris, or condensation droplets after
            labelling. Typical range: 20--500 px, scaling with image
            resolution. Default: 50.
        invert_edges: When ``True`` (default), the binary edge map is inverted
            before labelling so that non-edge regions can become colony objects.
            Set to ``False`` to label the edge pixels themselves, which is
            useful for inspecting edge closure and gap locations during
            parameter tuning. Default: True.
        connectivity: Pixel connectivity for connected-component labelling.
            ``1`` for 4-connectivity; ``2`` for 8-connectivity. Use ``2``
            (default) to bridge single-pixel diagonal gaps in Canny edge
            contours, preventing spurious region splits at diagonal steps.
            Default: 2.

    Returns:
        Image: Input image with ``objmap`` set to a labelled colony map where
        each retained connected region receives a unique integer label, and ``objmask``
        derived from the non-zero entries of that map.

    Raises:
        ValueError: If ``high_threshold`` is less than ``low_threshold``.

    References:
        [1] J. Canny, "A computational approach to edge detection," *IEEE
        Trans. Pattern Anal. Mach. Intell.*, vol. 8, no. 6, pp. 679--698,
        Nov. 1986.

    See Also:
        :doc:`/tutorials/notebooks/02_detecting_colonies` for a step-by-step
        tutorial demonstrating colony detection on real plate images.
        :doc:`/how_to/notebooks/choose_detection_algorithm` for a guide to
        selecting the right detector for your plate images.
        :doc:`/explanation/detection_strategies_compared` for an in-depth
        comparison of all detection strategies and their failure modes.
    """

    sigma: float = 1.0
    low_threshold: float = 0.1
    high_threshold: float = 0.2
    use_quantiles: bool = True
    min_size: int = 50
    invert_edges: bool = True
    connectivity: int = 2

    def _operate(self, image: Image | GridImage) -> Image:

        enhanced_matrix = image.detect_mat[:]

        # Apply Canny edge detection
        edges = feature.canny(
                image=enhanced_matrix,
                sigma=self.sigma,
                low_threshold=self.low_threshold,
                high_threshold=self.high_threshold,
                use_quantiles=self.use_quantiles,
        )

        # Invert edges to get regions (colonies) if requested
        if self.invert_edges:
            regions = ~edges
        else:
            regions = edges

        # Label connected components
        objmap, _ = ndimage.label(
                regions,
                structure=ndimage.generate_binary_structure(2, self.connectivity)
        )

        # Remove small objects (pass a boolean array when only one label
        # exists to avoid skimage's "single-label" ambiguity warning).
        if objmap.max() <= 1:
            objmap = morphology.remove_small_objects(
                    objmap.astype(bool), min_size=self.min_size
            ).astype(objmap.dtype)
        else:
            objmap = morphology.remove_small_objects(objmap, min_size=self.min_size)

        # Ensure correct dtype
        if objmap.dtype != image._OBJMAP_DTYPE:
            objmap = objmap.astype(image._OBJMAP_DTYPE)

        # Relabel to ensure consecutive labels
        image.objmap[:] = objmap
        image.objmap.relabel(connectivity=self.connectivity)

        return image


# Set the docstring so that it appears in the sphinx documentation
CannyDetector.apply.__doc__ = CannyDetector._operate.__doc__
