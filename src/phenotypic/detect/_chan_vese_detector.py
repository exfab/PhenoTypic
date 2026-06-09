from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from phenotypic._core._grid_image import GridImage
    from phenotypic._core._image import Image

from pydantic import Field
from skimage import segmentation, morphology
from scipy import ndimage

from ..abc_ import ObjectDetector
from ..tools_.typing_ import TuneSpec


class ChanVeseDetector(ObjectDetector):
    """Detect colonies by evolving a level-set contour that minimises region-intensity variance via the Chan-Vese method.

    Partition ``detect_mat`` into foreground and background by iterating a
    level-set that minimises the summed intensity variance inside and outside
    the detected regions. Because the energy is driven by region statistics
    rather than edge gradients, colonies with diffuse boundaries, gradual
    transitions into agar, or heterogeneous surface texture are captured
    cleanly as smooth, well-regularised outlines. The contour is initialised
    and evolved until the per-pixel change falls below ``tol`` or
    ``max_num_iter`` is reached, after which the binary mask is labelled into
    individual connected-component colonies.

    For a comparison of region-based and edge-based detection strategies, see
    :doc:`/explanation/detection_strategies_compared`.

    Best For:
        - Mucoid or spreading colonies whose boundaries lack sharp intensity
          gradients.
        - Plates with uneven colony pigmentation or sector-variant morphology
          that fragments threshold-based masks.
        - Low-contrast imaging conditions where colony and agar intensities
          overlap in the histogram.
        - Morphology studies requiring smooth, well-regularised colony outlines
          rather than jagged threshold masks.

    Consider Also:
        - :class:`OtsuDetector` when colonies and background form two distinct
          histogram peaks and a fast global threshold suffices.
        - :class:`HysteresisDetector` when colony brightness varies but edges
          remain reasonably sharp.
        - :class:`CannyDetector` when colonies are best delineated by edge
          contrast rather than interior region homogeneity.
        - :class:`WatershedDetector` when touching, compact colonies must be
          separated after an initial binary detection.

    Args:
        mu: Contour-length penalty weight in the Chan-Vese energy functional.
            Higher values produce shorter, smoother outlines and merge nearby
            colonies; lower values allow the contour to track fine boundary
            detail at the cost of noise sensitivity. Typical range: 0--1
            (lower for diffuse mucoid edges, higher for noisy plates).
            Default: 0.25.
        lambda1: Fidelity weight for intensity deviation inside detected
            regions. Increasing ``lambda1`` forces the contour to enclose
            pixels whose intensity matches the estimated colony mean more
            tightly. Equal weights (``lambda1`` = ``lambda2`` = 1.0) are the
            standard default recommended by Chan & Vese (2001); use asymmetric
            weights when colony and agar intensity distributions differ
            strongly. Default: 1.0.
        lambda2: Fidelity weight for intensity deviation outside detected
            regions (background). Increasing ``lambda2`` relative to
            ``lambda1`` penalises background heterogeneity more strongly
            (follows directly from the energy functional); useful when agar is
            uniform but colony texture is heterogeneous. Default: 1.0.
        max_num_iter: Hard cap on level-set evolution iterations. The
            algorithm stops early if ``tol`` is satisfied. Increase for
            complex or slowly converging shapes (mucoid, low-contrast);
            decrease to trade boundary accuracy for speed. Typical range:
            100--1000. Default: 500.
        tol: Early-stopping criterion: per-pixel level-set change normalised
            by image area. Smaller values give more precise convergence but
            require more iterations; values below the noise floor exhaust
            ``max_num_iter`` without improving results. Typical range:
            1e-5--1e-2. Default: 1e-3.
        dt: Step-size multiplier for each level-set iteration. Larger values
            converge faster but may lead to convergence problems on complex or
            low-contrast shapes; keep within 0.1--1.0 on real plate images.
            Default: 0.5.
        init_level_set: Shape used to initialise the level-set function
            before evolution. Accepted values: ``"checkerboard"`` (sin × sin
            pattern; fast multi-front convergence, well-suited to arrayed
            plates with spatially distributed colonies), ``"disk"`` (large
            circle shrinking inward; slower but more likely to detect implicit
            edges), ``"small disk"`` (small expanding circle at the image
            centre). Default: ``"checkerboard"``.
        min_size: Minimum connected-component area in pixels. Components
            smaller than this are discarded after labelling. Increase to
            remove condensation droplets, debris, or noise fragments; decrease
            for high-density formats where each colony occupies few pixels.
            Typical range: 10--500. Default: 50.
        connectivity: Pixel connectivity for connected-component labelling.
            ``1`` uses 4-connectivity (orthogonal neighbours only; keeps
            colonies that touch only at corners separate); ``2`` uses
            8-connectivity (all neighbours including diagonals; merges
            corner-touching regions). Default: 2.

    Returns:
        Image: Input image with ``objmask`` set to the binary segmentation
        mask and ``objmap`` set to consecutively labelled connected
        components.

    Raises:
        ValueError: If ``init_level_set`` is not one of the recognised
            initialisation strings.

    References:
        [1] T. F. Chan and L. A. Vese, "Active contours without edges,"
        *IEEE Trans. Image Process.*, vol. 10, no. 2, pp. 266--277,
        Feb. 2001.

        [2] P. Getreuer, "Chan-Vese segmentation," *Image Process. On
        Line*, vol. 2, pp. 214--224, 2012.

    See Also:
        :doc:`/tutorials/notebooks/02_detecting_colonies`
            Step-by-step tutorial for basic colony detection on plate images.
        :doc:`/how_to/notebooks/choose_detection_algorithm`
            Guide for selecting the right detector for your imaging conditions.
        :doc:`/explanation/detection_strategies_compared`
            In-depth comparison of all detection strategies and their
            failure modes.
    """

    mu: Annotated[float, TuneSpec(0.0, 1.0)] = 0.25
    lambda1: Annotated[float, TuneSpec(0.5, 2.0)] = 1.0
    lambda2: Annotated[float, TuneSpec(0.5, 2.0)] = 1.0
    max_num_iter: Annotated[int, TuneSpec(100, 1000)] = 500
    tol: Annotated[float, TuneSpec(1e-5, 1e-2, log=True)] = 1e-3
    dt: Annotated[float, TuneSpec(0.1, 1.0)] = 0.5
    init_level_set: str = "checkerboard"
    min_size: Annotated[int, TuneSpec(10, 500, log=True)] = 50
    connectivity: Annotated[int, TuneSpec(categories=[1, 2])] = Field(2, ge=1, le=2)

    def _operate(self, image: Image | GridImage) -> Image:
        """Apply Chan-Vese level-set segmentation to detect colonies.

        Reads the detection matrix, runs the Chan-Vese algorithm to produce a
        binary segmentation, then labels connected components and filters small
        objects. Sets image.objmap with consecutive colony labels.

        Args:
            image: Input image with ``detect_mat`` attribute (2D grayscale
                detection matrix).

        Returns:
            Image: Input image with ``objmap`` set to labeled colony map
            (consecutive integer labels, background = 0).
        """
        enhanced_matrix = image.detect_mat[:]

        # Run Chan-Vese segmentation → boolean mask
        cv_mask = segmentation.chan_vese(
            enhanced_matrix,
            mu=self.mu,
            lambda1=self.lambda1,
            lambda2=self.lambda2,
            max_num_iter=self.max_num_iter,
            tol=self.tol,
            dt=self.dt,
            init_level_set=self.init_level_set,
        )

        # Label connected components
        objmap, _ = ndimage.label(
            cv_mask,
            structure=ndimage.generate_binary_structure(2, self.connectivity),
        )

        # Remove small objects
        objmap = morphology.remove_small_objects(objmap, min_size=self.min_size)

        # Ensure correct dtype
        if objmap.dtype != image._OBJMAP_DTYPE:
            objmap = objmap.astype(image._OBJMAP_DTYPE)

        # Set objmap and relabel for consecutive IDs
        image.objmap[:] = objmap
        image.objmap.relabel(connectivity=self.connectivity)

        return image


# Set the docstring so that it appears in the sphinx documentation
ChanVeseDetector.apply.__doc__ = ChanVeseDetector._operate.__doc__
