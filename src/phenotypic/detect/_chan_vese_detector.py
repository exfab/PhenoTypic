from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._grid_image import GridImage
    from phenotypic._core._image import Image

from skimage import segmentation, morphology
from scipy import ndimage

from ..abc_ import ObjectDetector


class ChanVeseDetector(ObjectDetector):
    """Detect colonies by region-based level-set segmentation using the Chan-Vese energy functional.

    Partition the image into foreground and background by minimising an
    energy functional based on intensity homogeneity within each region.
    Because segmentation is driven by region statistics rather than edge
    gradients, colonies with diffuse boundaries, uneven texture, or gradual
    transitions into the agar background are captured cleanly. For algorithm
    details see :doc:`/explanation/detection_strategies_compared`.

    Args:
        mu: Edge-length penalty weight. Higher values produce smoother,
            rounder colony outlines; lower values preserve fine boundary
            detail. Typical range: 0.05--1.0. Default: 0.25.

        lambda1: Weight for intensity deviation inside detected regions.
            Increase to enforce uniform colony brightness. Default: 1.0.

        lambda2: Weight for intensity deviation outside detected regions.
            Increase when the agar background is homogeneous. Default: 1.0.

        max_num_iter: Maximum level-set iterations. Increase for complex
            images where convergence is slow; decrease for faster (but
            potentially incomplete) segmentation. Default: 500.

        tol: Convergence tolerance (L2 norm of level-set change). Smaller
            values require tighter convergence but more iterations.
            Default: 1e-3.

        dt: Step-size multiplier for level-set evolution. Larger values
            evolve faster but risk instability. Default: 0.5.

        init_level_set: Initialisation method for the level set. Accepted
            values: ``"checkerboard"``, ``"disk"``, ``"small disk"``.
            Checkerboard is robust for most plate images. Default:
            ``"checkerboard"``.

        min_size: Minimum colony area in pixels. Connected components
            smaller than this are removed as noise. Default: 50.

        connectivity: Pixel connectivity for labelling connected components.
            ``1`` for 4-connectivity, ``2`` for 8-connectivity. Default: 2.

    Returns:
        Image: Input image with ``objmask`` set to binary mask and
        ``objmap`` set to labeled connected components.

    Raises:
        ValueError: If ``init_level_set`` is not a recognised initialisation
            method.

    Best For:
        * Mucoid or fuzzy colonies whose edges lack sharp intensity gradients.
        * Plates with uneven colony pigmentation or heterogeneous surface
          texture that fragments threshold-based masks.
        * Low-contrast imaging where colony and agar intensities are similar.
        * Morphology studies where smooth, accurate colony outlines are
          required.

    Consider Also:
        * :class:`OtsuDetector` when colonies and background form two clear
          histogram peaks and a fast global threshold suffices.
        * :class:`HysteresisDetector` when colony brightness varies but edges
          are still reasonably sharp.
        * :class:`CannyDetector` when colonies are best delineated by edge
          contrast rather than region homogeneity.

    References:
        [1] T. F. Chan and L. A. Vese, "Active contours without edges,"
        *IEEE Trans. Image Process.*, vol. 10, no. 2, pp. 266--277, 2001.

    See Also:
        :doc:`/tutorials/notebooks/02_detecting_colonies`
            Step-by-step tutorial for basic colony detection.
        :doc:`/how_to/notebooks/choose_detection_algorithm`
            Guide for selecting the right detector for your plate images.
        :doc:`/explanation/detection_strategies_compared`
            In-depth comparison of all detection strategies.
    """

    def __init__(
        self,
        mu: float = 0.25,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
        max_num_iter: int = 500,
        tol: float = 1e-3,
        dt: float = 0.5,
        init_level_set: str = "checkerboard",
        min_size: int = 50,
        connectivity: int = 2,
    ):
        super().__init__()
        self.mu = mu
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.max_num_iter = max_num_iter
        self.tol = tol
        self.dt = dt
        self.init_level_set = init_level_set
        self.min_size = min_size
        self.connectivity = connectivity

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
