from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image, GridImage

from skimage import segmentation, morphology
from scipy import ndimage

from ..abc_ import ObjectDetector


class ChanVeseDetector(ObjectDetector):
    """Region-based active contour detector for colonies with fuzzy or soft edges.

    ChanVeseDetector applies the Chan-Vese level-set segmentation algorithm, which
    partitions an image into foreground and background by minimizing an energy
    functional based on intensity homogeneity within each region. Unlike threshold
    or edge detectors, Chan-Vese does not rely on sharp intensity gradients and can
    segment colonies with diffuse boundaries, uneven internal texture, or gradual
    transitions into the agar background.

    Args:
        mu: Edge length weight (default 0.25). Higher values penalize boundary
            length, producing smoother/rounder colony outlines. Primary tuning
            knob for colony morphology. Increase for noisy images or to suppress
            irregular protrusions; decrease to preserve fine boundary detail.

        lambda1: Weight for intensity deviation inside detected regions
            (default 1.0). Higher values force detected regions toward uniform
            internal intensity. Increase when colonies have consistent brightness.

        lambda2: Weight for intensity deviation outside detected regions
            (default 1.0). Higher values force the background toward uniform
            intensity. Increase when agar background is homogeneous.

        max_num_iter: Maximum iterations before stopping (default 500). Increase
            for complex images where convergence is slow; decrease for faster
            (but potentially incomplete) segmentation.

        tol: Convergence tolerance as L2 norm of level set change (default 1e-3).
            Smaller values require tighter convergence but more iterations.

        dt: Step size multiplier for level set evolution (default 0.5). Larger
            values evolve faster but risk instability; smaller values are more
            stable but slower.

        init_level_set: Initialization method for the level set (default
            ``"checkerboard"``). Options: ``"checkerboard"``, ``"disk"``,
            ``"small disk"``. Checkerboard is robust for most plate images.

        min_size: Minimum object area in pixels (default 50). Post-detection
            filter to remove noise, dust, or spurious small regions.

        connectivity: Connectivity for connected-component labeling (default 2).
            1 = 4-connected, 2 = 8-connected. Higher connectivity merges
            diagonally adjacent pixels into the same colony.

    Attributes:
        mu, lambda1, lambda2, max_num_iter, tol, dt, init_level_set,
        min_size, connectivity

    Returns:
        Image: Input image with objmap set to labeled colonies from Chan-Vese
        segmentation. Each unique positive integer identifies a distinct colony;
        background is 0.

    Raises:
        ValueError: If init_level_set is not a recognized initialization method.

    **Use cases**

    - **Fuzzy or mucoid colonies:** Colonies with soft, diffuse edges that lack
      sharp intensity gradients. Threshold and Canny methods fragment or miss
      these boundaries; Chan-Vese segments by region homogeneity instead.
    - **Uneven colony texture:** Heterogeneous internal pigmentation or surface
      texture that causes threshold-based methods to fragment colonies.
    - **Low-contrast plates:** Faint colonies on similarly toned agar where
      intensity differences are subtle. Chan-Vese leverages region statistics
      rather than absolute intensity cutoffs.
    - **Smooth boundary recovery:** When accurate colony outlines matter (e.g.,
      morphology measurements), Chan-Vese's mu parameter controls boundary
      smoothness directly.

    **Limitations**

    - **Computational cost:** Iterative level-set evolution is significantly slower
      than single-pass threshold or edge methods. Not suitable for real-time
      processing or very large batch jobs without reducing max_num_iter.
    - **Two-phase assumption:** Chan-Vese partitions into exactly two regions
      (foreground/background). Plates with multiple intensity populations (e.g.,
      mixed species with different pigmentation) may require post-processing.
    - **Parameter sensitivity:** mu, lambda1, and lambda2 interact. Poor choices
      can over-smooth (high mu) or produce noisy boundaries (low mu). Test on
      representative images before batch processing.
    - **Initialization dependence:** Results can vary with init_level_set choice,
      though ``"checkerboard"`` is robust for most cases.
    - **No edge awareness:** Unlike Canny, Chan-Vese does not explicitly detect
      edges. Colonies defined primarily by boundary sharpness (not region
      homogeneity) may be better served by edge-based detectors.

    **Parameter effects on colony detection**

    - **mu:** Primary control for boundary smoothness. Low mu (< 0.1) → jagged,
      detailed boundaries that follow noise. High mu (> 1.0) → very smooth,
      circular boundaries that may merge nearby colonies.
    - **lambda1/lambda2:** Balance foreground vs background homogeneity. Equal
      values (default) treat both symmetrically. Increase lambda1 to tighten
      foreground uniformity; increase lambda2 to enforce cleaner background.
    - **max_num_iter/tol:** Control convergence trade-off. More iterations with
      tighter tolerance → better segmentation but slower. For quick screening,
      reduce max_num_iter to 200.
    - **min_size:** Post-processing filter. Increase to suppress dust/debris;
      decrease to retain tiny colonies.

    Examples:
        Basic Chan-Vese detection for fuzzy colonies::

            from phenotypic import Image
            from phenotypic.detect import ChanVeseDetector

            plate = Image.imread("mucoid_plate.jpg")
            detector = ChanVeseDetector(mu=0.25, max_num_iter=500)
            detected = detector.apply(plate)
            num_colonies = detected.objects.count
            print(f"Detected {num_colonies} colonies via Chan-Vese")

        Pipeline with preprocessing for low-contrast plates::

            from phenotypic import ImagePipeline
            from phenotypic.enhance import GaussianBlur, CLAHE
            from phenotypic.detect import ChanVeseDetector

            pipeline = ImagePipeline([
                GaussianBlur(sigma=1.5),
                CLAHE(clip_limit=2.0),
                ChanVeseDetector(mu=0.3, lambda1=1.0, lambda2=1.0,
                                 max_num_iter=500, min_size=100)
            ])

            image = Image.imread("low_contrast_plate.jpg")
            result = pipeline.apply(image)
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
