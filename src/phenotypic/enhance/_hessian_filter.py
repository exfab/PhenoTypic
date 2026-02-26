from __future__ import annotations
from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image

from skimage.filters import hessian

from phenotypic.abc_ import ImageEnhancer


class HessianFilter(ImageEnhancer):
    """Hessian-based multi-scale edge and boundary detection.

    Computes eigenvalue-based Hessian filter across multiple scales to detect edges,
    boundaries, and ridge-like structures by analyzing image curvature. On agar plate
    images, this enhances colony edges, boundaries, and thin filamentous structures,
    making detection more robust across varying colony sizes and morphologies.

    Use cases (agar plates):
    - Enhance sharp boundaries between colonies and agar background.
    - Detect thin or elongated structures (filaments, branching) with poor contrast.
    - Preprocess before thresholding-based or morphological detection.
    - Improve edge clarity for size-invariant colony segmentation.
    - Analyze textured colonies or biofilms with complex internal structure.

    Tuning and effects:
    - sigmas: Range of standard deviations for Gaussian derivatives. Smaller sigmas
      detect finer edges; larger sigmas detect broader structural boundaries. Should
      span the expected range of colony feature scales (e.g., sigmas=range(1, 5)
      for 1-5 pixel width edges). Multi-scale analysis improves robustness.
    - alpha: Sensitivity to plate-like (flat) structure deviations. Lower values are
      more permissive; higher values stricter. Typical range: 0.1-1.0. Default 0.5.
    - beta: Sensitivity to blob-like (spherical) structure deviations. Lower values
      are more permissive; higher values stricter. Typical range: 0.1-1.0. Default 0.5.
    - gamma: Background suppression parameter; suppresses low-curvature regions (agar
      background). Larger values suppress background more aggressively. Typical
      range: 10-20. Default 15.
    - black_ridges: If True, detect dark ridges (colonies) on bright background.
      If False, detect bright ridges on dark background. For plate imaging with
      dark colonies on light agar, use True.
    - mode: How to handle image boundaries ('reflect', 'constant', 'nearest', 'mirror',
      'wrap'). Default 'reflect' works well for most images.
    - cval: Constant value used if mode='constant'. Default 0.

    Guidance:
    - Hessian works best on images with relatively clear edges and contrast.
    - Combine with mild smoothing (GaussianBlur) beforehand to suppress noise.
    - Start with sigmas=range(1, 4) and adjust based on expected colony edge widths.
    - For fine edges, use smaller sigmas; for broader boundaries, use larger sigmas.
    - Output is a ridge response map (grayscale), not binary; thresholding may
      be needed afterward for segmentation.
    - Multi-scale analysis (multiple sigmas) is more robust than single-scale.

    Caveats:
    - Output is a ridge/edge probability measure, not binary. Thresholding required.
    - Sensitive to noise and texture artifacts; noisy images may produce spurious edges.
    - On textured agar or with dust/condensation, require prior smoothing.
    - Computational cost increases with number of sigmas and image size.
    - May over-emphasize fine texture if sigmas are too small or gamma too small.
    - Parameters alpha, beta, gamma require tuning for different colony morphologies.

    Attributes:
        sigmas (tuple | list): Sequence of standard deviations for Hessian
            computation. Each sigma represents a scale; default (1, 2, 3).
        alpha (float): Plate-like structure sensitivity (0 to 1). Default 0.5.
        beta (float): Blob-like structure sensitivity (0 to 1). Default 0.5.
        gamma (float): Background suppression threshold. Default 15.
        black_ridges (bool): If True, detect dark structures on bright background;
            if False, detect bright structures on dark background. Default False.
        mode (str): Boundary handling mode for Gaussian convolution. Default 'reflect'.
        cval (float): Constant value for 'constant' mode boundary handling. Default 0.
    """

    def __init__(
            self,
            sigmas: Iterable[float] = (1, 2, 3),
            alpha: float = 0.5,
            beta: float = 0.5,
            gamma: float = 15,
            black_ridges: bool = False,
            mode: str = 'reflect',
            cval: float = 0,
    ):
        """
        Parameters:
            sigmas (tuple | list): Sequence of standard deviations for Gaussian
                derivatives. Smaller values detect finer edges, larger values
                detect thicker structures. Default (1, 2, 3).
            alpha (float): Sensitivity to plate-like structure deviations. Lower
                values are more permissive. Range: 0 to 1. Default 0.5.
            beta (float): Sensitivity to blob-like structure deviations. Lower
                values are more permissive. Range: 0 to 1. Default 0.5.
            gamma (float): Threshold for background suppression. Larger values
                suppress low-curvature regions (agar background) more aggressively.
                Default 15.
            black_ridges (bool): If True, detect dark ridges (colonies) on bright
                background. If False, detect bright ridges on dark background.
                For agar plates with dark colonies on light background, use True.
                Default False.
            mode (str): Boundary handling mode ('reflect', 'constant', 'nearest',
                'mirror', 'wrap'). Default 'reflect'.
            cval (float): Constant value used if mode='constant'. Default 0.
        """
        self.sigmas = sigmas
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.black_ridges = black_ridges
        self.mode = mode
        self.cval = cval

    def _operate(self, image: Image) -> Image:
        image.detect_mat[:] = hessian(
                image=image.detect_mat[:],
                sigmas=self.sigmas,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                black_ridges=self.black_ridges,
                mode=self.mode,
                cval=self.cval,
        )
        return image
