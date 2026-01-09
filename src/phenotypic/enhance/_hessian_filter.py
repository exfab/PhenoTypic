from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image

from skimage.filters import hessian

from phenotypic.abc_ import ImageEnhancer


class HessianFilter(ImageEnhancer):
    """Hessian-based edge detection to enhance boundaries and thin structures.

    Computes the eigenvalue-based Hessian filter to detect edges and boundaries
    by analyzing the curvature of the image. On agar plate images, this enhances
    colony edges and boundaries, making detection more robust.

    Use cases (agar plates):
    - Enhance sharp boundaries between colonies and agar background.
    - Detect thin or elongated structures with poor contrast.
    - Preprocess before thresholding-based detection.
    - Improve edge clarity for morphological analysis.

    Tuning and effects:
    - sigma: Scale at which edges are detected. Smaller values detect finer edges;
      larger values detect broader structural boundaries. Should match expected
      edge width (e.g., sigma=1.5 for sharp edges, sigma=3.0 for smooth boundaries).
    - mode: How to handle image boundaries ('reflect', 'constant', 'nearest', 'mirror',
      'wrap'). Default 'reflect' works well for most images.

    Guidance:
    - Hessian works best on images with relatively clear edges and contrast.
    - Combine with preprocessing (e.g., GaussianBlur) if image is noisy.
    - Start with sigma=1.5 and adjust based on colony edge sharpness.
    - Output is an edge magnitude map (grayscale), not binary; thresholding may
      be needed afterward for segmentation.

    Caveats:
    - Sensitive to noise and texture artifacts; noisy images may produce spurious edges.
    - On textured agar or with dust/condensation, may require prior smoothing.
    - Computational cost increases with sigma; use moderate values for speed.
    - Can over-emphasize fine texture if sigma is too small.

    Attributes:
        sigma (float): Standard deviation for Gaussian derivatives used in Hessian
            computation. Default 1.0.
        mode (str): Boundary handling mode for Gaussian convolution. Default 'reflect'.
    """

    def __init__(
            self,
            sigma: float = 1.0,
            mode: str = 'reflect',
    ):
        """
        Parameters:
            sigma (float): Standard deviation for Gaussian derivatives. Smaller values
                (0.5-1.0) detect fine edges; larger values (2.0-4.0) detect broader
                structures. Default 1.0.
            mode (str): Boundary handling mode ('reflect', 'constant', 'nearest',
                'mirror', 'wrap'). Default 'reflect'.
        """
        self.sigma = sigma
        self.mode = mode

    def _operate(self, image: Image) -> Image:
        image.enh_gray[:] = hessian(
                image=image.enh_gray[:],
                sigma=self.sigma,
                mode=self.mode,
        )
        return image
