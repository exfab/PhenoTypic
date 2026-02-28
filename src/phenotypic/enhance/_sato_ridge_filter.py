from __future__ import annotations
from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

from phenotypic.abc_ import ImageEnhancer


class SatoRidgeFilter(ImageEnhancer):
    """
    Sato tubeness filter to detect continuous ridge-like structures and colonies.

    Computes the Sato tubeness measure using the Hessian matrix eigenvalues
    to enhance continuous ridge structures (tubes, wrinkles, filamentous colonies,
    mycelial networks, and river-like features). On agar plate images, this highlights
    tube-like or ridge-like features and is particularly useful for organisms that form
    filamentous or branching morphologies (e.g., fungi, Bacillus, streptomycetes).

    Use cases (agar plates):
    - Enhance thin filamentous colonies or mycelial networks for better detection.
    - Detect continuous ridge-like structures that global thresholding might miss.
    - Improve segmentation of interconnected fungal networks or biofilm structures.
    - Preprocess images from filamentous organisms (molds, hyphae, root-like colonies).

    Tuning and effects:
    - sigmas: Range of standard deviations for Gaussian derivatives. Smaller sigmas
      detect finer structures; larger sigmas detect thicker features. Should span
      the expected range of colony feature scales (e.g., sigmas=range(1, 10, 2)
      for 1-10 pixel width structures).
    - black_ridges: If True, detect dark ridges (colonies) on bright background.
      If False, detect bright ridges on dark background. For plate imaging, use
      True when colonies appear darker than agar.
    - mode: Boundary handling when computing derivatives ('constant', 'reflect',
      'wrap', 'nearest', 'mirror'). Default 'reflect' usually works best.
    - cval: Fill value for 'constant' mode (default 0).

    Guidance:
    - Combine with mild smoothing (GaussianBlur) beforehand to suppress noise;
      Sato is sensitive to high-frequency artifacts, especially with small sigmas.
    - Start with sigmas=range(1, 5) for small features or range(3, 10) for thick tubes.
    - Choose sigmas that match expected feature widths (e.g., typical colony radius).
    - Sato typically has less sensitivity to parameter tuning compared to Frangi,
      making it a good first choice for ridge detection.

    Caveats:
    - Output is a tubeness map (probability measure), not binary. Thresholding is
      typically needed afterward.
    - Computationally expensive for large sigma ranges; consider limiting to 3-5
      different scales for speed.
    - May amplify noise if sigmas are too small; pre-smoothing helps.
    - Less responsive than Frangi to blobness/structuredness tuning; use Frangi
      if you need finer control over blob vs. tube detection.

    Attributes:
        sigmas (tuple | list): Sequence of standard deviations for Hessian
            computation. Each sigma represents a scale; default (1, 2, 3).
        black_ridges (bool): If True, detect dark structures on bright background;
            if False, detect bright structures on dark background. Default False.
        mode (str): Boundary handling mode ('constant', 'reflect', 'wrap',
            'nearest', 'mirror'). Default 'reflect'.
        cval (float): Fill value for 'constant' mode. Default 0.
    """

    def __init__(
            self,
            sigmas: Iterable[float] = (1, 2, 3),
            black_ridges: bool = False,
            mode: str = 'reflect',
            cval: float = 0,
    ):
        """
        Parameters:
            sigmas (tuple | list): Sequence of standard deviations for Gaussian
                derivatives. Smaller values detect finer features, larger values
                detect thicker structures. Default (1, 2, 3).
            black_ridges (bool): If True, detect dark ridges (colonies) on bright
                background. If False, detect bright ridges on dark background.
                For agar plates with dark colonies on light background, use True.
                Default False.
            mode (str): How to handle image boundaries. Options: 'constant'
                (pad with cval), 'reflect' (mirror), 'wrap' (tile), 'nearest'
                (replicate edge), 'mirror' (symmetric mirror). Default 'reflect'.
            cval (float): Fill value when mode='constant'. Default 0.
        """
        self.sigmas = sigmas
        self.black_ridges = black_ridges
        self.mode = mode
        self.cval = cval

    def _operate(self, image: Image) -> Image:
        # Manual Sato tubeness loop (replaces skimage.filters.sato) for
        # explicit deletion of Hessian intermediates, reducing peak memory
        # by ~50% for multi-sigma runs. Algorithm: Sato et al. 1998, eqs. 9/22.
        img = np.asarray(image.detect_mat[:], dtype=np.float32)

        if not self.black_ridges:
            img = -img

        filtered_max = np.zeros(img.shape, dtype=np.float32)

        for sigma in self.sigmas:
            hessian_elems = hessian_matrix(
                    img, sigma=sigma, mode=self.mode, cval=self.cval,
                    use_gaussian_derivatives=True,
            )
            eigvals = hessian_matrix_eigvals(hessian_elems)
            del hessian_elems

            eigvals = eigvals[:-1]
            filtered = (
                    sigma ** 2
                    * np.prod(np.maximum(eigvals, 0), axis=0) ** (1.0 / len(eigvals))
            )
            del eigvals
            np.maximum(filtered_max, filtered, out=filtered_max)
            del filtered

        image.detect_mat[:] = filtered_max
        return image
