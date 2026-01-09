from __future__ import annotations
from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image

from skimage.filters import frangi

from phenotypic.abc_ import ImageEnhancer


class FrangiVesselness(ImageEnhancer):
    """
    Frangi vesselness filter to detect elongated structures and colony branches.

    Computes the Frangi vesselness measure using the Hessian matrix eigenvalues
    to enhance line-like structures (thin filamentous colonies, branching patterns,
    mycelial networks). On agar plate images, this highlights elongated features
    and is particularly useful for organisms that form filamentous or branching
    morphologies (e.g., fungi, Bacillus).

    Use cases (agar plates):
    - Enhance branching filaments or network-like colonies for better detection.
    - Detect thin, elongated structures that global thresholding might miss.
    - Improve segmentation of interconnected mycelial networks or biofilms.
    - Preprocess images from fungal cultures that form visible hyphae.

    Tuning and effects:
    - sigmas: Range of standard deviations for Gaussian derivatives. Smaller sigmas
      detect finer structures; larger sigmas detect thicker features. Should span
      the expected range of colony feature scales (e.g., sigmas=range(1, 10, 2)
      for 1-10 pixel width structures).
    - alpha, beta: Sensitivity parameters for vesselness computation. Lower values
      make detection more permissive; higher values are stricter. Typical range:
      0.5-1.0 for both.
    - gamma: Background suppression parameter. Larger values suppress low-curvature
      regions (agar background). Typical range: 10-20.
    - black_ridges: If True, detect dark ridges (colonies) on bright background.
      If False, detect bright ridges on dark background. For plate imaging, use
      True when colonies appear darker than agar.

    Guidance:
    - Combine with mild smoothing (GaussianBlur) beforehand to suppress noise;
      Frangi is sensitive to high-frequency artifacts, especially with small sigmas.
    - Start with alpha=beta=0.5, gamma=15 and adjust based on colony morphology.
    - Choose sigmas that match expected feature widths (e.g., typical colony radius).

    Caveats:
    - Output is a vesselness map (probability measure), not binary. Thresholding is
      typically needed afterward.
    - Computationally expensive for large sigma ranges; consider limiting to 3-5
      different scales for speed.
    - May amplify noise or agar texture if gamma is too small or sigmas are too small.

    Attributes:
        sigmas (tuple | list): Sequence of standard deviations for Hessian
            computation. Each sigma represents a scale; default (1, 2, 3).
        alpha (float): Blobness parameter (0 to 1). Default 0.5.
        beta (float): Structuredness parameter (0 to 1). Default 0.5.
        gamma (float): Background suppression threshold. Default 15.
        black_ridges (bool): If True, detect dark structures on bright background;
            if False, detect bright structures on dark background. Default True.
    """

    def __init__(
            self,
            sigmas: Iterable[float] = (0.5, 1, 1.5),
            alpha: float = 0.5,
            beta: float = 0.5,
            gamma: float = None,
            black_ridges: bool = False,
    ):
        """
        Parameters:
            sigmas (tuple | list): Sequence of standard deviations for Gaussian
                derivatives. Smaller values detect finer features, larger values
                detect thicker structures. Default (0.5, 1, 1.5).
            alpha (float): Vesselness sensitivity to blobness. Lower values are
                more permissive. Range: 0 to 1. Default 0.5.
            beta (float): Vesselness sensitivity to structuredness. Lower values
                are more permissive. Range: 0 to 1. Default is None which uses
                half of the max Hessian norm.
            gamma (float): Threshold for background suppression. Larger values
                suppress low-curvature regions more aggressively. Default 15.
            black_ridges (bool): If True, detect dark ridges (colonies) on bright
                background. If False, detect bright ridges on dark background.
                For agar plates with dark colonies on light background, use True.
                Default True.
        """
        self.sigmas = sigmas
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.black_ridges = black_ridges

    def _operate(self, image: Image) -> Image:
        image.enh_gray[:] = frangi(
                image=image.enh_gray[:],
                sigmas=self.sigmas,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                black_ridges=self.black_ridges,
        )
        return image
