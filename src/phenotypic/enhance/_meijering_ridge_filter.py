from __future__ import annotations
from typing import Iterable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image

from skimage.filters import meijering

from phenotypic.abc_ import ImageEnhancer


class MeijeringRidgeFilter(ImageEnhancer):
    """
    Meijering neuriteness filter to detect fine ridge-like structures and delicate features.

    Computes the Meijering neuriteness measure using the Hessian matrix eigenvalues
    to enhance elongated, thread-like structures (neurites, delicate filaments, thin
    wrinkles, and river-like features). On agar plate images, this highlights fine
    ridge-like features and is particularly useful for organisms that form delicate
    filamentous or network-like morphologies (e.g., actinomycetes, fungal hyphae,
    bacterial networks).

    Use cases (agar plates):
    - Enhance delicate filamentous structures too thin for standard detection.
    - Detect fine wrinkles, grooves, or network features in biofilms.
    - Improve segmentation of sparse mycelial networks or bacterial filaments.
    - Preprocess images from organisms with fine, branching morphologies.

    Tuning and effects:
    - sigmas: Range of standard deviations for Gaussian derivatives. Smaller sigmas
      detect finer structures; larger sigmas detect thicker features. Should span
      the expected range of colony feature scales (e.g., sigmas=range(1, 5)
      for 1-5 pixel width structures, or range(3, 10) for thicker features).
    - alpha: Shape parameter controlling sensitivity to elongated vs. blob-like
      structures. Default None uses -1/(ndim+1). For 2D images, this becomes -1/3.
      Larger (less negative) values detect more blob-like features; smaller
      (more negative) values are stricter about linearity. Rarely requires manual
      tuning.
    - black_ridges: If True, detect dark ridges (colonies) on bright background.
      If False, detect bright ridges on dark background. For plate imaging, use
      True when colonies appear darker than agar.
    - mode: Boundary handling when computing derivatives ('constant', 'reflect',
      'wrap', 'nearest', 'mirror'). Default 'reflect' usually works best.
    - cval: Fill value for 'constant' mode (default 0).

    Guidance:
    - Combine with mild smoothing (GaussianBlur) beforehand to suppress noise;
      Meijering is sensitive to high-frequency artifacts, especially with small sigmas.
    - Start with sigmas=range(1, 5) for detecting fine filaments.
    - Leave alpha=None unless you have specific knowledge of structure linearity.
    - Choose sigmas smaller than for Sato when targeting very fine features.
    - Meijering is often more selective than Sato; use when you want to isolate
      thin, well-separated ridge structures.

    Caveats:
    - Output is a neuriteness map (probability measure), not binary. Thresholding is
      typically needed afterward.
    - Computationally expensive for large sigma ranges; consider limiting to 3-5
      different scales for speed.
    - May miss thick, blob-like structures; use Frangi for mixed blob/tube detection.
    - Less responsive than Frangi to blobness/structuredness tuning; use Frangi
      if you need finer control over feature characteristics.
    - Due to edge effects, results may be cropped by ~4 pixels (scikit-image behavior).

    Attributes:
        sigmas (tuple | list): Sequence of standard deviations for Hessian
            computation. Each sigma represents a scale; default (1, 2, 3).
        alpha (float | None): Shape parameter for elongated feature sensitivity.
            Default None uses -1/(ndim+1) for 2D images.
        black_ridges (bool): If True, detect dark structures on bright background;
            if False, detect bright structures on dark background. Default True.
        mode (str): Boundary handling mode ('constant', 'reflect', 'wrap',
            'nearest', 'mirror'). Default 'reflect'.
        cval (float): Fill value for 'constant' mode. Default 0.
    """

    def __init__(
            self,
            sigmas: Iterable[float] = (1, 2, 3),
            alpha: Optional[float] = None,
            black_ridges: bool = True,
            mode: str = 'reflect',
            cval: float = 0,
    ):
        """
        Parameters:
            sigmas (tuple | list): Sequence of standard deviations for Gaussian
                derivatives. Smaller values detect finer features, larger values
                detect thicker structures. Default (1, 2, 3).
            alpha (float | None): Shape parameter controlling linearity sensitivity.
                Default None uses -1/(ndim+1), which for 2D images is -1/3.
                Unlikely to require manual tuning in practice.
            black_ridges (bool): If True, detect dark ridges (colonies) on bright
                background. If False, detect bright ridges on dark background.
                For agar plates with dark colonies on light background, use True.
                Default True.
            mode (str): How to handle image boundaries. Options: 'constant'
                (pad with cval), 'reflect' (mirror), 'wrap' (tile), 'nearest'
                (replicate edge), 'mirror' (symmetric mirror). Default 'reflect'.
            cval (float): Fill value when mode='constant'. Default 0.
        """
        self.sigmas = sigmas
        self.alpha = alpha
        self.black_ridges = black_ridges
        self.mode = mode
        self.cval = cval

    def _operate(self, image: Image) -> Image:
        image.detect_mat[:] = meijering(
                image=image.detect_mat[:],
                sigmas=self.sigmas,
                alpha=self.alpha,
                black_ridges=self.black_ridges,
                mode=self.mode,
                cval=self.cval,
        )
        return image
