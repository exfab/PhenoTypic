from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from phenotypic import Image

from ..abc_ import ImageEnhancer


def homomorphic_filter(
    array: np.ndarray,
    sigma: float = 200.0,
    gamma_low: float = 0.5,
    gamma_high: float = 1.5,
    eps: float = 1e-6,
) -> np.ndarray:
    """Homomorphic filter for illumination correction on grayscale arrays.

    Args:
        array: Grayscale input, shape ``(H, W)``, float, range [0, 1].
        sigma: Gaussian sigma controlling the illumination/reflectance
            frequency cutoff. Larger values suppress broader illumination
            gradients while leaving colony-scale reflectance intact.
        gamma_low: Gain applied to low-frequency (illumination) component.
            Values < 1 suppress illumination variation across the plate.
        gamma_high: Gain applied to high-frequency (reflectance) component.
            Values > 1 enhance surface detail and colony contrast.
        eps: Small constant added before ``log`` to avoid ``log(0)``.

    Returns:
        Corrected image as float array, clipped to [0, 1].

    Assumes the observed image is a product of illumination and
    reflectance: ``I(x,y) = L(x,y) * R(x,y)``.  Taking the logarithm
    converts the multiplicative relationship into an additive one, allowing
    a Gaussian low-pass to separate the slowly varying illumination from the
    high-frequency reflectance.  Differential gains (``gamma_low``,
    ``gamma_high``) are then applied before exponentiating back to the
    linear domain.

    Processing steps:

    1. ``log_image = log(array + eps)``
    2. Low-pass via ``cv2.GaussianBlur`` (kernel size = ``int(6*sigma+1)``,
       forced odd).
    3. ``high_pass = log_image - low_pass``
    4. ``filtered_log = gamma_low * low_pass + gamma_high * high_pass``
    5. ``result = clip(exp(filtered_log) - eps, 0, 1)``

    Examples:
        >>> import numpy as np
        >>> arr = np.random.default_rng(0).random((64, 64)).astype(np.float32)
        >>> from phenotypic.enhance._homomorphic_filter import homomorphic_filter
        >>> out = homomorphic_filter(arr, sigma=10.0)
        >>> 0.0 <= out.min() and out.max() <= 1.0
        True
    """
    log_image = np.log(array.astype(np.float32) + eps)

    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1

    low_pass = cv2.GaussianBlur(
        log_image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma,
    )
    high_pass = log_image - low_pass

    filtered_log = gamma_low * low_pass + gamma_high * high_pass

    result = np.exp(filtered_log) - eps
    return np.clip(result, 0.0, 1.0)


class HomomorphicFilter(ImageEnhancer):
    """Homomorphic filtering for illumination correction on agar plate images.

    Separates illumination (low-frequency) and reflectance (high-frequency)
    components in the log domain, applies differential gains to suppress
    illumination gradients while boosting colony detail, then returns to the
    linear domain.  This is especially useful when plates suffer from
    vignetting, uneven scanner lighting, or shadow gradients that confound
    global thresholding.

    Args:
        sigma: Gaussian sigma for the illumination/reflectance frequency
            cutoff.  Should be large enough that the low-pass captures only
            the illumination field, not individual colonies.  A good starting
            point is several times the largest colony diameter.
        gamma_low: Gain for low frequencies (illumination).  Values < 1
            suppress slow illumination variation.
        gamma_high: Gain for high frequencies (reflectance).  Values > 1
            enhance colony surface detail and contrast.
        eps: Small constant added before ``log`` to avoid ``log(0)``.

    Returns:
        Image: Modified image with illumination-corrected ``detect_mat``
        clipped to [0, 1].

    Use cases (agar plates):
        - Correct vignetting or uneven scanner illumination so that
          colonies at the plate edge have comparable intensity to those
          in the centre.
        - Pre-processing before global thresholding (Otsu, triangle) to
          reduce sensitivity to illumination gradients.
        - Enhance reflectance detail on translucent or faintly pigmented
          colonies.

    Parameter effects:
        - **sigma:** Controls the spatial scale of the illumination
          estimate.  Too small and colony signal leaks into the low-pass,
          reducing their contrast.  Too large and fine illumination
          gradients are missed.  For a 4000 px plate scan, 150-300 is a
          reasonable range.
        - **gamma_low:** Decreasing below 1.0 progressively flattens
          illumination variation.  Setting to 0 removes illumination
          entirely (may produce unnatural results).
        - **gamma_high:** Increasing above 1.0 boosts colony contrast
          and surface texture.  Very high values amplify noise.
        - **eps:** Rarely needs adjustment; prevents numerical issues
          with zero-valued pixels.

    Caveats:
        - Works best on single-channel (grayscale) ``detect_mat``.
          Multichannel data is processed channel-by-channel by
          ``cv2.GaussianBlur``, which may not correspond to true
          illumination decomposition in colour space.
        - Very large ``sigma`` values produce very large blur kernels
          (kernel side = ``6*sigma + 1``), which increases computation
          time.

    Examples:
        Basic illumination correction:

        >>> from phenotypic.enhance import HomomorphicFilter
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> image = load_synth_yeast_plate()
        >>> enhancer = HomomorphicFilter(sigma=200.0)
        >>> result = enhancer.apply(image)
        >>> result.detect_mat[:].min() >= 0.0
        True

        In a detection pipeline:

        >>> from phenotypic import ImagePipeline
        >>> from phenotypic.enhance import HomomorphicFilter
        >>> from phenotypic.detect import OtsuDetector
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> image = load_synth_yeast_plate()
        >>> pipeline = ImagePipeline([
        ...     HomomorphicFilter(sigma=200.0),
        ...     OtsuDetector(),
        ... ])
        >>> result = pipeline.apply(image)
    """

    def __init__(
        self,
        sigma: float = 200.0,
        gamma_low: float = 0.5,
        gamma_high: float = 1.5,
        eps: float = 1e-6,
    ):
        """
        Parameters:
            sigma: Gaussian sigma for the illumination/reflectance cutoff.
                Larger values capture broader illumination gradients.  Start
                with a value several times the largest colony diameter.
            gamma_low: Gain for low frequencies (illumination).  < 1
                suppresses illumination variation; 1.0 leaves it unchanged.
            gamma_high: Gain for high frequencies (reflectance).  > 1
                enhances colony detail; 1.0 leaves it unchanged.
            eps: Offset to avoid ``log(0)``.  Rarely needs adjustment.
        """
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self.sigma = sigma
        self.gamma_low = gamma_low
        self.gamma_high = gamma_high
        self.eps = eps

    def _operate(self, image: Image) -> Image:
        image.detect_mat[:] = homomorphic_filter(
            array=image.detect_mat[:],
            sigma=self.sigma,
            gamma_low=self.gamma_low,
            gamma_high=self.gamma_high,
            eps=self.eps,
        )
        return image
