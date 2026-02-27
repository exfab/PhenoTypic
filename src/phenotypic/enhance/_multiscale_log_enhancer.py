from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image

import numpy as np
from scipy.ndimage import gaussian_laplace

from ..abc_ import ImageEnhancer

_SQRT2 = np.sqrt(2.0)


def multiscale_log_enhance(
    array: np.ndarray,
    min_radius: float = 3.0,
    max_radius: float = 12.0,
    num_scales: int = 12,
) -> np.ndarray:
    """Multi-scale Laplacian of Gaussian blob enhancement.

    Args:
        array (numpy.ndarray): 2-D grayscale array (float, typically 0-1).
        min_radius (float): Smallest target blob radius in pixels.
            The corresponding Gaussian sigma is ``min_radius / sqrt(2)``.
        max_radius (float): Largest target blob radius in pixels.
        num_scales (int): Number of logarithmically spaced scales between
            ``min_radius`` and ``max_radius``.

    Returns:
        numpy.ndarray: Scale-normalised LoG response, same shape as *array*.
            Each pixel holds the maximum absolute LoG response across all scales,
            multiplied by sigma squared for scale normalisation.  All values are
            non-negative.

    Raises:
        ValueError: If ``min_radius >= max_radius`` or ``num_scales < 1``.

    The Laplacian of Gaussian (LoG) is a classical blob detector.  At each scale
    sigma the raw LoG response is multiplied by sigma squared so that blobs of
    different sizes produce comparable peak magnitudes.  A max-projection across
    scales then selects the strongest response at every pixel, regardless of the
    blob size that produced it.

    Examples:
        >>> import numpy as np
        >>> from phenotypic.enhance._multiscale_log_enhancer import (
        ...     multiscale_log_enhance,
        ... )
        >>> rng = np.random.default_rng(0)
        >>> arr = rng.random((64, 64))
        >>> out = multiscale_log_enhance(arr)
        >>> out.shape
        (64, 64)
        >>> out.min() >= 0.0
        True
    """
    if min_radius <= 0:
        raise ValueError(f"min_radius must be positive, got {min_radius}")
    if min_radius >= max_radius:
        raise ValueError(
            f"min_radius ({min_radius}) must be less than max_radius ({max_radius})"
        )
    if num_scales < 1:
        raise ValueError(f"num_scales must be >= 1, got {num_scales}")

    min_sigma = min_radius / _SQRT2
    max_sigma = max_radius / _SQRT2
    sigmas = np.geomspace(min_sigma, max_sigma, num_scales)

    enhanced = np.zeros_like(array)
    for sigma in sigmas:
        log_response = gaussian_laplace(array, sigma=sigma)
        scale_norm = sigma ** 2 * np.abs(log_response)
        np.maximum(enhanced, scale_norm, out=enhanced)

    return enhanced


class MultiscaleLoGEnhancer(ImageEnhancer):
    """Multi-scale Laplacian of Gaussian blob enhancement for colony detection.

    Applies scale-normalised LoG filtering across a geometric series of
    Gaussian sigmas and returns the maximum response at each pixel.  Bright
    blob-like structures (colonies, inocula, droplets) on a darker background
    produce strong peaks regardless of their individual radii, making this a
    useful preprocessing step before thresholding or GMM-based segmentation.

    Use cases (agar plates):
    - Enhance round or near-round colony inocula before detection, making faint
      spots more visible relative to the agar background.
    - Detect inoculation spots across a range of sizes in a single pass without
      knowing the exact colony diameter in advance.
    - Pre-filter before a GMM or threshold step to accentuate blob-shaped
      features and suppress gradual illumination gradients.

    Tuning and effects:
    - min_radius / max_radius: Set the radius window (in pixels) that brackets
      the expected colony sizes.  Only blobs whose radii fall in this range
      will produce strong LoG responses.  Sigma is derived as radius / sqrt(2).
    - num_scales: More scales give finer resolution in blob size but cost more
      computation (one ``gaussian_laplace`` call per scale).  12 scales is a
      good default; reduce to 4-6 for speed when the size range is narrow.

    Caveats:
    - The absolute magnitude of the output depends on image contrast and blob
      sharpness.  Normalise or follow with a percentile-based threshold if
      downstream steps expect a fixed intensity range.
    - Very large ``max_radius`` values increase the effective kernel size
      (proportional to sigma) and can be slow on large images.
    - The LoG operator is isotropic; elongated structures (streaks, hyphae)
      are better handled by ridge filters (Frangi, Meijering).

    Attributes:
        min_radius (float): Smallest target blob radius in pixels.
        max_radius (float): Largest target blob radius in pixels.
        num_scales (int): Number of logarithmically spaced scales.

    Examples:
        Enhancing colony inocula on an agar plate:

        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from phenotypic.enhance import MultiscaleLoGEnhancer
        >>> image = load_synth_yeast_plate()
        >>> enhancer = MultiscaleLoGEnhancer(min_radius=3.0, max_radius=12.0)
        >>> enhanced = enhancer.apply(image)
        >>> enhanced.detect_mat.shape == image.detect_mat.shape
        True

        Inside a pipeline with subsequent detection:

        >>> from phenotypic import ImagePipeline
        >>> from phenotypic.enhance import MultiscaleLoGEnhancer
        >>> from phenotypic.detect import OtsuDetector
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> pipeline = ImagePipeline([
        ...     MultiscaleLoGEnhancer(min_radius=2.0, max_radius=8.0, num_scales=8),
        ...     OtsuDetector(),
        ... ])
        >>> image = load_synth_yeast_plate()
        >>> result = pipeline.apply(image)
        >>> result.objmask is not None
        True
    """

    def __init__(
        self,
        min_radius: float = 3.0,
        max_radius: float = 12.0,
        num_scales: int = 12,
    ):
        """
        Parameters:
            min_radius (float): Smallest target blob radius in pixels.  The
                corresponding sigma is ``min_radius / sqrt(2)``.  Blobs smaller
                than this will produce weaker responses.
            max_radius (float): Largest target blob radius in pixels.
                Blobs larger than this will also produce weaker responses.
            num_scales (int): Number of logarithmically spaced sigma values
                between ``min_radius / sqrt(2)`` and ``max_radius / sqrt(2)``.
                More scales give finer size resolution at the cost of speed.
        """
        if min_radius <= 0:
            raise ValueError(f"min_radius must be positive, got {min_radius}")
        if min_radius >= max_radius:
            raise ValueError(
                f"min_radius ({min_radius}) must be less than "
                f"max_radius ({max_radius})"
            )
        if num_scales < 1:
            raise ValueError(f"num_scales must be >= 1, got {num_scales}")

        self.min_radius = float(min_radius)
        self.max_radius = float(max_radius)
        self.num_scales = int(num_scales)

    def _operate(self, image: Image) -> Image:
        image.detect_mat[:] = multiscale_log_enhance(
            array=image.detect_mat[:],
            min_radius=self.min_radius,
            max_radius=self.max_radius,
            num_scales=self.num_scales,
        )
        return image
