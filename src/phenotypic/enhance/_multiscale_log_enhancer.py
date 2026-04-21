from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
from scipy.ndimage import gaussian_laplace

from ..abc_ import ImageEnhancer

_SQRT2 = np.sqrt(2.0)


class MultiscaleLoGEnhancer(ImageEnhancer):
    """Enhance blob-like colonies in ``detect_mat`` with scale-normalised Laplacian of Gaussian.

    Applies LoG filtering across a geometric series of Gaussian sigmas and
    returns the maximum response at each pixel. Bright blob-like structures
    (colonies, inocula, droplets) produce strong peaks regardless of size,
    making this a robust preprocessing step before thresholding or
    GMM-based segmentation.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Args:
        min_radius: Smallest target blob radius in pixels. Blobs smaller
            than this produce weaker responses. Typical range: 1.0--5.0
            at 512x768 resolution. Scale proportionally for higher
            resolutions. Default: 3.0.
        max_radius: Largest target blob radius in pixels. Blobs larger
            than this also produce weaker responses. Typical range:
            8.0--50.0 at 512x768 resolution. Default: 12.0.
        num_scales: Number of logarithmically spaced sigma values. More
            scales improve size discrimination at higher compute cost.
            Typical range: 4--20. Default: 12.

    Returns:
        Image: Input image with ``detect_mat`` replaced by the
        scale-normalised LoG response map. ``rgb`` and ``gray`` are
        unchanged.

    Raises:
        ValueError: If ``min_radius`` <= 0, ``min_radius`` >= ``max_radius``,
            or ``num_scales`` < 1.

    Best For:
        - Mixed-size colonies on mature plates where small emerging and
          large mature colonies must both be detected.
        - Sparse inoculation spots that are faint and nearly invisible
          against the agar background.
        - Low-contrast or shadowed regions where LoG emphasizes blob
          structure over absolute intensity.
        - Preprocessing before thresholding to sharpen blob boundaries
          and suppress uneven illumination.

    Consider Also:
        - :class:`SatoRidgeFilter` for elongated or filamentous structures
          where LoG's isotropic assumption is a poor fit.
        - :class:`LaplaceEnhancer` for simpler single-scale edge detection.
        - :class:`SubtractGaussian` when the primary issue is illumination
          gradients rather than blob enhancement.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a
        visual walkthrough of blob enhancement on plate images.
        :doc:`/explanation/what_enhancement_does` for background on
        scale-space blob detection and LoG theory.
    """

    def __init__(
        self,
        min_radius: float = 3.0,
        max_radius: float = 12.0,
        num_scales: int = 12,
    ):
        """Initialize MultiscaleLoGEnhancer with radius range and scale density.

        Args:
            min_radius (float): Smallest target blob radius in pixels. The
                corresponding Gaussian sigma is ``min_radius / sqrt(2)``. Blobs
                smaller than this radius produce weaker LoG responses. Typical
                range: 1.0–5.0 pixels at 512×768 resolution. Default: 3.0.
            max_radius (float): Largest target blob radius in pixels. The
                corresponding Gaussian sigma is ``max_radius / sqrt(2)``. Blobs
                larger than this radius also produce weaker responses. Typical
                range: 8.0–50.0 pixels at 512×768 resolution. Default: 12.0.
            num_scales (int): Number of logarithmically spaced sigma values
                between ``min_radius / sqrt(2)`` and ``max_radius / sqrt(2)``.
                Controls size-discrimination resolution. Larger values give finer
                blob size resolution but increase computation (one LoG evaluation
                per scale). Typical range: 4–20. Default: 12.

        Raises:
            ValueError: If min_radius <= 0, min_radius >= max_radius, or
                num_scales < 1.
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

    @staticmethod
    def _enhance(
        array: np.ndarray,
        min_radius: float = 3.0,
        max_radius: float = 12.0,
        num_scales: int = 12,
    ) -> np.ndarray:
        """Multi-scale Laplacian of Gaussian blob enhancement (core kernel).

        Applies scale-normalised Laplacian of Gaussian (LoG) filtering across a
        geometric series of Gaussian sigmas and returns the maximum response at each
        pixel. This is the core computation called by _operate() and available for
        direct use on standalone arrays.

        Args:
            array (numpy.ndarray): 2-D grayscale array, shape (height, width).
                Typically normalized to [0, 1] or [0, 255]. dtype should be float.
            min_radius (float): Smallest target blob radius in pixels. The
                corresponding Gaussian sigma is ``min_radius / sqrt(2)``. Blobs
                smaller than this produce weaker responses. Default: 3.0.
            max_radius (float): Largest target blob radius in pixels. The
                corresponding Gaussian sigma is ``max_radius / sqrt(2)``. Blobs
                larger than this also produce weaker responses. Default: 12.0.
            num_scales (int): Number of logarithmically spaced sigma values
                between ``min_radius / sqrt(2)`` and ``max_radius / sqrt(2)``.
                More scales provide finer size resolution at cost of speed.
                Default: 12.

        Returns:
            numpy.ndarray: Scale-normalised LoG response, same shape and dtype as
                *array*. Each pixel contains the maximum absolute LoG response across
                all scales, multiplied by sigma squared for scale normalization. All
                values are non-negative (≥ 0.0). Output range depends on input
                contrast and is typically [0, max_response] where max_response varies
                with image content.

        Raises:
            ValueError: If min_radius <= 0, min_radius >= max_radius, or
                num_scales < 1.

        Notes:
            At each scale σ, the Laplacian of Gaussian produces a response
            proportional to local curvature. The response is scaled by σ² so that
            blobs of different sizes produce comparable peak magnitudes. A max
            projection across scales selects the strongest response at each pixel,
            yielding size-invariant blob detection: a 3-pixel and a 12-pixel blob
            both produce strong peaks if they fall within [min_radius, max_radius].

            This function is the low-level kernel; for image processing via the
            PhenoTypic pipeline, use the MultiscaleLoGEnhancer class instead.

        Examples:
            Direct kernel use on a random array:

            >>> import numpy as np
            >>> from phenotypic.enhance._multiscale_log_enhancer import (
            ...     MultiscaleLoGEnhancer,
            ... )
            >>> rng = np.random.default_rng(0)
            >>> arr = rng.random((64, 64))
            >>> out = MultiscaleLoGEnhancer._enhance(arr)
            >>> out.shape
            (64, 64)
            >>> out.min() >= 0.0
            True

            LoG enhancement on a synthetic image with known blob:

            >>> import numpy as np
            >>> from phenotypic.enhance._multiscale_log_enhancer import (
            ...     MultiscaleLoGEnhancer,
            ... )
            >>> # Create a simple blob (Gaussian)
            >>> y, x = np.ogrid[:100, :100]
            >>> blob = np.exp(-((x - 50)**2 + (y - 50)**2) / 100.0)
            >>> out = MultiscaleLoGEnhancer._enhance(blob, min_radius=3, max_radius=12)
            >>> peak_pos = np.unravel_index(out.argmax(), out.shape)
            >>> abs(peak_pos[0] - 50) <= 2 and abs(peak_pos[1] - 50) <= 2
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

    def _operate(self, image: Image) -> Image:
        image.detect_mat[:] = self._enhance(
            array=image.detect_mat[:],
            min_radius=self.min_radius,
            max_radius=self.max_radius,
            num_scales=self.num_scales,
        )
        return image
