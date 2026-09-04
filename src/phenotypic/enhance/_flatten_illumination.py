from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

# `cv2` is imported inside the methods that use it. A module-scope import
# would load OpenCV on `import phenotypic` for every entry point, though only
# a run reaching this operation needs it.
import numpy as np
from pydantic import Field

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from ..abc_ import BackgroundSubtraction
from ..sdk_.typing_ import TuneSpec


class FlattenIllumination(BackgroundSubtraction):
    """Correct uneven illumination in ``detect_mat`` using homomorphic frequency-domain filtering.

    Separates the illumination (low-frequency) and reflectance (high-frequency)
    components in the log domain, applies differential gains to suppress
    brightness gradients while boosting colony detail, then exponentiates back
    to the linear domain. The result is clipped to [0.0, 1.0]. Particularly
    effective for plates with vignetting, scanner lighting bands, or shadow
    gradients where spatial-domain subtraction is insufficient.

    For how enhancement fits into the pipeline, see
    :doc:`/explanation/what_enhancement_does`.

    Best For:
        - Plates with visible vignetting or radial brightness falloff from
          DSLR or flatbed scanner optics.
        - Scanner images with horizontal or vertical lighting bands across
          the plate surface.
        - Plates where uneven agar thickness causes variable background
          brightness.
        - Pre-conditioning before global thresholding on unevenly lit plates.

    Consider Also:
        - :class:`SubtractGaussian` for a simpler spatial-domain background
          subtraction when the illumination gradient is smooth and mild.
        - :class:`EnhanceLocalContrast` when the problem is local contrast
          variation rather than a large-scale illumination gradient.
        - :class:`SubtractRollingBall` for morphological background estimation
          when the gradient is driven by colony proximity to the plate edge.

    Args:
        sigma: Gaussian sigma controlling the spatial scale of the estimated
            illumination field in pixels. Must be large enough to span entire
            colonies so the low-pass captures only the gradient, not colony
            signal. A reasonable starting point is several times the diameter
            of the largest colony. Typical range: 40--300 (resolution-dependent).
            Default: 200.0.
        gamma_low: Gain applied to the illumination (low-frequency) component.
            Values below 1.0 suppress illumination variation; 1.0 leaves it
            unchanged. Typical range: 0.3--0.8. Default: 0.5.
        gamma_high: Gain applied to the reflectance (high-frequency) component.
            Values above 1.0 enhance colony contrast and surface detail; 1.0
            leaves it unchanged. Typical range: 1.0--2.5. Default: 1.5.
        eps: Small additive constant before the logarithm to prevent log(0)
            on zero-valued pixels. Typical range: 1e-8--1e-4. Rarely needs
            adjustment for standard [0, 1] normalised images. Default: 1e-6.

    Returns:
        Image: Input image with ``detect_mat`` illumination-corrected and
        clipped to [0.0, 1.0]. ``rgb`` and ``gray`` are unchanged.

    Raises:
        ValueError: If ``sigma`` is not positive.

    See Also:
        :doc:`/how_to/notebooks/enhance_low_contrast` for a comparison of
        contrast and illumination correction methods on real plate images.
        :doc:`/explanation/what_enhancement_does` for how homomorphic filtering
        separates illumination from reflectance.
    """

    # TODO: review bound (unverified vs literature)
    sigma: Annotated[float, TuneSpec(40.0, 300.0, log=True)] = Field(200.0, gt=0.0)
    # TODO: review bound (unverified vs literature)
    gamma_low: Annotated[float, TuneSpec(0.3, 0.8)] = 0.5
    # TODO: review bound (unverified vs literature)
    gamma_high: Annotated[float, TuneSpec(1.0, 2.5)] = 1.5
    eps: Annotated[float, TuneSpec(tunable=False)] = 1e-6

    @staticmethod
    def _filter(
        array: np.ndarray,
        sigma: float = 200.0,
        gamma_low: float = 0.5,
        gamma_high: float = 1.5,
        eps: float = 1e-6,
    ) -> np.ndarray:
        """Apply homomorphic filtering to a single grayscale array.

        Separates illumination and reflectance components in the log domain,
        applies differential gains, and returns corrected image in linear domain.

        Args:
            array (np.ndarray): Grayscale image, shape (H, W), dtype float32,
                range [0.0, 1.0]. Supports both single-channel (H, W) and
                multichannel (H, W, C) arrays; multichannel is processed
                per-channel independently by cv2.GaussianBlur.
            sigma (float): Gaussian sigma for the low-pass filter (illumination
                estimation). Larger values suppress broader illumination
                gradients. Kernel size is int(6*sigma + 1), forced to be odd.
                Typical range: 20–300 pixels depending on image resolution and
                illumination artifact scale. Default is 200.0.
            gamma_low (float): Gain for low-frequency (illumination) component.
                Values < 1.0 suppress illumination variation; 1.0 is no change.
                Default is 0.5.
            gamma_high (float): Gain for high-frequency (reflectance) component.
                Values > 1.0 enhance colony detail; 1.0 is no change. Default is
                1.5.
            eps (float): Small constant added before logarithm to prevent log(0).
                Default is 1e-6. Rarely needs adjustment; use smaller values
                only if working with normalized or very low-intensity arrays.

        Returns:
            np.ndarray: Corrected image, shape matching input, dtype float32,
            range [0.0, 1.0]. Clipped to ensure valid range.

        Raises:
            ValueError: If sigma <= 0 (caught in __init__, not here).

        Processing Pipeline:

        1. Convert to float32 and compute log domain: log_image = log(array + eps)
        2. Estimate illumination field via Gaussian blur:
           - Kernel size: int(6*sigma + 1), forced odd
           - low_pass = cv2.GaussianBlur(log_image, (ksize, ksize), sigma, sigma)
        3. Compute reflectance residual: high_pass = log_image - low_pass
        4. Apply differential gains: filtered_log = gamma_low * low_pass + gamma_high * high_pass
        5. Exponentiate and clip: result = clip(exp(filtered_log) - eps, 0.0, 1.0)

        Notes:

        - This is a static method, so it can be called independently for testing
          or external use.
        - Multichannel arrays (H, W, C) are processed per-channel by
          cv2.GaussianBlur, which may not produce physically correct illumination
          decomposition. For color-aware processing, consider converting to
          single-channel (e.g., grayscale or LAB L-channel) before filtering.
        - Very large sigma values increase computation time and memory usage
          (kernel side = 6*sigma + 1 can exceed 1200 pixels). For high-resolution
          images, consider downsampling or reducing sigma.
        """
        log_image = np.log(array.astype(np.float32) + eps)

        ksize = int(6 * sigma + 1)
        if ksize % 2 == 0:
            ksize += 1

        import cv2

        low_pass = cv2.GaussianBlur(
            log_image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma,
        )
        high_pass = log_image - low_pass

        filtered_log = gamma_low * low_pass + gamma_high * high_pass

        result = np.exp(filtered_log) - eps
        return np.clip(result, 0.0, 1.0)

    def _operate(self, image: Image) -> Image:
        image.detect_mat[:] = self._filter(
            array=image.detect_mat[:],
            sigma=self.sigma,
            gamma_low=self.gamma_low,
            gamma_high=self.gamma_high,
            eps=self.eps,
        )
        return image
