# Generalized Anscombe Transform implementation adapted from:
#   pymultiscale (https://github.com/broxtronix/pymultiscale)
#   Author: Michael Broxton (broxtronix)
#
# Reference:
#   M. Makitalo and A. Foi, "Optimal Inversion of the Generalized Anscombe
#   Transformation for Poisson-Gaussian Noise", IEEE Trans. Image Process., 2013.

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from phenotypic import Image

from ..abc_ import ImageEnhancer


class AnscombeInverse(ImageEnhancer):
    """Inverse Generalized Anscombe Transform to restore original scale.

    Applies the closed-form approximation of the exact unbiased inverse of
    the Generalized Anscombe Transform. This converts variance-stabilized
    data back to the original intensity scale after denoising in the GAT
    domain.

    Args:
        gain: Camera gain in electrons per ADU. Default 1.0.
        mu: Read noise mean (baseline offset). Default 0.0.
        sigma: Read noise standard deviation. Default 0.0 (pure Poisson).
        scale_factor: Converts normalized [0,1] data to counts. If None
            (default), auto-detects from image metadata: 255 for 8-bit,
            65535 for 16-bit, or falls back to 255.

    Returns:
        Image with detect_mat restored to [0, 1] intensity range.

    Raises:
        ValueError: If gain <= 0, sigma < 0, or scale_factor <= 0.

    Always use ``AnscombeInverse`` paired with :class:`AnscombeForward` in a
    pipeline. Both must use identical ``gain``, ``mu``, ``sigma``, and
    ``scale_factor`` values. Place denoising operations between Forward and
    Inverse.

    Any intermediate operations that have a ``clip`` parameter **must** set
    ``clip=False``, because the GAT domain produces values outside [0, 1]
    (typically ~1-32 for 8-bit images). Clipping would destroy the
    transformed signal.

    Attributes:
        gain (float): Camera gain (electrons/ADU). Default 1.0.
        mu (float): Read noise mean. Default 0.0.
        sigma (float): Read noise standard deviation. Default 0.0.
        scale_factor (float | None): Converts normalized data to counts.

    References:
        - Generalized Anscombe Transform implementation adapted from
          pymultiscale (https://github.com/broxtronix/pymultiscale) by
          Michael Broxton (broxtronix).
        - M. Makitalo and A. Foi, "Optimal Inversion of the Generalized
          Anscombe Transformation for Poisson-Gaussian Noise", IEEE Trans.
          Image Process., 2013.

    Examples:
        Pair with AnscombeForward in a pipeline for Poisson noise handling:

        >>> from phenotypic import ImagePipeline
        >>> from phenotypic.enhance import (
        ...     AnscombeForward, AnscombeInverse, GaussianBlur,
        ...     BilateralDenoise,
        ... )
        >>> pipeline = ImagePipeline([
        ...     AnscombeForward(gain=1.0, sigma=0.0, scale_factor=255.0),
        ...     GaussianBlur(sigma=1.0),
        ...     BilateralDenoise(sigma_spatial=10, clip=False),
        ...     AnscombeInverse(gain=1.0, sigma=0.0, scale_factor=255.0),
        ... ])

        Apply to a colony plate image:

        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from phenotypic.enhance import (
        ...     AnscombeForward, AnscombeInverse, GaussianBlur,
        ... )
        >>> from phenotypic import ImagePipeline
        >>> image = load_synth_yeast_plate()
        >>> pipeline = ImagePipeline([
        ...     AnscombeForward(gain=1.0, sigma=0.0, scale_factor=255.0),
        ...     GaussianBlur(sigma=1.0),
        ...     AnscombeInverse(gain=1.0, sigma=0.0, scale_factor=255.0),
        ... ])
        >>> result = pipeline.apply(image)
    """

    def __init__(
            self,
            gain: float = 1.0,
            mu: float = 0.0,
            sigma: float = 0.0,
            scale_factor: float | None = None,
    ):
        """
        Parameters:
            gain (float): Camera gain in electrons per ADU. Must match the
                value used in AnscombeForward. Default 1.0.
            mu (float): Read noise mean. Must match the value used in
                AnscombeForward. Default 0.0.
            sigma (float): Read noise standard deviation. Must match the
                value used in AnscombeForward. Default 0.0.
            scale_factor (float | None): Converts counts back to normalized
                [0,1] range. Must match the value used in AnscombeForward.
                If None (default), auto-detects from image metadata.
        """
        if gain <= 0:
            raise ValueError(f"gain must be > 0, got {gain}")
        if sigma < 0:
            raise ValueError(f"sigma must be >= 0, got {sigma}")
        if scale_factor is not None and scale_factor <= 0:
            raise ValueError(
                f"scale_factor must be > 0, got {scale_factor}"
            )

        self.gain = float(gain)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.scale_factor = (
            float(scale_factor) if scale_factor is not None else None
        )

    def _get_scale_factor(self, image: Image) -> float:
        """Get scale factor, auto-detecting from image metadata.

        Args:
            image: The Image to get scale factor for.

        Returns:
            Scale factor for converting counts back to normalized [0,1].
        """
        if self.scale_factor is not None:
            return self.scale_factor

        bit_depth = getattr(image.metadata, "bit_depth", None)
        if bit_depth == 8:
            return 255.0
        elif bit_depth == 16:
            return 65535.0
        else:
            return 255.0

    def _operate(self, image: Image) -> Image:
        """Apply inverse GAT to restore detect_mat to [0, 1]."""
        scale_factor = self._get_scale_factor(image)
        denoised = self._inverse_generalized_anscombe(
            image.detect_mat[:], self.mu, self.sigma, self.gain
        )
        image.detect_mat[:] = (denoised / scale_factor).clip(0.0, 1.0)
        return image

    @staticmethod
    def _inverse_generalized_anscombe(
            x: np.ndarray,
            mu: float,
            sigma: float,
            gain: float = 1.0,
    ) -> np.ndarray:
        """Inverse Generalized Anscombe Transform (closed-form).

        Applies the closed-form approximation of the exact unbiased inverse
        of the Generalized Anscombe variance-stabilizing transformation.

        The input signal x is transformed back into counts based on the
        assumption that the original signal follows the Poisson-Gaussian
        noise model:
            x = gain * p + n

        Args:
            x: Variance-stabilized array (from forward transform).
            mu: Read noise mean.
            sigma: Read noise standard deviation.
            gain: Camera gain (electrons/ADU). Default 1.0.

        Returns:
            Reconstructed array in counts domain.

        Reference:
            https://github.com/broxtronix/pymultiscale
        """
        test = np.maximum(x, 1.0)
        inv_test = np.reciprocal(test)

        result = test * test
        result *= 0.25                                          # (test/2)^2

        result += (0.25 * np.sqrt(1.5)) * inv_test              # test^-1 term

        inv_test_sq = inv_test * inv_test
        result -= 1.375 * inv_test_sq                            # test^-2 term

        result += (0.625 * np.sqrt(1.5)) * (inv_test_sq * inv_test)  # test^-3

        result -= 0.125 + sigma ** 2
        np.maximum(result, 0.0, out=result)
        result *= gain
        result += mu
        np.nan_to_num(result, nan=0.0, copy=False)
        return result
