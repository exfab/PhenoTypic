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
    from phenotypic._core._image import Image

from ..abc_ import ImageEnhancer


class AnscombeInverse(ImageEnhancer):
    """Apply the inverse Generalized Anscombe Transform to restore original scale.

    Converts variance-stabilized data back to the [0, 1] intensity range
    using the closed-form approximation of the exact unbiased inverse. Always
    pair with :class:`AnscombeForward` in a pipeline, placing denoising
    operations between the forward and inverse transforms. Both must use
    identical parameter values.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Best For:
        - Completing an Anscombe-based denoising pipeline.
        - Restoring biologically meaningful intensities after GAT-domain
          denoising.
        - Fluorescence or low-light plate workflows that require
          variance-stabilized processing.

    Consider Also:
        - :class:`AnscombeForward` which must precede this operation.
        - :class:`BM3DDenoiser` for direct denoising without domain
          transforms.
        - :class:`NonLocalMeansDenoiser` for patch-based denoising that
          does not require a variance-stabilizing step.

    Args:
        gain: Camera gain in electrons per ADU. Must match the value
            used in :class:`AnscombeForward`. Default: 1.0.
        mu: Read noise mean (baseline offset). Must match the forward
            transform. Default: 0.0.
        sigma: Read noise standard deviation. Must match the forward
            transform. Default: 0.0.
        scale_factor: Converts counts back to normalized [0,1] range.
            Must match the forward transform. If ``None`` (default),
            auto-detects from image metadata.

    Returns:
        Image: Input image with ``detect_mat`` restored to [0, 1]
        intensity range. ``rgb`` and ``gray`` are unchanged.

    Raises:
        ValueError: If ``gain`` <= 0, ``sigma`` < 0, or ``scale_factor`` <= 0.

    References:
        [1] F. J. Anscombe, "The transformation of Poisson, binomial and
        negative-binomial data," *Biometrika*, vol. 35, no. 3/4,
        pp. 246--254, Dec. 1948.

        [2] M. Makitalo and A. Foi, "Optimal inversion of the generalized
        Anscombe transformation for Poisson-Gaussian noise," *IEEE Trans.
        Image Process.*, vol. 22, no. 1, pp. 91--103, Jan. 2013.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a
        visual walkthrough of enhancement pipelines on plate images.
        :doc:`/explanation/what_enhancement_does` for background on
        variance-stabilizing transforms and denoising strategies.
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
