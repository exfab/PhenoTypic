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


class AnscombeForward(ImageEnhancer):
    """Forward Generalized Anscombe Transform for variance stabilization.

    Applies the forward Generalized Anscombe Transform (GAT) to convert
    Poisson-Gaussian noise (common in photon-counting imaging, fluorescence
    microscopy, and low-light photography) into approximately Gaussian noise.
    After this transform, standard Gaussian denoisers (wavelets, BM3D,
    bilateral filters) work effectively on the stabilized signal.

    Args:
        gain: Camera gain in electrons per ADU. Default 1.0.
        mu: Read noise mean (baseline offset). Default 0.0.
        sigma: Read noise standard deviation. Default 0.0 (pure Poisson).
        scale_factor: Converts normalized [0,1] data to counts. If None
            (default), auto-detects from image metadata: 255 for 8-bit,
            65535 for 16-bit, or falls back to 255.

    Returns:
        Image with detect_mat in variance-stabilized (sqrt-scaled) domain.
        Values are typically in the range ~1-32 for 8-bit source images.

    Raises:
        ValueError: If gain <= 0, sigma < 0, or scale_factor <= 0.

    Always use ``AnscombeForward`` paired with :class:`AnscombeInverse` in a
    pipeline, with denoising operations between them. Both must use identical
    ``gain``, ``mu``, ``sigma``, and ``scale_factor`` values.

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
        Pair with AnscombeInverse in a pipeline for Poisson noise handling:

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
            gain (float): Camera gain in electrons per ADU. Higher gain
                amplifies both signal and noise. Default 1.0 assumes
                unity gain.
            mu (float): Read noise mean (baseline offset). For calibrated
                cameras, typically near 0. Default 0.0.
            sigma (float): Read noise standard deviation. Set to 0 for
                pure Poisson noise. Increase for cameras with significant
                read noise (e.g., 1-5 for CCD sensors). Default 0.0.
            scale_factor (float | None): Converts normalized [0,1] data
                to counts. If None (default), auto-detects from image
                metadata. Set manually if auto-detection fails or for
                raw count data (use 1.0).
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
            Scale factor for converting normalized [0,1] data to counts.
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
        """Apply forward GAT to variance-stabilize detect_mat."""
        scale_factor = self._get_scale_factor(image)
        data = image.detect_mat[:].copy() * scale_factor
        transformed = self._generalized_anscombe(
            data, self.mu, self.sigma, self.gain
        )
        image.detect_mat[:] = transformed
        return image

    @staticmethod
    def _generalized_anscombe(
            x: np.ndarray,
            mu: float,
            sigma: float,
            gain: float = 1.0,
    ) -> np.ndarray:
        """Forward Generalized Anscombe Transform.

        Compute the generalized Anscombe variance stabilizing transform,
        which assumes the data is a mixture of Poisson and Gaussian noise.

        The input signal x follows the Poisson-Gaussian noise model:
            x = gain * p + n
        where gain is the camera gain, mu and sigma are the read noise
        mean and standard deviation.

        Values less than or equal to 0 are handled by clamping to 0.

        Args:
            x: Input array (counts).
            mu: Read noise mean.
            sigma: Read noise standard deviation.
            gain: Camera gain (electrons/ADU). Default 1.0.

        Returns:
            Variance-stabilized array.

        Reference:
            https://github.com/broxtronix/pymultiscale
        """
        y = gain * x + (gain ** 2) * 3.0 / 8.0 + sigma ** 2 - gain * mu
        return (2.0 / gain) * np.sqrt(np.maximum(y, 0.0))
