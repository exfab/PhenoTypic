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


class AnscombeForward(ImageEnhancer):
    """Apply the forward Generalized Anscombe Transform for variance stabilization.

    Converts Poisson-Gaussian noise into approximately Gaussian noise by
    applying a variance-stabilizing square-root transformation to ``detect_mat``.
    After this transform, standard Gaussian denoisers (wavelets, BM3D, bilateral
    filters) work effectively on the stabilized signal. Always pair with
    :class:`AnscombeInverse` in a pipeline, with denoising operations between
    them; both must use identical parameter values.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Args:
        gain: Camera gain in electrons per ADU. Typical range: 0.1--10.0.
            Default: 1.0.
        mu: Read noise mean (baseline offset). Typical range: 0.0--50.0.
            Default: 0.0.
        sigma: Read noise standard deviation. Set to 0 for pure Poisson
            noise. Typical range: 0.0--10.0. Default: 0.0.
        scale_factor: Converts normalized [0,1] data to counts. If ``None``
            (default), auto-detects from image metadata: 255 for 8-bit,
            65535 for 16-bit.

    Returns:
        Image: Input image with ``detect_mat`` in variance-stabilized
        (sqrt-scaled) domain. ``rgb`` and ``gray`` are unchanged.

    Raises:
        ValueError: If ``gain`` <= 0, ``sigma`` < 0, or ``scale_factor`` <= 0.

    Best For:
        - Low-light or fluorescence plate images with photon-counting noise.
        - Images from CCD/CMOS sensors where noise is Poisson-dominated.
        - Enabling Gaussian denoisers on data with signal-dependent noise.

    Consider Also:
        - :class:`BM3DDenoiser` for direct denoising without domain transforms.
        - :class:`BilateralDenoise` when edge preservation matters more than
          noise model accuracy.
        - :class:`VisuShrinkEnhancer` for wavelet denoising without a
          variance-stabilizing step.

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
        data = image.detect_mat[:] * scale_factor
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
        y = x * gain
        y += (gain ** 2) * 3.0 / 8.0 + sigma ** 2 - gain * mu
        np.maximum(y, 0.0, out=y)
        np.sqrt(y, out=y)
        y *= 2.0 / gain
        return y
