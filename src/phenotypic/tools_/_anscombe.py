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


def gat_forward(
        x: np.ndarray,
        mu: float,
        sigma: float,
        gain: float = 1.0,
) -> np.ndarray:
    """Forward Generalized Anscombe Transform.

    Variance-stabilize a Poisson-Gaussian noisy signal so its residual noise
    is approximately Gaussian with unit variance, enabling Gaussian denoisers
    (BM3D, wavelet shrinkage, NLM, bilateral) to operate optimally.

    The input signal ``x`` follows the noise model::

        x = gain * p + n,   p ~ Poisson,   n ~ N(mu, sigma**2)

    Values of the transformed array can be inverted with :func:`gat_inverse`
    using the same parameters. Negative arguments to the square root are
    clamped to 0.

    Args:
        x: Input array in counts (i.e. pre-scaled, not [0, 1]). Not
            modified -- internal arithmetic operates on a fresh array.
        mu: Read-noise mean (baseline offset).
        sigma: Read-noise standard deviation.
        gain: Camera gain in electrons per ADU. Default 1.0.

    Returns:
        Variance-stabilized array (newly allocated).
    """
    y = x * gain
    y += (gain ** 2) * 3.0 / 8.0 + sigma ** 2 - gain * mu
    np.maximum(y, 0.0, out=y)
    np.sqrt(y, out=y)
    y *= 2.0 / gain
    return y


def gat_inverse(
        x: np.ndarray,
        mu: float,
        sigma: float,
        gain: float = 1.0,
) -> np.ndarray:
    """Inverse Generalized Anscombe Transform (closed-form unbiased).

    Closed-form approximation of the exact unbiased inverse of
    :func:`gat_forward` from Mäkitalo & Foi (TIP 2013). Restores the
    expected counts-domain signal from a denoised stabilized array.

    Args:
        x: Variance-stabilized array (output of :func:`gat_forward`,
            optionally passed through a Gaussian denoiser).
        mu: Read-noise mean. Must match the forward transform.
        sigma: Read-noise standard deviation. Must match the forward
            transform.
        gain: Camera gain. Must match the forward transform.

    Returns:
        Reconstructed array in counts-domain.
    """
    test = np.maximum(x, 1.0)
    inv_test = np.reciprocal(test)

    result = test * test
    result *= 0.25                                              # (test/2)^2

    result += (0.25 * np.sqrt(1.5)) * inv_test                  # test^-1 term

    inv_test_sq = inv_test * inv_test
    result -= 1.375 * inv_test_sq                                # test^-2 term

    result += (0.625 * np.sqrt(1.5)) * (inv_test_sq * inv_test)  # test^-3

    result -= 0.125 + sigma ** 2
    np.maximum(result, 0.0, out=result)
    result *= gain
    result += mu
    np.nan_to_num(result, nan=0.0, copy=False)
    return result


def resolve_scale_factor(
        image: Image,
        override: float | None,
) -> float:
    """Resolve the count-scale multiplier for an image.

    Converts normalized [0, 1] data to counts. Auto-detects from
    ``image.metadata.bit_depth`` (8-bit -> 255, 16-bit -> 65535).
    Defaults to 255 for unknown bit depth.

    Args:
        image: Image whose metadata provides the bit-depth hint.
        override: Explicit scale factor; takes precedence when not None.

    Returns:
        Multiplier converting [0, 1] floats to count values.
    """
    if override is not None:
        return override

    bit_depth = getattr(image.metadata, "bit_depth", None)
    if bit_depth == 8:
        return 255.0
    if bit_depth == 16:
        return 65535.0
    return 255.0
