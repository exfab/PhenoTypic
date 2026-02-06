# Generalized Anscombe Transform implementation adapted from:
#   pymultiscale (https://github.com/broxtronix/pymultiscale)
#   Author: Michael Broxton (broxtronix)
#
# Reference:
#   M. Makitalo and A. Foi, "Optimal Inversion of the Generalized Anscombe
#   Transformation for Poisson-Gaussian Noise", IEEE Trans. Image Process., 2013.

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
    from phenotypic import Image, ImagePipeline

from ..abc_ import ImageEnhancer
from ..tools_.mixin import ClipControlMixin


class AnscombeTransformDenoise(ClipControlMixin, ImageEnhancer):
    """
    Variance-stabilizing denoising using the Generalized Anscombe Transform for Poisson-Gaussian noise.

    Applies the Generalized Anscombe Transform (GAT) to convert Poisson-Gaussian noise
    (common in photon-counting imaging, fluorescence microscopy, and low-light photography)
    into approximately Gaussian noise. This enables standard Gaussian denoisers (wavelets,
    BM3D, bilateral filters) to work effectively on images with photon noise. After
    denoising in the transformed domain, the inverse GAT restores the original intensity
    scale while preserving denoised signal.

    Use cases (agar plates):
    - Low-light or fluorescence imaging where photon counting (Poisson) noise dominates.
    - Scientific cameras (CCD, sCMOS) with known gain and read noise parameters.
    - Images from imaging systems with mixed Poisson-Gaussian noise characteristics.
    - Combining powerful Gaussian denoisers (BM3D, wavelets) with Poisson noise.
    - Pre-processing for colony detection when photon noise degrades edge definition.

    Tuning and effects:
    - inner_enhancer: The denoiser applied in variance-stabilized domain. Use BM3DDenoiser,
      VisuShrinkEnhancer, or BayesShrinkEnhancer for strong denoising. Use GaussianBlur for
      simple smoothing. Can be a single ImageEnhancer or an entire ImagePipeline.
    - gain: Camera gain in electrons per ADU (analog-to-digital unit). Higher gain means
      larger signal but also amplified noise. Default 1.0 assumes unity gain.
    - mu: Read noise mean (offset). For calibrated cameras, typically near 0. Non-zero
      values shift the baseline intensity level.
    - sigma: Read noise standard deviation. Set to 0 for pure Poisson noise (ideal photon
      counting). Increase for cameras with significant read noise (e.g., 1-5 for CCD).
    - scale_factor: Converts normalized [0,1] image data to counts for the transform.
      If None (default), auto-detects from image metadata: 255 for 8-bit, 65535 for 16-bit.
      Manual override supported. Incorrect scale_factor reduces denoising effectiveness.

    Caveats:
    - Requires camera calibration: Best results need accurate gain, mu, sigma values
      from camera characterization. Default values assume pure Poisson noise.
    - The inner enhancer operates on transformed (sqrt-scaled) values, not [0,1]. Some
      enhancers with hardcoded thresholds may need parameter adjustment.
    - Computational overhead: Two transforms plus inner enhancer add processing time.
      Not needed if noise is already approximately Gaussian.
    - For very low counts (< 5 photons/pixel), the GAT approximation becomes less accurate.
    - The inverse transform uses a closed-form approximation, not exact inversion.

    Attributes:
        inner_enhancer (ImageEnhancer | ImagePipeline): The denoiser applied after variance
            stabilization. Any operation with .apply(image) interface.
        gain (float): Camera gain (electrons/ADU). Default 1.0.
        mu (float): Read noise mean. Default 0.0.
        sigma (float): Read noise standard deviation. Default 0.0 (pure Poisson).
        scale_factor (float | None): Converts normalized data to counts. Default None
            (auto-detect from image metadata).

    References:
        - Generalized Anscombe Transform implementation adapted from pymultiscale
          (https://github.com/broxtronix/pymultiscale) by Michael Broxton (broxtronix).
        - M. Makitalo and A. Foi, "Optimal Inversion of the Generalized Anscombe
          Transformation for Poisson-Gaussian Noise", IEEE Trans. Image Process., 2013.

    Examples:
        Basic usage with wavelet denoiser for low-light colony images:

        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from phenotypic.enhance import AnscombeTransformDenoise, VisuShrinkEnhancer
        >>> image = load_synth_yeast_plate()
        >>> # Apply GAT + wavelet denoising for photon noise
        >>> denoiser = AnscombeTransformDenoise(
        ...     inner_enhancer=VisuShrinkEnhancer(sigma=None),
        ...     gain=1.0,
        ...     sigma=0.0,  # Pure Poisson noise
        ...     scale_factor=255.0  # 8-bit image
        ... )
        >>> denoised = denoiser.apply(image)
        >>> # Original rgb/gray unchanged, detect_mat denoised

        Using with ImagePipeline as inner enhancer:

        >>> from phenotypic import ImagePipeline
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from phenotypic.enhance import (
        ...     AnscombeTransformDenoise, GaussianBlur, BilateralDenoise
        ... )
        >>> image = load_synth_yeast_plate()
        >>> # Create multi-step inner denoiser
        >>> inner_pipeline = ImagePipeline([
        ...     GaussianBlur(sigma=0.5),
        ...     BilateralDenoise(sigma_spatial=10)
        ... ])
        >>> # Wrap in GAT for Poisson noise handling
        >>> gat_denoiser = AnscombeTransformDenoise(
        ...     inner_enhancer=inner_pipeline,
        ...     gain=1.0,
        ...     sigma=0.0,
        ...     scale_factor=255.0
        ... )
        >>> result = gat_denoiser.apply(image)

        Pipeline integration for complete preprocessing:

        >>> from phenotypic import ImagePipeline
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from phenotypic.enhance import (
        ...     AnscombeTransformDenoise, GaussianBlur, CLAHE
        ... )
        >>> from phenotypic.detect import OtsuDetector
        >>> image = load_synth_yeast_plate()
        >>> # Build pipeline with GAT denoising
        >>> pipeline = ImagePipeline([
        ...     AnscombeTransformDenoise(
        ...         inner_enhancer=GaussianBlur(sigma=1.0),
        ...         gain=1.0,
        ...         sigma=0.0,
        ...         scale_factor=255.0
        ...     ),
        ...     CLAHE(clip_limit=0.02),
        ...     OtsuDetector()
        ... ])
        >>> result = pipeline.apply(image)
        >>> colonies = result.objects
    """

    def __init__(
            self,
            inner_enhancer: Union[ImageEnhancer, "ImagePipeline"] = None,
            gain: float = 1.0,
            mu: float = 0.0,
            sigma: float = 0.0,
            scale_factor: float | None = None,
    ):
        """
        Parameters:
            inner_enhancer (ImageEnhancer | ImagePipeline): The denoiser to apply in the
                variance-stabilized domain. Must have a callable .apply(image, inplace=bool)
                method. Use BM3DDenoiser, VisuShrinkEnhancer, BayesShrinkEnhancer for strong
                denoising, or GaussianBlur for simple smoothing. Can be a single ImageEnhancer
                or an entire ImagePipeline for multi-step denoising.
            gain (float): Camera gain in electrons per ADU. Higher gain amplifies both signal
                and noise. Default 1.0 assumes unity gain. Obtain from camera calibration
                or manufacturer specs for best results.
            mu (float): Read noise mean (baseline offset). For calibrated cameras, typically
                near 0. Non-zero values account for detector bias. Default 0.0.
            sigma (float): Read noise standard deviation. Set to 0 for pure Poisson noise
                (ideal photon-counting sensors). Increase for cameras with significant
                read noise (e.g., 1-5 for CCD sensors). Default 0.0.
            scale_factor (float | None): Converts normalized [0,1] image data to counts
                for the transform. If None (default), auto-detects from image metadata:
                uses 255 for 8-bit origin, 65535 for 16-bit, or falls back to 255.
                Set manually if auto-detection fails or for raw count data (use 1.0).
        """
        # Validate inner_enhancer has apply method (duck typing)
        if inner_enhancer is None:
            from phenotypic.enhance import NonLocalMeansDenoiser

            inner_enhancer = NonLocalMeansDenoiser(
                    patch_size=3,
                    search_distance=5
            )

        if not hasattr(inner_enhancer, "apply") or not callable(inner_enhancer.apply):
            raise TypeError(
                    f"inner_enhancer must be an ImageEnhancer or ImagePipeline with an "
                    f"apply() method, got {type(inner_enhancer).__name__}"
            )

        # Validate gain
        if gain <= 0:
            raise ValueError(f"gain must be > 0, got {gain}")

        # Validate sigma (read noise std)
        if sigma < 0:
            raise ValueError(f"sigma must be >= 0, got {sigma}")

        # Validate scale_factor if provided
        if scale_factor is not None and scale_factor <= 0:
            raise ValueError(f"scale_factor must be > 0, got {scale_factor}")

        self.inner_enhancer = inner_enhancer
        self.gain = float(gain)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.scale_factor = float(scale_factor) if scale_factor is not None else None

    def _get_scale_factor(self, image: Image) -> float:
        """Get scale factor, auto-detecting from image metadata if not specified.

        Args:
            image: The Image to get scale factor for.

        Returns:
            Scale factor for converting normalized [0,1] data to counts.
        """
        if self.scale_factor is not None:
            return self.scale_factor

        # Auto-detect from image metadata
        bit_depth = getattr(image.metadata, "bit_depth", None)
        if bit_depth == 8:
            return 255.0
        elif bit_depth == 16:
            return 65535.0
        else:
            # Default to 255 for 8-bit assumption
            return 255.0

    def _operate(self, image: Image) -> Image:
        """Apply GAT denoising: forward transform -> inner enhancer -> inverse transform.

        The GAT produces sqrt-scaled values (typically 1-32 for 8-bit images). To preserve
        correct denoising behavior, we disable output clipping on the inner enhancer using
        ClipControlMixin._disable_clipping(). This ensures the inner enhancer operates
        in the GAT domain without losing data to [0,1] clipping.
        """
        # 0. Get scale factor (auto-detect or manual)
        scale_factor = self._get_scale_factor(image)

        # 1. Scale detect_mat to counts domain
        data = image.detect_mat[:].copy() * scale_factor

        # 2. Apply forward Generalized Anscombe Transform
        transformed = self._generalized_anscombe(data, self.mu, self.sigma, self.gain)

        # 3. Store GAT-transformed data in detect_mat
        image.detect_mat[:] = transformed

        # 4. Create clip-disabled copy of inner enhancer
        # This ensures the inner enhancer preserves GAT-scale values (~1-32)
        clip_disabled_enhancer = self._disable_clipping(self.inner_enhancer)

        # 5. Apply clip-disabled inner enhancer
        # Check if the enhancer supports reset parameter (ImagePipeline does)
        # and pass reset=False to prevent resetting detect_mat to original grayscale
        apply_params = {"inplace": True}
        sig = inspect.signature(clip_disabled_enhancer.apply)
        if "reset" in sig.parameters:
            apply_params["reset"] = False

        image = clip_disabled_enhancer.apply(image, **apply_params)

        # 6. Apply inverse Generalized Anscombe Transform
        denoised = self._inverse_generalized_anscombe(
                image.detect_mat[:], self.mu, self.sigma, self.gain
        )

        # 7. Scale back to [0, 1] range and clip
        image.detect_mat[:] = (denoised / scale_factor).clip(0.0, 1.0)

        return image

    @staticmethod
    def _generalized_anscombe(
            x: np.ndarray, mu: float, sigma: float, gain: float = 1.0
    ) -> np.ndarray:
        """Forward Generalized Anscombe Transform (variance stabilization).

        Compute the generalized Anscombe variance stabilizing transform, which assumes
        that the data provided to it is a mixture of Poisson and Gaussian noise.

        The input signal x is assumed to follow the Poisson-Gaussian noise model:
            x = gain * p + n
        where gain is the camera gain and mu and sigma are the read noise mean and
        standard deviation.

        We assume that x contains only positive values. Values that are less than or
        equal to 0 are handled by taking the maximum with 0.

        Note: This transform will show some bias for counts less than about 20.

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

    @staticmethod
    def _inverse_generalized_anscombe(
            x: np.ndarray, mu: float, sigma: float, gain: float = 1.0
    ) -> np.ndarray:
        """Inverse Generalized Anscombe Transform (closed-form approximation).

        Applies the closed-form approximation of the exact unbiased inverse of the
        Generalized Anscombe variance-stabilizing transformation.

        The input signal x is transformed back into a Poisson random variable based
        on the assumption that the original signal from which it was derived follows
        the Poisson-Gaussian noise model:
            x = gain * p + n
        where gain is the camera gain and mu and sigma are the read noise mean and
        standard deviation.

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
        exact_inverse = (
                np.power(test / 2.0, 2.0)
                + 1.0 / 4.0 * np.sqrt(3.0 / 2.0) * np.power(test, -1.0)
                - 11.0 / 8.0 * np.power(test, -2.0)
                + 5.0 / 8.0 * np.sqrt(3.0 / 2.0) * np.power(test, -3.0)
                - 1.0 / 8.0
                - sigma ** 2
        )
        exact_inverse = np.maximum(0.0, exact_inverse)
        exact_inverse *= gain
        exact_inverse += mu
        # Handle NaN values (from division by zero or other numerical issues)
        exact_inverse = np.where(np.isnan(exact_inverse), 0.0, exact_inverse)
        return exact_inverse
