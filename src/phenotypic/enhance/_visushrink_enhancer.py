from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from skimage.restoration import denoise_wavelet

from ..abc_ import ImageEnhancer


class VisuShrinkEnhancer(ImageEnhancer):
    """Denoise ``detect_mat`` with universal VisuShrink wavelet thresholding.

    Applies wavelet-domain denoising with a single universal threshold across
    all subbands, designed to remove all Gaussian noise with high probability.
    Faster than :class:`BayesShrinkEnhancer` but may over-smooth regions with
    low noise. Preserves colony edges better than Gaussian blur.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Best For:
        - Scanner banding and flatbed scanner noise removal.
        - High-ISO camera images where colony boundaries must remain sharp.
        - Agar granularity and condensation speckle suppression before
          detection.
        - Pre-filtering before edge detection to avoid noise amplification.

    Consider Also:
        - :class:`BayesShrinkEnhancer` for adaptive thresholding that
          preserves more detail in regions with varying noise levels.
        - :class:`BM3DDenoiser` for state-of-the-art structured noise
          removal at higher computational cost.
        - :class:`BilateralDenoise` for edge-preserving smoothing without
          wavelet decomposition.

    Args:
        sigma: Noise standard deviation in [0, 1] scale. ``None`` (default)
            auto-estimates via MAD. Typical range: 0.01--0.05 for moderate
            scanner/camera noise. Too high causes over-smoothing.
        wavelet: Wavelet family. ``'db2'`` (default) balances smoothness and
            locality; ``'db4'`` captures more detail. Must be orthogonal.
        mode: Thresholding mode. ``'soft'`` (default) produces smoother
            results for additive noise; ``'hard'`` preserves edges more.
        wavelet_levels: Decomposition depth. ``None`` (default) uses max-3
            automatically. Higher values give finer denoising.
        clip: Clip output to [0, 1]. Default: ``True``. Set to ``False``
            when using with variance-stabilizing transforms (e.g., GAT).

    Returns:
        Image: Input image with ``detect_mat`` denoised via universal
        wavelet thresholding. ``rgb`` and ``gray`` are unchanged.

    References:
        [1] D. L. Donoho and I. M. Johnstone, "Ideal spatial adaptation by
        wavelet shrinkage," *Biometrika*, vol. 81, no. 3, pp. 425--455,
        Sep. 1994.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a
        visual walkthrough of denoising pipelines on plate images.
        :doc:`/explanation/what_enhancement_does` for background on
        wavelet denoising and threshold selection strategies.
    """

    def __init__(
            self,
            sigma: float | None = None,
            wavelet: str = "db2",
            mode: Literal["soft", "hard"] = "soft",
            wavelet_levels: int | None = None,
            clip: bool = True,
    ):
        """Initialize VisuShrink wavelet denoiser.

        Parameters:
            sigma (float | None): Noise standard deviation in [0, 1] scale. None
                (default) auto-estimates via median absolute deviation (MAD).
                For reference: 8-bit noise σ=10/255 ≈ 0.04 in normalized scale.
                Typical values: 0.01-0.05 for moderate scanner/camera noise.
                Start with auto-estimation, then tune if needed.
            wavelet (str): Wavelet type from PyWavelets. 'db2' (default) is a
                good general choice. 'db4' for more detail, 'sym2' for symmetry.
                Must be orthogonal (db*, sym*) for proper noise handling.
            mode (Literal['soft', 'hard']): Threshold type. 'soft' (default)
                produces smoother results for additive noise. 'hard' preserves
                edges more but may leave noise artifacts.
            wavelet_levels (int | None): Decomposition depth. None (default)
                uses max-3 automatically. Higher = finer denoising, slower.
            clip (bool): Whether to clip output to [0, 1] range. Default True.
                Set to False when using with variance-stabilizing transforms
                (e.g., GAT) that require preserving the original scale.
        """
        self.sigma = sigma
        self.wavelet = wavelet
        self.mode = mode
        self.wavelet_levels = wavelet_levels
        self.clip = clip

    def _operate(self, image: Image) -> Image:
        """Apply VisuShrink wavelet denoising to detection matrix.

        Returns:
            Modified Image with denoised detect_mat
        """
        denoised = denoise_wavelet(
                image=image.detect_mat[:],
                sigma=self.sigma,
                wavelet=self.wavelet,
                mode=self.mode,
                wavelet_levels=self.wavelet_levels,
                method="VisuShrink",
                channel_axis=None,
                rescale_sigma=True,
        )
        if self.clip:
            denoised = denoised.clip(0.0, 1.0)
        image.detect_mat[:] = denoised
        return image
