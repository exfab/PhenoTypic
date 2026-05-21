from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from skimage.restoration import denoise_wavelet

from ..abc_ import ImageDenoiser
from ..tools_.mixin import _GATSupportMixin


class VisuShrinkEnhancer(_GATSupportMixin, ImageDenoiser):
    """Denoise ``detect_mat`` with universal VisuShrink wavelet thresholding.

    Applies wavelet-domain denoising with a single universal threshold across
    all subbands, designed to remove all Gaussian noise with high probability.
    Faster than :class:`BayesShrinkEnhancer` but may over-smooth regions with
    low noise. Preserves colony edges better than Gaussian blur.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Args:
        sigma: Noise standard deviation in [0, 1] scale. ``None`` (default)
            auto-estimates via MAD. Typical range: 0.01--0.05 for moderate
            scanner/camera noise. Too high causes over-smoothing. Ignored
            when ``use_gat=True`` (the stabilized-domain value 1.0 is used
            internally).
        wavelet: Wavelet family. ``'db2'`` (default) balances smoothness and
            locality; ``'db4'`` captures more detail. Must be orthogonal.
        mode: Thresholding mode. ``'soft'`` (default) produces smoother
            results for additive noise; ``'hard'`` preserves edges more.
        wavelet_levels: Decomposition depth. ``None`` (default) uses max-3
            automatically. Higher values give finer denoising.
        clip: Clip output to [0, 1]. Default: ``True``. Automatically
            deferred when ``use_gat=True``.
        rescale_sigma: skimage's internal rescale flag. Default ``True``.
            Automatically forced to ``False`` when ``use_gat=True``.
        use_gat: Wrap denoising in the Generalized Anscombe Transform.
            Default: ``False``. See
            :class:`phenotypic.tools_.mixin._GATSupportMixin`.
        gat_gain, gat_mu, gat_read_sigma, gat_scale_factor: GAT parameters.

    Returns:
        Image: Input image with ``detect_mat`` denoised via universal
        wavelet thresholding. ``rgb`` and ``gray`` are unchanged.

    Best For:
        - Scanner banding and flatbed scanner noise removal.
        - High-ISO camera images where colony boundaries must remain sharp.
        - Agar granularity and condensation speckle suppression before
          detection.
        - Pre-filtering before edge detection to avoid noise amplification.
        - Poisson-Gaussian noise via ``use_gat=True``.

    Consider Also:
        - :class:`BayesShrinkEnhancer` for adaptive thresholding that
          preserves more detail in regions with varying noise levels.
        - :class:`BM3DDenoiser` for state-of-the-art structured noise
          removal at higher computational cost.
        - :class:`BilateralDenoise` for edge-preserving smoothing without
          wavelet decomposition.

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

    _GAT_NOISE_PARAMS: ClassVar[dict[str, float]] = {"sigma": 1.0}
    _GAT_DEFER_ATTRS: ClassVar[tuple[str, ...]] = ("clip", "rescale_sigma")

    sigma: float | None = None
    wavelet: str = "db2"
    mode: Literal["soft", "hard"] = "soft"
    wavelet_levels: int | None = None
    clip: bool = True
    rescale_sigma: bool = True

    def _operate(self, image: Image) -> Image:
        """Apply VisuShrink wavelet denoising to detection matrix."""
        self._gat_apply(image, "detect_mat", self._denoise_detect_mat)
        return image

    def _denoise_detect_mat(self, image: Image) -> None:
        denoised = denoise_wavelet(
                image=image.detect_mat[:],
                sigma=self.sigma,
                wavelet=self.wavelet,
                mode=self.mode,
                wavelet_levels=self.wavelet_levels,
                method="VisuShrink",
                channel_axis=None,
                rescale_sigma=self.rescale_sigma,
        )
        if self.clip:
            denoised = denoised.clip(0.0, 1.0)
        image.detect_mat[:] = denoised
