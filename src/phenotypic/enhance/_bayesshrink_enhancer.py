from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from skimage.restoration import denoise_wavelet

from ..abc_ import ImageDenoiser
from ..tools_.mixin import _GATSupportMixin


class BayesShrinkEnhancer(_GATSupportMixin, ImageDenoiser):
    """Denoise ``detect_mat`` with adaptive BayesShrink wavelet thresholding.

    Applies wavelet-domain denoising with per-subband adaptive thresholds
    computed from local statistics. Preserves more fine detail than
    :class:`VisuShrinkEnhancer` by denoising aggressively only where noise
    is high and gently where signal dominates.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Args:
        sigma: Noise standard deviation in [0, 1] scale. ``None`` (default)
            auto-estimates via MAD. Typical range: 0.01--0.05 for moderate
            scanner/camera noise. Accurate estimation improves adaptive
            threshold quality. Ignored when ``use_gat=True`` (the
            stabilized-domain value 1.0 is used internally).
        wavelet: Wavelet family. ``'db2'`` (default) balances smoothness and
            locality; ``'db4'`` preserves finer details. Must be orthogonal.
        mode: Thresholding mode. ``'soft'`` (default) produces smoother
            results; ``'hard'`` preserves edges more aggressively.
        wavelet_levels: Decomposition depth. ``None`` (default) uses max-3
            automatically. Higher values allow finer noise/signal separation.
        clip: Clip output to [0, 1]. Default: ``True``. Automatically
            deferred when ``use_gat=True``.
        rescale_sigma: Rescale ``sigma`` to match the wavelet decomposition
            scale (skimage's internal flag). Default: ``True``. Automatically
            forced to ``False`` when ``use_gat=True`` because the stabilized
            domain is no longer in [0, 1].
        use_gat: Wrap denoising in the Generalized Anscombe Transform for
            Poisson-Gaussian noise. Default: ``False``. See
            :class:`phenotypic.tools_.mixin._GATSupportMixin`.
        gat_gain, gat_mu, gat_read_sigma, gat_scale_factor: GAT parameters
            (see ``_GATSupportMixin``).

    Returns:
        Image: Input image with ``detect_mat`` denoised via adaptive wavelet
        thresholding. ``rgb`` and ``gray`` are unchanged.

    Best For:
        - Images with spatially varying noise from uneven illumination.
        - Preserving colony texture and internal morphology during denoising.
        - Scanner noise and camera artifacts on plates where fine detail
          matters for downstream measurement.
        - Pre-filtering before feature extraction or texture analysis.
        - Poisson-Gaussian noise via ``use_gat=True``.

    Consider Also:
        - :class:`VisuShrinkEnhancer` for faster denoising with a universal
          threshold when spatial noise uniformity is acceptable.
        - :class:`BM3DDenoiser` for state-of-the-art denoising of structured
          noise patterns.
        - :class:`LocalEdgeDenoise` for edge-preserving smoothing without
          wavelet decomposition.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a
        visual walkthrough of enhancement pipelines on plate images.
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
        """Apply BayesShrink adaptive wavelet denoising to detection matrix."""
        self._gat_apply(image, "detect_mat", self._denoise_detect_mat)
        return image

    def _denoise_detect_mat(self, image: Image) -> None:
        denoised = denoise_wavelet(
                image=image.detect_mat[:],
                sigma=self.sigma,
                wavelet=self.wavelet,
                mode=self.mode,
                wavelet_levels=self.wavelet_levels,
                method="BayesShrink",
                channel_axis=None,
                rescale_sigma=self.rescale_sigma,
        )
        if self.clip:
            denoised = denoised.clip(0.0, 1.0)
        image.detect_mat[:] = denoised
