from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from skimage.restoration import denoise_wavelet

from ..abc_ import ImageEnhancer


class BayesShrinkEnhancer(ImageEnhancer):
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
            threshold quality.
        wavelet: Wavelet family. ``'db2'`` (default) balances smoothness and
            locality; ``'db4'`` preserves finer details. Must be orthogonal.
        mode: Thresholding mode. ``'soft'`` (default) produces smoother
            results; ``'hard'`` preserves edges more aggressively.
        wavelet_levels: Decomposition depth. ``None`` (default) uses max-3
            automatically. Higher values allow finer noise/signal separation.
        clip: Clip output to [0, 1]. Default: ``True``. Set to ``False``
            when using with variance-stabilizing transforms (e.g., GAT).

    Returns:
        Image: Input image with ``detect_mat`` denoised via adaptive wavelet
        thresholding. ``rgb`` and ``gray`` are unchanged.

    Best For:
        - Images with spatially varying noise from uneven illumination.
        - Preserving colony texture and internal morphology during denoising.
        - Scanner noise and camera artifacts on plates where fine detail
          matters for downstream measurement.
        - Pre-filtering before feature extraction or texture analysis.

    Consider Also:
        - :class:`VisuShrinkEnhancer` for faster denoising with a universal
          threshold when spatial noise uniformity is acceptable.
        - :class:`BM3DDenoiser` for state-of-the-art denoising of structured
          noise patterns.
        - :class:`BilateralDenoise` for edge-preserving smoothing without
          wavelet decomposition.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a
        visual walkthrough of enhancement pipelines on plate images.
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
        """Initialize BayesShrink adaptive wavelet denoiser.

        Parameters:
            sigma (float | None): Noise standard deviation in [0, 1] scale. None
                (default) auto-estimates. More accurate sigma improves adaptive
                thresholding quality. Typical: 0.01-0.05 for moderate noise.
            wavelet (str): Wavelet type. 'db2' (default) is general-purpose.
                'db4' for finer details, 'sym2' for symmetric filters.
            mode (Literal['soft', 'hard']): 'soft' (default) for smoother
                denoising, 'hard' for sharper edges with possible noise residue.
            wavelet_levels (int | None): Decomposition depth. None (default)
                uses max-3. Increase for very noisy images.
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
        """Apply BayesShrink adaptive wavelet denoising to detection matrix.

        Returns:
            Modified Image with denoised detect_mat
        """
        denoised = denoise_wavelet(
                image=image.detect_mat[:],
                sigma=self.sigma,
                wavelet=self.wavelet,
                mode=self.mode,
                wavelet_levels=self.wavelet_levels,
                method="BayesShrink",
                channel_axis=None,
                rescale_sigma=True,
        )
        if self.clip:
            denoised = denoised.clip(0.0, 1.0)
        image.detect_mat[:] = denoised
        return image
