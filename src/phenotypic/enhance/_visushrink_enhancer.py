from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, ClassVar, Literal

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from skimage.restoration import denoise_wavelet

from ..abc_ import ImageDenoiser
from ..sdk_.mixin import _GATSupportMixin
from ..sdk_.typing_ import TuneSpec


class VisuShrinkEnhancer(_GATSupportMixin, ImageDenoiser):
    """Denoise ``detect_mat`` with universal VisuShrink wavelet thresholding.

    Decomposes the image into wavelet subbands and zeros all coefficients
    below the universal threshold T = sigma * sqrt(2 * log(N)), which is
    near-minimax optimal for Gaussian white noise. The universal threshold
    with the auto-estimated sigma is conservative and can over-smooth; the
    skimage gallery recommends supplying a manual sigma below the
    auto-estimate (e.g. half) for better visual quality.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Best For:
        - Flatbed scanner banding and CCD read-noise removal before detection.
        - High-ISO camera plate images where colony boundaries must remain
          sharp after denoising.
        - Agar granularity and condensation speckle suppression before edge
          detection or thresholding.
        - Batch pipelines where a single universal threshold is preferred
          over per-subband tuning.

    Consider Also:
        - :class:`BayesShrinkEnhancer` for adaptive per-subband thresholding
          that preserves more colony texture detail at variable noise levels.
        - :class:`EnhanceBlockMatch` for state-of-the-art structured noise removal
          at higher computational cost.
        - :class:`LocalEdgeDenoise` for edge-preserving spatial smoothing
          without wavelet decomposition.

    Args:
        sigma: Noise standard deviation on the [0, 1] intensity scale.
            Controls the universal threshold T = sigma * sqrt(2 * log(N)).
            ``None`` (default) auto-estimates via MAD of finest-scale wavelet
            detail coefficients; the universal threshold with this estimate
            is conservative and often over-smooths — the skimage gallery
            recommends a manual sigma below the auto-estimate (e.g. half)
            for better visual quality. Typical range: 0.01--0.05 for
            standard scanner/camera
            noise. Has no effect when ``use_gat=True`` (the threshold is set
            using the stabilized-domain value 1.0 internally).
        wavelet: Wavelet family name (PyWavelets string). Use only orthogonal
            families: ``'db2'`` (default, compact support, good edge
            localisation), ``'db4'`` (more vanishing moments, suppresses
            smooth background gradients better), ``'sym4'`` or ``'sym6'``
            (near-symmetric, reduced Gibbs ringing). Biorthogonal wavelets
            produce coloured noise in subbands and are not recommended.
        mode: Thresholding mode. ``'soft'`` (default) subtracts the
            threshold from surviving coefficients, producing continuous
            output without ringing; recommended for Gaussian noise removal
            before detection. ``'hard'`` preserves coefficient amplitudes
            above the threshold exactly, retaining sharper edges at the cost
            of Gibbs-like artefacts at the threshold boundary.
        wavelet_levels: Number of decomposition levels. ``None`` (default)
            uses the library heuristic (max levels minus 3). Higher values
            denoise at larger spatial scales, risking removal of genuine
            low-contrast colony signal; lower values leave coarse banding
            intact. Valid range: 1 to floor(log2(min_image_dimension)).
        clip: Clip output to [0, 1]. Default: ``True``. Soft thresholding
            on float inputs can produce slightly negative values near dark
            edges; clipping eliminates these. Automatically deferred to
            ``False`` when ``use_gat=True``.
        rescale_sigma: Allow skimage to rescale the user-supplied sigma
            proportionally when it converts integer-dtype inputs to float
            internally. Default: ``True``. Has no observable effect on the
            float32 ``detect_mat`` used in this project. Automatically
            forced to ``False`` when ``use_gat=True``.

        # GAT parameters (active only when use_gat=True)
        use_gat: Wrap denoising in the Generalized Anscombe Transform to
            handle Poisson-Gaussian noise (e.g., low-light fluorescence or
            high-ISO DSLR images of colonies). Default: ``False``.
        gat_gain: Camera gain in electrons per ADU. Scales the Poisson
            noise component in the GAT model. Typical range 0.1--10.0
            (consumer DSLR ~0.5--3.0, scientific CCD ~1.0--10.0). Leave at
            1.0 for normalized images without calibrated gain. Default: 1.0.
        gat_mu: Read-noise mean (baseline DC offset). Set to 0.0 if
            dark-frame subtraction has been applied. Default: 0.0.
        gat_read_sigma: Standard deviation of additive Gaussian read noise.
            ``0.0`` (default) assumes pure Poisson noise. For scientific
            cameras with measurable read noise, obtain from spec sheet in
            e- RMS and convert: read_sigma_norm = read_sigma_e / (gain *
            max_counts).
        gat_scale_factor: Multiplier converting normalized [0, 1] data to
            photon counts before the GAT forward pass. ``None`` (default)
            auto-detects from image bit depth (8-bit → 255, 16-bit → 65535).
            Override when the effective bit depth differs from metadata (e.g.,
            14-bit sensor stored as 16-bit padded: use 16383 not 65535).

    Returns:
        Image: Input image with ``detect_mat`` denoised via universal wavelet
        thresholding. ``rgb`` and ``gray`` are unchanged.

    References:
        [1] D. L. Donoho and I. M. Johnstone, "Ideal spatial adaptation by
        wavelet shrinkage," *Biometrika*, vol. 81, no. 3, pp. 425--455,
        Sep. 1994.
        [2] M. Mäkitalo and A. Foi, "Optimal inversion of the generalized
        Anscombe transformation for Poisson-Gaussian noise," *IEEE Trans.
        Image Process.*, vol. 22, no. 1, pp. 91--103, Jan. 2013.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a
        visual walkthrough of denoising pipelines on plate images.
        :doc:`/how_to/notebooks/denoise_low_light` for wavelet denoising
        strategies on low-light plate images.
        :doc:`/explanation/what_enhancement_does` for background on wavelet
        thresholding and the VisuShrink threshold derivation.
    """

    _GAT_NOISE_PARAMS: ClassVar[dict[str, float]] = {"sigma": 1.0}
    _GAT_DEFER_ATTRS: ClassVar[tuple[str, ...]] = ("clip", "rescale_sigma")

    sigma: float | None = None
    wavelet: str = "db2"
    mode: Literal["soft", "hard"] = "soft"
    wavelet_levels: Annotated[int | None, TuneSpec(2, 6)] = None
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
