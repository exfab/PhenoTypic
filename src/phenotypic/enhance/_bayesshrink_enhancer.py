from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, ClassVar, Literal

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from skimage.restoration import denoise_wavelet

from ..abc_ import ImageDenoiser
from ..tools_.mixin import _GATSupportMixin
from ..tools_.typing_ import TuneSpec


class BayesShrinkEnhancer(_GATSupportMixin, ImageDenoiser):
    """Denoise ``detect_mat`` with adaptive per-subband BayesShrink wavelet thresholding.

    Decomposes the detection matrix into wavelet subbands and applies a separate
    soft-threshold to each, computed from the estimated signal and noise
    variances. This preserves colony texture more faithfully than a universal
    threshold while suppressing scanner and camera noise. For algorithm details
    see :doc:`/explanation/what_enhancement_does`.

    Best For:
        - Flatbed scanner images where CCD read noise is spatially uniform
          and the auto-estimated sigma is reliable.
        - Plates requiring fine colony texture retention before downstream
          morphology measurement or texture analysis.
        - Pre-processing before ridge or edge enhancers that amplify noise
          alongside structure.
        - Low-light fluorescence plate images with Poisson-dominated shot
          noise when ``use_gat=True`` is enabled.
        - Pipelines where denoising must be fully serialisable and
          reproducible across runs.

    Consider Also:
        - :class:`VisuShrinkEnhancer` when a single universal threshold
          across all subbands is preferred for speed or simplicity.
        - :class:`BM3DDenoiser` for structured scanner banding or systematic
          CCD patterned noise that wavelet thresholding does not fully remove.
        - :class:`NonLocalMeansDenoiser` when patch-based self-similarity
          in the agar or colony texture can be exploited.
        - :class:`MedianFilter` for salt-and-pepper impulse noise rather
          than Gaussian additive noise.

    Args:
        sigma: Noise standard deviation on the [0, 1] intensity scale used
            to compute per-subband BayesShrink thresholds. ``None`` (default)
            auto-estimates via the MAD of the finest-scale HH subband
            (``median(|coeff|) / 0.6745``). Typical manual override: 0.01--0.05
            for moderate scanner noise. Raising sigma shrinks more coefficients
            and smooths more aggressively; lowering it retains more texture
            alongside residual noise. Ignored when ``use_gat=True`` — the
            stabilised-domain value 1.0 is used internally. Default: ``None``.
        wavelet: PyWavelets wavelet family string. Must be an orthogonal
            wavelet for BayesShrink's variance-preservation property to hold.
            ``'db2'`` (default) has compact support and handles colony edges
            well; ``'db4'`` represents smoother signals more sparsely and may
            suit filamentous fungi hyphae. Accepted examples: ``'db1'``--
            ``'db8'``, ``'sym2'``--``'sym8'``. Default: ``'db2'``.
        mode: Wavelet coefficient thresholding mode. ``'soft'`` (default)
            shrinks each coefficient toward zero by the threshold amount,
            producing a smooth, ringing-free output consistent with
            BayesShrink's Bayesian risk derivation. ``'hard'`` zeros
            coefficients below the threshold and leaves those above unchanged;
            this can introduce pseudo-Gibbs artefacts near sharp colony edges.
            Default: ``'soft'``.
        wavelet_levels: Decomposition depth. ``None`` (default) uses
            ``max_possible_levels - 3``, a conservative choice that avoids
            the coarsest subbands where reliable per-subband threshold
            estimation fails. Practical range: 2--8 integers; fewer than 3
            leaves coarse-scale noise untouched; more than ``max - 3`` risks
            over-smoothing broad colony texture. Default: ``None``.
        clip: Clamp output to [0, 1] after reconstruction. Soft thresholding
            can produce values marginally outside [0, 1] due to floating-point
            accumulation. Default: ``True``. Automatically set to ``False``
            inside the GAT region when ``use_gat=True``.
        rescale_sigma: Allow skimage to rescale ``sigma`` to match each
            subband's energy (the statistically correct behaviour for
            orthonormal wavelets). Default: ``True``. Automatically forced to
            ``False`` when ``use_gat=True`` because the stabilised domain
            has a different variance profile. Default: ``True``.

        # GAT parameters — only active when use_gat=True
        use_gat: Wrap the denoise call in a forward Generalised Anscombe
            Transform (GAT) → denoise at fixed sigma=1.0 → exact unbiased
            inverse GAT pipeline. Enables correct denoising under mixed
            Poisson-Gaussian noise (e.g. fluorescence plate readers, low-light
            incubator cameras). Leave ``False`` for standard flatbed scanner
            images where additive Gaussian noise dominates. Default: ``False``.
        gat_gain: Camera gain in electrons per ADU, used by the GAT to model
            Poisson variance scaling. Obtain from the sensor datasheet or a
            photon-transfer curve. Only relevant when ``use_gat=True``.
            Default: 1.0.
        gat_mu: Read-noise mean (DC baseline offset) in ADU before
            [0, 1] normalisation. Set to the dark-current bias level if the
            image has not been background-subtracted. Only relevant when
            ``use_gat=True``. Default: 0.0.
        gat_read_sigma: Standard deviation of the Gaussian read-noise
            component (electrons RMS). Setting this to the manufacturer's
            read noise improves stabilisation accuracy under mixed
            Poisson-Gaussian conditions. Only relevant when ``use_gat=True``.
            Default: 0.0.
        gat_scale_factor: Multiplier converting the [0, 1] normalised
            ``detect_mat`` back to photon counts before the forward GAT.
            ``None`` auto-detects from ``image.metadata.bit_depth`` (255 for
            8-bit, 65535 for 16-bit). Only relevant when ``use_gat=True``.
            Default: ``None``.

    Returns:
        Image: Input image with ``detect_mat`` denoised via adaptive wavelet
        thresholding. ``rgb`` and ``gray`` are unchanged.

    References:
        [1] S. G. Chang, B. Yu, and M. Vetterli, "Adaptive wavelet
        thresholding for image denoising and compression," *IEEE Trans.
        Image Process.*, vol. 9, no. 9, pp. 1532--1546, Sep. 2000.

        [2] D. L. Donoho and I. M. Johnstone, "Ideal spatial adaptation by
        wavelet shrinkage," *Biometrika*, vol. 81, no. 3, pp. 425--455,
        Sep. 1994.

        [3] M. Mäkitalo and A. Foi, "Optimal inversion of the generalized
        Anscombe transformation for Poisson-Gaussian noise," *IEEE Trans.
        Image Process.*, vol. 22, no. 1, pp. 91--103, Jan. 2013.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a
        visual walkthrough of enhancement pipelines on plate images.
        :doc:`/how_to/notebooks/denoise_low_light` for guidance on choosing
        between BayesShrink, BM3D, and non-local means for low-light plates.
        :doc:`/explanation/what_enhancement_does` for background on
        wavelet denoising and threshold selection strategies.
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
