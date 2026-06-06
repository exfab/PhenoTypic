from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
from skimage.restoration import denoise_wavelet

from ..abc_ import ImageCorrector
from ..tools_.mixin import _GATSupportMixin


class VisuShrinkCorrector(_GATSupportMixin, ImageCorrector):
    """Denoise all image components using the near-minimax VisuShrink universal wavelet threshold.

    Apply VisuShrink wavelet denoising to RGB (if present), grayscale, and
    detection matrix simultaneously, using a single global threshold derived
    from the estimated noise level and image size. All three image
    representations are updated in one pass, keeping them mutually consistent.
    The universal threshold is deliberately conservative and tends to over-smooth
    relative to adaptive alternatives.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Best For:
        - Quick, consistent denoising of raw plate scans before archival or
          publication where a single noise level characterizes the whole image.
        - Plates with uniform scanner or camera noise where per-subband
          adaptation provides little additional benefit.
        - Removing broad-field scanner banding or read noise from all
          image components before downstream multi-channel analysis.

    Consider Also:
        - :class:`BayesShrinkCorrector` for adaptive per-subband thresholds
          that preserve finer colony detail at the cost of slightly more
          computation.
        - :class:`StableDenoise` for GAT-stabilized BM3D collaborative
          filtering when Poisson-Gaussian noise modelling is important.
        - :class:`VisuShrinkEnhancer` when only the detection matrix should
          be denoised and RGB and gray must remain untouched.

    Args:
        sigma: Noise standard deviation on the [0, 1] intensity scale.
            ``None`` auto-estimates from the median absolute deviation of
            the finest wavelet detail subband (MAD / 0.6745). The universal
            threshold T = sigma * sqrt(2 * log(N)) is conservative; reducing
            sigma by factors of 2--4 below the auto-estimate gives sharper
            output. Internally retargeted to 1.0 for gray and detect_mat
            when ``use_gat=True``; RGB is not GAT-wrapped. Default: ``None``.
        wavelet: PyWavelets wavelet family name. ``'db2'`` is general-purpose
            with compact support; ``'db4'`` has more vanishing moments and
            wider support, representing smooth colony interiors more sparsely.
            Default: ``'db2'``.
        mode: Wavelet coefficient thresholding mode. ``'soft'`` shrinks
            surviving coefficients toward zero, producing smoother output
            preferred for detection preprocessing. ``'hard'`` zeroes
            coefficients below the threshold and leaves the rest unchanged,
            retaining sharper edges but introducing discontinuities that can
            appear as ringing near sharp boundaries. Default: ``'soft'``.
        wavelet_levels: Number of wavelet decomposition levels. ``None`` uses
            the library heuristic (maximum possible minus three). Practical
            manual range: 2--6. More levels suppress noise at coarser spatial
            scales; fewer levels restrict denoising to fine-grained noise.
            Default: ``None``.
        convert2ycbcr: Convert RGB to YCbCr before denoising so luminance
            and chrominance channels are thresholded independently, preserving
            colony hue for pigmentation assays. Applies only when RGB data is
            present. Default: ``True``.
        rescale_sigma: Allow skimage to internally rescale sigma when
            converting between dtypes. Automatically deferred to ``False``
            during the gray and detect_mat passes when ``use_gat=True`` to
            prevent double-scaling the stabilized-domain noise level. Default:
            ``True``.
        clip: Clamp gray and detect_mat outputs to [0, 1] after denoising.
            Soft thresholding can produce marginally negative values near dark
            colony edges; clipping eliminates these. Automatically deferred to
            ``False`` inside the GAT region so the inverse transform operates
            on the full stabilized signal before the final clamp. Default:
            ``True``.

        # GAT parameters (active only when use_gat=True)
        use_gat: Wrap gray and detect_mat denoising in the Generalized
            Anscombe Transform, converting Poisson-Gaussian noise into
            approximately unit-variance Gaussian noise so the VisuShrink
            threshold is correctly scaled across intensity levels. RGB is not
            transformed. Enable for fluorescence plate readers or low-light
            incubator images. Default: ``False``.
        gat_gain: Camera gain in electrons per ADU. Typical range: 0.5--10
            e-/ADU depending on sensor. Obtain from the manufacturer datasheet
            or a mean-variance calibration. Default: ``1.0``.
        gat_mu: Read-noise mean (dark-current baseline offset) in count units
            consistent with ``gat_scale_factor``. Set to 0.0 when the image
            has been bias-subtracted, covering most flatbed-scanner workflows.
            Default: ``0.0``.
        gat_read_sigma: Read-noise standard deviation in count units
            consistent with ``gat_scale_factor``. Zero assumes pure Poisson
            noise; supplying the sensor read-noise spec (a few to a few tens of
            counts for typical scientific CCDs) improves stabilization at low
            signal levels. Default: ``0.0``.
        gat_scale_factor: Multiplier converting normalized [0, 1] float data
            to photon counts before the forward GAT. ``None`` auto-detects
            from image bit depth (8-bit: 255, 16-bit: 65535). Override for
            non-standard bit depths such as 12-bit sensors stored in a 16-bit
            container (4095). Default: ``None``.

    Returns:
        Image: Input image with all components (``rgb``, ``gray``,
        ``detect_mat``) transformed by VisuShrink wavelet denoising. All
        three representations are updated in a single pass.

    References:
        [1] D. L. Donoho and I. M. Johnstone, "Ideal spatial adaptation by
        wavelet shrinkage," *Biometrika*, vol. 81, no. 3, pp. 425--455,
        Sep. 1994.

        [2] M. Mäkitalo and A. Foi, "Optimal inversion of the generalized
        Anscombe transformation for Poisson-Gaussian noise," *IEEE Trans.
        Image Process.*, vol. 22, no. 1, pp. 91--103, Jan. 2013.

    See Also:
        :doc:`/how_to/notebooks/denoise_low_light` for a visual walkthrough
        of wavelet denoising on plate images.
        :doc:`/explanation/image_quality_noise_contrast_structure` for
        background on noise models and denoising strategy selection.
    """

    _GAT_NOISE_PARAMS: ClassVar[dict[str, float]] = {"sigma": 1.0}
    _GAT_DEFER_ATTRS: ClassVar[tuple[str, ...]] = ("rescale_sigma", "clip")

    sigma: float | None = None
    wavelet: str = "db2"
    mode: Literal["soft", "hard"] = "soft"
    wavelet_levels: int | None = None
    convert2ycbcr: bool = True
    rescale_sigma: bool = True
    clip: bool = True

    def _operate(self, image: Image) -> Image:
        """Apply VisuShrink wavelet denoising to all image components."""
        if not image.rgb.isempty():
            self._denoise_rgb(image)
        self._gat_apply(image, "gray", self._denoise_gray)
        self._gat_apply(image, "detect_mat", self._denoise_detect_mat)
        return image

    def _denoise_rgb(self, image: Image) -> None:
        denoised_rgb = denoise_wavelet(
            image=image.rgb[:],
            sigma=self.sigma,
            wavelet=self.wavelet,
            mode=self.mode,
            wavelet_levels=self.wavelet_levels,
            method="VisuShrink",
            convert2ycbcr=self.convert2ycbcr,
            channel_axis=-1,
            rescale_sigma=self.rescale_sigma,
        )
        image._data.rgb = denoised_rgb.clip(0, 255).astype(np.uint8)

    def _denoise_gray(self, image: Image) -> None:
        denoised_gray = denoise_wavelet(
            image=image.gray[:],
            sigma=self.sigma,
            wavelet=self.wavelet,
            mode=self.mode,
            wavelet_levels=self.wavelet_levels,
            method="VisuShrink",
            channel_axis=None,
            rescale_sigma=self.rescale_sigma,
        )
        if self.clip:
            denoised_gray = denoised_gray.clip(0.0, 1.0)
        image._data.gray = denoised_gray

    def _denoise_detect_mat(self, image: Image) -> None:
        denoised_enh = denoise_wavelet(
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
            denoised_enh = denoised_enh.clip(0.0, 1.0)
        image._data.detect_mat = denoised_enh
