from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
from skimage.restoration import denoise_wavelet

from ..abc_ import ImageCorrector
from ..tools_.mixin import _GATSupportMixin


class VisuShrinkCorrector(_GATSupportMixin, ImageCorrector):
    """Denoise all image components using a universal VisuShrink wavelet threshold.

    Apply VisuShrink wavelet denoising to RGB (if present), grayscale, and
    detection matrix simultaneously. Unlike :class:`VisuShrinkEnhancer`,
    which modifies only the detection matrix, this corrector transforms all
    image representations to maintain cross-component consistency.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Args:
        sigma: Noise standard deviation. ``None`` auto-estimates from
            the image. Retargeted to 1.0 when ``use_gat=True`` (applies
            only to gray and detect_mat passes; RGB stays out of GAT).
            Default: ``None``.
        wavelet: Wavelet family name. ``'db2'`` is general-purpose;
            ``'db4'`` preserves finer detail. Default: ``'db2'``.
        mode: Thresholding mode. ``'soft'`` produces smoother results;
            ``'hard'`` retains sharper edges. Default: ``'soft'``.
        wavelet_levels: Number of decomposition levels. ``None`` uses
            the maximum minus three (automatic). Default: ``None``.
        convert2ycbcr: Denoise RGB in YCbCr space so luminance and
            chrominance are handled separately, preserving colony color.
            Only applies when RGB data is present. Default: ``True``.
        rescale_sigma: skimage flag controlling internal sigma rescaling.
            Default: ``True``. Auto-deferred for the gray/detect_mat
            passes when ``use_gat=True``.
        use_gat: Wrap gray and detect_mat denoising in the Generalized
            Anscombe Transform. RGB is not transformed. Default: ``False``.
        gat_gain, gat_mu, gat_read_sigma, gat_scale_factor: GAT parameters.

    Returns:
        Image: Input image with all components (RGB, gray, detect_mat)
        transformed by VisuShrink wavelet denoising.

    Best For:
        - Quick, uniform denoising of raw plate scans for archival or
          publication where a single threshold is acceptable.
        - Removing scanner noise from all image components before
          downstream multi-channel analysis.
        - Plates with relatively uniform noise where adaptive subband
          thresholding is not necessary.

    Consider Also:
        - :class:`BayesShrinkCorrector` for adaptive subband thresholds
          that preserve finer colony detail.
        - :class:`StableDenoise` for variance-stabilized BM3D denoising
          when Poisson-Gaussian noise modelling is important.
        - :class:`VisuShrinkEnhancer` when only the detection matrix
          should be denoised (non-destructive to RGB and gray).

    See Also:
        :doc:`/how_to/notebooks/correct_color_cast` for combining
        denoising with color correction workflows.
    """

    _GAT_NOISE_PARAMS: ClassVar[dict[str, float]] = {"sigma": 1.0}
    _GAT_DEFER_ATTRS: ClassVar[tuple[str, ...]] = ("rescale_sigma", "clip")

    def __init__(
        self,
        sigma: float | None = None,
        wavelet: str = "db2",
        mode: Literal["soft", "hard"] = "soft",
        wavelet_levels: int | None = None,
        convert2ycbcr: bool = True,
        rescale_sigma: bool = True,
        clip: bool = True,
        **kwargs,
    ):
        """Initialize VisuShrink corrector for all image components.

        Parameters:
            sigma (float | None): Noise level. None (default) auto-estimates.
                Retargeted to 1.0 when ``use_gat=True``.
            wavelet (str): Wavelet type. 'db2' (default).
            mode (Literal['soft', 'hard']): 'soft' (default) or 'hard'.
            wavelet_levels (int | None): None = max-3.
            convert2ycbcr (bool): Denoise RGB in YCbCr (default True).
            rescale_sigma (bool): skimage internal flag. Default True.
                Auto-deferred during GAT.
            **kwargs: Forwarded to :class:`_GATSupportMixin`.
        """
        super().__init__(**kwargs)
        self.sigma = sigma
        self.wavelet = wavelet
        self.mode = mode
        self.wavelet_levels = wavelet_levels
        self.convert2ycbcr = convert2ycbcr
        self.rescale_sigma = rescale_sigma
        self.clip = clip

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
