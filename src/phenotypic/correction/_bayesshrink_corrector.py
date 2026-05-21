from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
from skimage.restoration import denoise_wavelet

from ..abc_ import ImageCorrector
from ..tools_.mixin import _GATSupportMixin


class BayesShrinkCorrector(_GATSupportMixin, ImageCorrector):
    """Denoise all image components using adaptive BayesShrink wavelet thresholding.

    Apply subband-adaptive wavelet denoising to RGB (if present), grayscale,
    and detection matrix simultaneously. BayesShrink estimates a separate
    threshold for each wavelet subband, preserving fine colony detail while
    suppressing noise more selectively than a universal threshold.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Args:
        sigma: Noise standard deviation. ``None`` auto-estimates from the
            finest wavelet subband. Typical range: 0.01--0.1 for normalized
            images. Retargeted to 1.0 when ``use_gat=True`` (applies only
            to the gray and detect_mat passes; RGB stays out of GAT).
            Default: ``None``.
        wavelet: Wavelet family name. ``'db2'`` balances smoothness and
            locality; ``'db4'`` preserves finer spatial detail. Default:
            ``'db2'``.
        mode: Thresholding mode. ``'soft'`` produces smoother results;
            ``'hard'`` retains sharper edges with possible noise residue.
            Default: ``'soft'``.
        wavelet_levels: Number of decomposition levels. ``None`` uses the
            maximum minus three (automatic). Default: ``None``.
        convert2ycbcr: Denoise RGB in YCbCr space so luminance and
            chrominance are handled separately, preserving colony color.
            Only applies when RGB data is present. Default: ``True``.
        rescale_sigma: skimage flag controlling internal sigma rescaling.
            Default: ``True``. Automatically forced to ``False`` for the
            gray/detect_mat passes when ``use_gat=True`` (RGB pass keeps
            its caller-supplied value because GAT is bypassed there).
        use_gat: Wrap gray and detect_mat denoising in the Generalized
            Anscombe Transform for Poisson-Gaussian noise. RGB is not
            transformed. Default: ``False``.
        gat_gain, gat_mu, gat_read_sigma, gat_scale_factor: GAT parameters.

    Returns:
        Image: Input image with all components (RGB, gray, detect_mat)
        transformed by adaptive wavelet denoising.

    Best For:
        - Plates imaged with aging or high-ISO cameras that introduce
          spatially varying sensor noise.
        - RGB plate scans destined for publication where color fidelity
          and fine detail must be preserved.
        - Pre-processing before multi-channel feature extraction (color
          composition and morphology).

    Consider Also:
        - :class:`VisuShrinkCorrector` when a faster, simpler universal
          threshold is acceptable.
        - :class:`StableDenoise` for variance-stabilized BM3D denoising
          of grayscale channels with Poisson-Gaussian noise.
        - :class:`BayesShrinkEnhancer` when only the detection matrix
          should be denoised (non-destructive to RGB and gray).

    References:
        [1] S. G. Chang, B. Yu, and M. Vetterli, "Adaptive wavelet
        thresholding for image denoising and compression," *IEEE Trans.
        Image Process.*, vol. 9, no. 9, pp. 1532--1546, Sep. 2000.

    See Also:
        :doc:`/how_to/notebooks/correct_color_cast` for a walkthrough of
        denoising plate images before color analysis.
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
        """Apply BayesShrink adaptive wavelet denoising to all image components.

        RGB stays outside the GAT region (uint8, color-mixed channels);
        gray and detect_mat are GAT-wrapped when ``use_gat=True``.
        """
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
            method="BayesShrink",
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
            method="BayesShrink",
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
            method="BayesShrink",
            channel_axis=None,
            rescale_sigma=self.rescale_sigma,
        )
        if self.clip:
            denoised_enh = denoised_enh.clip(0.0, 1.0)
        image._data.detect_mat = denoised_enh
