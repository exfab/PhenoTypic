from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
from skimage.restoration import denoise_wavelet

from ..abc_ import ImageCorrector


class BayesShrinkCorrector(ImageCorrector):
    """Denoise all image components using adaptive BayesShrink wavelet thresholding.

    Apply subband-adaptive wavelet denoising to RGB (if present), grayscale,
    and detection matrix simultaneously. BayesShrink estimates a separate
    threshold for each wavelet subband, preserving fine colony detail while
    suppressing noise more selectively than a universal threshold.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Args:
        sigma: Noise standard deviation. ``None`` auto-estimates from the
            finest wavelet subband. Typical range: 0.01--0.1 for normalized
            images. Default: ``None``.
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

    def __init__(
        self,
        sigma: float | None = None,
        wavelet: str = "db2",
        mode: Literal["soft", "hard"] = "soft",
        wavelet_levels: int | None = None,
        convert2ycbcr: bool = True,
    ):
        """Initialize BayesShrink adaptive corrector for all image components.

        Parameters:
            sigma (float | None): Noise level. None (default) auto-estimates.
                BayesShrink benefits from accurate sigma for optimal adaptive
                thresholding. Test explicit values if auto-estimation seems off.
            wavelet (str): Wavelet type. 'db2' (default) is general-purpose.
                'db4' for finer detail preservation.
            mode (Literal['soft', 'hard']): 'soft' (default) for smoothness,
                'hard' for sharper edges with possible noise residue.
            wavelet_levels (int | None): Levels. None = max-3 (automatic).
            convert2ycbcr (bool): Denoise RGB in YCbCr (True, default) for
                better color preservation. Only applies when RGB exists.
        """
        self.sigma = sigma
        self.wavelet = wavelet
        self.mode = mode
        self.wavelet_levels = wavelet_levels
        self.convert2ycbcr = convert2ycbcr

    def _operate(self, image: Image) -> Image:
        """Apply BayesShrink adaptive wavelet denoising to all image components.

        Returns:
            Modified Image with all components denoised adaptively
        """
        # Denoise RGB if present
        if not image.rgb.isempty():
            denoised_rgb = denoise_wavelet(
                image=image.rgb[:],
                sigma=self.sigma,
                wavelet=self.wavelet,
                mode=self.mode,
                wavelet_levels=self.wavelet_levels,
                method="BayesShrink",
                convert2ycbcr=self.convert2ycbcr,
                channel_axis=-1,
                rescale_sigma=True,
            )
            image._data.rgb = denoised_rgb.clip(0, 255).astype(np.uint8)

        # Always denoise gray
        denoised_gray = denoise_wavelet(
            image=image.gray[:],
            sigma=self.sigma,
            wavelet=self.wavelet,
            mode=self.mode,
            wavelet_levels=self.wavelet_levels,
            method="BayesShrink",
            channel_axis=None,
            rescale_sigma=True,
        )
        image._data.gray = denoised_gray.clip(0.0, 1.0)

        # Always denoise detect_mat
        denoised_enh = denoise_wavelet(
            image=image.detect_mat[:],
            sigma=self.sigma,
            wavelet=self.wavelet,
            mode=self.mode,
            wavelet_levels=self.wavelet_levels,
            method="BayesShrink",
            channel_axis=None,
            rescale_sigma=True,
        )
        image._data.detect_mat = denoised_enh.clip(0.0, 1.0)

        return image
