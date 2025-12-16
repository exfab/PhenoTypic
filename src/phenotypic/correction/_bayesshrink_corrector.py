from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from phenotypic import Image

import numpy as np
from skimage.restoration import denoise_wavelet

from ..abc_ import ImageCorrector


class BayesShrinkCorrector(ImageCorrector):
    """Adaptive wavelet denoising with BayesShrink for all image components.

    Applies BayesShrink wavelet denoising to RGB (if present), grayscale, and enhanced
    grayscale simultaneously. BayesShrink computes adaptive thresholds for each wavelet
    subband, producing higher quality results than VisuShrink by preserving fine details
    while removing noise.

    Use cases (agar plates):
    - Denoise RGB images for publication while preserving colony color and texture.
    - Remove spatially varying noise (e.g., from uneven illumination or aging cameras)
      across all image components.
    - Clean up scanner artifacts in archival RGB images with minimal detail loss.
    - Pre-process for multi-channel feature extraction (color composition + morphology).

    Tuning and effects:
    - sigma: Noise level. None (default) auto-estimates. BayesShrink uses sigma to
      compute subband-adaptive thresholds, so accurate estimation improves quality.
    - wavelet: 'db2' (default) balances smoothness and locality. 'db4' for finer details.
    - mode: 'soft' (default) for smoother denoising, 'hard' for sharper edges.
    - convert2ycbcr: If True (default), RGB is denoised in YCbCr space (luminance and
      chrominance handled separately), which preserves colony color better.

    Caveats:
    - Modifies ALL image components irreversibly. For non-destructive preprocessing,
      use BayesShrinkEnhancer instead.
    - Requires more computation than VisuShrink due to adaptive threshold calculation.
    - Assumes Gaussian noise; not ideal for impulse noise (salt-and-pepper).
    - Does not correct illumination gradients; combine with background subtraction.

    Attributes:
        sigma (float | None): Noise std deviation. None = auto-estimate.
        wavelet (str): Wavelet family. Default 'db2'.
        mode (Literal['soft', 'hard']): Thresholding mode. Default 'soft'.
        wavelet_levels (int | None): Decomposition levels. None = max-3.
        convert2ycbcr (bool): Denoise RGB in YCbCr space. Default True.

    Examples:
        .. dropdown:: High-quality RGB denoising for publication

            .. code-block:: python

                from phenotypic import Image
                from phenotypic.correction import BayesShrinkCorrector

                image = Image.imread('raw_plate_scan.jpg')

                # BayesShrink preserves more detail than VisuShrink
                corrector = BayesShrinkCorrector()
                denoised = corrector.apply(image)

                # All components denoised with adaptive thresholding
                assert not np.array_equal(denoised.rgb[:], image.rgb[:])
                assert not np.array_equal(denoised.gray[:], image.gray[:])

        .. dropdown:: Fine detail preservation with db4 wavelet

            .. code-block:: python

                from phenotypic import Image
                from phenotypic.correction import BayesShrinkCorrector

                image = Image.imread('high_res_plate.jpg')

                # db4 preserves finer details than default db2
                corrector = BayesShrinkCorrector(wavelet='db4')
                denoised = corrector.apply(image)

                # Better detail preservation for texture analysis

        .. dropdown:: Spatially varying noise handling

            .. code-block:: python

                from phenotypic import Image
                from phenotypic.correction import BayesShrinkCorrector

                # Image with uneven illumination (varying noise levels)
                image = Image.imread('vignette_plate.jpg')

                # BayesShrink adapts to local noise levels
                corrector = BayesShrinkCorrector()
                denoised = corrector.apply(image)

                # Aggressive denoising in high-noise regions, conservative in low-noise
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

    @staticmethod
    def _operate(
        image: Image,
        sigma: float | None = None,
        wavelet: str = "db2",
        mode: Literal["soft", "hard"] = "soft",
        wavelet_levels: int | None = None,
        convert2ycbcr: bool = True,
    ) -> Image:
        """Apply BayesShrink adaptive wavelet denoising to all image components.

        Parameters:
            image: Image object to denoise
            sigma: Noise level estimate
            wavelet: Wavelet type
            mode: 'soft' or 'hard' thresholding
            wavelet_levels: Decomposition levels
            convert2ycbcr: Convert RGB to YCbCr before denoising

        Returns:
            Modified Image with all components denoised adaptively
        """
        # Denoise RGB if present
        if not image.rgb.isempty():
            denoised_rgb = denoise_wavelet(
                image=image.rgb[:],
                sigma=sigma,
                wavelet=wavelet,
                mode=mode,
                wavelet_levels=wavelet_levels,
                method="BayesShrink",
                convert2ycbcr=convert2ycbcr,
                channel_axis=-1,
                rescale_sigma=True,
            )
            image._data.rgb = denoised_rgb.clip(0, 255).astype(np.uint8)

        # Always denoise gray
        denoised_gray = denoise_wavelet(
            image=image.gray[:],
            sigma=sigma,
            wavelet=wavelet,
            mode=mode,
            wavelet_levels=wavelet_levels,
            method="BayesShrink",
            channel_axis=None,
            rescale_sigma=True,
        )
        image._data.gray = denoised_gray.clip(0.0, 1.0)

        # Always denoise enh_gray
        denoised_enh = denoise_wavelet(
            image=image.enh_gray[:],
            sigma=sigma,
            wavelet=wavelet,
            mode=mode,
            wavelet_levels=wavelet_levels,
            method="BayesShrink",
            channel_axis=None,
            rescale_sigma=True,
        )
        image._data.enh_gray = denoised_enh.clip(0.0, 1.0)

        return image
