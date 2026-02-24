from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from phenotypic import Image

import numpy as np
from skimage.restoration import denoise_wavelet

from ..abc_ import ImageCorrector


class VisuShrinkCorrector(ImageCorrector):
    """Wavelet denoising with VisuShrink for all image components (RGB, gray, detect_mat).

    Applies VisuShrink wavelet denoising to the entire image, modifying RGB (if present),
    grayscale, and detection matrix simultaneously. Unlike VisuShrinkEnhancer (which
    modifies only detect_mat), this corrector updates all image representations, ensuring
    consistency across components.

    Use cases (agar plates):
    - Denoise raw RGB plate images for archival/publication while maintaining color fidelity.
    - Remove scanner noise from all image components before downstream analysis.
    - Clean up camera noise in RGB images while denoising the grayscale simultaneously.
    - Pre-process images for multi-channel analysis (color + morphology).

    Tuning and effects:
    - sigma: Noise level. For RGB, denoise_wavelet handles channel-specific scaling
      internally. For grayscale, sigma is in [0, 1] scale. None (default) auto-estimates.
    - wavelet: 'db2' (default) for general use. 'db4' for finer details.
    - mode: 'soft' (default) for smoother results, 'hard' for sharper edges.
    - convert2ycbcr: If True (default), RGB denoising happens in YCbCr color space
      (Y=luminance, CbCr=chrominance). This typically produces better results because
      luminance and color are denoised separately, preserving colony color better.

    Caveats:
    - Modifies ALL image components, including original RGB. Cannot be undone without
      reloading the image. For non-destructive preprocessing, use VisuShrinkEnhancer.
    - VisuShrink's universal threshold tends to over-smooth; consider BayesShrinkCorrector
      for adaptive, detail-preserving denoising.
    - Does not handle illumination gradients; combine with background correction.
    - Slower than simple Gaussian blur, especially for RGB images.

    Attributes:
        sigma (float | None): Noise std deviation. None = auto-estimate.
        wavelet (str): Wavelet family. Default 'db2'.
        mode (Literal['soft', 'hard']): Thresholding mode. Default 'soft'.
        wavelet_levels (int | None): Decomposition levels. None = max-3.
        convert2ycbcr (bool): Denoise RGB in YCbCr space. Default True.

    Examples:
        Denoise RGB image for archival quality:

        >>> from phenotypic import Image
        >>> from phenotypic.correction import VisuShrinkCorrector
        >>> image = Image.imread('raw_plate_scan.jpg')  # doctest: +SKIP
        >>> corrector = VisuShrinkCorrector()
        >>> denoised = corrector.apply(image)  # doctest: +SKIP
        >>> # All components modified
        >>> assert not np.array_equal(denoised.rgb[:], image.rgb[:])  # doctest: +SKIP
        >>> assert not np.array_equal(denoised.gray[:], image.gray[:])  # doctest: +SKIP

        Denoise grayscale-only image gracefully:

        >>> from phenotypic import Image
        >>> from phenotypic.correction import VisuShrinkCorrector
        >>> # Grayscale image (no RGB)
        >>> image = Image.imread('gray_plate.tif')  # doctest: +SKIP
        >>> # Works without error, denoises gray and detect_mat
        >>> corrector = VisuShrinkCorrector()
        >>> denoised = corrector.apply(image)  # doctest: +SKIP
        >>> # Only gray and detect_mat modified (no RGB to modify)
        >>> assert not np.array_equal(denoised.gray[:], image.gray[:])  # doctest: +SKIP

        Color-preserving denoising with YCbCr conversion:

        >>> from phenotypic import Image
        >>> from phenotypic.correction import VisuShrinkCorrector
        >>> image = Image.imread('color_plate.jpg')  # doctest: +SKIP
        >>> # Default: denoise in YCbCr (better color preservation)
        >>> corrector = VisuShrinkCorrector(convert2ycbcr=True)
        >>> result = corrector.apply(image)  # doctest: +SKIP
        >>> # Colony colors preserved better than RGB-space denoising
    """

    def __init__(
        self,
        sigma: float | None = None,
        wavelet: str = "db2",
        mode: Literal["soft", "hard"] = "soft",
        wavelet_levels: int | None = None,
        convert2ycbcr: bool = True,
    ):
        """Initialize VisuShrink corrector for all image components.

        Parameters:
            sigma (float | None): Noise level. None (default) auto-estimates.
                For RGB, denoise_wavelet handles internal scaling.
            wavelet (str): Wavelet type. 'db2' (default) is general-purpose.
            mode (Literal['soft', 'hard']): 'soft' (default) for smoothness.
            wavelet_levels (int | None): Levels. None = max-3.
            convert2ycbcr (bool): Denoise RGB in YCbCr space (True, default)
                for better color preservation. Only applies if RGB exists.
        """
        self.sigma = sigma
        self.wavelet = wavelet
        self.mode = mode
        self.wavelet_levels = wavelet_levels
        self.convert2ycbcr = convert2ycbcr

    def _operate(self, image: Image) -> Image:
        """Apply VisuShrink wavelet denoising to all image components.

        Returns:
            Modified Image with all components denoised
        """
        # Denoise RGB if present
        if not image.rgb.isempty():
            denoised_rgb = denoise_wavelet(
                image=image.rgb[:],
                sigma=self.sigma,
                wavelet=self.wavelet,
                mode=self.mode,
                wavelet_levels=self.wavelet_levels,
                method="VisuShrink",
                convert2ycbcr=self.convert2ycbcr,
                channel_axis=-1,
                rescale_sigma=True,
            )
            image._data.rgb = denoised_rgb.clip(0, 255).astype(np.uint8)

        # Always denoise gray (luminance grayscale)
        denoised_gray = denoise_wavelet(
            image=image.gray[:],
            sigma=self.sigma,
            wavelet=self.wavelet,
            mode=self.mode,
            wavelet_levels=self.wavelet_levels,
            method="VisuShrink",
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
            method="VisuShrink",
            channel_axis=None,
            rescale_sigma=True,
        )
        image._data.detect_mat = denoised_enh.clip(0.0, 1.0)

        return image
