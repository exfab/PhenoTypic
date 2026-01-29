from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image

from skimage.restoration import denoise_nl_means

from ..abc_ import ImageEnhancer


class NonLocalMeansDenoiser(ImageEnhancer):
    """
    Non-local means denoising for suppressing noise while preserving texture.

    Performs non-local means denoising on images, which is particularly effective
    at removing Gaussian noise from agar plates while preserving fine colony details
    and edges. Unlike simple Gaussian or median filtering, non-local means compares
    patches across the image to identify similar structures, enabling preservation
    of thin colony boundaries and internal texture.

    Use cases (agar plates):
    - Remove scanner noise and agar granularity without excessive blurring of colony edges.
    - Denoise low-contrast or faint colonies where Gaussian blur would cause loss of detail.
    - Preserve colony texture and morphology while reducing speckle and dust artifacts.
    - Pre-filter before edge detection (e.g., `SobelFilter`) to avoid amplifying noise.

    Tuning and effects:
    - patch_size: Larger patches (e.g., 7-15) capture more structure and are slower;
      smaller patches (5-7) are faster but may miss textures. For colonies, 7 is typically
      a good balance.
    - patch_distance: Larger search width (e.g., 11-21) considers more similar patches
      at higher computational cost; smaller values (5-7) run faster but may miss good
      matches far from the pixel. Default of 11 usually works well.
    - h: Controls the decay in patch weights. Larger h allows more smoothing between
      dissimilar patches (more blur); smaller h is more conservative. A rule of thumb:
      h ≈ sigma (noise level). Too large h causes over-smoothing and colony merging.
    - fast_mode: If True (default), uses a faster algorithm with slightly lower quality
      but much better performance. For interactive work, fast_mode=True. For publication-quality
      results, consider fast_mode=False, but with longer runtime.
    - sigma: The expected noise standard deviation. If provided, the algorithm accounts
      for this when computing patch similarity, often improving denoising quality. Set to 0
      (default) to disable.

    Caveats:
    - Non-local means is slower than Gaussian blur, especially with fast_mode=False.
    - Computational complexity grows with patch_distance and patch_size.
    - For very large search radii or patch sizes, memory usage can become significant.
    - Excessive smoothing (large h) can merge adjacent colonies just like Gaussian blur.
    - Not suitable for images with strong structural artifacts (e.g., dust particles larger
      than patch_size); morphological operations may be preferable.

    Attributes:
        patch_size (int): Size of patches (in pixels) used for similarity comparison.
        patch_distance (int): Maximal distance in pixels where to search for similar patches.
        h (float): Cut-off distance controlling patch weight decay (higher = more smoothing).
        fast_mode (bool): If True, use faster algorithm; if False, use original algorithm.
        sigma (float): Expected noise standard deviation for improved patch weighting.
    """

    def __init__(
            self,
            patch_size: int = 5,
            patch_distance: int = 11,
            h: float = 0.5,
            *,
            fast_mode: bool = True,
            sigma: float = 0.0,
    ):
        """
        Parameters:
            patch_size (int): Size of patches used for comparison. Larger patches capture
                more structure but are slower. Start with 5-7 for agar plates; increase to 11-15
                for heavily noisy images. Default: 7.
            patch_distance (int): Maximal distance in pixels to search for similar patches.
                Larger values find more candidates at higher cost. Default: 11.
            h (float): Cut-off distance controlling smoothness. Typical rule of thumb:
                h ≈ sigma (noise level). Increase to ~1.5*sigma for more smoothing.
                Default: 0.1.
            fast_mode (bool): If True (default), use faster variant with uniform spatial weighting.
                If False, use original algorithm with Gaussian spatial weighting (slower but
                potentially better quality). Default: True.
            sigma (float): Noise standard deviation. If provided (> 0), improves patch weighting
                by accounting for expected noise variance. Start with estimate_sigma() output.
                Default: 0.0 (disabled).
        """
        self.patch_size = int(patch_size)
        self.patch_distance = int(patch_distance)
        self.h = float(h)
        self.fast_mode = bool(fast_mode)
        self.sigma = float(sigma)

    def _operate(self, image: Image) -> Image:
        """Apply non-local means denoising to enhanced grayscale."""
        denoised = denoise_nl_means(
                image=image.enh_gray[:],
                patch_size=self.patch_size,
                patch_distance=self.patch_distance,
                h=self.h,
                fast_mode=self.fast_mode,
                sigma=self.sigma,
                preserve_range=True,
        )
        image.enh_gray[:] = denoised
        return image
