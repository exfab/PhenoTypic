from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from skimage.restoration import denoise_nl_means

from ..abc_ import ImageEnhancer


class NonLocalMeansDenoiser(ImageEnhancer):
    """Denoise ``detect_mat`` with non-local means patch-based filtering.

    Compares patches across the image to identify similar structures and
    averages them, preserving thin colony boundaries and internal texture
    better than simple Gaussian or median filtering. Particularly effective
    at removing Gaussian noise and agar granularity.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Args:
        patch_size: Size of patches used for comparison in pixels. Larger
            patches capture more structure but are slower. Typical range:
            5--15. Default: 5.
        search_dist: Maximum search distance for similar patches in pixels.
            Larger values find more candidates at higher cost. Typical
            range: 5--21. Default: 11.
        h: Cut-off distance controlling smoothness. Rule of thumb:
            ``h`` ~= noise level (sigma). Higher values produce more
            smoothing. Default: 0.5.
        fast_mode: If ``True``, use faster variant with uniform spatial
            weighting. If ``False`` (default), use original algorithm
            with Gaussian spatial weighting.
        sigma: Expected noise standard deviation. Values > 0 improve
            patch weighting by accounting for expected noise variance.
            Default: 0.0 (disabled).

    Returns:
        Image: Input image with ``detect_mat`` denoised via non-local
        means filtering. ``rgb`` and ``gray`` are unchanged.

    Best For:
        - Scanner noise and agar granularity where colony edges must stay
          sharp.
        - Low-contrast or faint colonies where Gaussian blur would cause
          loss of detail.
        - Preserving colony texture and morphology during speckle and dust
          removal.
        - Pre-filtering before edge detection to avoid amplifying noise.

    Consider Also:
        - :class:`BM3DDenoiser` for state-of-the-art structured noise
          removal at higher computational cost.
        - :class:`BilateralDenoise` for faster edge-preserving denoising
          without patch comparison.
        - :class:`BayesShrinkEnhancer` for adaptive wavelet denoising with
          spatially varying thresholds.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a
        visual walkthrough of denoising pipelines on plate images.
        :doc:`/how_to/notebooks/denoise_low_light` for non-local means
        and other denoising strategies on low-light plate images.
    """

    def __init__(
            self,
            patch_size: int = 5,
            search_dist: int = 11,
            h: float = 0.5,
            *,
            fast_mode: bool = False,
            sigma: float = 0.0,
    ):
        """
        Parameters:
            patch_size (int): Size of patches used for comparison. Larger patches capture
                more structure but are slower. Start with 5-7 for agar plates; increase to 11-15
                for heavily noisy images. Default: 7.
            search_dist (int): Maximal distance in pixels to search for similar patches.
                Larger values find more candidates at higher cost. Default: 11.
            h (float): Cut-off distance controlling smoothness. Typical rule of thumb:
                h ≈ sigma (noise level). Increase to ~1.5*sigma for more smoothing.
                Default: 0.1.
            fast_mode (bool): If True, use faster variant with uniform spatial weighting.
                If False, use original algorithm with Gaussian spatial weighting (slower but
                potentially better quality). Default: False.
            sigma (float): Noise standard deviation. If provided (> 0), improves patch weighting
                by accounting for expected noise variance. Start with estimate_sigma() output.
                Default: 0.0 (disabled).
        """
        self.patch_size = int(patch_size)
        self.search_dist = int(search_dist)
        self.h = float(h)
        self.fast_mode = bool(fast_mode)
        self.sigma = float(sigma)

    def _operate(self, image: Image) -> Image:
        """Apply non-local means denoising to detection matrix."""
        denoised = denoise_nl_means(
                image=image.detect_mat[:],
                patch_size=self.patch_size,
                patch_distance=self.search_dist,
                h=self.h,
                fast_mode=self.fast_mode,
                sigma=self.sigma,
                preserve_range=True,
        )
        image.detect_mat[:] = denoised
        return image
