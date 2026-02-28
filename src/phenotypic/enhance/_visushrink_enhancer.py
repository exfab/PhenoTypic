from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from phenotypic import Image

from skimage.restoration import denoise_wavelet

from ..abc_ import ImageEnhancer


class VisuShrinkEnhancer(ImageEnhancer):
    """Wavelet denoising with universal VisuShrink thresholding for plate images.

    Applies wavelet-domain denoising using the VisuShrink method, which uses a single
    universal threshold across all wavelet subbands. This is effective for removing
    Gaussian noise from agar plates (scanner artifacts, CCD noise) while preserving
    sharp colony edges better than Gaussian blur.

    Use cases (agar plates):
    - Remove scanner banding and flatbed scanner noise without blurring colonies.
    - Denoise high-ISO camera images while preserving colony boundaries.
    - Suppress agar granularity and condensation speckle before detection.
    - Pre-filter before edge detection (Sobel, Canny) to avoid noise amplification.

    Tuning and effects:
    - sigma: Noise standard deviation in [0, 1] scale (matching detect_mat range).
      None (default) auto-estimates via MAD. Typical values: 0.01-0.05 for
      moderate noise (equivalent to σ=2.5-12.75 on 8-bit). Too high causes
      over-smoothing and colony merging.
    - wavelet: 'db2' (default) balances smoothness and locality. 'db4' captures
      more detail. 'sym2' offers symmetric filters. Must be orthogonal wavelet.
    - mode: 'soft' (default) for additive noise produces smoother results;
      'hard' preserves more edges but may leave residual noise.
    - wavelet_levels: None (default) uses max-3 levels automatically. More
      levels = finer noise removal but higher computation.

    Caveats:
    - VisuShrink uses a universal threshold designed to remove ALL noise with
      high probability, which can over-smooth compared to BayesShrinkEnhancer.
    - Not suitable for images with spatially varying noise levels (use
      BayesShrinkEnhancer instead for adaptive thresholding).
    - Does not correct illumination gradients; combine with background
      subtraction (GaussianSubtract, SubtractRollingBall) if needed.
    - Slower than Gaussian blur but faster than BM3D or non-local means.

    Attributes:
        sigma (float | None): Noise standard deviation in [0, 1]. None = auto-
            estimate via MAD. Typical: 0.01-0.05 for scanner/camera noise.
        wavelet (str): Wavelet family ('db2', 'db4', 'sym2', etc.). Default 'db2'.
        mode (Literal['soft', 'hard']): Thresholding mode. 'soft' recommended
            for additive noise. Default 'soft'.
        wavelet_levels (int | None): Decomposition levels. None = max-3 (auto).

    Examples:
        Basic denoising of scanner noise with defaults:

        >>> from phenotypic import Image
        >>> from phenotypic.enhance import VisuShrinkEnhancer
        >>> image = Image.imread('agar_plate.jpg')  # doctest: +SKIP
        >>> enhancer = VisuShrinkEnhancer()
        >>> denoised = enhancer.apply(image)  # doctest: +SKIP
        >>> # Original RGB/gray untouched, detect_mat is denoised
        >>> assert np.array_equal(image.rgb[:], denoised.rgb[:])  # doctest: +SKIP
        >>> assert np.array_equal(image.gray[:], denoised.gray[:])  # doctest: +SKIP
        >>> # detect_mat is different
        >>> assert not np.array_equal(image.detect_mat[:], denoised.detect_mat[:])  # doctest: +SKIP

        Custom parameters for heavily noisy images:

        >>> from phenotypic import Image
        >>> from phenotypic.enhance import VisuShrinkEnhancer
        >>> image = Image.imread('high_noise_plate.jpg')  # doctest: +SKIP
        >>> # Use db4 for finer details, more decomposition levels
        >>> enhancer = VisuShrinkEnhancer(
        ...     wavelet='db4',
        ...     wavelet_levels=5,
        ...     sigma=0.08  # Higher noise estimate
        ... )
        >>> denoised = enhancer.apply(image)  # doctest: +SKIP

        Chaining with other enhancers for robust preprocessing:

        >>> from phenotypic import Image, ImagePipeline
        >>> from phenotypic.enhance import (
        ...     VisuShrinkEnhancer, CLAHE, SubtractGaussian
        ... )
        >>> from phenotypic.detect import OtsuDetector
        >>> image = Image.imread('plate.jpg')  # doctest: +SKIP
        >>> # Build preprocessing pipeline
        >>> pipeline = ImagePipeline()
        >>> pipeline.add(SubtractGaussian(sigma=50))  # Remove background
        >>> pipeline.add(VisuShrinkEnhancer(sigma=0.03))  # Denoise
        >>> pipeline.add(CLAHE(clip_limit=0.02))  # Enhance local contrast
        >>> pipeline.add(OtsuDetector())  # Detect colonies
        >>> result = pipeline.apply(image)  # doctest: +SKIP
        >>> colonies = result.objects  # doctest: +SKIP
    """

    def __init__(
            self,
            sigma: float | None = None,
            wavelet: str = "db2",
            mode: Literal["soft", "hard"] = "soft",
            wavelet_levels: int | None = None,
            clip: bool = True,
    ):
        """Initialize VisuShrink wavelet denoiser.

        Parameters:
            sigma (float | None): Noise standard deviation in [0, 1] scale. None
                (default) auto-estimates via median absolute deviation (MAD).
                For reference: 8-bit noise σ=10/255 ≈ 0.04 in normalized scale.
                Typical values: 0.01-0.05 for moderate scanner/camera noise.
                Start with auto-estimation, then tune if needed.
            wavelet (str): Wavelet type from PyWavelets. 'db2' (default) is a
                good general choice. 'db4' for more detail, 'sym2' for symmetry.
                Must be orthogonal (db*, sym*) for proper noise handling.
            mode (Literal['soft', 'hard']): Threshold type. 'soft' (default)
                produces smoother results for additive noise. 'hard' preserves
                edges more but may leave noise artifacts.
            wavelet_levels (int | None): Decomposition depth. None (default)
                uses max-3 automatically. Higher = finer denoising, slower.
            clip (bool): Whether to clip output to [0, 1] range. Default True.
                Set to False when using with variance-stabilizing transforms
                (e.g., GAT) that require preserving the original scale.
        """
        self.sigma = sigma
        self.wavelet = wavelet
        self.mode = mode
        self.wavelet_levels = wavelet_levels
        self.clip = clip

    def _operate(self, image: Image) -> Image:
        """Apply VisuShrink wavelet denoising to detection matrix.

        Returns:
            Modified Image with denoised detect_mat
        """
        denoised = denoise_wavelet(
                image=image.detect_mat[:],
                sigma=self.sigma,
                wavelet=self.wavelet,
                mode=self.mode,
                wavelet_levels=self.wavelet_levels,
                method="VisuShrink",
                channel_axis=None,
                rescale_sigma=True,
        )
        if self.clip:
            denoised = denoised.clip(0.0, 1.0)
        image.detect_mat[:] = denoised
        return image
