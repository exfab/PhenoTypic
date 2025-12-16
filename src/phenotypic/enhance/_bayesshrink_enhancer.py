from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from phenotypic import Image

from skimage.restoration import denoise_wavelet

from ..abc_ import ImageEnhancer


class BayesShrinkEnhancer(ImageEnhancer):
    """Wavelet denoising with adaptive BayesShrink thresholding for plate images.

    Applies wavelet-domain denoising using the BayesShrink method, which computes
    separate adaptive thresholds for each wavelet subband based on local statistics.
    This produces higher quality denoising than VisuShrink by preserving more detail
    in regions with low noise and aggressive denoising in noisy regions.

    Use cases (agar plates):
    - Remove scanner noise and camera artifacts while preserving fine colony details.
    - Denoise images with spatially varying noise (uneven illumination causes
      varying noise levels across the plate).
    - Preserve colony texture and morphology better than VisuShrink or Gaussian blur.
    - Pre-filter for feature extraction when colony internal structure matters.

    Tuning and effects:
    - sigma: Noise standard deviation in [0, 1] scale. None (default) auto-estimates
      via MAD. Typical: 0.01-0.05 for scanner/camera noise. BayesShrink uses this
      to compute subband-specific thresholds, so accurate estimation improves quality.
    - wavelet: 'db2' (default) balances smoothness and locality. 'db4' preserves
      finer details. Must be orthogonal wavelet for proper noise statistics.
    - mode: 'soft' (default) produces smoother results; 'hard' preserves edges
      more aggressively but may leave noise.
    - wavelet_levels: None (default) uses max-3 automatically. More levels allow
      finer noise/signal separation.

    Caveats:
    - Requires more computation than VisuShrink due to subband-specific threshold
      calculation, but typically better quality.
    - Assumes Gaussian noise; may underperform with impulse noise (use MedianFilter).
    - Does not correct illumination gradients; use background subtraction first.
    - For very small colonies (few pixels), may slightly blur boundaries; consider
      VisuShrink or reduce wavelet_levels.

    Attributes:
        sigma (float | None): Noise standard deviation in [0, 1]. None = auto-
            estimate. Accurate sigma improves adaptive threshold quality.
        wavelet (str): Wavelet family ('db2', 'db4', 'sym2'). Default 'db2'.
        mode (Literal['soft', 'hard']): Thresholding mode. 'soft' recommended.
        wavelet_levels (int | None): Decomposition levels. None = max-3.

    Examples:
        .. dropdown:: Basic denoising with adaptive BayesShrink

            .. code-block:: python

                from phenotypic import Image
                from phenotypic.enhance import BayesShrinkEnhancer

                image = Image.imread('agar_plate.jpg')
                enhancer = BayesShrinkEnhancer()
                denoised = enhancer.apply(image)

                # Original data preserved, enh_gray denoised
                assert np.array_equal(image.rgb[:], denoised.rgb[:])
                assert np.array_equal(image.gray[:], denoised.gray[:])

        .. dropdown:: BayesShrink vs VisuShrink comparison

            .. code-block:: python

                from phenotypic import Image
                from phenotypic.enhance import BayesShrinkEnhancer, VisuShrinkEnhancer

                image = Image.imread('plate.jpg')

                # BayesShrink: Adaptive, preserves more detail
                bayes = BayesShrinkEnhancer().apply(image)

                # VisuShrink: Universal threshold, more aggressive smoothing
                visu = VisuShrinkEnhancer().apply(image)

                # Results are different
                assert not np.array_equal(bayes.enh_gray[:], visu.enh_gray[:])
                # BayesShrink typically preserves more fine structure

        .. dropdown:: Fine detail preservation for texture analysis

            .. code-block:: python

                from phenotypic import Image, ImagePipeline
                from phenotypic.enhance import BayesShrinkEnhancer, UnsharpMask
                from phenotypic.measure import MeasureFeatures

                image = Image.imread('high_res_plate.jpg')

                # Denoise while preserving colony texture
                pipeline = ImagePipeline()
                pipeline.add(BayesShrinkEnhancer(wavelet='db4'))  # Fine details
                pipeline.add(UnsharpMask(radius=1.5, amount=1.0))  # Enhance edges

                result = pipeline.apply(image)

                # Now measure morphology with full texture information
                measured = MeasureFeatures().apply(result)
                features = measured.objects
    """

    def __init__(
        self,
        sigma: float | None = None,
        wavelet: str = "db2",
        mode: Literal["soft", "hard"] = "soft",
        wavelet_levels: int | None = None,
    ):
        """Initialize BayesShrink adaptive wavelet denoiser.

        Parameters:
            sigma (float | None): Noise standard deviation in [0, 1] scale. None
                (default) auto-estimates. More accurate sigma improves adaptive
                thresholding quality. Typical: 0.01-0.05 for moderate noise.
            wavelet (str): Wavelet type. 'db2' (default) is general-purpose.
                'db4' for finer details, 'sym2' for symmetric filters.
            mode (Literal['soft', 'hard']): 'soft' (default) for smoother
                denoising, 'hard' for sharper edges with possible noise residue.
            wavelet_levels (int | None): Decomposition depth. None (default)
                uses max-3. Increase for very noisy images.
        """
        self.sigma = sigma
        self.wavelet = wavelet
        self.mode = mode
        self.wavelet_levels = wavelet_levels

    @staticmethod
    def _operate(
        image: Image,
        sigma: float | None = None,
        wavelet: str = "db2",
        mode: Literal["soft", "hard"] = "soft",
        wavelet_levels: int | None = None,
    ) -> Image:
        """Apply BayesShrink adaptive wavelet denoising to enhanced grayscale.

        Parameters:
            image: Image object to denoise
            sigma: Noise level in [0, 1] scale
            wavelet: Wavelet type
            mode: 'soft' or 'hard' thresholding
            wavelet_levels: Decomposition levels

        Returns:
            Modified Image with denoised enh_gray
        """
        denoised = denoise_wavelet(
            image=image.enh_gray[:],
            sigma=sigma,
            wavelet=wavelet,
            mode=mode,
            wavelet_levels=wavelet_levels,
            method="BayesShrink",
            channel_axis=None,
            rescale_sigma=True,
        )
        image.enh_gray[:] = denoised.clip(0.0, 1.0)
        return image
