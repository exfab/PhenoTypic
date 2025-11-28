from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING: from phenotypic import Image
import bm3d
from bm3d.profiles import BM3DStages

from ..abc_ import ImageEnhancer


class BM3DDenoiser(ImageEnhancer):
    """
    Block-matching and 3D collaborative filtering denoising for plate images.

    Applies BM3D, a state-of-the-art denoising algorithm that groups similar
    image patches and performs collaborative filtering in the transform domain.
    This is particularly effective for removing structured noise patterns and
    preserving fine colony details on agar plates (e.g., scanner artifacts,
    systematic CCD noise, or subtle textures from imaging hardware).

    Use cases (agar plates):
    - Remove structured camera/scanner noise while preserving sharp colony edges
      and fine morphological features (e.g., wrinkles, satellite colonies).
    - Suppress noise from low-light imaging or high ISO settings without the
      over-smoothing typical of simple Gaussian blur.
    - Pre-process before edge detection or feature extraction when image
      quality is poor but colony structures must remain intact.

    Tuning and effects:
    - sigma_psd: Estimated noise standard deviation in [0, 1] scale (matching
      normalized image range). Start with automatic estimation (None) or typical
      values 0.01-0.05 for moderate noise, 0.05-0.15 for very noisy images.
      For reference: 8-bit noise of σ=10/255 ≈ 0.04; 16-bit noise of σ=300/65535
      ≈ 0.005. Too low preserves noise; too high removes real colony texture.
    - stage_arg: Controls whether to run fast ('hard_thresholding') or complete
      ('all_stages') denoising. 'all_stages' produces cleaner results but is
      slower; 'hard_thresholding' is faster and often sufficient for plates.
    - Operates on normalized [0,1] float data directly from enh_gray.

    Caveats:
    - Computationally expensive, especially on high-resolution images. Consider
      downsampling or cropping for exploratory analysis.
    - Requires accurate `sigma_psd` estimate in [0, 1] scale. If unknown, use
      None for auto-estimation or test multiple values to avoid under/over-
      denoising. Noise magnitude differs between 8-bit and 16-bit originals.
    - Does not correct illumination gradients; combine with background
      subtraction (e.g., `GaussianSubtract`, `RollingBallRemoveBG`) if needed.
    - May slightly blur very fine colony features if sigma_psd is too high.

    Attributes:
        sigma_psd (float | None): Noise standard deviation in [0, 1] normalized
            scale. If None, BM3D auto-estimates from the image. Typical values:
            0.01-0.05 for moderate noise (e.g., 8-bit with σ=5-15), 0.05-0.15
            for heavy noise. 16-bit images typically have lower relative noise.
        stage_arg (Literal["all_stages", "hard_thresholding"]): Processing mode.
            'all_stages' applies both hard thresholding and Wiener filtering
            (slower, highest quality); 'hard_thresholding' runs only the first
            stage (faster, good for most plates).
    """

    def __init__(
            self,
            sigma_psd: float = 0.02,
            *,
            stage_arg: Literal["all_stages", "hard_thresholding"] = "all_stages",
    ):
        """
        Parameters:
            sigma_psd (float): Noise level estimate in [0, 1] normalized
                scale. None for auto-estimation; otherwise specify as standard
                deviation matching the normalized image range. Start with 0.02-0.05
                for typical scanner noise on plates (equivalent to σ=5-12 on 8-bit).
                Higher value -> more noise.
            stage_arg (Literal["all_stages", "hard_thresholding"]): Denoising
                stages to run. 'all_stages' gives best quality at the cost of
                speed; 'hard_thresholding' is faster and adequate for routine
                plate analysis.
        """
        if not isinstance(sigma_psd, (int, float)):
            raise TypeError("sigma_psd must be a number or None")
        if sigma_psd < 0:
            raise ValueError("sigma_psd must be non-negative")
        self.sigma_psd = float(sigma_psd)

        if stage_arg not in ["all_stages", "hard_thresholding"]:
            raise ValueError("stage_arg must be 'all_stages' or 'hard_thresholding'")
        else:
            self.stage_arg = stage_arg

    def _operate(self, image: Image) -> Image:
        # enh_gray is guaranteed to be in [0, 1] range, which BM3D expects

        denoised = bm3d.bm3d(
                image.enh_gray[:],
                sigma_psd=self.sigma_psd,
                stage_arg=self._convert_stage_arg(self.stage_arg),
        )

        image.enh_gray[:] = denoised
        return image

    def _convert_stage_arg(self, stage_arg: Literal["all_stages", "hard_thresholding"]):

        match stage_arg:
            case "hard_thresholding":
                return BM3DStages.HARD_THRESHOLDING
            case "all_stages":
                return BM3DStages.ALL_STAGES
            case _:
                raise ValueError(f"Unknown stage arg: {stage_arg}")
