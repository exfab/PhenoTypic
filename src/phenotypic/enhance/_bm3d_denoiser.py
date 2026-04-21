from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from phenotypic._core._image import Image
import bm3d
from bm3d.profiles import BM3DStages

from ..abc_ import ImageEnhancer


class BM3DDenoiser(ImageEnhancer):
    """Denoise ``detect_mat`` with block-matching and 3D collaborative filtering.

    Groups similar image patches and filters them jointly in the transform
    domain, preserving fine colony details while removing structured noise
    patterns (scanner artifacts, systematic CCD noise, imaging hardware
    texture). Produces higher-quality results than simple Gaussian blur at
    significantly higher computational cost.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Args:
        sigma_psd: Noise standard deviation in [0, 1] normalized scale.
            Typical range: 0.01--0.05 for moderate noise, 0.05--0.15 for
            heavy noise. Too low preserves noise; too high removes colony
            texture. Default: 0.02.
        block_size: Block size for BM3D patch matching. Default: 8.
        stage_arg: Processing mode. ``'all_stages'`` (default) applies both
            hard thresholding and Wiener filtering for highest quality;
            ``'hard_thresholding'`` runs only the first stage for faster
            processing.
        clip: Clip output to [0, 1]. Default: ``True``. Set to ``False``
            when using with variance-stabilizing transforms (e.g., GAT).

    Returns:
        Image: Input image with ``detect_mat`` denoised via BM3D
        collaborative filtering. ``rgb`` and ``gray`` are unchanged.

    Best For:
        - Structured camera or scanner noise on plate images.
        - Low-light imaging where high ISO introduces patterned noise.
        - Preserving fine morphological features (wrinkles, satellite
          colonies) during denoising.

    Consider Also:
        - :class:`BilateralDenoise` for faster edge-preserving denoising
          when structured noise is not the primary concern.
        - :class:`NonLocalMeansDenoiser` for patch-based denoising with
          lower computational overhead.
        - :class:`VisuShrinkEnhancer` for fast wavelet denoising when
          speed matters more than quality.

    References:
        [1] K. Dabov, A. Foi, V. Katkovnik, and K. Egiazarian, "Image
        denoising by sparse 3-D transform-domain collaborative filtering,"
        *IEEE Trans. Image Process.*, vol. 16, no. 8, pp. 2080--2095,
        Aug. 2007.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a
        visual walkthrough of denoising pipelines on plate images.
        :doc:`/how_to/notebooks/denoise_low_light` for BM3D and other
        denoising strategies on low-light plate images.
    """

    def __init__(
            self,
            sigma_psd: float = 0.02,
            block_size: int = 8,
            *,
            stage_arg: Literal["all_stages", "hard_thresholding"] = "all_stages",
            clip: bool = True,
    ):
        """
        Parameters:
            sigma_psd (float): Noise level estimate in [0, 1] normalized
                scale. Start with 0.02-0.05 for typical scanner noise on
                plates (equivalent to σ=5-12 on 8-bit).
                Higher value -> more noise.
            block_size (int): Block size for BM3D denoising. Default is 8.
            stage_arg (Literal["all_stages", "hard_thresholding"]): Denoising
                stages to run. 'all_stages' gives best quality at the cost of
                speed; 'hard_thresholding' is faster and adequate for routine
                plate analysis.
            clip (bool): Whether to clip output to [0, 1] range. Default True.
                Set to False when using with variance-stabilizing transforms
                (e.g., GAT) that require preserving the original scale.
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

        self.block_size = block_size
        self.clip = clip

    def _operate(self, image: Image) -> Image:
        # detect_mat is guaranteed to be in [0, 1] range, which BM3D expects
        profile = bm3d.BM3DProfile()
        profile.bs_ht = self.block_size
        profile.bs_wiener = self.block_size
        denoised = bm3d.bm3d(
                image.detect_mat[:],
                profile=profile,
                sigma_psd=self.sigma_psd,
                stage_arg=self._convert_stage_arg(self.stage_arg),
        )

        if self.clip:
            denoised = denoised.clip(0.0, 1.0)
        image.detect_mat[:] = denoised
        return image

    def _convert_stage_arg(self, stage_arg: Literal["all_stages", "hard_thresholding"]):
        match stage_arg:
            case "hard_thresholding":
                return BM3DStages.HARD_THRESHOLDING
            case "all_stages":
                return BM3DStages.ALL_STAGES
            case _:
                raise ValueError(f"Unknown stage arg: {stage_arg}")
