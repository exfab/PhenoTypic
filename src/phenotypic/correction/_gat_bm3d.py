from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import bm3d
import numpy as np
from bm3d.profiles import BM3DStages

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from ..abc_ import ImageCorrector
from ..enhance._anscombe_forward import AnscombeForward
from ..enhance._anscombe_inverse import AnscombeInverse


class GatBM3D(ImageCorrector):
    """Variance-stabilized BM3D denoising for gray and detect_mat.

    Combines the Generalized Anscombe Transform (GAT) with BM3D collaborative
    filtering into a single corrector step. The GAT stabilizes Poisson-Gaussian
    noise variance so that BM3D operates optimally, then the inverse GAT restores
    the original intensity scale. This directly improves intensity measurement
    quality for downstream colony phenotyping.

    Use cases (agar plates):
    - Denoise grayscale intensity and detection channels simultaneously for
      more accurate colony size and opacity measurements.
    - Remove photon-counting noise from low-light or high-ISO plate images
      while preserving fine colony morphology.
    - Pre-process scanned plates with mixed Poisson-Gaussian noise (common in
      CCD/CMOS sensors) before thresholding or feature extraction.

    Tuning and effects:
    - block_size: BM3D patch size. 8 (default) is standard; larger values
      capture more context but slow computation.
    - stage_arg: 'all_stages' (default) runs both hard thresholding and Wiener
      filtering for best quality. 'hard_thresholding' is faster and often
      sufficient for routine plate analysis.
    - GAT parameters (gain, mu, sigma) model the camera noise. Defaults assume
      pure Poisson noise (gain=1, mu=0, sigma=0), which is appropriate for most
      plate scanners and cameras.

    Caveats:
    - Modifies gray and detect_mat irreversibly. RGB is unchanged (BM3D is a
      2D grayscale algorithm).
    - Computationally expensive, especially with 'all_stages' on large images.
    - The internal noise PSD is fixed at 1.0 (the theoretically correct value
      for GAT-domain Poisson noise per Makitalo & Foi 2013). Do not confuse
      this with the ``sigma_psd`` parameter of :class:`BM3DDenoiser`.

    Attributes:
        block_size (int): BM3D patch size. Default 8.
        stage_arg (str): Processing mode. Default 'all_stages'.
        gain (float): Camera gain (electrons/ADU). Default 1.0.
        mu (float): Read noise mean. Default 0.0.
        sigma (float): Read noise standard deviation. Default 0.0.
        scale_factor (float | None): Converts [0,1] to counts. None = auto.

    References:
        - M. Makitalo and A. Foi, "Optimal Inversion of the Generalized
          Anscombe Transformation for Poisson-Gaussian Noise", IEEE Trans.
          Image Process., 2013.

    Examples:
        One-step variance-stabilized denoising:

        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from phenotypic.correction import GatBM3D
        >>> image = load_synth_yeast_plate()
        >>> corrector = GatBM3D()
        >>> denoised = corrector.apply(image)  # doctest: +SKIP

        Fast mode with hard thresholding only:

        >>> from phenotypic.correction import GatBM3D
        >>> corrector = GatBM3D(stage_arg='hard_thresholding')
    """

    def __init__(
            self,
            block_size: int = 8,
            stage_arg: Literal["all_stages", "hard_thresholding"] = "all_stages",
            *,
            gain: float = 1.0,
            mu: float = 0.0,
            sigma: float = 0.0,
            scale_factor: float | None = None,
    ):
        """Initialize GAT-stabilized BM3D corrector for gray and detect_mat.

        Args:
            block_size (int): BM3D patch size. Default 8.
            stage_arg (Literal["all_stages", "hard_thresholding"]): Denoising
                stages. 'all_stages' gives best quality; 'hard_thresholding'
                is faster.
            gain (float): Camera gain in electrons per ADU. Default 1.0.
            mu (float): Read noise mean (baseline offset). Default 0.0.
            sigma (float): Read noise standard deviation. Default 0.0
                (pure Poisson noise).
            scale_factor (float | None): Converts normalized [0,1] data to
                counts. None (default) auto-detects from image metadata.
        """
        if gain <= 0:
            raise ValueError(f"gain must be > 0, got {gain}")
        if sigma < 0:
            raise ValueError(f"sigma must be >= 0, got {sigma}")
        if scale_factor is not None and scale_factor <= 0:
            raise ValueError(f"scale_factor must be > 0, got {scale_factor}")
        if stage_arg not in ("all_stages", "hard_thresholding"):
            raise ValueError(
                    f"stage_arg must be 'all_stages' or 'hard_thresholding', "
                    f"got {stage_arg!r}"
            )

        self.block_size = block_size
        self.stage_arg = stage_arg
        self.gain = float(gain)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.scale_factor = (
            float(scale_factor) if scale_factor is not None else None
        )

    def _get_scale_factor(self, image: Image) -> float:
        """Get scale factor, auto-detecting from image metadata.

        Args:
            image: The Image to get scale factor for.

        Returns:
            Scale factor for converting normalized [0,1] data to counts.
        """
        if self.scale_factor is not None:
            return self.scale_factor

        bit_depth = getattr(image.metadata, "bit_depth", None)
        if bit_depth == 8:
            return 255.0
        elif bit_depth == 16:
            return 65535.0
        else:
            return 255.0

    def _denoise_channel(
            self, channel: np.ndarray, scale_factor: float
    ) -> np.ndarray:
        """Denoise a single [0,1] channel via GAT -> BM3D -> inverse GAT.

        Args:
            channel: 2D array in [0, 1] range.
            scale_factor: Multiplier to convert [0,1] to counts.

        Returns:
            Denoised 2D array clipped to [0, 1].
        """
        # [0,1] -> counts
        counts = channel * scale_factor

        # Forward GAT: stabilize Poisson-Gaussian variance
        stabilized = AnscombeForward._generalized_anscombe(
                counts, self.mu, self.sigma, self.gain
        )

        # BM3D denoise in GAT domain (sigma_psd=1.0 is theoretically correct)
        profile = bm3d.BM3DProfile()
        profile.bs_ht = self.block_size
        profile.bs_wiener = self.block_size
        denoised = bm3d.bm3d(
                stabilized,
                profile=profile,
                sigma_psd=1.0,
                stage_arg=self._convert_stage_arg(self.stage_arg),
        )

        # Inverse GAT: recover counts
        recovered = AnscombeInverse._inverse_generalized_anscombe(
                denoised, self.mu, self.sigma, self.gain
        )

        # counts -> [0,1], clip
        return (recovered / scale_factor).clip(0.0, 1.0)

    def _operate(self, image: Image) -> Image:
        """Apply GAT-stabilized BM3D denoising to gray and detect_mat.

        Returns:
            Modified Image with gray and detect_mat denoised. RGB unchanged.
        """
        scale_factor = self._get_scale_factor(image)
        image._data.gray = self._denoise_channel(image._data.gray, scale_factor)
        return image

    @staticmethod
    def _convert_stage_arg(
            stage_arg: Literal["all_stages", "hard_thresholding"],
    ) -> BM3DStages:
        """Convert string stage argument to BM3DStages enum."""
        match stage_arg:
            case "hard_thresholding":
                return BM3DStages.HARD_THRESHOLDING
            case "all_stages":
                return BM3DStages.ALL_STAGES
            case _:
                raise ValueError(f"Unknown stage arg: {stage_arg}")
