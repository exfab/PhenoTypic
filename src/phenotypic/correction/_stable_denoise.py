from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import bm3d
import numpy as np
from bm3d.profiles import BM3DStages
from pydantic import field_validator

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from ..abc_ import ImageCorrector
from ..tools_._anscombe import gat_forward, gat_inverse


class StableDenoise(ImageCorrector):
    """Denoise grayscale channels using variance-stabilized BM3D collaborative filtering.

    Combine the Generalized Anscombe Transform (GAT) with BM3D denoising
    in a single corrector step. The GAT stabilizes Poisson-Gaussian noise
    variance so that BM3D operates optimally, then the inverse GAT
    restores the original intensity scale. Writing through the gray
    accessor triggers a detect_mat reset, so downstream reads reflect
    the denoised result.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Args:
        block_size: BM3D patch side length in pixels. Larger values
            capture more context but increase computation. Default: ``8``.
        stage_arg: Processing stages. ``'all_stages'`` runs hard
            thresholding followed by Wiener filtering for best quality;
            ``'hard_thresholding'`` is faster. Default: ``'all_stages'``.
        gain: Camera gain in electrons per ADU. Default: ``1.0``.
        mu: Read-noise mean (baseline offset). Default: ``0.0``.
        sigma: Read-noise standard deviation. ``0.0`` assumes pure
            Poisson noise, appropriate for most plate scanners.
            Default: ``0.0``.
        scale_factor: Multiplier converting normalized [0, 1] data to
            photon counts. ``None`` auto-detects from image bit depth.
            Default: ``None``.

    Returns:
        Image: Input image with grayscale channel denoised via the
        accessor cascade. RGB is unchanged.

    Raises:
        ValueError: If ``gain`` is not positive, ``sigma`` is negative,
            ``scale_factor`` is not positive, or ``stage_arg`` is not a
            recognized value.

    Best For:
        - Low-light or high-ISO plate images with photon-counting
          (Poisson-Gaussian) noise.
        - Improving intensity measurement accuracy before colony size
          or opacity quantification.
        - CCD/CMOS scanned plates where mixed noise models apply.

    Consider Also:
        - :class:`BayesShrinkCorrector` when all components (including
          RGB) need denoising simultaneously.
        - :class:`BM3DDenoiser` for enhancer-only BM3D on the detection
          matrix without modifying grayscale.
        - :class:`VisuShrinkCorrector` for a faster wavelet-based
          alternative when Poisson noise modelling is not required.

    References:
        [1] M. Makitalo and A. Foi, "Optimal inversion of the
        generalized Anscombe transformation for Poisson-Gaussian noise,"
        *IEEE Trans. Image Process.*, vol. 22, no. 1, pp. 91--103,
        Jan. 2013.

    See Also:
        :doc:`/how_to/notebooks/correct_color_cast` for combining
        denoising with color correction workflows.
    """

    block_size: int = 8
    stage_arg: Literal["all_stages", "hard_thresholding"] = "all_stages"
    gain: float = 1.0
    mu: float = 0.0
    sigma: float = 0.0
    scale_factor: float | None = None

    @field_validator("gain", mode="before")
    @classmethod
    def _validate_gain(cls, gain: float) -> float:
        """Require ``gain`` to be positive; coerce to ``float``.

        Reproduces the pre-migration ``__init__`` guard and ``float()``
        coercion exactly.
        """
        if gain <= 0:
            raise ValueError(f"gain must be > 0, got {gain}")
        return float(gain)

    @field_validator("mu", mode="before")
    @classmethod
    def _coerce_mu(cls, mu: float) -> float:
        """Coerce ``mu`` to ``float`` (pre-migration ``__init__`` did this)."""
        return float(mu)

    @field_validator("sigma", mode="before")
    @classmethod
    def _validate_sigma(cls, sigma: float) -> float:
        """Require ``sigma`` to be non-negative; coerce to ``float``.

        Reproduces the pre-migration ``__init__`` guard and ``float()``
        coercion exactly.
        """
        if sigma < 0:
            raise ValueError(f"sigma must be >= 0, got {sigma}")
        return float(sigma)

    @field_validator("scale_factor", mode="before")
    @classmethod
    def _validate_scale_factor(
            cls, scale_factor: float | None
    ) -> float | None:
        """Require ``scale_factor`` to be positive when supplied.

        Reproduces the pre-migration ``__init__`` guard and ``float()``
        coercion exactly: ``None`` passes through unchanged.
        """
        if scale_factor is None:
            return None
        if scale_factor <= 0:
            raise ValueError(f"scale_factor must be > 0, got {scale_factor}")
        return float(scale_factor)

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
        stabilized = gat_forward(counts, self.mu, self.sigma, self.gain)

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
        recovered = gat_inverse(denoised, self.mu, self.sigma, self.gain)

        # counts -> [0,1], clip
        return (recovered / scale_factor).clip(0.0, 1.0)

    def _operate(self, image: Image) -> Image:
        """Apply GAT-stabilized BM3D denoising to grayscale channel.

        Writes denoised result via ``image.gray[:]`` accessor, which
        triggers ``detect_mat.reset()`` so downstream detect_mat reads
        reflect the denoised grayscale.

        Returns:
            Modified Image with gray denoised via accessor cascade.
            RGB unchanged.
        """
        scale_factor = self._get_scale_factor(image)
        image.gray[:] = self._denoise_channel(image._data.gray, scale_factor)
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
