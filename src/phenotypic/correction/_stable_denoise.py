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
from ..tools_.colourspace import decode_srgb, encode_srgb


class StableDenoise(ImageCorrector):
    """Denoise the grayscale channel using GAT-stabilized BM3D collaborative filtering.

    Combine the Generalized Anscombe Transform (GAT) with BM3D block-matching
    and 3D filtering in a single corrector step. The forward GAT converts
    Poisson-Gaussian mixed noise to approximately unit-variance Gaussian so BM3D
    operates with a theoretically correct noise model; the exact unbiased inverse
    GAT then restores the original intensity scale. Writing back through the gray
    accessor automatically resets ``detect_mat``, so downstream reads reflect the
    denoised result.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Best For:
        - Low-light or high-ISO plate images where photon shot noise (Poisson
          component) is comparable to or larger than read noise.
        - Improving intensity measurement accuracy before colony size or
          opacity quantification on CCD/sCMOS imaged plates.
        - Fluorescence plate reader images with signal-dependent noise where
          a purely Gaussian denoiser would leave structured residuals.

    Consider Also:
        - :class:`BayesShrinkCorrector` when all image components including
          RGB need simultaneous wavelet denoising.
        - :class:`BM3DDenoiser` for BM3D applied only to the detection matrix
          without altering the grayscale channel.
        - :class:`VisuShrinkCorrector` for a faster wavelet alternative when
          Poisson noise modelling is not required.

    Args:
        block_size: Side length in pixels of the 2D patches used in BM3D
            block-matching. Larger values capture more spatial context and
            denoise smooth agar backgrounds more effectively; smaller values
            preserve fine colony texture and thin hyphal edges. Typical range:
            4--16. Default: ``8``.
        stage_arg: BM3D processing stages. ``'all_stages'`` runs the
            two-stage pipeline (hard-thresholding basic estimate followed by
            Wiener collaborative filtering) for best denoising quality.
            ``'hard_thresholding'`` skips the Wiener stage for roughly
            40--50% faster execution at the cost of more residual noise.
            Default: ``'all_stages'``.
        gain: Camera gain in electrons per ADU for the GAT noise model.
            Scales the Poisson variance component in the forward transform.
            Typical range: 0.5--10 e-/ADU depending on sensor; obtain from the
            sensor datasheet or a mean-variance calibration. Must be positive.
            Default: ``1.0``.
        mu: Read-noise mean (dark-current baseline offset) in count units
            consistent with ``scale_factor``. Set to 0.0 when the image has
            been bias-subtracted, which covers most plate-scanner workflows.
            Default: ``0.0``.
        sigma: Read-noise standard deviation in count units consistent with
            ``scale_factor``. ``0.0`` assumes pure Poisson noise, appropriate
            for most plate scanners under normal exposure. Supplying the sensor
            read-noise spec (a few to a few tens of counts for typical
            scientific CCDs) improves stabilization in low-signal regions. Must
            be non-negative. Default: ``0.0``.
        scale_factor: Multiplier converting normalized [0, 1] grayscale data
            to photon counts before the forward GAT. ``None`` auto-detects
            from image bit depth (8-bit: 255, 16-bit: 65535). Override for
            non-standard bit depths such as 12-bit sensors stored in a 16-bit
            container (4095). Must be positive when supplied. Default:
            ``None``.

    Returns:
        Image: Input image with the grayscale channel replaced by the
        GAT-stabilized BM3D-denoised result. ``rgb`` is unchanged; ``detect_mat``
        is automatically reset via the gray accessor cascade.

    Raises:
        ValueError: If ``gain`` is not positive, ``sigma`` is negative,
            ``scale_factor`` is not positive when supplied, or ``stage_arg``
            is not a recognized value.

    References:
        [1] K. Dabov, A. Foi, V. Katkovnik, and K. Egiazarian, "Image
        denoising by sparse 3-D transform-domain collaborative filtering,"
        *IEEE Trans. Image Process.*, vol. 16, no. 8, pp. 2080--2095,
        Aug. 2007.

        [2] M. Mäkitalo and A. Foi, "Optimal inversion of the generalized
        Anscombe transformation for Poisson-Gaussian noise," *IEEE Trans.
        Image Process.*, vol. 22, no. 1, pp. 91--103, Jan. 2013.

    See Also:
        :doc:`/how_to/notebooks/denoise_low_light` for a walkthrough of
        denoising low-light plate images.
        :doc:`/explanation/image_quality_noise_contrast_structure` for
        background on Poisson-Gaussian noise models and denoiser selection.
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

        The channel is assumed sRGB-encoded: it is decoded to linear
        light before stabilization and re-encoded to sRGB afterwards, so
        the GAT's Poisson-Gaussian model operates in linear light rather
        than on gamma-warped data.

        Caveat: the grayscale channel is a luma sum of gamma-encoded RGB
        (``skimage.color.rgb2gray``). Decoding that sum with the
        per-channel sRGB curve only *approximates* true linear luminance
        -- far better than no linearization, but not exact. The exact fix
        would linearize RGB per-channel then recompute luminance.

        Args:
            channel: 2D array in [0, 1] range.
            scale_factor: Multiplier to convert [0,1] to counts.

        Returns:
            Denoised 2D array clipped to [0, 1].
        """
        # sRGB -> linear light so the GAT sees linear photon counts
        working = decode_srgb(channel)

        # [0,1] -> counts
        counts = working * scale_factor

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
        result = (recovered / scale_factor).clip(0.0, 1.0)

        # linear light -> sRGB to restore the original encoding
        return encode_srgb(result).clip(0.0, 1.0)

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
