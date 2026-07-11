from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import bm3d
import numpy as np
from pydantic import field_validator

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from ..abc_ import ImageCorrector
from ..sdk_.mixin import NormalizedOutputMixin
from ..sdk_.typing_ import TuneSpec
from ..sdk_._anscombe import gat_forward, gat_inverse, resolve_scale_factor
from ..sdk_.colourspace import decode_srgb, encode_srgb


class ColorDenoise(NormalizedOutputMixin, ImageCorrector):
    """Denoise an RGB plate image using color block-matching 3D filtering (CBM3D).

    Apply the color extension of BM3D jointly across the three sRGB channels
    in the linear-light domain, decorrelating color into a luminance-chrominance
    space and computing patch grouping once on the luminance channel for reuse
    across the chrominance channels. Writing the cleaned RGB through the accessor
    cascade automatically rebuilds ``gray`` and ``detect_mat`` from the corrected
    data.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Best For:
        - RGB plate scans where colony color composition is subsequently
          measured and chrominance fidelity at colony boundaries must be
          preserved.
        - Structured camera or scanner sensor noise that should be suppressed
          while preserving colony color fidelity.
        - Publication-quality plate figures requiring visually clean,
          color-accurate RGB images.
        - Low-light or high-ISO DSLR plate captures with Poisson-Gaussian
          mixed noise, using ``use_gat=True``.

    Consider Also:
        - :class:`BayesShrinkCorrector` for faster wavelet-based denoising of
          all components when CBM3D runtime is prohibitive.
        - :class:`DenoiseBlockMatch` for GAT-stabilized BM3D on the grayscale
          channel when only intensity measurements are needed.
        - :class:`NonLocalMeansDenoiser` when a simpler, lower-cost denoiser
          for the detection matrix is sufficient.

    Args:
        sigma_psd: Gaussian noise standard deviation in the linear-light
            [0, 1] domain after sRGB decoding. Typical range: 0.01--0.05
            for moderate flatbed-scanner noise; 0.05--0.15 for heavy noise
            from high-ISO or low-light captures. Too low leaves structured
            noise intact; too high erases fine colony texture and inter-colony
            gap detail. Ignored when ``use_gat=True``; the stabilized-domain
            value 1.0 is used internally. Default: ``0.02``.
        block_size: Side length in pixels of the 2D patches used for
            block-matching. Sets both the hard-thresholding and Wiener
            stage block sizes. Larger values capture more spatial context
            and improve denoising of uniform agar backgrounds but increase
            computation quadratically. Typical range: 4--16. Default: ``8``.
        norm: Output range policy applied to the sRGB-re-encoded result before
            rescaling to the original integer dtype. ``"clip"`` (default)
            saturates values outside [0, 1], preventing rare BM3D overshoot
            near high-contrast colony edges from causing integer wraparound.
            ``"rescale"`` remaps the full observed range onto [0, 1]; ``None``
            passes values through untouched.

        # GAT parameters (active only when use_gat=True)
        use_gat: Wrap the per-channel CBM3D call in the Generalized Anscombe
            Transform, converting Poisson-Gaussian mixed noise to
            approximately unit-variance Gaussian so BM3D operates optimally.
            Enable for fluorescence plate readers or high-ISO DSLR images
            where shot noise dominates. Default: ``False``.
        gat_gain: Camera gain in electrons per ADU for the GAT noise model.
            Scales the Poisson variance component. Typical range: 0.5--10
            e-/ADU depending on sensor; obtain from the manufacturer datasheet
            or a mean-variance calibration. Default: ``1.0``.
        gat_mu: Read-noise mean (dark-current baseline offset) in count units
            consistent with ``gat_scale_factor``. Set to 0.0 when the image
            has been bias-subtracted, which covers most plate-scanner
            workflows. Default: ``0.0``.
        gat_read_sigma: Read-noise standard deviation in count units
            consistent with ``gat_scale_factor``. Zero assumes pure Poisson
            noise; supplying the sensor read-noise spec (a few to a few tens of
            counts for typical scientific CCDs) improves stabilization accuracy
            at low signal levels. Default: ``0.0``.
        gat_scale_factor: Multiplier converting normalized [0, 1] linear-light
            data to photon counts before the forward GAT. ``None``
            auto-detects from the image bit depth (8-bit: 255, 16-bit:
            65535). Override for non-standard bit depths such as 12-bit
            sensors stored in a 16-bit container (4095). Default: ``None``.

    Returns:
        Image: Input image with ``rgb`` replaced by the CBM3D-denoised
        result. ``gray`` and ``detect_mat`` are automatically recomputed
        from the cleaned RGB via the accessor cascade.

    Raises:
        ValueError: If the image has no RGB data, if ``sigma_psd`` is
            negative, ``block_size`` is not positive, ``gat_gain`` is not
            positive, ``gat_read_sigma`` is negative, or
            ``gat_scale_factor`` is not positive.

    References:
        [1] K. Dabov, A. Foi, V. Katkovnik, and K. Egiazarian, "Image
        denoising by sparse 3-D transform-domain collaborative filtering,"
        *IEEE Trans. Image Process.*, vol. 16, no. 8, pp. 2080--2095,
        Aug. 2007.

        [2] K. Dabov, A. Foi, V. Katkovnik, and K. Egiazarian, "Color
        image denoising via sparse 3D collaborative filtering with
        grouping constraint in luminance-chrominance space," in *Proc.
        IEEE Int. Conf. Image Process. (ICIP)*, 2007, pp. I-313--I-316.

        [3] M. Mäkitalo and A. Foi, "Optimal inversion of the generalized
        Anscombe transformation for Poisson-Gaussian noise," *IEEE Trans.
        Image Process.*, vol. 22, no. 1, pp. 91--103, Jan. 2013.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a
        visual walkthrough of denoising pipelines on plate images.
        :doc:`/explanation/image_quality_noise_contrast_structure` for
        background on noise models and denoiser selection.
    """

    sigma_psd: Annotated[float, TuneSpec(0.01, 0.15, log=True)] = 0.02
    block_size: Annotated[int, TuneSpec(categories=(4, 8, 16))] = 8
    use_gat: bool = False
    # GAT noise-model parameters are sensor-calibration constants, not search
    # targets — keep their Field bounds, exclude from tuning.
    gat_gain: Annotated[float, TuneSpec(tunable=False)] = 1.0
    gat_mu: Annotated[float, TuneSpec(tunable=False)] = 0.0
    gat_read_sigma: Annotated[float, TuneSpec(tunable=False)] = 0.0
    gat_scale_factor: Annotated[float | None, TuneSpec(tunable=False)] = None

    @field_validator("sigma_psd")
    @classmethod
    def _check_sigma_psd(cls, sigma_psd: float) -> float:
        """Require a non-negative noise estimate."""
        if sigma_psd < 0:
            raise ValueError(f"sigma_psd must be non-negative, got {sigma_psd}")
        return sigma_psd

    @field_validator("block_size")
    @classmethod
    def _check_block_size(cls, block_size: int) -> int:
        """Require a positive BM3D block size."""
        if block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size}")
        return block_size

    @field_validator("gat_gain")
    @classmethod
    def _check_gat_gain(cls, gat_gain: float) -> float:
        """Require a positive camera gain for the GAT."""
        if gat_gain <= 0:
            raise ValueError(f"gat_gain must be > 0, got {gat_gain}")
        return gat_gain

    @field_validator("gat_read_sigma")
    @classmethod
    def _check_gat_read_sigma(cls, gat_read_sigma: float) -> float:
        """Require a non-negative read-noise standard deviation."""
        if gat_read_sigma < 0:
            raise ValueError(
                    f"gat_read_sigma must be >= 0, got {gat_read_sigma}"
            )
        return gat_read_sigma

    @field_validator("gat_scale_factor")
    @classmethod
    def _check_gat_scale_factor(
            cls, gat_scale_factor: float | None
    ) -> float | None:
        """Require a positive scale factor when supplied; ``None`` passes through."""
        if gat_scale_factor is None:
            return None
        if gat_scale_factor <= 0:
            raise ValueError(
                    f"gat_scale_factor must be > 0, got {gat_scale_factor}"
            )
        return gat_scale_factor

    def _operate(self, image: Image) -> Image:
        """Apply CBM3D denoising to the RGB channels.

        The image is assumed sRGB-encoded: it is decoded to linear light
        before filtering and re-encoded to sRGB afterwards. The denoised
        RGB is written through ``set_image``, whose ``_set_from_array``
        cascade rebuilds ``gray`` and ``detect_mat``.
        """
        if image.rgb.isempty():
            raise ValueError(
                    "ColorDenoise requires a 3-channel RGB image; this image "
                    "has no RGB data."
            )

        # Read via the private data attribute: the public ``rgb[:]`` getter
        # marks the underlying array non-writeable, which would poison the
        # write-back below.
        raw = np.asarray(image._data.rgb)
        dtype = raw.dtype
        vmax = np.iinfo(dtype).max
        rgb01 = raw.astype(np.float64) / vmax

        # Plate captures are assumed sRGB-encoded. Denoise in linear light
        # -- BM3D's Gaussian-noise model and the GAT's Poisson model are
        # both correct there -- then re-encode the result to sRGB.
        rgb_lin = decode_srgb(rgb01)

        if self.use_gat:
            denoised = self._denoise_gat(rgb_lin, image)
        else:
            denoised = self._denoise_plain(rgb_lin)

        denoised = self._apply_norm(encode_srgb(denoised))

        # Always clamp to the dtype range to avoid integer wraparound.
        rescaled = (denoised * vmax).round().clip(0, vmax).astype(dtype)
        # set_image() is the documented full-replacement API; it routes
        # through _set_from_array, rebuilding gray and detect_mat from the
        # cleaned RGB.
        image.set_image(rescaled)
        return image

    def _build_profile(self) -> bm3d.BM3DProfile:
        """Build a BM3D profile with the configured block size."""
        profile = bm3d.BM3DProfile()
        profile.bs_ht = self.block_size
        profile.bs_wiener = self.block_size
        return profile

    def _denoise_plain(self, rgb_lin: np.ndarray) -> np.ndarray:
        """Run CBM3D directly on linear-light RGB in [0, 1]."""
        denoised = bm3d.bm3d_rgb(
                rgb_lin, self.sigma_psd, self._build_profile(), "opp"
        )
        return np.asarray(denoised, dtype=np.float64)

    def _denoise_gat(self, rgb_lin: np.ndarray, image: Image) -> np.ndarray:
        """Run CBM3D in the GAT-stabilized domain.

        Each channel is converted to photon counts, variance-stabilized
        by the forward GAT, denoised with ``sigma_psd=1.0`` (theoretically
        correct in the stabilized domain), then restored by the
        closed-form inverse GAT.
        """
        scale = resolve_scale_factor(image, self.gat_scale_factor)
        counts = rgb_lin * scale
        stabilized = gat_forward(
                counts, self.gat_mu, self.gat_read_sigma, self.gat_gain
        )
        denoised = bm3d.bm3d_rgb(stabilized, 1.0, self._build_profile(), "opp")
        recovered = gat_inverse(
                denoised, self.gat_mu, self.gat_read_sigma, self.gat_gain
        )
        return np.asarray(recovered / scale, dtype=np.float64)
