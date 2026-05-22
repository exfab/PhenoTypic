from __future__ import annotations

from typing import TYPE_CHECKING

import bm3d
import numpy as np
from pydantic import field_validator

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from ..abc_ import ImageCorrector
from ..tools_._anscombe import gat_forward, gat_inverse, resolve_scale_factor
from ..tools_.colourspace import decode_srgb, encode_srgb


class ColorDenoise(ImageCorrector):
    """Denoise an RGB plate image with color block-matching 3D filtering (CBM3D).

    Run the color extension of BM3D jointly across the three RGB channels,
    preserving inter-channel color fidelity while removing structured
    sensor noise. Unlike the grayscale denoisers (:class:`BM3DDenoiser`,
    :class:`StableDenoise`), CBM3D decorrelates color into a
    luminance-chrominance opponent space, computes patch grouping **once**
    on the luminance channel, and reuses those groups for all three
    channels. The shared grouping prevents color fringing and is markedly
    cheaper than denoising each channel independently.

    As an :class:`~phenotypic.abc_.ImageCorrector`, the denoised RGB is
    written back through the ``rgb`` accessor, so ``gray`` and
    ``detect_mat`` are automatically rebuilt from the cleaned RGB.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Args:
        sigma_psd: Gaussian noise standard deviation, interpreted in the
            **linear-light** domain (after sRGB linearization). Typical
            range: 0.01--0.05 for moderate noise, 0.05--0.15 for heavy
            noise. Too low preserves noise; too high erases colony
            texture. Ignored when ``use_gat=True`` (the stabilized-domain
            value 1.0 is used internally). Default: 0.02.
        block_size: Side length in pixels of the BM3D patches; sets both
            the hard-thresholding and Wiener block sizes. Larger blocks
            capture more context at higher cost. Default: 8.
        clip: Clip the result to [0, 1] before rescaling to the original
            dtype. Default: ``True``. Out-of-range values are always
            clamped to the dtype range regardless, to avoid integer
            wraparound.
        use_gat: Wrap denoising in the per-channel Generalized Anscombe
            Transform for Poisson-Gaussian sensor noise. After
            stabilization the RGB noise is approximately unit-variance
            white Gaussian, so ``sigma_psd=1.0`` is used internally.
            Default: ``False``.
        gat_gain: Camera gain in electrons per ADU, used by the GAT.
            Default: 1.0.
        gat_mu: Read-noise mean (baseline offset) used by the GAT.
            Default: 0.0.
        gat_read_sigma: Read-noise standard deviation used by the GAT.
            Default: 0.0.
        gat_scale_factor: Multiplier converting normalized [0, 1] data to
            photon counts. ``None`` auto-detects from the image bit depth
            (8-bit -> 255, 16-bit -> 65535). Default: ``None``.

    Returns:
        Image: Input image with ``rgb`` denoised by CBM3D. ``gray`` and
        ``detect_mat`` are recomputed from the cleaned RGB via the
        accessor cascade.

    Raises:
        ValueError: If the image has no RGB data, if ``sigma_psd`` is
            negative, ``block_size`` is not positive, ``gat_gain`` is not
            positive, ``gat_read_sigma`` is negative, or
            ``gat_scale_factor`` is not positive.

    Best For:
        - RGB plate scans where colony color composition is measured and
          chrominance fidelity must be preserved.
        - Publication-quality figures requiring color-accurate denoising.
        - Structured camera/scanner noise that should be removed without
          introducing color fringing at colony boundaries.
        - Low-light or high-ISO captures with Poisson-Gaussian noise via
          ``use_gat=True``.

    Consider Also:
        - :class:`BM3DDenoiser` for BM3D on the detection matrix only
          (non-destructive to RGB and gray).
        - :class:`StableDenoise` for variance-stabilized BM3D on the
          grayscale channel.
        - :class:`BayesShrinkCorrector` for faster wavelet denoising of
          all components when CBM3D's runtime is prohibitive.

    Notes:
        The input is assumed to be sRGB-encoded (the standard for plate
        captures). The RGB is decoded to linear light before filtering --
        BM3D's Gaussian-noise model and the GAT's Poisson model are both
        most correct in linear light -- and the denoised result is
        re-encoded to sRGB, so the output carries the same encoding as
        the input. The denoiser always uses the ``'opp'`` opponent color
        transform internally.

        **Performance:** CBM3D is computationally expensive -- denoising a
        full-resolution plate image can take minutes. Crop to a region of
        interest or downsample for interactive tuning.

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

    Examples:
        Construct the corrector with custom parameters:

        >>> from phenotypic.correction import ColorDenoise
        >>> denoiser = ColorDenoise(sigma_psd=0.03, block_size=8)
        >>> denoiser.sigma_psd
        0.03
        >>> denoiser.use_gat
        False

        The corrector is fully serializable for reproducible pipelines:

        >>> restored = ColorDenoise.model_validate(denoiser.model_dump())
        >>> restored.sigma_psd == denoiser.sigma_psd
        True
    """

    sigma_psd: float = 0.02
    block_size: int = 8
    clip: bool = True
    use_gat: bool = False
    gat_gain: float = 1.0
    gat_mu: float = 0.0
    gat_read_sigma: float = 0.0
    gat_scale_factor: float | None = None

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

        denoised = encode_srgb(denoised)

        if self.clip:
            denoised = denoised.clip(0.0, 1.0)

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
