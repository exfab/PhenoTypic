from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from pydantic import field_validator
from skimage.restoration import denoise_bilateral

from ..abc_ import ImageDenoiser
from ..tools_.mixin import _GATSupportMixin


class LocalEdgeDenoise(_GATSupportMixin, ImageDenoiser):
    """Denoise ``detect_mat`` with fast, local edge-preserving (bilateral) filtering.

    Averages pixel values based on both spatial proximity and intensity
    similarity within a *local* neighborhood, preserving sharp colony
    boundaries while smoothing uniform regions such as agar background.
    Effectively removes scanner noise, agar grain, dust speckles, and
    condensation artifacts without blurring colony edges.

    This is the lightweight, single-pass edge-preserving denoiser: it
    weighs only each pixel's local neighborhood. For stronger cleanup that
    searches the whole image for similar patches, step up to the non-local
    :class:`NonLocalMeansDenoiser` or :class:`BM3DDenoiser`.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Args:
        sigma_color: Intensity similarity weighting. Small values
            (0.02--0.05) preserve subtle boundaries; medium values
            (0.05--0.15) balance denoising and edge preservation; large
            values (0.2--0.5) smooth aggressively. ``None`` (default)
            auto-estimates from image statistics. Retargeted to 1.0 when
            ``use_gat=True``.
        sigma_spatial: Spatial distance weighting in pixels. Small values
            (1--5) apply local denoising; medium values (10--20) smooth
            regionally; large values (30--50) smooth wide areas. Keep
            below the minimum colony diameter. Default: 15. Not affected
            by GAT (purely spatial parameter).
        win_size: Window size for filter computation. ``None`` (default)
            auto-calculates from ``sigma_spatial``.
        mode: Boundary handling. Accepted values: ``'constant'``,
            ``'edge'``, ``'symmetric'``, ``'reflect'``, ``'wrap'``.
            Default: ``'constant'``.
        cval: Fill value when ``mode='constant'``. Default: 0.
        clip: Clip output to [0, 1]. Default: ``True``. Automatically
            deferred when ``use_gat=True``.
        use_gat: Wrap denoising in the Generalized Anscombe Transform.
            Default: ``False``.
        gat_gain, gat_mu, gat_read_sigma, gat_scale_factor: GAT parameters.

    Returns:
        Image: Input image with ``detect_mat`` smoothed by bilateral
        filtering. ``rgb`` and ``gray`` are unchanged.

    Best For:
        - Noisy or grainy agar scans from high-ISO photography or old scanners.
        - Plates with surface condensation, dust speckles, or uneven agar
          texture.
        - Preprocessing before thresholding when colony edges must remain
          sharp.
        - Low-quality captures where colony morphology must be preserved.

    Consider Also:
        - :class:`NonLocalMeansDenoiser` for stronger denoising of repetitive
          textures at higher computational cost.
        - :class:`BM3DDenoiser` for state-of-the-art structured noise removal.
        - :class:`SubtractGaussian` when the primary problem is illumination
          gradients rather than pixel-level noise.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a
        visual walkthrough of denoising pipelines on plate images.
        :doc:`/how_to/notebooks/denoise_low_light` for edge-preserving
        denoising strategies on low-light plate images.
    """

    _GAT_NOISE_PARAMS: ClassVar[dict[str, float]] = {"sigma_color": 1.0}
    _GAT_DEFER_ATTRS: ClassVar[tuple[str, ...]] = ("clip",)

    sigma_color: float | None = None
    sigma_spatial: float = 15
    win_size: int | None = None
    mode: Literal["constant", "edge", "symmetric", "reflect", "wrap"] = "constant"
    cval: float = 0
    clip: bool = True

    @field_validator("sigma_spatial")
    @classmethod
    def _check_sigma_spatial(cls, sigma_spatial: float) -> float:
        """Require a positive spatial sigma (matches the legacy guard)."""
        if sigma_spatial <= 0:
            raise ValueError("sigma_spatial must be > 0")
        return sigma_spatial

    @field_validator("sigma_color")
    @classmethod
    def _check_sigma_color(cls, sigma_color: float | None) -> float | None:
        """Require a positive intensity sigma or ``None`` (matches the legacy guard)."""
        if sigma_color is not None and sigma_color <= 0:
            raise ValueError("sigma_color must be > 0 or None")
        return sigma_color

    def _operate(self, image: Image) -> Image:
        """Apply bilateral denoising to reduce noise while preserving colony edges."""
        self._gat_apply(image, "detect_mat", self._denoise_detect_mat)
        return image

    def _denoise_detect_mat(self, image: Image) -> None:
        # denoise_bilateral may require a writable array, so create a copy
        result = denoise_bilateral(
                image=image.detect_mat[:].copy(),
                sigma_color=self.sigma_color,
                sigma_spatial=self.sigma_spatial,
                win_size=self.win_size,
                mode=self.mode,
                cval=self.cval,
                channel_axis=None,
        )
        if self.clip:
            result = result.clip(0.0, 1.0)
        image.detect_mat[:] = result
