from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, Optional

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from pydantic import Field, field_validator
from skimage.restoration import denoise_bilateral

from ..abc_ import ImageDenoiser
from ..sdk_.mixin import _GATSupportMixin
from ..sdk_.typing_ import TuneSpec


class LocalEdgeDenoise(_GATSupportMixin, ImageDenoiser):
    """Denoise ``detect_mat`` with local edge-preserving bilateral filtering.

    Weights each pixel by both spatial proximity and intensity similarity
    within a local neighbourhood, smoothing uniform agar regions while
    keeping colony boundaries sharp. The optimal intensity range weight
    scales linearly with the image noise floor, so noisier images benefit
    from a larger ``sigma_color``.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Best For:
        - Noisy or grainy agar scans from high-ISO photography or older
          flatbed scanners where colony edges must remain sharp.
        - Plates with surface condensation, dust speckles, or uneven agar
          texture.
        - Fast single-pass preprocessing before thresholding when only local
          smoothing is needed.
        - Low-quality captures where colony morphology must be preserved.

    Consider Also:
        - :class:`NonLocalMeansDenoiser` when repetitive agar texture warrants
          whole-image patch search for stronger denoising.
        - :class:`EnhanceBlockMatch` for state-of-the-art structured noise removal
          at higher computational cost.
        - :class:`SubtractGaussian` when the primary problem is a slow-varying
          illumination gradient rather than pixel-level noise.

    Args:
        sigma_color: Intensity similarity weighting on the [0, 1] scale.
            Small values (0.02--0.05) preserve subtle colony boundaries;
            medium values (0.05--0.15) balance denoising and edge
            preservation; large values (0.2--0.5) smooth aggressively
            across edges. ``None`` (default) auto-estimates from the image
            standard deviation. The optimal value is approximately
            proportional to the noise standard deviation of the image.
            Automatically retargeted to 1.0 when ``use_gat=True``.
        sigma_spatial: Spatial distance weighting in pixels. Small values
            (1--5) apply local denoising only; medium values (10--20)
            smooth regionally; large values (30--50) blend wide areas. Keep
            below the minimum colony diameter to avoid smearing adjacent
            colonies together. Default: 15.
        win_size: Filter window half-size in pixels. ``None`` (default)
            auto-computes as ``max(5, 2*ceil(3*sigma_spatial)+1)``. Override
            only to cap memory use when ``sigma_spatial`` is very large.
        mode: Boundary handling. Accepted values: ``'constant'``,
            ``'edge'``, ``'symmetric'``, ``'reflect'``, ``'wrap'``.
            ``'constant'`` (default) pads with ``cval``; ``'edge'`` or
            ``'reflect'`` avoids darkening at plate borders when the plate
            fills the frame.
        cval: Fill value used when ``mode='constant'``. Set to the mean
            background intensity to suppress spurious dark-band artefacts at
            plate edges. Default: 0.0.
        clip: Clip output to [0, 1]. Default: ``True``. Automatically
            deferred to ``False`` when ``use_gat=True``.

        # GAT parameters (active only when use_gat=True)
        use_gat: Wrap denoising in the Generalized Anscombe Transform to
            stabilize Poisson-Gaussian noise variance before filtering.
            Most beneficial for low-light photographs or low-exposure scans
            where shot noise dominates. Default: ``False``.
        gat_gain: Camera gain in electrons per ADU. Scales the Poisson
            noise component in the GAT model. Typical range 0.1--10.0;
            leave at 1.0 for normalized images without calibrated gain.
            Default: 1.0.
        gat_mu: Read-noise mean (baseline DC offset). Set to zero if
            dark-frame subtraction has already been applied. Default: 0.0.
        gat_read_sigma: Standard deviation of additive Gaussian read noise.
            ``0.0`` (default) assumes pure Poisson noise. Typical flatbed
            scanner values: 0.004--0.02 on the [0, 1] scale.
        gat_scale_factor: Multiplier converting normalized [0, 1] data to
            photon counts before the GAT forward pass. ``None`` (default)
            auto-detects from image bit depth (8-bit → 255, 16-bit → 65535).

    Returns:
        Image: Input image with ``detect_mat`` smoothed by bilateral
        filtering. ``rgb`` and ``gray`` are unchanged.

    References:
        [1] C. Tomasi and R. Manduchi, "Bilateral filtering for gray and
        color images," in *Proc. ICCV*, Bombay, 1998, pp. 839--846.
        [2] M. Mäkitalo and A. Foi, "Optimal inversion of the generalized
        Anscombe transformation for Poisson-Gaussian noise," *IEEE Trans.
        Image Process.*, vol. 22, no. 1, pp. 91--103, Jan. 2013.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a
        visual walkthrough of denoising pipelines on plate images.
        :doc:`/how_to/notebooks/denoise_low_light` for edge-preserving
        denoising strategies on low-light plate images.
        :doc:`/explanation/what_enhancement_does` for bilateral filter
        theory and parameter selection guidance.
    """

    _GAT_NOISE_PARAMS: ClassVar[dict[str, float]] = {"sigma_color": 1.0}
    _GAT_DEFER_ATTRS: ClassVar[tuple[str, ...]] = ("clip",)

    # Categorical (not a float range) so the optimizer can try ``None`` — the
    # auto-estimate (noise-std) mode — alongside representative explicit sigmas.
    # A plain ``TuneSpec(0.02, 0.5)`` window would never sample the ``None`` arm.
    sigma_color: Annotated[
        Optional[float],
        TuneSpec(categories=[None, 0.02, 0.05, 0.1, 0.2, 0.5]),
    ] = None
    sigma_spatial: Annotated[float, TuneSpec(1.0, 50.0, log=True)] = Field(15, gt=0.0)
    win_size: Annotated[Optional[int], TuneSpec(tunable=False)] = None
    mode: Literal["constant", "edge", "symmetric", "reflect", "wrap"] = "constant"
    cval: Annotated[float, TuneSpec(tunable=False)] = 0
    clip: bool = True

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
