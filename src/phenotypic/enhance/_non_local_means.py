from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, ClassVar

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from skimage.restoration import denoise_nl_means

from ..abc_ import ImageDenoiser
from ..tools_.mixin import _GATSupportMixin
from ..tools_.typing_ import TuneSpec


class NonLocalMeansDenoiser(_GATSupportMixin, ImageDenoiser):
    """Denoise ``detect_mat`` with non-local means patch-based filtering.

    Compares patches across the image to identify similar structures and
    averages them, preserving thin colony boundaries and internal texture
    better than simple Gaussian or median filtering. Particularly effective
    at removing Gaussian noise and agar granularity.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Args:
        patch_size: Size of patches used for comparison in pixels. Larger
            patches capture more structure but are slower. Typical range:
            5--15. Default: 5.
        search_dist: Maximum search distance for similar patches in pixels.
            Larger values find more candidates at higher cost. Typical
            range: 5--21. Default: 11.
        h: Cut-off distance controlling smoothness. Rule of thumb:
            ``h`` ~= noise level (sigma). Higher values produce more
            smoothing. Both ``h`` and ``sigma`` retarget to 1.0 when
            ``use_gat=True``. Default: 0.5.
        fast_mode: If ``True``, use faster variant with uniform spatial
            weighting. If ``False`` (default), use original algorithm
            with Gaussian spatial weighting.
        sigma: Expected noise standard deviation. Values > 0 improve
            patch weighting by accounting for expected noise variance.
            Retargets to 1.0 when ``use_gat=True``. Default: 0.0.
        use_gat: Wrap denoising in the Generalized Anscombe Transform.
            Default: ``False``. See
            :class:`phenotypic.tools_.mixin._GATSupportMixin`.
        gat_gain, gat_mu, gat_read_sigma, gat_scale_factor: GAT parameters.

    Returns:
        Image: Input image with ``detect_mat`` denoised via non-local
        means filtering. ``rgb`` and ``gray`` are unchanged.

    Best For:
        - Scanner noise and agar granularity where colony edges must stay
          sharp.
        - Low-contrast or faint colonies where Gaussian blur would cause
          loss of detail.
        - Preserving colony texture and morphology during speckle and dust
          removal.
        - Pre-filtering before edge detection to avoid amplifying noise.
        - Poisson-Gaussian noise via ``use_gat=True`` (Deledalle et al.
          2010 demonstrate that NLM benefits from variance stabilization
          for moderate-to-high photon counts).

    Consider Also:
        - :class:`BM3DDenoiser` for state-of-the-art structured noise
          removal at higher computational cost.
        - :class:`LocalEdgeDenoise` for faster edge-preserving denoising
          without patch comparison.
        - :class:`BayesShrinkEnhancer` for adaptive wavelet denoising with
          spatially varying thresholds.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a
        visual walkthrough of denoising pipelines on plate images.
        :doc:`/how_to/notebooks/denoise_low_light` for non-local means
        and other denoising strategies on low-light plate images.
    """

    _GAT_NOISE_PARAMS: ClassVar[dict[str, float]] = {"h": 1.0, "sigma": 1.0}
    _GAT_DEFER_ATTRS: ClassVar[tuple[str, ...]] = ()

    patch_size: Annotated[int, TuneSpec(5, 15, step=2)] = 5
    search_dist: Annotated[int, TuneSpec(5, 21, step=2)] = 11
    h: Annotated[float, TuneSpec(0.1, 2.0, log=True)] = 0.5
    fast_mode: bool = False
    sigma: Annotated[float, TuneSpec(tunable=False)] = 0.0

    def _operate(self, image: Image) -> Image:
        """Apply non-local means denoising to detection matrix."""
        self._gat_apply(image, "detect_mat", self._denoise_detect_mat)
        return image

    def _denoise_detect_mat(self, image: Image) -> None:
        denoised = denoise_nl_means(
                image=image.detect_mat[:],
                patch_size=self.patch_size,
                patch_distance=self.search_dist,
                h=self.h,
                fast_mode=self.fast_mode,
                sigma=self.sigma,
                preserve_range=True,
        )
        image.detect_mat[:] = denoised
