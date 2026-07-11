from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, ClassVar

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from skimage.restoration import denoise_nl_means

from ..abc_ import ImageDenoiser
from ..sdk_.mixin import _GATSupportMixin
from ..sdk_.typing_ import TuneSpec


class NonLocalMeansDenoiser(_GATSupportMixin, ImageDenoiser):
    """Denoise ``detect_mat`` with non-local means patch-based filtering.

    Compares small patches across the entire image to identify structurally
    similar regions and averages them, exploiting the repetitive texture of
    arrayed colony plates to remove noise while preserving thin colony
    boundaries. Stronger than bilateral filtering for images with repetitive
    agar background patterns, at higher computational cost.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Best For:
        - Scanner noise and agar granularity where colony edges must stay
          sharp.
        - Low-contrast or faint colonies where Gaussian blur would cause
          loss of detail.
        - Dense arrayed plates whose repetitive colony pattern provides many
          similar patches for effective averaging.
        - Pre-filtering before edge detection to reduce noise without
          amplifying gradients.

    Consider Also:
        - :class:`EnhanceBlockMatch` for state-of-the-art structured noise removal
          at higher computational cost.
        - :class:`LocalEdgeDenoise` for faster edge-preserving denoising
          without whole-image patch search.
        - :class:`BayesShrinkEnhancer` for adaptive wavelet denoising that
          applies spatially varying thresholds.

    Args:
        patch_size: Side length of the square patches compared during
            denoising in pixels. Larger patches capture more structural
            context and suppress noise more robustly, but are slower and
            risk spanning adjacent colony boundaries on dense plates.
            Typical range: 3--11 (skimage library default is 7); keep
            ≤ 5 for fine hyphal structures to avoid cross-branch
            averaging. Default: 5.
        search_dist: Half-side of the square search window for patch
            candidates. Larger values find more similar patches at
            quadratic computational cost (the skimage library default is
            11, i.e. a 23x23 window). On crowded 384-well plates a smaller
            value (5--7) avoids pulling patches from neighbouring colony
            positions. Default: 11.
        h: Cut-off distance in the patch-similarity weight kernel. Patches
            with squared distance greater than ``h`` squared receive
            exponentially diminished weight. Rule of thumb: ``h``
            approximately equals the noise standard deviation. Both ``h``
            and ``sigma`` are automatically retargeted to 1.0 when
            ``use_gat=True``. Default: 0.5.
        fast_mode: If ``True``, use the faster uniform-weight variant
            (integral-image algorithm); if ``False`` (default), use the
            original Gaussian-weighted algorithm which preserves colony
            edges marginally better. When ``sigma`` is provided, pair
            ``fast_mode=True`` with ``h ≈ 0.8 * sigma`` and
            ``fast_mode=False`` with ``h ≈ 0.6 * sigma``. Default:
            ``False``.
        sigma: Known noise standard deviation on the [0, 1] scale.
            When > 0, noise variance is subtracted from patch distances,
            improving weight accuracy for structurally similar patches.
            Set to 0.0 to disable. Retargeted to 1.0 when
            ``use_gat=True``. Default: 0.0.

        # GAT parameters (active only when use_gat=True)
        use_gat: Wrap denoising in the Generalized Anscombe Transform to
            handle Poisson-Gaussian noise (e.g., low-light fluorescence
            images of colonies). Default: ``False``.
        gat_gain: Camera gain in electrons per ADU. Scales the Poisson
            noise component. Typical range 0.1--10.0; leave at 1.0 for
            normalized images without calibrated gain. Default: 1.0.
        gat_mu: Read-noise mean (baseline DC offset). Set to 0.0 if
            dark-frame subtraction has been applied. Default: 0.0.
        gat_read_sigma: Standard deviation of additive Gaussian read noise
            on the [0, 1] scale. ``0.0`` (default) assumes pure Poisson
            noise. Scientific CMOS cameras: typically 0.001--0.05.
        gat_scale_factor: Multiplier converting normalized [0, 1] data to
            photon counts before the GAT forward pass. ``None`` (default)
            auto-detects from image bit depth (8-bit → 255, 16-bit → 65535).

    Returns:
        Image: Input image with ``detect_mat`` denoised via non-local means
        filtering. ``rgb`` and ``gray`` are unchanged.

    References:
        [1] A. Buades, B. Coll, and J.-M. Morel, "A non-local algorithm for
        image denoising," in *Proc. CVPR*, vol. 2, 2005, pp. 60--65.
        [2] M. Lebrun, "Non-local means denoising," *Image Process. On
        Line*, vol. 2012, pp. 208--212, 2012.
        [3] M. Mäkitalo and A. Foi, "Optimal inversion of the generalized
        Anscombe transformation for Poisson-Gaussian noise," *IEEE Trans.
        Image Process.*, vol. 22, no. 1, pp. 91--103, Jan. 2013.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a
        visual walkthrough of denoising pipelines on plate images.
        :doc:`/how_to/notebooks/denoise_low_light` for non-local means
        and other denoising strategies on low-light plate images.
        :doc:`/explanation/what_enhancement_does` for patch-similarity
        denoising theory and parameter selection guidance.
    """

    _GAT_NOISE_PARAMS: ClassVar[dict[str, float]] = {"h": 1.0, "sigma": 1.0}
    _GAT_DEFER_VALUES: ClassVar[dict[str, Any]] = {}

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
