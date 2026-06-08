from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, ClassVar, Literal

if TYPE_CHECKING:
    from phenotypic._core._image import Image
import bm3d
from bm3d.profiles import BM3DStages
from pydantic import Field

from ..abc_ import ImageDenoiser
from ..tools_.mixin import _GATSupportMixin
from ..tools_.typing_ import TuneSpec


class BM3DDenoiser(_GATSupportMixin, ImageDenoiser):
    """Denoise ``detect_mat`` using block-matching and 3-D collaborative filtering.

    Groups similar image patches across the plate and filters them jointly
    in a 3-D transform domain, removing scanner banding, CCD read patterns,
    and imaging-hardware texture while retaining colony edges and fine
    morphological features. The optional second Wiener-filtering stage
    further sharpens colony boundaries relative to the initial
    hard-thresholding pass. For algorithm details see
    :doc:`/explanation/what_enhancement_does`.

    Best For:
        - Structured patterned noise from scanner CCD rows or camera
          sensor banding visible across the plate background.
        - Low-light incubator or plate-reader images where high ISO
          introduces spatially correlated noise.
        - Plates where fine colony morphology — wrinkle texture, satellite
          colonies, rough biofilm edges — must survive denoising.
        - Pipelines that already use ``'all_stages'`` for highest-quality
          deliverables and can absorb the additional compute.
        - Poisson-Gaussian mixed noise (fluorescence readers, sCMOS cameras)
          via ``use_gat=True``.

    Consider Also:
        - :class:`BayesShrinkEnhancer` for per-subband adaptive wavelet
          denoising with lower computational cost when structured patterns
          are not the primary concern.
        - :class:`NonLocalMeansDenoiser` for patch-based denoising at
          moderately lower computational overhead.
        - :class:`VisuShrinkEnhancer` when speed matters more than quality
          and a universal wavelet threshold is acceptable.
        - :class:`MedianFilter` for isolated salt-and-pepper impulse noise
          rather than Gaussian or structured noise.

    Args:
        sigma_psd: Noise standard deviation on the [0, 1] normalised
            intensity scale. Acts as the noise-model oracle for both the
            hard-thresholding and Wiener stages. Typical range: 0.01--0.05
            for moderate scanner noise; 0.05--0.15 for heavy noise. Setting
            too low leaves structured patterns intact; too high smooths away
            colony texture alongside noise. Ignored when ``use_gat=True`` —
            the stabilised-domain value 1.0 is used internally. Default: 0.02.
        block_size: Side length (pixels) of square patches used for
            block-matching. Larger blocks capture more self-similar structure
            and increase denoising strength at the cost of speed and potential
            over-smoothing of fine colony detail. Typical range: 4--16; integer
            powers of 2 are conventional. Default: 8.
        stage_arg: Processing pipeline. ``'all_stages'`` (default) runs the
            hard-thresholding stage followed by a Wiener-filtering stage using
            the HT estimate as an oracle, producing better boundary sharpness.
            ``'hard_thresholding'`` runs only the first stage — approximately
            2× faster with slightly coarser colony edges. Accepted values:
            ``'all_stages'``, ``'hard_thresholding'``. Default: ``'all_stages'``.
        clip: Clamp output to [0, 1] after BM3D aggregation. BM3D can
            produce values marginally outside [0, 1] due to weighted patch
            accumulation. Default: ``True``. Automatically set to ``False``
            inside the GAT region when ``use_gat=True``.

        # GAT parameters — only active when use_gat=True
        use_gat: Wrap the BM3D call in a forward Generalised Anscombe
            Transform (GAT) → denoise at fixed sigma_psd=1.0 → exact
            unbiased inverse GAT pipeline. Enables correct denoising under
            mixed Poisson-Gaussian noise from low-light sensors. Leave
            ``False`` for standard flatbed scanner images where additive
            Gaussian read noise dominates. Default: ``False``.
        gat_gain: Camera gain in electrons per ADU, used by the GAT to model
            Poisson variance scaling. Obtain from the sensor datasheet or a
            photon-transfer curve. Only relevant when ``use_gat=True``.
            Default: 1.0.
        gat_mu: Read-noise mean (DC baseline offset) in ADU before
            [0, 1] normalisation. Set to the dark-current bias level if the
            image has not been background-subtracted. Only relevant when
            ``use_gat=True``. Default: 0.0.
        gat_read_sigma: Standard deviation of the Gaussian read-noise
            component (electrons RMS). Setting this to the manufacturer's
            read noise improves stabilisation accuracy under mixed
            Poisson-Gaussian conditions. Only relevant when ``use_gat=True``.
            Default: 0.0.
        gat_scale_factor: Multiplier converting the [0, 1] normalised
            ``detect_mat`` back to photon counts before the forward GAT.
            ``None`` auto-detects from ``image.metadata.bit_depth`` (255 for
            8-bit, 65535 for 16-bit). Supply an explicit value when the
            ``detect_mat`` was normalised by a range other than the bit-depth
            maximum. Only relevant when ``use_gat=True``. Default: ``None``.

    Returns:
        Image: Input image with ``detect_mat`` denoised via BM3D
        collaborative filtering. ``rgb`` and ``gray`` are unchanged.

    References:
        [1] K. Dabov, A. Foi, V. Katkovnik, and K. Egiazarian, "Image
        denoising by sparse 3-D transform-domain collaborative filtering,"
        *IEEE Trans. Image Process.*, vol. 16, no. 8, pp. 2080--2095,
        Aug. 2007.

        [2] M. Mäkitalo and A. Foi, "Optimal inversion of the generalized
        Anscombe transformation for Poisson-Gaussian noise," *IEEE Trans.
        Image Process.*, vol. 22, no. 1, pp. 91--103, Jan. 2013.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a
        visual walkthrough of denoising pipelines on plate images.
        :doc:`/how_to/notebooks/denoise_low_light` for BM3D and other
        denoising strategies on low-light plate images.
        :doc:`/explanation/what_enhancement_does` for background on
        block-matching and collaborative filtering.
    """

    _GAT_NOISE_PARAMS: ClassVar[dict[str, float]] = {"sigma_psd": 1.0}
    _GAT_DEFER_ATTRS: ClassVar[tuple[str, ...]] = ("clip",)

    sigma_psd: Annotated[float, TuneSpec(0.01, 0.15, log=True)] = Field(0.02, ge=0.0)
    block_size: Annotated[int, TuneSpec(tunable=False)] = 8
    stage_arg: Literal["all_stages", "hard_thresholding"] = "all_stages"
    clip: bool = True

    def _operate(self, image: Image) -> Image:
        # detect_mat is guaranteed to be in [0, 1] range, which BM3D expects
        # (or in stabilized counts when use_gat=True; the mixin handles the
        # forward/inverse round-trip and noise-param retargeting).
        self._gat_apply(image, "detect_mat", self._denoise_detect_mat)
        return image

    def _denoise_detect_mat(self, image: Image) -> None:
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

    def _convert_stage_arg(
            self, stage_arg: Literal["all_stages", "hard_thresholding"]
    ):
        match stage_arg:
            case "hard_thresholding":
                return BM3DStages.HARD_THRESHOLDING
            case "all_stages":
                return BM3DStages.ALL_STAGES
            case _:
                raise ValueError(f"Unknown stage arg: {stage_arg}")
