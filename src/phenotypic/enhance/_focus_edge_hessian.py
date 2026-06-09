from __future__ import annotations
from typing import Annotated, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from pydantic import field_validator
from skimage.filters import hessian

from phenotypic.abc_ import FocusEdge
from phenotypic.tools_.typing_ import TuneSpec


class FocusEdgeHessian(FocusEdge):
    """Enhance colony boundaries and ridge-like structures via multi-scale Hessian filtering.

    Computes Hessian matrix eigenvalues at each sigma scale and combines them
    into a hybrid vesselness score that highlights regions with high curvature
    — colony edges, filament ridges, biofilm features — while suppressing flat
    agar background. Unlike the Frangi filter, background pixels (with response
    ≤ 0) are set to 1, so the output map inverts the agar-to-colony contrast
    and is best interpreted as a ridge-strength mask. For algorithm details see
    :doc:`/explanation/what_enhancement_does`.

    Best For:
        - Sharp colony-agar boundaries on plates with compact bacterial or
          yeast colonies resolved at 1--5 px edge width.
        - Size-heterogeneous plates where a tuple of sigmas spanning the full
          colony size range provides scale-invariant edge response.
        - Thin filaments and branching structures with low intensity contrast
          that global thresholding misses.
        - Textured colonies or biofilms where internal ridge structure aids
          downstream morphology analysis.

    Consider Also:
        - :class:`FocusEdgeFrangi` for strictly elongated hyphae and mycelial
          networks where background suppression via the adaptive gamma rule
          is preferred.
        - :class:`FocusEdgeMeijering` for very fine neurite-like filaments
          where the analytic shape-parameter optimum is preferred.
        - :class:`FocusEdgeLaplace` for simpler single-scale second-derivative
          edge detection without multi-scale parameter tuning.

    Args:
        sigmas: Gaussian standard deviations (pixels) at which the Hessian
            is evaluated. Each value responds most strongly to ridges and
            edges whose cross-sectional half-width is approximately that
            number of pixels; the per-pixel maximum across all scales is
            taken. Typical range: ``(1, 2, 3)`` for standard plate scans;
            extend to ``(1, 5)`` or wider when colony sizes vary broadly.
            A reasonable starting point for whole-plate scans at 600 dpi is
            ``(1, 2, 3)``; add larger sigmas for mature large colonies or
            thick filaments. Default: ``(1, 2, 3)``.
        alpha: Plate-likeness sensitivity in the vesselness formula. In 2-D
            images this parameter has no numerical effect because the
            plate-sensitivity ratio is undefined and omitted from the 2-D
            formula; it is included only for compatibility with 3-D use.
            Typical range: 0.1--1.0. Default: 0.5.
        beta: Blob-likeness sensitivity. Lower values make the filter more
            permissive of rounded, curved, or imperfect ridges (useful for
            circular colony edges); higher values restrict responses to more
            elongated, line-like structures. Typical range: 0.1--1.0.
            Default: 0.5.
        gamma: Fixed background suppression threshold applied to the Hessian
            Frobenius norm. Only regions with norm above this level produce
            nonzero responses; regions below it are set to 1 (background
            convention of the hybrid Hessian filter). Typical range: 10--20.
            Lower values (5--10) recover faint colony edges; higher values
            (20--25) sharpen the contrast between ridges and flat agar.
            Default: 15.
        black_ridges: Polarity of the target ridges. ``False`` (default)
            detects bright ridges on a dark background, matching the
            ``detect_mat`` convention where colonies and hyphae appear bright.
            ``True`` detects dark ridges on a bright background. Default: ``False``.
        mode: Boundary padding mode for Gaussian derivative computation.
            Accepted values: ``'reflect'``, ``'constant'``, ``'nearest'``,
            ``'mirror'``, ``'wrap'``. Default: ``'reflect'``.
        cval: Fill value used when ``mode='constant'``. Has no effect for
            any other mode. Default: 0.

    Returns:
        Image: Input image with ``detect_mat`` replaced by the Hessian
        ridge response map. ``rgb`` and ``gray`` are unchanged.

    References:
        [1] A. F. Frangi, W. J. Niessen, K. L. Vincken, and M. A. Viergever,
        "Multiscale vessel enhancement filtering," in *Proc. MICCAI*, 1998,
        pp. 130--137.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a
        visual walkthrough of ridge and edge enhancement on plate images.
        :doc:`/explanation/what_enhancement_does` for background on
        Hessian-based structure detection.
    """

    sigmas: tuple[float, ...] = (1, 2, 3)
    alpha: float = 0.5
    beta: Annotated[float, TuneSpec(0.1, 1.0)] = 0.5
    gamma: Annotated[float, TuneSpec(5.0, 25.0, log=True)] = 15
    black_ridges: bool = False
    mode: str = "reflect"
    cval: Annotated[float, TuneSpec(tunable=False)] = 0

    @field_validator("sigmas", mode="before")
    @classmethod
    def _coerce_sigmas(cls, sigmas: Iterable[float]) -> tuple[float, ...]:
        """Coerce any iterable of sigmas to a tuple.

        Reproduces the pre-migration ``__setattr__`` override, which
        normalized ``sigmas`` (passed as a list or other iterable) to a
        tuple before storing it.
        """
        return tuple(sigmas)

    def _operate(self, image: Image) -> Image:
        image.detect_mat[:] = hessian(
                image=image.detect_mat[:],
                sigmas=self.sigmas,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                black_ridges=self.black_ridges,
                mode=self.mode,
                cval=self.cval,
        )
        return image
