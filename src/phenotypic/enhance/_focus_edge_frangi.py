from __future__ import annotations
from typing import Annotated, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from pydantic import field_validator
from skimage.filters import frangi

from phenotypic.abc_ import FocusEdge
from phenotypic.tools_.typing_ import TuneSpec


class FocusEdgeFrangi(FocusEdge):
    """Enhance elongated structures in ``detect_mat`` using multi-scale Frangi vesselness filtering.

    Computes Hessian matrix eigenvalues at each sigma scale and combines them
    into a vesselness score that responds strongly to ridge-like features —
    hyphae, mycelial branches, biofilm edges — while suppressing blob-like
    colonies and flat agar background. The output is a [0, 1] probability-like
    map that typically feeds into :class:`FilamentousFungiDetector` or a
    hysteresis threshold. For algorithm details see
    :doc:`/explanation/filamentous_fungi_algorithm`.

    Best For:
        - Filamentous fungi (*Neurospora*, *Aspergillus*) with branching
          hyphae resolved to 2--8 px wide at the imaging scale.
        - Mycelial networks and biofilm ridge structures that global intensity
          thresholding misses.
        - Pre-filtering before :class:`FilamentousFungiDetector` to produce a
          clean hyphal evidence map.
        - Plates where hyphae span a range of widths and a multi-sigma sweep
          is needed to cover all branch generations.

    Consider Also:
        - :class:`FocusEdgeMeijering` for very fine, isolated filaments where
          the analytic alpha optimum and simpler parameterisation are preferred.
        - :class:`FocusEdgeSato` when continuous tubular structures with
          different eigenvalue-ratio behaviour are the target.
        - :class:`FocusEdgePhase` for contrast-invariant edge enhancement
          on plates with uneven illumination.
        - :class:`StructureSmoothing` for anisotropic pre-smoothing along
          hyphal orientation before ridge detection.

    Args:
        sigmas: Gaussian standard deviations (pixels) at which the Hessian
            is evaluated. Each sigma responds most strongly to ridges whose
            cross-sectional half-width is approximately that value. Include
            sigmas spanning the full range of expected hyphal widths; the
            per-pixel maximum across all scales is taken, so additional sigmas
            can only raise the response. A reasonable starting point for agar
            plate scans at 600 dpi, where hyphae typically appear 2--8 px
            wide, is ``(0.5, 1, 1.5)`` to ``(1, 2, 3, 4)``; extend the upper
            bound for mature thick mycelium or coarser imaging resolution.
            Default: ``(0.5, 1, 1.5)``.
        alpha: Plate-likeness sensitivity in the vesselness formula.
            Controls how strongly the filter penalises structures that deviate
            from a purely elongated ridge. In 2-D images this parameter has no
            numerical effect because the plate-sensitivity ratio is undefined
            and omitted from the 2-D vesselness formula; it is included only
            for compatibility with 3-D use. Typical range: 0.1--1.0.
            Default: 0.5.
        beta: Blob-likeness sensitivity. Lower values make the filter more
            permissive of rounded or imperfect ridges (useful at branching
            junctions and thick hyphal segments); higher values restrict
            responses to more purely elongated structures. Typical range:
            0.1--1.0. Default: 0.5.
        gamma: Background suppression threshold based on the Hessian
            Frobenius norm. ``None`` (default) uses half the maximum Hessian
            norm per scale, adapting to the actual contrast in each image.
            An explicit positive value provides a fixed threshold useful when
            comparing results across images with different illumination.
            Default: ``None``.
        black_ridges: Polarity of the target ridges. ``False`` (default)
            detects bright ridges on a dark background, matching the
            ``detect_mat`` convention where hyphae appear bright. ``True``
            detects dark ridges on a bright background (e.g. transmitted-light
            microscopy). Default: ``False``.

    Returns:
        Image: Input image with ``detect_mat`` replaced by the [0, 1]
        vesselness response map. ``rgb`` and ``gray`` are unchanged.

    References:
        [1] A. F. Frangi, W. J. Niessen, K. L. Vincken, and M. A. Viergever,
        "Multiscale vessel enhancement filtering," in *Proc. MICCAI*, 1998,
        pp. 130--137.

    See Also:
        :doc:`/tutorials/notebooks/10_detecting_filamentous_fungi` for a
        visual walkthrough of filamentous fungi detection.
        :doc:`/explanation/filamentous_fungi_algorithm` for the theory behind
        Hessian-based vesselness filtering.
    """

    sigmas: tuple[float, ...] = (0.5, 1, 1.5)
    alpha: float = 0.5
    beta: Annotated[float, TuneSpec(0.1, 1.0)] = 0.5
    gamma: float | None = None
    black_ridges: bool = False

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
        image.detect_mat[:] = frangi(
                image=image.detect_mat[:],
                sigmas=self.sigmas,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                black_ridges=self.black_ridges,
        )
        return image
