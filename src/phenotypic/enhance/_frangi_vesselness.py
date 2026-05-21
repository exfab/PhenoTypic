from __future__ import annotations
from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from pydantic import field_validator
from skimage.filters import frangi

from phenotypic.abc_ import ImageEnhancer


class FrangiVesselness(ImageEnhancer):
    """Enhance tubular structures in detect_mat using Hessian-based vesselness filtering.

    Computes the Frangi vesselness measure from Hessian matrix eigenvalues at
    multiple scales, producing a response map that highlights elongated features
    (hyphae, branches, mycelial networks). The output is a probability-like map
    (0--1) that typically requires thresholding before detection.

    For algorithm details, see :doc:`/explanation/filamentous_fungi_algorithm`.

    Args:
        sigmas: Scales (standard deviations) for Hessian computation. Smaller
            values detect finer structures; larger values detect thicker ones.
            Span the expected range of hyphal widths in pixels. Default:
            ``(0.5, 1, 1.5)``.
        alpha: Blobness sensitivity (0--1). Lower is more permissive.
            Default: 0.5.
        beta: Structuredness sensitivity (0--1). Lower is more permissive.
            Default: 0.5.
        gamma: Background suppression threshold. Larger values suppress
            low-curvature (flat) regions more aggressively. ``None`` uses
            half of the max Hessian norm. Default: ``None``.
        black_ridges: If ``True``, detect dark ridges on bright background.
            If ``False``, detect bright ridges on dark background.
            Default: ``False``.

    Returns:
        Image: Input image with ``detect_mat`` set to the vesselness response
        map. ``rgb`` and ``gray`` are unchanged.

    Best For:
        - Filamentous fungi (*Neurospora*, *Aspergillus*) with branching hyphae.
        - Thin, elongated structures that global thresholding misses.
        - Interconnected mycelial networks or biofilm structures.
        - Pre-filtering before ``FilamentousFungiDetector``.

    Consider Also:
        - :class:`MeijeringRidgeFilter` for neurite-like structures with fewer
          parameters to tune.
        - :class:`SatoRidgeFilter` for ridge detection with different
          sensitivity characteristics.
        - :class:`PhaseCongruencyEnhancer` for illumination-invariant edge
          enhancement of filaments.

    References:
        [1] A. F. Frangi, W. J. Niessen, K. L. Vincken, and M. A. Viergever,
        "Multiscale vessel enhancement filtering," in *MICCAI*, 1998,
        pp. 130--137.

    See Also:
        :doc:`/tutorials/notebooks/10_detecting_filamentous_fungi` for a
        visual walkthrough of filamentous fungi detection.
        :doc:`/explanation/filamentous_fungi_algorithm` for the theory behind
        Hessian-based vesselness filtering.

    """

    sigmas: tuple[float, ...] = (0.5, 1, 1.5)
    alpha: float = 0.5
    beta: float = 0.5
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
