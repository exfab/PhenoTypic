from __future__ import annotations
from typing import Iterable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from pydantic import field_validator
from skimage.filters import meijering

from phenotypic.abc_ import FocusEdge


class FocusEdgeMeijering(FocusEdge):
    """Enhance fine filamentous ridges in ``detect_mat`` using the Meijering neuriteness filter.

    Computes the Meijering neuriteness measure from Hessian eigenvalues at
    each sigma scale, using an analytically derived shape parameter that
    maximally suppresses blob-like and isotropic responses while favouring
    elongated, thread-like features. The output is a [0, 1] ridge-strength
    map suited to detecting delicate hyphae, actinomycete filaments, and
    fine biofilm ridge networks. For algorithm details see
    :doc:`/explanation/what_enhancement_does`.

    Best For:
        - Delicate fungal hyphae and actinomycete filaments resolved to
          2--5 px width where the analytic shape optimum provides maximum
          elongation selectivity.
        - Sparse mycelial networks with well-separated filaments where
          sensitivity to thin ridges is more important than junction detection.
        - Fine biofilm grooves or wrinkle networks in bacterial colony
          morphology plates.
        - Pipelines where minimising the number of parameters to tune is
          a priority (alpha defaults to the analytic optimum).

    Consider Also:
        - :class:`FocusEdgeFrangi` for broader mycelial networks with
          independent control over plate-like and blob-like sensitivity.
        - :class:`FocusEdgeSato` for thicker, continuous tubular structures
          with a different eigenvalue-combination strategy.
        - :class:`FocusEdgeHessian` when combined edge and colony boundary
          detection alongside filament ridges is needed.
        - :class:`StructureSmoothing` for anisotropic pre-smoothing along
          hyphal orientation before this ridge filter.

    Args:
        sigmas: Gaussian standard deviations (pixels) at which the Hessian
            is evaluated. Each value responds most strongly to ridges whose
            cross-sectional half-width is approximately that number of pixels;
            the per-pixel maximum across all scales is taken. Typical range:
            ``(1, 2, 3)`` for standard plate scans; extend the lower bound to
            0.5 for very thin filaments at high magnification or the upper
            bound to 5--8 for thick mycelial mats. Default: ``(1, 2, 3)``.
        alpha: Shape parameter controlling how the Hessian eigenvalues are
            combined into a neuriteness score. ``None`` (default) uses the
            analytically derived optimum ``-1/(ndim+1)`` (``-1/3`` for 2-D
            images), which maximally suppresses blob-like structures while
            retaining elongated ridges. Manual values closer to 0 reduce blob
            suppression; more-negative values sharpen the ridge/blob
            discrimination. Default: ``None``.
        black_ridges: Polarity of the target ridges. ``False`` (default)
            detects bright ridges on a dark background, matching the
            ``detect_mat`` convention where hyphae appear bright. ``True``
            detects dark ridges on a bright background. Default: ``False``.
        mode: Boundary padding mode for Gaussian derivative computation.
            Accepted values: ``'constant'``, ``'reflect'``, ``'wrap'``,
            ``'nearest'``, ``'mirror'``. Default: ``'reflect'``.
        cval: Fill value used when ``mode='constant'``. Has no effect for
            any other mode. Default: 0.

    Returns:
        Image: Input image with ``detect_mat`` replaced by the Meijering
        neuriteness response map. ``rgb`` and ``gray`` are unchanged.

    References:
        [1] E. Meijering, M. Jacob, J.-C. F. Sarria, P. Steiner, H. Hirling,
        and M. Unser, "Design and validation of a tool for neurite tracing
        and analysis in fluorescence microscopy images," *Cytometry Part A*,
        vol. 58, no. 2, pp. 167--176, Apr. 2004.

    See Also:
        :doc:`/tutorials/notebooks/10_detecting_filamentous_fungi` for a
        visual walkthrough of filamentous fungi detection using ridge filters.
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a
        broader enhancement pipeline walkthrough on plate images.
        :doc:`/explanation/what_enhancement_does` for background on
        Hessian-based ridge detection methods.
    """

    sigmas: tuple[float, ...] = (1, 2, 3)
    alpha: Optional[float] = None
    black_ridges: bool = False
    mode: str = "reflect"
    cval: float = 0

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
        image.detect_mat[:] = meijering(
                image=image.detect_mat[:],
                sigmas=self.sigmas,
                alpha=self.alpha,
                black_ridges=self.black_ridges,
                mode=self.mode,
                cval=self.cval,
        )
        return image
