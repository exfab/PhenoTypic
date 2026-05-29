from __future__ import annotations
from typing import Iterable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from pydantic import field_validator
from skimage.filters import meijering

from phenotypic.abc_ import ImageEnhancer


class MeijeringRidgeFilter(ImageEnhancer):
    """Enhance fine ridge-like structures in ``detect_mat`` with the Meijering neuriteness filter.

    Computes the Meijering neuriteness measure from Hessian matrix eigenvalues
    to highlight elongated, thread-like structures such as delicate filaments,
    thin wrinkles, and network-like features. More selective than
    :class:`SatoRidgeFilter` for very fine, well-separated ridges.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Args:
        sigmas: Sequence of standard deviations for Gaussian derivatives.
            Smaller values detect finer structures; larger values detect
            thicker features. Typical range: ``(1, 2, 3)`` to
            ``range(1, 10)``. Default: ``(1, 2, 3)``.
        alpha: Shape parameter controlling linearity sensitivity. ``None``
            (default) uses ``-1/(ndim+1)`` which is ``-1/3`` for 2D
            images. Rarely requires manual tuning.
        black_ridges: If ``True``, detect dark ridges on bright background.
            If ``False`` (default), detect bright ridges on dark background.
        mode: Boundary handling. Accepted values: ``'constant'``,
            ``'reflect'``, ``'wrap'``, ``'nearest'``, ``'mirror'``.
            Default: ``'reflect'``.
        cval: Fill value when ``mode='constant'``. Default: 0.

    Returns:
        Image: Input image with ``detect_mat`` replaced by the Meijering
        neuriteness response map. ``rgb`` and ``gray`` are unchanged.

    Best For:
        - Delicate filamentous structures too thin for standard detection
          (actinomycetes, fungal hyphae, bacterial networks).
        - Fine wrinkles, grooves, or network features in biofilms.
        - Sparse mycelial networks or bacterial filaments that require
          sensitive ridge detection.

    Consider Also:
        - :class:`SatoRidgeFilter` for thicker, continuous tubular
          structures with less sensitivity to parameter tuning.
        - :class:`HessianFilter` for combined edge and ridge detection
          with blob sensitivity control.
        - :class:`StructureSmoothing` for enhancing directional
          structures via anisotropic smoothing before ridge detection.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a
        visual walkthrough of ridge enhancement on plate images.
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
