from __future__ import annotations
from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
from pydantic import field_validator
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

from phenotypic.abc_ import ImageEnhancer


class SatoRidgeFilter(ImageEnhancer):
    """Enhance tubular and ridge-like structures in ``detect_mat`` with the Sato tubeness filter.

    Computes the Sato tubeness measure from Hessian matrix eigenvalues to
    highlight continuous ridge structures such as filamentous colonies,
    mycelial networks, and branching morphologies. Less sensitive to
    parameter tuning than Frangi, making it a good first choice for ridge
    detection.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Args:
        sigmas: Sequence of standard deviations for Gaussian derivatives.
            Smaller values detect finer structures; larger values detect
            thicker features. Typical range: ``(1, 2, 3)`` to
            ``range(1, 10, 2)``. Default: ``(1, 2, 3)``.
        black_ridges: If ``True``, detect dark ridges on bright background.
            If ``False`` (default), detect bright ridges on dark background.
        mode: Boundary handling. Accepted values: ``'constant'``,
            ``'reflect'``, ``'wrap'``, ``'nearest'``, ``'mirror'``.
            Default: ``'reflect'``.
        cval: Fill value when ``mode='constant'``. Default: 0.

    Returns:
        Image: Input image with ``detect_mat`` replaced by the Sato
        tubeness response map. ``rgb`` and ``gray`` are unchanged.

    Best For:
        - Thin filamentous colonies or mycelial networks (fungi, Bacillus,
          streptomycetes).
        - Continuous ridge-like structures that global thresholding misses.
        - Interconnected fungal networks or biofilm structures.
        - Organisms with branching or root-like colony morphologies.

    Consider Also:
        - :class:`MeijeringRidgeFilter` for very fine neurite-like filaments
          where higher selectivity is needed.
        - :class:`HessianFilter` for combined edge and ridge detection with
          blob sensitivity control.
        - :class:`StructureSmoothing` for anisotropic smoothing
          that enhances directional structures before ridge detection.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a
        visual walkthrough of ridge enhancement on plate images.
        :doc:`/explanation/what_enhancement_does` for background on
        Hessian-based ridge detection methods.
    """

    sigmas: tuple[float, ...] = (1, 2, 3)
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
        # Manual Sato tubeness loop (replaces skimage.filters.sato) for
        # explicit deletion of Hessian intermediates, reducing peak memory
        # by ~50% for multi-sigma runs. Algorithm: Sato et al. 1998, eqs. 9/22.
        img = np.asarray(image.detect_mat[:], dtype=np.float32)

        if not self.black_ridges:
            img = -img

        filtered_max = np.zeros(img.shape, dtype=np.float32)

        for sigma in self.sigmas:
            hessian_elems = hessian_matrix(
                    img, sigma=sigma, mode=self.mode, cval=self.cval,
                    use_gaussian_derivatives=True,
            )
            eigvals = hessian_matrix_eigvals(hessian_elems)
            del hessian_elems

            eigvals = eigvals[:-1]
            filtered = (
                    sigma ** 2
                    * np.prod(np.maximum(eigvals, 0), axis=0) ** (1.0 / len(eigvals))
            )
            del eigvals
            np.maximum(filtered_max, filtered, out=filtered_max)
            del filtered

        image.detect_mat[:] = filtered_max
        return image
