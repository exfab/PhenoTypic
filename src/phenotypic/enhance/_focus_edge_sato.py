from __future__ import annotations
from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
from pydantic import field_validator
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

from phenotypic.abc_ import FocusEdge


class FocusEdgeSato(FocusEdge):
    """Enhance hyphal ridges and tubular colony structures in ``detect_mat`` using the Sato tubeness filter.

    Computes a tubeness response from Hessian matrix eigenvalues at each
    specified scale, then takes the per-pixel maximum across all scales to
    produce a response map where bright ridges correspond to continuous
    filamentous structures. Intermediates are deleted between scales to
    reduce peak memory usage.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Best For:
        - Thin filamentous colonies and mycelial networks (Aspergillus,
          Neurospora, streptomycetes).
        - Continuous ridge-like morphologies that global thresholding misses.
        - Branching or root-like colony forms requiring multi-scale detection.
        - Plates where hyphal width varies across the image and a single
          sigma would miss structures at one end of the size range.

    Consider Also:
        - :class:`FocusEdgeFrangi` when blob-versus-tube discrimination is
          needed via explicit alpha/beta sensitivity controls.
        - :class:`FocusEdgeHessian` for combined edge and ridge detection with
          adjustable background suppression.
        - :class:`StructureSmoothing` when anisotropic preprocessing is needed
          to reinforce coherent hyphal orientation before ridge detection.

    Args:
        sigmas: Scales (standard deviations in pixels) at which the Hessian
            is evaluated. Each sigma responds maximally to ridges whose
            cross-sectional half-width matches that value. The output is the
            per-pixel maximum across all scales, so extra sigmas can only
            raise the response. Typical tuple span: ``(1, 2, 3)`` for
            standard 300--600 dpi scans where hyphae are 2--8 px wide;
            extend to ``range(1, 10, 2)`` for thick mature filaments or lower
            magnification. A reasonable starting point for standard agar plate
            scans is to span the expected minimum and maximum hyphal width in
            pixels, e.g. ``(1, 3, 5)`` for 600 dpi images of Neurospora.
            Default: ``(1, 2, 3)``.
        black_ridges: Detect dark ridges on a bright background when ``True``.
            ``False`` (default) detects bright ridges on a dark background,
            matching the ``detect_mat`` convention where colonies appear bright.
        mode: Boundary handling for Gaussian derivative convolution. Accepted
            values: ``'constant'``, ``'reflect'``, ``'wrap'``, ``'nearest'``,
            ``'mirror'``. ``'reflect'`` (default) mirrors image data at the
            border, minimising spurious ridge responses at plate edges.
            Use ``'constant'`` (with ``cval`` set to the background level)
            only when stitching multi-tile acquisitions.
        cval: Fill value used when ``mode='constant'``. Has no effect for
            other border modes. Default: 0.

    Returns:
        Image: Input image with ``detect_mat`` replaced by the Sato tubeness
        response map. Brighter pixels indicate stronger ridge-like structures
        at the sampled scales. ``rgb`` and ``gray`` are unchanged.

    References:
        [1] Y. Sato et al., "Three-dimensional multi-scale line filter for
        segmentation and visualization of curvilinear structures in medical
        images," *Med. Image Anal.*, vol. 2, no. 2, pp. 143--168, Jun. 1998.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a
        visual walkthrough of ridge enhancement on plate images.
        :doc:`/explanation/what_enhancement_does` for background on
        Hessian-based ridge detection methods.
        :doc:`/tutorials/notebooks/10_detecting_filamentous_fungi` for
        filamentous fungi detection pipelines that use this enhancer.
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
