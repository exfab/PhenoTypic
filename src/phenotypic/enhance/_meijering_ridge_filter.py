from __future__ import annotations
from typing import Iterable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

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
        - :class:`CoherenceEnhancingDiffusion` for enhancing directional
          structures via anisotropic smoothing before ridge detection.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a
        visual walkthrough of ridge enhancement on plate images.
        :doc:`/explanation/what_enhancement_does` for background on
        Hessian-based ridge detection methods.
    """

    def __init__(
            self,
            sigmas: Iterable[float] = (1, 2, 3),
            alpha: Optional[float] = None,
            black_ridges: bool = False,
            mode: str = 'reflect',
            cval: float = 0,
    ):
        """
        Parameters:
            sigmas (tuple | list): Sequence of standard deviations for Gaussian
                derivatives. Smaller values detect finer features, larger values
                detect thicker structures. Default (1, 2, 3).
            alpha (float | None): Shape parameter controlling linearity sensitivity.
                Default None uses -1/(ndim+1), which for 2D images is -1/3.
                Unlikely to require manual tuning in practice.
            black_ridges (bool): If True, detect dark ridges (colonies) on bright
                background. If False, detect bright ridges on dark background.
                For agar plates with dark colonies on light background, use True.
                Default False.
            mode (str): How to handle image boundaries. Options: 'constant'
                (pad with cval), 'reflect' (mirror), 'wrap' (tile), 'nearest'
                (replicate edge), 'mirror' (symmetric mirror). Default 'reflect'.
            cval (float): Fill value when mode='constant'. Default 0.
        """
        self.sigmas = sigmas
        self.alpha = alpha
        self.black_ridges = black_ridges
        self.mode = mode
        self.cval = cval

    def __setattr__(self, name: str, value: object) -> None:
        if name == "sigmas" and value is not None:
            value = tuple(value)  # type: ignore[arg-type]
        super().__setattr__(name, value)

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
