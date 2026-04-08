from __future__ import annotations
from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from skimage.filters import hessian

from phenotypic.abc_ import ImageEnhancer


class HessianFilter(ImageEnhancer):
    """Enhance edges and ridge-like structures via multi-scale Hessian filtering.

    Computes eigenvalue-based Hessian responses across multiple scales to
    highlight colony boundaries, thin filamentous structures, and ridge-like
    features in ``detect_mat``. Multi-scale analysis makes detection robust
    across varying colony sizes and morphologies.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Best For:
        - Sharp boundaries between colonies and agar background.
        - Thin or elongated structures (filaments, branching) with poor
          contrast.
        - Size-invariant colony edge enhancement before thresholding.
        - Textured colonies or biofilms with complex internal structure.

    Consider Also:
        - :class:`SatoRidgeFilter` for continuous tube-like structures where
          Hessian eigenvalue ratios provide cleaner ridge responses.
        - :class:`MeijeringRidgeFilter` for very fine neurite-like filaments.
        - :class:`LaplaceEnhancer` for simpler second-derivative edge
          detection without multi-scale analysis.

    Args:
        sigmas: Sequence of standard deviations for Gaussian derivatives.
            Smaller values detect finer edges; larger values detect broader
            structures. Typical range: ``(1, 2, 3)`` to ``(1, 5)``.
            Default: ``(1, 2, 3)``.
        alpha: Sensitivity to plate-like structure deviations. Lower values
            are more permissive. Typical range: 0.1--1.0. Default: 0.5.
        beta: Sensitivity to blob-like structure deviations. Lower values
            are more permissive. Typical range: 0.1--1.0. Default: 0.5.
        gamma: Background suppression threshold. Larger values suppress
            low-curvature regions (agar background) more aggressively.
            Typical range: 10--20. Default: 15.
        black_ridges: If ``True``, detect dark ridges on bright background.
            If ``False`` (default), detect bright ridges on dark background.
        mode: Boundary handling. Accepted values: ``'reflect'``,
            ``'constant'``, ``'nearest'``, ``'mirror'``, ``'wrap'``.
            Default: ``'reflect'``.
        cval: Fill value when ``mode='constant'``. Default: 0.

    Returns:
        Image: Input image with ``detect_mat`` replaced by the Hessian
        ridge response map. ``rgb`` and ``gray`` are unchanged.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a
        visual walkthrough of ridge and edge enhancement on plate images.
        :doc:`/explanation/what_enhancement_does` for background on
        Hessian-based structure detection.
    """

    def __init__(
            self,
            sigmas: Iterable[float] = (1, 2, 3),
            alpha: float = 0.5,
            beta: float = 0.5,
            gamma: float = 15,
            black_ridges: bool = False,
            mode: str = 'reflect',
            cval: float = 0,
    ):
        """
        Parameters:
            sigmas (tuple | list): Sequence of standard deviations for Gaussian
                derivatives. Smaller values detect finer edges, larger values
                detect thicker structures. Default (1, 2, 3).
            alpha (float): Sensitivity to plate-like structure deviations. Lower
                values are more permissive. Range: 0 to 1. Default 0.5.
            beta (float): Sensitivity to blob-like structure deviations. Lower
                values are more permissive. Range: 0 to 1. Default 0.5.
            gamma (float): Threshold for background suppression. Larger values
                suppress low-curvature regions (agar background) more aggressively.
                Default 15.
            black_ridges (bool): If True, detect dark ridges (colonies) on bright
                background. If False, detect bright ridges on dark background.
                For agar plates with dark colonies on light background, use True.
                Default False.
            mode (str): Boundary handling mode ('reflect', 'constant', 'nearest',
                'mirror', 'wrap'). Default 'reflect'.
            cval (float): Constant value used if mode='constant'. Default 0.
        """
        self.sigmas = sigmas
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.black_ridges = black_ridges
        self.mode = mode
        self.cval = cval

    def __setattr__(self, name: str, value: object) -> None:
        if name == "sigmas" and value is not None:
            value = tuple(value)  # type: ignore[arg-type]
        super().__setattr__(name, value)

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
