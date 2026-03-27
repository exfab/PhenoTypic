from __future__ import annotations
from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from skimage.filters import frangi

from phenotypic.abc_ import ImageEnhancer


class FrangiVesselness(ImageEnhancer):
    """Enhance tubular structures in detect_mat using Hessian-based vesselness filtering.

    Computes the Frangi vesselness measure from Hessian matrix eigenvalues at
    multiple scales, producing a response map that highlights elongated features
    (hyphae, branches, mycelial networks). The output is a probability-like map
    (0--1) that typically requires thresholding before detection.

    For algorithm details, see :doc:`/explanation/filamentous_fungi_algorithm`.

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

    References:
        [1] A. F. Frangi, W. J. Niessen, K. L. Vincken, and M. A. Viergever,
        "Multiscale vessel enhancement filtering," in *MICCAI*, 1998,
        pp. 130--137.

    See Also:
        :doc:`/tutorials/notebooks/10_detecting_filamentous_fungi` for a
        visual walkthrough of filamentous fungi detection.
        :doc:`/explanation/filamentous_fungi_algorithm` for the theory behind
        Hessian-based vesselness filtering.

    Attributes:
        sigmas (tuple | list): Sequence of standard deviations for Hessian
            computation. Each sigma represents a scale; default (1, 2, 3).
        alpha (float): Blobness parameter (0 to 1). Default 0.5.
        beta (float): Structuredness parameter (0 to 1). Default 0.5.
        gamma (float): Background suppression threshold. Default 15.
        black_ridges (bool): If True, detect dark structures on bright background;
            if False, detect bright structures on dark background. Default False.
    """

    def __init__(
            self,
            sigmas: Iterable[float] = (0.5, 1, 1.5),
            alpha: float = 0.5,
            beta: float = 0.5,
            gamma: float = None,
            black_ridges: bool = False,
    ):
        """
        Parameters:
            sigmas (tuple | list): Sequence of standard deviations for Gaussian
                derivatives. Smaller values detect finer features, larger values
                detect thicker structures. Default (0.5, 1, 1.5).
            alpha (float): Vesselness sensitivity to blobness. Lower values are
                more permissive. Range: 0 to 1. Default 0.5.
            beta (float): Vesselness sensitivity to structuredness. Lower values
                are more permissive. Range: 0 to 1. Default is None which uses
                half of the max Hessian norm.
            gamma (float): Threshold for background suppression. Larger values
                suppress low-curvature regions more aggressively. Default 15.
            black_ridges (bool): If True, detect dark ridges (colonies) on bright
                background. If False, detect bright ridges on dark background.
                For agar plates with dark colonies on light background, use True.
                Default False.
        """
        self.sigmas = sigmas
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.black_ridges = black_ridges

    def __setattr__(self, name: str, value: object) -> None:
        if name == "sigmas" and value is not None:
            value = tuple(value)  # type: ignore[arg-type]
        super().__setattr__(name, value)

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
