from __future__ import annotations

from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from skimage.morphology import skeletonize

from phenotypic.abc_ import ObjectRefiner


class Skeletonize(ObjectRefiner):
    """Reduce object masks to single-pixel-wide skeletons via medial axis thinning.

    Compresses each object region to its medial axis (centerline), preserving
    topological connectivity while discarding boundary and interior pixels.
    Useful for distilling colony morphology to its core branching structure
    for filament or spreading phenotype analysis.

    Args:
        method: Thinning algorithm. ``"zhang"`` is fast and optimized for
            clean 2D masks. ``"lee"`` is more robust to noise and works on
            2D/3D. ``None`` auto-selects based on dimensionality. Default:
            None.

    Returns:
        Image: Input image with ``objmask`` replaced by the single-pixel-wide
        skeleton.

    Raises:
        ValueError: If an invalid ``method`` is provided.

    Best For:
        - Extracting colony centerlines for elongation or orientation
          analysis.
        - Analyzing branching patterns in filamentous fungi or spreading
          bacterial phenotypes.
        - Simplifying masks for spatial graph analysis or hyphae tracking.
        - Reducing boundary noise before measuring advanced morphological
          features.

    Consider Also:
        - :class:`Thinning` for iterative boundary peeling with control
          over the number of iterations.
        - :class:`MaskGradient` when you need boundary outlines rather
          than medial axes.
        - :class:`MaskErosion` for uniform inward shrinking that preserves
          filled regions.

    See Also:
        :doc:`/how_to/notebooks/refine_noisy_boundaries` for skeleton-based
        analysis workflows.
        :doc:`/explanation/refinement_strategies` for a comparison of
        morphological refinement methods.
    """

    method: Literal["zhang", "lee"] | None = None

    def _operate(self, image: Image) -> Image:
        image.objmask[:] = skeletonize(image.objmask.copy(), method=self.method)
        return image
