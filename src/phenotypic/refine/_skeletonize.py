from __future__ import annotations

from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from skimage.morphology import skeletonize

from phenotypic.abc_ import ObjectRefiner


class Skeletonize(ObjectRefiner):
    """Reduce object masks to single-pixel-wide skeletons via medial-axis thinning.

    Compresses each detected region to its medial axis (centerline),
    preserving topological connectivity while discarding all interior and
    boundary pixels. The result captures the branching structure and
    elongation of each colony without the area contribution of filled masks.

    For a comparison of morphological refinement methods, see
    :doc:`/explanation/refinement_strategies`.

    Best For:
        - Extracting colony centerlines for elongation or orientation
          analysis.
        - Analyzing branching patterns in filamentous fungi or spreading
          bacterial phenotypes.
        - Simplifying masks before spatial graph analysis or hyphal
          network tracing.
        - Reducing boundary noise before measuring advanced morphological
          features such as tortuosity or branch count.

    Consider Also:
        - :class:`Thinning` for iterative boundary peeling with explicit
          control over the number of thinning steps.
        - :class:`MaskGradient` when boundary outlines are needed rather
          than medial axes.
        - :class:`MaskErosion` for uniform inward shrinking that preserves
          filled object interiors.

    Args:
        method: Skeletonization algorithm. ``"zhang"`` applies Zhang--Suen
            fast parallel thinning for 2-D binary masks. ``"lee"`` uses
            Lee's octree-based algorithm, which also supports 3-D arrays.
            ``None`` selects automatically based on array dimensionality.
            Default: None.

    Returns:
        Image: Input image with ``objmask`` replaced by the
        single-pixel-wide medial-axis skeleton. ``objmap`` is unchanged.

    Raises:
        ValueError: If ``method`` is not ``"zhang"``, ``"lee"``, or
            ``None``.

    See Also:
        :doc:`/how_to/notebooks/refine_noisy_boundaries` for skeleton-based
        analysis workflows on real plate images.
        :doc:`/explanation/refinement_strategies` for a comparison of
        morphological refinement methods.
    """

    method: Literal["zhang", "lee"] | None = None

    def _operate(self, image: Image) -> Image:
        image.objmask[:] = skeletonize(image.objmask.copy(), method=self.method)
        return image
