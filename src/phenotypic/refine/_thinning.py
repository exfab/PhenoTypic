from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from skimage.morphology import thin

from phenotypic.abc_ import ObjectRefiner


class Thinning(ObjectRefiner):
    """Progressively thin object masks by iteratively removing outer boundary pixels.

    Strips boundary pixels one layer at a time while preserving topological
    connectivity, gradually reducing each object toward single-pixel-wide
    structures. Explicit iteration control makes it suitable for anything
    from gentle edge cleanup to full centerline extraction.

    For a comparison of morphological refinement methods, see
    :doc:`/explanation/refinement_strategies`.

    Best For:
        - Gently cleaning diffuse or ragged colony boundaries before
          morphological measurements with a small number of iterations.
        - Gradually separating lightly touching colonies via controlled
          pixel removal without applying a full watershed.
        - Preparing masks for graph-based analysis by iterating to
          single-pixel-wide structures.
        - Preserving the overall shape of filamentous colonies while
          reducing spurious boundary protrusions.

    Consider Also:
        - :class:`Skeletonize` for direct medial-axis extraction without
          per-iteration control.
        - :class:`MaskErosion` for uniform inward shrinking governed by a
          configurable structuring element rather than iterative peeling.
        - :class:`SeparateObjects` for watershed-based separation when
          colonies are firmly merged rather than lightly touching.

    Args:
        max_num_iter: Maximum number of thinning iterations. ``None`` runs
            until convergence, yielding a fully thinned single-pixel-wide
            skeleton (Guo--Hall thinning, distinct from the Zhang--Suen/Lee
            algorithms in :class:`Skeletonize`, so the output is similar but
            not identical). Small values (1--5) provide gentle boundary
            cleanup; larger values thin more aggressively. Default: None.

    Returns:
        Image: Input image with ``objmask`` thinned by up to
        ``max_num_iter`` iterations. Assigning ``objmask`` rebuilds
        ``objmap`` from the thinned mask.

    See Also:
        :doc:`/how_to/notebooks/refine_noisy_boundaries` for thinning-based
        boundary cleanup workflows on real plate images.
        :doc:`/explanation/refinement_strategies` for a comparison of
        morphological refinement methods.
    """

    max_num_iter: int | None = None

    def _operate(self, image: Image) -> Image:
        image.objmask[:] = thin(image.objmask[:], max_num_iter=self.max_num_iter)
        return image
