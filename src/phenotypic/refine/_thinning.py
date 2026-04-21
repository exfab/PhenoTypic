from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from skimage.morphology import thin

from phenotypic.abc_ import ObjectRefiner


class Thinning(ObjectRefiner):
    """Progressively thin object masks by iteratively removing outer pixels while preserving connectivity.

    Strips away boundary pixels one layer at a time, gradually reducing object
    width toward single-pixel structures. Unlike skeletonization, thinning
    offers explicit iteration control, making it useful for gentle boundary
    cleanup (few iterations) or full skeleton extraction (convergence).

    Args:
        max_num_iter: Maximum thinning iterations. ``None`` iterates until
            convergence (full skeleton). A small value (1--3) provides
            gentle boundary cleanup; a large value (10--50) thins
            aggressively. Default: None.

    Returns:
        Image: Input image with ``objmask`` thinned by the specified number
        of iterations.

    Raises:
        ValueError: If ``max_num_iter`` is negative.

    Best For:
        - Gradually separating touching or overlapping colonies via controlled
          pixel removal.
        - Clarifying diffuse colony boundaries before morphological
          measurements.
        - Preparing masks for graph-based analysis by converting to
          single-pixel skeletons.
        - De-noising colony edges while preserving filamentous structures.

    Consider Also:
        - :class:`Skeletonize` for direct medial-axis extraction without
          iterative control.
        - :class:`MaskEroder` for uniform inward shrinking with a
          configurable structuring element.
        - :class:`SeparateObjects` for watershed-based separation of
          touching colonies.

    See Also:
        :doc:`/how_to/notebooks/refine_noisy_boundaries` for thinning-based
        boundary cleanup workflows.
        :doc:`/explanation/refinement_strategies` for a comparison of
        morphological refinement methods.
    """

    def __init__(self, max_num_iter: int | None = None):
        """Initialize the thinner.

        Args:
            max_num_iter (int | None): Upper limit on iterations. Use:

                - None (default) to iterate until convergence, yielding a full skeleton.
                - A small int (e.g., 1-3) for gentle boundary cleanup while preserving
                  colony bulk.
                - A large int (e.g., 10-50) for aggressive thinning to single-pixel structures.

                Choosing max_num_iter is a trade-off: few iterations preserve colony
                size/robustness but may leave overlaps; many iterations separate more
                aggressively but risk removing small filaments or creating fragmentation.
        """
        super().__init__()
        self.max_num_iter: int | None = max_num_iter

    def _operate(self, image: Image) -> Image:
        image.objmask[:] = thin(image.objmask[:], max_num_iter=self.max_num_iter)
        return image
