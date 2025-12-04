from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image

import numpy as np
from skimage.morphology import thin

from phenotypic.abc_ import ObjectRefiner


class Thinning(ObjectRefiner):
    """Progressively thin object masks to single-pixel-wide structures via morphological erosion.

    Intuition:
        Thinning iteratively removes outer pixels while preserving connectivity and 2x2 blocks,
        gradually reducing object width to 1 pixel. Unlike skeletonization, thinning does not
        require solving a medial axis problem, making it simpler and more predictable. On agar
        plates, thinning can clarify colony boundaries, separate overlapping regions, or prepare
        masks for filament analysis by stripping away outer noise layers step-by-step.

    Why this is useful for agar plates:
        Colonies touching or with diffuse edges can be gradually separated by removing outer
        pixels without aggressive erosion. Thinning is gentler than opening and can preserve
        thin filamentous structures better than binary erosion, making it ideal for multi-pass
        boundary cleaning before advanced morphological measurements.

    Use cases:
        - Gradually separating touching or overlapping colonies via controlled pixel removal.
        - Clarifying colony boundaries by removing diffuse outer layers.
        - Preparing masks for graph-based analysis (converting to 1-pixel skeletons).
        - De-noising edges while preserving internal structure (e.g., before branching analysis).

    Caveats:
        - Each iteration removes pixels; too many iterations will obliterate small features or
          break thin filaments.
        - Thinning alone does not always separate touching colonies; combine with opening or
          seed-based separation for better results.
        - The algorithm preserves 2x2 blocks, which can leave small "square" artifacts in
          very compact colonies.
        - On highly fragmented or noise-laden masks, iterations may not converge smoothly;
          pre-clean with SmallObjectRemover.

    Attributes:
        max_num_iter (int | None): Maximum number of thinning iterations. Each iteration
            removes outer-layer pixels while maintaining topology.
            - None (default): Iterate until convergence (no changes detected).
            - Positive int: Stop after N iterations regardless of convergence.

            Larger max_num_iter values progressively thin more aggressively. For separating
            touching colonies, try 1-3 iterations; for full skeletonization, set max_num_iter
            higher (10-50 depending on colony size).

    Examples:
        .. dropdown:: Gradually thin colonies to clarify boundaries

            >>> from phenotypic.refine import Thinning
            >>> op = Thinning(max_num_iter=2)
            >>> image = op.apply(image, inplace=True)  # doctest: +SKIP

        .. dropdown:: Thin to convergence (full skeleton)

            >>> from phenotypic.refine import Thinning
            >>> op = Thinning()  # Iterate until no change
            >>> image = op.apply(image, inplace=True)  # doctest: +SKIP

    Raises:
        ValueError: If ``max_num_iter`` is negative (checked during operation).
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
