from __future__ import annotations

from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image

import numpy as np
from skimage.morphology import skeletonize

from phenotypic.abc_ import ObjectRefiner


class Skeletonize(ObjectRefiner):
    """Reduce object masks to single-pixel-wide skeletons using medial axis thinning.

    Intuition:
        Skeletonization compresses object regions to their medial axes (centerlines),
        preserving topological structure while reducing to 1-pixel width. On agar plates,
        this distills colony morphology to its core branching structure, useful for
        analyzing filamentous or spreading phenotypes without boundary noise. The method
        efficiently extracts the 'backbone' of colony shape.

    Why this is useful for agar plates:
        Colonies may have ragged edges, uneven staining, or noise that obscures their true
        spreading pattern. Skeletons expose the essential branching or directional growth,
        enabling more robust morphological features (e.g., branch count, elongation)
        and simplifying hyphae or filament tracking in fungal cultures.

    Use cases:
        - Extracting colony centerlines for elongation or orientation analysis.
        - Analyzing branching patterns in spreading/filamentous phenotypes.
        - Simplifying masks for spatial graph analysis or filament tracking.
        - Reducing noise in masks before measuring advanced morphological features.

    Caveats:
        - Skeletonization destroys width information; use only if you need topology,
          not colony boundary details.
        - Thin or poorly-defined colonies may produce fragmented or spurious skeleton branches.
        - Method choice (Zhang vs. Lee) can affect branch detection; Zhang is optimized
          for clean 2D images, Lee is more robust but may produce slightly thicker structures.
        - Isolated noise pixels may create spurious skeleton branches; apply cleanup
          (e.g., SmallObjectRemover) before skeletonizing.

    Attributes:
        method (Literal["zhang", "lee"] | None): Thinning algorithm to use.
            - "zhang": Fast, optimized for 2D images with clean topology. May produce
              thin artifacts on noisy images.
            - "lee": Works on 2D/3D, more robust to noise and irregular boundaries.
              Slightly slower than Zhang.
            - None: Auto-select based on image dimensionality (Zhang for 2D, Lee for 3D).

    Examples:
        .. dropdown:: Reduce filamentous colony to medial axis skeleton

            >>> from phenotypic.refine import Skeletonize
            >>> op = Skeletonize(method="zhang")
            >>> image = op.apply(image, inplace=True)  # doctest: +SKIP

    Raises:
        ValueError: If an invalid ``method`` is provided (checked during operation).
    """

    def __init__(self, method: Literal["zhang", "lee"] | None = None):
        """Initialize the skeletonizer.

        Args:
            method (Literal["zhang", "lee"] | None): Algorithm for skeletonization.
                - "zhang": Optimized for 2D images; fast, produces thin skeletons.
                  Best for well-defined colony boundaries.
                - "lee": Works on 2D/3D; more robust to noisy or irregular boundaries.
                  Slightly slower but preserves topology better on challenging images.
                - None: Automatically selects Zhang for 2D and Lee for 3D.

                Choosing the right method depends on image quality: clean, binary
                masks benefit from Zhang; noisier masks or fungal hyphae benefit
                from Lee.
        """
        super().__init__()
        self.method: Literal["zhang", "lee"] | None = method

    def _operate(self, image: Image) -> Image:
        image.objmask[:] = skeletonize(image.objmask[:], method=self.method)
        return image
