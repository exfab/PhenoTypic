from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import pandas as pd
import numpy as np
from skimage.measure import regionprops_table
import math

from ..abc_ import ObjectRefiner
from ..tools_.constants_ import OBJECT


class LowCircularityRemover(ObjectRefiner):
    """Remove objects whose Polsby-Popper circularity falls below a cutoff.

    Computes circularity as ``4 * pi * area / perimeter^2`` for each labeled
    object and discards those below the threshold. Keeps well-formed, roughly
    circular colonies while filtering out elongated artifacts, merged blobs,
    and segmentation debris.

    Best For:
        - Post-threshold cleanup to exclude elongated scratches or merged
          colonies before phenotyping.
        - Enforcing morphology consistency in high-throughput grid assays.
        - Plates with round yeast or bacterial colonies where irregular
          detections indicate artifacts.

    Consider Also:
        - :class:`SmallObjectRemover` when artifacts are distinguished by
          size rather than shape.
        - :class:`BorderObjectRemover` when irregular detections cluster
          near plate edges.
        - :class:`MaskOpener` for smoothing jagged boundaries before
          circularity filtering.

    Args:
        cutoff: Minimum Polsby-Popper circularity in ``[0, 1]`` required to
            retain an object. Typical range: 0.5--0.9. Higher values keep
            only near-circular shapes; lower values tolerate irregular
            morphologies. Default: 0.785.

    Returns:
        Image: Input image with ``objmap`` and ``objmask`` updated to exclude
        objects below the circularity cutoff.

    Raises:
        ValueError: If ``cutoff`` is outside ``[0, 1]``.

    See Also:
        :doc:`/how_to/notebooks/refine_noisy_boundaries` for shape-based
        cleanup workflows.
        :doc:`/explanation/refinement_strategies` for a comparison of
        morphological refinement methods.
    """

    def __init__(self, cutoff: float = 0.785):
        """Initialize the remover.

        Args:
            cutoff (float): Minimum allowed circularity in [0, 1]. Increasing
                the cutoff favors compact, round objects (often cleaner masks),
                whereas lowering it retains irregular colonies but may keep more
                debris or merged objects.

        Raises:
            ValueError: If ``cutoff`` is outside [0, 1].
        """
        if cutoff < 0 or cutoff > 1:
            raise ValueError("threshold should be a number between 0 and 1.")
        self.cutoff = cutoff

    def _operate(self, image: Image) -> Image:
        # Create intial measurement table
        table = (
            pd.DataFrame(
                    regionprops_table(
                            label_image=image.objmap[:],
                            intensity_image=image.gray[:],
                            properties=["label", "area", "perimeter"],
                    )
            )
            .rename(columns={"label": OBJECT.LABEL})
            .set_index(OBJECT.LABEL)
        )

        # Calculate circularity based on Polsby-Popper Score
        table["circularity"] = (4 * math.pi * table["area"]) / (table["perimeter"] ** 2)

        passing_objects = table[table["circularity"] > self.cutoff]
        failed_object_boolean_indices = ~(
            np.isin(
                    element=image.objmap[:],
                    test_elements=passing_objects.index.to_numpy()
            )
        )
        image.objmap[failed_object_boolean_indices] = 0
        return image
