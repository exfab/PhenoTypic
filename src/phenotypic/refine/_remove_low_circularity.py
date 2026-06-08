from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import pandas as pd
import numpy as np
from pydantic import field_validator
from skimage.measure import regionprops_table
import math

from ..abc_ import ObjectRefiner
from phenotypic.schema import OBJECT


class RemoveLowCircularity(ObjectRefiner):
    """Remove objects whose Polsby-Popper circularity score falls below a threshold.

    Computes ``4π × area / perimeter²`` for each labeled object and
    discards those below ``cutoff``. A perfect circle scores 1.0; elongated
    or jagged shapes score lower. Well-formed round colonies are retained
    while scratches, merged blobs, and irregular segmentation debris are
    removed.

    For an overview of morphological refinement strategies, see
    :doc:`/explanation/refinement_strategies`.

    Best For:
        - Post-threshold cleanup to exclude elongated scratches, merged
          colony blobs, or agar-edge artefacts before size or shape
          measurement.
        - High-throughput grid assays where all genuine colonies are
          expected to be round and irregular detections indicate artefacts.
        - Plates with round yeast or bacterial colonies where any
          non-circular detection is reliably spurious.

    Consider Also:
        - :class:`SmallObjectRemover` when artefacts are better
          distinguished by area than by shape, or when colony morphology
          is legitimately non-circular.
        - :class:`RemoveBorderObjects` when irregular detections cluster
          near the plate edge because of partial cropping.
        - :class:`MaskOpening` for smoothing jagged boundaries before
          circularity filtering so that genuine colonies are not penalised
          by pixelation artefacts in the perimeter measurement.

    Args:
        cutoff: Minimum Polsby-Popper circularity score in ``[0, 1]``
            required to retain an object. A perfect circle has a score of
            1.0; elongated or irregular shapes score lower. The default
            0.785 (π/4) is the area ratio of a circle inscribed in its
            bounding square and provides a reasonable boundary between
            compact colonies and elongated artefacts. Higher values enforce
            stricter roundness; lower values tolerate irregular morphology.
            Typical range: 0.5--0.9. Default: 0.785.

    Returns:
        Image: Input image with ``objmap`` and ``objmask`` updated to
        exclude all objects whose circularity is at or below ``cutoff``.
        ``rgb``, ``gray``, and ``detect_mat`` are unchanged.

    Raises:
        ValueError: If ``cutoff`` is outside ``[0, 1]``.

    See Also:
        :doc:`/how_to/notebooks/refine_noisy_boundaries` for shape-based
        cleanup workflows on real plate images.
        :doc:`/explanation/refinement_strategies` for a comparison of
        morphological refinement methods.
    """

    cutoff: float = 0.785

    @field_validator("cutoff")
    @classmethod
    def _validate_cutoff(cls, cutoff: float) -> float:
        """Reject a ``cutoff`` outside ``[0, 1]``.

        Reproduces the pre-migration ``__init__`` guard verbatim.
        """
        if cutoff < 0 or cutoff > 1:
            raise ValueError("threshold should be a number between 0 and 1.")
        return cutoff

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
