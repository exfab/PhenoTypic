from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import pandas as pd
import numpy as np
from skimage.measure import regionprops_table
import math

from ..abc_ import ObjectRefiner
from ..tools.constants_ import OBJECT


class LowCircularityRemover(ObjectRefiner):
    """Remove objects with circularity below a specified cutoff.

    Intuition:
        Single bacterial/fungal colonies on agar often appear approximately
        round. Irregular, elongated, or fragmented shapes can indicate merged
        colonies, scratches, agar texture, or segmentation errors. Filtering by
        circularity keeps well-formed colonies and removes unlikely shapes.

    Why this is useful for agar plates:
        Circular colonies produce more reliable area and intensity measurements.
        Removing low-circularity detections reduces bias from streaks, debris,
        or incomplete segmentation near plate edges and grid borders.

    Use cases:
        - Post-threshold cleanup to exclude elongated artifacts or merged
          colonies before counting or phenotyping.
        - Enforcing morphology consistency in high-throughput grid assays.

    Caveats:
        - Some colonies are intrinsically irregular (wrinkled, spreading,
          filamentous). A high cutoff may incorrectly remove these phenotypes.
        - Perimeter estimates on low-resolution masks can be noisy, slightly
          biasing the circularity calculation.

    Attributes:
        cutoff (float): Minimum Polsby–Popper circularity required to keep an
            object, in [0, 1]. Higher values retain only near-circular shapes
            (sharper shape constraints) and can improve edge sharpness in the
            kept set but may reduce recall for irregular colonies.

    Examples:
        >>> from phenotypic.refine import LowCircularityRemover
        >>> op = LowCircularityRemover(cutoff=0.8)
        >>> image = op.apply(image, inplace=True)  # doctest: +SKIP
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
        if cutoff < 0 or cutoff > 1: raise ValueError('threshold should be a number between 0 and 1.')
        self.cutoff = cutoff

    def _operate(self, image: Image) -> Image:
        # Create intial measurement table
        table = (pd.DataFrame(regionprops_table(label_image=image.objmap[:], intensity_image=image.gray[:],
                                                properties=['label', 'area', 'perimeter']
                                                )
                              )
                 .rename(columns={'label': OBJECT.LABEL})
                 .set_index(OBJECT.LABEL))

        # Calculate circularity based on Polsby-Popper Score
        table['circularity'] = (4*math.pi*table['area'])/(table['perimeter'] ** 2)

        passing_objects = table[table['circularity'] > self.cutoff]
        failed_object_boolean_indices = ~(
            np.isin(element=image.objmap[:], test_elements=passing_objects.index.to_numpy()))
        image.objmap[failed_object_boolean_indices] = 0
        return image
