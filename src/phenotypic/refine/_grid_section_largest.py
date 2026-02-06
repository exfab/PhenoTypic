from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from phenotypic import GridImage

from phenotypic.abc_ import ObjectRefiner
from phenotypic.measure import MeasureSize
from phenotypic.tools_.measurement_info_ import SIZE, GRID
from phenotypic.tools_.constants_ import OBJECT


class GridSectionLargest(ObjectRefiner):
    """Identifies and retains only the largest objects within each grid section.

    This class processes an image that contains microbial colonies segmented into
    grid sections on solid media agar plates. The goal is to identify and retain
    only the largest object within each grid section of the image, effectively
    filtering out smaller objects and noise.

    """

    def _operate(self, image: GridImage) -> GridImage:
        size_table = MeasureSize().measure(image, include_meta=True)
        max_idx = size_table.groupby(
                by=GRID.SECTION_NUM,
                observed=True
        )[SIZE.AREA].idxmax()
        max_size_labels = size_table.loc[max_idx, OBJECT.LABEL].to_numpy()

        # Drop objects not the largest
        nonmax_mask = ~np.isin(image.objmap[:], max_size_labels)
        image.objmap[nonmax_mask] = 0
        return image
