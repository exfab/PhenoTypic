from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from phenotypic import GridImage

from phenotypic.abc_ import GridObjectRefiner
from phenotypic.tools.constants_ import GRID, OBJECT


class MergeGridObj(GridObjectRefiner):
    """This operation merges all the objects within a grid section into one object per
    section."""

    def _operate(self, image: GridImage) -> GridImage:
        # Cache the objmap into memory and make a copy for editing
        objmap = image.objmap[:]
        new_objmap = image.objmap[:].copy()

        grid_info: pd.DataFrame = image.grid.info()
        for i, gs in enumerate(grid_info[GRID.SECTION_NUM].unique()):
            subtable = grid_info.loc[grid_info.loc[:, GRID.SECTION_NUM] == gs, :]

            new_objmap[np.isin(objmap, subtable.loc[:, OBJECT.LABEL])] = i + 1
