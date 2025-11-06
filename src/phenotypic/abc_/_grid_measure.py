from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import GridImage

import pandas as pd

from phenotypic.abc_ import MeasureFeatures
from phenotypic.tools.exceptions_ import GridImageInputError, OutputValueError
from phenotypic.tools.funcs_ import validate_measure_integrity
from abc import ABC


class GridMeasureFeatures(MeasureFeatures, ABC):

    @validate_measure_integrity()
    def measure(self, image: GridImage) -> pd.DataFrame:
        from phenotypic import GridImage

        if not isinstance(image, GridImage): raise GridImageInputError()
        output = super().measure(image)
        if not isinstance(output, pd.DataFrame): raise OutputValueError("pandas.DataFrame")
        return output
