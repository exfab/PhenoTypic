from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenoscope import Image

import pandas as pd

from phenoscope.abstract import FeatureExtractor
from phenoscope.grid.abstract import GridOperation


class GridFinder(FeatureExtractor, GridOperation):
    """
    GridFinder measures grid information from the objects in various ways. Using the names here allow for streamlined integration.
    Unlike other Grid series interfaces, GridExtractors can work on regular images, and should not be dependent on the GridImage class.

    """
    def _operate(self, image: Image) -> pd.DataFrame:
        pass
