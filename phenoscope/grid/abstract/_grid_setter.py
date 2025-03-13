import pandas as pd

from phenoscope import Image
from phenoscope.abstract import FeatureExtractor
from phenoscope.grid.abstract import GridOperation
from phenoscope.util.constants import C_Grid


class GridSetter(FeatureExtractor, GridOperation):
    """
    Grid setters measure grid information from the objects in various ways. Using the names here allow for streamlined integration.
    Unlike other Grid series interfaces, GridExtractors can work on regular images and gridded images
    """
    def _operate(self, image: Image) -> pd.DataFrame:
        pass
