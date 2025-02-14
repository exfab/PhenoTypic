import pandas as pd

from phenoscope import Image
from phenoscope.interface import FeatureExtractor
from phenoscope.grid.interface import GridOperation
from phenoscope.util.constants import INTERFACE_ERROR_MSG


class GridExtractor(FeatureExtractor, GridOperation):
    """
    Grid extractores extract grid information from the objects in various ways. Using the names here allow for streamlined integration.
    Unlike other Grid series interfaces, GridExtractors can work on regular images and gridded images
    """
    LABEL_GRID_ROW_NUM = 'Grid_RowNum'
    LABEL_GRID_ROW_INTERVAL = 'Grid_RowInterval'

    LABEL_GRID_COL_NUM = 'Grid_ColNum'
    LABEL_GRID_COL_INTERVAL = 'Grid_ColInterval'

    LABEL_GRID_SECTION_NUM = 'Grid_SectionNum'
    LABEL_GRID_SECTION_IDX = 'Grid_SectionIndex'
    LABEL_GRID_SECTION_INTERVAL = 'Grid_SectionInterval'

    # To be implemented in subclass constructors.
    n_rows: int = None
    n_cols: int = None

    def _operate(self, image: Image) -> pd.DataFrame:
        pass
