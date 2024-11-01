import numpy as np
from typing import Optional

from phenoscope.grid import GriddedImage
from phenoscope.grid.interface import GridMapModifier
from phenoscope.grid.feature_extraction import _grid_linreg_stats_extractor


class LinRegResidualOutlierModifier(GridMapModifier):
    def __init__(self, axis: Optional[int] = None, stddev_multiplier=1.5):
        self.axis = axis  # Either none, 0, or 1
        self.stddev_multiplier = stddev_multiplier

    def _operate(self, image: GriddedImage):
        # TODO: Finish Implementation
        grid_info = image.grid_info
