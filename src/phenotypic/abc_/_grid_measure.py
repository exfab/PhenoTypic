from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import GridImage

import pandas as pd

from phenotypic.abc_ import MeasureFeatures
from phenotypic.tools.exceptions_ import GridImageInputError, OutputValueError
from phenotypic.tools.funcs_ import validate_measure_integrity
from abc import ABC


class GridMeasureFeatures(MeasureFeatures, ABC):
    """Extract feature measurements from detected colonies in GridImage objects.

    GridMeasureFeatures is a type-safe wrapper around MeasureFeatures that enforces
    GridImage input type. It is to MeasureFeatures what GridOperation is to ImageOperation:
    a specialization for grid-aware (arrayed plate) analysis.

    **Purpose**

    Use GridMeasureFeatures when implementing measurement operations that extract
    quantitative metrics from colonies in grid-structured agar plate images. Like
    MeasureFeatures, it returns pandas DataFrames with one row per detected colony.
    The only difference is that it requires GridImage input, making explicit that your
    measurement may leverage grid structure (well positions, row/column layout) if desired.

    **GridImage vs Image**

    - **Image:** Generic image with optional, unvalidated grid information.
    - **GridImage:** Specialized Image subclass with validated grid structure (row/column
      layout, well positions, grid alignment). Suitable for 96-well, 384-well, or other
      arrayed plate formats.

    **When to use GridMeasureFeatures vs MeasureFeatures**

    - **MeasureFeatures:** Measurement works equally well on any Image (with or without
      grid). Examples: colony size, color composition, morphology metrics that are
      computed globally. Use when grid structure is irrelevant.

    - **GridMeasureFeatures:** Measurement leverages grid structure or assumes well-level
      organization. Examples: per-well growth metrics, grid-aligned morphology, measurements
      that depend on row/column position. Use when grid structure is essential or enhances
      the measurement.

    **Implementation Pattern**

    Inherit from GridMeasureFeatures and implement ``_operate()`` as normal:

    .. code-block:: python

        from phenotypic.abc_ import GridMeasureFeatures
        import pandas as pd

        class GridMeasureWellOccupancy(GridMeasureFeatures):
            '''Measure fraction of well area occupied by colonies.'''

            def _operate(self, image: GridImage) -> pd.DataFrame:
                # image is guaranteed to be GridImage with grid structure
                # Implement your grid-aware measurement here
                results = pd.DataFrame(...)
                return results

    **Typical Use Cases**

    - Per-well phenotypic analysis where well position matters
    - Grid-based filtering (e.g., "measure only colonies in the center wells")
    - Well-normalized metrics (e.g., colony area relative to well size)
    - Multi-well experiments where you need to track which well each measurement came from

    **Notes**

    - The ``measure()`` method is inherited from MeasureFeatures; the only difference
      is input type validation.
    - Returns pandas.DataFrame with one row per detected object, first column is
      OBJECT.LABEL (matching image.objmap labels).
    - GridImage must have valid grid structure set before measuring. Typically set
      by GridFinder or GridCorrector operations in the pipeline.
    - All helper methods from MeasureFeatures (mean, median, sum, etc.) are available.

    Examples:
        .. dropdown:: Grid-aware measurement of colony size per well

            .. code-block:: python

                from phenotypic import GridImage
                from phenotypic.abc_ import GridMeasureFeatures
                from phenotypic.tools.constants_ import OBJECT
                import pandas as pd

                class MeasureWellOccupancy(GridMeasureFeatures):
                    \"\"\"Measure total area occupied in each well.\"\"\"

                    def _operate(self, image: GridImage) -> pd.DataFrame:
                        # Use grid accessor to calculate per-well metrics
                        area = self._calculate_sum(image.objmask[:], image.objmap[:])
                        well_info = image.grid.info()  # Get well assignments

                        # Combine area with well location
                        results = pd.DataFrame({
                            'WellArea': area,
                        })
                        results.insert(0, OBJECT.LABEL, image.objects.labels2series())
                        return results

                # Usage
                from phenotypic import Image
                from phenotypic.detect import OtsuDetector

                image = Image.from_image_path('plate.jpg')
                image = OtsuDetector().operate(image)

                grid_image = GridImage(image)
                grid_image.detect_grid()  # Establish grid structure

                measurer = MeasureWellOccupancy()
                df = measurer.measure(grid_image)  # Returns grid-aware measurements
    """

    @validate_measure_integrity()
    def measure(self, image: GridImage) -> pd.DataFrame:
        from phenotypic import GridImage

        if not isinstance(image, GridImage):
            raise GridImageInputError()
        output = super().measure(image)
        if not isinstance(output, pd.DataFrame):
            raise OutputValueError("pandas.DataFrame")
        return output
