from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._grid_image import GridImage

import pandas as pd

from phenotypic.abc_ import MeasureFeatures
from phenotypic.tools_.exceptions_ import GridImageInputError, OutputValueError
from phenotypic.tools_.funcs_ import validate_measure_integrity
from abc import ABC


class GridMeasureFeatures(MeasureFeatures, ABC):
    """Extract feature measurements from detected colonies in GridImage objects.

    GridMeasureFeatures is a type-safe wrapper around MeasureFeatures that enforces
    GridImage input type. It is to MeasureFeatures what GridOperation is to ImageOperation:
    a specialization for grid-aware (arrayed plate) analysis.

    **What is GridMeasureFeatures?**

    GridMeasureFeatures enables measurements that leverage well structure in arrayed plate images:

    - **Type-safe wrapper:** Enforces GridImage input (guaranteed valid grid structure with rows, cols, well positions).
    - **Grid-aware measurements:** Access per-well positions, row/column layout via image.grid.info(), enabling
      well-normalized and position-aware metrics.
    - **Per-well metrics:** Compute measurements per colony but with awareness of which well it belongs to, well size,
      and row/column position.
    - **Relationship to MeasureFeatures:** Inherits all methods from MeasureFeatures (parent class). Only difference
      is stricter input type validation (GridImage required) plus guaranteed access to grid information.
    - **Output format:** Returns pandas.DataFrame with one row per detected colony, matching image.objmap labels.
      Can easily map back to grid layout.
    - **Well-normalized analysis:** Enables computing metrics relative to well size/position (e.g., colony area as
      fraction of well area, normalized by well position on plate).

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

    **Quick Decision Guide**

    Choose GridMeasureFeatures vs MeasureFeatures:

    - **GridMeasureFeatures:** Measurement depends on or benefits from well structure. Examples: per-well
      metrics, well-normalized values, measurements filtered by well position, edge well exclusion.
    - **MeasureFeatures:** Measurement works equally well on any Image. Examples: colony size, color,
      morphology computed globally without well awareness or position dependency.
    - **Type safety:** GridMeasureFeatures enforces GridImage; MeasureFeatures accepts any Image (grid optional).
    - **Well-level data:** Use GridMeasureFeatures when you need image.grid.info() for per-well analysis and
      when grid structure is essential to your measurement.
    - **Multi-well experiments:** GridMeasureFeatures simplifies tracking which well each colony belongs to,
      row/column position, and inter-well comparisons.
    - **Subclass reference:** Both GridMeasureSize and GridMeasureShape inherit from GridMeasureFeatures.
    - **Performance:** GridMeasureFeatures has minimal overhead vs MeasureFeatures; use it when grid structure matters.
    - **Flexibility:** If uncertain whether you need grid structure, start with MeasureFeatures; upgrade to
      GridMeasureFeatures if well-aware logic becomes necessary.

    **Implementation Pattern**

    Inherit from GridMeasureFeatures and implement ``_operate()`` as an instance method:

    .. code-block:: python

        from phenotypic.abc_ import GridMeasureFeatures
        from phenotypic import GridImage
        from phenotypic.tools_.constants_ import OBJECT
        import pandas as pd

        class GridMeasureWellOccupancy(GridMeasureFeatures):
            '''Measure fraction of well area occupied by colonies.'''

            def __init__(self, normalize: bool = True):
                super().__init__()
                self.normalize = normalize

            def _operate(self, image: GridImage) -> pd.DataFrame:
                # image is guaranteed to be GridImage with validated grid structure
                grid_info = image.grid.info()  # Access well positions and layout
                # grid_info contains: 'grid_shape', 'well_centers', 'well_size', etc.

                nrows, ncols = grid_info['grid_shape']
                well_size = grid_info['well_size']

                # Calculate area occupied by colonies using MeasureFeatures methods
                area = self._calculate_sum(image.objmask[:], image.objmap[:])

                # Optional: normalize by well size for grid-aware analysis
                if self.normalize and well_size > 0:
                    area = area / (well_size ** 2)

                # Build results DataFrame with colony labels (required by contract)
                results = pd.DataFrame({'WellArea': area})
                results.insert(0, OBJECT.LABEL, image.objects.labels2series())
                return results

    **Typical Use Cases**

    - **Per-well phenotypic analysis:** Extract growth metrics where well position and size matter for normalization
      (e.g., center wells may grow differently from edge wells due to evaporation).
    - **Grid-based filtering:** Measure only colonies in specific well regions (e.g., measure only center wells in
      high-variance experiments, exclude border wells due to contamination risk).
    - **Well-normalized metrics:** Compute colony area relative to well size, colony density within well boundaries,
      or occupancy rates per well in a 96-well or 384-well plate.
    - **Multi-well experiments:** Track which well each colony occupies via grid.info(), enabling growth curve fitting
      per well and inter-well statistical comparisons.
    - **Quality control filtering:** Exclude edge wells from analysis where plate handling artifacts are common, or
      normalize measurements by well position.

    **Helper Methods Available**

    Inherit all measurement methods from MeasureFeatures parent class:

    - ``_calculate_sum()`` - Total measurement per object (e.g., total area)
    - ``_calculate_mean()`` - Average measurement per object (e.g., mean intensity)
    - ``_calculate_median()`` - Median measurement per object
    - ``_calculate_std()`` - Standard deviation per object
    - ``_calculate_percentile()`` - Percentile-based measurements (e.g., 95th percentile size)
    - ``_extract_properties_per_object()`` - Extract properties from regionprops or similar

    **Grid Information Available**

    Access grid structure via ``image.grid.info()`` in _operate():

    - ``'grid_shape'`` - Tuple (nrows, ncols) of well layout
    - ``'well_centers'`` - List of (row_px, col_px) well center coordinates in pixel space
    - ``'well_size'`` - Typical well size in pixels (varies by format: 96-well vs 384-well)
    - ``'rotation_angle'`` - Current grid rotation angle in degrees
    - ``'alignment'`` - Alignment state/quality metrics

    **Notes**

    - The ``measure()`` method is inherited from MeasureFeatures; the only difference
      is input type validation (GridImage required).
    - Returns pandas.DataFrame with one row per detected object, first column is
      OBJECT.LABEL (matching image.objmap labels).
    - GridImage must have valid grid structure set before measuring. Typically set
      by GridFinder or GridCorrector operations in the pipeline.
    - All helper methods from MeasureFeatures (mean, median, sum, etc.) are available.
    - DataFrame returned can be easily mapped back to grid layout using image.grid.info().

    **Relationship to MeasureFeatures and GridOperation**

    - **MeasureFeatures:** Parent class providing all measurement logic. GridMeasureFeatures enforces
      GridImage input and guarantees grid structure is available via image.grid.info().
    - **GridOperation:** Base class for all grid-aware operations. GridMeasureFeatures is a specialized
      subclass combining measurement functionality with grid awareness.
    - **Pipeline integration:** Typical order: detect_colonies (ObjectDetector) → measure_features (GridMeasureFeatures) →
      analyze_growth (e.g., growth curve fitting per well).

    **Well-to-Colony Mapping Patterns**

    Common workflows for mapping measurements back to grid:

    - **Per-well aggregation:** Group colonies by well position; compute statistics per well (mean, std, count).
    - **Well-position filtering:** Exclude edge wells, measure only center wells, or apply position-dependent normalization.
    - **Row/column analysis:** Group by row or column; detect spatial patterns or plate gradients.
    - **Multi-well comparisons:** Compare growth metrics across wells for identification of outliers or resistance.

    **Output DataFrame Structure**

    Standard format returned by GridMeasureFeatures._operate():

    - **First column:** OBJECT.LABEL (int) - Matches image.objmap labels, one row per colony
    - **Measurement columns:** Your custom measurements (Area, Intensity, Circularity, etc.)
    - **Mapping to grid:** Use image.grid.info()['well_centers'] + label → pixel position → well assignment

    **Validation Checklist**

    Before shipping GridMeasureFeatures subclass:

    - Verify grid.info() call succeeds and returns expected keys
    - Test on GridImage with known grid (use load_synth_yeast_plate())
    - Ensure DataFrame output matches expected column structure (OBJECT.LABEL first, then measurements)
    - Check that measurements make sense (e.g., area > 0, intensity in expected range)
    - Validate on both small (96-well) and large (384-well) plates if possible

    **Known Implementations**

    Reference implementations in the PhenoTypic framework:

    - **GridMeasureSize:** Measures colony size and morphological features per detected object in GridImage.
      Extends GridMeasureFeatures with size-specific methods and grid-aware normalization.
    - **GridMeasureShape:** Measures colony shape/circularity/symmetry features. Demonstrates how grid structure
      can influence shape interpretation (e.g., elongation along grid rows).
    - **MeasureSize (MeasureFeatures):** Parent class implementation. GridMeasureSize extends this with grid awareness.

    **Testing GridMeasureFeatures Implementations**

    Best practices for testing new GridMeasureFeatures subclasses:

    - Use ``load_synth_yeast_plate()`` from phenotypic.data (creates GridImage with detected colonies).
    - Verify DataFrame output structure: first column OBJECT.LABEL, subsequent columns are measurements.
    - Test with edge cases: image with no colonies detected, single large colony, multiple overlapping colonies.
    - Check well-mapping: ensure measurements map correctly to grid positions via image.grid.info().
    - Validate row/column filtering logic if using position-aware filtering (center wells, edge exclusion).

    Examples:
        Grid-aware measurement of colony size per well:

        >>> from phenotypic import GridImage
        >>> from phenotypic.abc_ import GridMeasureFeatures
        >>> from phenotypic.tools_.constants_ import OBJECT
        >>> import pandas as pd
        >>> class MeasureWellOccupancy(GridMeasureFeatures):
        ...     '''Measure total area occupied in each well.'''
        ...
        ...     def _operate(self, image: GridImage) -> pd.DataFrame:
        ...         # Use grid accessor to calculate per-well metrics
        ...         area = self._calculate_sum(image.objmask[:], image.objmap[:])
        ...         well_info = image.grid.info()  # Get well assignments
        ...         # Combine area with well location
        ...         results = pd.DataFrame({
        ...             'WellArea': area,
        ...         })
        ...         results.insert(0, OBJECT.LABEL, image.objects.labels2series())
        ...         return results
        >>> # Usage
        >>> from phenotypic import Image
        >>> from phenotypic.detect import OtsuDetector
        >>> image = Image('plate.jpg')
        >>> image = OtsuDetector().operate(image)
        >>> grid_image = GridImage(image)
        >>> grid_image.detect_grid()  # Establish grid structure
        >>> measurer = MeasureWellOccupancy()
        >>> df = measurer.measure(grid_image)  # Returns grid-aware measurements

        Grid-aware filtering: measure only center wells:

        >>> from phenotypic.abc_ import GridMeasureFeatures
        >>> from phenotypic import GridImage
        >>> import pandas as pd
        >>> class MeasureCenterWells(GridMeasureFeatures):
        ...     '''Measure size only for colonies in center wells (exclude edge wells).'''
        ...
        ...     def _operate(self, image: GridImage) -> pd.DataFrame:
        ...         grid_info = image.grid.info()
        ...         nrows, ncols = grid_info['grid_shape']
        ...         # Get all measurements first
        ...         all_areas = self._calculate_sum(image.objmask[:], image.objmap[:])
        ...         labels = image.objects.labels2series()
        ...         # Filter: keep only colonies in center wells
        ...         center_mask = self._get_center_well_mask(grid_info, nrows, ncols)
        ...         # Return measurements for center-well colonies only
        ...         results = pd.DataFrame({'Area': all_areas[center_mask]})
        ...         results.insert(0, 'OBJECT.LABEL', labels[center_mask])
        ...         return results
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
