from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import GridImage
from phenotypic.abc_ import GridMeasureFeatures, MeasurementInfo

import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
from phenotypic.tools.constants_ import OBJECT, BBOX, GRID


class GRID_SPREAD(MeasurementInfo):
    """Grid section spatial spread measurements.

    Provides measurements for evaluating spatial distribution of colonies within
    grid sections of arrayed microbial assays.
    """

    @classmethod
    def category(cls):
        return "GridSpread"

    OBJECT_SPREAD = (
        "ObjectSpread",
        "Sum of squared pairwise Euclidean distances between all unique colony pairs within a grid section. Quantifies spatial dispersion of colonies in a grid cell. Higher values indicate greater spread from the section center, suggesting over-segmentation, multi-detections, or colonies growing beyond expected boundaries. Used to identify problematic grid sections requiring refinement or quality review.",
    )


class MeasureGridSpread(GridMeasureFeatures):
    """Quantify spatial distribution of colonies within grid sections of arrayed assays.

    This class measures colony clustering and spread within each grid cell (well) of a high-throughput
    microbial phenotyping plate. It computes the sum of squared pairwise distances between all colony
    pairs in each section, revealing whether multiple colonies are dispersed within a well or tightly
    clustered near the center.

    **Intuition:** In ideal arrayed assays, each well contains a single localized colony at the expected
    position. High ObjectSpread values indicate multiple colonies, fragmented growth, or spread beyond
    the well boundary. This metric helps identify sections with detection ambiguity or atypical growth
    patterns that may compromise phenotypic measurements.

    **Use cases (agar plates):**
    - Detect over-segmentation: Multiple detected objects in a single well instead of one cohesive colony.
    - Identify spreading or invasive growth: Colonies extending far beyond their designated grid position
      into adjacent areas.
    - Flag wells with questionable data quality for manual review or exclusion from downstream analysis.
    - Assess plate quality: Systematic high spread values across the plate suggest uneven agar surface,
      condensation issues, or poor inoculation technique.
    - Prioritize sections for refinement operations (e.g., object merging or filtering).

    **Caveats:**
    - ObjectSpread depends on colony density and size; small plates with tight spacing have inherently
      different baselines than larger assays.
    - Spread values are not normalized by well area; compare only within the same plate type and grid.
    - Touching or overlapping colonies may register as a single large object (low spread) or two smaller
      objects (high spread) depending on detection algorithm performance. Use in conjunction with object
      count and boundary metrics for robust quality assessment.
    - The metric is sensitive to very small or very large colonies; outliers in position or size can
      disproportionately inflate spread values.

    Returns:
        pd.DataFrame: Section-level statistics with columns:
            - Section numbers (index from grid).
            - Count: Number of colonies detected in each section.
            - ObjectSpread: Sum of squared pairwise Euclidean distances between colonies.
                Sorted in descending order by ObjectSpread.

    Examples:
        .. dropdown:: Measure colony spread across a plate

            .. code-block:: python

                from phenotypic import GridImage
                from phenotypic.detect import OtsuDetector
                from phenotypic.measure import MeasureGridSpread

                # Load a plate with grid
                grid_image = GridImage.from_image_path("plate_384well.jpg", grid_shape=(16, 24))

                # Detect colonies
                detector = OtsuDetector()
                grid_image = detector.operate(grid_image)

                # Measure spread per well
                spreader = MeasureGridSpread()
                spread_results = spreader.operate(grid_image)

                # Find wells with high spread (potential over-segmentation)
                high_spread = spread_results.nlargest(10, 'GridSpread_ObjectSpread')
                print(f"Top 10 problematic wells:")
                print(high_spread)

        .. dropdown:: Identify over-segmented wells

            .. code-block:: python

                # Flag wells with both multiple detections AND high spread
                multi_obj = spread_results[spread_results['count'] > 1]
                high_spread_multi = multi_obj[
                    multi_obj['GridSpread_ObjectSpread'] > spread_results['GridSpread_ObjectSpread'].quantile(0.75)
                ]
                print(f"Wells needing refinement: {list(high_spread_multi.index)}")
    """

    @staticmethod
    def _operate(image: GridImage) -> pd.DataFrame:
        gs_table = image.grid.info()
        gs_counts = pd.DataFrame(gs_table.loc[:, str(GRID.SECTION_NUM)].value_counts())

        obj_spread = []
        for gs_bindex in gs_counts.index:
            curr_gs_subtable = gs_table.loc[
                gs_table.loc[:, str(GRID.SECTION_NUM)] == gs_bindex, :
            ]

            x_vector = curr_gs_subtable.loc[:, str(BBOX.CENTER_CC)]
            y_vector = curr_gs_subtable.loc[:, str(BBOX.CENTER_RR)]
            obj_vector = np.array(list(zip(x_vector, y_vector)))
            gs_distance_matrix = distance_matrix(x=obj_vector, y=obj_vector, p=2)

            obj_spread.append(np.sum(np.unique(gs_distance_matrix) ** 2))
        gs_counts.insert(loc=1, column=str(GRID_SPREAD.OBJECT_SPREAD), value=pd.Series(obj_spread))
        gs_counts.sort_values(by=str(GRID_SPREAD.OBJECT_SPREAD), ascending=False, inplace=True)
        return gs_counts


MeasureGridSpread.__doc__ = GRID_SPREAD.append_rst_to_doc(MeasureGridSpread)
