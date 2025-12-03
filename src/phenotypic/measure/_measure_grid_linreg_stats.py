from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import GridImage

import pandas as pd
from scipy.spatial.distance import euclidean

from phenotypic.abc_ import GridMeasureFeatures, MeasurementInfo
from phenotypic.tools.constants_ import OBJECT, BBOX, GRID


class GRID_LINREG_STATS(MeasurementInfo):
    """Grid linear regression statistics and residual errors.

    Provides measurements for evaluating grid alignment quality and detecting off-grid
    colonies in arrayed microbial assays.
    """

    @classmethod
    def category(cls):
        return "GridLinReg"

    ROW_LINREG_M = (
        "RowM",
        "Slope of row-wise linear regression fit across column positions. Measures systematic drift in row alignment. Values near 0 indicate horizontal rows; non-zero values suggest rotational misalignment or systematic row curvature across the plate.",
    )
    ROW_LINREG_B = (
        "RowB",
        "Intercept of row-wise linear regression fit. Represents the expected row coordinate when column position is 0. Combined with slope, defines the expected row trend line for quality assessment and position prediction.",
    )
    COL_LINREG_M = (
        "ColM",
        "Slope of column-wise linear regression fit across row positions. Measures systematic drift in column alignment. Values near 0 indicate vertical columns; non-zero values suggest rotational misalignment or systematic column curvature across the plate.",
    )
    COL_LINREG_B = (
        "ColB",
        "Intercept of column-wise linear regression fit. Represents the expected column coordinate when row position is 0. Combined with slope, defines the expected column trend line for quality assessment and position prediction.",
    )
    PRED_RR = (
        "PredRR",
        "Predicted row coordinate from column-wise linear regression. Uses the column position and column regression parameters (ColM, ColB) to estimate where the row coordinate should be if the grid were perfectly aligned. Used for calculating residual errors and detecting misaligned colonies.",
    )
    PRED_CC = (
        "PredCC",
        "Predicted column coordinate from row-wise linear regression. Uses the row position and row regression parameters (RowM, RowB) to estimate where the column coordinate should be if the grid were perfectly aligned. Used for calculating residual errors and detecting misaligned colonies.",
    )
    RESIDUAL_ERR = (
        "ResidualError",
        "Euclidean distance between the actual colony centroid and the predicted position from linear regression. Quantifies how far each colony deviates from the expected grid pattern. High values indicate misdetections, off-grid growth, or local plate warping. Used by refinement operations to filter outliers and select the most plausible colony per grid cell.",
    )


class MeasureGridLinRegStats(GridMeasureFeatures):
    """Evaluate grid alignment quality for arrayed microbial colonies using linear regression.

    This class measures how well detected colonies align to expected grid positions in arrayed
    phenotyping assays. It computes row-wise and column-wise linear regression fits across the
    grid, then quantifies each colony's deviation from the predicted position using residual error.

    **Intuition:** In high-throughput microbial phenotyping, 96-well and 384-well plates have
    expected regular grid patterns. Deviations indicate off-grid growth, misdetections, plate
    warping, or imaging artifacts. High residual errors flag problematic detections that may
    need refinement or filtering before downstream analysis.

    **Use cases (agar plates):**
    - Identify colonies that grew outside their designated grid position (e.g., spreading into
      adjacent wells on a plate).
    - Detect systematic alignment issues (e.g., rotation, shear) across the plate to validate
      grid detection quality.
    - Filter or weight colonies by detection confidence based on deviation from expected position
      (used by refinement operations to select most plausible colony per grid cell).
    - Quantify plate warping or uneven agar surface by analyzing residual error patterns across rows/columns.

    **Caveats:**
    - Regression assumes a linear relationship; severely warped plates may not fit the linear model well.
    - Residual errors are sensitive to grid detection accuracy; incorrect grid estimates propagate to
      inflated residuals for correctly detected colonies.
    - Threshold selection for outlier filtering is application-dependent; conservatively use the 95th
      percentile residual error within a plate for robust quality control.

    Args:
        section_num (Optional[int], optional): Grid section number to restrict measurements to.
            If None, measurements are computed across the entire grid. Defaults to None.

    Attributes:
        section_num (Optional[int]): Section number for targeted grid region analysis.

    Returns:
        pd.DataFrame: Measurement results indexed by object label. Includes per-object metrics:
            - RowM, RowB: Row regression slope and intercept (1 value per row).
            - ColM, ColB: Column regression slope and intercept (1 value per column).
            - PredRR, PredCC: Predicted row and column centroids from regression.
            - ResidualError: Euclidean distance between actual and predicted centroid.

    Examples:
        .. dropdown:: Measure grid alignment for an arrayed plate

            .. code-block:: python

                from phenotypic import GridImage
                from phenotypic.detect import OtsuDetector
                from phenotypic.measure import MeasureGridLinRegStats

                # Load a plate image with grid information
                grid_image = GridImage.from_image_path("plate_96well.jpg", grid_shape=(8, 12))

                # Detect colonies
                detector = OtsuDetector()
                grid_image = detector.operate(grid_image)

                # Measure grid alignment quality
                measurer = MeasureGridLinRegStats()
                results = measurer.operate(grid_image)

                # Identify off-grid colonies
                outliers = results[results['GridLinReg_ResidualError'] > 10.0]
                print(f"Found {len(outliers)} colonies with high misalignment")

        .. dropdown:: Measure alignment within a single section/well

            .. code-block:: python

                # Measure only colonies in section 5 (useful for troubleshooting
                # a specific well or region)
                measurer = MeasureGridLinRegStats(section_num=5)
                section_results = measurer.operate(grid_image)
                # Results contain grid stats and residual errors only for section 5
    """

    def __init__(self, section_num: Optional[int] = None):
        super().__init__()
        self.section_num = section_num

    def _operate(self, image: GridImage) -> pd.DataFrame:
        # Collect the relevant section info. If no section was specified perform calculation on the entire grid info table.
        if self.section_num is None:
            section_info = image.grid.info().reset_index(drop=False)
        else:
            grid_info = image.grid.info().reset_index(drop=False)
            section_info = grid_info.loc[
                grid_info.loc[:, str(GRID.SECTION_NUM)] == self.section_num, :
            ]

        # Get the current row-wise linreg info
        row_m, row_b = image.grid.get_centroid_alignment_info(axis=0)

        # Convert arrays to dataframe for join operation
        row_linreg_info = pd.DataFrame(
            data={
                str(GRID_LINREG_STATS.ROW_LINREG_M): row_m,
                str(GRID_LINREG_STATS.ROW_LINREG_B): row_b,
            },
            index=pd.Index(data=range(image.grid.nrows), name=str(GRID.ROW_NUM)),
        )

        section_info = pd.merge(
            left=section_info,
            right=row_linreg_info,
            left_on=str(GRID.ROW_NUM),
            right_on=str(GRID.ROW_NUM),
        )

        # NOTE: Row linear regression(CC) -> pred RR
        section_info.loc[:, str(GRID_LINREG_STATS.PRED_RR)] = (
            section_info.loc[:, str(BBOX.CENTER_CC)]
            * section_info.loc[:, str(GRID_LINREG_STATS.ROW_LINREG_M)]
            + section_info.loc[:, str(GRID_LINREG_STATS.ROW_LINREG_B)]
        )

        # Get the current column linreg info
        col_m, col_b = image.grid.get_centroid_alignment_info(axis=1)

        # convert array to dataframe for join operation
        col_linreg_info = pd.DataFrame(
            data={
                str(GRID_LINREG_STATS.COL_LINREG_M): col_m,
                str(GRID_LINREG_STATS.COL_LINREG_B): col_b,
            },
            index=pd.Index(data=range(image.grid.ncols), name=str(GRID.COL_NUM)),
        )

        section_info = pd.merge(
            left=section_info,
            right=col_linreg_info,
            left_on=str(GRID.COL_NUM),
            right_on=str(GRID.COL_NUM),
        )

        # NOTE: Col linear regression(RR) -> pred CC
        section_info.loc[:, str(GRID_LINREG_STATS.PRED_CC)] = (
            section_info.loc[:, str(BBOX.CENTER_RR)]
            * section_info.loc[:, str(GRID_LINREG_STATS.COL_LINREG_M)]
            + section_info.loc[:, str(GRID_LINREG_STATS.COL_LINREG_B)]
        )

        # Calculate the distance each object is from it's predicted center. This is the residual error
        section_info.loc[:, str(GRID_LINREG_STATS.RESIDUAL_ERR)] = (
            section_info.apply(
                lambda row: euclidean(
                    u=[row[str(BBOX.CENTER_CC)], row[str(BBOX.CENTER_RR)]],
                    v=[
                        row[str(GRID_LINREG_STATS.PRED_CC)],
                        row[str(GRID_LINREG_STATS.PRED_RR)],
                    ],
                ),
                axis=1,
            )
        )

        return section_info.set_index(OBJECT.LABEL)


MeasureGridLinRegStats.__doc__ = GRID_LINREG_STATS.append_rst_to_doc(MeasureGridLinRegStats)
