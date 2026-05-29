"""Grid linear regression statistics and residual errors."""

from ._measurement_info import MeasurementInfo


class GRID_LINREG_STATS(MeasurementInfo):
    """Evaluate grid alignment quality using row-wise and column-wise linear regression.

    Fit linear regressions to colony centroid positions along each row
    and column of the grid, then compute per-colony residual error
    (Euclidean distance between observed and predicted centroid). High
    residual errors flag off-grid growth, misdetections, or plate
    warping.
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
