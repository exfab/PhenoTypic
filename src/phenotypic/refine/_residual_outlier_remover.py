from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._grid_image import GridImage

import numpy as np
from pydantic import AliasChoices, Field

from phenotypic.abc_ import GridObjectRefiner
from phenotypic.measure import MeasureGridLinRegStats
from phenotypic.tools_.measurement_info import GRID_LINREG_STATS, GRID


class ResidualOutlierRemover(GridObjectRefiner):
    """Remove objects with large positional residuals in noisy grid rows or columns.

    Fits linear-regression trends to colony centroids along each row and
    column, identifies rows/columns with high variability (coefficient of
    variance above threshold), and removes objects whose residual error
    exceeds an IQR-based cutoff within those noisy lines.

    Args:
        axis: Axis to analyze. ``None`` analyzes both rows and columns,
            ``0`` rows only, ``1`` columns only. Restricting the axis
            speeds processing and targets known problem directions.
            Default: None.
        stddev_multiplier: IQR-based cutoff multiplier for outlier removal.
            Lower values prune more aggressively; higher values are
            conservative. Typical range: 1.0--3.0. Default: 1.5.
        max_coeff_variance: Maximum coefficient of variance (std/mean)
            allowed before a row/column is considered noisy and eligible
            for outlier pruning. Typical range: 1--5. Default: 1.

    Returns:
        Image: Input image with ``objmap`` and ``objmask`` updated to exclude
        positional outliers from noisy grid rows/columns.

    Best For:
        - Cleaning rows or columns with off-grid detections from condensation,
          glare, or debris before measuring growth.
        - Stabilizing grid registration when a subset of positions is noisy.
        - Plates where most grid lines are well-aligned but a few contain
          spurious artifacts.

    Consider Also:
        - :class:`ReduceMultipleGridObjects` for reducing multi-detections to
          one per cell rather than pruning outliers within rows/columns.
        - :class:`GridAlignmentRefiner` for full grid-aware filtering using
          dominant-object-per-cell selection.
        - :class:`GridOversizedObjectRemover` when the problem is oversized
          detections rather than positional outliers.

    See Also:
        :doc:`/how_to/notebooks/refine_noisy_boundaries` for grid-based
        outlier removal workflows.
        :doc:`/explanation/refinement_strategies` for a comparison of
        grid refinement approaches.
    """

    #: Axis selection — ``None`` for both, ``0`` for row, ``1`` for column.
    axis: int | None = None
    #: Robust residual cutoff multiplier. The public constructor keyword is
    #: the legacy ``stddev_multiplier``; the attribute the algorithm reads is
    #: ``cutoff_multiplier`` (the pre-migration ``__init__`` renamed the
    #: parameter on assignment). ``AliasChoices`` accepts both the legacy
    #: keyword and the field name so existing call sites and JSON
    #: round-trips keep working.
    cutoff_multiplier: float = Field(
            default=1.5,
            validation_alias=AliasChoices("stddev_multiplier", "cutoff_multiplier"),
    )
    max_coeff_variance: int = 1

    def _operate(self, image: GridImage) -> GridImage:
        """Identify and remove residual outliers per noisy row/column.

        Args:
            image (GridImage): Grid image with object map and grid metadata.

        Returns:
            GridImage: Modified grid image with outlier objects removed.

        Raises:
            ValueError: If parameters are misconfigured in a way that prevents
                computation (propagated from measurement utilities).
        """
        # Generate cached version of grid_info
        linreg_stat_extractor = MeasureGridLinRegStats()
        grid_info = linreg_stat_extractor.measure(image)

        # Create container to hold the id of objects to be removed
        outlier_obj_ids = []

        # Row-wise residual outlier discovery
        if self.axis is None or self.axis == 0:
            # Calculate the coefficient of variance (std/mean)
            #   Collect the standard deviation
            row_variance = grid_info.groupby(str(GRID.ROW_NUM))[
                str(GRID_LINREG_STATS.RESIDUAL_ERR)
            ].std()

            #   Divide standard deviation by mean
            row_variance = (
                    row_variance
                    / grid_info.groupby(str(GRID.ROW_NUM))[
                        str(GRID_LINREG_STATS.RESIDUAL_ERR)
                    ].mean()
            )

            over_limit_row_variance = row_variance.loc[
                row_variance > self.max_coeff_variance
                ]

            # Collect outlier objects in the nrows with a variance over the maximum
            for row_idx in over_limit_row_variance.index:
                row_err = grid_info.loc[
                    grid_info.loc[:, str(GRID.ROW_NUM)] == row_idx,
                    str(GRID_LINREG_STATS.RESIDUAL_ERR),
                ]
                row_err_mean = row_err.mean()
                row_q3, row_q1 = row_err.quantile([0.75, 0.25])
                row_iqr = row_q3 - row_q1

                # row_stddev = row_err.std()
                # upper_row_cutoff = row_err_mean + row_stddev * self.cutoff_multiplier

                upper_row_cutoff = row_err_mean + row_iqr * self.cutoff_multiplier
                outlier_obj_ids += row_err.loc[
                    row_err >= upper_row_cutoff
                    ].index.tolist()

        # Column-wise residual outlier discovery
        if self.axis is None or self.axis == 1:
            # Calculate the coefficient of variance (std/mean)
            #   Collect the standard deviation
            col_variance = grid_info.groupby(str(GRID.COL_NUM))[
                str(GRID_LINREG_STATS.RESIDUAL_ERR)
            ].std()

            #   Divide standard deviation by mean
            col_variance = (
                    col_variance
                    / grid_info.groupby(str(GRID.COL_NUM))[
                        str(GRID_LINREG_STATS.RESIDUAL_ERR)
                    ].mean()
            )

            over_limit_col_variance = col_variance.loc[
                col_variance > self.max_coeff_variance
                ]

            # Collect outlier objects in the columns with a variance over the maximum
            for col_idx in over_limit_col_variance.index:
                col_err = grid_info.loc[
                    grid_info.loc[:, str(GRID.COL_NUM)] == col_idx,
                    str(GRID_LINREG_STATS.RESIDUAL_ERR),
                ]
                col_err_mean = col_err.mean()
                col_q3, col_q1 = col_err.quantile([0.75, 0.25])
                col_iqr = col_q3 - col_q1
                # col_stddev = col_err.std()

                upper_col_cutoff = col_err_mean + col_iqr * self.cutoff_multiplier
                outlier_obj_ids += col_err.loc[
                    col_err >= upper_col_cutoff
                    ].index.tolist()

        # Remove objects from obj map
        image.objmap[np.isin(image.objmap[:], outlier_obj_ids)] = 0

        return image
