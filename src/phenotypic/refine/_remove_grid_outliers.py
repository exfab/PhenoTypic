from __future__ import annotations
from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from phenotypic._core._grid_image import GridImage

import numpy as np
from pydantic import AliasChoices, Field

from phenotypic.abc_ import GridObjectRefiner
from phenotypic.measure import MeasureGridLinRegStats
from phenotypic.schema import GRID_LINREG_STATS, GRID
from phenotypic.sdk_.typing_ import TuneSpec


class RemoveGridOutliers(GridObjectRefiner):
    """Remove positional outliers from noisy grid rows or columns using IQR-based residual pruning.

    Fits linear-regression trends to colony centroids along each row and
    column, identifies lines whose residual coefficient of variance exceeds
    ``max_coeff_variance``, then removes objects whose residual error exceeds
    the mean plus ``cutoff_multiplier`` times the IQR within those noisy
    lines. Rows and columns with low variance are left untouched.

    Requires a ``GridImage`` with grid metadata already populated.

    For a comparison of grid refinement approaches, see
    :doc:`/explanation/refinement_strategies`.

    Best For:
        - Plates where most rows and columns are well-aligned but a few
          contain off-grid detections from condensation, glare, or debris.
        - Stabilizing grid registration before size or intensity measurement
          when a subset of row or column positions is noisy.
        - Batch runs where individual plates occasionally have edge rows
          with aberrant detections that should not propagate to measurements.

    Consider Also:
        - :class:`ReduceSectionsByLine` when the goal is reducing each cell
          to one detection rather than pruning outliers within lines.
        - :class:`GridAlignmentRefiner` for full grid-aware filtering using
          dominant-object-per-cell selection without variance-based gating.
        - :class:`GridOversizedObjectRemover` when the problem is objects
          spanning multiple grid sections rather than positional outliers.

    Args:
        axis: Grid axis to analyze. ``None`` processes both rows and
            columns; ``0`` restricts to rows; ``1`` restricts to columns.
            Restricting the axis speeds processing when the problem
            direction is known. Default: ``None``.
        cutoff_multiplier: IQR-based residual cutoff multiplier. Objects
            whose residual exceeds ``mean + cutoff_multiplier × IQR``
            within a noisy line are removed. Lower values prune more
            aggressively; higher values are conservative. Also accepted
            as ``stddev_multiplier`` for backwards compatibility. Typical
            range: 1.0--3.0. Default: 1.5.
        max_coeff_variance: Maximum coefficient of variance (std / mean
            of residuals) allowed for a row or column before it is
            considered noisy and subjected to outlier pruning. Lines below
            this threshold are left untouched. Typical range: 1--5.
            Default: 1.

    Returns:
        Image: Input image with ``objmap`` and ``objmask`` updated to
        exclude positional outliers from noisy grid rows and columns.
        ``rgb``, ``gray``, and ``detect_mat`` are unchanged.

    See Also:
        :doc:`/how_to/notebooks/refine_noisy_boundaries` for grid-based
        outlier removal workflows.
        :doc:`/explanation/refinement_strategies` for a comparison of
        grid refinement approaches.
    """

    #: Axis selection — ``None`` for both, ``0`` for row, ``1`` for column.
    axis: Annotated[int | None, TuneSpec(tunable=False)] = None
    #: Robust residual cutoff multiplier. The public constructor keyword is
    #: the legacy ``stddev_multiplier``; the attribute the algorithm reads is
    #: ``cutoff_multiplier`` (the pre-migration ``__init__`` renamed the
    #: parameter on assignment). ``AliasChoices`` accepts both the legacy
    #: keyword and the field name so existing call sites and JSON
    #: round-trips keep working.
    cutoff_multiplier: Annotated[float, TuneSpec(1.0, 3.0)] = Field(
            default=1.5,
            validation_alias=AliasChoices("stddev_multiplier", "cutoff_multiplier"),
            description=(
                "IQR-based cutoff multiplier for outlier removal. Lower "
                "values prune more aggressively; higher values are "
                "conservative. Typical range: 1.0--3.0. Default: 1.5."
            ),
    )
    max_coeff_variance: Annotated[int, TuneSpec(1, 5)] = 1

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
