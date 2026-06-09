"""Grid-aligned object refinement - keeps only grid-aligned colonies from detection results.

Refines detected colonies by filtering to keep only the dominant object in each grid cell,
useful for removing off-grid artifacts and enforcing grid structure on detection results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import gc
import numpy as np

from phenotypic.abc_ import ObjectRefiner
from phenotypic.tools_.mixin import GridInferenceMixin
from phenotypic.tools_.funcs_ import validate_operation_integrity
from phenotypic.tools_.typing_ import TuneSpec


class GridAlignmentRefiner(GridInferenceMixin, ObjectRefiner):
    """Enforce grid structure by retaining the dominant object per grid cell.

    Infers or reads grid geometry from the image, partitions the detection
    results into grid cells, and keeps one object per cell according to the
    chosen selection strategy. Off-grid artifacts, dust, and spurious detections
    outside expected colony positions are removed. For ``GridImage`` inputs the
    grid layout is read directly; for plain ``Image`` inputs it is estimated
    from the detected mask.

    For an overview of grid refinement strategies, see
    :doc:`/explanation/refinement_strategies`.

    Best For:
        - High-throughput arrayed plates (96-well, 384-well, pinned cultures)
          where colonies should align with known well positions.
        - Post-detection cleanup when results contain off-grid debris or dust
          that survived threshold-based detection.
        - Experiments that require exactly one colony label per grid position
          before downstream measurement.

    Consider Also:
        - :class:`RefineBySineFit` when colony intensities are heterogeneous
          and sinusoidal template correlation improves grid estimation.
        - :class:`KeepSectionLargest` for a simpler largest-per-cell strategy
          on ``GridImage`` inputs without grid inference overhead.
        - :class:`ReduceSectionsByLine` for regression-based multi-detection
          reduction within grid cells.

    Args:
        smoothing_sigma: Standard deviation of the Gaussian kernel applied to
            row and column projection profiles before peak detection during grid
            inference. Higher values merge closely spaced peaks in noisy profiles;
            lower values preserve fine grid periodicity. Typical range: 0.5--5.0.
            Default: 2.0.
        min_peak_distance: Minimum pixel distance between adjacent detected grid
            peaks. ``None`` auto-estimates as half the expected inter-colony
            spacing. Default: None.
        peak_prominence: Minimum peak prominence for grid peak detection. ``None``
            auto-calculates as 10 % of the profile signal range. Default: None.
        edge_refinement: Refine inferred grid-cell edges using local intensity
            minima after initial peak detection. Improves cell boundary accuracy
            for unevenly lit or slightly rotated plates. Default: True.
        selection_mode: Strategy for selecting one object per grid cell.
            ``"dominant"`` keeps the largest object by pixel count;
            ``"centered"`` keeps the object whose centroid is closest to the
            cell centre; ``"regularized"`` fits a global regular-grid model
            then re-selects. Default: ``"dominant"``.
        split_merged: Attempt to split merged objects within a cell before
            applying the selection strategy. Useful when adjacent colonies are
            partially fused in the detection result. Default: False.

    Returns:
        Image: Input image with ``objmap`` filtered to one grid-aligned object
        per cell and ``objmask`` updated to match.

    Raises:
        ValueError: If grid inference fails or the image lacks detection results.

    See Also:
        :doc:`/how_to/notebooks/refine_noisy_boundaries` for grid-based cleanup
        workflows on real plate images.
    """

    smoothing_sigma: Annotated[float, TuneSpec(0.5, 5.0, log=True)] = 2.0
    min_peak_distance: Annotated[int | None, TuneSpec(5, 100, log=True)] = None
    peak_prominence: float | None = None
    edge_refinement: bool = True
    selection_mode: Literal["dominant", "centered", "regularized"] = "dominant"
    split_merged: bool = False

    @validate_operation_integrity("image.rgb", "image.gray", "image.detect_mat")
    def apply(self, image: Image, inplace: bool = False) -> Image:
        return super().apply(image=image, inplace=inplace)

    def _operate(self, image: Image) -> Image:
        """Refine detected objects to grid-aligned colonies.

        This method filters the object map to keep only the dominant object within each
        grid cell. Objects are reassigned new labels (1, 2, 3, ...) to ensure contiguous
        labeling after filtering.

        Returns:
            Image: Modified image with filtered objmap and updated objmask.
        """
        from phenotypic import GridImage

        # Get existing objmap
        objmap = image.objmap[:]

        # Determine grid edges (GridImage or infer)
        if isinstance(image, GridImage):
            row_edges = np.round(image.grid.get_row_edges()).astype(int)
            col_edges = np.round(image.grid.get_col_edges()).astype(int)
        else:
            objmask = image.objmask[:]
            # Use inherited mixin methods (call via class since _operate is static)
            nrows, ncols = GridAlignmentRefiner._infer_grid_shape(objmask)

            row_edges = GridAlignmentRefiner._estimate_edges(
                    objmask,
                    axis=0,
                    n_bins=nrows,
                    smoothing_sigma=self.smoothing_sigma,
                    min_peak_distance=self.min_peak_distance,
                    peak_prominence=self.peak_prominence,
            )
            col_edges = GridAlignmentRefiner._estimate_edges(
                    objmask,
                    axis=1,
                    n_bins=ncols,
                    smoothing_sigma=self.smoothing_sigma,
                    min_peak_distance=self.min_peak_distance,
                    peak_prominence=self.peak_prominence,
            )

            if self.edge_refinement:
                row_edges = GridAlignmentRefiner._refine_edges(objmask, row_edges,
                                                               axis=0)
                col_edges = GridAlignmentRefiner._refine_edges(objmask, col_edges,
                                                               axis=1)

        # Clip and unique edges
        row_edges = np.clip(np.unique(row_edges), 0, objmap.shape[0])
        col_edges = np.clip(np.unique(col_edges), 0, objmap.shape[1])

        # Assign objects per grid cell using selection strategy
        refined_map = self._assign_grid_objects(
                objmap, row_edges, col_edges, self.selection_mode, image._OBJMAP_DTYPE,
                intensity=image.detect_mat[:], split_merged=self.split_merged,
        )

        # Update image with refined map
        image.objmap[:] = refined_map
        image.objmap.relabel(connectivity=1)

        gc.collect()

        return image
