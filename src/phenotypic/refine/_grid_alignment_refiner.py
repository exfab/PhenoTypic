"""Grid-aligned object refinement - keeps only grid-aligned colonies from detection results.

Refines detected colonies by filtering to keep only the dominant object in each grid cell,
useful for removing off-grid artifacts and enforcing grid structure on detection results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image

import gc
import numpy as np

from phenotypic.abc_ import ObjectRefiner
from phenotypic.tools_._grid_inference_mixin import GridInferenceMixin
from phenotypic.tools_.funcs_ import validate_operation_integrity


class GridAlignmentRefiner(GridInferenceMixin, ObjectRefiner):
    """Refines detected objects by keeping only grid-aligned colonies using dominant-object-per-cell filtering.

    GridAlignmentRefiner filters detection results to retain only one dominant object per
    grid cell, removing off-grid artifacts and enforcing regular grid structure on colony
    detections. This is particularly useful for high-throughput arrayed cultures where
    colonies should align with known well positions.

    Args:
        smoothing_sigma: Gaussian smoothing standard deviation for row/column intensity
            profiles during grid inference. Default 2.0. Higher values smooth noise but
            may merge adjacent peaks.
        min_peak_distance: Minimum pixel distance between detected grid peaks. If None
            (default), automatically estimated as half the expected colony spacing.
        peak_prominence: Minimum prominence threshold for peak detection. If None (default),
            auto-calculated as 10% of signal range.
        edge_refinement: If True (default), refine grid edges using local intensity minima
            to improve grid alignment accuracy.

    Returns:
        Image: Input image with filtered objmap containing only grid-aligned objects.
            objmask is automatically updated to match refined objmap. All image data
            (rgb, gray, enh_gray) remain unchanged.

    Raises:
        ValueError: If grid inference fails or image lacks detection results (no objmap).

    **Use cases**

    - **Gridded plate images:** Remove off-grid noise and dust for accurate well-based
      phenotyping on 96-well, 384-well, or pinned culture formats.
    - **Post-detection cleanup:** Apply after ObjectDetector when detections contain
      spurious off-grid objects or artifacts.
    - **Explicit grid enforcement:** Use with GridImage to snap detections to known
      well positions when exact grid coordinates are available.
    - **Multi-detector workflows:** Combine outputs from different detectors and keep
      grid-aligned objects from highest-confidence detection.

    **Limitations**

    - Assumes regular grid geometry; fails on irregular colony spacing or missing positions.
    - Grid inference on regular Image is less accurate than explicit GridImage specification.
    - Requires colonies to cluster within grid cells; fails if colonies straddle boundaries.
    - Best for yeast-like circular colonies; less suitable for filamentous or irregular
      morphologies that may not align cleanly with grid.
    - Removes off-grid objects entirely; use GridObjectRefiner for gentler boundary handling.

    **Parameter effects on grid detection**

    - **smoothing_sigma:** Higher values improve robustness to noise but may merge
      adjacent peaks. Set to 0 to disable smoothing (faster, less robust).
    - **edge_refinement:** When True, places grid edges at valleys between colonies
      rather than fixed positions, improving accuracy for uneven plate lighting.
    - **min_peak_distance, peak_prominence:** Lower values detect more peaks (find more
      grid positions); higher values are more selective. Auto-tuning usually works well.

    Examples:
        Basic usage with GridImage and explicit grid dimensions:

        >>> from phenotypic import GridImage
        >>> from phenotypic.detect import OtsuDetector
        >>> from phenotypic.refine import GridAlignmentRefiner
        >>> from phenotypic.data import load_synth_plate
        >>>
        >>> # Load gridded plate image
        >>> grid_image = load_synth_plate()  # Returns GridImage with 8x12 grid
        >>> detector = OtsuDetector()
        >>> detected = detector.apply(grid_image)
        >>>
        >>> # Refine to keep only grid-aligned objects
        >>> refiner = GridAlignmentRefiner()
        >>> refined = refiner.apply(detected)
        >>>
        >>> # Check results
        >>> print(f"Before: {detected.objmap[:].max()} objects")
        >>> print(f"After:  {refined.objmap[:].max()} objects (grid-aligned)")

        Integration into full processing pipeline with grid inference:

        >>> from phenotypic import Image, ImagePipeline
        >>> from phenotypic.enhance import GaussianBlur, CLAHE
        >>> from phenotypic.detect import RoundPeaksDetector
        >>> from phenotypic.refine import GridAlignmentRefiner
        >>> from phenotypic.measure import MeasureShape
        >>>
        >>> # Build pipeline with detection and refinement
        >>> pipeline = ImagePipeline([
        ...     GaussianBlur(sigma=1.5),
        ...     CLAHE(clip_limit=2.0),
        ...     RoundPeaksDetector(smoothing_sigma=2.0),
        ...     GridAlignmentRefiner(edge_refinement=True),  # Clean up detection
        ...     MeasureShape()
        ... ])
        >>>
        >>> # Process image (no explicit grid needed)
        >>> image = Image.imread("noisy_plate.jpg")
        >>> result = pipeline.apply(image)
        >>>
        >>> print(f"Cleaned colonies: {len(result.objects)}")
    """

    def __init__(
            self,
            smoothing_sigma: float = 2.0,
            min_peak_distance: int | None = None,
            peak_prominence: float | None = None,
            edge_refinement: bool = True,
    ):
        """Initialize GridAlignmentRefiner with grid inference parameters.

        Args:
            smoothing_sigma: Gaussian smoothing sigma for intensity profiles.
            min_peak_distance: Minimum distance between grid peaks.
            peak_prominence: Minimum prominence for peak detection.
            edge_refinement: Enable edge refinement via local intensity minima.
        """
        super().__init__()
        self.smoothing_sigma = smoothing_sigma
        self.min_peak_distance = min_peak_distance
        self.peak_prominence = peak_prominence
        self.edge_refinement = edge_refinement

    @validate_operation_integrity("image.rgb", "image.gray", "image.enh_gray")
    def apply(self, image: Image, inplace: bool = False) -> Image:
        return super().apply(image=image, inplace=inplace)

    @staticmethod
    def _operate(
            image: Image,
            smoothing_sigma: float,
            min_peak_distance: int | None,
            peak_prominence: float | None,
            edge_refinement: bool,
    ) -> Image:
        """Refine detected objects to grid-aligned colonies.

        This method filters the object map to keep only the dominant object within each
        grid cell. Objects are reassigned new labels (1, 2, 3, ...) to ensure contiguous
        labeling after filtering.

        Args:
            image: Image object with existing objmask and objmap from detection.
            smoothing_sigma: Gaussian smoothing for grid inference.
            min_peak_distance: Minimum distance between peaks.
            peak_prominence: Minimum prominence for peaks.
            edge_refinement: Whether to refine edges.

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
                    smoothing_sigma=smoothing_sigma,
                    min_peak_distance=min_peak_distance,
                    peak_prominence=peak_prominence,
            )
            col_edges = GridAlignmentRefiner._estimate_edges(
                    objmask,
                    axis=1,
                    n_bins=ncols,
                    smoothing_sigma=smoothing_sigma,
                    min_peak_distance=min_peak_distance,
                    peak_prominence=peak_prominence,
            )

            if edge_refinement:
                row_edges = GridAlignmentRefiner._refine_edges(objmask, row_edges,
                                                               axis=0)
                col_edges = GridAlignmentRefiner._refine_edges(objmask, col_edges,
                                                               axis=1)

        # Clip and unique edges
        row_edges = np.clip(np.unique(row_edges), 0, objmap.shape[0])
        col_edges = np.clip(np.unique(col_edges), 0, objmap.shape[1])

        # Assign dominant object per grid cell
        refined_map = np.zeros_like(objmap, dtype=image._OBJMAP_DTYPE)
        label_counter = 1

        for r in range(len(row_edges) - 1):
            r0, r1 = row_edges[r], row_edges[r + 1]
            for c in range(len(col_edges) - 1):
                c0, c1 = col_edges[c], col_edges[c + 1]

                # Get region in this grid cell
                region = objmap[r0:r1, c0:c1]

                if region.size == 0:
                    continue

                # Find all unique labels (except background 0)
                uniq, counts = np.unique(region, return_counts=True)
                valid = uniq != 0

                if not np.any(valid):
                    continue

                uniq = uniq[valid]
                counts = counts[valid]

                # Keep dominant label (most pixels in cell)
                dominant_label = uniq[np.argmax(counts)]
                mask = region == dominant_label

                if np.any(mask):
                    refined_map[r0:r1, c0:c1][mask] = label_counter
                    label_counter += 1

        # Update image with refined map
        image.objmap[:] = refined_map
        image.objmap.relabel(connectivity=1)

        gc.collect()

        return image
