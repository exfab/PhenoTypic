from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import gc
from typing import Annotated, Literal, Optional

import numpy as np
import scipy.ndimage as ndimage
from pydantic import Field

from phenotypic.abc_ import ObjectDetector
from phenotypic.sdk_.mixin import GridInferenceMixin
from phenotypic.sdk_.typing_ import TuneSpec
import skimage.morphology as morphology
from phenotypic.detect._grid_peak_common import (
    _round_odd as _grid_peak_round_odd,
    grid_peak_threshold_mask,
)


class RoundPeaksDetector(GridInferenceMixin, ObjectDetector):
    """Detect round colonies on gridded plates by row/column peak analysis (gitter algorithm).

    Threshold the plate image, project row and column intensity sums to
    detect periodic peaks, infer grid edges from peak positions, and assign
    one colony per grid cell. This implements the gitter algorithm optimised
    for pinned microbial culture plates with circular colonies arranged in
    regular arrays (96, 384, 1536 formats). For a full comparison see
    :doc:`/explanation/detection_strategies_compared`.

    Best For:
        - Pinned yeast or bacterial plates with colonies arranged in a
          regular rectangular grid.
        - Plates where colony shape is approximately circular and colonies
          are well-separated or only mildly touching.
        - High-throughput screens where automatic grid inference eliminates
          the need for manual grid specification.
        - Workflows that require one-colony-per-cell assignment for
          downstream quantification.

    Consider Also:
        - :class:`SinePeakDetector` when colony sizes are heterogeneous or
          some wells are empty, where rank cross-correlation locates the
          grid more robustly than direct peak finding.
        - :class:`WatershedDetector` when colonies are densely packed and
          touching but not arranged on a regular grid.
        - :class:`FilamentousFungiDetector` when colonies exhibit spreading,
          filamentous growth that violates the round-colony assumption.
        - :class:`ManualGridPointDetector` when colony positions are known a
          priori from robotic spotting coordinates.

    Args:
        thresh_method: Thresholding method for binary mask creation.
            Accepted values: ``'otsu'`` (default), ``'mean'``, ``'local'``,
            ``'triangle'``, ``'minimum'``, ``'isodata'``, ``'li'``.
            ``'otsu'`` works well for most standardised imaging setups;
            ``'local'`` adapts to spatial illumination gradients. Default:
            ``'otsu'``.
        subtract_background: Apply white tophat transform to remove uneven
            illumination before thresholding. Disable on plates with
            uniform lighting to save compute time. Default: True.
        remove_noise: Apply morphological opening to remove small noise
            artefacts from the binary mask. Default: True.
        footprint_width: Width in pixels for the background subtraction
            kernel. When a GridImage is provided, an adaptive kernel sized
            to 1.5x colony spacing is used instead, making this a fallback
            for plain Image inputs. Typical range: 4--20. Default: 6.
        noise_radius: Radius of the diamond structuring element for
            morphological noise removal. Increase for larger noise
            artefacts. Typical range: 1--3. Default: 1.
        smoothing_sigma: Gaussian sigma for smoothing row/column intensity
            profiles before peak detection. Higher values suppress noise but
            may merge adjacent colony peaks. Set to 0 to disable smoothing.
            Typical range: 0.0--5.0. Default: 2.0.
        min_peak_distance: Minimum pixel distance between detected peaks.
            When ``None``, automatically estimated from grid dimensions.
            Default: None.
        peak_prominence: Minimum prominence threshold for peak detection.
            When ``None``, auto-calculated as 0.1 × signal range. Higher
            values are more selective. Default: None.
        edge_refinement: Refine grid edges using weighted local intensity
            profiles for improved accuracy. Default: True.
        selection_mode: Strategy for choosing one object per grid cell.
            ``"dominant"`` keeps the largest object by pixel count.
            ``"centered"`` keeps the object whose centroid is closest to
            the cell centre. ``"regularized"`` fits a global regular-grid
            model from median row/column centroids, then re-selects per
            cell — best for pinned arrays. Default: ``"dominant"``.
        split_merged: Pre-split merged colonies that span multiple grid
            cells using EDT watershed before grid assignment. Set to False
            when colonies are well-separated and splitting is unnecessary.
            Default: True.

    Returns:
        Image: Input image with ``objmask`` set to a binary colony mask
        and ``objmap`` set to a labelled colony map with one label per
        grid cell.

    Raises:
        ValueError: If an invalid thresholding method is specified.

    References:
        [1] O. Wagih and L. Parts, "gitter: A robust and accurate method
        for quantification of colony sizes from plate images," *G3
        (Bethesda)*, vol. 4, no. 3, pp. 547--552, 2014.
        doi: 10.1534/g3.113.009431.

    See Also:
        :doc:`/tutorials/notebooks/02_detecting_colonies`
            Step-by-step tutorial for basic colony detection.
        :doc:`/how_to/notebooks/choose_detection_algorithm`
            Guide for selecting the right detector for your plate images.
        :doc:`/explanation/detection_strategies_compared`
            In-depth comparison of all detection strategies.
    """

    thresh_method: Literal[
        "otsu", "mean", "local", "triangle", "minimum", "isodata", "li"
    ] = "otsu"
    subtract_background: bool = True
    remove_noise: bool = True
    # TODO: review bound (unverified vs literature)
    footprint_width: Annotated[int, TuneSpec(4, 20)] = Field(6, ge=1)
    noise_radius: Annotated[int, TuneSpec(1, 3)] = Field(1, ge=1)
    smoothing_sigma: Annotated[float, TuneSpec(0.0, 5.0)] = 2.0
    min_peak_distance: Annotated[Optional[int], TuneSpec(tunable=False)] = None
    peak_prominence: Annotated[Optional[float], TuneSpec(tunable=False)] = None
    edge_refinement: bool = True
    selection_mode: Literal["dominant", "centered", "regularized"] = "dominant"
    split_merged: bool = True

    @staticmethod
    def _round_odd(n: int) -> int:
        """Round to nearest odd integer (minimum 3)."""
        return _grid_peak_round_odd(n)

    def _operate(self, image: Image) -> Image:
        """
        Detect colonies in the image using the gitter algorithm.

        This method performs the _core detection workflow:
        1. Extract grid dimensions (if GridImage)
        2. Threshold the detection matrix with adaptive kernel sizing
        3. Remove noise if requested
        4. Label connected components
        5. Determine or estimate grid edges
        6. Assign dominant colonies to grid cells
        7. Create final object map

        Args:
            image: Image object to process. Can be a regular Image or GridImage.

        Returns:
            Image: The processed image with updated objmask and objmap.
        """
        from phenotypic import GridImage

        enh_matrix = image.detect_mat[:]
        self._log_memory_usage("getting detection matrix")

        # Extract grid dimensions early for adaptive kernel sizing
        if isinstance(image, GridImage):
            nrows, ncols = image.nrows, image.ncols
        else:
            nrows = ncols = None

        objmask = self._thresholding(enh_matrix, nrows=nrows, ncols=ncols)
        self._log_memory_usage("thresholding")

        if self.remove_noise:
            objmask = morphology.opening(
                    objmask, footprint=morphology.diamond(radius=self.noise_radius)
            )
            self._log_memory_usage("noise removal")

        # Keep a copy of the mask we intend to use for downstream measurements
        image.objmask[:] = objmask

        labeled, num_features = ndimage.label(
                objmask, structure=ndimage.generate_binary_structure(
                        rank=2,
                        connectivity=2)
        )
        self._log_memory_usage(f"labeling ({num_features} features)")

        # Determine grid edges either from GridImage or by estimating from the binary mask
        if isinstance(image, GridImage):
            row_edges = np.round(image.grid.get_row_edges()).astype(int)
            col_edges = np.round(image.grid.get_col_edges()).astype(int)
        else:
            row_edges = col_edges = None

        if row_edges is None or col_edges is None:
            # Estimate edges using peak finding on row/col sums
            nrows, ncols = self._infer_grid_shape(objmask)
            self._log_memory_usage(f"inferred grid shape: {nrows}x{ncols}")

            row_edges = self._estimate_edges(
                    objmask,
                    axis=0,
                    n_bins=nrows,
                    smoothing_sigma=self.smoothing_sigma,
                    min_peak_distance=self.min_peak_distance,
                    peak_prominence=self.peak_prominence,
            )
            col_edges = self._estimate_edges(
                    objmask,
                    axis=1,
                    n_bins=ncols,
                    smoothing_sigma=self.smoothing_sigma,
                    min_peak_distance=self.min_peak_distance,
                    peak_prominence=self.peak_prominence,
            )
            self._log_memory_usage("edge estimation")

            # Refine edges if requested
            if self.edge_refinement:
                row_edges = self._refine_edges(objmask, row_edges, axis=0)
                col_edges = self._refine_edges(objmask, col_edges, axis=1)
                self._log_memory_usage("edge refinement")

        row_edges = np.clip(np.unique(row_edges), 0, objmask.shape[0])
        col_edges = np.clip(np.unique(col_edges), 0, objmask.shape[1])

        # Assign colonies to grid cells using selection strategy
        objmap = self._assign_grid_objects(
                labeled, row_edges, col_edges, self.selection_mode, image._OBJMAP_DTYPE,
                intensity=enh_matrix, split_merged=self.split_merged,
        )

        # Fallback if no regions were labeled (e.g., grid inference failed)
        if objmap.max() == 0:
            objmap = labeled.astype(image._OBJMAP_DTYPE, copy=False)

        self._log_memory_usage("grid cell assignment")

        image.objmap[:] = objmap
        image.objmap.relabel(connectivity=1)

        gc.collect()  # Force garbage collection
        self._log_memory_usage(
                "final cleanup", include_process=True, include_tracemalloc=True
        )

        return image

    def _thresholding(
            self,
            matrix: np.ndarray,
            nrows: int | None = None,
            ncols: int | None = None,
    ) -> np.ndarray:
        """
        Threshold the image to create a binary mask of foreground colonies.

        This method applies optional background subtraction followed by one of
        several thresholding algorithms to separate colonies from background.

        Args:
            matrix: 2D detection matrix array with pixel intensities.
            nrows: Number of grid rows (from GridImage). When provided, the
                background subtraction kernel is adaptively sized to 1.5x
                the colony spacing along the row axis.
            ncols: Number of grid columns (from GridImage). When provided,
                the background subtraction kernel is adaptively sized to 1.5x
                the colony spacing along the column axis.

        Returns:
            np.ndarray: Binary mask where True/1 indicates colony pixels,
                False/0 indicates background.

        Raises:
            ValueError: If an invalid thresholding method is specified.
        """
        return grid_peak_threshold_mask(
            matrix,
            thresh_method=self.thresh_method,
            subtract_background=self.subtract_background,
            footprint_width=self.footprint_width,
            nrows=nrows,
            ncols=ncols,
            round_odd=self._round_odd,
        )
