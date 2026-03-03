from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import gc
from typing import Literal

import numpy as np
import scipy.ndimage as ndimage

from phenotypic.abc_ import ObjectDetector
from phenotypic.tools_.mixin import GridInferenceMixin
import skimage.filters as filters
import skimage.morphology as morphology


class RoundPeaksDetector(GridInferenceMixin, ObjectDetector):
    """Grid-based colony detector using row/column peak detection (gitter algorithm).

    RoundPeaksDetector identifies colonies in gridded plate images by analyzing
    row and column intensity profiles to detect periodic peaks, estimating grid
    edges, and assigning colonies to grid cells. This implements the gitter algorithm
    originally developed for R, optimized for pinned microbial culture plates with
    circular colonies arranged in regular patterns.

    Args:
        thresh_method: Thresholding method ('otsu', 'mean', 'local', 'triangle',
            'minimum', 'isodata', 'li'). Default 'otsu'. Controls binary mask creation.

        subtract_background: If True (default), apply white tophat transform to
            remove uneven illumination before thresholding.

        remove_noise: If True (default), apply morphological opening to remove small
            noise artifacts from the binary mask.

        footprint_width: Width in pixels for the background subtraction kernel.
            Default 6. When a GridImage is provided, an adaptive kernel sized to
            1.5x colony spacing is used instead.

        noise_radius: Radius for the diamond structuring element used in
            morphological noise removal. Default 1 (3x3 diamond, matching gitter).
            Increase for larger noise artifacts.

        smoothing_sigma: Gaussian smoothing of row/column intensity profiles before
            peak detection. Default 2.0. Higher values smooth noise but may merge peaks.

        min_peak_distance: Minimum pixel distance between detected peaks. If None,
            automatically estimated from grid dimensions.

        peak_prominence: Minimum prominence threshold for peak detection. If None,
            auto-calculated as 0.1 * signal range. Higher values are more selective.

        edge_refinement: If True (default), refine grid edges using local intensity
            profiles for improved accuracy.

        selection_mode: Strategy for choosing one object per grid cell. ``"dominant"``
            (default) keeps the largest object by pixel count. ``"centered"`` keeps
            the object whose centroid is closest to the cell center. ``"regularized"``
            uses a two-pass approach that fits a global regular-grid model from median
            row/column centroids, then re-selects per cell. Best for pinned arrays.

        split_merged: If True (default), pre-split merged colonies that span
            multiple grid cells using EDT watershed before grid assignment. This
            detects two colonies that have grown into each other by finding
            multiple EDT peaks in different grid cells. Set to False to skip
            splitting (e.g., when colonies are well-separated).

    Attributes:
        thresh_method, subtract_background, remove_noise, footprint_radius,
        noise_radius, smoothing_sigma, min_peak_distance, peak_prominence,
        edge_refinement

    Returns:
        Image: Input image with objmask (binary colony mask) and objmap (labeled
        colonies assigned to grid cells) set.

    Raises:
        ValueError: If invalid thresholding method specified.

    **Use cases**

    - **Gridded plate images:** Colonies arranged in regular arrays (96-well, 384-well
      plates, pinned cultures). Peak detection exploits this structure.
    - **Circular colonies:** Works best for yeast-like spherical growth. Less suitable
      for filamentous fungi or irregular morphologies.
    - **Batch processing:** Efficient grid inference enables high-throughput analysis
      without manual grid specification (though GridImage with explicit dimensions
      is more accurate).

    **Limitations**

    - Grid inference from binary mask alone is less accurate than explicit GridImage
      specification. For best results, use with GridImage when grid parameters known.
    - Assumes regular grid geometry. Works poorly with irregular colony spacing or
      missing grid positions.
    - Best for yeast-like morphologies. Less suitable for filamentous, spreading, or
      irregular colony shapes.
    - Computational cost: Peak detection and edge refinement add overhead vs simple
      thresholding.

    **Parameter effects on colony detection**

    - **thresh_method:** Different histogram assumptions (Otsu=variance, mean=simple,
      local=adaptive). Affects mask quality and downstream peak detection.
    - **subtract_background, remove_noise:** Remove preprocessing artifacts (vignetting,
      dust, noise) that can create spurious peaks.
    - **smoothing_sigma:** Balances noise robustness vs peak resolution. Higher values
      smooth noise but may merge adjacent colonies.

    Examples:
        Basic grid detection with default parameters::

            from phenotypic import Image
            from phenotypic.detect import RoundPeaksDetector

            plate = Image.imread("plate_grid.jpg")
            detector = RoundPeaksDetector()
            detected = detector.apply(plate)
            num_colonies = detected.objects.count
            print(f"Detected {num_colonies} colonies in grid")

        Pipeline with preprocessing for noisy plate images::

            from phenotypic import ImagePipeline
            from phenotypic.enhance import GaussianBlur, CLAHE
            from phenotypic.detect import RoundPeaksDetector

            pipeline = ImagePipeline([
                GaussianBlur(sigma=1.5),
                CLAHE(clip_limit=2.0),
                RoundPeaksDetector(thresh_method='otsu', smoothing_sigma=2.0)
            ])

            image = Image.imread("plate_grid.jpg")
            result = pipeline.apply(image)

    References:
        Wagih, O. and Parts, L. (2014). gitter: a robust and accurate method for
        quantification of colony sizes from plate images. G3 (Bethesda), 4(3), 547-552.
    """

    def __init__(
            self,
            thresh_method: Literal[
                "otsu", "mean", "local", "triangle", "minimum", "isodata", "li"
            ] = "otsu",
            subtract_background: bool = True,
            remove_noise: bool = True,
            footprint_width: int = 6,
            noise_radius: int = 1,
            smoothing_sigma: float = 2.0,
            min_peak_distance: int | None = None,
            peak_prominence: float | None = None,
            edge_refinement: bool = True,
            selection_mode: Literal["dominant", "centered", "regularized"] = "dominant",
            split_merged: bool = True,
    ):
        """
        Initialize the RoundPeaksDetector with specified parameters.

        Args:
            thresh_method: Method for thresholding the image. Options are:
                'otsu' (default), 'mean', 'local', 'triangle', 'minimum',
                'isodata', 'li'.
            subtract_background: If True, apply white tophat transform to remove
                background variations before thresholding.
            remove_noise: If True, apply morphological opening to remove small
                noise artifacts from the binary mask.
            footprint_width: Width in pixels for the background subtraction kernel.
                When a GridImage is provided, an adaptive kernel sized to 1.5x
                colony spacing is used instead, making this a fallback.
            noise_radius: Radius for the diamond structuring element used in
                morphological noise removal. Default 1 (3x3 diamond, matching
                gitter). Increase for larger noise artifacts.
            smoothing_sigma: Standard deviation for Gaussian smoothing of intensity
                profiles before peak detection. Set to 0 to disable smoothing.
            min_peak_distance: Minimum allowed distance between detected peaks.
                If None, automatically estimated from grid dimensions.
            peak_prominence: Minimum prominence required for peak detection.
                If None, automatically calculated as 0.1 * signal range.
            edge_refinement: If True, refine grid edges using weighted intensity
                profiles for improved accuracy.
            selection_mode: Strategy for choosing one object per grid cell.
                'dominant' (default) keeps the largest, 'centered' keeps
                the most centred, 'regularized' uses a global fit.
            split_merged: If True (default), pre-split merged colonies that
                span multiple grid cells using EDT watershed before assignment.
        """
        super().__init__()

        self.thresh_method = thresh_method
        self.subtract_background = subtract_background
        self.footprint_radius = footprint_width
        self.noise_radius = noise_radius
        self.remove_noise = remove_noise
        self.smoothing_sigma = smoothing_sigma
        self.min_peak_distance = min_peak_distance
        self.peak_prominence = peak_prominence
        self.edge_refinement = edge_refinement
        self.selection_mode = selection_mode
        self.split_merged = split_merged

    @staticmethod
    def _round_odd(n: int) -> int:
        """Round to nearest odd integer (minimum 3)."""
        n = max(n, 3)
        return n if n % 2 == 1 else n + 1

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
        # Adaptive kernel sizing: use grid spacing when available, fallback otherwise
        if nrows is not None:
            bg_h = self._round_odd(round((matrix.shape[0] / nrows) * 1.5))
            bg_w = self._round_odd(round((matrix.shape[1] / (ncols or nrows)) * 1.5))
            kernel = morphology.footprint_rectangle((bg_h, bg_w))
        else:
            dim = self._round_odd(self.footprint_radius * 2)
            kernel = morphology.footprint_rectangle((dim, dim))

        enh_matrix = matrix.copy()  # Work on a copy to avoid modifying input

        # Isolate bright foreground colonies via white tophat (image - opening)
        if self.subtract_background:
            enh_matrix = morphology.white_tophat(enh_matrix, kernel)

        # Apply selected thresholding method
        match self.thresh_method:
            case "otsu":
                thresh = filters.threshold_otsu(enh_matrix)
            case "mean":
                thresh = filters.threshold_mean(enh_matrix)
            case "local":
                block_size = max(
                        self.footprint_radius * 2 + 1, 3
                )  # Ensure odd block size
                thresh = filters.threshold_local(enh_matrix, block_size=block_size)
            case "triangle":
                thresh = filters.threshold_triangle(enh_matrix)
            case "minimum":
                thresh = filters.threshold_minimum(enh_matrix)
            case "isodata":
                thresh = filters.threshold_isodata(enh_matrix)
            case "li":
                thresh = filters.threshold_li(enh_matrix)
            case _:
                # Default to Otsu if method not recognized
                thresh = filters.threshold_otsu(enh_matrix)

        return enh_matrix >= thresh
