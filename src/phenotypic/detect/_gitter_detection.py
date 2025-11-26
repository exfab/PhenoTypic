from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import gc
from typing import Literal

import numpy as np
import scipy.ndimage as ndimage
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from functools import partial

from phenotypic.abc_ import ObjectDetector
import skimage.filters as filters
import skimage.morphology as morphology


class GitterDetector(ObjectDetector):
    """
    Class for detecting colonies in gridded plate images using the gitter algorithm.

    The GitterDetector implements a Python version of the gitter colony detection
    algorithm originally developed for R. This method is specifically designed for
    quantifying pinned microbial cultures arranged in a regular grid pattern on
    agar plates. The algorithm works by:

    1. Thresholding the image to create a binary mask of colonies
    2. Analyzing row and column intensity profiles to detect periodic peaks
    3. Estimating grid edges based on peak positions
    4. Assigning pixels to grid cells and identifying dominant colonies

    This approach is robust to irregular colonies, noise, variable illumination,
    and other common plate imaging artifacts.

    Note:
        For best results, use preprocessing such as `GaussianBlur` or other
        enhancement techniques before detection. The detector works best with
        images where colonies are clearly visible against the background.

    Warning:
        Grid inference from the binary mask alone (when not using GridImage)
        may be less accurate than providing explicit grid information. For
        optimal results, use with GridImage when grid parameters are known.

    Attributes:
        thresh_method (str): Thresholding method to use for binary mask creation.
            Options: 'otsu', 'mean', 'local', 'triangle', 'minimum', 'isodata'.
            Default is 'otsu'.
        subtract_background (bool): Whether to apply white tophat background
            subtraction before thresholding. Helps with uneven illumination.
        remove_noise (bool): Whether to apply binary opening to remove small
            noise artifacts after thresholding.
        footprint_radius (int): Radius for morphological operations (noise removal
            and background subtraction kernels).
        smoothing_sigma (float): Standard deviation for Gaussian smoothing of
            row/column sums before peak detection. Higher values increase
            robustness to noise but may merge nearby peaks. Set to 0 to disable.
        min_peak_distance (int | None): Minimum distance between peaks in pixels.
            If None, automatically estimated from grid dimensions. Prevents
            detection of spurious peaks too close together.
        peak_prominence (float | None): Minimum prominence of peaks for detection.
            If None, automatically estimated from signal statistics. Higher values
            are more selective.
        edge_refinement (bool): Whether to refine grid edges using local intensity
            profiles. Improves accuracy but adds computational cost.

    References:
        Wagih, O. and Parts, L. (2014). gitter: a robust and accurate method for
        quantification of colony sizes from plate images. G3 (Bethesda), 4(3), 547-552.
        https://omarwagih.github.io/gitter/
    """

    def __init__(self,
                 thresh_method: Literal['otsu', 'mean', 'local', 'triangle', 'minimum', 'isodata'] = 'otsu',
                 subtract_background: bool = True,
                 remove_noise: bool = True,
                 footprint_radius: int = 3,
                 smoothing_sigma: float = 2.0,
                 min_peak_distance: int | None = None,
                 peak_prominence: float | None = None,
                 edge_refinement: bool = True):
        """
        Initialize the GitterDetector with specified parameters.

        Args:
            thresh_method: Method for thresholding the image. Options are:
                'otsu' (default), 'mean', 'local', 'triangle', 'minimum', 'isodata'.
            subtract_background: If True, apply white tophat transform to remove
                background variations before thresholding.
            remove_noise: If True, apply morphological opening to remove small
                noise artifacts from the binary mask.
            footprint_radius: Radius in pixels for morphological operations.
                Larger values remove larger noise but may erode colony edges.
            smoothing_sigma: Standard deviation for Gaussian smoothing of intensity
                profiles before peak detection. Set to 0 to disable smoothing.
            min_peak_distance: Minimum allowed distance between detected peaks.
                If None, automatically estimated from grid dimensions.
            peak_prominence: Minimum prominence required for peak detection.
                If None, automatically calculated as 0.1 * signal range.
            edge_refinement: If True, refine grid edges using weighted intensity
                profiles for improved accuracy.
        """
        super().__init__()

        self.thresh_method = thresh_method
        self.subtract_background = subtract_background
        self.footprint_radius = footprint_radius
        self.remove_noise = remove_noise
        self.smoothing_sigma = smoothing_sigma
        self.min_peak_distance = min_peak_distance
        self.peak_prominence = peak_prominence
        self.edge_refinement = edge_refinement

    def _operate(self, image: Image) -> Image:
        """
        Detect colonies in the image using the gitter algorithm.

        This method performs the core detection workflow:
        1. Threshold the enhanced grayscale image
        2. Remove noise if requested
        3. Label connected components
        4. Determine or estimate grid edges
        5. Assign dominant colonies to grid cells
        6. Create final object map

        Args:
            image: Image object to process. Can be a regular Image or GridImage.

        Returns:
            Image: The processed image with updated objmask and objmap.
        """
        from phenotypic import GridImage

        enh_matrix = image.enh_gray[:]
        self._log_memory_usage("getting enhanced gray")

        objmask = self._thresholding(enh_matrix)
        self._log_memory_usage("after thresholding")

        if self.remove_noise:
            objmask = morphology.binary_opening(objmask, morphology.diamond(radius=self.footprint_radius))
            self._log_memory_usage("after noise removal")

        # Keep a copy of the mask we intend to use for downstream measurements
        image.objmask[:] = objmask

        labeled, num_features = ndimage.label(objmask, structure=ndimage.generate_binary_structure(2, 2))
        self._log_memory_usage(f"after labeling ({num_features} features)")

        # Determine grid edges either from GridImage or by estimating from the binary mask
        if isinstance(image, GridImage):
            row_edges = np.round(image.grid.get_row_edges()).astype(int)
            col_edges = np.round(image.grid.get_col_edges()).astype(int)
            nrows, ncols = image.nrows, image.ncols
        else:
            nrows = ncols = None
            row_edges = col_edges = None

        if row_edges is None or col_edges is None:
            # Estimate edges using peak finding on row/col sums
            nrows, ncols = self._infer_grid_shape(objmask)
            self._log_memory_usage(f"inferred grid shape: {nrows}x{ncols}")

            row_edges = self._estimate_edges(objmask, axis=0, n_bins=nrows)
            col_edges = self._estimate_edges(objmask, axis=1, n_bins=ncols)
            self._log_memory_usage("after edge estimation")

            # Refine edges if requested
            if self.edge_refinement:
                row_edges = self._refine_edges(objmask, row_edges, axis=0)
                col_edges = self._refine_edges(objmask, col_edges, axis=1)
                self._log_memory_usage("after edge refinement")

        row_edges = np.clip(np.unique(row_edges), 0, objmask.shape[0])
        col_edges = np.clip(np.unique(col_edges), 0, objmask.shape[1])

        objmap = np.zeros_like(labeled, dtype=image._OBJMAP_DTYPE)
        label_counter = 1

        # Assign dominant colonies to each grid cell
        for r in range(len(row_edges) - 1):
            r0, r1 = row_edges[r], row_edges[r + 1]
            for c in range(len(col_edges) - 1):
                c0, c1 = col_edges[c], col_edges[c + 1]
                region = labeled[r0:r1, c0:c1]
                if region.size == 0:
                    continue
                uniq, counts = np.unique(region, return_counts=True)
                valid = uniq != 0
                uniq = uniq[valid]
                counts = counts[valid]
                if uniq.size == 0:
                    continue
                dominant_label = uniq[np.argmax(counts)]
                mask = region == dominant_label
                if np.any(mask):
                    objmap[r0:r1, c0:c1][mask] = label_counter
                    label_counter += 1

        # Fallback if no regions were labeled (e.g., grid inference failed)
        if label_counter == 1:
            objmap = labeled.astype(image._OBJMAP_DTYPE, copy=False)

        self._log_memory_usage("after grid cell assignment")

        image.objmap[:] = objmap
        image.objmap.relabel(connectivity=1)

        gc.collect()  # Force garbage collection
        self._log_memory_usage("final cleanup", include_process=True, include_tracemalloc=True)

        return image

    def _thresholding(self, matrix: np.ndarray) -> np.ndarray:
        """
        Threshold the image to create a binary mask of foreground colonies.

        This method applies optional background subtraction followed by one of
        several thresholding algorithms to separate colonies from background.

        Args:
            matrix: 2D enhanced grayscale array with pixel intensities.

        Returns:
            np.ndarray: Binary mask where True/1 indicates colony pixels,
                False/0 indicates background.

        Raises:
            ValueError: If an invalid thresholding method is specified.
        """
        kernel = morphology.footprint_rectangle((self.footprint_radius * 2, self.footprint_radius * 2))
        enh_matrix = matrix.copy()  # Work on a copy to avoid modifying input

        # Subtract background using white tophat to handle uneven illumination
        if self.subtract_background:
            tophat_res = morphology.white_tophat(enh_matrix, kernel)
            enh_matrix = enh_matrix - tophat_res

        # Apply selected thresholding method
        match self.thresh_method:
            case 'otsu':
                thresh = filters.threshold_otsu(enh_matrix)
            case 'mean':
                thresh = filters.threshold_mean(enh_matrix)
            case 'local':
                block_size = max(self.footprint_radius * 2 + 1, 3)  # Ensure odd block size
                thresh = filters.threshold_local(enh_matrix, block_size=block_size)
            case 'triangle':
                thresh = filters.threshold_triangle(enh_matrix)
            case 'minimum':
                thresh = filters.threshold_minimum(enh_matrix)
            case 'isodata':
                thresh = filters.threshold_isodata(enh_matrix)
            case _:
                # Default to Otsu if method not recognized
                thresh = filters.threshold_otsu(enh_matrix)

        return enh_matrix >= thresh

    def _clean_and_sum_binary(self, binary_image: np.ndarray, p: float = 0.2, axis: int = 0) -> np.ndarray:
        """
        Compute projection sums while removing problematic edge artifacts.

        This method identifies rows (axis=0) or columns (axis=1) near image edges
        that contain abnormally long stretches of foreground pixels (likely artifacts
        or plate edges) and excludes them from the sum to avoid spurious peaks.

        Args:
            binary_image: Binary mask of detected colonies.
            p: Proportion of image dimension to use as threshold for
                detecting problematic long runs (default: 0.2 = 20%).
            axis: Direction to sum along following numpy convention.
                - axis=0: Sum along rows (collapse rows → column sums for row edge detection)
                - axis=1: Sum along columns (collapse columns → row sums for column edge detection)

        Returns:
            np.ndarray: 1D array of cleaned sums along the specified axis.
                Problematic edge regions are set to 0.

        Note:
            This cleaning step helps avoid detecting false peaks from plate
            edges or imaging artifacts that span large portions of rows/columns.
        """
        # Calculate threshold based on image dimensions
        # For axis=0: we're summing columns, so check for long runs across columns
        # For axis=1: we're summing rows, so check for long runs across rows
        if axis == 0:
            c = p * binary_image.shape[1]  # Threshold based on number of columns
            n_slices = binary_image.shape[0]  # Number of rows to iterate through
        else:
            c = p * binary_image.shape[0]  # Threshold based on number of rows
            n_slices = binary_image.shape[1]  # Number of columns to iterate through

        # Identify problematic rows/columns with long stretches of 1s
        problematic = np.zeros(n_slices, dtype=bool)

        for i in range(n_slices):
            if axis == 0:
                slice_data = binary_image[i, :]  # Get row i
            else:
                slice_data = binary_image[:, i]  # Get column i

            # Run-length encoding to find stretches of 1s
            diff = np.diff(np.concatenate(([0], slice_data.astype(int), [0])))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            lengths = ends - starts

            # Check if any stretch of 1s is longer than threshold
            if len(lengths) > 0 and np.any(lengths > c):
                problematic[i] = True

        # Compute sums along the specified axis
        sums = np.sum(binary_image, axis=axis, dtype=np.float64)

        # Split problematic array in half and zero out problematic regions at edges
        mid = len(problematic) // 2
        left_prob = problematic[:mid]
        right_prob = problematic[mid:]

        # Zero out sums for problematic regions at edges
        if np.any(left_prob):
            last_prob = np.where(left_prob)[0][-1]
            sums[:last_prob + 1] = 0

        if np.any(right_prob):
            first_prob = np.where(right_prob)[0][0] + mid
            sums[first_prob:] = 0

        return sums

    def _estimate_edges(self, binary_image: np.ndarray, axis: int, n_bins: int) -> np.ndarray:
        """
        Estimate grid edges by detecting periodic peaks in row/column intensity sums.

        This method implements the core of the gitter algorithm by analyzing the
        projection of colonies onto rows or columns. It detects peaks corresponding
        to colony centers and derives grid edges between them.

        Args:
            binary_image: Binary mask of detected colonies.
            axis: Direction for edge detection (0 for row edges, 1 for column edges).
            n_bins: Expected number of grid bins (rows or columns).

        Returns:
            np.ndarray: Array of edge positions including image borders.
                Length is n_bins + 1.

        Note:
            The method applies smoothing to the intensity profile before peak
            detection to improve robustness. If automatic peak detection fails
            to find enough peaks, it falls back to evenly-spaced bins.
        """
        # Get cleaned sums along the specified axis
        sums = self._clean_and_sum_binary(binary_image, axis=axis)

        # Apply Gaussian smoothing if requested to reduce noise
        if self.smoothing_sigma > 0:
            sums = gaussian_filter1d(sums, sigma=self.smoothing_sigma)

        # Calculate expected spacing between colonies
        image_size = binary_image.shape[1 - axis]  # Size along the summed dimension
        expected_spacing = max(image_size // max(n_bins, 1), 1)

        # Determine peak detection parameters
        min_distance = self.min_peak_distance if self.min_peak_distance is not None else max(expected_spacing // 2, 1)

        # Calculate prominence if not provided
        if self.peak_prominence is not None:
            prominence = self.peak_prominence
        else:
            signal_range = np.max(sums) - np.min(sums)
            prominence = 0.1 * signal_range if signal_range > 0 else None

        # Detect peaks with prominence and distance constraints
        peaks, properties = find_peaks(sums, distance=min_distance, prominence=prominence)

        if peaks.size < n_bins:
            # Fallback: enforce evenly spaced peaks if auto detection under-fits
            peaks = np.linspace(
                start=expected_spacing // 2,
                stop=image_size - expected_spacing // 2,
                num=n_bins,
                dtype=int
            )
        elif peaks.size > n_bins:
            # Keep the strongest n_bins peaks by height
            peak_heights = sums[peaks]
            top_indices = np.argsort(peak_heights)[-n_bins:]
            peaks = np.sort(peaks[top_indices])

        # Derive edges midway between peaks
        if len(peaks) > 1:
            # Calculate midpoints between consecutive peaks
            midpoints = ((peaks[:-1] + peaks[1:]) / 2).astype(int)
            # Prepend/append image borders
            edges = np.concatenate(([0], midpoints, [image_size]))
        else:
            # Fallback for single or no peaks: evenly divide the space
            edges = np.linspace(0, image_size, n_bins + 1, dtype=int)

        # Ensure we have exactly n_bins + 1 edges
        if edges.size > n_bins + 1:
            edges = edges[:n_bins + 1]
        elif edges.size < n_bins + 1:
            missing = (n_bins + 1) - edges.size
            edges = np.concatenate((edges, np.full(missing, image_size)))

        return edges.astype(int)

    def _refine_edges(self, binary_image: np.ndarray, edges: np.ndarray, axis: int) -> np.ndarray:
        """
        Refine grid edges using local intensity profiles for improved accuracy.

        This method adjusts edge positions by analyzing the intensity distribution
        near each initial edge estimate. It shifts edges to positions of minimum
        intensity (background) between colonies.

        Args:
            binary_image: Binary mask of detected colonies.
            edges: Initial edge estimates from peak detection.
            axis: Direction of edges (0 for row edges, 1 for column edges).

        Returns:
            np.ndarray: Refined edge positions.

        Note:
            This refinement step can significantly improve accuracy by placing
            edges in the valleys between colonies rather than at fixed positions.
        """
        refined_edges = edges.copy()
        sums = np.sum(binary_image, axis=axis, dtype=np.float64)

        # Refine each internal edge (not the borders)
        for i in range(1, len(edges) - 1):
            edge_pos = edges[i]
            # Define search window around current edge
            search_radius = min(10, (edges[i+1] - edges[i-1]) // 4)
            search_start = max(0, edge_pos - search_radius)
            search_end = min(len(sums), edge_pos + search_radius + 1)

            # Find position of minimum intensity in search window
            search_window = sums[search_start:search_end]
            if len(search_window) > 0:
                local_min_idx = np.argmin(search_window)
                refined_edges[i] = search_start + local_min_idx

        return refined_edges.astype(int)

    def _infer_grid_shape(self, binary_image: np.ndarray) -> tuple[int, int]:
        """
        Infer grid dimensions from the binary mask when not explicitly provided.

        This method estimates the number of rows and columns in the grid by
        counting connected components and assuming a roughly rectangular layout.
        Common plate formats (96-well, 384-well) are used as fallbacks.

        Args:
            binary_image: Binary mask of detected colonies.

        Returns:
            tuple[int, int]: Estimated (n_rows, n_cols) for the grid.

        Note:
            This is a best-effort estimate. For accurate results, provide
            grid dimensions explicitly using GridImage.
        """
        labeled, num = ndimage.label(binary_image)
        if num == 0:
            # Default to 96-well plate format (8x12)
            return 8, 12

        # Estimate based on aspect ratio and colony count
        aspect_ratio = binary_image.shape[1] / binary_image.shape[0]

        if aspect_ratio > 1.3:  # Wide plate (likely 8x12 or similar)
            # Try 8x12 (96 wells), 16x24 (384 wells), etc.
            if num <= 100:
                return 8, 12
            elif num <= 400:
                return 16, 24
            else:
                approx_rows = int(np.ceil(np.sqrt(num / aspect_ratio)))
                approx_cols = int(np.ceil(np.sqrt(num * aspect_ratio)))
                return approx_rows, approx_cols
        else:
            # Square-ish layout
            approx_side = int(np.ceil(np.sqrt(num)))
            return approx_side, max(approx_side, 1)

