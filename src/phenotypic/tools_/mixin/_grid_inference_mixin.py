"""Mixin providing grid inference capabilities for gridded plate image processing.

This module contains GridInferenceMixin, which provides static methods for inferring
grid structure from colony patterns. Used by both ObjectDetector (RoundPeaksDetector)
and ObjectRefiner (GridAlignmentRefiner) subclasses.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import scipy.ndimage as ndimage
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


class GridInferenceMixin:
    """Mixin providing grid inference capabilities from binary masks.

    Provides static methods for inferring grid structure from colony patterns using
    peak detection on row/column projections. Used by detectors and refiners
    that work with gridded plate images (96-well, 384-well formats, pinned cultures).

    All methods are static to support parallelization in pipeline operations.

    This is an internal utility for advanced users creating custom grid-based operations.
    Most users should use RoundPeaksDetector for detection or GridAlignmentRefiner for
    post-detection refinement directly.
    """

    @staticmethod
    def _clean_and_sum_binary(
        binary_image: np.ndarray, p: float = 0.2, axis: int = 0
    ) -> np.ndarray:
        """Compute projection sums while removing problematic edge artifacts.

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
            sums[: last_prob + 1] = 0

        if np.any(right_prob):
            first_prob = np.where(right_prob)[0][0] + mid
            sums[first_prob:] = 0

        return sums

    @staticmethod
    def _estimate_edges(
        binary_image: np.ndarray,
        axis: int,
        n_bins: int,
        smoothing_sigma: float = 2.0,
        min_peak_distance: int | None = None,
        peak_prominence: float | None = None,
    ) -> np.ndarray:
        """Estimate grid edges by detecting periodic peaks in row/column intensity sums.

        This method implements the core of the gitter algorithm by analyzing the
        projection of colonies onto rows or columns. It detects peaks corresponding
        to colony centers and derives grid edges between them.

        Args:
            binary_image: Binary mask of detected colonies.
            axis: Direction for edge detection (0 for row edges, 1 for column edges).
            n_bins: Expected number of grid bins (rows or columns).
            smoothing_sigma: Gaussian smoothing standard deviation for intensity profile.
                Default 2.0. Higher values smooth noise but may merge peaks.
            min_peak_distance: Minimum pixel distance between detected peaks. If None,
                automatically estimated from grid dimensions.
            peak_prominence: Minimum prominence threshold for peak detection. If None,
                auto-calculated as 0.1 * signal range.

        Returns:
            np.ndarray: Array of edge positions including image borders.
                Length is n_bins + 1.

        Note:
            The method applies smoothing to the intensity profile before peak
            detection to improve robustness. If automatic peak detection fails
            to find enough peaks, it falls back to evenly-spaced bins.
        """
        # Get cleaned sums along the specified axis
        sums = GridInferenceMixin._clean_and_sum_binary(binary_image, axis=axis)

        # Apply Gaussian smoothing if requested to reduce noise
        if smoothing_sigma > 0:
            sums = gaussian_filter1d(sums, sigma=smoothing_sigma)

        # Calculate expected spacing between colonies
        image_size = binary_image.shape[1 - axis]  # Size along the summed dimension
        expected_spacing = max(image_size // max(n_bins, 1), 1)

        # Determine peak detection parameters
        min_distance = (
            min_peak_distance if min_peak_distance is not None else max(expected_spacing // 2, 1)
        )

        # Calculate prominence if not provided
        if peak_prominence is not None:
            prominence = peak_prominence
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
                dtype=int,
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
            edges = edges[: n_bins + 1]
        elif edges.size < n_bins + 1:
            missing = (n_bins + 1) - edges.size
            edges = np.concatenate((edges, np.full(missing, image_size)))

        return edges.astype(int)

    @staticmethod
    def _refine_edges(
        binary_image: np.ndarray, edges: np.ndarray, axis: int
    ) -> np.ndarray:
        """Refine grid edges using local intensity profiles for improved accuracy.

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
            search_radius = min(10, (edges[i + 1] - edges[i - 1]) // 4)
            search_start = max(0, edge_pos - search_radius)
            search_end = min(len(sums), edge_pos + search_radius + 1)

            # Find position of minimum intensity in search window
            search_window = sums[search_start:search_end]
            if len(search_window) > 0:
                local_min_idx = np.argmin(search_window)
                refined_edges[i] = search_start + local_min_idx

        return refined_edges.astype(int)

    @staticmethod
    def _infer_grid_shape(binary_image: np.ndarray) -> tuple[int, int]:
        """Infer grid dimensions from the binary mask when not explicitly provided.

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

    @staticmethod
    def _select_cell_object(
        region: np.ndarray,
        selection_mode: Literal["dominant", "centered"],
    ) -> int | None:
        """Select a single object label from a grid cell region.

        Args:
            region: 2D labeled array for a single grid cell (0 = background).
            selection_mode: Strategy for choosing the object.
                ``"dominant"`` picks the label with the most pixels.
                ``"centered"`` picks the label whose centroid is closest
                to the cell center.

        Returns:
            The selected label, or ``None`` if the region contains no
            foreground objects.

        Raises:
            ValueError: If *selection_mode* is not ``"dominant"`` or
                ``"centered"``.
        """
        uniq, counts = np.unique(region, return_counts=True)
        valid = uniq != 0
        if not np.any(valid):
            return None
        uniq = uniq[valid]
        counts = counts[valid]

        if selection_mode == "dominant":
            return int(uniq[np.argmax(counts)])

        if selection_mode == "centered":
            center_r = region.shape[0] / 2.0
            center_c = region.shape[1] / 2.0
            centroids = ndimage.center_of_mass(
                np.ones_like(region), region, uniq.tolist()
            )
            best_label = None
            best_dist = np.inf
            for label, (cr, cc) in zip(uniq, centroids):
                dist = (cr - center_r) ** 2 + (cc - center_c) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_label = int(label)
            return best_label

        raise ValueError(
            f"Unknown selection_mode {selection_mode!r}. "
            "Use 'dominant' or 'centered'."
        )

    @staticmethod
    def _assign_grid_objects(
        labeled: np.ndarray,
        row_edges: np.ndarray,
        col_edges: np.ndarray,
        selection_mode: Literal["dominant", "centered", "regularized"],
        dtype: np.dtype[Any] | type,
    ) -> np.ndarray:
        """Assign one object per grid cell using the chosen selection strategy.

        Args:
            labeled: Full labeled array (0 = background, >0 = objects).
            row_edges: Sorted array of row boundary positions.
            col_edges: Sorted array of column boundary positions.
            selection_mode: ``"dominant"`` keeps the largest object per cell,
                ``"centered"`` keeps the most centred object, and
                ``"regularized"`` uses a two-pass global fit to recover
                expected colony positions from median row/column centroids.
            dtype: NumPy dtype for the output array.

        Returns:
            A new labeled array with sequential labels (1, 2, 3, ...).
        """
        nrows = len(row_edges) - 1
        ncols = len(col_edges) - 1
        refined_map = np.zeros_like(labeled, dtype=dtype)

        if selection_mode in ("dominant", "centered"):
            per_cell_mode: Literal["dominant", "centered"] = (
                "dominant" if selection_mode == "dominant" else "centered"
            )
            label_counter = 1
            for r in range(nrows):
                r0, r1 = row_edges[r], row_edges[r + 1]
                for c in range(ncols):
                    c0, c1 = col_edges[c], col_edges[c + 1]
                    region = labeled[r0:r1, c0:c1]
                    if region.size == 0:
                        continue
                    chosen = GridInferenceMixin._select_cell_object(
                        region, per_cell_mode
                    )
                    if chosen is not None:
                        mask = region == chosen
                        if np.any(mask):
                            refined_map[r0:r1, c0:c1][mask] = label_counter
                            label_counter += 1
            return refined_map

        # --- regularized: two-pass global approach ---

        # Pass 1: collect initial candidates using dominant selection
        candidates: list[tuple[int, int, float, float]] = []  # (row_idx, col_idx, abs_r, abs_c)
        for r in range(nrows):
            r0, r1 = row_edges[r], row_edges[r + 1]
            for c in range(ncols):
                c0, c1 = col_edges[c], col_edges[c + 1]
                region = labeled[r0:r1, c0:c1]
                if region.size == 0:
                    continue
                chosen = GridInferenceMixin._select_cell_object(region, "dominant")
                if chosen is not None:
                    # Centroid in local coords
                    local_coords = ndimage.center_of_mass(
                        (region == chosen).astype(np.uint8)
                    )
                    abs_r = local_coords[0] + r0
                    abs_c = local_coords[1] + c0
                    candidates.append((r, c, abs_r, abs_c))

        # Fit global grid model: median row/column positions
        expected_r = np.empty(nrows)
        expected_c = np.empty(ncols)

        for i in range(nrows):
            row_vals = [abs_r for (ri, _, abs_r, _) in candidates if ri == i]
            if row_vals:
                expected_r[i] = float(np.median(row_vals))
            else:
                # Fallback: cell center
                expected_r[i] = (row_edges[i] + row_edges[i + 1]) / 2.0

        for j in range(ncols):
            col_vals = [abs_c for (_, cj, _, abs_c) in candidates if cj == j]
            if col_vals:
                expected_c[j] = float(np.median(col_vals))
            else:
                expected_c[j] = (col_edges[j] + col_edges[j + 1]) / 2.0

        # Pass 2: re-select using expected positions
        label_counter = 1
        for r in range(nrows):
            r0, r1 = row_edges[r], row_edges[r + 1]
            for c in range(ncols):
                c0, c1 = col_edges[c], col_edges[c + 1]
                region = labeled[r0:r1, c0:c1]
                if region.size == 0:
                    continue

                uniq, counts = np.unique(region, return_counts=True)
                valid = uniq != 0
                if not np.any(valid):
                    continue
                uniq = uniq[valid]

                # Compute centroid of each object and pick closest to expected
                centroids = ndimage.center_of_mass(
                    np.ones_like(region), region, uniq.tolist()
                )
                best_label = None
                best_dist = np.inf
                for lbl, (cr, cc) in zip(uniq, centroids):
                    abs_cr = cr + r0
                    abs_cc = cc + c0
                    dist = (abs_cr - expected_r[r]) ** 2 + (abs_cc - expected_c[c]) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        best_label = int(lbl)

                if best_label is not None:
                    mask = region == best_label
                    if np.any(mask):
                        refined_map[r0:r1, c0:c1][mask] = label_counter
                        label_counter += 1

        return refined_map
