"""Grid-aligned object refinement using sinusoidal cross-correlation.

Refines detected colonies by filtering to keep only the dominant object in each grid cell,
using FFT-based sinusoidal cross-correlation (gitter-faithful, Wagih & Parts 2014) for
grid edge estimation. More robust to outlier colonies than simple peak-finding because
rank-based Spearman correlation is insensitive to monotonic intensity transformations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import gc

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, fftconvolve, medfilt
from scipy.stats import rankdata

from phenotypic.abc_ import ObjectRefiner
from phenotypic.tools_.mixin import GridInferenceMixin
from phenotypic.tools_.funcs_ import validate_operation_integrity


class SineAlignmentRefiner(GridInferenceMixin, ObjectRefiner):
    """Retain only grid-aligned colonies using sinusoidal cross-correlation for grid estimation.

    Estimates grid edges by computing FFT-based normalized cross-correlation
    against a sinusoidal template of expected colony periodicity, then keeps
    one dominant object per cell. Rank-based (Spearman) correlation provides
    robustness to outlier colony intensities and monotonic intensity
    transformations compared to simple peak-finding.

    Args:
        smoothing_sigma: Gaussian smoothing sigma for intensity profiles.
            Typical range: 0.5--5.0. Higher values smooth noise but may
            merge adjacent peaks. Default: 2.0.
        min_peak_distance: Minimum pixel distance between grid peaks.
            ``None`` auto-estimates. Default: None.
        peak_prominence: Minimum prominence for peak detection. ``None``
            auto-calculates. Default: None.
        edge_refinement: Refine grid edges using local intensity minima.
            Default: True.
        correlation_threshold: Minimum NCC value for a valid peak.
            Typical range: 0.1--0.6. Lower values accept weaker matches;
            higher values are more selective. Default: 0.3.
        selection_mode: Strategy for choosing one object per cell.
            ``"dominant"`` keeps the largest, ``"centered"`` keeps the
            most centered, ``"regularized"`` fits a global model.
            Default: ``"dominant"``.

    Returns:
        Image: Input image with ``objmap`` filtered to grid-aligned objects
        and ``objmask`` updated to match.

    Raises:
        ValueError: If grid inference fails or image lacks detection results.

    Best For:
        - Gridded plates (96-well, 384-well, pinned cultures) where colony
          intensities are heterogeneous or unevenly grown.
        - Post-detection cleanup when simple peak-finding grid estimation
          is unreliable.
        - Plates with variable colony sizes or uneven growth where rank-based
          correlation outperforms direct intensity matching.

    Consider Also:
        - :class:`GridAlignmentRefiner` for faster grid estimation when
          colony intensities are relatively uniform.
        - :class:`GridSectionLargest` for a simpler largest-per-cell
          strategy on GridImage inputs.
        - :class:`ReduceMultipleGridObjects` for regression-based multi-
          detection reduction within grid cells.

    References:
        [1] O. Wagih and L. Parts, "gitter: a robust and accurate method
        for quantification of colony sizes from plate images," *G3
        (Bethesda)*, vol. 4, no. 3, pp. 547--552, 2014.

    See Also:
        :doc:`/how_to/notebooks/refine_noisy_boundaries` for grid-based
        cleanup workflows.
        :doc:`/explanation/refinement_strategies` for a comparison of
        grid refinement approaches.
    """

    def __init__(
            self,
            smoothing_sigma: float = 2.0,
            min_peak_distance: int | None = None,
            peak_prominence: float | None = None,
            edge_refinement: bool = True,
            correlation_threshold: float = 0.3,
            selection_mode: Literal["dominant", "centered", "regularized"] = "dominant",
            split_merged: bool = False,
    ):
        """Initialize SineAlignmentRefiner with grid inference and correlation parameters.

        Args:
            smoothing_sigma: Gaussian smoothing sigma for intensity profiles.
            min_peak_distance: Minimum distance between grid peaks.
            peak_prominence: Minimum prominence for peak detection.
            edge_refinement: Enable edge refinement via local intensity minima.
            correlation_threshold: Minimum NCC value for valid peaks.
            selection_mode: Strategy for choosing the object per grid cell.
                'dominant' (default) keeps the largest, 'centered' keeps
                the most centred, 'regularized' uses a global fit.
            split_merged: If True, pre-split merged colonies that span
                multiple grid cells using EDT watershed before assignment.
                Default False for refiners (splitting is more useful during
                initial detection).
        """
        super().__init__()
        self.smoothing_sigma = smoothing_sigma
        self.min_peak_distance = min_peak_distance
        self.peak_prominence = peak_prominence
        self.edge_refinement = edge_refinement
        self.correlation_threshold = correlation_threshold
        self.selection_mode = selection_mode
        self.split_merged = split_merged

    @validate_operation_integrity("image.rgb", "image.gray", "image.detect_mat")
    def apply(self, image: Image, inplace: bool = False) -> Image:
        return super().apply(image=image, inplace=inplace)

    def _operate(self, image: Image) -> Image:
        """Refine detected objects to grid-aligned colonies using sinusoidal cross-correlation.

        This method filters the object map to keep only the dominant object within each
        grid cell. Grid edges are estimated using FFT-based normalized cross-correlation
        against a sinusoidal template for robust peak detection. Objects are reassigned
        new labels (1, 2, 3, ...) to ensure contiguous labeling after filtering.

        Returns:
            Image: Modified image with filtered objmap and updated objmask.
        """
        from phenotypic import GridImage

        # Get existing objmap
        objmap = image.objmap[:]

        # Determine grid edges (GridImage or infer via sine cross-correlation)
        if isinstance(image, GridImage):
            row_edges = np.round(image.grid.get_row_edges()).astype(int)
            col_edges = np.round(image.grid.get_col_edges()).astype(int)
        else:
            objmask = image.objmask[:]
            nrows, ncols = self._infer_grid_shape(objmask)

            row_edges = self._estimate_edges(
                    objmask,
                    axis=0,
                    n_bins=nrows,
            )
            col_edges = self._estimate_edges(
                    objmask,
                    axis=1,
                    n_bins=ncols,
            )

            if self.edge_refinement:
                row_edges = self._refine_edges(objmask, row_edges, axis=0)
                col_edges = self._refine_edges(objmask, col_edges, axis=1)

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

    def _estimate_edges(self, binary_image: np.ndarray, axis: int, n_bins: int, **kwargs: object) -> np.ndarray:  # type: ignore[override]
        """Estimate grid edges using sinusoidal cross-correlation.

        Overrides GridInferenceMixin._estimate_edges with a gitter-faithful
        approach: generates a sine template matching expected colony periodicity,
        computes FFT-based normalized cross-correlation against the projection
        signal, and selects peaks from the correlation output. Rank-based
        (Spearman) correlation provides robustness to outliers and monotonic
        intensity transformations.

        Args:
            binary_image: Binary mask of detected colonies.
            axis: Direction for edge detection (0 for row edges, 1 for column edges).
            n_bins: Expected number of grid bins (rows or columns).

        Returns:
            np.ndarray: Array of edge positions including image borders.
                Length is n_bins + 1.
        """
        # 1. Clean projection sums (from mixin)
        sums = GridInferenceMixin._clean_and_sum_binary(binary_image, axis=axis)

        # 2. Gaussian smooth
        if self.smoothing_sigma > 0:
            sums = gaussian_filter1d(sums, sigma=self.smoothing_sigma)

        # 3. Signal enhancement: multiply by median-filtered version
        image_size = binary_image.shape[1 - axis]
        expected_spacing = max(image_size // max(n_bins, 1), 1)
        window_size = max(expected_spacing, 3)
        medfilt_kernel = max(window_size // 3, 3)
        if medfilt_kernel % 2 == 0:
            medfilt_kernel += 1
        enhanced = sums * medfilt(sums, kernel_size=medfilt_kernel)

        # 4. Rank transform for Spearman robustness
        ranked_signal = rankdata(enhanced).astype(np.float64)

        # 5. Rank sine template
        template = np.sin(np.linspace(-np.pi, 2 * np.pi, window_size))
        ranked_template = rankdata(template).astype(np.float64)

        # 6. FFT normalized cross-correlation
        ncc = self._normalized_cross_correlation(ranked_signal, ranked_template)

        # 7. Threshold low correlations
        ncc[ncc < self.correlation_threshold] = 0

        # 8. Find peaks
        min_distance = (
            self.min_peak_distance if self.min_peak_distance is not None
            else max(expected_spacing // 2, 1)
        )
        if self.peak_prominence is not None:
            prominence: float | None = self.peak_prominence
        else:
            signal_range = np.max(ncc) - np.min(ncc)
            prominence = 0.1 * signal_range if signal_range > 0 else None

        peaks, _ = find_peaks(ncc, distance=min_distance, prominence=prominence)

        # 9. Select best n_bins peaks by correlation height, sorted by position
        if peaks.size > n_bins:
            peak_heights = ncc[peaks]
            top_indices = np.argsort(peak_heights)[-n_bins:]
            peaks = np.sort(peaks[top_indices])
        elif peaks.size < n_bins:
            # Fallback: evenly spaced peaks
            peaks = np.linspace(
                start=expected_spacing // 2,
                stop=image_size - expected_spacing // 2,
                num=n_bins,
                dtype=int,
            )

        # 10. Derive edges at midpoints
        if len(peaks) > 1:
            midpoints = ((peaks[:-1] + peaks[1:]) / 2).astype(int)
            edges = np.concatenate(([0], midpoints, [image_size]))
        else:
            edges = np.linspace(0, image_size, n_bins + 1, dtype=int)

        # Ensure exactly n_bins + 1 edges
        if edges.size > n_bins + 1:
            edges = edges[:n_bins + 1]
        elif edges.size < n_bins + 1:
            missing = (n_bins + 1) - edges.size
            edges = np.concatenate((edges, np.full(missing, image_size)))

        return edges.astype(int)

    @staticmethod
    def _normalized_cross_correlation(signal: np.ndarray, template: np.ndarray) -> np.ndarray:
        """FFT-based normalized cross-correlation.

        Computes the normalized cross-correlation between a signal and a
        template using FFT convolution for O(N log N) performance. The
        result is clipped to [-1, 1].

        Args:
            signal: 1D input signal array.
            template: 1D template array (typically shorter than signal).

        Returns:
            np.ndarray: Normalized cross-correlation values, same length as
                signal, clipped to [-1, 1].
        """
        n = len(signal)
        k = len(template)

        # Zero-mean template
        template_mean = np.mean(template)
        template_zm = template - template_mean
        template_norm = np.sqrt(np.sum(template_zm ** 2))

        if template_norm < 1e-10:
            return np.zeros(n)

        # Cross-correlation via FFT
        xcorr = fftconvolve(signal, template_zm[::-1], mode="same")

        # Local statistics via FFT with ones kernel
        ones_kernel = np.ones(k)
        local_sum = fftconvolve(signal, ones_kernel, mode="same")
        local_sum_sq = fftconvolve(signal ** 2, ones_kernel, mode="same")

        local_mean = local_sum / k
        # Use sum-of-squares form: sqrt(sum((x-mean)^2)) to match template_norm scale
        local_energy = np.maximum(local_sum_sq - local_sum ** 2 / k, 0)
        local_std = np.sqrt(local_energy)

        # Normalize (suppress divide-by-zero where denom is near-zero)
        denom = local_std * template_norm
        safe_denom = np.where(denom > 1e-10, denom, 1.0)
        ncc = np.where(
            denom > 1e-10,
            (xcorr - local_mean * np.sum(template_zm)) / safe_denom,
            0.0,
        )

        return np.clip(ncc, -1.0, 1.0)
