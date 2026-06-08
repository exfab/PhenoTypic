"""Grid-aligned object refinement using sinusoidal cross-correlation.

Refines detected colonies by filtering to keep only the dominant object in each grid cell,
using FFT-based sinusoidal cross-correlation for grid edge estimation. This is inspired
by projection-based grid quantification work such as gitter, but the sinusoidal template
and rank-transform correlation are implementation-specific extensions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

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
from phenotypic.tools_.typing_ import TuneSpec


class RefineBySineFit(GridInferenceMixin, ObjectRefiner):
    """Retain only grid-aligned colonies using sinusoidal cross-correlation for grid edge estimation.

    Estimates grid edges by computing FFT-based normalized cross-correlation
    of rank-transformed projection profiles against a sinusoidal template of
    expected colony periodicity, then retains one object per cell according
    to the chosen selection strategy. Rank-transforming the profile before
    correlation (a Spearman-style step) makes the grid estimate robust to
    outlier colony intensities and to monotonic intensity transformations,
    compared to direct peak-finding on raw intensity profiles.

    For a comparison of grid refinement approaches, see
    :doc:`/explanation/refinement_strategies`.

    Best For:
        - Gridded plates (96-well, 384-well, pinned cultures) where colony
          sizes or intensities are heterogeneous or unevenly grown.
        - Post-detection cleanup when simple projection-peak grid estimation
          misses or misaligns grid lines due to variable colony brightness.
        - Plates with large empty sectors where direct intensity peak-finding
          loses periodicity signal.

    Consider Also:
        - :class:`GridAlignmentRefiner` for faster grid estimation when
          colony intensities are relatively uniform across the plate.
        - :class:`KeepSectionLargest` for a simpler largest-per-cell
          strategy on ``GridImage`` inputs where grid metadata is
          already available.
        - :class:`ReduceSectionsByLine` for regression-based reduction of
          multi-detections within grid cells after grid edges are known.

    Args:
        smoothing_sigma: Standard deviation (pixels) of the Gaussian applied
            to row and column projection profiles before cross-correlation.
            Higher values suppress noise in sparse or uneven profiles but
            risk merging adjacent peaks on dense plates. Typical range:
            0.5--5.0. Default: 2.0.
        min_peak_distance: Minimum pixel distance between detected grid peaks.
            ``None`` auto-estimates from expected colony spacing. Increase
            to avoid spurious sub-peaks on large plates. Default: ``None``.
        peak_prominence: Minimum peak prominence in the cross-correlation
            profile. ``None`` auto-computes as 10% of the profile range.
            Raise to suppress weak periodicity signals from noisy plates.
            Default: ``None``.
        edge_refinement: Refine estimated grid edges by snapping to local
            intensity minima between detected peaks. Disable on plates with
            variable inter-colony spacing. Default: ``True``.
        correlation_threshold: Minimum normalized cross-correlation value
            required to accept a peak as a valid grid position. Lower values
            recover peaks on sparse or faint plates; higher values are more
            selective and suppress false periodicity. Typical range:
            0.1--0.6. Default: 0.3.
        selection_mode: Strategy for keeping one object per grid cell.
            ``"dominant"`` retains the largest object by area; ``"centered"``
            retains the object closest to the cell centroid; ``"regularized"``
            fits a global spatial model. Default: ``"dominant"``.
        split_merged: Attempt to split objects that span multiple grid cells
            before applying per-cell selection. Default: ``False``.

    Returns:
        Image: Input image with ``objmap`` filtered to at most one grid-
        aligned object per cell and ``objmask`` updated to match. ``rgb``,
        ``gray``, and ``detect_mat`` are unchanged.

    Raises:
        ValueError: If grid inference fails or image lacks detection results.

    References:
        [1] O. Wagih and L. Parts, "gitter: a robust and accurate method
        for quantification of colony sizes from plate images," *G3
        (Bethesda)*, vol. 4, no. 3, pp. 547--552, Mar. 2014. Prior work on
        correlation-based grid-colony quantification from plate images; the
        sinusoidal cross-correlation and rank-transform steps used here are
        not drawn from this paper.

    See Also:
        :doc:`/how_to/notebooks/refine_noisy_boundaries` for grid-based
        cleanup workflows on real plate images.
        :doc:`/explanation/refinement_strategies` for a comparison of
        grid refinement approaches and grid inference methods.
    """

    smoothing_sigma: Annotated[float, TuneSpec(0.5, 5.0, log=True)] = 2.0
    min_peak_distance: Annotated[int | None, TuneSpec(5, 100, log=True)] = None
    peak_prominence: float | None = None
    edge_refinement: bool = True
    correlation_threshold: Annotated[float, TuneSpec(0.1, 0.6)] = 0.3
    selection_mode: Literal["dominant", "centered", "regularized"] = "dominant"
    split_merged: bool = False

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

    def _estimate_edges(self, binary_image: np.ndarray, axis: int, n_bins: int,
                        **kwargs: object) -> np.ndarray:  # type: ignore[override]
        """Estimate grid edges using sinusoidal cross-correlation.

        Overrides GridInferenceMixin._estimate_edges with a projection-based
        approach: generates a sine template matching expected colony periodicity,
        computes FFT-based normalized cross-correlation against the projection
        signal, and selects peaks from the correlation output. Rank-based
        (Spearman-style) correlation provides robustness to outliers and
        monotonic intensity transformations. The sinusoidal template and
        rank-transform steps are not from the gitter paper cited above.

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
    def _normalized_cross_correlation(signal: np.ndarray,
                                      template: np.ndarray) -> np.ndarray:
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
