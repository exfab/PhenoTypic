from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import gc
from typing import Literal

import numpy as np
import scipy.ndimage as ndimage
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, fftconvolve, medfilt
from scipy.stats import rankdata

from phenotypic.abc_ import ObjectDetector
from phenotypic.tools_.mixin import GridInferenceMixin
import skimage.filters as filters
import skimage.morphology as morphology


class SinePeakDetector(GridInferenceMixin, ObjectDetector):
    """Detect colonies on gridded plates using sinusoidal cross-correlation peak finding.

    Generate a sinusoidal template matching the expected colony periodicity,
    compute FFT-based rank (Spearman) cross-correlation against row and
    column projection signals, and select peaks from the correlation output
    to locate grid positions. Rank-based correlation is insensitive to
    outlier colonies and monotonic intensity transformations, making this
    more robust than direct peak finding on plates with heterogeneous
    growth. For a full comparison see
    :doc:`/explanation/detection_strategies_compared`.

    Best For:
        - Gridded plates (96-well, 384-well, pinned arrays) where colonies
          are arranged in a regular periodic pattern.
        - Plates with heterogeneous colony sizes or uneven growth where
          rank-based correlation outperforms intensity-based peak finding.
        - High-throughput batch processing of arrayed plates without
          manual grid specification.
        - Plates where a small number of empty or very faint wells would
          mislead direct intensity peak detection.

    Consider Also:
        - :class:`RoundPeaksDetector` for a simpler grid detector when
          colony intensities are uniform and direct peak finding suffices.
        - :class:`OtsuDetector` when colonies are not gridded and a global
          threshold is appropriate.
        - :class:`RankOtsuDetector` when spatial illumination variation is
          the primary challenge rather than grid localisation.

    Args:
        thresh_method: Thresholding method for binary mask creation.
            Accepted values: ``'otsu'`` (default), ``'mean'``, ``'local'``,
            ``'triangle'``, ``'minimum'``, ``'isodata'``, ``'li'``.
            ``'otsu'`` works well for most standardised setups; ``'local'``
            adapts to spatial illumination gradients. Default: ``'otsu'``.
        subtract_background: Apply white tophat transform to remove uneven
            illumination before thresholding. Default: True.
        remove_noise: Apply morphological opening to remove small noise
            artefacts from the binary mask. Default: True.
        footprint_width: Width in pixels for the background subtraction
            kernel. When a GridImage is provided, an adaptive kernel sized
            to 1.5x colony spacing is used instead, making this a fallback
            for plain Image inputs. Typical range: 4--20. Default: 6.
        noise_radius: Radius of the diamond structuring element for
            morphological noise removal. Typical range: 1--3. Default: 1.
        smoothing_sigma: Gaussian standard deviation for smoothing
            row/column intensity profiles before cross-correlation. Higher
            values suppress noise but may merge adjacent peaks. Set to 0 to
            disable. Typical range: 0.0--5.0. Default: 2.0.
        min_peak_distance: Minimum pixel distance between detected peaks.
            When ``None``, automatically estimated from grid dimensions.
            Default: None.
        peak_prominence: Minimum prominence for peak detection. When
            ``None``, auto-calculated as 0.1 × signal range. Higher values
            are more selective. Default: None.
        edge_refinement: Refine grid edges using weighted local intensity
            profiles for improved accuracy. Default: True.
        correlation_threshold: Minimum normalised cross-correlation score
            for a sinusoidal template peak to be accepted as a valid grid
            position. Lower values recover grid cells with weak colony
            signal (sparse plates); higher values reject false grid
            positions at the cost of missing faint wells. Typical range:
            0.1--0.5. Default: 0.3.
        selection_mode: Strategy for choosing one object per grid cell.
            ``"dominant"`` keeps the largest object by pixel count.
            ``"centered"`` keeps the object whose centroid is closest to
            the cell centre. ``"regularized"`` fits a global regular-grid
            model from median centroids, then re-selects per cell — best
            for pinned arrays. Default: ``"dominant"``.
        split_merged: Pre-split merged colonies spanning multiple grid
            cells using EDT watershed before grid assignment. Set to False
            when colonies are well-separated. Default: True.

    Returns:
        Image: Input image with ``objmask`` set to binary mask and
        ``objmap`` set to labeled connected components with one label per
        grid cell.

    Raises:
        ValueError: If ``thresh_method`` is not one of the accepted
            values.

    References:
        [1] O. Wagih and L. Parts, "gitter: a robust and accurate method
        for quantification of colony sizes from plate images,"
        *G3 (Bethesda)*, vol. 4, no. 3, pp. 547--552, 2014.
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
    footprint_width: int = 6
    noise_radius: int = 1
    smoothing_sigma: float = 2.0
    min_peak_distance: int | None = None
    peak_prominence: float | None = None
    edge_refinement: bool = True
    correlation_threshold: float = 0.3
    selection_mode: Literal["dominant", "centered", "regularized"] = "dominant"
    split_merged: bool = True

    @staticmethod
    def _round_odd(n: int) -> int:
        """Round to nearest odd integer (minimum 3)."""
        n = max(n, 3)
        return n if n % 2 == 1 else n + 1

    def _operate(self, image: Image) -> Image:
        """
        Detect colonies using sinusoidal cross-correlation grid estimation.

        This method performs the core detection workflow:
        1. Extract grid dimensions (if GridImage)
        2. Threshold the detection matrix with adaptive kernel sizing
        3. Remove noise if requested
        4. Label connected components
        5. Determine or estimate grid edges (via sinusoidal cross-correlation)
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
            # Estimate edges using sinusoidal cross-correlation on row/col sums
            nrows, ncols = self._infer_grid_shape(objmask)
            self._log_memory_usage(f"inferred grid shape: {nrows}x{ncols}")

            row_edges = self._estimate_edges(objmask, axis=0, n_bins=nrows)
            col_edges = self._estimate_edges(objmask, axis=1, n_bins=ncols)
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
            dim = self._round_odd(self.footprint_width * 2)
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
                        self.footprint_width * 2 + 1, 3
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

    def _estimate_edges(self, binary_image: np.ndarray, axis: int, n_bins: int, **kwargs: object) -> np.ndarray:  # type: ignore[override]
        """Estimate grid edges using sinusoidal cross-correlation.

        Overrides GridInferenceMixin._estimate_edges with a projection-based
        approach: generates a sine template matching expected colony periodicity,
        computes FFT-based normalized cross-correlation against the projection
        signal, and selects peaks from the correlation output. Rank-based
        (Spearman-style) correlation provides robustness to outliers and
        monotonic intensity transformations. The sinusoidal template and
        rank-transform steps are project-specific extensions rather than
        steps from the gitter paper cited above.

        Args:
            binary_image: Binary mask of detected colonies.
            axis: Direction for edge detection (0 for row edges, 1 for column edges).
            n_bins: Expected number of grid bins (rows or columns).

        Returns:
            np.ndarray: Array of edge positions including image borders.
                Length is n_bins + 1.

        Note:
            Unlike the mixin's static ``_estimate_edges``, this instance method
            reads ``smoothing_sigma``, ``min_peak_distance``, ``peak_prominence``,
            and ``correlation_threshold`` from ``self``.
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


SinePeakDetector.apply.__doc__ = SinePeakDetector._operate.__doc__
