from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image

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
    """Grid-based colony detector using sinusoidal cross-correlation peak detection.

    SinePeakDetector identifies colonies in gridded plate images by generating a
    sinusoidal template matching expected colony periodicity, computing FFT-based
    normalized cross-correlation against projection signals, and selecting peaks
    from the correlation output. This implements a gitter-faithful approach
    (Wagih & Parts, 2014) that is more robust to irregular colony intensities
    than simple peak finding, because rank-based correlation is insensitive to
    outliers and monotonic intensity transformations.

    The detector builds on the RoundPeaksDetector workflow (threshold, label,
    grid assignment) but replaces the mixin's static ``_estimate_edges`` with an
    instance method that uses sinusoidal cross-correlation for edge estimation.

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
            cross-correlation. Default 2.0. Higher values smooth noise but may
            merge peaks.

        min_peak_distance: Minimum pixel distance between detected peaks. If None,
            automatically estimated from grid dimensions.

        peak_prominence: Minimum prominence threshold for peak detection. If None,
            auto-calculated as 0.1 * signal range. Higher values are more selective.

        edge_refinement: If True (default), refine grid edges using local intensity
            profiles for improved accuracy.

        correlation_threshold: Minimum normalized cross-correlation value for a
            peak to be considered valid. Default 0.3. Correlation values below this
            threshold are zeroed before peak detection. Lower values accept weaker
            matches; higher values are more selective.

    Attributes:
        thresh_method, subtract_background, remove_noise, footprint_radius,
        noise_radius, smoothing_sigma, min_peak_distance, peak_prominence,
        edge_refinement, correlation_threshold

    Returns:
        Image: Input image with objmask (binary colony mask) and objmap (labeled
        colonies assigned to grid cells) set.

    Raises:
        ValueError: If invalid thresholding method specified.

    **Use cases**

    - **Gridded plate images:** Colonies arranged in regular arrays (96-well, 384-well
      plates, pinned cultures). Sinusoidal cross-correlation exploits periodic structure
      more robustly than direct peak finding.
    - **Variable colony intensity:** Rank-based (Spearman) correlation is insensitive
      to outlier colonies or uneven growth, making this detector suitable for plates
      with heterogeneous colony sizes.
    - **Batch processing:** Efficient FFT-based correlation enables high-throughput
      analysis without manual grid specification (though GridImage with explicit
      dimensions is more accurate).

    **Limitations**

    - Grid inference from binary mask alone is less accurate than explicit GridImage
      specification. For best results, use with GridImage when grid parameters known.
    - Assumes regular grid geometry. Works poorly with irregular colony spacing or
      missing grid positions.
    - Best for yeast-like morphologies. Less suitable for filamentous, spreading, or
      irregular colony shapes.
    - Slightly higher computational cost than RoundPeaksDetector due to FFT-based
      cross-correlation.

    **Parameter effects on colony detection**

    - **thresh_method:** Different histogram assumptions (Otsu=variance, mean=simple,
      local=adaptive). Affects mask quality and downstream correlation.
    - **subtract_background, remove_noise:** Remove preprocessing artifacts (vignetting,
      dust, noise) that can create spurious correlation peaks.
    - **smoothing_sigma:** Balances noise robustness vs peak resolution. Higher values
      smooth noise but may merge adjacent colonies.
    - **correlation_threshold:** Controls sensitivity to weak matches. Lower values
      detect more peaks (including false positives); higher values are more selective
      but may miss faint colonies.

    Examples:
        Basic grid detection with sinusoidal cross-correlation::

            from phenotypic import Image
            from phenotypic.detect import SinePeakDetector

            plate = Image.imread("plate_grid.jpg")
            detector = SinePeakDetector()
            detected = detector.apply(plate)
            num_colonies = detected.objects.count
            print(f"Detected {num_colonies} colonies in grid")

        Pipeline with preprocessing for noisy plate images::

            from phenotypic import ImagePipeline
            from phenotypic.enhance import GaussianBlur, CLAHE
            from phenotypic.detect import SinePeakDetector

            pipeline = ImagePipeline([
                GaussianBlur(sigma=1.5),
                CLAHE(clip_limit=2.0),
                SinePeakDetector(
                    thresh_method='otsu',
                    smoothing_sigma=2.0,
                    correlation_threshold=0.25,
                )
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
            correlation_threshold: float = 0.3,
    ):
        """
        Initialize the SinePeakDetector with specified parameters.

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
                profiles before cross-correlation. Set to 0 to disable smoothing.
            min_peak_distance: Minimum allowed distance between detected peaks.
                If None, automatically estimated from grid dimensions.
            peak_prominence: Minimum prominence required for peak detection.
                If None, automatically calculated as 0.1 * signal range.
            edge_refinement: If True, refine grid edges using weighted intensity
                profiles for improved accuracy.
            correlation_threshold: Minimum normalized cross-correlation value for
                a peak to be considered valid. Default 0.3. Values below this
                threshold are zeroed before peak detection.
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
        self.correlation_threshold = correlation_threshold

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
