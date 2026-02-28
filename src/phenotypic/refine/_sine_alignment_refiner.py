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
    """Refines detected objects by keeping only grid-aligned colonies using sinusoidal cross-correlation.

    SineAlignmentRefiner filters detection results to retain only one dominant object per
    grid cell, using FFT-based normalized cross-correlation against a sinusoidal template
    for grid edge estimation. This implements a gitter-faithful approach (Wagih & Parts,
    2014) that is more robust to irregular colony intensities than the simple peak-finding
    used by GridAlignmentRefiner, because rank-based (Spearman) correlation is insensitive
    to outliers and monotonic intensity transformations.

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
        correlation_threshold: Minimum normalized cross-correlation value for a peak to
            be considered valid. Default 0.3. Correlation values below this threshold are
            zeroed before peak detection. Lower values accept weaker matches; higher
            values are more selective.
        selection_mode: Strategy for choosing one object per grid cell. ``"dominant"``
            (default) keeps the largest object by pixel count. ``"centered"`` keeps
            the object whose centroid is closest to the cell center. ``"regularized"``
            uses a two-pass approach that fits a global regular-grid model from median
            row/column centroids, then re-selects per cell. Best for pinned arrays.

    Returns:
        Image: Input image with filtered objmap containing only grid-aligned objects.
            objmask is automatically updated to match refined objmap. All image data
            (rgb, gray, detect_mat) remain unchanged.

    Raises:
        ValueError: If grid inference fails or image lacks detection results (no objmap).

    **Use cases**

    - **Gridded plate images:** Remove off-grid noise and dust for accurate well-based
      phenotyping on 96-well, 384-well, or pinned culture formats.
    - **Post-detection cleanup:** Apply after ObjectDetector when detections contain
      spurious off-grid objects or artifacts. More robust than GridAlignmentRefiner
      when colony intensities are heterogeneous.
    - **Explicit grid enforcement:** Use with GridImage to snap detections to known
      well positions when exact grid coordinates are available.
    - **Variable colony intensity:** Rank-based correlation handles plates with
      heterogeneous colony sizes or uneven growth better than direct peak finding.

    **Limitations**

    - Assumes regular grid geometry; fails on irregular colony spacing or missing positions.
    - Grid inference on regular Image is less accurate than explicit GridImage specification.
    - Requires colonies to cluster within grid cells; fails if colonies straddle boundaries.
    - Slightly higher computational cost than GridAlignmentRefiner due to FFT-based
      cross-correlation.
    - Best for yeast-like circular colonies; less suitable for filamentous or irregular
      morphologies that may not align cleanly with grid.

    **Parameter effects on grid detection**

    - **smoothing_sigma:** Higher values improve robustness to noise but may merge
      adjacent peaks. Set to 0 to disable smoothing (faster, less robust).
    - **edge_refinement:** When True, places grid edges at valleys between colonies
      rather than fixed positions, improving accuracy for uneven plate lighting.
    - **correlation_threshold:** Controls sensitivity to weak matches. Lower values
      detect more peaks (including false positives); higher values are more selective
      but may miss faint colonies.
    - **min_peak_distance, peak_prominence:** Lower values detect more peaks (find more
      grid positions); higher values are more selective. Auto-tuning usually works well.

    Examples:
        Basic usage with GridImage and explicit grid dimensions:

        >>> from phenotypic import GridImage
        >>> from phenotypic.detect import OtsuDetector
        >>> from phenotypic.refine import SineAlignmentRefiner
        >>> from phenotypic.data import load_synth_yeast_plate
        >>>
        >>> # Load gridded plate image
        >>> grid_image = load_synth_yeast_plate()  # Returns GridImage with 8x12 grid
        >>> detector = OtsuDetector()
        >>> detected = detector.apply(grid_image)
        >>>
        >>> # Refine to keep only grid-aligned objects (sine cross-correlation)
        >>> refiner = SineAlignmentRefiner()
        >>> refined = refiner.apply(detected)
        >>>
        >>> print(f"Before: {detected.objmap[:].max()} objects")
        >>> print(f"After:  {refined.objmap[:].max()} objects (grid-aligned)")

        Integration into full processing pipeline with grid inference:

        >>> from phenotypic import Image, ImagePipeline
        >>> from phenotypic.enhance import GaussianBlur, CLAHE
        >>> from phenotypic.detect import RoundPeaksDetector
        >>> from phenotypic.refine import SineAlignmentRefiner
        >>> from phenotypic.measure import MeasureShape
        >>>
        >>> # Build pipeline with detection and sine-based refinement
        >>> pipeline = ImagePipeline([
        ...     GaussianBlur(sigma=1.5),
        ...     CLAHE(clip_limit=2.0),
        ...     RoundPeaksDetector(smoothing_sigma=2.0),
        ...     SineAlignmentRefiner(correlation_threshold=0.25),
        ...     MeasureShape()
        ... ])
        >>>
        >>> # Process image (no explicit grid needed)
        >>> image = Image.imread("noisy_plate.jpg")
        >>> result = pipeline.apply(image)
        >>>
        >>> print(f"Cleaned colonies: {len(result.objects)}")

    References:
        Wagih, O. and Parts, L. (2014). gitter: a robust and accurate method for
        quantification of colony sizes from plate images. G3 (Bethesda), 4(3), 547-552.
    """

    def __init__(
            self,
            smoothing_sigma: float = 2.0,
            min_peak_distance: int | None = None,
            peak_prominence: float | None = None,
            edge_refinement: bool = True,
            correlation_threshold: float = 0.3,
            selection_mode: Literal["dominant", "centered", "regularized"] = "dominant",
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
        """
        super().__init__()
        self.smoothing_sigma = smoothing_sigma
        self.min_peak_distance = min_peak_distance
        self.peak_prominence = peak_prominence
        self.edge_refinement = edge_refinement
        self.correlation_threshold = correlation_threshold
        self.selection_mode = selection_mode

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
            objmap, row_edges, col_edges, self.selection_mode, image._OBJMAP_DTYPE
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
