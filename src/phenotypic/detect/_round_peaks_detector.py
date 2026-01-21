from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image

import gc
from typing import Literal

import numpy as np
import scipy.ndimage as ndimage
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

from phenotypic.abc_ import ObjectDetector
from phenotypic.tools_._grid_inference_mixin import GridInferenceMixin
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

        footprint_width: Radius in pixels for morphological kernels (noise removal,
            background subtraction). Default 6. Larger values remove larger noise.

        smoothing_sigma: Gaussian smoothing of row/column intensity profiles before
            peak detection. Default 2.0. Higher values smooth noise but may merge peaks.

        min_peak_distance: Minimum pixel distance between detected peaks. If None,
            automatically estimated from grid dimensions.

        peak_prominence: Minimum prominence threshold for peak detection. If None,
            auto-calculated as 0.1 * signal range. Higher values are more selective.

        edge_refinement: If True (default), refine grid edges using local intensity
            profiles for improved accuracy.

    Attributes:
        thresh_method, subtract_background, remove_noise, footprint_radius,
        smoothing_sigma, min_peak_distance, peak_prominence, edge_refinement

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
            smoothing_sigma: float = 2.0,
            min_peak_distance: int | None = None,
            peak_prominence: float | None = None,
            edge_refinement: bool = True,
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
            footprint_width: width in pixels for morphological operations.
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
        self.footprint_radius = footprint_width
        self.remove_noise = remove_noise
        self.smoothing_sigma = smoothing_sigma
        self.min_peak_distance = min_peak_distance
        self.peak_prominence = peak_prominence
        self.edge_refinement = edge_refinement

    def _operate(self, image: Image) -> Image:
        """
        Detect colonies in the image using the gitter algorithm.

        This method performs the _core detection workflow:
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
        self._log_memory_usage("thresholding")

        if self.remove_noise:
            objmask = morphology.binary_opening(
                    objmask, morphology.diamond(radius=self.footprint_radius)
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
            nrows, ncols = image.nrows, image.ncols
        else:
            nrows = ncols = None
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
        kernel = morphology.footprint_rectangle(
                (self.footprint_radius * 2, self.footprint_radius * 2)
        )
        enh_matrix = matrix.copy()  # Work on a copy to avoid modifying input

        # Subtract background using white tophat to handle uneven illumination
        if self.subtract_background:
            tophat_res = morphology.white_tophat(enh_matrix, kernel)
            enh_matrix = enh_matrix - tophat_res

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
