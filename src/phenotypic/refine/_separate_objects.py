"""Grid-based watershed segmentation for separating touching colonies.

Refines detected colonies by separating touching/merged objects using watershed segmentation
seeded at grid intersection points inferred from colony layout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image

import gc
import numpy as np

from phenotypic.abc_ import ObjectRefiner
from phenotypic.tools_._grid_inference_mixin import GridInferenceMixin
from phenotypic.tools_.funcs_ import validate_operation_integrity


class SeparateObjects(GridInferenceMixin, ObjectRefiner):
    """Separate touching/merged colonies using grid-based watershed segmentation.

    SeparateObjects segments colonies by inferring grid structure from colony layout
    and using grid intersection points as seed markers for watershed segmentation.
    This region-growing approach effectively separates touching colonies that may have
    been merged by thresholding while preserving individual colony boundaries within
    each grid cell.

    Args:
        smoothing_sigma: Gaussian smoothing standard deviation for row/column intensity
            profiles during grid inference. Default 2.0. Higher values smooth noise but
            may merge adjacent peaks.
        min_peak_distance: Minimum pixel distance between detected grid peaks. If None
            (default), automatically estimated from expected colony spacing.
        peak_prominence: Minimum prominence threshold for peak detection. If None (default),
            auto-calculated as 10% of signal range.
        erosion_radius: Radius for morphological erosion footprint when computing object
            boundaries. Default 1. Controls how aggressively boundaries are detected.

    Returns:
        Image: Input image with refined objmap where touching colonies are separated
            into distinct regions. objmask is automatically updated. All image data
            (rgb, gray, enh_gray) remain unchanged.

    Raises:
        ValueError: If grid inference fails or image lacks detection results (no objmap).

    **Use cases**

    - **Touching/overlapping colonies:** Separates colonies that touch or overlap
      where simple thresholding merges them into a single detection.
    - **High-throughput screening:** Grid-based seeding works well for arrayed formats
      (96-well, 384-well, pinned cultures) where colonies align to known positions.
    - **Post-detection refinement:** Apply after ObjectDetector (e.g., Otsu, Hysteresis)
      when detections contain merged objects that need individualization.
    - **Variable colony sizes:** Watershed respects local intensity gradients, adapting
      to different colony sizes within the same grid better than geometric methods.

    **Limitations**

    - Assumes regular grid geometry; fails on irregular colony spacing or missing positions.
    - Grid inference on regular Image is less accurate than explicit GridImage specification.
    - Requires colonies to cluster near grid centers; fails if colonies straddle grid boundaries.
    - Assumes at least some colonies present; completely empty plates will fail grid inference.
    - Best for circular/regular colonies; less effective for highly irregular morphologies.
    - May over-segment noisy detections with many spurious peaks if grid inference is poor.

    **Parameter effects on separation quality**

    - **smoothing_sigma:** Higher values improve robustness to noise in grid inference
      but may merge adjacent peaks, leading to incomplete separation. Lower values
      are more sensitive to noise. Set to 0 to disable smoothing (faster, less robust).
    - **erosion_radius:** Larger values make boundaries thicker, forcing more aggressive
      separation but potentially oversplitting single colonies. Smaller values preserve
      more natural boundaries but may fail on touching regions. Range: 1-3 typically works.
    - **min_peak_distance, peak_prominence:** Lower values detect more grid peaks; higher
      values are more selective. Auto-tuning usually works well; manual adjustment helps
      if grid inference fails on unusual plate layouts.

    Examples:
        Basic usage with detected image to separate touching colonies:

        >>> from phenotypic import Image
        >>> from phenotypic.detect import OtsuDetector
        >>> from phenotypic.refine import SeparateObjects
        >>> from phenotypic.data import load_synth_plate
        >>>
        >>> # Load and detect
        >>> image = load_synth_plate()
        >>> detector = OtsuDetector()
        >>> detected = detector.apply(image)
        >>>
        >>> # Separate touching colonies using grid-based watershed
        >>> separator = SeparateObjects(smoothing_sigma=2.0, erosion_radius=1)
        >>> separated = separator.apply(detected)
        >>>
        >>> print(f"Before: {detected.objmap[:].max()} colonies")
        >>> print(f"After:  {separated.objmap[:].max()} colonies (touching separated)")

        Integration into full processing pipeline with grid inference:

        >>> from phenotypic import Image, ImagePipeline
        >>> from phenotypic.enhance import GaussianBlur, CLAHE
        >>> from phenotypic.detect import OtsuDetector
        >>> from phenotypic.refine import SeparateObjects, SmallObjectRemover
        >>> from phenotypic.measure import MeasureShape
        >>>
        >>> # Build pipeline with detection, separation, and measurement
        >>> pipeline = ImagePipeline([
        ...     GaussianBlur(sigma=1.5),
        ...     CLAHE(clip_limit=2.0),
        ...     OtsuDetector(),
        ...     SeparateObjects(smoothing_sigma=2.0, erosion_radius=1),
        ...     SmallObjectRemover(min_size=50),
        ...     MeasureShape()
        ... ])
        >>>
        >>> # Process image (grid inferred automatically)
        >>> image = Image.imread("plate.jpg")
        >>> result = pipeline.apply(image)
        >>>
        >>> print(f"Separated and measured: {len(result.objects)} individual colonies")
    """

    def __init__(
            self,
            smoothing_sigma: float = 2.0,
            min_peak_distance: int | None = None,
            peak_prominence: float | None = None,
            erosion_radius: int = 1,
    ):
        """Initialize SeparateObjects with grid inference and watershed parameters.

        Args:
            smoothing_sigma: Gaussian smoothing sigma for intensity profiles.
            min_peak_distance: Minimum distance between grid peaks.
            peak_prominence: Minimum prominence for peak detection.
            erosion_radius: Radius for erosion footprint in boundary detection.
        """
        super().__init__()
        self.smoothing_sigma = smoothing_sigma
        self.min_peak_distance = min_peak_distance
        self.peak_prominence = peak_prominence
        self.erosion_radius = erosion_radius

    @validate_operation_integrity("image.rgb", "image.gray", "image.enh_gray")
    def apply(self, image: Image, inplace: bool = False) -> Image:
        return super().apply(image=image, inplace=inplace)

    @staticmethod
    def _make_elevation_map(objmask: np.ndarray) -> np.ndarray:
        """Compute elevation map for watershed segmentation.

        Args:
            objmask: Binary mask of detected objects.

        Returns:
            Elevation map where higher values = colony centers, lower values = boundaries.

        Note:
            Current implementation uses distance transform (inverted for watershed).
            This method is designed to be easily modified for alternative elevation
            strategies without changing the main _operate() logic.
        """
        from scipy.ndimage import distance_transform_edt

        # Distance transform: higher values at colony centers
        distance = distance_transform_edt(objmask)
        return -distance  # Negate for watershed (wants valleys at peaks)

    @staticmethod
    def _make_boundary_mask(objmask: np.ndarray, erosion_radius: int) -> np.ndarray:
        """Compute boundary mask from object mask via erosion.

        Args:
            objmask: Binary mask of detected objects.
            erosion_radius: Radius for erosion footprint.

        Returns:
            Binary mask where True indicates object boundaries.
        """
        from skimage import morphology

        footprint = morphology.disk(erosion_radius)
        eroded = morphology.erosion(objmask, footprint)
        boundaries = objmask & ~eroded
        return boundaries

    @staticmethod
    def _create_grid_seeds(
            objmask: np.ndarray,
            row_peaks: np.ndarray,
            col_peaks: np.ndarray,
    ) -> np.ndarray:
        """Create watershed seed markers at grid intersection points.

        Args:
            objmask: Binary mask of detected objects.
            row_peaks: Row positions for grid intersections.
            col_peaks: Column positions for grid intersections.

        Returns:
            Integer marker array with unique labels at valid seed positions.

        Note:
            Seeds are only placed where objmask is True, automatically
            handling empty grid cells. Out-of-bounds positions are skipped.
        """
        markers = np.zeros_like(objmask, dtype=np.int32)
        label_id = 1

        height, width = objmask.shape

        for row_pos in row_peaks:
            # Skip out-of-bounds rows
            if row_pos < 0 or row_pos >= height:
                continue
            for col_pos in col_peaks:
                # Skip out-of-bounds columns
                if col_pos < 0 or col_pos >= width:
                    continue
                # Only place seed if there's an object at this position
                if objmask[row_pos, col_pos]:
                    markers[row_pos, col_pos] = label_id
                    label_id += 1

        return markers

    @staticmethod
    def _operate(
            image: Image,
            smoothing_sigma: float,
            min_peak_distance: int | None,
            peak_prominence: float | None,
            erosion_radius: int,
    ) -> Image:
        """Separate touching colonies using grid-based watershed segmentation.

        This method infers grid structure from colony patterns and uses grid intersection
        points as seed markers for watershed segmentation, effectively separating touching
        colonies into distinct regions.

        Args:
            image: Image object with existing objmask and objmap from detection.
            smoothing_sigma: Gaussian smoothing for grid inference.
            min_peak_distance: Minimum distance between peaks.
            peak_prominence: Minimum prominence for peaks.
            erosion_radius: Radius for erosion in boundary detection.

        Returns:
            Image: Modified image with separated objmap and updated objmask.
        """
        from skimage import segmentation

        objmask = image.objmask[:]

        # Infer grid dimensions from colony patterns
        nrows, ncols = SeparateObjects._infer_grid_shape(objmask)

        # Estimate grid edges along both axes
        row_edges = SeparateObjects._estimate_edges(
                objmask,
                axis=0,
                n_bins=nrows,
                smoothing_sigma=smoothing_sigma,
                min_peak_distance=min_peak_distance,
                peak_prominence=peak_prominence,
        )
        col_edges = SeparateObjects._estimate_edges(
                objmask,
                axis=1,
                n_bins=ncols,
                smoothing_sigma=smoothing_sigma,
                min_peak_distance=min_peak_distance,
                peak_prominence=peak_prominence,
        )

        # Calculate peak positions as midpoints between edges
        row_peaks = ((row_edges[:-1] + row_edges[1:]) / 2).astype(int)
        col_peaks = ((col_edges[:-1] + col_edges[1:]) / 2).astype(int)

        # Create seeds at grid intersections (handles empty cells automatically)
        markers = SeparateObjects._create_grid_seeds(objmask, row_peaks, col_peaks)

        # Compute boundary mask using helper
        boundaries = SeparateObjects._make_boundary_mask(objmask, erosion_radius)

        # Compute elevation map (easy to swap strategy later)
        elevation = SeparateObjects._make_elevation_map(boundaries)

        # Watershed segmentation
        objmap = segmentation.watershed(
                elevation,
                markers=markers,
                mask=objmask,  # Limit to detected regions only
        )

        # Convert to proper dtype
        if objmap.dtype != image._OBJMAP_DTYPE:
            objmap = objmap.astype(image._OBJMAP_DTYPE)

        # Update image with separated map
        image.objmap[:] = objmap
        image.objmap.relabel(connectivity=1)

        gc.collect()

        return image
