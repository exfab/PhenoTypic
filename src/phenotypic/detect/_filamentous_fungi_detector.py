from __future__ import annotations
from typing import TYPE_CHECKING, Union
import numpy as np
import gc

if TYPE_CHECKING:
    from phenotypic import Image, ImagePipeline  # type: ignore

from scipy.ndimage import center_of_mass
from skimage.morphology import disk, erosion
from skimage.segmentation import watershed
from skimage.measure import label

from phenotypic.abc_ import ObjectDetector
from phenotypic import ImagePipeline
from phenotypic.enhance import (
    MedianFilter,
    BM3DDenoiser,
    BayesShrinkEnhancer,
    CLAHE,
    GrayOpening,
    GaussianBlur,
    GaussianSubtract
)

from phenotypic.detect import OtsuDetector, RoundPeaksDetector, SecondaryOtsuDetector
from phenotypic.refine import MaskOpener, GridSectionLargest


class FilamentousFungiDetector(ObjectDetector):
    """Detects and separates filamentous fungi using two-stage detection and watershed.

    FilamentousFungiDetector uses two detection strategies to segment filamentous fungi:
    (1) center_detector identifies compact fungal centers/nuclei, (2) overall_detector captures
    the complete fungal structure including spreading hyphae. The detector combines these
    by filtering centers to those overlapping with the overall structure, then uses the
    filtered centers as seed markers for watershed segmentation with boundary-constrained
    elevation map. This approach effectively separates touching filamentous fungi that have
    irregular, spreading morphology.

    Args:
        center_detector: ObjectDetector or ImagePipeline that identifies fungal centers/nuclei.
            Should produce small, compact regions at center points. Examples: OtsuDetector(),
            RoundPeaksDetector(), or preprocessing pipeline with GaussianBlur() + detector.

        overall_detector: ObjectDetector or ImagePipeline that captures complete fungal
            structures including hyphae and spreading edges. Should produce full fungal body
            masks. Examples: TriangleDetector(), CannyDetector(), or custom detector with
            lower threshold than center_detector.

        erosion_radius: Radius in pixels for morphological erosion to compute boundaries
            (default 1). Boundaries computed as: objmask - erosion(objmask, disk(radius)).
            Controls thickness of boundary region.

        boundary_cost: Cost assigned to boundaries in elevation map (default 1e6). Higher
            values create stronger barriers preventing watershed from crossing boundaries.
            Controls watershed constraint strength.

        compactness: Watershed compactness parameter (default 0.0). Usually 0 for
            boundary-constrained watershed to allow free region growing constrained only
            by boundaries. Higher values enforce more compact segments.

        connectivity: Connectivity for region labeling (1=4-connected, 2=8-connected,
            default 1). Controls how adjacent pixels merge into regions.

    Returns:
        Image: Input image with objmask (binary mask) and objmap (labeled fungi) set.
            Each labeled fungus is separated by watershed segmentation based on filtered
            centers. objmask is True for all fungal pixels within watershed regions.

    Raises:
        TypeError: If center_detector or overall_detector are not ObjectDetector or
            ImagePipeline instances.
        ValueError: If no centers detected, no overall structure detected, or no centers
            overlap with overall structure after filtering.
        RuntimeError: If watershed segmentation fails to produce valid regions.

    **Use cases**

    - **Filamentous fungal colonies:** Irregular spreading structures that require separate
      center and overall detection strategies for accurate segmentation.
    - **Touching/overlapping fungi:** Watershed separation using center seeds effectively
      separates filaments in close contact where simple thresholding merges them.
    - **Variable morphology:** Two-stage approach adapts to fungi with varying sizes,
      growth patterns, and hyphae density.
    - **Grid-based fungal cultures:** Works on GridImage with multiple wells containing
      filamentous fungi; can be integrated into ImagePipeline.
    - **High-throughput fungal phenotyping:** Enables batch processing of fungal plate
      images with consistent separation quality.

    **Limitations**

    - Requires two compatible detectors: centers must overlap significantly with overall
      structure, or ValueError is raised. Tuning both detectors is necessary.
    - More computationally expensive than single-detector methods due to two detection
      passes and watershed segmentation.
    - Watershed quality depends on center detection accuracy: missing centers cause under-
      segmentation; spurious centers cause over-segmentation.
    - May over-segment if centers are too numerous or too close together (smaller than
      erosion_radius).
    - Less suitable for circular, yeast-like colonies; use WatershedDetector instead
      for round morphologies.

    **Parameter effects on separation quality**

    - **erosion_radius:** Larger radius creates thicker boundary barriers (stronger
      constraint), may prevent watershed from reaching small hyphae. Smaller radius allows
      finer separation but weaker boundaries. Typical range: 1-3 pixels.

    - **boundary_cost:** Higher values strongly prevent watershed from crossing boundaries
      (more conservative separation). Lower values allow more flexible region growing
      (less conservative). Typical range: 1e4 - 1e8.

    - **compactness:** For boundary-constrained watershed, keep at 0.0 (default) to allow
      free region growing. Non-zero values enforce compactness which may conflict with
      boundary constraints.

    - **connectivity:** Use 1 (4-connected, default) for conservative labeling that respects
      diagonal gaps. Use 2 (8-connected) to merge diagonally-touching regions.

    Examples:
        Detect and separate filamentous fungi with center and overall detection:

        >>> from phenotypic.detect import FilamentousFungiDetector, OtsuDetector, TriangleDetector
        >>> from phenotypic.data import load_synth_plate
        >>>
        >>> # Create detector: centers detected via OtsuDetector, overall via TriangleDetector
        >>> detector = FilamentousFungiDetector(
        ...     center_detector=OtsuDetector(ignore_zeros=True),
        ...     overall_detector=TriangleDetector(),
        ...     erosion_radius=1,
        ...     boundary_cost=1e6
        ... )
        >>>
        >>> # Note: load_synth_plate returns circular colonies; example is illustrative
        >>> image = load_synth_plate()
        >>> result = detector.apply(image)
        >>> num_fungi = result.objmap[:].max()
        >>> print(f"Detected and separated {num_fungi} fungal colonies")

        Integration in processing pipeline with preprocessing and refinement:

        >>> from phenotypic import ImagePipeline
        >>> from phenotypic.enhance import GaussianBlur, CLAHE
        >>> from phenotypic.refine import SmallObjectRemover
        >>> from phenotypic.data import load_synth_plate
        >>>
        >>> # Build pipeline with enhancement, two-stage fungi detection, and cleanup
        >>> pipeline = ImagePipeline([
        ...     GaussianBlur(sigma=1.5),
        ...     CLAHE(clip_limit=2.0),
        ...     FilamentousFungiDetector(
        ...         center_detector=ImagePipeline([
        ...             GaussianBlur(sigma=0.5),
        ...             OtsuDetector()
        ...         ]),
        ...         overall_detector=TriangleDetector(),
        ...         erosion_radius=1
        ...     ),
        ...     SmallObjectRemover(min_size=100)
        ... ])
        >>>
        >>> image = load_synth_plate()
        >>> result = pipeline.apply(image)
        >>> print(f"Final separated fungi: {result.objmap[:].max()}")
    """
    __overall_pipe = ImagePipeline(
            ops=[
                MedianFilter(),
                BM3DDenoiser(),
                CLAHE(kernel_size=500),
                GrayOpening(),
                BayesShrinkEnhancer(
                        wavelet="db4",
                        mode="hard"
                ),
                OtsuDetector(ignore_zeros=True),
            ]
    )

    __center_pipe = ImagePipeline(
            ops=[
                GaussianBlur(sigma=5),
                GaussianSubtract(sigma=500),
                RoundPeaksDetector(thresh_method="triangle"),
                SecondaryOtsuDetector(ignore_zeros=True),
                MaskOpener(),
                GridSectionLargest(),
            ]
    )

    def __init__(
            self,
            center_detector: Union[ObjectDetector, 'ImagePipeline', None] = None,
            overall_detector: Union[ObjectDetector, 'ImagePipeline', None] = None,
            erosion_radius: int = 1,
            boundary_cost: float = 1e6,
            compactness: float = 0.0,
            connectivity: int = 1,
    ):
        super().__init__()

        # Type validation (allow None for serialization/deserialization)
        from phenotypic import ImagePipeline

        if center_detector is not None and not isinstance(center_detector,
                                                          (ObjectDetector,
                                                           ImagePipeline)):
            raise TypeError(
                    "center_detector must be an ObjectDetector or ImagePipeline instance, "
                    f"got {type(center_detector).__name__}"
            )
        if overall_detector is not None and not isinstance(overall_detector,
                                                           (ObjectDetector,
                                                            ImagePipeline)):
            raise TypeError(
                    "overall_detector must be an ObjectDetector or ImagePipeline instance, "
                    f"got {type(overall_detector).__name__}"
            )

        self.center_detector = center_detector if center_detector \
            else self.__center_pipe

        self.overall_detector = overall_detector if overall_detector \
            else self.__overall_pipe
        
        self.erosion_radius = erosion_radius
        self.boundary_cost = boundary_cost
        self.compactness = compactness
        self.connectivity = connectivity

    def _operate(self, image: 'Image') -> 'Image':
        """Detect and separate filamentous fungi using two-stage detection and watershed.

        Algorithm:
        1. Run center_detector to find fungal centers
        2. Run overall_detector to capture complete fungal structures
        3. Compute boundaries via morphological erosion
        4. Filter centers to keep only those overlapping with overall structure
        5. Convert filtered centers to watershed markers (centroid-based)
        6. Create boundary-constrained elevation map (flat with high cost at boundaries)
        7. Run watershed segmentation using markers and elevation map
        8. Set objmask and objmap with watershed results
        """

        from phenotypic import ImagePipeline

        # Validate that detectors are set before operation
        if self.center_detector is None:
            raise ValueError(
                    "center_detector is required but not set. "
                    "Provide a detector when creating FilamentousFungiDetector."
            )
        if self.overall_detector is None:
            raise ValueError(
                    "overall_detector is required but not set. "
                    "Provide a detector when creating FilamentousFungiDetector."
            )

        # Step 1: Apply center_detector
        if isinstance(self.center_detector, ImagePipeline):
            center_result = self.center_detector.apply(image, inplace=False,
                                                       reset=False)
        else:
            center_result = self.center_detector.apply(image, inplace=False)
        center_objmask = center_result.objmask[:]
        center_objmap = center_result.objmap[:]

        # Validate centers detected
        if center_objmap.max() == 0:
            raise ValueError(
                    "No centers detected by center_detector. Cannot perform watershed "
                    "separation. Try adjusting center_detector parameters or using a "
                    "different detection strategy."
            )

        self._log_memory_usage("after center detection")

        # Step 2: Apply overall_detector
        if isinstance(self.overall_detector, ImagePipeline):
            overall_result = self.overall_detector.apply(image, inplace=False,
                                                         reset=False)
        else:
            overall_result = self.overall_detector.apply(image, inplace=False)

        overall_objmask = overall_result.objmask[:]

        # Validate overall structure detected
        if overall_result.num_objects == 0:
            raise ValueError(
                    "No overall structure detected by overall_detector. Cannot create "
                    "watershed mask. Try adjusting overall_detector parameters."
            )

        self._log_memory_usage("after overall detection")

        # Step 3: Filter centers to keep only those overlapping with overall structure
        # Use _filter_centers_by_overlap with correct argument order
        filtered_center_objmask = self._filter_centers_by_overlap(
                center_mask=center_objmask, overall_mask=overall_objmask
        )
        overlap_objmap = label(filtered_center_objmask)
        num_centers = overlap_objmap.max()

        # Step 4: Compute boundaries via morphological erosion
        footprint = disk(self.erosion_radius)
        eroded = erosion(overall_objmask, footprint=footprint)
        boundaries = overall_objmask & ~eroded
        del eroded, footprint

        self._log_memory_usage("after boundary computation")

        # Validate at least one center remains after filtering
        if num_centers == 0:
            raise ValueError(
                    "No centers overlap with overall structure after filtering. "
                    "Check that center_detector and overall_detector are compatible "
                    "(detecting the same objects)."
            )

        self._log_memory_usage("after overlap filtering")

        # Step 5: Create watershed markers from filtered centers
        markers = self._create_markers_from_centers(overlap_objmap)

        self._log_memory_usage("after marker creation")

        # Step 6: Create boundary-constrained elevation map
        elevation = self._create_boundary_elevation_map(
                image.enh_gray.shape, boundaries, self.boundary_cost
        )

        self._log_memory_usage(
                "after elevation map creation", include_tracemalloc=True
        )

        # Step 7: Watershed segmentation
        # Debug: Check marker placement
        markers_in_mask = np.logical_and(markers > 0, overall_objmask)
        num_markers_in_mask = markers_in_mask.sum()

        objmap = watershed(
                image=elevation,
                markers=markers,
                mask=overall_objmask,
                compactness=self.compactness,
                connectivity=self.connectivity,
        )

        self._log_memory_usage(
                "after watershed segmentation",
                include_process=True,
                include_tracemalloc=True,
        )

        # Validate watershed produced valid result
        if objmap.max() == 0:
            raise RuntimeError(
                    f"Watershed segmentation produced empty result. "
                    f"Created {markers.max()} markers, {num_markers_in_mask} markers in mask. "
                    f"Elevation range: [{elevation.min():.2f}, {elevation.max():.2f}], "
                    f"boundaries pixels: {boundaries.sum()}, "
                    f"mask pixels: {overall_objmask.sum()}. "
                    f"Markers shape: {markers.shape}, Elevation shape: {elevation.shape}, "
                    f"Mask shape: {overall_objmask.shape}"
            )

        # Convert to proper dtype
        if objmap.dtype != image._OBJMAP_DTYPE:
            objmap = objmap.astype(image._OBJMAP_DTYPE)

        # Step 8: Set results
        # Note: do not relabel, it undoes watershed
        image.objmap[:] = objmap

        # Cleanup
        del center_result, overall_result, markers, elevation, boundaries
        gc.collect()

        self._log_memory_usage(
                "final cleanup and relabeling",
                include_process=True,
                include_tracemalloc=True,
        )

        return image

    @staticmethod
    def _filter_mask_by_overlap(mask_to_clean, reference_mask):
        """
        Retain only objects in mask_to_clean that overlap with reference_mask.

        Args:
            mask_to_clean (np.ndarray): Binary mask to filter (2D boolean or uint8)
            reference_mask (np.ndarray): Binary mask defining valid regions (2D boolean or uint8)

        Returns:
            np.ndarray: Filtered binary mask with same shape as mask_to_clean

        Raises:
            ValueError: If masks don't have compatible spatial overlap
        """
        # Label connected components in mask to clean
        labeled = label(mask_to_clean)

        # Handle potential size mismatch by finding overlapping region
        min_h = min(mask_to_clean.shape[0], reference_mask.shape[0])
        min_w = min(mask_to_clean.shape[1], reference_mask.shape[1])

        # Compute intersection in overlapping region
        intersection = labeled[:min_h, :min_w] * reference_mask[:min_h, :min_w]

        # Find which labels have overlap
        overlapping_labels = np.unique(intersection[intersection > 0])

        # Create output mask retaining only overlapping objects
        filtered_mask = np.isin(labeled, overlapping_labels)

        return filtered_mask.astype(mask_to_clean.dtype)

    @staticmethod
    def _filter_centers_by_overlap(
            center_mask: np.ndarray, overall_mask: np.ndarray
    ) -> np.ndarray:
        """Filter center mask to keep only centers overlapping with overall structure.

        Args:
            center_mask: Binary mask of detected centers (2D boolean or uint8)
            overall_mask: Binary mask of overall structure (2D boolean or uint8)

        Returns:
            Filtered binary mask containing only overlapping centers

        Raises:
            ValueError: If masks have incompatible shapes
        """

        # Label centers
        labeled_centers = label(center_mask)  # type: ignore
        labeled_centers = labeled_centers.astype(np.uint16)

        # Handle size mismatch by cropping to overlapping region
        min_h: int = min(center_mask.shape[0], overall_mask.shape[0])
        min_w: int = min(center_mask.shape[1], overall_mask.shape[1])

        labeled_centers_crop = labeled_centers[:min_h, :min_w]
        overall_mask_crop = overall_mask[:min_h, :min_w]

        # Find center labels that overlap with overall structure
        overlap_region = (labeled_centers_crop > 0) & (overall_mask_crop > 0)
        overlapping_labels_all = np.unique(labeled_centers_crop[overlap_region])
        overlapping_labels: np.ndarray = overlapping_labels_all[
            overlapping_labels_all > 0]

        # Create filtered binary mask keeping only overlapping centers
        filtered_mask = np.isin(labeled_centers, overlapping_labels)

        return filtered_mask.astype(center_mask.dtype)

    @staticmethod
    def _create_markers_from_centers(center_objmap: np.ndarray) -> np.ndarray:
        """Create watershed markers from center objects using centroids.

        Computes centroid of each labeled center region and places a unique
        marker at that position. Ensures exactly one marker per center object.

        Args:
            center_objmap: Labeled map of center objects (2D integer array)

        Returns:
            Marker array with unique integers at centroid positions (2D int32)

        Note:
            Uses scipy.ndimage.center_of_mass() for accurate centroid computation.
            Rounds to integers for pixel coordinates.
        """

        # Create empty markers array
        markers = np.zeros_like(center_objmap, dtype=np.int32)

        # Get unique center labels (excluding background 0)
        center_labels_all = np.unique(center_objmap)
        center_labels: np.ndarray = center_labels_all[center_labels_all > 0]

        # Place marker at centroid of each center
        for marker_id, label_val in enumerate(center_labels, start=1):
            # Compute centroid
            mask = center_objmap == label_val
            centroid = center_of_mass(mask)

            # Round to integer coordinates
            row = int(round(centroid[0]))
            col = int(round(centroid[1]))

            # Bounds checking to ensure marker is placed inside the image
            if 0 <= row < markers.shape[0] and 0 <= col < markers.shape[1]:
                markers[row, col] = marker_id

        return markers

    @staticmethod
    def _create_boundary_elevation_map(
            shape: tuple, boundaries: np.ndarray, boundary_cost: float
    ) -> np.ndarray:
        """Create boundary-constrained elevation map for watershed.

        Creates flat elevation map with high cost at boundaries to create
        watershed basins separated by boundary ridges.

        Args:
            shape: Shape of output elevation map (from image.enh_gray.shape)
            boundaries: Binary mask of boundary pixels (high cost region)
            boundary_cost: Cost value assigned to boundary pixels

        Returns:
            Elevation map (2D float32 array) with boundaries marked as ridges
        """

        # Create flat elevation map
        elevation = np.zeros(shape, dtype=np.float32)

        # Add high cost at boundaries
        elevation[boundaries] = boundary_cost

        return elevation
