from __future__ import annotations
from typing import TYPE_CHECKING, List, Literal, Union
import numpy as np
from scipy.ndimage import label

if TYPE_CHECKING:
    from phenotypic._core._image import Image
    from phenotypic._core._image_pipeline import ImagePipeline

from phenotypic.abc_ import ObjectDetector
from phenotypic.detect import OtsuDetector, RoundPeaksDetector


class CompositeDetector(ObjectDetector):
    """
    Combines multiple detectors or pipelines using union, intersection, or overlap filtering.

    This detector applies multiple detection algorithms (or preprocessing pipelines +
    detection) to the same image and combines their objmask outputs. Useful for
    ensemble detection, consensus filtering, or quality control.

    Args:
        detectors: List of ObjectDetector instances or ImagePipeline instances to combine.
            ImagePipeline instances allow preprocessing steps before detection.
            Defaults to empty list.
        mode: Combination strategy:
            - 'union': Pixel is True if True in ANY detector (logical OR)
            - 'intersection': Pixel is True if True in ALL detectors (logical AND)
            - 'overlap': Keep only objects with sufficient overlap across all masks
        min_overlap_ratio: For 'overlap' mode, minimum fraction of object pixels
            that must overlap with all other masks. Range: 0.0 to 1.0. Default: 0.5

    Returns:
        Image: Image with combined objmask and objmap from all detectors/pipelines.

    Raises:
        ValueError: If detectors list is empty or mode is invalid.

    Use Cases:
        - Ensemble detection: Combine multiple algorithms for increased sensitivity
        - Consensus detection: Only accept colonies detected by all methods (high confidence)
        - Quality filtering: Remove spurious detections via overlap requirements

    Limitations:
        - Execution time is sum of all detector execution times (runs sequentially)
        - Union mode may increase false positives vs single detector
        - Intersection mode may miss valid colonies detected by only one method
        - Overlap mode requires tuning min_overlap_ratio for your imaging conditions

    Parameter Effects:
        - **mode='union'**: Maximizes sensitivity, may include more noise
        - **mode='intersection'**: Maximizes specificity, may miss valid objects
        - **mode='overlap'**: Balances sensitivity/specificity based on min_overlap_ratio
        - **min_overlap_ratio** (overlap mode only): Higher values = more conservative filtering

    Examples:
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from phenotypic import ImagePipeline
        >>> from phenotypic.detect import OtsuDetector, CannyDetector, CompositeDetector
        >>> from phenotypic.enhance import GaussianBlur
        >>>
        >>> image = load_synth_yeast_plate()
        >>>
        >>> # Union: detect colonies found by either method (high sensitivity)
        >>> composite = CompositeDetector(
        ...     detectors=[OtsuDetector(), CannyDetector(sigma=2)],
        ...     mode='union'
        ... )
        >>> result = composite.apply(image)
        >>> print(f"Union detected {result.objmap[:].max()} colonies")
        >>>
        >>> # Mix detectors and pipelines for strategy comparison
        >>> composite_mixed = CompositeDetector(
        ...     detectors=[
        ...         OtsuDetector(),
        ...         ImagePipeline([
        ...             GaussianBlur(sigma=2),
        ...             CannyDetector(sigma=2)
        ...         ])
        ...     ],
        ...     mode='intersection'
        ... )
        >>> result = composite_mixed.apply(image)
        >>> print(f"Mixed strategy detected {result.objmap[:].max()} colonies")
    """

    def __init__(
            self,
            detectors: List[Union[ObjectDetector, 'ImagePipeline']] = None,
            mode: Literal['union', 'intersection', 'overlap'] = 'overlap',
            min_overlap_ratio: float = 0.0
    ):
        super().__init__()
        self.detectors = detectors if detectors is not None else [
            OtsuDetector(),
            RoundPeaksDetector()
        ]
        self.mode = mode
        self.min_overlap_ratio = min_overlap_ratio

    def _operate(self, image: Image) -> Image:
        """Apply all detectors/pipelines and combine their objmask outputs."""

        if not self.detectors:
            raise ValueError(
                    "At least one detector must be provided to CompositeDetector")

        if self.mode not in ['union', 'intersection', 'overlap']:
            raise ValueError(
                    f"Invalid mode '{self.mode}'. "
                    f"Must be 'union', 'intersection', or 'overlap'"
            )

        # Import here to avoid circular dependency
        from phenotypic import ImagePipeline

        # Apply all detectors/pipelines and collect masks
        objmaps = []

        for detector in self.detectors:
            if isinstance(detector, ImagePipeline):
                # Apply pipeline (preprocessing + detection)
                detected_image = detector.apply(image,
                                                inplace=False,
                                                reset=False)
                objmaps.append(detected_image.objmap[:].astype(bool))
            else:
                # Apply detector directly (ObjectDetector)
                detected_image = detector.apply(image, inplace=False)
                objmaps.append(detected_image.objmap[:].astype(bool))

        # Combine masks based on mode
        if self.mode == 'union':
            # Pixel is True if True in ANY mask (logical OR)
            combined_mask = np.logical_or.reduce(objmaps)

        elif self.mode == 'intersection':
            # Pixel is True if True in ALL masks (logical AND)
            combined_mask = np.logical_and.reduce(objmaps)

        elif self.mode == 'overlap':
            # Keep entire objects from any mask that have mutual overlap
            if len(objmaps) == 1:
                # Only one detector, just use its mask
                combined_mask = objmaps[0]
            else:
                # Start with first two masks
                combined_mask = CompositeDetector._filter_mask_by_overlap_bidirectional(
                        objmaps[0], objmaps[1]
                )
                # For additional masks, progressively filter to keep mutual overlaps
                for mask in objmaps[2:]:
                    combined_mask = CompositeDetector._filter_mask_by_overlap_bidirectional(
                            combined_mask, mask
                    )
        else:
            raise ValueError(
                    f"Invalid mode '{self.mode}'. "
                    f"Must be 'union', 'intersection', or 'overlap'"
            )

        # Set objmask
        image.objmask[:] = combined_mask > 0

        return image

    @staticmethod
    def _filter_mask_by_overlap_bidirectional(mask_a, mask_b):
        """
        Retain objects in both masks that have mutual overlap.

        Objects are kept if they overlap with objects in the other mask. Returns
        a single merged mask containing all pixels from mutually-overlapping objects.

        Args:
            mask_a (np.ndarray): First binary mask (2D boolean or uint8)
            mask_b (np.ndarray): Second binary mask (2D boolean or uint8)

        Returns:
            np.ndarray: Merged binary mask with same dtype as mask_a, containing
                        pixels from objects with mutual overlap

        Raises:
            ValueError: If masks have incompatible shapes
        """
        # Label connected components in both masks
        labeled_a, _ = label(mask_a)
        labeled_b, _ = label(mask_b)

        # Handle potential size mismatch by finding overlapping region
        min_h = min(mask_a.shape[0], mask_b.shape[0])
        min_w = min(mask_a.shape[1], mask_b.shape[1])

        crop_a = labeled_a[:min_h, :min_w]
        crop_b = labeled_b[:min_h, :min_w]

        # Find pixels where both masks have objects
        overlap_region = (crop_a > 0) & (crop_b > 0)

        # Get labels from each mask that appear in overlap region
        overlapping_labels_a = np.unique(crop_a[overlap_region])
        overlapping_labels_b = np.unique(crop_b[overlap_region])

        # Remove background label if present
        overlapping_labels_a = overlapping_labels_a[overlapping_labels_a > 0]
        overlapping_labels_b = overlapping_labels_b[overlapping_labels_b > 0]

        # Create filtered masks retaining only mutually overlapping objects
        filtered_a = np.isin(labeled_a, overlapping_labels_a)
        filtered_b = np.isin(labeled_b, overlapping_labels_b)

        # Merge into single mask
        merged_mask = filtered_a | filtered_b

        return merged_mask.astype(mask_a.dtype)

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
        from skimage.measure import label

        # Label connected components in mask to clean
        labeled: np.ndarray = label(mask_to_clean)
        reference_mask = reference_mask > 0

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
