from __future__ import annotations
from typing import TYPE_CHECKING, Annotated, Any, List, Literal
import numpy as np
from pydantic import Field, field_validator
from scipy.ndimage import label

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from phenotypic.abc_ import ObjectDetector
from phenotypic.detect import OtsuDetector, RoundPeaksDetector
from phenotypic.sdk_.typing_ import OperationField, TuneSpec


class CompositeDetector(ObjectDetector):
    """Detect colonies by combining multiple detectors via union, intersection, or overlap filtering.

    Apply two or more detection algorithms (or preprocessing pipelines ending
    in a detector) to the same plate image and merge their binary masks.
    Ensemble combination improves sensitivity (union), specificity
    (intersection), or both (overlap filtering). This is especially useful
    for challenging plates where no single algorithm captures all colonies
    reliably. For a full comparison see
    :doc:`/explanation/detection_strategies_compared`.

    Best For:
        - Plates where different colony sub-populations respond to different
          detection algorithms (e.g., bright colonies via Otsu, faint
          colonies via triangle thresholding).
        - Consensus-based quality control that accepts only colonies
          confirmed by all methods.
        - Ensemble strategies that maximise recall by unioning masks from
          complementary algorithms.
        - Benchmarking workflows that compare detector agreement across
          methods.

    Consider Also:
        - :class:`OtsuDetector` or :class:`HysteresisDetector` when a
          single detector already captures all colonies reliably.
        - :class:`WatershedDetector` when the primary challenge is
          separating touching colonies rather than combining detection
          strategies.

    Args:
        detectors: List of :class:`~phenotypic.abc_.ObjectDetector` or
            :class:`~phenotypic.ImagePipeline` instances to combine.
            Pipelines allow preprocessing steps before detection. Defaults
            to ``[OtsuDetector(), RoundPeaksDetector()]`` when not
            specified.
        mode: Combination strategy. ``'union'`` marks a pixel as colony if
            any detector flags it (logical OR, maximises sensitivity).
            ``'intersection'`` requires all detectors to agree (logical AND,
            maximises specificity). ``'overlap'`` retains whole objects that
            have mutual spatial overlap across masks (balances sensitivity
            and specificity). Default: ``'overlap'``.
        min_overlap_ratio: For ``'overlap'`` mode, minimum fraction of
            object pixels that must overlap with another mask before an
            object is retained. At the default 0.0 any mutually overlapping
            object is kept; larger values bias towards objects confirmed by
            more than one detector. Typical range: 0.0--0.5. Default: 0.0.

    Returns:
        Image: Input image with ``objmask`` set to the combined binary
        colony mask and ``objmap`` derived from the merged mask.

    Raises:
        ValueError: If ``detectors`` is empty or ``mode`` is not one of
            ``'union'``, ``'intersection'``, or ``'overlap'``.

    See Also:
        :doc:`/tutorials/notebooks/02_detecting_colonies`
            Step-by-step tutorial for basic colony detection.
        :doc:`/how_to/notebooks/choose_detection_algorithm`
            Guide for selecting the right detector for your plate images.
        :doc:`/explanation/detection_strategies_compared`
            In-depth comparison of all detection strategies.
    """

    # Each detector may be an ``ObjectDetector`` or a nested
    # ``ImagePipeline``. ``OperationField`` keeps the concrete class of
    # each entry across a JSON round-trip (a bare ``model_dump`` of an
    # ``ObjectDetector | ImagePipeline`` union would lose the subclass).
    # ``| None`` permits an empty list slot: the GUI builder marks an
    # unfilled detector slot in an in-progress pipeline with ``None``
    # (pre-migration the un-validated list accepted these implicitly).
    detectors: List[OperationField | None] = Field(
        default_factory=lambda: [OtsuDetector(), RoundPeaksDetector()]
    )
    mode: Literal['union', 'intersection', 'overlap'] = 'overlap'
    min_overlap_ratio: Annotated[float, TuneSpec(0.0, 0.5)] = Field(0.0, ge=0.0, le=1.0)

    @field_validator("detectors", mode="before")
    @classmethod
    def _default_detectors(cls, value: Any) -> Any:
        """Map an explicit ``None`` onto the default detector pair.

        The pre-migration ``__init__`` accepted ``detectors=None`` as the
        "unset" sentinel and substituted ``[OtsuDetector(),
        RoundPeaksDetector()]``. The ``default_factory`` covers the
        omitted-argument case; this validator preserves the legacy
        explicit-``None`` call.
        """
        if value is None:
            return [OtsuDetector(), RoundPeaksDetector()]
        return value

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
            if detector is None:
                # An unfilled list slot (the GUI builder marks an empty
                # detector slot with None); nothing to apply — skip it.
                continue
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
                        objmaps[0],
                        objmaps[1],
                        min_overlap_ratio=self.min_overlap_ratio,
                )
                # For additional masks, progressively filter to keep mutual overlaps
                for mask in objmaps[2:]:
                    combined_mask = CompositeDetector._filter_mask_by_overlap_bidirectional(
                            combined_mask,
                            mask,
                            min_overlap_ratio=self.min_overlap_ratio,
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
    def _filter_mask_by_overlap_bidirectional(
            mask_a,
            mask_b,
            min_overlap_ratio: float = 0.0,
    ):
        """
        Retain objects in both masks that have mutual overlap.

        Objects are kept if they overlap with objects in the other mask by at
        least ``min_overlap_ratio`` of their own area. Returns a single merged
        mask containing all pixels from retained objects.

        Args:
            mask_a (np.ndarray): First binary mask (2D boolean or uint8)
            mask_b (np.ndarray): Second binary mask (2D boolean or uint8)
            min_overlap_ratio (float): Minimum object-area fraction that must
                overlap with the other mask. At 0.0, any non-zero overlap keeps
                the object.

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

        overlapping_labels_a = CompositeDetector._labels_meeting_overlap_ratio(
                crop_a,
                overlap_region,
                min_overlap_ratio,
        )
        overlapping_labels_b = CompositeDetector._labels_meeting_overlap_ratio(
                crop_b,
                overlap_region,
                min_overlap_ratio,
        )

        # Create filtered masks retaining only mutually overlapping objects
        filtered_a = np.isin(labeled_a, overlapping_labels_a)
        filtered_b = np.isin(labeled_b, overlapping_labels_b)

        # Merge into single mask
        merged_mask = filtered_a | filtered_b

        return merged_mask.astype(mask_a.dtype)

    @staticmethod
    def _labels_meeting_overlap_ratio(
            labeled_mask: np.ndarray,
            overlap_region: np.ndarray,
            min_overlap_ratio: float,
    ) -> list[int]:
        """Return labels whose overlap fraction meets the configured threshold."""
        kept_labels: list[int] = []
        for label_id in np.unique(labeled_mask[labeled_mask > 0]):
            object_pixels = labeled_mask == label_id
            overlap_pixels = int(np.count_nonzero(object_pixels & overlap_region))
            if overlap_pixels == 0:
                continue
            if min_overlap_ratio <= 0.0:
                kept_labels.append(int(label_id))
                continue

            object_area = int(np.count_nonzero(object_pixels))
            if overlap_pixels / object_area >= min_overlap_ratio:
                kept_labels.append(int(label_id))
        return kept_labels

    @staticmethod
    def _filter_mask_by_overlap(mask_to_clean, reference_mask):
        """
        Retain only objects in mask that overlap with reference_mask.

        Args:
            mask_to_clean (np.ndarray): Binary mask to filter (2D boolean or uint8)
            reference_mask (np.ndarray): Binary mask defining valid regions (2D boolean or uint8)

        Returns:
            np.ndarray: Filtered binary mask with same shape as mask

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
