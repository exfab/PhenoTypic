from __future__ import annotations

from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image

from phenotypic.abc_ import ObjectRefiner

import numpy as np
from skimage.morphology import dilation, erosion


class MaskGradient(ObjectRefiner):
    """Extract object boundaries via morphological gradient.

    Intuition:
        The morphological gradient is the difference between dilation and erosion
        of the same mask. It produces a thin outline showing the boundary pixels
        of each object, preserving edges while removing interior and exterior
        pixels. On agar plates, this extracts colony perimeters for edge-based
        analysis, boundary visualization, or edge-specific measurements without
        the interior colony mass.

    Why this is useful for agar plates:
        Colony edges carry information about growth morphology, spreading,
        filamentous extensions, and optical properties. Extracting edges via
        morphological gradient enables edge-focused phenotyping (e.g., boundary
        roughness, circularity), visualization of colony contours, and detection
        of edge-specific features (halos, haloes, pigmentation patterns).

    Use cases:
        - Extract colony perimeters for edge-based morphological measurements.
        - Create boundary masks for edge-specific color or texture analysis.
        - Visualize colony outlines for QC overlays on raw images.
        - Detect spreading or filamentous edges extending from colony cores.

    Caveats:
        - Produces hollow edge masks, not filled objects suitable for standard
          morphological measurements (use on filled masks for meaningful results).
        - Large footprints produce thick, imprecise boundaries that lose detail
          and smooth away fine features.
        - Not suitable for downstream operations expecting solid filled masks
          (e.g., area measurements); intended for boundary analysis only.
        - Gradient edges may appear disconnected for colonies with complex or
          highly irregular shapes.

    Attributes:
        footprint (Literal["auto", "square", "diamond", "disk"] | np.ndarray | None):
            Structuring element used for gradient computation. Controls edge
            thickness and neighborhood size.
        radius (int): Footprint radius in pixels. Larger values produce thicker,
            less precise boundaries.

    Examples:
        .. dropdown:: Extract colony boundaries for edge analysis

            >>> from phenotypic.refine import MaskGradient
            >>> from phenotypic import Image
            >>> from phenotypic.detect import OtsuDetector
            >>> image = Image.imread("colony_plate.jpg")  # doctest: +SKIP
            >>> detected = OtsuDetector().apply(image)  # doctest: +SKIP
            >>> # Extract edges with auto-scaled footprint
            >>> refiner = MaskGradient(footprint='auto')  # doctest: +SKIP
            >>> edges = refiner.apply(detected)  # doctest: +SKIP
            >>> # Or use a small disk footprint (radius 1) for thin, precise edges
            >>> refiner = MaskGradient(footprint='disk', radius=1)  # doctest: +SKIP
            >>> edges = refiner.apply(detected, inplace=False)  # doctest: +SKIP

    Raises:
        AttributeError: If an invalid ``footprint`` type is provided (checked
            during operation).
    """

    def __init__(
            self,
            footprint: Literal[
                           "auto", "square", "diamond", "disk"] | np.ndarray | None = None,
            radius: int = 1
    ):
        """Initialize the gradient extractor.

        Args:
            footprint (Literal["auto", "square", "diamond", "disk"] | np.ndarray | None):
                Structuring element for gradient computation. Use:
                - "auto" to select a disk footprint scaled to image size,
                - a NumPy array to pass a custom footprint,
                - one of the named shapes ("disk", "square", "diamond") with
                  a specified radius,
                - or ``None`` to use the library default.

                Larger radii produce thicker boundaries with less precision but
                more robustness to noise.
            radius (int): Footprint radius in pixels when using named shapes
                or auto-scaling. Default: 1 pixel (thin, precise boundaries).
        """
        super().__init__()
        self.footprint = footprint
        self.radius = radius

    def _operate(self, image: Image) -> Image:
        if self.footprint == "auto":
            footprint = self._make_footprint(
                    "disk", radius=max(1, round(np.min(image.shape)*0.002))
            )
        elif isinstance(self.footprint, np.ndarray):
            footprint = self.footprint
        elif self.footprint in self._footprint_shapes:
            footprint = self._make_footprint(self.footprint, radius=self.radius)
        elif not self.footprint:
            footprint = None
        else:
            raise AttributeError("Invalid footprint type")

        # Compute morphological gradient: dilated - eroded
        mask = image.objmask[:]
        dilated_mask = dilation(mask, footprint=footprint)
        eroded_mask = erosion(mask, footprint=footprint)
        gradient_mask = dilated_mask & ~eroded_mask  # Boundary pixels

        image.objmask[:] = gradient_mask
        return image
