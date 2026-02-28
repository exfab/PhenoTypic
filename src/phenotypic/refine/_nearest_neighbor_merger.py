from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import regionprops_table
import pandas as pd

from ..abc_ import ObjectRefiner


class NearestNeighborMerger(ObjectRefiner):
    """Merge colonies to their nearest neighbor within distance threshold.

    Intuition:
        Unlike transitive merging which creates chains, nearest-neighbor merging
        is one-directional: each object merges to its single closest neighbor if
        within threshold. This prevents cascading merges while still addressing
        fragmented colonies. Combined with size filtering, this is effective for
        removing small noise/debris artifacts near real colonies without over-merging
        distinct colonies that happen to be close together.

    Why this is useful for agar plates:
        Small fragments from dust, agar texture, or sensor noise often appear as
        independent labels near larger, real colonies. Merging only small objects
        to their nearest neighbor provides selective cleanup: genuine small colonies
        remain untouched while noise is absorbed into nearby larger structures.
        This is more conservative than transitive merging and better preserves
        distinct objects in crowded plates.

    Use cases:
        - Removing small noise artifacts near real colonies while preserving
          distinct small colonies that may be legitimate.
        - Cleaning up dust, salt-and-pepper noise, or agar texture artifacts
          without risk of cascading merges.
        - Post-processing when you want to be conservative: merge only objects
          below a size threshold and only to their nearest neighbor.
        - Working with size-biased detection artifacts where small fragments
          are noise but large objects are colonies of interest.

    Caveats:
        - One-directional merging creates asymmetric behavior: object A merges
          to B, but B may merge to C, creating indirect chains. Not guaranteed
          to merge all objects within threshold distance to each other.
        - Without size filtering (min_size=None), all objects merge to their
          nearest neighbor, even large well-formed colonies. This is rarely
          desired unless you want to explicitly merge all nearby objects.
        - Labels may remain non-consecutive after merging (acceptable for
          functional purposes but may affect visualization).
        - Small objects that are equidistant from multiple neighbors will merge
          to one arbitrarily based on KDTree ordering.

    Attributes:
        distance_threshold (float): Maximum distance to nearest neighbor for
            merging (pixels). Objects with nearest neighbor beyond this distance
            remain independent. Typical range: 15-40 pixels. Smaller values
            preserve more independence; larger values merge more aggressively.
        min_size (int | None): If provided, only objects with area < min_size
            are candidates for merging. Objects >= min_size are preserved
            independently and act as anchor targets. Default: 50 pixels. Set to
            None to merge all objects regardless of size (usually not recommended).

    Examples:
        Remove small noise by merging only small objects to neighbors:

        >>> from phenotypic.refine import NearestNeighborMerger
        >>> from phenotypic import Image
        >>> from phenotypic.detect import OtsuDetector
        >>> image = Image('noisy_plate.jpg')
        >>> detected = OtsuDetector().apply(image)
        >>> # Merge only objects smaller than 50 pixels to their nearest colony
        >>> merger = NearestNeighborMerger(
        ...     distance_threshold=25,
        ...     min_size=50
        ... )
        >>> cleaned = merger.apply(detected)  # doctest: +SKIP
        >>> print(f"Removed small artifacts: {detected.objmap[:].max()} -> {cleaned.objmap[:].max()}")  # doctest: +SKIP
    """

    def __init__(self, distance_threshold: float = 20.0, min_size: Optional[int] = 50):
        """Initialize the merger.

        Args:
            distance_threshold (float): Maximum distance to nearest neighbor for
                merging. Objects farther than this remain independent.
            min_size (int | None): Minimum area to preserve independently.
                Objects smaller than this merge to nearest neighbor if within
                distance_threshold. Larger objects remain untouched.

        Raises:
            ValueError: If distance_threshold is not positive or if min_size is
                provided and not positive.
        """
        if distance_threshold <= 0:
            raise ValueError("distance_threshold must be positive")
        if min_size is not None and min_size <= 0:
            raise ValueError("min_size must be positive if provided")
        self.distance_threshold = distance_threshold
        self.min_size = min_size

    def _operate(self, image: Image) -> Image:
        """Apply nearest-neighbor distance-based merging to objmap.

        Algorithm:
            1. Extract labels, centroids, and areas from labeled map
            2. Build KDTree from centroids
            3. Query k=2 nearest neighbors (first is self, second is actual nearest)
            4. For each object:
               - If object is large (>= min_size), preserve independently
               - If object is small and nearest neighbor within threshold, merge
               - Otherwise preserve independently
            5. Apply merge mapping to objmap

        Args:
            image: Image object with populated objmap from prior detection.

        Returns:
            Image object with merged objmap and unchanged RGB/gray/detect_mat.
        """
        objmap = image.objmap[:]

        # Edge cases: empty or single object
        if objmap.max() == 0:
            return image
        if objmap.max() == 1:
            return image

        # Extract properties
        props = regionprops_table(
            label_image=objmap, properties=["label", "centroid", "area"]
        )
        df = pd.DataFrame(props)

        labels = df["label"].values
        centroids = df[["centroid-0", "centroid-1"]].values
        areas = df["area"].values

        # Build KDTree for spatial queries
        tree = cKDTree(centroids)

        # Query k=2 nearest neighbors (self + actual nearest)
        distances, indices = tree.query(centroids, k=2)

        # Build merge map
        merge_map = {}

        for i, label in enumerate(labels):
            # Check size filter: preserve large objects independently
            if self.min_size is not None and areas[i] >= self.min_size:
                merge_map[label] = label
                continue

            # Get actual nearest neighbor (second result, first is self)
            nearest_idx = indices[i, 1]
            nearest_label = labels[nearest_idx]
            distance_to_nearest = distances[i, 1]

            # Merge to nearest if within threshold
            if distance_to_nearest <= self.distance_threshold:
                merge_map[label] = nearest_label
            else:
                merge_map[label] = label

        # Apply merge mapping to objmap
        remap = np.vectorize(lambda lbl: merge_map.get(lbl, lbl))
        merged_objmap = remap(objmap)

        # Write result (no relabeling for this approach)
        image.objmap[:] = merged_objmap

        return image
