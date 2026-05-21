from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
from pydantic import field_validator
from scipy.spatial import cKDTree
from skimage.measure import regionprops_table
import pandas as pd

from ..abc_ import ObjectRefiner


class NearestNeighborMerger(ObjectRefiner):
    """Merge small fragments into their nearest neighboring colony.

    Each object below ``min_size`` is absorbed into its single closest
    neighbor within ``distance_threshold``. Unlike transitive merging, this
    is one-directional and conservative — it cleans up debris without
    cascading merges across the plate.

    Args:
        distance_threshold: Maximum centroid distance (pixels) for merging.
            Objects beyond this distance remain independent. Typical range:
            15--40. Default: 25.
        min_size: Only objects with area below this threshold are merge
            candidates. Larger objects serve as anchor targets. ``None``
            merges all objects (rarely desired). Default: 50.

    Returns:
        Image: Input image with ``objmask`` and ``objmap`` updated after
        merging small objects into their nearest neighbors.

    Best For:
        - Absorbing dust or noise fragments near real colonies.
        - Cleaning up small detection artifacts without risking cascading merges.
        - Size-selective cleanup where only fragments below a threshold merge.

    Consider Also:
        - :class:`TransitiveDistanceMerger` for chained merging of all nearby
          objects regardless of size.
        - :class:`SmallToLargeMerger` for merging small objects into the
          largest nearby colony.
        - :class:`MaskCloser` for bridging narrow gaps morphologically.

    See Also:
        :doc:`/how_to/notebooks/merge_fragmented_detections` for fragment
        merging strategies.
        :doc:`/explanation/refinement_strategies` for choosing the right
        refinement approach.
    """

    distance_threshold: float = 20.0
    min_size: int | None = 50

    @field_validator("distance_threshold")
    @classmethod
    def _validate_distance_threshold(cls, distance_threshold: float) -> float:
        """Reject a non-positive ``distance_threshold``.

        Reproduces the pre-migration ``__init__`` guard verbatim.
        """
        if distance_threshold <= 0:
            raise ValueError("distance_threshold must be positive")
        return distance_threshold

    @field_validator("min_size")
    @classmethod
    def _validate_min_size(cls, min_size: int | None) -> int | None:
        """Reject a non-positive ``min_size`` when one is provided.

        Reproduces the pre-migration ``__init__`` guard verbatim.
        """
        if min_size is not None and min_size <= 0:
            raise ValueError("min_size must be positive if provided")
        return min_size

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
