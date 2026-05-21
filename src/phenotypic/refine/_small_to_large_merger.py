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


class SmallToLargeMerger(ObjectRefiner):
    """Merge small colony fragments into their nearest large colony using hierarchical size-based merging.

    Partitions objects into small (below size threshold) and large (at or
    above), then absorbs each small fragment into the nearest large neighbor
    within the distance threshold. Large colonies serve as stable anchors
    and never merge with each other, preventing false consolidation of
    distinct colonies.

    Args:
        distance_threshold: Maximum centroid-to-centroid distance in pixels
            for merging a small fragment into a large colony. Typical
            range: 10--50. Should be smaller than the minimum distance
            between distinct large colonies. Default: 30.0.
        size_threshold: Pixel area separating small fragments from large
            anchor colonies. Objects below this are merge candidates;
            objects at or above are preserved as anchors. Typical range:
            50--200. Default: 100.

    Returns:
        Image: Input image with ``objmap`` updated so that small fragments
        are relabeled to their nearest large colony.

    Raises:
        ValueError: If ``distance_threshold`` or ``size_threshold`` is not
            positive.

    Best For:
        - Fragmented detections from heterogeneous pigmentation or uneven
          illumination where satellites cluster around a main colony.
        - Post-watershed over-segmentation where one colony splits into a
          large core plus small peripheral regions.
        - Removing small debris near real colonies without merging distinct
          large colonies.
        - Plates with severe lighting gradients that produce satellite
          fragments around main detections.

    Consider Also:
        - :class:`TransitiveDistanceMerger` when all nearby objects should
          merge regardless of size, including large-to-large merging.
        - :class:`NearestNeighborMerger` for simple nearest-neighbor
          merging without size partitioning.
        - :class:`SmallObjectRemover` when small fragments should be
          discarded entirely rather than absorbed.

    See Also:
        :doc:`/how_to/notebooks/merge_fragmented_detections` for fragment
        merging workflows.
        :doc:`/explanation/refinement_strategies` for a comparison of
        merging strategies.
    """

    distance_threshold: float = 30.0
    size_threshold: int = 100

    @field_validator("distance_threshold")
    @classmethod
    def _validate_distance_threshold(cls, distance_threshold: float) -> float:
        """Reject a non-positive ``distance_threshold``.

        Reproduces the pre-migration ``__init__`` guard verbatim.
        """
        if distance_threshold <= 0:
            raise ValueError("distance_threshold must be positive")
        return distance_threshold

    @field_validator("size_threshold")
    @classmethod
    def _validate_size_threshold(cls, size_threshold: int) -> int:
        """Reject a non-positive ``size_threshold``.

        Reproduces the pre-migration ``__init__`` guard verbatim.
        """
        if size_threshold <= 0:
            raise ValueError("size_threshold must be positive")
        return size_threshold

    def _operate(self, image: Image) -> Image:
        """Apply small-to-large hierarchical merging to objmap.

        Algorithm:
            1. Extract label, centroid, area for all objects
            2. Partition into small (< threshold) and large (>= threshold)
            3. If no large objects exist, return unchanged
            4. Build KDTree from large centroids only
            5. Query nearest large colony for each small object
            6. Merge small -> large if distance <= threshold
            7. Apply merge mapping to objmap

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

        # Partition into small and large
        small_df = df[df["area"] < self.size_threshold]
        large_df = df[df["area"] >= self.size_threshold]

        # Edge case: no large objects to merge into
        if len(large_df) == 0:
            return image

        # Edge case: no small objects to merge
        if len(small_df) == 0:
            return image

        # Extract large object properties
        large_centroids = large_df[["centroid-0", "centroid-1"]].values
        large_labels = large_df["label"].values

        # Build KDTree from large centroids only
        tree = cKDTree(large_centroids)

        # Extract small object properties
        small_centroids = small_df[["centroid-0", "centroid-1"]].values
        small_labels = small_df["label"].values

        # Query nearest large colony for each small object
        distances, indices = tree.query(small_centroids, k=1)

        # Initialize merge map: large objects map to themselves
        merge_map = {lbl: lbl for lbl in large_labels}

        # Add small objects: merge to nearest large if within threshold
        for i, small_label in enumerate(small_labels):
            nearest_large_label = large_labels[indices[i]]
            distance_to_nearest = distances[i]

            if distance_to_nearest <= self.distance_threshold:
                merge_map[small_label] = nearest_large_label
            else:
                merge_map[small_label] = small_label  # Too far, keep independent

        # Apply merge mapping to objmap
        remap = np.vectorize(lambda lbl: merge_map.get(lbl, lbl))
        merged_objmap = remap(objmap)

        # Write result
        image.objmap[:] = merged_objmap

        return image
