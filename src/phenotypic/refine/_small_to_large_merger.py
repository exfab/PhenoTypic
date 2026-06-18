from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
from pydantic import Field
from scipy.spatial import cKDTree
from skimage.measure import regionprops_table
import pandas as pd

from ..abc_ import ObjectRefiner
from ..sdk_.typing_ import TuneSpec


class SmallToLargeMerger(ObjectRefiner):
    """Merge small colony fragments into their nearest large colony using size-partitioned merging.

    Partitions detected objects into small fragments (below ``size_threshold``)
    and large anchor colonies (at or above), then absorbs each fragment into
    the nearest anchor within ``distance_threshold`` pixels. Large colonies
    are never merged with each other, preventing false consolidation of
    distinct colonies.

    For a comparison of fragment merging strategies, see
    :doc:`/explanation/refinement_strategies`.

    Best For:
        - Fragmented detections from heterogeneous colony pigmentation or
          uneven illumination that produce satellite regions around a main
          colony body.
        - Post-watershed over-segmentation where one colony splits into a
          large core and small peripheral fragments.
        - Plates with lighting gradients that generate small spurious
          detections adjacent to genuine large colonies.
        - Scenarios where small isolated debris should be absorbed into
          nearby real colonies rather than discarded outright.

    Consider Also:
        - :class:`MergeFragmentChains` when large-to-large merging is also
          needed, not only small-to-large absorption.
        - :class:`NearestNeighborMerger` for nearest-neighbor merging
          without any size-based partitioning.
        - :class:`SmallObjectRemover` when small fragments should be
          discarded entirely rather than absorbed into a parent colony.

    Args:
        distance_threshold: Maximum centroid-to-centroid distance in pixels
            within which a small fragment is absorbed into the nearest large
            colony. Set smaller than the minimum expected spacing between
            distinct large colonies to avoid cross-colony merges. Typical
            range: 10--50. Default: 30.0.
        size_threshold: Pixel area boundary separating merge candidates
            (below threshold) from immovable anchor colonies (at or above).
            Typical range: 50--200. Scale upward for high-resolution scans
            where colony areas are larger. Default: 100.

    Returns:
        Image: Input image with ``objmap`` updated so that small fragments
        within range are relabeled to match their nearest large colony.

    Raises:
        ValueError: If ``distance_threshold`` or ``size_threshold`` is not
            positive.

    See Also:
        :doc:`/how_to/notebooks/merge_fragmented_detections` for fragment
        merging workflows on real plate images.
        :doc:`/explanation/refinement_strategies` for a comparison of
        merging strategies.
    """

    distance_threshold: Annotated[float, TuneSpec(10.0, 50.0)] = Field(
            default=30.0, gt=0
    )
    size_threshold: Annotated[int, TuneSpec(50, 200, log=True)] = Field(
            default=100, gt=0
    )

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
