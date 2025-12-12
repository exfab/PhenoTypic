from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image

import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import regionprops_table
import pandas as pd

from ..abc_ import ObjectRefiner


class SmallToLargeMerger(ObjectRefiner):
    """Merge small colony fragments into nearby large colonies (hierarchical merging).

    Intuition:
        Fragmented colonies often produce one large central detection plus small
        satellite fragments from uneven pigmentation or lighting. This refiner
        implements hierarchical merging: small fragments are absorbed into their
        nearest large neighbor (which never merge), preserving large colonies as
        stable anchors. This is more targeted than distance-based merging and
        explicitly preserves the structure of well-formed colonies.

    Why this is useful for agar plates:
        Heterogeneous pigmentation, uneven illumination, or aggressive thresholding
        can fragment a single colony into a large central region plus small satellites.
        Size-based filtering assumes large = real colony and small = artifact or
        fragment. By merging only small objects into large neighbors, you reconstruct
        the full colony footprint while avoiding the risk of merging two distinct
        large colonies that happen to be close together.

    Use cases:
        - Cleaning up fragmented detections from heterogeneous colony pigmentation
          (e.g., highly pigmented or mucoid colonies with internal voids).
        - Removing small debris/dust artifacts near real colonies without merging
          distinct colonies.
        - Post-processing watershed over-segmentation where one colony becomes
          multiple regions.
        - Correcting detections on plates with severe lighting gradients that
          create satellite fragments around main colonies.

    Caveats:
        - If no large colonies exist (all objects below size_threshold), no
          merging occurs. May require tuning size_threshold to your image
          characteristics.
        - Small colonies far from any large colony remain independent. This is
          usually desired (preserves isolated small colonies) but may leave some
          noise artifacts if distance_threshold is small.
        - Large colonies never merge, even if they are extremely close together.
          If you need to merge large colonies, use TransitiveDistanceMerger instead.
        - Cannot distinguish between legitimate small colonies and noise fragments
          based on size alone. A small but viable colony below size_threshold will
          be absorbed into a nearby large colony.
        - Multiple small fragments near the same large colony all merge to that
          colony, potentially distorting its shape if fragments are far from the
          main body.

    Attributes:
        distance_threshold (float): Maximum distance in pixels from small fragment
            to large colony for merging. Smaller values (10-20) are conservative
            and reduce risk of merging unrelated fragments. Larger values (30-50)
            clean more aggressively but may merge distant small colonies. Should
            be smaller than minimum distance between distinct large colonies.
        size_threshold (int): Pixel area separating "small" fragments from "large"
            anchor colonies. Objects with area < size_threshold are candidates for
            merging; objects >= size_threshold are preserved as anchors. Tune based
            on expected colony size. Typical range: 50-200 pixels depending on
            imaging resolution. Smaller values classify more objects as large
            anchors; larger values classify more as small fragments.

    Examples:
        .. dropdown:: Merge small fragments into parent colonies

            >>> from phenotypic.refine import SmallToLargeMerger
            >>> from phenotypic import Image
            >>> from phenotypic.detect import OtsuDetector
            >>> image = Image('fragmented_plate.jpg')
            >>> detected = OtsuDetector().apply(image)
            >>> # Merge fragments <100px into nearby colonies >100px
            >>> merger = SmallToLargeMerger(
            ...     distance_threshold=40,
            ...     size_threshold=100
            ... )
            >>> refined = merger.apply(detected)  # doctest: +SKIP
            >>> print(f"Consolidated fragments: {detected.objmap[:].max()} -> {refined.objmap[:].max()}")  # doctest: +SKIP
    """

    def __init__(self, distance_threshold: float = 30.0, size_threshold: int = 100):
        """Initialize the merger.

        Args:
            distance_threshold (float): Maximum distance from small fragment to
                large colony for merging (pixels).
            size_threshold (int): Minimum area for an object to be considered a
                "large" anchor colony. Smaller objects are candidates for merging.

        Raises:
            ValueError: If distance_threshold or size_threshold are not positive.
        """
        if distance_threshold <= 0:
            raise ValueError("distance_threshold must be positive")
        if size_threshold <= 0:
            raise ValueError("size_threshold must be positive")
        self.distance_threshold = distance_threshold
        self.size_threshold = size_threshold

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
            Image object with merged objmap and unchanged RGB/gray/enh_gray.
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
