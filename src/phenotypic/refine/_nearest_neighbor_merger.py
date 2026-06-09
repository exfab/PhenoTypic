from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
from pydantic import Field, field_validator
from scipy.spatial import cKDTree
from skimage.measure import regionprops_table
import pandas as pd

from ..abc_ import ObjectRefiner
from ..tools_.typing_ import TuneSpec


class NearestNeighborMerger(ObjectRefiner):
    """Absorb small colony fragments into their nearest neighboring detection.

    For each object whose area is below ``min_size``, finds the single
    nearest centroid within ``distance_threshold`` and reassigns the
    fragment's label to that neighbor. The merge is one-directional and
    non-transitive: large anchor colonies are never themselves merged,
    which avoids cascading chains across the plate.

    For a comparison of merging strategies, see
    :doc:`/explanation/refinement_strategies`.

    Best For:
        - Absorbing dust or debris fragments that appear near genuine
          colonies after thresholding or edge detection.
        - Size-selective cleanup where only small artefacts below an area
          threshold need merging while large colonies remain independent.
        - Plates with scattered micro-colonies or satellite spots that
          should count as part of the nearest parent colony.

    Consider Also:
        - :class:`MergeFragmentChains` when transitive closure is needed
          so that chains of nearby fragments (A near B near C) merge into
          one detection even if the endpoints are far apart.
        - :class:`SmallToLargeMerger` when small fragments should
          specifically merge into the largest nearby colony rather than
          the geometrically closest one.
        - :class:`MaskClosing` for morphological bridging of narrow gaps
          without relabeling objects.

    Args:
        distance_threshold: Maximum centroid-to-centroid distance in pixels
            within which a small fragment is eligible to merge with its
            nearest neighbor. Fragments beyond this distance remain
            independent. Typical range: 10--50. Default: 20.0.
        min_size: Area threshold in pixels. Objects with area strictly below
            this value are merge candidates; objects at or above serve as
            immovable anchors. Set to ``None`` to make all objects eligible
            for merging. Typical range: 20--200. Default: 50.

    Returns:
        Image: Input image with ``objmap`` updated after absorbing small
        fragments into their nearest neighbors. ``objmask``, ``rgb``,
        ``gray``, and ``detect_mat`` are unchanged.

    Raises:
        ValueError: If ``distance_threshold`` is not positive.
        ValueError: If ``min_size`` is provided and not positive.

    See Also:
        :doc:`/how_to/notebooks/merge_fragmented_detections` for fragment
        merging workflows on real plate images.
        :doc:`/explanation/refinement_strategies` for choosing the right
        merging approach.
    """

    distance_threshold: Annotated[float, TuneSpec(10.0, 50.0)] = Field(
            default=20.0, gt=0
    )
    min_size: Annotated[int | None, TuneSpec(20, 200, log=True)] = 50

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
