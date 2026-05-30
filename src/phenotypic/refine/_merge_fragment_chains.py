from __future__ import annotations

from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
from pydantic import field_validator
from scipy.spatial import cKDTree
from skimage.measure import regionprops_table
import pandas as pd

from ..abc_ import ObjectRefiner


class MergeFragmentChains(ObjectRefiner):
    """Merge nearby colony fragments using transitive closure of centroid distances.

    Finds all pairs of objects with centroids within the distance threshold,
    then applies union-find with path compression to transitively merge
    connected groups. If A is near B and B is near C, all three merge into
    one detection even if A and C are not directly within threshold. Labels
    are relabeled consecutively after merging.

    Args:
        distance_threshold: Maximum centroid-to-centroid distance in pixels
            for merging. Typical range: 10--30. Lower values are
            conservative; higher values merge more aggressively but risk
            combining distinct colonies via transitive chains. Default: 20.0.

    Returns:
        Image: Input image with ``objmap`` updated so that transitively
        connected fragments share a single label, relabeled consecutively.

    Raises:
        ValueError: If ``distance_threshold`` is not positive.

    Best For:
        - Repairing fragmented detections from watershed over-segmentation
          where a single colony splits into multiple touching regions.
        - Consolidating micro-fragments and satellite spots caused by
          thresholding artifacts or agar texture.
        - Correcting detections on plates with harsh shadows or glare that
          create internal voids within colony masks.
        - Post-processing after aggressive noise removal that leaves
          fragmented edges.

    Consider Also:
        - :class:`SmallToLargeMerger` when only small fragments should
          merge into large anchors, preserving distinct large colonies.
        - :class:`NearestNeighborMerger` for simple pairwise nearest-
          neighbor merging without transitive closure.
        - :class:`MaskClosing` for morphological closing that bridges small
          gaps without relabeling.

    See Also:
        :doc:`/how_to/notebooks/merge_fragmented_detections` for fragment
        merging workflows.
        :doc:`/explanation/refinement_strategies` for a comparison of
        merging strategies.
    """

    distance_threshold: float = 20.0

    @field_validator("distance_threshold")
    @classmethod
    def _validate_distance_threshold(cls, distance_threshold: float) -> float:
        """Reject a non-positive ``distance_threshold``.

        Reproduces the pre-migration ``__init__`` guard verbatim.
        """
        if distance_threshold <= 0:
            raise ValueError("distance_threshold must be positive")
        return distance_threshold

    @staticmethod
    def _find_root(label: int, parent: Dict[int, int]) -> int:
        """Find root parent with path compression (iterative version for pickling)."""
        root = label
        # Find root
        while parent[root] != root:
            root = parent[root]
        # Path compression: update all nodes to point directly to root
        current = label
        while parent[current] != root:
            next_node = parent[current]
            parent[current] = root
            current = next_node
        return root

    @staticmethod
    def _apply_merge_map(objmap: np.ndarray, merge_map: Dict[int, int]) -> np.ndarray:
        """Apply merge mapping to object map."""
        vfunc = np.vectorize(lambda lbl: merge_map.get(lbl, lbl), otypes=[np.uint16])
        return vfunc(objmap)

    @staticmethod
    def _relabel_consecutive(objmap: np.ndarray) -> np.ndarray:
        """Relabel merged groups to consecutive integers starting from 1."""
        unique_labels = np.unique(objmap)
        unique_labels = unique_labels[unique_labels > 0]  # Exclude background (0)

        if len(unique_labels) == 0:
            return objmap

        # Create relabeling map
        relabel_map = {
            old_label: new_label for new_label, old_label in enumerate(unique_labels,
                                                                       start=1)
        }
        relabel_map[0] = 0  # Background stays 0

        # Apply relabeling
        vfunc = np.vectorize(lambda lbl: relabel_map.get(lbl, lbl), otypes=[np.uint16])
        return vfunc(objmap)

    def _operate(self, image: Image) -> Image:
        """Apply transitive distance-based merging to objmap.

        Algorithm:
            1. Extract centroids from labeled map using regionprops
            2. Build KDTree for efficient spatial queries
            3. Find all pairs of objects within distance_threshold
            4. Use union-find with path compression to build transitive closure
            5. Remap labels to merged groups
            6. Relabel consecutively from 1

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

        # Extract centroids and labels
        props = regionprops_table(
                label_image=objmap, properties=["label", "centroid"]
        )
        df = pd.DataFrame(props)

        labels = df["label"].values
        centroids = df[["centroid-0", "centroid-1"]].values

        # Build KDTree for spatial queries
        tree = cKDTree(centroids)

        # Find all pairs within distance threshold
        pairs = tree.query_pairs(r=self.distance_threshold, output_type="ndarray")

        # Edge case: no pairs within threshold
        if len(pairs) == 0:
            return image

        # Union-Find with path compression using static method
        parent = {label: label for label in labels}

        # Apply union-find to pairs
        for i, j in pairs:
            label_i = labels[i]
            label_j = labels[j]
            root_i = self._find_root(label_i, parent)
            root_j = self._find_root(label_j, parent)
            if root_i != root_j:
                # Merge to smaller label for stability
                if root_i < root_j:
                    parent[root_j] = root_i
                else:
                    parent[root_i] = root_j

        # Build merge mapping using static method find_root
        merge_map = {}
        for label in labels:
            merge_map[label] = self._find_root(label, parent)

        # Apply merge mapping and relabel consecutively
        merged_objmap = self._apply_merge_map(objmap, merge_map)
        relabeled = self._relabel_consecutive(merged_objmap)

        # Write result
        image.objmap[:] = relabeled

        return image
