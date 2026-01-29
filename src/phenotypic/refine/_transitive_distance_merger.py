from __future__ import annotations

from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from phenotypic import Image

import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import regionprops_table
import pandas as pd

from ..abc_ import ObjectRefiner


class TransitiveDistanceMerger(ObjectRefiner):
    """Merge colonies with centroids within distance threshold using transitive closure.

    Intuition:
        Fragmented colony detections from uneven illumination or pigment heterogeneity
        often appear as multiple small regions clustered together. When colonies are
        close enough spatially, they likely represent fragments of a single organism.
        This operation uses transitive closure to merge all connected fragments: if
        colony A is near B, and B is near C, then A, B, and C all merge into a single
        detection, even if A and C are not directly within the threshold distance.

    Why this is useful for agar plates:
        Aggressive thresholding or contrast enhancement can fragment a single colony
        into multiple labeled regions. Merging by proximity reconstructs the true
        colony shape, correcting inflated colony counts and improving downstream
        area and intensity measurements. This is particularly effective for colonies
        with heterogeneous pigmentation or those growing on plates with uneven lighting.

    Use cases:
        - Repairing fragmented detections from watershed over-segmentation where a
          single colony is split into multiple touching regions.
        - Consolidating micro-fragments (satellite spots) around main colonies caused
          by thresholding artifacts or agar texture.
        - Post-processing after aggressive noise removal that leaves fragmented edges.
        - Correcting detections in images with harsh shadows or glare that create
          internal voids within colony masks.

    Caveats:
        - Setting distance_threshold too high incorrectly merges distinct colonies
          that happen to be close together, reducing count accuracy and creating
          artificially large merged objects.
        - Transitive closure can cascade through long chains of nearby objects,
          potentially merging objects that are far apart if connected through
          intermediate fragments. Choose threshold conservatively to avoid this.
        - Does not consider size, shape, or intensity similarity—purely spatial
          proximity. May merge unrelated artifacts if they happen to be close.
        - More computationally expensive than NearestNeighborMerger for large
          numbers of objects due to pairwise distance calculations and union-find
          operations.
        - Cannot distinguish between legitimate separate colonies and fragments of
          a single colony based on spatial information alone.

    Attributes:
        distance_threshold (float): Maximum centroid-to-centroid distance in pixels
            for merging colonies. Lower values (5-15 pixels) are safer and produce
            fewer false merges, suitable for high-resolution images with well-separated
            colonies. Higher values (20-50 pixels) merge more aggressively, useful for
            fragmented detections in noisy images but risk combining distinct colonies.
            Typical range: 10-30 pixels depending on imaging resolution and colony
            density.

    Examples:
        Merge fragmented colonies using transitive distance threshold:

        >>> from phenotypic.refine import TransitiveDistanceMerger
        >>> from phenotypic import Image
        >>> from phenotypic.detect import OtsuDetector
        >>> image = Image.imread('fragmented_plate.jpg')
        >>> detected = OtsuDetector().apply(image)
        >>> merger = TransitiveDistanceMerger(distance_threshold=25.0)
        >>> merged = merger.apply(detected)  # doctest: +SKIP
        >>> print(f"Merged {detected.objmap[:].max()} fragments into {merged.objmap[:].max()} colonies")  # doctest: +SKIP
    """

    def __init__(self, distance_threshold: float = 20.0):
        """Initialize the merger.

        Args:
            distance_threshold (float): Maximum centroid-to-centroid distance in
                pixels for merging colonies. Increasing this value merges more
                fragments but risks combining distinct colonies. Should be smaller
                than the minimum expected distance between separate colonies.

        Raises:
            ValueError: If distance_threshold is not positive.
        """
        if distance_threshold <= 0:
            raise ValueError("distance_threshold must be positive")
        self.distance_threshold = distance_threshold

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
            Image object with merged objmap and unchanged RGB/gray/enh_gray.
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
