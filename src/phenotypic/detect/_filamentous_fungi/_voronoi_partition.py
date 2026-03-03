"""Euclidean Voronoi partition with connectivity-based fragment correction.

Replaces watershed-based territory assignment with a two-phase approach:

1. Pure Euclidean Voronoi partition (each pixel to nearest seed)
2. Connected-component-based fragment reassignment
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, label as ndi_label


def euclidean_voronoi_assign(
        markers: np.ndarray,
        mask: np.ndarray,
) -> np.ndarray:
    """Assign masked pixels to nearest marker via Euclidean Voronoi partition.

    Only mask connected components that contain at least one seed are
    labeled.  Disconnected mask regions with no seed remain 0, matching
    the connectivity-aware behavior of watershed.

    Args:
        markers: 2D int32 array with seed labels at marker positions.
        mask: Binary mask restricting the assignment region.

    Returns:
        2D int32 labeled array. Each masked pixel has the label of its
        nearest marker by Euclidean distance. Pixels outside mask or in
        seedless connected components are 0.
    """
    effective_markers = markers.copy()
    effective_markers[~mask > 0] = 0

    seed_mask = effective_markers > 0
    if not seed_mask.any():
        return np.zeros(mask.shape, dtype=np.int32)

    _, nearest_idx = distance_transform_edt(~seed_mask, return_indices=True)

    voronoi_labels = effective_markers[
        nearest_idx[0], nearest_idx[1]
    ].astype(np.int32)
    voronoi_labels[~mask] = 0

    # Zero out labels in mask CCs that contain no seeds
    cc_map, n_cc = ndi_label(mask)  # type: ignore[misc]
    seeded_ccs = np.unique(cc_map[seed_mask])
    has_seed = np.zeros(n_cc + 1, dtype=bool)
    has_seed[seeded_ccs] = True
    voronoi_labels[~has_seed[cc_map]] = 0

    return voronoi_labels


def connectivity_correct_labels(
        voronoi_labels: np.ndarray,
        mask: np.ndarray,
        markers: np.ndarray,
) -> np.ndarray:
    """Correct Voronoi misassignments using foreground connectivity.

    Runs connected-component analysis on the foreground mask and checks
    which seeds each component contains:

    - **Case A** (single seed): reassign entire component to that seed.
    - **Case B** (multiple seeds): keep Voronoi labels unchanged.
    - **Case C** (no seed): keep Voronoi labels (nearest-seed by EDT).

    Args:
        voronoi_labels: Label map from ``euclidean_voronoi_assign``.
        mask: Binary foreground mask (same as used for Voronoi).
        markers: Seed marker array (same as used for Voronoi).

    Returns:
        Corrected label map with fragment reassignments applied.
    """
    corrected = voronoi_labels.copy()

    cc_map, n_cc = ndi_label(mask)  # type: ignore[misc]

    seed_rows, seed_cols = np.nonzero(markers > 0)
    cc_to_seeds: dict[int, set[int]] = {}
    for r, c in zip(seed_rows, seed_cols):
        cc_id = int(cc_map[r, c])
        if cc_id > 0:
            cc_to_seeds.setdefault(cc_id, set()).add(int(markers[r, c]))

    for cc_id in range(1, n_cc + 1):
        seeds = cc_to_seeds.get(cc_id, set())

        if len(seeds) == 1:
            # Case A: entire CC belongs to the single reachable seed
            cc_pixels = cc_map == cc_id
            corrected[cc_pixels] = next(iter(seeds))

        # Case B (multiple seeds): keep Voronoi labels unchanged
        # Case C (no seed): keep Voronoi labels (nearest by EDT)

    return corrected
