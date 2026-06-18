"""Euclidean Voronoi partition with connectivity-based fragment correction.

Replaces watershed-based territory assignment with a two-phase approach:

1. Pure Euclidean Voronoi partition (each pixel to nearest seed)
2. Connected-component-based fragment reassignment
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, label as ndi_label
from scipy.sparse import coo_matrix
from skimage.measure import label as skimage_label
from skimage.segmentation import expand_labels


def euclidean_voronoi_assign(
        markers: np.ndarray,
        mask: np.ndarray,
        restrict_to_seeded_cc: bool = True,
) -> np.ndarray:
    """Assign masked pixels to nearest marker via Euclidean Voronoi partition.

    Args:
        markers: 2D int32 array with seed labels at marker positions.
        mask: Binary mask restricting the assignment region.
        restrict_to_seeded_cc: When True (default), only mask connected
            components that contain at least one seed are labeled;
            disconnected mask regions with no seed remain 0.  When False,
            seeds are used globally — every mask pixel gets the label of
            its nearest seed regardless of connectivity.

    Returns:
        2D int32 labeled array. Each masked pixel has the label of its
        nearest marker by Euclidean distance. Pixels outside mask are 0.
        When *restrict_to_seeded_cc* is True, pixels in seedless
        connected components are also 0.
    """
    if mask.dtype != np.bool_:
        mask = mask > 0

    seed_mask = markers > 0
    if restrict_to_seeded_cc:
        effective_markers = markers.copy()
        effective_markers[~mask] = 0
        seed_mask = effective_markers > 0
    else:
        effective_markers = markers

    if not seed_mask.any():
        return np.zeros(mask.shape, dtype=np.int32)

    _, nearest_idx = distance_transform_edt(~seed_mask, return_indices=True)

    voronoi_labels = effective_markers[
        nearest_idx[0], nearest_idx[1]
    ].astype(np.int32)
    voronoi_labels[~mask] = 0

    if restrict_to_seeded_cc:
        cc_map, n_cc = ndi_label(mask)  # type: ignore[misc]
        seeded_ccs = np.unique(cc_map[seed_mask])
        has_seed = np.zeros(n_cc + 1, dtype=bool)
        has_seed[seeded_ccs] = True
        voronoi_labels[~has_seed[cc_map]] = 0

    return voronoi_labels


def _mode(arr: np.ndarray) -> int:
    """Return the most frequent value in a 1-D integer array (ignoring 0)."""
    counts = np.bincount(arr)
    counts[0] = 0
    return int(np.argmax(counts))


def connectivity_correct_labels(
        voronoi_labels: np.ndarray,
        mask: np.ndarray,
        markers: np.ndarray,
) -> np.ndarray:
    """Correct Voronoi misassignments using fragment-based iterative fill.

    Separates seed pixels from unseeded fragments, then iteratively
    assigns each fragment the mode label of its neighborhood until
    convergence.

    Args:
        voronoi_labels: Label map from ``euclidean_voronoi_assign``.
        mask: Binary foreground mask (same as used for Voronoi).
        markers: Seed marker array (same as used for Voronoi).

    Returns:
        Corrected label map with fragment reassignments applied.
    """
    if mask.dtype != np.bool_:
        mask = mask > 0

    seed_mask = markers > 0

    # Fragment map — per-voronoi-value CC labeling.
    # skimage.measure.label connects only same-value neighbors; 0 is bg.
    fragment_voronoi = np.where(seed_mask, 0, voronoi_labels)
    fragment_map = skimage_label(fragment_voronoi, connectivity=1)
    n_frags = int(fragment_map.max())

    # Seeded map — labels only at seed pixels; fragments zeroed.
    seeded_map = voronoi_labels.copy()
    seeded_map[fragment_map > 0] = 0
    # Anchor seeds even outside mask (voronoi_labels is 0 there).
    seeded_map[seed_mask] = markers[seed_mask]

    if n_frags == 0:
        seeded_map[~mask] = 0
        return seeded_map

    # COO index of all fragment pixels (built once).
    frag_coo = coo_matrix(fragment_map)
    frag_label = np.zeros(n_frags + 1, dtype=np.int32)  # LUT: frag_id → label

    unfilled = set(range(1, n_frags + 1))

    while unfilled:
        # Expand filled regions by 1 pixel.
        expanded = expand_labels(seeded_map, distance=1)

        # Look up expanded values at all fragment pixel positions.
        expanded_at_frags = expanded[frag_coo.row, frag_coo.col]

        filled_this_pass = []
        for frag_id in sorted(unfilled):
            frag_px = frag_coo.data == frag_id
            vals = expanded_at_frags[frag_px]
            nonzero = vals[vals > 0]
            if nonzero.size == 0:
                continue
            frag_label[frag_id] = _mode(nonzero)
            filled_this_pass.append(frag_id)

        if not filled_this_pass:
            break

        # LUT paint all newly-filled fragments in one vectorized step.
        newly_filled_px = np.isin(frag_coo.data, filled_this_pass)
        assigned = frag_label[frag_coo.data[newly_filled_px]]
        seeded_map[
            frag_coo.row[newly_filled_px],
            frag_coo.col[newly_filled_px],
        ] = assigned

        unfilled -= set(filled_this_pass)

    seeded_map[~mask] = 0
    return seeded_map
