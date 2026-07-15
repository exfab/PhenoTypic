"""Grid-Voronoi labeling helpers for filamentous-colony detection.

Pure array functions extracted from FilamentousFungiDetector. No Image,
operation, or GUI types (see package CLAUDE.md import contract).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import center_of_mass
from skimage.measure import label

from ..branch_pathfinding import connectivity_correct_labels, euclidean_voronoi_assign


def filter_mask_by_overlap(mask: np.ndarray, reference_mask: np.ndarray) -> np.ndarray:
    """Retain only connected components of ``mask`` that overlap ``reference_mask``.

    Args:
        mask: Binary mask to filter (2D boolean or uint8).
        reference_mask: Binary mask defining valid regions.

    Returns:
        Filtered binary mask, same dtype/shape as ``mask``.
    """
    labeled = label(mask)
    min_h = min(mask.shape[0], reference_mask.shape[0])
    min_w = min(mask.shape[1], reference_mask.shape[1])
    intersection = labeled[:min_h, :min_w] * reference_mask[:min_h, :min_w]
    overlapping_labels = np.unique(intersection[intersection > 0])
    max_label = int(labeled.max())
    keep = np.zeros(max_label + 1, dtype=labeled.dtype)
    keep[overlapping_labels] = overlapping_labels
    return keep[labeled].astype(mask.dtype, copy=False)


def markers_from_centroids(objmap: np.ndarray) -> np.ndarray:
    """Create Voronoi seed markers at each positive label's centroid.

    Args:
        objmap: Labeled integer array (each object a unique positive ID).

    Returns:
        2D int32 marker array with one seed per centroid.
    """
    labels = np.unique(objmap)
    labels = labels[labels > 0]
    markers = np.zeros(objmap.shape, dtype=np.int32)
    for marker_id, lbl in enumerate(labels, start=1):
        com = center_of_mass(objmap == lbl)
        r = min(int(round(com[0])), objmap.shape[0] - 1)
        c = min(int(round(com[1])), objmap.shape[1] - 1)
        markers[r, c] = marker_id
    return markers


def partition_by_grid_voronoi(markers: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Voronoi-partition ``mask`` pixels by nearest ``markers`` seed and correct connectivity."""
    voronoi_map = euclidean_voronoi_assign(
        markers=markers, mask=mask, restrict_to_seeded_cc=False,
    )
    return connectivity_correct_labels(
        voronoi_labels=voronoi_map, mask=mask, markers=markers,
    )
