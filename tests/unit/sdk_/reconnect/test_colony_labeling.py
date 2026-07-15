import numpy as np

from phenotypic.sdk_.reconnect import (
    filter_mask_by_overlap,
    markers_from_centroids,
    partition_by_grid_voronoi,
)


def test_filter_mask_by_overlap_drops_non_overlapping_cc():
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:5, 2:5] = True          # CC A — overlaps reference
    mask[12:15, 12:15] = True      # CC B — does not
    ref = np.zeros((20, 20), dtype=bool)
    ref[3, 3] = True
    out = filter_mask_by_overlap(mask, ref)
    assert out[3, 3]                # A kept
    assert not out[13, 13]         # B dropped


def test_markers_from_centroids_one_seed_per_label():
    objmap = np.zeros((20, 20), dtype=np.int32)
    objmap[2:6, 2:6] = 1
    objmap[12:16, 12:16] = 2
    markers = markers_from_centroids(objmap)
    assert markers.dtype == np.int32
    assert set(np.unique(markers)) == {0, 1, 2}
    assert int(markers[np.array([3, 4]).mean().round().astype(int), 3]) or markers[markers > 0].size == 2


def test_partition_by_grid_voronoi_labels_two_blobs():
    mask = np.zeros((20, 40), dtype=bool)
    mask[8:12, 4:8] = True
    mask[8:12, 32:36] = True
    markers = np.zeros((20, 40), dtype=np.int32)
    markers[10, 6] = 1
    markers[10, 34] = 2
    labels = partition_by_grid_voronoi(markers, mask)
    assert set(np.unique(labels[mask])) == {1, 2}
    assert labels[10, 6] == 1 and labels[10, 34] == 2
