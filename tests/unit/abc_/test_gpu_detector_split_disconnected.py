"""Tests for GpuDetector.split_disconnected_labels (connectivity relabel).

A SAM-style detector can paint ONE instance label across two spatially
disconnected blobs (e.g. a single predicted mask covering distant regions, or
a tile-merged objmap). When ``split_disconnected_labels`` is set (the default),
``_write_object_output`` relabels the instance objmap by connected components so
each connected region becomes its own instance. Torch-free: a hand-crafted fake
detector returns the merged objmap directly.
"""

import numpy as np
from pydantic import PrivateAttr

from phenotypic._core._image import Image
from phenotypic.abc_ import GpuDetector


class _MergedObjmapDetector(GpuDetector):
    """Returns a fixed objmap whose label 1 spans two disconnected blobs.

    Two 2x2 blocks at opposite corners both carry label 1 — exactly the
    "distant objects, same label" defect the relabel step fixes.
    """

    _loaded: bool = PrivateAttr(default=False)

    def _ensure_model_loaded(self) -> None:
        self._loaded = True

    def _infer_one(self, sample):
        h, w = sample.shape[:2]
        objmap = np.zeros((h, w), dtype=np.uint16)
        objmap[0:2, 0:2] = 1  # top-left blob
        objmap[h - 2:h, w - 2:w] = 1  # bottom-right blob, SAME label
        return objmap


def _tiny_image() -> Image:
    return Image(arr=np.zeros((10, 10, 3), dtype=np.uint8))


class TestFields:
    def test_defaults(self):
        det = _MergedObjmapDetector()
        assert det.split_disconnected_labels is True
        assert det.connectivity == 2

    def test_fields_are_serializable_pydantic_fields(self):
        assert "split_disconnected_labels" in GpuDetector.model_fields
        assert "connectivity" in GpuDetector.model_fields


class TestSplitBehavior:
    def test_default_splits_disconnected_label_into_two_instances(self):
        det = _MergedObjmapDetector()  # split on by default
        out = det.apply(_tiny_image(), inplace=False)
        objmap = out.objmap[:]
        # Two disconnected blobs that shared label 1 are now two instances.
        assert out.num_objects == 2
        labels = set(np.unique(objmap)) - {0}
        assert labels == {1, 2}

    def test_disabled_keeps_single_shared_label(self):
        det = _MergedObjmapDetector(split_disconnected_labels=False)
        out = det.apply(_tiny_image(), inplace=False)
        # Without the relabel the two distant blobs keep the one label.
        assert out.num_objects == 1
        assert set(np.unique(out.objmap[:])) - {0} == {1}

    def test_connectivity_one_still_splits_diagonal_gap(self):
        # The two blobs are far apart, so 4- vs 8-connectivity both split them.
        det = _MergedObjmapDetector(connectivity=1)
        out = det.apply(_tiny_image(), inplace=False)
        assert out.num_objects == 2
