from __future__ import annotations

import numpy as np

from phenotypic import Image
from phenotypic.detect import WatershedDetector


def test_watershed_detector_does_not_relabel_when_relabel_false(monkeypatch):
    """``relabel=False`` preserves raw watershed labels by skipping relabel()."""
    detect_mat = np.zeros((32, 32), dtype=float)
    detect_mat[8:12, 8:12] = 1.0
    detect_mat[20:24, 20:24] = 1.0
    image = Image(arr=detect_mat)

    def fail_relabel(*args, **kwargs):
        raise AssertionError("relabel should not be called")

    monkeypatch.setattr(type(image.objmap), "relabel", fail_relabel)

    WatershedDetector(min_size=1, relabel=False).apply(image)
