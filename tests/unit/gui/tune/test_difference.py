"""Unit tests for the A/B difference overlay backend (tune Curate B-i, task B2).

The Curate view can pin one trial and diff a second against it: which colonies
both pipelines agree on, which only A found, which only B found. ``_overlays``
provides the pure backend:

* :func:`difference_objects` — partition object ids into ``both`` / ``only_a`` /
  ``only_b`` via greedy IoU matching.
* :func:`render_difference` — color object outlines by agreement
  (both = grey, only-A = sky, only-B = orange) using the Okabe-Ito data palette.
* :func:`cell_disagreement` — count grid cells whose per-cell colony counts
  differ between two ``GridImage`` segmentations.
"""
from __future__ import annotations

import numpy as np

from phenotypic import ImagePipeline
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import GaussianBlur
from phenotypic.gui._design import OI_GREY, OI_ORANGE, OI_SKY
from phenotypic.gui.tune._overlays import (
    cell_disagreement,
    difference_objects,
    render_difference,
)


def _three_blob_objmap_a() -> np.ndarray:
    """Three separate 2x2 blobs labeled 1, 2, 3."""
    arr = np.zeros((6, 12), dtype=int)
    arr[1:3, 1:3] = 1
    arr[1:3, 5:7] = 2
    arr[1:3, 9:11] = 3
    return arr


def _two_blob_plus_extra_objmap_b() -> np.ndarray:
    """Blobs matching A.1 and A.2, plus a disjoint B-only blob labeled 9."""
    arr = np.zeros((6, 12), dtype=int)
    arr[1:3, 1:3] = 1
    arr[1:3, 5:7] = 2
    arr[4:6, 9:11] = 9
    return arr


def test_difference_objects_partitions_both_only_a_only_b() -> None:
    a = _three_blob_objmap_a()
    b = _two_blob_plus_extra_objmap_b()
    diff = difference_objects(a, b, tau=0.5)
    assert sorted(diff.both) == [1, 2]
    assert sorted(diff.only_a) == [3]
    assert sorted(diff.only_b) == [9]


def test_render_difference_returns_rgb_array() -> None:
    a = _three_blob_objmap_a()
    b = _two_blob_plus_extra_objmap_b()
    plate = np.zeros((6, 12, 3), dtype=np.uint8)
    out = render_difference(plate, a, b, tau=0.5)
    assert out.ndim == 3 and out.shape[2] in (3, 4)
    assert out.shape[:2] == a.shape


def test_render_difference_uses_okabe_ito_tokens() -> None:
    # The three agreement colors are the Okabe-Ito data-palette tokens, not
    # hard-coded hex — guard against drift away from _design.
    a = _three_blob_objmap_a()
    b = _two_blob_plus_extra_objmap_b()
    plate = np.zeros((6, 12, 3), dtype=np.uint8)
    out = render_difference(plate, a, b, tau=0.5)

    def _rgb(hex_color: str) -> tuple[int, int, int]:
        h = hex_color.lstrip("#")
        return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]

    present = {tuple(px) for px in out.reshape(-1, out.shape[2])[:, :3]}
    assert _rgb(OI_GREY) in present  # both
    assert _rgb(OI_SKY) in present  # only-A
    assert _rgb(OI_ORANGE) in present  # only-B


def test_cell_disagreement_counts_differing_cells() -> None:
    # Two real segmentations of the synthetic plate disagree on some grid cells
    # once a heavy blur merges neighbouring colonies.
    sharp = ImagePipeline(ops=[GaussianBlur(sigma=1.0), OtsuDetector()]).apply(
        load_synth_yeast_plate()
    )
    blurred = ImagePipeline(ops=[GaussianBlur(sigma=6.0), OtsuDetector()]).apply(
        load_synth_yeast_plate()
    )
    n = cell_disagreement(sharp, blurred)
    assert isinstance(n, int)
    assert n > 0
    # A segmentation never disagrees with itself.
    assert cell_disagreement(sharp, sharp) == 0


def test_cell_disagreement_handles_absent_sections() -> None:
    # A section absent from one series is a zero count, not a KeyError.
    class _FakeGrid:
        def __init__(self, counts: dict[int, int]) -> None:
            import pandas as pd

            self._series = pd.Series(counts, dtype=int)

        def get_section_counts(self, ascending: bool = False):  # noqa: ANN001
            return self._series

    class _FakeGridImage:
        def __init__(self, counts: dict[int, int]) -> None:
            self.grid = _FakeGrid(counts)

    # Cell 0 agrees (1 vs 1); cell 1 differs (2 vs 0=absent); cell 2 differs
    # (absent=0 vs 3). Two cells disagree.
    a = _FakeGridImage({0: 1, 1: 2})
    b = _FakeGridImage({0: 1, 2: 3})
    assert cell_disagreement(a, b) == 2
