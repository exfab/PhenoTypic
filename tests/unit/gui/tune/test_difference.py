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


class _FakeGrid:
    def __init__(self, counts: dict[int, int]) -> None:
        import pandas as pd

        self._series = pd.Series(counts, dtype=int)

    def get_section_counts(self, ascending: bool = False):  # noqa: ANN001
        return self._series


class _FakeGridImage:
    def __init__(self, counts: dict[int, int]) -> None:
        self.grid = _FakeGrid(counts)


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

    # Decoded HERE rather than with `_design.hex_to_rgb`, on purpose: the
    # code under test uses that function, so sharing it would let a bug in
    # it make both sides agree and this test pass. An independent witness
    # is the point -- do not dedupe these two lines.
    def _rgb(hex_color: str) -> tuple[int, int, int]:
        h = hex_color.lstrip("#")
        return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]

    present = {tuple(px) for px in out.reshape(-1, out.shape[2])[:, :3]}
    assert _rgb(OI_GREY) in present  # both
    assert _rgb(OI_SKY) in present  # only-A
    assert _rgb(OI_ORANGE) in present  # only-B


def test_render_difference_bounds_output_to_max_dim() -> None:
    # Full-res difference arrays are a browser memory/perf hazard (a 4000x6000
    # plate is ~72 MB per go.Image, cached at full res in the LRU). The caller
    # passes max_dim to clamp the longest side; the plate is anti-aliased and
    # BOTH objmaps are downscaled label-aware (nearest-neighbor) and
    # consistently, so the diff still matches on the downscaled maps.
    big_a = np.zeros((1200, 1600), dtype=int)
    big_a[100:300, 100:300] = 1
    big_a[100:300, 900:1100] = 2
    big_b = np.zeros((1200, 1600), dtype=int)
    big_b[100:300, 100:300] = 1  # agrees with A.1
    plate = np.zeros((1200, 1600, 3), dtype=np.uint8)

    out = render_difference(plate, big_a, big_b, tau=0.5, max_dim=640)
    assert out.ndim == 3 and out.shape[2] in (3, 4)
    # The longest spatial side is clamped to max_dim.
    assert max(out.shape[:2]) <= 640
    # Aspect ratio is roughly preserved (1600x1200 -> ~640x480).
    assert out.shape[0] < out.shape[1]


def test_render_difference_max_dim_none_is_full_res() -> None:
    # max_dim=None keeps the legacy full-resolution behaviour (B-i contract:
    # un-downscaled for outline correctness when the caller wants full res).
    a = _three_blob_objmap_a()
    b = _two_blob_plus_extra_objmap_b()
    plate = np.zeros((6, 12, 3), dtype=np.uint8)
    out = render_difference(plate, a, b, tau=0.5, max_dim=None)
    assert out.shape[:2] == a.shape


def test_cell_disagreement_counts_differing_cells() -> None:
    a = _FakeGridImage({0: 1, 1: 1, 2: 2})
    b = _FakeGridImage({0: 1, 1: 0, 2: 3})
    n = cell_disagreement(a, b)
    assert isinstance(n, int)
    assert n == 2
    # A segmentation never disagrees with itself.
    assert cell_disagreement(a, a) == 0


def test_cell_disagreement_handles_absent_sections() -> None:
    # A section absent from one series is a zero count, not a KeyError.
    # Cell 0 agrees (1 vs 1); cell 1 differs (2 vs 0=absent); cell 2 differs
    # (absent=0 vs 3). Two cells disagree.
    a = _FakeGridImage({0: 1, 1: 2})
    b = _FakeGridImage({0: 1, 2: 3})
    assert cell_disagreement(a, b) == 2
