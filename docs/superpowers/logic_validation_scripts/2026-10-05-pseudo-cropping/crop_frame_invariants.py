"""Re-derive the numeric invariants the pseudo-cropping spec rests on.

Spec: docs/superpowers/specs/2026-10-05-pseudo-cropping/design.md

Independent witness: stdlib + numpy only, never imports ``phenotypic``. Exits
non-zero on the first failed check.

Checks:
    1. Offset composition (spec §3.2): slicing a slice of an image equals one
       slice of the original at the summed offset.
    2. Pad/slice identity (spec §5.1, §5.3): zero-padding an ROI into its canvas
       and slicing it back returns the ROI byte-for-byte, and every pixel
       outside the ROI is zero.
    3. Coordinate regeneration (spec §4): a coordinate measured in the ROI plus
       the frame offset addresses the same pixel in the canvas.
    4. Pad-overflow rule (spec §3.4): PadImage keeps the frame iff the shifted
       window still fits inside the canvas.
"""

from __future__ import annotations

import sys

import numpy as np

RNG = np.random.default_rng(20261005)
CANVAS = (37, 53)


def _window(offset, shape):
    return (
        slice(offset[0], offset[0] + shape[0]),
        slice(offset[1], offset[1] + shape[1]),
    )


def _is_valid(offset, shape, canvas):
    return (
        offset[0] >= 0
        and offset[1] >= 0
        and offset[0] + shape[0] <= canvas[0]
        and offset[1] + shape[1] <= canvas[1]
    )


def check_offset_composition() -> None:
    canvas = RNG.integers(0, 255, size=(*CANVAS, 3), dtype=np.uint8)
    for _ in range(500):
        r0 = int(RNG.integers(0, CANVAS[0] - 2))
        c0 = int(RNG.integers(0, CANVAS[1] - 2))
        r1 = int(RNG.integers(r0 + 2, CANVAS[0] + 1))
        c1 = int(RNG.integers(c0 + 2, CANVAS[1] + 1))
        first = canvas[r0:r1, c0:c1]
        h, w = first.shape[:2]
        a0 = int(RNG.integers(0, h - 1))
        b0 = int(RNG.integers(0, w - 1))
        a1 = int(RNG.integers(a0 + 1, h + 1))
        b1 = int(RNG.integers(b0 + 1, w + 1))
        second = first[a0:a1, b0:b1]
        offset = (r0 + a0, c0 + b0)
        assert np.array_equal(
            second, canvas[_window(offset, second.shape)]
        ), "offset composition failed"


def check_pad_slice_identity() -> None:
    for dtype in (np.uint8, np.uint16, np.float32):
        for _ in range(200):
            h = int(RNG.integers(1, CANVAS[0] + 1))
            w = int(RNG.integers(1, CANVAS[1] + 1))
            offset = (
                int(RNG.integers(0, CANVAS[0] - h + 1)),
                int(RNG.integers(0, CANVAS[1] - w + 1)),
            )
            roi = (RNG.random((h, w)) * 200 + 1).astype(dtype)
            padded = np.zeros(CANVAS, dtype=dtype)
            padded[_window(offset, roi.shape)] = roi
            back = padded[_window(offset, roi.shape)]
            assert back.tobytes() == roi.tobytes(), "pad/slice identity failed"
            outside = np.ones(CANVAS, dtype=bool)
            outside[_window(offset, roi.shape)] = False
            assert not padded[outside].any(), "non-zero pixel outside the ROI"


def check_coordinate_regeneration() -> None:
    canvas = np.arange(CANVAS[0] * CANVAS[1]).reshape(CANVAS)
    offset = (5, 11)
    roi = canvas[_window(offset, (20, 30))]
    for rr in range(roi.shape[0]):
        for cc in range(roi.shape[1]):
            assert roi[rr, cc] == canvas[rr + offset[0], cc + offset[1]], (
                "Bbox + Frame_Offset does not address the same canvas pixel"
            )


def check_pad_overflow_rule() -> None:
    shape = (10, 12)
    offset = (4, 6)
    assert _is_valid(offset, shape, CANVAS)
    # Padding within the cropped margin: still inside the canvas -> kept.
    pad_top, pad_left, pad_bottom, pad_right = 4, 6, 2, 3
    new_offset = (offset[0] - pad_top, offset[1] - pad_left)
    new_shape = (shape[0] + pad_top + pad_bottom, shape[1] + pad_left + pad_right)
    assert _is_valid(new_offset, new_shape, CANVAS), "in-canvas pad must keep frame"
    # Padding past the original edge: offset goes negative -> dropped.
    new_offset = (offset[0] - 5, offset[1])
    new_shape = (shape[0] + 5, shape[1])
    assert not _is_valid(new_offset, new_shape, CANVAS), "overflow must drop frame"
    # Padding past the far edge: exceeds the canvas -> dropped.
    new_shape = (CANVAS[0] - offset[0] + 1, shape[1])
    assert not _is_valid(offset, new_shape, CANVAS), "far-edge overflow must drop"


CHECKS = (
    check_offset_composition,
    check_pad_slice_identity,
    check_coordinate_regeneration,
    check_pad_overflow_rule,
)


def run_all_checks() -> int:
    for check in CHECKS:
        try:
            check()
        except AssertionError as exc:
            print(f"FAIL {check.__name__}: {exc}")
            return 1
        print(f"ok   {check.__name__}")
    return 0


if __name__ == "__main__":
    sys.exit(run_all_checks())
