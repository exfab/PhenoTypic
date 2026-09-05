#!/usr/bin/env python3
"""Re-derive the numeric claims behind the crop-path fix in the Scatter tab spec.

The spec asserts three things a reader would otherwise take on faith:

1. ``uint16 -> uint8`` via ``np.ndarray.astype`` is a modular reduction
   (``value & 0xFF``), so it is NON-MONOTONIC: a source delta of +1 across a
   256 boundary becomes an output delta of -255.
2. Over the measured source range of a real colony crop
   (19061..38171 on object 24 of ``d000466_280_003``), that produces ~74.6
   wraps of the 256 cycle, which is why the output reads as noise rather than
   as a dim or washed-out image.
3. Scaling against a fixed ``(lo, hi)`` range and clipping is monotonic
   non-decreasing, which is the invariant the regression test pins.

Depends only on the standard library + numpy. Never imports ``phenotypic``:
this file must stay runnable against the spec alone.

Exit code 0 = every claim reproduced. Non-zero = the spec and the arithmetic
have drifted apart.
"""

from __future__ import annotations

import sys

import numpy as np

# Measured from the migrated subset, object 24 of
# results/7-24-26_redo_full/zarr/d000466_280_003_2026-07-26_06-34-47.ome.zarr
CROP_LO, CROP_HI = 19061, 38171
# Per-image display range taken from the whole of rgb/4 in that store.
IMAGE_LO, IMAGE_HI = 20511, 44047

failures: list[str] = []


def check(name: str, condition: bool, detail: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}: {detail}")
    if not condition:
        failures.append(name)


def truncate(values: np.ndarray) -> np.ndarray:
    """What the current code does: a bare narrowing cast."""
    return values.astype(np.uint8)


def scale(values: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """What the fix does: scale against a fixed range, then clip."""
    out = (values.astype(np.float64) - lo) / (hi - lo) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


print("1. the cast is a modular reduction, not a scale")
probes = np.array([18315, 18175, 18176, 31783], dtype=np.uint16)
got = truncate(probes)
expected = probes & 0xFF
check(
    "astype equals bitwise-and 0xFF",
    bool(np.array_equal(got, expected)),
    f"{probes.tolist()} -> {got.tolist()}",
)
check(
    "crossing a 256 boundary inverts a +1 step",
    int(truncate(np.array([18176], np.uint16))[0])
    - int(truncate(np.array([18175], np.uint16))[0])
    == -255,
    "18175 -> 255, 18176 -> 0, so +1 in source becomes -255 in output",
)

print("\n2. the measured crop range spans many wraps")
span = CROP_HI - CROP_LO
wraps = span / 256.0
check(
    "the colony crop spans > 70 wraps of the 256 cycle",
    wraps > 70.0,
    f"{CROP_LO}..{CROP_HI} = {span} levels = {wraps:.1f} wraps",
)

ramp = np.arange(CROP_LO, CROP_HI + 1, dtype=np.uint16)
truncated = truncate(ramp).astype(np.int16)
descents = int((np.diff(truncated) < 0).sum())
check(
    "a monotonic source ramp descends once per wrap under truncation",
    descents == int(np.floor(wraps)) or descents == int(np.ceil(wraps)),
    f"{descents} descending steps across the ramp (~{wraps:.1f} expected)",
)

print("\n3. scaling is monotonic non-decreasing — the regression invariant")
scaled = scale(ramp, IMAGE_LO, IMAGE_HI).astype(np.int16)
check(
    "no descending step anywhere in the scaled ramp",
    bool((np.diff(scaled) >= 0).all()),
    f"min step {int(np.diff(scaled).min())}, max step {int(np.diff(scaled).max())}",
)
check(
    "truncation violates the same invariant",
    not bool((np.diff(truncated) >= 0).all()),
    f"{descents} descending steps — this is what the regression test catches",
)
check(
    "scaling clips rather than wrapping when the crop exceeds the image range",
    int(scale(np.array([IMAGE_HI + 5000], np.uint16), IMAGE_LO, IMAGE_HI)[0]) == 255,
    "a value above hi saturates at 255 instead of folding to a low number",
)

print("\n4. neighbour-delta separation is large enough to be a usable check")
rng = np.random.default_rng(0)
smooth = np.clip(
    np.cumsum(rng.normal(0, 12, 4096)) + (CROP_LO + CROP_HI) / 2, CROP_LO, CROP_HI
).astype(np.uint16)
d_trunc = float(np.abs(np.diff(truncate(smooth).astype(np.int16))).mean())
d_scaled = float(np.abs(np.diff(scale(smooth, IMAGE_LO, IMAGE_HI).astype(np.int16))).mean())
check(
    "truncation inflates the mean neighbour delta by more than 5x",
    d_trunc > 5 * d_scaled,
    f"truncated {d_trunc:.1f} vs scaled {d_scaled:.1f}",
)

print()
if failures:
    print(f"FAILED: {len(failures)} claim(s) did not reproduce: {', '.join(failures)}")
    sys.exit(1)
print("All claims reproduced.")
sys.exit(0)
