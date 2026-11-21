"""Re-derive the memory figures quoted in the time-series feasibility report.

The report (docs/superpowers/reports/2026-11-21-time-series-operations/feasibility.md)
argues that a series operation which holds every frame of a plate in memory is not
viable at plate resolution, while one that streams spatial tiles across all frames is.
This script recomputes those numbers from the per-pixel layer sizes alone, so the
argument does not rest on the prose.

Per-pixel footprint of one in-memory ``Image`` (``_image_data_manager.py:575-597``):
``rgb`` is 3 channels of uint8 (1 B) or uint16 (2 B); ``gray`` and ``detect_mat`` are
float32 (4 B each); ``objmap`` is a sparse CSC matrix and is excluded here, as are
lazily cached colour spaces (``image.color.Lab`` etc.), which only add to the total.
A CLI worker also retains a copy of the decoded RGB (``Image._retain_original``,
``_cli_process_single.py:322-326``) unless ``--drop-originals`` is given; that copy
is reported separately because it is a CLI choice, not part of the Image model.

The frame size (24 MP) and series length (48 frames) are ILLUSTRATIVE assumptions, not
measured properties of any dataset. Change them below to match a real experiment.

Depends only on the standard library and never imports ``phenotypic``. Exits non-zero
if any asserted relationship fails.
"""

from __future__ import annotations

import sys

MB = 1_000_000
GB = 1_000_000_000

RGB_CHANNELS = 3
FLOAT32_BYTES = 4


def bytes_per_pixel(rgb_bytes_per_channel: int) -> int:
    """Dense bytes per pixel for rgb + gray + detect_mat."""
    return RGB_CHANNELS * rgb_bytes_per_channel + 2 * FLOAT32_BYTES


def retained_original_bytes_per_pixel(rgb_bytes_per_channel: int) -> int:
    """The CLI worker's retained decoded-RGB snapshot."""
    return RGB_CHANNELS * rgb_bytes_per_channel


def frame_bytes(height: int, width: int, rgb_bytes_per_channel: int) -> int:
    return height * width * bytes_per_pixel(rgb_bytes_per_channel)


def tile_stack_bytes(n_frames: int, tile: int) -> int:
    """One float32 detect_mat tile read from every frame of a series."""
    return n_frames * tile * tile * FLOAT32_BYTES


def check_memory_claims() -> list[str]:
    failures: list[str] = []
    height, width = 4000, 6000          # illustrative 24 MP plate image
    n_frames = 48                       # illustrative: hourly for two days
    tile = 1024                         # illustrative spatial tile edge

    per_px_u8, per_px_u16 = bytes_per_pixel(1), bytes_per_pixel(2)
    f8, f16 = frame_bytes(height, width, 1), frame_bytes(height, width, 2)
    s8, s16 = n_frames * f8, n_frames * f16
    t = tile_stack_bytes(n_frames, tile)

    print(f"bytes/pixel  uint8 RGB: {per_px_u8}   uint16 RGB: {per_px_u16}")
    with_original = (per_px_u8 + retained_original_bytes_per_pixel(1),
                     per_px_u16 + retained_original_bytes_per_pixel(2))
    print(f"bytes/pixel incl. CLI retained original  uint8: {with_original[0]}"
          f"   uint16: {with_original[1]}")
    print(f"one frame    uint8: {f8 / MB:.0f} MB   uint16: {f16 / MB:.0f} MB")
    print(f"{n_frames} frames    uint8: {s8 / GB:.1f} GB   uint16: {s16 / GB:.1f} GB")
    print(f"{n_frames}x{tile}^2 float32 tile stack: {t / MB:.0f} MB")
    # ImageOperation.apply copies by default (abc_/_image_operation.py
    # _apply_to_single_image), so a naive per-frame apply inside a series op
    # transiently doubles a frame.
    print(f"whole series + one transient frame copy, uint8: {(s8 + f8) / GB:.1f} GB")

    if per_px_u8 != 11 or per_px_u16 != 14:
        failures.append("per-pixel footprint is not 11 B (uint8) / 14 B (uint16)")
    if with_original != (14, 20):
        failures.append("with retained original, footprint is not 14 B / 20 B")
    if round(f8 / MB) != 264 or round(f16 / MB) != 336:
        failures.append("24 MP frame is not 264 MB (uint8) / 336 MB (uint16)")
    if not (12.6 * GB < s8 < 12.8 * GB):
        failures.append("48-frame uint8 series is not ~12.7 GB")
    if not (16.0 * GB < s16 < 16.2 * GB):
        failures.append("48-frame uint16 series is not ~16.1 GB")
    if round(t / MB) != 201:
        failures.append("48 x 1024^2 float32 tile stack is not ~201 MB")
    # The load-bearing comparison: streaming tiles needs well under a single
    # frame's memory, while holding the series needs ~48 frames' worth.
    if not t < f8:
        failures.append("tile stack is not smaller than one frame")
    if not s8 / t > 50:
        failures.append("whole-series residency is not >50x the tile stack")
    return failures


def check_reference_cache_thrash() -> list[str]:
    """Read counts for windowed raw-frame references through ReferenceContext.

    ReferenceContext keeps a process-wide LRU of 2 loaded images
    (_reference_context.py, _IMAGE_CACHE_SIZE = 2). An op that reads the k raw
    frames preceding each target is simulated two ways:

    - best case: ONE worker processes a series in frame order, so consecutive
      targets share their windows through the cache;
    - worst case: every target lands in a different worker process (the CLI
      spreads images across loky workers and SLURM array tasks), so no cache
      entry is ever reused and each target reads its whole window.
    """
    failures: list[str] = []
    cache_size = 2

    def in_order_reads(n_frames: int, k: int) -> int:
        cache: list[int] = []
        reads = 0
        for target in range(n_frames):
            for frame in [f for f in range(target - k, target) if f >= 0]:
                if frame in cache:
                    cache.remove(frame)
                    cache.append(frame)
                    continue
                reads += 1
                cache.append(frame)
                if len(cache) > cache_size:
                    cache.pop(0)
        return reads

    def no_reuse_reads(n_frames: int, k: int) -> int:
        return sum(min(target, k) for target in range(n_frames))

    n = 48
    rows = {k: (in_order_reads(n, k), no_reuse_reads(n, k)) for k in (1, 3, 4, n)}
    for k, (best, worst) in rows.items():
        label = "k=n" if k == n else f"k={k}"
        print(f"{label:>4} window over {n} frames: best {best:>5} reads, worst {worst:>5} reads")

    # k <= 3 in frame order: the 2-entry LRU already holds the overlap, so each
    # target costs one new read (the frame just before it).
    if rows[1][0] != n - 1 or rows[3][0] != n - 1:
        failures.append("k in {1, 3}, in order, should read each frame once")
    # k = 4 in order: the overlap (3 frames) exceeds the cache, so it thrashes.
    # Only the first few targets, whose windows are truncated at frame 0, get
    # any reuse; the total is within k reads of the no-reuse count.
    if not rows[4][1] - 4 <= rows[4][0] <= rows[4][1]:
        failures.append("k=4 in order should thrash to ~the no-reuse count")
    # A whole-series window recomputed per target is quadratic in reads in
    # either case: n(n-1)/2 with no reuse, within n of that in order.
    quadratic = n * (n - 1) // 2
    if rows[n][1] != quadratic or not (quadratic - n <= rows[n][0] <= quadratic):
        failures.append("k=n per-target reads should be ~n(n-1)/2")
    return failures


def run_all_checks() -> int:
    failures = check_memory_claims() + check_reference_cache_thrash()
    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    print("OK" if not failures else f"{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run_all_checks())
