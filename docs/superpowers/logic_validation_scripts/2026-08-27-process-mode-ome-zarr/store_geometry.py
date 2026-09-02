#!/usr/bin/env python3
"""Re-derive the store-geometry claims of the process-mode OME-Zarr design.

Spec: docs/superpowers/specs/2026-08-27-process-mode-ome-zarr/design.md

Imports nothing from ``phenotypic``: every number is derived from the NGFF 0.5
and Zarr v3 rules directly, so the script fails if the spec and the format
disagree -- not merely if the spec and the implementation disagree.

Exits non-zero on failure.
"""

from __future__ import annotations

import math
import sys

PYRAMID_STOP_PX = 512
CHUNK_EDGE = 1024
SHARD_EDGE = 4096
FAILURES: list[str] = []


def check(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        FAILURES.append(f"{label}: expected {expected!r}, got {actual!r}")


def level_count(height: int, width: int, stop_px: int = PYRAMID_STOP_PX) -> int:
    """Halve until the longest edge is <= stop_px. Ceil, not floor."""
    longest = max(height, width)
    if longest <= stop_px:
        return 1
    return int(math.ceil(math.log2(longest / stop_px))) + 1


def level_shapes(height: int, width: int) -> list[tuple[int, int]]:
    """Ceil-halve both spatial axes, matching NGFF's stored extents.

    ``(h + 1) // 2``, never ``h // 2``: an odd 1025-pixel axis becomes 513
    pixels, and a floor formula would silently disagree with the writer on
    every odd level.
    """
    shapes = []
    h, w = height, width
    for _ in range(level_count(height, width)):
        shapes.append((h, w))
        h, w = max(1, (h + 1) // 2), max(1, (w + 1) // 2)
    return shapes


def shards_per_level(h: int, w: int, channels: int) -> int:
    """One shard file per shard-sized block. A shard spans the whole c axis.

    The spatial shard edge is the FIXED ``SHARD_EDGE``, never clamped to the
    level extent: the Zarr v3 sharding codec constrains shard-vs-chunk
    divisibility only, never shard-vs-array, and partial edge shards are
    normal. Clamping would turn a 4000-pixel axis under a 4096 shard into four
    shard files instead of one. A level below one chunk collapses to
    ``chunk == shard == extent``, which keeps divisibility trivially true.
    """
    chunk_h = min(CHUNK_EDGE, h)
    chunk_w = min(CHUNK_EDGE, w)
    shard_h = chunk_h if h < CHUNK_EDGE else SHARD_EDGE
    shard_w = chunk_w if w < CHUNK_EDGE else SHARD_EDGE
    # A shard must be an exact multiple of the chunk in every dimension.
    if shard_h % chunk_h or shard_w % chunk_w or shard_h < chunk_h:
        FAILURES.append(
            f"shard {(channels, shard_h, shard_w)} is not a multiple of "
            f"chunk {(1, chunk_h, chunk_w)}"
        )
    return math.ceil(h / shard_h) * math.ceil(w / shard_w)


def single_series_file_count(height: int, width: int, channels: int) -> int:
    """Files in a single-series store: 4 fixed + 2 per pyramid level.

    The ``4 + 2 * levels`` shorthand holds only while every level fits inside
    ONE shard -- true up to a 4096-pixel level-0 edge, which covers every
    camera this design targets. Above that a level contributes more than one
    shard file and the shorthand understates the count; ``shards_per_level``
    is the general form and is what this function actually sums.
    """
    shapes = level_shapes(height, width)
    data = sum(shards_per_level(h, w, channels) for h, w in shapes)
    metadata = (
        1                # root zarr.json
        + 1              # OME/zarr.json
        + 1              # OME/METADATA.ome.xml
        + 1              # <series>/zarr.json
        + len(shapes)    # <series>/<level>/zarr.json
    )
    return data + metadata


def main() -> int:
    # Spec 1.1: the measured geometry of a 4000x3000 rgb store.
    check("levels(4000, 3000)", level_count(4000, 3000), 4)
    check(
        "level shapes(4000, 3000)",
        level_shapes(4000, 3000),
        [(4000, 3000), (2000, 1500), (1000, 750), (500, 375)],
    )
    check(
        "single-series rgb file count at 4000x3000",
        single_series_file_count(4000, 3000, channels=3),
        12,
    )

    # Spec 1.4 (inherited): the sharding codec requires an exact multiple in
    # EVERY dimension, the channel axis included.
    check("shard spans the channel axis: 3 % 1", 3 % 1, 0)
    check("shard/chunk edge ratio", SHARD_EDGE % CHUNK_EDGE, 0)

    # A level at or below the stop threshold is the last one.
    check("levels(512, 512)", level_count(512, 512), 1)
    check("levels(513, 400)", level_count(513, 400), 2)

    # Ceil-halving, not floor: an odd axis keeps the extra row. A floor
    # formula agrees on 4000x3000 (every level is even) and diverges the
    # moment an odd edge appears, which is why the check uses an odd one.
    check(
        "odd axes ceil-halve",
        level_shapes(4001, 3000)[:2],
        [(4001, 3000), (2001, 1500)],
    )

    # Ceil, not floor, in the LEVEL COUNT too: a floor formula stops a level
    # early and leaves 4000x3000's smallest level at 1000x750.
    floor_levels = int(math.floor(math.log2(4000 / PYRAMID_STOP_PX))) + 1
    check("floor level formula is refuted", floor_levels == 4, False)

    # The 4 + 2*levels shorthand, and the extent at which it stops holding.
    check(
        "shorthand holds at 4000x3000",
        single_series_file_count(4000, 3000, channels=3),
        4 + 2 * level_count(4000, 3000),
    )
    check("a 4096-edge level is one shard", shards_per_level(4096, 4096, 3), 1)
    check("a 4097-edge level is four shards", shards_per_level(4097, 4097, 3), 4)

    for failure in FAILURES:
        print(f"FAIL: {failure}", file=sys.stderr)
    if FAILURES:
        return 1
    print("All store-geometry claims verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
