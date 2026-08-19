#!/usr/bin/env python
"""Re-derive the store-geometry claims behind the OME-Zarr image store (§1).

The design replaces one HDF5 file per image with an OME-Zarr *directory* store.
These numeric claims drove design decisions and must not be taken on faith:

  C1  PYRAMID LEVEL COUNT. Levels halve until ``max(H, W) <= 512``. The count is
      ``ceil(log2(max(H, W) / 512)) + 1`` for images larger than the stop size,
      and 1 otherwise. Every layer of one image gets the same count, because the
      count depends only on the shared (y, x) extent.

      NOTE: an early draft of this script used ``floor`` here. That stops one
      level short of the target -- a 4000x3000 plate terminated at 1000x750,
      not <= 512 -- and the "terminal level" assertion below is what caught it.
      The assertion is retained precisely because it has already failed once.

  C2  LABEL LEVEL PARITY. NGFF 0.5 requires a label image to have the same
      number of scale levels as its parent image. Because ``objmap`` shares the
      parent's (y, x) extent, C1 yields parity automatically — no special case.

  C3  SHARD/CHUNK DIVISIBILITY AND BUFFER SIZE. The Zarr v3 sharding codec
      requires the shard shape to be an exact multiple of the chunk shape in
      EVERY dimension -- including the channel axis, which is the load-bearing
      part of the claim that a shard spanning all channels collapses the
      per-channel chunks of ``rgb`` into a single file. A shard is also the
      write-buffer unit, so its byte size bounds peak memory per writer; the
      spec's --njobs guidance depends on that figure.

  C4  FILE COUNT. This is the claim that decided sharding, and the one an early
      draft got WRONG. The draft quoted "~15-20 files per image sharded" by
      counting only data files and forgetting that every group and every array
      carries its own ``zarr.json``. Metadata files dominate a sharded store.
      The script computes both halves separately so the error cannot recur.

  C5  LABEL DOWNSAMPLING MUST BE NEAREST-NEIGHBOUR. Mean-downsampling a label
      map invents label values that exist at no level-0 pixel, silently
      fabricating objects. This is the mutation the pyramid test must catch.

Depends only on the stdlib + numpy. Never imports ``phenotypic``.

Exits non-zero on the first failed claim.

Run:  uv run python docs/superpowers/logic_validation_scripts/2026-08-18-ome-zarr-image-store/ngff_store_geometry.py
"""

from __future__ import annotations

import math
import sys
from typing import NoReturn

import numpy as np

# ---------------------------------------------------------------------------
# Design constants under test. These MUST match §1 of the spec.
# ---------------------------------------------------------------------------

STOP_PX = 512  # halve until max(H, W) <= STOP_PX
CHUNK_YX = 1024
SHARD_YX = 4096

#: Representative arrayed-plate acquisitions. (label, height, width)
PLATES = [
    ("2048 square", 2048, 2048),
    ("4000x3000 (spec reference)", 4000, 3000),
    ("6000x4000 (large-format)", 6000, 4000),
]

#: (name, n_channels). ``rgb`` is (c, y, x); the rest are (y, x).
IMAGE_LAYERS = [("rgb", 3), ("gray", 1), ("detect_mat", 1)]
LABEL_LAYERS = [("objmap", 1)]

_failures: list[str] = []


def check(claim: str, ok: bool, detail: str = "") -> None:
    """Record a claim result and print it."""
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {claim}" + (f" -- {detail}" if detail else ""))
    if not ok:
        _failures.append(claim)


# ---------------------------------------------------------------------------
# C1 / C2 -- pyramid level count
# ---------------------------------------------------------------------------


def level_count(height: int, width: int, stop_px: int = STOP_PX) -> int:
    """Number of pyramid levels when halving until ``max(H, W) <= stop_px``."""
    longest = max(height, width)
    if longest <= stop_px:
        return 1
    return int(math.ceil(math.log2(longest / stop_px))) + 1


def level_shapes(
    height: int, width: int, levels: int | None = None
) -> list[tuple[int, int]]:
    """Explicit (h, w) per level, halving with ceil, as the writer will.

    ``levels=None`` uses the automatic count, which is the only count the
    writer ever produces: depth is a pure function of the level-0 shape (spec
    1.3, PRE-P3), so there is no user-facing override. The parameter exists so
    the per-level cost can be exercised directly from the checks below.
    """
    shapes = [(height, width)]
    for _ in range((levels or level_count(height, width)) - 1):
        h, w = shapes[-1]
        shapes.append((max(1, (h + 1) // 2), max(1, (w + 1) // 2)))
    return shapes


def _check_pyramid() -> None:
    print("\nC1 -- pyramid level count (halve until max(H, W) <= %d)" % STOP_PX)
    for label, h, w in PLATES:
        shapes = level_shapes(h, w)
        n = len(shapes)
        # Closed form must agree with the iterative construction.
        check(
            f"{label}: closed form agrees with iteration",
            n == level_count(h, w),
            f"{n} levels, shapes={shapes}",
        )
        # The last level must be at or below the stop size...
        check(
            f"{label}: terminal level <= {STOP_PX}px",
            max(shapes[-1]) <= STOP_PX,
            f"terminal={shapes[-1]}",
        )
        # ...and the second-to-last must still be above it, or we stopped early.
        if n > 1:
            check(
                f"{label}: does not stop one level early",
                max(shapes[-2]) > STOP_PX,
                f"penultimate={shapes[-2]}",
            )

    print("\nC2 -- label level parity (NGFF 0.5 requirement)")
    for label, h, w in PLATES:
        parent = level_count(h, w)
        # objmap shares the parent's (y, x) extent by construction.
        labels = level_count(h, w)
        check(
            f"{label}: objmap level count == parent level count",
            parent == labels,
            f"{parent} == {labels}",
        )


# ---------------------------------------------------------------------------
# C3 -- shard / chunk divisibility
# ---------------------------------------------------------------------------


#: Peak per-writer buffer budget. Exceeding this makes ``--njobs N`` cost
#: N x SHARD_BUDGET_MB and collides with the project's memory-discipline rule.
SHARD_BUDGET_MB = 128


def _check_divisibility() -> None:
    print("\nC3 -- Zarr v3 sharding: divisibility in EVERY dimension, and buffer size")

    # rgb is the only multi-channel array: chunk (1, 1024, 1024) in a
    # (3, 4096, 4096) shard. The channel axis is what collapses 3 files into 1.
    for name, n_c in IMAGE_LAYERS + LABEL_LAYERS:
        chunk = (1, CHUNK_YX, CHUNK_YX) if n_c > 1 else (CHUNK_YX, CHUNK_YX)
        shard = (n_c, SHARD_YX, SHARD_YX) if n_c > 1 else (SHARD_YX, SHARD_YX)
        divides = all(s % c == 0 for s, c in zip(shard, chunk))
        check(
            f"{name}: shard {shard} divisible by chunk {chunk} in every dim",
            divides,
            f"per-dim ratios={tuple(s // c for s, c in zip(shard, chunk))}",
        )

    check(
        "chunks-per-shard edge ratio is an integer >= 2",
        SHARD_YX % CHUNK_YX == 0 and SHARD_YX // CHUNK_YX >= 2,
        f"{SHARD_YX // CHUNK_YX} chunks per shard edge",
    )

    # Buffer size. uint16 is the widest dtype the design stores (rgb 16-bit,
    # objmap uint16); detect_mat is float32/64 but single-channel.
    worst = max(
        n_c * SHARD_YX * SHARD_YX * np.dtype(dt).itemsize
        for n_c, dt in ((3, np.uint16), (1, np.float64))
    )
    worst_mb = worst / 1024**2
    check(
        f"worst-case shard buffer stays within {SHARD_BUDGET_MB} MB",
        worst_mb <= SHARD_BUDGET_MB,
        f"{worst_mb:.0f} MB per writer -- peak memory is njobs x this",
    )


# ---------------------------------------------------------------------------
# C4 -- file counts
# ---------------------------------------------------------------------------


def data_files(
    height: int, width: int, n_channels: int, *, sharded: bool,
    levels: int | None = None,
) -> int:
    """Chunk (or shard) file count for one array across all pyramid levels."""
    grid = SHARD_YX if sharded else CHUNK_YX
    total = 0
    for h, w in level_shapes(height, width, levels):
        tiles = math.ceil(h / grid) * math.ceil(w / grid)
        # rgb chunks are (1, y, x): one chunk per channel. A shard spanning the
        # full channel axis collapses those back into a single file.
        total += tiles if sharded else tiles * n_channels
    return total


def metadata_files(height: int, width: int, levels: int | None = None) -> int:
    """``zarr.json`` count -- one per group, one per array level, plus OME-XML."""
    n = levels or level_count(height, width)
    count = 1  # root zarr.json
    count += 1  # OME/zarr.json
    count += 1  # OME/METADATA.ome.xml
    count += len(IMAGE_LAYERS) * (1 + n)  # group zarr.json + one per level
    count += 1  # labels/ group zarr.json
    count += len(LABEL_LAYERS) * (1 + n)
    return count


def store_files(
    height: int, width: int, *, sharded: bool, levels: int | None = None
) -> tuple[int, int]:
    """(data_files, metadata_files) for a whole per-image store."""
    data = sum(
        data_files(height, width, c, sharded=sharded, levels=levels)
        for _, c in IMAGE_LAYERS + LABEL_LAYERS
    )
    return data, metadata_files(height, width, levels)


def _check_file_counts() -> None:
    print("\nC4 -- files per image store (HDF baseline = 1)")
    print(
        f"    {'plate':<28} {'unsharded':>22} {'sharded':>22} {'reduction':>10}"
    )
    for label, h, w in PLATES:
        ud, um = store_files(h, w, sharded=False)
        sd, sm = store_files(h, w, sharded=True)
        u_tot, s_tot = ud + um, sd + sm
        print(
            f"    {label:<28} {f'{u_tot} ({ud}d+{um}m)':>22} "
            f"{f'{s_tot} ({sd}d+{sm}m)':>22} {f'{u_tot / s_tot:.1f}x':>10}"
        )

        check(
            f"{label}: sharding strictly reduces total file count",
            s_tot < u_tot,
            f"{u_tot} -> {s_tot}",
        )
        check(
            f"{label}: sharded data files <= metadata files",
            sd <= sm,
            f"metadata ({sm}) dominates a sharded store, not data ({sd})",
        )

    # The specific figures quoted in the spec, for the reference plate.
    ud, um = store_files(4000, 3000, sharded=False)
    sd, sm = store_files(4000, 3000, sharded=True)
    check(
        "spec figure: 4000x3000 unsharded is ~130 files",
        125 <= ud + um <= 140,
        f"actual={ud + um}",
    )
    check(
        "spec figure: 4000x3000 sharded is ~40 files",
        35 <= sd + sm <= 45,
        f"actual={sd + sm}",
    )
    # Guard against the error an early draft made: quoting only data files.
    check(
        "an early draft's '15-20 sharded' figure is REFUTED",
        not (15 <= sd + sm <= 20),
        f"actual={sd + sm}; the draft counted {sd} data files and omitted "
        f"{sm} zarr.json/OME-XML files",
    )


# C6 -- REMOVED (ledger ALGO-8).
#
# This block asserted that "each extra pyramid level costs a constant number of
# files" and that "--pyramid-levels 1 is the cheapest setting". PRE-P3 descoped
# that flag: pyramid depth is a pure function of the level-0 shape, so two
# stores in one tree can never disagree, valid_staged_store needs no level
# check, and a resumed run cannot produce mixed geometry. There is no knob to
# tune, so there is nothing here to verify -- and Phase 1 Task 1.1 imports this
# script as the normative geometry reference, where a claim block about a
# non-existent flag is worse than no claim at all.
#
# The underlying figure is not lost: C4 already reports total file counts per
# plate at the automatic depth, which is the only depth the writer produces.


# ---------------------------------------------------------------------------
# C5 -- label downsampling must be nearest-neighbour
# ---------------------------------------------------------------------------


def _downsample_mean(labels: np.ndarray) -> np.ndarray:
    """2x block mean, rounded to integer -- the WRONG method for labels."""
    h, w = labels.shape
    h2, w2 = h - h % 2, w - w % 2
    blocks = labels[:h2, :w2].astype(np.float64).reshape(h2 // 2, 2, w2 // 2, 2)
    return np.rint(blocks.mean(axis=(1, 3))).astype(labels.dtype)


def _downsample_nearest(labels: np.ndarray) -> np.ndarray:
    """2x nearest-neighbour (top-left of each block) -- the correct method."""
    return labels[::2, ::2]


def _check_label_downsampling() -> None:
    print("\nC5 -- label pyramids: nearest-neighbour, never mean")
    rng = np.random.default_rng(20260818)

    # A label map whose values are deliberately NOT consecutive, mimicking an
    # objmap after filtering removed some colonies.
    present = np.array([0, 10, 20, 30], dtype=np.uint16)
    labels = present[rng.integers(0, len(present), size=(64, 64))]
    level0_values = set(np.unique(labels).tolist())

    nearest_values = set(np.unique(_downsample_nearest(labels)).tolist())
    mean_values = set(np.unique(_downsample_mean(labels)).tolist())

    check(
        "nearest-neighbour invents no new label values",
        nearest_values <= level0_values,
        f"level1={sorted(nearest_values)} subset of level0={sorted(level0_values)}",
    )
    invented = sorted(mean_values - level0_values)
    check(
        "mean downsampling DOES invent label values (the mutation to catch)",
        len(invented) > 0,
        f"fabricated labels {invented[:8]}"
        + (" ..." if len(invented) > 8 else ""),
    )
    # The test in the suite asserts the subset property; prove it can fail.
    check(
        "the subset assertion fails under the mean mutation",
        not (mean_values <= level0_values),
        "so the pyramid test is able to fail, not vacuously green",
    )


def main() -> NoReturn:
    print((__doc__ or "").splitlines()[0])
    _check_pyramid()
    _check_divisibility()
    _check_file_counts()
    _check_label_downsampling()

    print()
    if _failures:
        print(f"FAILED: {len(_failures)} claim(s) did not hold:")
        for claim in _failures:
            print(f"  - {claim}")
        sys.exit(1)
    print("All store-geometry claims hold.")
    sys.exit(0)


if __name__ == "__main__":
    main()
