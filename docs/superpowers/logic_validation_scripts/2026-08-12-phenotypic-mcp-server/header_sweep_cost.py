#!/usr/bin/env python3
"""Is §10.6.1's parent header sweep cheap enough for `deploy_plan`'s work class?

THE CLAIM UNDER TEST
--------------------
Spec §10.6.1 gives `deploy_plan {scope:"full"}` a two-tier check whose first tier
"**Always** — header sweep | `W0`, no decode, no slot | Read dimensions, bit depth,
and channel count from every parent image header."

§1.6.1 defines `W0` as returning **in under one second**. The sweep's cost is
therefore load-bearing: if reading N image headers off a shared filesystem takes
materially longer than a second, the sweep is not `W0` no matter how the human
gate around it is arranged, and §5.3's class has to say so.

This script re-derives that number from scratch. It does NOT import `phenotypic`,
and it does not import tifffile/PIL either — it parses the TIFF/PNG/JPEG headers
directly from bytes. Two reasons: the repo convention for logic-validation scripts
is stdlib + numpy/scipy only, and more usefully, a hand-rolled parser measures the
*floor* — the irreducible I/O — rather than whatever a library happens to layer on
top. A real implementation using tifffile can only be slower than this, so a
failure here is conclusive while a pass is a lower bound.

WHY COLD CACHE IS THE NUMBER THAT MATTERS
-----------------------------------------
A warm-cache sweep measures RAM. `deploy_plan` runs against a parent the agent has
typically NOT just read — that is the whole point of promoting from a subset — so
the cold path is the honest one. We evict per-file with `posix_fadvise(DONTNEED)`,
which needs no privileges, rather than `drop_caches`, which needs root.

USAGE
    python3 header_sweep_cost.py <image-dir> [--repeats 3] [--budget-s 1.0]

Exits non-zero if the cold-cache sweep exceeds the budget.
"""

from __future__ import annotations

import argparse
import os
import statistics
import struct
import sys
import time
from pathlib import Path

IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

# How much of the front of a file we read to find the header. TIFF IFDs can sit
# anywhere, so this is a hint, not a guarantee -- see _read_tiff_header.
PROBE_BYTES = 65536


class HeaderUnreadable(Exception):
    """The bytes did not parse as a header we understand."""


def _read_png_header(fh) -> tuple[int, int, int, int]:
    """(width, height, bit_depth, channels) from a PNG IHDR chunk."""
    fh.seek(8)  # signature
    length, ctype = struct.unpack(">I4s", fh.read(8))
    if ctype != b"IHDR":
        raise HeaderUnreadable("first chunk is not IHDR")
    w, h, depth, color_type = struct.unpack(">IIBB", fh.read(10))
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color_type)
    if channels is None:
        raise HeaderUnreadable(f"unknown PNG color type {color_type}")
    return w, h, depth, channels


def _read_jpeg_header(fh) -> tuple[int, int, int, int]:
    """(width, height, bit_depth, channels) from the first SOF marker."""
    fh.seek(2)  # SOI
    while True:
        b = fh.read(1)
        if not b:
            raise HeaderUnreadable("no SOF marker before EOF")
        if b != b"\xff":
            continue
        marker = fh.read(1)
        while marker == b"\xff":  # fill bytes
            marker = fh.read(1)
        m = marker[0]
        if m in (0xD8, 0xD9) or 0xD0 <= m <= 0xD7:
            continue
        (seg_len,) = struct.unpack(">H", fh.read(2))
        # SOF0..SOF15 except the non-frame markers DHT(C4), JPG(C8), DAC(CC)
        if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
            depth, h, w, ncomp = struct.unpack(">BHHB", fh.read(6))
            return w, h, depth, ncomp
        fh.seek(seg_len - 2, os.SEEK_CUR)


# TIFF tags we care about, and how many bytes each field type occupies.
_TIFF_TAGS = {256: "width", 257: "height", 258: "bits", 277: "channels"}
_TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}


def _read_tiff_header(fh) -> tuple[int, int, int, int]:
    """(width, height, bit_depth, channels) from the first IFD.

    Handles both byte orders and both classic TIFF and BigTIFF. Only the four
    tags above are decoded; everything else is skipped by arithmetic, which is
    what keeps this to one or two reads per file.
    """
    fh.seek(0)
    order_raw = fh.read(2)
    if order_raw == b"II":
        e = "<"
    elif order_raw == b"MM":
        e = ">"
    else:
        raise HeaderUnreadable("not a TIFF byte-order mark")

    (version,) = struct.unpack(e + "H", fh.read(2))
    if version == 42:
        big = False
        (ifd_off,) = struct.unpack(e + "I", fh.read(4))
    elif version == 43:
        big = True
        offsize, zero = struct.unpack(e + "HH", fh.read(4))
        if offsize != 8 or zero != 0:
            raise HeaderUnreadable("malformed BigTIFF header")
        (ifd_off,) = struct.unpack(e + "Q", fh.read(8))
    else:
        raise HeaderUnreadable(f"unknown TIFF version {version}")

    fh.seek(ifd_off)
    if big:
        (n_entries,) = struct.unpack(e + "Q", fh.read(8))
        entry_size, count_fmt = 20, "Q"
    else:
        (n_entries,) = struct.unpack(e + "H", fh.read(2))
        entry_size, count_fmt = 12, "I"

    if n_entries > 4096:
        raise HeaderUnreadable(f"implausible IFD entry count {n_entries}")

    found: dict[str, int] = {}
    entries = fh.read(n_entries * entry_size)
    for i in range(n_entries):
        chunk = entries[i * entry_size : (i + 1) * entry_size]
        if len(chunk) < entry_size:
            break
        tag, ftype = struct.unpack_from(e + "HH", chunk, 0)
        name = _TIFF_TAGS.get(tag)
        if name is None:
            continue
        (count,) = struct.unpack_from(e + count_fmt, chunk, 4)
        value_off = 8 if big else 8
        size = _TYPE_SIZE.get(ftype, 0) * count
        inline_cap = 8 if big else 4
        if size <= inline_cap:
            if ftype in (3, 8):
                (val,) = struct.unpack_from(e + "H", chunk, value_off)
            elif ftype in (4, 9):
                (val,) = struct.unpack_from(e + "I", chunk, value_off)
            elif ftype in (16, 17):
                (val,) = struct.unpack_from(e + "Q", chunk, value_off)
            else:
                continue
        else:
            # Out-of-line value. This is the case that costs an extra seek, and
            # BitsPerSample for an RGB image lands here -- worth counting honestly.
            (ptr,) = struct.unpack_from(e + ("Q" if big else "I"), chunk, value_off)
            here = fh.tell()
            fh.seek(ptr)
            raw = fh.read(2 if ftype in (3, 8) else 4)
            fh.seek(here)
            val = struct.unpack(e + ("H" if len(raw) == 2 else "I"), raw)[0]
        found[name] = val

    if "width" not in found or "height" not in found:
        raise HeaderUnreadable("IFD carried no ImageWidth/ImageLength")
    return (
        found["width"],
        found["height"],
        found.get("bits", 8),
        found.get("channels", 1),
    )


def read_header(path: Path) -> tuple[int, int, int, int]:
    with open(path, "rb") as fh:
        magic = fh.read(4)
        if magic.startswith(b"\x89PNG"):
            return _read_png_header(fh)
        if magic.startswith(b"\xff\xd8"):
            return _read_jpeg_header(fh)
        if magic[:2] in (b"II", b"MM"):
            return _read_tiff_header(fh)
        raise HeaderUnreadable(f"unrecognized magic {magic!r}")


def evict(paths: list[Path]) -> bool:
    """Drop each file from the page cache. Returns False if unsupported."""
    if not hasattr(os, "posix_fadvise"):
        return False
    for p in paths:
        try:
            fd = os.open(p, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    return True


def sweep(paths: list[Path]) -> tuple[float, int, int]:
    """Read every header. Returns (elapsed_s, n_ok, n_failed)."""
    ok = bad = 0
    t0 = time.perf_counter()
    for p in paths:
        try:
            read_header(p)
            ok += 1
        except (HeaderUnreadable, OSError, struct.error):
            bad += 1
    return time.perf_counter() - t0, ok, bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image_dir", type=Path)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--budget-s", type=float, default=1.0,
                    help="the W0 ceiling from spec 1.6.1")
    args = ap.parse_args()

    paths = sorted(
        p for p in args.image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        print(f"FAIL: no images under {args.image_dir}", file=sys.stderr)
        return 1

    print(f"Directory : {args.image_dir}")
    print(f"Images    : {len(paths)}")
    total_bytes = sum(p.stat().st_size for p in paths)
    print(f"Total size: {total_bytes / 2**30:.1f} GiB "
          f"({total_bytes / len(paths) / 2**20:.1f} MiB/image)")
    fs = os.popen(f"df -PT {args.image_dir!s} 2>/dev/null | awk 'NR==2{{print $2}}'").read().strip()
    print(f"Filesystem: {fs or 'unknown'}")
    print(f"Budget    : {args.budget_s:.2f} s  (spec 1.6.1 W0 ceiling)")
    print()

    # Correctness first. A fast sweep that read nothing proves nothing.
    elapsed, ok, bad = sweep(paths)
    print(f"Parse check: {ok} headers parsed, {bad} failed")
    if ok == 0:
        print("FAIL: parsed no headers at all -- the timings below would be "
              "measuring open() and nothing else.", file=sys.stderr)
        return 1
    if bad > len(paths) * 0.05:
        print(f"FAIL: {bad}/{len(paths)} headers unparseable (>5%); "
              "the timing is not representative of a real sweep.", file=sys.stderr)
        return 1
    dims = read_header(paths[0])
    print(f"  e.g. {paths[0].name}: {dims[0]}x{dims[1]}, "
          f"{dims[2]}-bit, {dims[3]}ch")
    print()

    can_evict = evict(paths[:1])
    if not can_evict:
        print("WARNING: posix_fadvise unavailable; COLD numbers are not cold.")

    cold, warm = [], []
    for i in range(args.repeats):
        if can_evict:
            evict(paths)
        c, _, _ = sweep(paths)
        w, _, _ = sweep(paths)  # immediately after: page cache is hot
        cold.append(c)
        warm.append(w)
        print(f"  run {i + 1}:  cold {c:7.3f} s    warm {w:7.3f} s")

    cold_med, warm_med = statistics.median(cold), statistics.median(warm)
    print()
    print(f"COLD median : {cold_med:7.3f} s  "
          f"({cold_med / len(paths) * 1e3:.2f} ms/image)")
    print(f"WARM median : {warm_med:7.3f} s  "
          f"({warm_med / len(paths) * 1e3:.2f} ms/image)")
    print()

    # Extrapolate to the spec's worked example so the answer is quotable.
    per_image = cold_med / len(paths)
    for n in (480, 5000, 50000):
        print(f"  extrapolated to {n:6d} images: {per_image * n:8.2f} s")
    print()

    if cold_med <= args.budget_s:
        print(f"VERDICT: WITHIN BUDGET. The cold sweep of {len(paths)} headers "
              f"took {cold_med:.3f} s, under the {args.budget_s:.2f} s W0 ceiling.")
        print("  Spec 10.6.1's tier-1 sweep is defensible as W0 AT THIS SCALE.")
        print(f"  Note the extrapolation above: it stops being W0 somewhere past "
              f"{int(args.budget_s / per_image)} images, so the class is "
              f"dataset-size-dependent and the spec should say so.")
        return 0

    print(f"VERDICT: OVER BUDGET. The cold sweep took {cold_med:.3f} s against a "
          f"{args.budget_s:.2f} s ceiling.")
    print("  Spec 5.3 cannot call deploy_plan {scope:'full'} a W0 tool. Either")
    print("  reclassify it (already done as W1 -- this measurement confirms it was")
    print("  necessary, not merely tidy), or move the sweep off the plan path.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
