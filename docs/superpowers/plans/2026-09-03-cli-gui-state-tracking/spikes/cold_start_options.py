"""Same question, without the ordering bias.

The first run measured B (hash, no overlay) and then C (stat) in one process
over the SAME paths. B ran against a cold GPFS metadata cache and C against a
warm one, so C's advantage is partly just having gone second -- the same
page-cache confound P0's S-2 spike was rewritten to avoid.

Fix, borrowed from that spike: give each lane a DISJOINT half of the markers,
seeded and shuffled, so both read paths the other never touched. Then run each
lane in both orders and report all four numbers.
"""
import hashlib
import json
import random
import sys
import time
from pathlib import Path

root = Path(
    "/bigdata/exfab/anguy344/projects/ucr_029_e_d_Maresca/data/results/2026-08-11"
)
markers = sorted(
    (root / ".phenotypic" / "progress" / "image_complete").rglob("*.json")
)
random.Random(20260904).shuffle(markers)
half = len(markers) // 2
lanes = {"first": markers[:half], "second": markers[half:]}


def _resolve(ms):
    no_ov, all3 = [], []
    for m in ms:
        try:
            arts = json.loads(m.read_text(encoding="utf-8")).get("artifacts") or {}
        except (OSError, ValueError):
            continue
        for name, desc in arts.items():
            rel = desc.get("path")
            if not isinstance(rel, str):
                continue
            p = root / rel
            p = p / "zarr.json" if name == "store" else p
            all3.append(p)
            if name != "overlay":
                no_ov.append(p)
    return no_ov, all3


def _hash(paths):
    t = time.perf_counter()
    total = 0
    for p in paths:
        try:
            with p.open("rb") as fh:
                h = hashlib.sha256()
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
                    total += len(chunk)
        except OSError:
            pass
    return time.perf_counter() - t, total


def _stat(paths):
    t = time.perf_counter()
    for p in paths:
        try:
            st = p.stat()
            _ = (st.st_size, st.st_mtime_ns)
        except OSError:
            pass
    return time.perf_counter() - t


which = sys.argv[1]  # "B" or "C"
lane = sys.argv[2]   # "first" or "second"
no_ov, all3 = _resolve(lanes[lane])

if which == "B":
    sec, byts = _hash(no_ov)
    per = sec / max(len(no_ov), 1) * 1000
    print(f"B lane={lane} seconds={sec:7.1f} files={len(no_ov)} "
          f"bytes={byts/1e9:.2f}GB per_file_ms={per:.1f}")
else:
    sec = _stat(all3)
    per = sec / max(len(all3), 1) * 1000
    print(f"C lane={lane} seconds={sec:7.1f} files={len(all3)} "
          f"per_file_ms={per:.1f}")
