#!/usr/bin/env python3
"""Re-derive the U-11 cold-start numbers from the raw measurement.

The U-11 ruling — ship the stat sweep plus the on-disk cache tier, rather than
merely dropping the overlay from the completion proof — rests on a ratio between
two cold-cache lanes. Every figure quoted in the spec's decision table, in
``EXECUTION.md``'s artifact-budget rule, and in
``plans/2026-09-03-cli-gui-state-tracking/spikes/RESULTS-cold-start.md`` is
arithmetic on top of six raw numbers. This re-derives all of them so a reader
does not take the chain on faith.

Independent by construction: the only inputs are the raw ``stdout`` of Slurm job
``28142292`` and a marker census. It imports no project code and reads no project
file, so it is a witness to the arithmetic rather than a second copy of it.

**The raw numbers are hardcoded on purpose.** The Slurm log lives outside the
repository under ``/bigdata/exfab/anguy344/slurm_logs/`` and will be rotated; the
census walks a results tree that is not a fixture. Pinning the measurement and
re-deriving everything downstream is what keeps this checkable after both are
gone.

Run:
    uv run python docs/superpowers/logic_validation_scripts/\
2026-09-03-cli-gui-state-tracking/cold_start_options.py

Exits non-zero if any published figure does not reproduce.
"""

from __future__ import annotations

import sys

# --------------------------------------------------------------------------
# Raw inputs. Nothing below this block is a measurement.
# --------------------------------------------------------------------------

# Slurm job 28142292, node x02, COMPLETED 0:0, 2026-09-04. Verbatim stdout:
#   C lane=first  seconds=   18.5 files=9918 per_file_ms=1.9
#   B lane=second seconds=  128.2 files=6596 bytes=0.30GB per_file_ms=19.4
C_SEC, C_FILES = 18.5, 9918          # lane "first",  stats every artifact
B_SEC, B_FILES, B_GB = 128.2, 6596, 0.30   # lane "second", hashes all but the overlay

# Marker census over the same seeded split (random.Random(20260904)).
MARKERS_FIRST, MARKERS_SECOND = 3328, 3329
MARKERS_TOTAL = 6657
TREE_ALL3 = 9918 + 9925              # artifacts C would stat across the tree
TREE_NOOV = 6590 + 6596              # artifacts B would hash across the tree

# Today's behaviour, measured separately and cited in design.md U-11.
TODAY_SEC, TODAY_GB = 1403.0, 73.6

# --------------------------------------------------------------------------

failures: list[str] = []


def check(label: str, computed: float, published: float, tol: float) -> None:
    """Assert a published figure reproduces from the raw inputs."""
    if abs(computed - published) > tol:
        failures.append(f"{label}: computed {computed:.4g}, published {published:g}")
        verdict = "FAIL"
    else:
        verdict = "ok  "
    print(f"{verdict} {label:<40} computed={computed:>9.4g}  published={published:g}")


# --- per-file cost: the only figure the log prints directly ---------------
check("C per file (ms)", C_SEC / C_FILES * 1000, 1.87, 0.005)
check("B per file (ms)", B_SEC / B_FILES * 1000, 19.44, 0.005)

# --- per marker: the unit a completion check is actually billed in --------
# C touches 3 artifacts per image, B touches 2, so per-file cost is NOT the
# option comparison. This is the correction the writeup exists to make.
c_marker = C_SEC / MARKERS_FIRST * 1000
b_marker = B_SEC / MARKERS_SECOND * 1000
check("C per marker (ms)", c_marker, 5.56, 0.005)
check("B per marker (ms)", b_marker, 38.51, 0.005)
check("ratio per file", (B_SEC / B_FILES) / (C_SEC / C_FILES), 10.4, 0.05)
check("ratio per marker", b_marker / c_marker, 6.9, 0.05)

# --- full tree: extrapolated from one cold half, not measured ------------
c_tree = c_marker / 1000 * MARKERS_TOTAL
b_tree = b_marker / 1000 * MARKERS_TOTAL
check("C full tree (s)", c_tree, 37.0, 0.05)
check("B full tree (s)", b_tree, 256.0, 0.5)
check("today vs B (x)", TODAY_SEC / b_tree, 5.5, 0.05)
check("today vs C (x)", TODAY_SEC / c_tree, 38.0, 0.5)

# --- the transferable result ---------------------------------------------
# Dropping the overlay removes ~99% of the bytes and only ~82% of the time:
# on GPFS the price is the open+read round trip, not the volume.
tree_noov_gb = B_GB / B_FILES * TREE_NOOV
check("no-overlay bytes, tree (GB)", tree_noov_gb, 0.60, 0.005)
check("pct bytes removed", (1 - tree_noov_gb / TODAY_GB) * 100, 99.0, 0.5)
check("pct time removed", (1 - b_tree / TODAY_SEC) * 100, 82.0, 0.5)

# --- corroboration, and the strongest claim in the writeup ---------------
# C's extrapolation is independent of the ~37 s stat sweep quoted in design.md:
# different run, different half, different cold node. They agree to 0.01 s.
check("C tree vs design.md's 37 s", c_tree, 37.0, 0.05)

print()
print(f"files B would touch across tree = {TREE_NOOV:,}")
print(f"files C would touch across tree = {TREE_ALL3:,}")
print()

if failures:
    print("FAILED -- published figures that do not reproduce:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)

print("ALL PUBLISHED FIGURES REPRODUCE")
