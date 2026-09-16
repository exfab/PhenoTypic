# Lazy startup — before/after measurements (Task 7)

**Script:** `docs/superpowers/plans/2026-09-11-lazy-startup/measure_startup.py`, best of 5 fresh
interpreters per path after one warm-up. Numbers are *reported, never asserted* — timing on a
shared machine is noisy, and the guard tests pin the module set instead.

## Which machine, and why these numbers replace the committed baseline

**The pair below was taken on the same machine, back-to-back, on an otherwise idle node.**

| | Before | After |
|---|---|---|
| Commit | `d105397d` (`.worktrees/lazy-baseline`) | `0d0bc221` (branch head at measurement) |
| Platform | `Linux-4.18.0-553.144.1.el8_10.x86_64` | same |
| Python | 3.12.10 | same |
| Repeats | best of 5 | best of 5 |

The `before` leg was **re-measured rather than taken from `startup-before.json`**. That committed
file records `"platform": "macOS-26.6.2-arm64-arm-64bit"`, `"cwd": "/private/tmp/pht-lazy-base"` —
it was measured on a laptop, while this branch is developed and run on the UCR HPCC. The spec
requires the two legs be taken *on the same machine*, and the difference is not cosmetic: cold
GPFS makes `import phenotypic` **4.790 s here against 1.572 s on the laptop**, roughly 3x. Pairing
the macOS `before` with an HPCC `after` would have reported a saving that neither machine ever
produced.

`baseline.md`'s macOS table is kept as history. The HPCC `before` is committed beside this file as
`startup-before-hpcc.json`; `startup-after.json` is the `after`.

The baseline worktree is detached at `d105397d`, the last commit before any source change on this
branch. Its `src/` is byte-identical to the spec's `BASE_PRE` (`f0f9a544`) — verified with
`git diff --stat f0f9a544 HEAD -- src/`, which is empty — and `measure_startup.py` is byte-identical
in both worktrees, so the two legs ran the same procedure over the intended two trees.

## Results

| Path | Before (best of 5) | After (best of 5) | Change |
|---|---|---|---|
| bare interpreter | 0.031 s | 0.030 s | -0.001 s |
| `import phenotypic` | 4.790 s | **0.047 s** | **-4.743 s** |
| `from phenotypic import Image` | 4.863 s | 2.549 s | -2.314 s |
| `python -m phenotypic --help` | 5.013 s | **0.437 s** | **-4.576 s** |
| `phenotypic-gui --help` | 5.867 s | **0.460 s** | **-5.407 s** |
| hub to servable (no request) | 6.261 s | 3.971 s | -2.290 s |
| hub + first `/builder/` request | 6.495 s | 5.082 s | -1.413 s |

`import phenotypic` also drops from ~3300 modules to **71**. That figure was re-measured at Task 7
rather than carried forward from Phase 1, because Task 5's Step 5b.1 changed `import importlib` to
`import importlib.util` in `_startup_perf`, which every `import phenotypic` pays. Measured:
`importlib.util` adds **0** modules — the submodule is already in `sys.modules` at interpreter
start — and the count is still exactly 71. (This is what withdrew the Phase 2 review's M2.)

## How to read it

The three paths the spec set out to fix are the three that collapse: bare `import phenotypic` is
now **~100x faster**, and both `--help` paths are **~11-13x faster** and no longer load any of
`HEAVY_STARTUP_MODULES`. `--help` no longer pays for numba, cv2, h5py, mahotas, bm3d, colour,
plotly or Dash to print a usage string.

**`from phenotypic import Image` stays at 2.549 s, and that is the design, not a shortfall.**
Deferring scipy, skimage and pandas is an explicit spec non-goal — 49/94/81 module-level importers,
and the core `Image` handler chain needs them. What it no longer loads is `colour`, `h5py`,
`matplotlib.pyplot` and `plotly`, which is what tier 4 asserts.

The two hub numbers improve by less because the run console stays eager by design: constructing it
starts the SLURM observer, so deferring it would change behaviour rather than move an import. The
first `/builder/` request is slower than "hub to servable" by design too — that is the deferred
builder build arriving where it was asked to arrive, on first use rather than at boot.

**One caveat on the hub rows.** They were measured before the `get_registry()` fix (`0e832d22`),
which in its worst case performs one duplicate `discover()` when a first `/builder/` and a first
`/analysis/` request race. That path is not exercised by this single-threaded measurement, so these
numbers are unaffected; a concurrent first-request workload could see one extra discover once per
process.
