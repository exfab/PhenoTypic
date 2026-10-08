# SubtractPolySurface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Also REQUIRED before any subagent is dispatched: the `orchestrate-subagent` skill** (user's global CLAUDE.md).

**Goal:** Ship `phenotypic.enhance.SubtractPolySurface`, a Gwyddion-faithful polynomial background flattener for `detect_mat` with `method = offset | plane | polynomial | line` and an opt-in robust (sigma-clipped) fit.

**Architecture:** A private float64 kernel module (`_poly_surface_kernels.py`) implements the numeric contract on bare arrays; a thin pydantic operation (`_subtract_poly_surface.py`) owns fields, validation, dtype and `norm`. Correctness is pinned three ways: behavioural tests from the spec contract, a golden fixture produced by a separate clean-room **oracle** task that runs Gwyddion 2.71 itself, and a mutation matrix.

**Tech Stack:** Python 3.12, numpy (`polynomial.legendre`, `linalg.lstsq`), scipy (`stats.median_abs_deviation`), pydantic v2, pytest; gcc + system glib 2.56 for the oracle harness only.

**Spec:** `docs/superpowers/specs/2026-10-08-subtract-poly-surface/design.md` — with
`drift-register.md` and `references.md` beside it. **Read the spec before any task;** this plan
argues from it and cites it as `§n`.

## Clean-room rule (read first — it changes how this plan is written)

The spec (§8, decision Q8) makes this a **clean-room port** of GPL-2.0+ Gwyddion into an
Apache-2.0 codebase, following
`docs/superpowers/specs/2026-07-13-fungi-detection-method-ports/refs/nfa/ATTRIBUTION.md`.

- **Implementer side — Tasks 0, 2–6, 8, 9.** Must **not** open anything under
  `/bigdata/exfab/anguy344/reference-sources/2026-10-08-subtract-poly-surface/gwyddion-2.71/` or
  `.../gwyddion-user-guide/`. Works from `design.md` §4 (the source-free contract),
  `drift-register.md`, the logic-validation scripts, the tests below, and the oracle's numeric
  fixture. The astropy files in that tree (BSD-3-Clause) are permitted.
- **Oracle side — Task 1 only**, run by a **separate agent** that may read the Gwyddion tree.
- **Task 7** (fixture test) is implementer side: it consumes only Task 1's numbers.

Because the author of this plan has read the Gwyddion source, **this plan gives complete test
code and exact interfaces, but deliberately no implementation bodies** for the kernel and the
operation. Each implementation step names the function, its signature, the spec section that
defines its behaviour, and the tests that pin it; the implementer writes the body. This
overrides the writing-plans norm of code in every step, for the provenance reason above.

## Global Constraints

- `uv` is the only runner: `uv run …`; never bare `python`/`pip` (root `CLAUDE.md`).
- Focused test runs: `QT_QPA_PLATFORM=offscreen uv run pytest <paths> -q --no-header -p no:randomly -o addopts= -m "not slow"` (the `run-phenotypic-test` skill: never `-n auto`, never `-x` for a baseline).
- Any run expected to exceed a couple of minutes is a Slurm job (`slurm-job` skill); never `preempt`.
- `uv run ruff check --fix <explicit paths>` only — never bare.
- Operations are pydantic models: annotated class-level fields, keyword-only construction, no `__init__`, closed sets as `Literal` aliases in `phenotypic.sdk_.typing_` (`adding-an-operation` skill).
- Every numeric field on an `enhance/` op is covered by a `TuneSpec` or `Field` bound, and every `TuneSpec` window lies inside its `Field` bounds (tune gates).
- Lazy imports: import `scipy.stats` inside the function that uses it (root `CLAUDE.md` *Gotchas*; guard `tests/unit/ci/test_startup_imports.py`).
- `detect_mat` is **float32**; kernel math is float64; the op casts back (spec §4.9, §5).
- **Citations in shipped code (`src/`, `tests/`) are literature only, and only literature that exists** (user, 2026-10-08): Nečas & Klapetek (2012), doi:10.2478/s11534-011-0096-2 (Gwyddion); Rousseeuw & Croux (1993), doi:10.1080/01621459.1993.10476408 (MAD scale). Never cite Gwyddion or astropy source files or line numbers in code, docstrings, comments or test docstrings — the clean-room record of those lives only in the spec's `references.md` / `drift-register.md`. Referencing spec sections (`§4.3`) and drift rows (`D7`) is fine.
- Defaults, verbatim from spec §3: `method="plane"`, `order=3` (2–11), `independent=True`, `line_order=1` (0–5), `line_axis="row"`, `fit="lstsq"`, `clip_sigma=3.0` (>0), `max_iter=10` (≥1), `norm="clip"`; `MAX_FIT_POINTS = 262_144`.
- Commits end with the two attribution lines:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01E7T821DynJYz1hkGLKtWCt`.
- Work happens in the worktree `/bigdata/exfab/anguy344/PhenoTypic/.claude/worktrees/subtract-poly-surface` (branch `worktree-subtract-poly-surface`). Do not `cd` into any *other* worktree from an isolated session.

## Review Focus

Inputs the spec implies but no contract test exercises, most likely to bite a user first. Each
has a test added to its owning task.

1. **A perfectly flat `detect_mat`** (blank agar crop, synthetic constant). Every method under both
   fits must return finite output at its §4.6 level; the robust loop must stop on a zero MAD, not
   divide by it. → Task 5, `TestReviewFocus::test_constant_image_*`.
2. **A majority-saturated `detect_mat`** (≥ 50% pixels exactly 1.0 from overexposure) under
   `fit="robust"`: MAD can be 0 over the kept set. Must return finite output, no exception. →
   Task 5, `TestReviewFocus::test_majority_saturated_image_is_finite`.
3. **Dark colonies on bright agar** (a plate the user forgot to invert) under `fit="robust"`:
   symmetric clipping (drift D15) must still recover the background. → Task 5,
   `TestReviewFocus::test_dark_colonies_are_clipped_too`.
4. **A detected `GridImage`** (`load_synth_yeast_plate()`): only `detect_mat` may change —
   `rgb`, `gray` and `objmap` stay byte-identical. → Task 6,
   `TestOperationContract::test_grid_image_only_detect_mat_changes`.
5. **Geometry at the validity boundary**: plane on 2×2 and polynomial order 3 on 4×4 must work;
   3×4 at order 3, and `line_axis="column"` with `H < line_order + 1`, must raise a `ValueError`
   naming the problem. → Task 5 `TestValidation`, Task 6 `test_apply_time_errors_keep_their_cause`.

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `src/phenotypic/enhance/_poly_surface_kernels.py` (create) | float64 kernels: coordinates, term sets, Legendre design, lstsq + rank check, robust loop, surface evaluation, subsampled surface fit, line leveling, `flatten_surface` dispatch + geometry validation | 2–5 |
| `src/phenotypic/enhance/_subtract_poly_surface.py` (create) | the operation: fields, docstring, `_operate` | 6 |
| `src/phenotypic/sdk_/typing_.py` (modify) | `SurfaceMethod`, `SurfaceFit`, `LineAxis` aliases | 6 |
| `src/phenotypic/enhance/__init__.py` (modify) | export | 6 |
| `tests/unit/enhance/_poly_surface_synth.py` (create) | shared synthetic plates (mirrors the validation script) | 2 |
| `tests/unit/enhance/test_poly_surface_kernels.py` (create) | kernel tests | 2–5 |
| `tests/unit/enhance/test_subtract_poly_surface.py` (create) | operation tests | 6 |
| `tests/unit/enhance/test_subtract_poly_surface_gwyddion.py` (create) | golden-fixture test | 7 |
| `tests/fixtures/enhance/subtract_poly_surface_gwyddion.{npz,json}` + `SUBTRACT_POLY_SURFACE_ATTRIBUTION.md` (create) | oracle output + provenance | 1, 7 |
| `docs/superpowers/plans/2026-10-08-subtract-poly-surface/oracle_inputs.py` (create) | deterministic oracle inputs + case list | 1 |
| `docs/superpowers/plans/2026-10-08-subtract-poly-surface/run_mutations.py` (create) | mutation harness (drives shipped code, so it lives beside the plan, not in `logic_validation_scripts/`) | 8 |
| `tests/unit/abc_/test_enhancer_taxonomy.py`, `tests/unit/tune/test_enhance_annotations.py`, `docs/source/explanation/what_enhancement_does.md` (modify) | registration | 6 |
| `docs/superpowers/reports/2026-10-08-subtract-poly-surface/` (create) | mutation matrix, gate results | 8, 9 |

**Dependency DAG:** 0 → {1, 2}; 2 → 3 → 4 → 5 → 6; {1, 5} → 7; {6, 7} → 8 → 9. Task 1 runs in
parallel with 2–6.

---

### Task 0: Worktree environment and baseline

**Files:** none.

- [ ] **Step 1: Sync the worktree environment** (slow on GPFS; run in the background if needed)

Run: `uv sync --group dev --group test-qt --extra gui --extra napari`
Expected: exits 0; `.venv/` exists in the worktree.

- [ ] **Step 2: Record a baseline for the files this change touches**

Run:
```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/enhance/test_detect_mat_invariant.py \
  tests/unit/abc_/test_enhancer_taxonomy.py tests/unit/tune/test_enhance_annotations.py \
  tests/unit/tune/test_annotation_coverage.py tests/unit/tune/test_annotation_subset_invariant.py \
  -q --no-header -p no:randomly -o addopts= -m "not slow"
```
Expected: all pass (or note pre-existing failures verbatim in the task report — they are not this change's).

- [ ] **Step 3: Confirm the logic-validation scripts pass in this environment**

Run each:
```bash
uv run python docs/superpowers/logic_validation_scripts/2026-10-08-subtract-poly-surface/basis_equivalence.py
uv run python docs/superpowers/logic_validation_scripts/2026-10-08-subtract-poly-surface/level_conventions.py
uv run python docs/superpowers/logic_validation_scripts/2026-10-08-subtract-poly-surface/robust_and_subsample.py
```
Expected: each ends `0 failure(s)` and exits 0.

---

### Task 1: Gwyddion oracle fixture (ORACLE SIDE — separate agent, may read the Gwyddion tree)

**Files:**
- Create: `docs/superpowers/plans/2026-10-08-subtract-poly-surface/oracle_inputs.py`
- Create: `tests/fixtures/enhance/subtract_poly_surface_gwyddion.npz`
- Create: `tests/fixtures/enhance/subtract_poly_surface_gwyddion.json`
- Create: `tests/fixtures/enhance/SUBTRACT_POLY_SURFACE_ATTRIBUTION.md`
- Outside the repo (never committed): `/bigdata/exfab/anguy344/reference-sources/2026-10-08-subtract-poly-surface/oracle/` — harness C source, build tree, logs.

**Interfaces:**
- Consumes: Gwyddion 2.71 tarball at `/bigdata/exfab/anguy344/reference-sources/2026-10-08-subtract-poly-surface/gwyddion-2.71/gwyddion-2.71.tar.xz`; `references.md` §4 for which Gwyddion functions realise each method.
- Produces (Task 7 depends on this **exact** schema):
  - `.npz`: for each input name `X`, key `input__X` (float64, shape `(H, W)`); for each case name `C`, key `output__C` (float64, same shape as its input).
  - `.json`:
    ```json
    {"kind": "gwyddion-oracle",            // or "contract-rederivation" (fallback, Step 6)
     "gwyddion_version": "2.71",
     "tarball_sha256": "2df721befccbe4d5ee2ba564b32e69341f8ce1de637e2045838a09a2d46b5dba",
     "harness_sha256": "<sha256 of the harness .c>",
     "build": "<compiler, glib version, flags>",
     "generated_utc": "<ISO-8601>",
     "cases": [{"name": "plane__A", "input": "A", "method": "plane",
                "order": null, "independent": null, "line_order": null, "line_axis": null}, ...]}
    ```

- [ ] **Step 1: Write the deterministic inputs and case list**

```python
"""Deterministic inputs and case list for the SubtractPolySurface Gwyddion oracle.

Writes inputs.npz (float64 arrays A, B, C) and cases.json into the directory given as argv[1].
Imports numpy only; never phenotypic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SHAPES = {"A": (23, 31), "B": (32, 32), "C": (17, 40)}


def make_input(height: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    i, j = np.mgrid[0:height, 0:width].astype(float)
    u, v = 2 * j / (width - 1) - 1, 2 * i / (height - 1) - 1
    smooth = 0.45 + 0.12 * u - 0.08 * v + 0.05 * u * v + 0.04 * u**3 - 0.03 * v**2 * u
    rows = rng.normal(0.0, 0.02, (height, 1)) + rng.normal(0.0, 0.01, (height, 1)) * u
    return smooth + rows + rng.normal(0.0, 0.01, (height, width))


def make_cases() -> list[dict]:
    cases = []
    for x in SHAPES:
        cases.append(dict(name=f"offset__{x}", input=x, method="offset",
                          order=None, independent=None, line_order=None, line_axis=None))
        cases.append(dict(name=f"plane__{x}", input=x, method="plane",
                          order=None, independent=None, line_order=None, line_axis=None))
    for x in ("A", "C"):
        for order in (2, 3, 5):
            for independent in (True, False):
                cases.append(dict(name=f"poly{order}_{'ind' if independent else 'tot'}__{x}", input=x,
                                  method="polynomial", order=order, independent=independent,
                                  line_order=None, line_axis=None))
        for degree in (0, 1, 3):
            for axis in ("row", "column"):
                cases.append(dict(name=f"line{degree}_{axis}__{x}", input=x, method="line",
                                  order=None, independent=None, line_order=degree, line_axis=axis))
    return cases


def write_oracle_inputs(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {name: make_input(h, w, seed=101 + k) for k, (name, (h, w)) in enumerate(SHAPES.items())}
    np.savez(out_dir / "inputs.npz", **arrays)
    (out_dir / "cases.json").write_text(json.dumps(make_cases(), indent=1))


if __name__ == "__main__":
    write_oracle_inputs(Path(sys.argv[1]))
```

Run: `uv run python docs/superpowers/plans/2026-10-08-subtract-poly-surface/oracle_inputs.py /bigdata/exfab/anguy344/reference-sources/2026-10-08-subtract-poly-surface/oracle/io`
Expected: `inputs.npz` and `cases.json` (30 cases) written.

- [ ] **Step 2: Build Gwyddion's `libgwyddion` + `libprocess` headless, as a Slurm job**

Inside `.../oracle/build/`, extract the tarball and build only what the harness links. Gwyddion's
`configure` may insist on GTK; if so, compile `libgwyddion/*.c` and `libprocess/*.c` directly with
`gcc -O2 -fPIC $(pkg-config --cflags glib-2.0 gobject-2.0)` into a static archive (system glib is
2.56 with headers; gcc 8.5). Generate `config.h`/`gwyconfig.h` from `configure` with GTK disabled
if it gets that far, else hand-write the minimal defines it reports missing. Submit with the
`slurm-job` skill (`--partition=short --cpus-per-task=8 --mem=16G --time=1:00:00`, logs on
`/bigdata`, never `/scratch`).
Expected: a linkable archive. **If after one honest attempt the build cannot be made to link,
skip to Step 6 (fallback)** and say so in the report.

- [ ] **Step 3: Write the harness (outside the repo) and run it**

A C program that reads `inputs.npz`'s arrays (export them first to raw little-endian `.f64` files
with a sidecar shape, from Python) and `cases.json`, and for each case calls, on a fresh
`GwyDataField` (`xres = W`, `yres = H`, row-major data copied from the numpy array):

| method | Gwyddion call sequence (see `references.md` §4) |
|---|---|
| `offset` | `gwy_data_field_add(f, -gwy_data_field_get_avg(f))` (Zero Mean Value) |
| `plane` | `gwy_data_field_fit_plane(f, &a, &bx, &by)`; `c = -0.5*(bx*xres + by*yres)`; `gwy_data_field_plane_level(f, c, bx, by)` (the Plane Level module's sequence) |
| `polynomial` | build `term_powers` exactly as `polylevel.c` does for `independent` / total degree with `col_degree = row_degree = max_degree = order`; `gwy_data_field_fit_poly(f, NULL, nterms, term_powers, FALSE, NULL)`; `gwy_data_field_subtract_poly(f, nterms, term_powers, coeffs)` |
| `line`, `row` | degree 0: `gwy_data_field_find_row_shifts_trimmed_mean(f, NULL, GWY_MASK_IGNORE, 0.0, 0)` then `gwy_data_field_subtract_row_shifts`; degree ≥ 1: `gwy_data_field_row_level_poly(f, NULL, GWY_MASK_IGNORE, degree, NULL)` (Align Rows' Polynomial branch, `linematch.c:375-379`) |
| `line`, `column` | `gwy_data_field_flip_xy` into a new field, the `row` sequence, `flip_xy` back (`linematch.c:362-369, 400-404`) |

Write each output as raw `.f64`. Run it inside the same Slurm allocation as Step 2 or a new
short job.
Expected: 30 outputs, all finite.

- [ ] **Step 4: Assemble the fixture in the repo**

Pack inputs as `input__A|B|C` and outputs as `output__<case>` into
`tests/fixtures/enhance/subtract_poly_surface_gwyddion.npz` (`np.savez_compressed`), and write the
`.json` with the schema above (`kind = "gwyddion-oracle"`). Sanity-check with numpy only:
`offset` outputs have mean ≈ 0; `polynomial` outputs have mean ≈ 0; `line` outputs have every
line's mean ≈ the input's mean; `plane` outputs have mean ≈ `mean(z) + ½(bx+by)` computed from
the closed-form per-pixel slopes. These are spec §4.6; a failure here means a harness bug — fix
the harness, never the expectation.

- [ ] **Step 5: Write `tests/fixtures/enhance/SUBTRACT_POLY_SURFACE_ATTRIBUTION.md`**

```markdown
# SubtractPolySurface golden fixture — attribution

`subtract_poly_surface_gwyddion.npz` holds numeric outputs of Gwyddion 2.71
(Nečas & Klapetek, *Cent. Eur. J. Phys.* 10(1):181–188, 2012, doi:10.2478/s11534-011-0096-2),
GPL-2.0-or-later, computed by a harness that links Gwyddion's libgwyddion/libprocess on
deterministic inputs from `docs/superpowers/plans/2026-10-08-subtract-poly-surface/oracle_inputs.py`.

Only numbers are in this repository. No Gwyddion source, and not the harness (which calls
Gwyddion's API), is distributed, imported, linked, copied, or transcribed by PhenoTypic. The
production implementation (`src/phenotypic/enhance/_poly_surface_kernels.py`) was written by an
implementer who received `design.md` (a source-free behavioural contract), `drift-register.md`,
the logic-validation scripts, and this fixture — and did not open the Gwyddion source or user
guide. Provenance (tarball sha256, harness sha256, build) is in the sibling `.json`.
```

- [ ] **Step 6 (fallback only): contract re-derivation**

If Step 2 failed, generate the outputs with an independent numpy implementation of spec §4 using
**monomial** bases and normal equations (not Legendre — so it is not the implementer's method),
set `"kind": "contract-rederivation"`, and change the attribution file's first paragraph to say
the fixture pins the spec's contract, not Gwyddion. Report this prominently: it is the weaker
fixture.

- [ ] **Step 7: Commit (fixture + inputs script + attribution only)**

```bash
git add docs/superpowers/plans/2026-10-08-subtract-poly-surface/oracle_inputs.py \
  tests/fixtures/enhance/subtract_poly_surface_gwyddion.npz \
  tests/fixtures/enhance/subtract_poly_surface_gwyddion.json \
  tests/fixtures/enhance/SUBTRACT_POLY_SURFACE_ATTRIBUTION.md
git commit -m "test(fixtures): Gwyddion 2.71 oracle outputs for SubtractPolySurface"
```

---

### Task 2: Kernel foundation — coordinates, term sets, Legendre design, solve, evaluate

**Files:**
- Create: `src/phenotypic/enhance/_poly_surface_kernels.py`
- Create: `tests/unit/enhance/_poly_surface_synth.py`
- Create: `tests/unit/enhance/test_poly_surface_kernels.py`

**Interfaces — Produces:**
```python
MAX_FIT_POINTS: Final[int] = 262_144

def normalized_axis(n: int) -> np.ndarray: ...
    # float64 array 2k/(n-1) - 1, k = 0..n-1 (spec §4.1). ValueError("... at least 2 ...") if n < 2.
def term_powers(degree: int, independent: bool) -> tuple[tuple[int, int], ...]: ...
    # (p, q) = (power of u, power of v). p-major, then q ascending.
    # independent: all p, q in 0..degree; total: p + q <= degree (spec §3, §4.2).
def legendre_design(u: np.ndarray, v: np.ndarray, degree: int,
                    terms: Sequence[tuple[int, int]]) -> np.ndarray: ...
    # (n_points, n_terms); column k = L_p(u) * L_q(v) for terms[k] = (p, q). u, v are 1-D.
def solve_least_squares(a: np.ndarray, z: np.ndarray) -> np.ndarray: ...
    # numpy.linalg.lstsq; ValueError mentioning "rank" if rank < a.shape[1] (drift D2).
def evaluate_surface(coef: np.ndarray, terms: Sequence[tuple[int, int]], degree: int,
                     height: int, width: int) -> np.ndarray: ...
    # (height, width) via numpy.polynomial.legendre.leggrid2d(v_axis, u_axis, C), C[q, p] = coef[k] (spec §4.2).
```

- [ ] **Step 1: Write the shared synthetic-plate helper**

`tests/unit/enhance/_poly_surface_synth.py`:
```python
"""Synthetic plates for the SubtractPolySurface tests.

Mirrors docs/superpowers/logic_validation_scripts/2026-10-08-subtract-poly-surface/
robust_and_subsample.py, whose measurements set every threshold the tests assert.
"""

from __future__ import annotations

import numpy as np

NOISE = 0.01


def grid(height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return float64 (i, j, u, v) grids; u, v are normalized to [-1, 1] (design.md §4.1)."""
    i, j = np.mgrid[0:height, 0:width].astype(float)
    return i, j, 2 * j / (width - 1) - 1, 2 * i / (height - 1) - 1


def colony_domes(height: int, width: int, cover: float, rng: np.random.Generator,
                 radius: float = 16.0) -> np.ndarray:
    """Soft-edged colony domes (amplitude 0.15-0.45) until ``cover`` of the pixels are colony."""
    i, j, _, _ = grid(height, width)
    colonies = np.zeros((height, width))
    while (colonies > 0).mean() < cover:
        cy, cx = rng.uniform(0, height), rng.uniform(0, width)
        rr = np.hypot(i - cy, j - cx) / radius
        colonies = np.maximum(colonies, rng.uniform(0.15, 0.45) * np.sqrt(np.clip(1 - rr**2, 0, None)))
    return colonies


def surface_plate(height: int = 300, width: int = 450, cover: float = 0.25,
                  seed: int = 2025) -> tuple[np.ndarray, np.ndarray]:
    """A plate whose background is exactly a tensor-order-3 polynomial. Returns (z, background)."""
    rng = np.random.default_rng(seed)
    _, _, u, v = grid(height, width)
    background = 0.30 + 0.08 * u - 0.05 * v + 0.06 * u * u + 0.04 * v * v - 0.03 * u * v + 0.02 * u**3 * v
    return background + colony_domes(height, width, cover, rng) + rng.normal(0, NOISE, (height, width)), background


def line_plate(height: int = 300, width: int = 450, cover: float = 0.25,
               seed: int = 2026) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A plate whose background is exactly degree 1 along every row (per-row offset and slope).

    Returns (z, background, colony_mask).
    """
    rng = np.random.default_rng(seed)
    _, _, u, v = grid(height, width)
    background = (0.30 + 0.08 * u - 0.05 * v + 0.04 * v * v
                  + rng.normal(0, 0.03, (height, 1)) + rng.normal(0, 0.02, (height, 1)) * u)
    colonies = colony_domes(height, width, cover, rng)
    return background + colonies + rng.normal(0, NOISE, (height, width)), background, colonies > 0


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
```

- [ ] **Step 2: Write the failing tests**

`tests/unit/enhance/test_poly_surface_kernels.py`:
```python
"""Kernel tests for SubtractPolySurface (design.md §4; drift-register.md)."""

from __future__ import annotations

import numpy as np
import pytest

from phenotypic.enhance._poly_surface_kernels import (
    evaluate_surface,
    legendre_design,
    normalized_axis,
    solve_least_squares,
    term_powers,
)

from ._poly_surface_synth import grid


class TestNormalizedAxis:
    def test_endpoints_and_spacing(self):
        t = normalized_axis(5)
        np.testing.assert_array_equal(t, [-1.0, -0.5, 0.0, 0.5, 1.0])
        assert t.dtype == np.float64

    def test_divides_by_n_minus_one(self):
        """Gwyddion's normalization reaches +1 exactly at the last sample (n-1, not n)."""
        assert normalized_axis(4)[-1] == 1.0
        assert normalized_axis(7)[3] == 0.0

    def test_rejects_fewer_than_two_samples(self):
        with pytest.raises(ValueError, match="at least 2"):
            normalized_axis(1)


class TestTermPowers:
    def test_independent_is_the_tensor_product(self):
        assert term_powers(2, independent=True) == (
            (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2))

    def test_total_degree_keeps_p_plus_q_at_most_degree(self):
        assert term_powers(2, independent=False) == ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0))

    @pytest.mark.parametrize("degree", [0, 1, 2, 3, 5, 11])
    def test_term_counts(self, degree):
        assert len(term_powers(degree, independent=True)) == (degree + 1) ** 2
        assert len(term_powers(degree, independent=False)) == (degree + 1) * (degree + 2) // 2


class TestLegendreDesign:
    def test_column_is_legendre_of_u_times_legendre_of_v(self):
        u, v = np.array([0.3]), np.array([-0.4])
        a = legendre_design(u, v, 2, ((1, 0), (0, 1), (2, 1)))
        p2_of_u = 0.5 * (3 * 0.3**2 - 1)
        np.testing.assert_allclose(a[0], [0.3, -0.4, p2_of_u * -0.4], atol=1e-15)

    @pytest.mark.parametrize("independent", [True, False])
    def test_spans_the_monomial_term_set(self, independent):
        """Drift D1: same span as Gwyddion's monomials, so any monomial surface is fitted exactly."""
        rng = np.random.default_rng(0)
        _, _, u, v = grid(13, 17)
        terms = term_powers(3, independent)
        z = sum(c * u**p * v**q for c, (p, q) in zip(rng.normal(size=len(terms)), terms))
        a = legendre_design(u.ravel(), v.ravel(), 3, terms)
        np.testing.assert_allclose(a @ solve_least_squares(a, z.ravel()), z.ravel(), atol=1e-12)


class TestSolveLeastSquares:
    def test_recovers_exact_coefficients(self):
        rng = np.random.default_rng(1)
        a = rng.normal(size=(40, 4))
        x = np.array([0.5, -1.0, 2.0, 0.25])
        np.testing.assert_allclose(solve_least_squares(a, a @ x), x, atol=1e-12)

    def test_rank_deficient_design_raises(self):
        """Drift D2: Gwyddion silently zeroes the coefficients; we raise."""
        a = np.ones((10, 2))
        with pytest.raises(ValueError, match="rank"):
            solve_least_squares(a, np.arange(10.0))


class TestEvaluateSurface:
    @pytest.mark.parametrize("independent", [True, False])
    def test_matches_dense_design_evaluation(self, independent):
        rng = np.random.default_rng(2)
        height, width, degree = 11, 19, 3
        terms = term_powers(degree, independent)
        coef = rng.normal(size=len(terms))
        _, _, u, v = grid(height, width)
        dense = (legendre_design(u.ravel(), v.ravel(), degree, terms) @ coef).reshape(height, width)
        surface = evaluate_surface(coef, terms, degree, height, width)
        assert surface.shape == (height, width)
        np.testing.assert_allclose(surface, dense, atol=1e-13)
```

- [ ] **Step 3: Run to verify they fail**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/enhance/test_poly_surface_kernels.py -q --no-header -p no:randomly -o addopts= -m "not slow"`
Expected: collection error — `ModuleNotFoundError: No module named 'phenotypic.enhance._poly_surface_kernels'`.

- [ ] **Step 4: Implement the five functions and the constant**

Create `src/phenotypic/enhance/_poly_surface_kernels.py` with a module docstring stating: private,
float64, implements `design.md` §4, clean-room (no Gwyddion source consulted), and drift rows D1/D2.
Implement exactly the five signatures in **Interfaces** above, behaviour per spec §4.1–§4.2:
`legvander` per axis multiplied column-wise; `numpy.linalg.lstsq(..., rcond=None)` with the rank
check; `leggrid2d` with a zero-filled `(degree+1, degree+1)` coefficient matrix indexed `[q, p]`.
Google-style docstrings on each. No scipy import needed yet.

- [ ] **Step 5: Run to verify they pass**

Run: same command as Step 3. Expected: all pass.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/enhance/_poly_surface_kernels.py tests/unit/enhance/_poly_surface_synth.py tests/unit/enhance/test_poly_surface_kernels.py
git add src/phenotypic/enhance/_poly_surface_kernels.py tests/unit/enhance/_poly_surface_synth.py tests/unit/enhance/test_poly_surface_kernels.py
git commit -m "feat(enhance): poly-surface kernel foundation (Legendre design, lstsq, evaluation)"
```

---

### Task 3: Robust loop and subsampled surface fit

**Files:**
- Modify: `src/phenotypic/enhance/_poly_surface_kernels.py`
- Modify: `tests/unit/enhance/test_poly_surface_kernels.py`

**Interfaces:**
- Consumes: Task 2's five functions.
- Produces:
```python
class RobustFit(NamedTuple):
    coef: np.ndarray    # (n_terms,)
    kept: np.ndarray    # bool (n_points,), the final inlier set
    rounds: int         # refits performed after the initial fit (0..max_iter)

def robust_least_squares(a: np.ndarray, z: np.ndarray, *, clip_sigma: float,
                         max_iter: int) -> RobustFit: ...
    # spec §4.3 exactly: initial fit on all points; per round residuals at ALL points, center =
    # median(r[kept]), scale = scipy.stats.median_abs_deviation(r[kept], scale="normal");
    # stop if scale == 0; new = kept & (|r - center| <= clip_sigma*scale) (clipped never return);
    # stop keeping the previous fit if new.sum() < n_terms (D3); stop if new.sum() == kept.sum();
    # else kept = new, refit, rounds += 1.

def fit_surface_coefficients(z: np.ndarray, *, degree: int, independent: bool, fit: str,
                             clip_sigma: float, max_iter: int,
                             max_fit_points: int = MAX_FIT_POINTS) -> np.ndarray: ...
    # spec §4.8: strided subsample when z.size > max_fit_points, per-axis stride
    # s_a = min(s, max(1, (n_a - 1) // max(degree, 1))), s = ceil(sqrt(z.size / max_fit_points));
    # subsampled points keep FULL-grid coordinates; fit = "lstsq" -> solve_least_squares,
    # "robust" -> robust_least_squares(...).coef. Returns coefficients in term_powers(degree, independent) order.
```

- [ ] **Step 1: Append the failing tests**

Add to `tests/unit/enhance/test_poly_surface_kernels.py` (extend the import list with
`MAX_FIT_POINTS`, `RobustFit`, `fit_surface_coefficients`, `robust_least_squares`, and
`from ._poly_surface_synth import NOISE, rmse, surface_plate`):
```python
class TestRobustLeastSquares:
    def test_exact_data_returns_exact_coefficients(self):
        _, _, u, v = grid(20, 30)
        a = legendre_design(u.ravel(), v.ravel(), 1, term_powers(1, independent=False))
        x = np.array([0.4, 0.02, -0.03])
        fit = robust_least_squares(a, a @ x, clip_sigma=3.0, max_iter=10)
        assert isinstance(fit, RobustFit)
        np.testing.assert_allclose(fit.coef, x, atol=1e-12)

    def test_zero_scale_stops_without_dividing(self):
        """A constant residual set has MAD 0; the loop must stop, not divide (Review Focus 1)."""
        fit = robust_least_squares(np.ones((50, 1)), np.full(50, 0.37), clip_sigma=3.0, max_iter=10)
        assert fit.rounds == 0
        np.testing.assert_allclose(fit.coef, [0.37])
        assert fit.kept.all()

    def test_recovers_background_under_colonies(self):
        """R1/R2 at 25% cover: robust within 0.5 sigma, plain lstsq beyond 1 sigma, >= 10x apart."""
        z, background = surface_plate(cover=0.25, seed=2025)
        _, _, u, v = grid(*z.shape)
        terms = term_powers(3, independent=True)
        a = legendre_design(u.ravel(), v.ravel(), 3, terms)
        robust = (a @ robust_least_squares(a, z.ravel(), clip_sigma=3.0, max_iter=10).coef).reshape(z.shape)
        plain = (a @ solve_least_squares(a, z.ravel())).reshape(z.shape)
        assert rmse(robust, background) < 0.5 * NOISE
        assert rmse(plain, background) > 1.0 * NOISE
        assert rmse(plain, background) > 10 * rmse(robust, background)

    def test_clipped_points_never_return(self):
        """The mask is carried forward between rounds, so the kept set only shrinks (spec §4.3)."""
        z, _ = surface_plate(cover=0.40, seed=2030)
        _, _, u, v = grid(*z.shape)
        a = legendre_design(u.ravel(), v.ravel(), 3, term_powers(3, independent=True))
        previous = None
        for rounds in range(1, 6):
            kept = robust_least_squares(a, z.ravel(), clip_sigma=3.0, max_iter=rounds).kept
            if previous is not None:
                assert not (kept & ~previous).any(), f"a clipped point returned at max_iter={rounds}"
            previous = kept

    def test_max_iter_caps_the_rounds_and_convergence_stops_early(self):
        z, _ = surface_plate(cover=0.10, seed=2031)
        _, _, u, v = grid(*z.shape)
        a = legendre_design(u.ravel(), v.ravel(), 3, term_powers(3, independent=True))
        assert robust_least_squares(a, z.ravel(), clip_sigma=3.0, max_iter=1).rounds == 1
        converged = robust_least_squares(a, z.ravel(), clip_sigma=3.0, max_iter=50)
        assert 1 <= converged.rounds < 50

    def test_never_fits_from_fewer_points_than_terms(self):
        """Drift D3: a round that would leave < n_terms inliers is rejected, so no rank error."""
        rng = np.random.default_rng(3)
        for _ in range(300):
            n = int(rng.integers(4, 9))
            a = np.stack([np.ones(n), np.linspace(-1, 1, n), np.linspace(-1, 1, n) ** 2], axis=1)
            z = rng.standard_cauchy(n)
            fit = robust_least_squares(a, z, clip_sigma=1.0, max_iter=10)
            assert fit.kept.sum() >= 3
            assert np.isfinite(fit.coef).all()


class TestFitSurfaceCoefficients:
    def test_below_the_cap_every_pixel_is_used(self):
        z, _ = surface_plate(height=60, width=90, cover=0.0, seed=2032)
        _, _, u, v = grid(*z.shape)
        terms = term_powers(3, independent=True)
        direct = solve_least_squares(legendre_design(u.ravel(), v.ravel(), 3, terms), z.ravel())
        coef = fit_surface_coefficients(z, degree=3, independent=True, fit="lstsq",
                                        clip_sigma=3.0, max_iter=10)
        np.testing.assert_allclose(coef, direct, atol=1e-13)

    def test_subsampled_fit_is_within_the_standard_error_bound(self):
        """R4: RMS distance from the full fit <= 3*sigma*sqrt(p/n_sub)."""
        z, _ = surface_plate(height=200, width=300, cover=0.0, seed=2033)
        terms = term_powers(3, independent=True)
        full = fit_surface_coefficients(z, degree=3, independent=True, fit="lstsq",
                                        clip_sigma=3.0, max_iter=10, max_fit_points=10**9)
        sub = fit_surface_coefficients(z, degree=3, independent=True, fit="lstsq",
                                       clip_sigma=3.0, max_iter=10, max_fit_points=6000)
        n_sub = len(range(0, 200, 4)) * len(range(0, 300, 4))  # stride ceil(sqrt(60000/6000)) = 4
        bound = 3 * NOISE * np.sqrt(len(terms) / n_sub)
        diff = rmse(evaluate_surface(sub, terms, 3, 200, 300), evaluate_surface(full, terms, 3, 200, 300))
        assert 0 < diff <= bound

    def test_per_axis_stride_keeps_enough_rows_on_a_thin_image(self):
        """Spec §4.8: a 4 x 5000 strip at degree 3 must keep >= 4 distinct rows (no rank error)."""
        rng = np.random.default_rng(4)
        _, _, u, v = grid(4, 5000)
        z = 0.3 + 0.1 * u * v**3 + rng.normal(0, NOISE, (4, 5000))
        coef = fit_surface_coefficients(z, degree=3, independent=True, fit="lstsq",
                                        clip_sigma=3.0, max_iter=10, max_fit_points=100)
        assert np.isfinite(coef).all()

    def test_default_cap_is_the_spec_constant(self):
        assert MAX_FIT_POINTS == 262_144
```

- [ ] **Step 2: Run to verify they fail**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/enhance/test_poly_surface_kernels.py -q --no-header -p no:randomly -o addopts= -m "not slow"`
Expected: ImportError for `RobustFit` / `robust_least_squares` / `fit_surface_coefficients`.

- [ ] **Step 3: Implement `RobustFit`, `robust_least_squares`, `fit_surface_coefficients`**

Exactly the Interfaces block above and spec §4.3, §4.8. `scipy.stats` is imported **inside**
`robust_least_squares`. The docstring describes the loop and may reference drift rows D3,
D12–D15; it cites no source file (Global Constraints: literature only). The MAD's normal
consistency scale may be cited as Rousseeuw & Croux (1993), doi:10.1080/01621459.1993.10476408.

- [ ] **Step 4: Run to verify they pass**

Run: same as Step 2. Expected: all pass.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/enhance/_poly_surface_kernels.py tests/unit/enhance/test_poly_surface_kernels.py
git add src/phenotypic/enhance/_poly_surface_kernels.py tests/unit/enhance/test_poly_surface_kernels.py
git commit -m "feat(enhance): robust sigma-clipped fit and subsampled surface fit for poly-surface kernels"
```

---

### Task 4: Line leveling

**Files:**
- Modify: `src/phenotypic/enhance/_poly_surface_kernels.py`
- Modify: `tests/unit/enhance/test_poly_surface_kernels.py`

**Interfaces:**
- Consumes: Task 2–3 functions.
- Produces:
```python
def level_lines(z: np.ndarray, *, line_order: int, fit: str, clip_sigma: float,
                max_iter: int) -> np.ndarray: ...
    # Levels ROWS of z (the caller transposes for columns). spec §4.5 steps 2-4:
    # avg = z.mean() computed BEFORE leveling; each row fitted with legvander(normalized_axis(W), line_order);
    # "lstsq": one lstsq call with rows as right-hand sides; "robust": §4.3 independently per row,
    # processed in blocks of _LINE_BLOCK = 256 rows; returns z - fit + avg (float64, new array).
```

- [ ] **Step 1: Append the failing tests**

Add `level_lines` to the import and `from ._poly_surface_synth import line_plate`:
```python
class TestLevelLines:
    def test_removes_per_row_offsets_and_slopes_exactly(self):
        rng = np.random.default_rng(5)
        _, _, u, _ = grid(40, 60)
        z = 0.4 + rng.normal(0, 0.05, (40, 1)) + rng.normal(0, 0.03, (40, 1)) * u
        out = level_lines(z, line_order=1, fit="lstsq", clip_sigma=3.0, max_iter=10)
        np.testing.assert_allclose(out, np.full_like(z, z.mean()), atol=1e-12)

    @pytest.mark.parametrize("line_order", [0, 1, 3])
    def test_every_row_is_leveled_to_the_input_mean(self, line_order):
        """Spec §4.5 / L4: avg is the input's global mean, taken BEFORE leveling."""
        z, _, _ = line_plate(height=50, width=80, cover=0.2, seed=2034)
        out = level_lines(z, line_order=line_order, fit="lstsq", clip_sigma=3.0, max_iter=10)
        np.testing.assert_allclose(out.mean(axis=1), np.full(50, z.mean()), atol=1e-12)

    def test_degree_zero_equals_the_row_shift_form(self):
        """L5: z - (rowmean - mean(rowmeans)) on an unmasked field."""
        z, _, _ = line_plate(height=30, width=45, cover=0.1, seed=2035)
        rowmeans = z.mean(axis=1, keepdims=True)
        out = level_lines(z, line_order=0, fit="lstsq", clip_sigma=3.0, max_iter=10)
        np.testing.assert_allclose(out, z - (rowmeans - rowmeans.mean()), atol=1e-12)

    def test_robust_recovers_rows_that_are_mostly_background(self):
        """R2 per line: rows < 20% own colony fraction within 0.5 sigma; lstsq worse than 1 sigma."""
        z, background, colonies = line_plate(cover=0.25, seed=2026)
        avg = z.mean()
        robust_fit = z - level_lines(z, line_order=1, fit="robust", clip_sigma=3.0, max_iter=10) + avg
        plain_fit = z - level_lines(z, line_order=1, fit="lstsq", clip_sigma=3.0, max_iter=10) + avg
        sparse = colonies.mean(axis=1) < 0.2
        err = lambda f: np.sqrt(np.mean((f[sparse] - background[sparse]) ** 2, axis=1))
        assert sparse.sum() > 50
        assert err(robust_fit).max() < 0.5 * NOISE
        assert err(plain_fit).max() > 1.0 * NOISE

    def test_rows_are_independent_across_block_boundaries(self):
        """Robust per-row state must not leak between rows or blocks (515 rows > 2 blocks of 256)."""
        z, _, _ = line_plate(height=515, width=64, cover=0.2, seed=2036)
        whole = level_lines(z, line_order=1, fit="robust", clip_sigma=3.0, max_iter=10) - z.mean()
        for r in (0, 255, 256, 511, 514):
            row = z[r:r + 1]
            alone = level_lines(row, line_order=1, fit="robust", clip_sigma=3.0, max_iter=10) - row.mean()
            np.testing.assert_allclose(whole[r], alone[0], atol=1e-12)

    def test_short_heavy_tailed_rows_never_fail(self):
        """Drift D3 per line: a round leaving < line_order+1 inliers is rejected for that row."""
        rng = np.random.default_rng(6)
        z = rng.standard_cauchy((200, 6))
        out = level_lines(z, line_order=3, fit="robust", clip_sigma=1.0, max_iter=10)
        assert np.isfinite(out).all()

    def test_a_row_that_is_mostly_colony_is_not_shifted_by_the_mean(self):
        """Drift D3: Gwyddion's degenerate-row path adds +avg to the raw row; ours keeps a fit."""
        rng = np.random.default_rng(7)
        z = 0.3 + rng.normal(0, NOISE, (20, 100))
        z[7, 5:95] += 0.4                                    # row 7 is 90% colony
        out = level_lines(z, line_order=1, fit="robust", clip_sigma=3.0, max_iter=10)
        assert np.isfinite(out[7]).all()
        assert not np.allclose(out[7], z[7] + z.mean())
```

- [ ] **Step 2: Run to verify they fail**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/enhance/test_poly_surface_kernels.py -q --no-header -p no:randomly -o addopts= -m "not slow"`
Expected: ImportError for `level_lines`.

- [ ] **Step 3: Implement `level_lines`** (and private `_LINE_BLOCK = 256`) per the Interfaces block and spec §4.5, §4.3 (per line), §4.8 (blocks; batched weighted normal equations via `numpy.linalg.solve` are allowed for `line_order ≤ 5`).

- [ ] **Step 4: Run to verify they pass**

Run: same as Step 2. Expected: all pass.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/enhance/_poly_surface_kernels.py tests/unit/enhance/test_poly_surface_kernels.py
git add src/phenotypic/enhance/_poly_surface_kernels.py tests/unit/enhance/test_poly_surface_kernels.py
git commit -m "feat(enhance): per-line polynomial leveling (lstsq + per-line robust) for poly-surface kernels"
```

---

### Task 5: `flatten_surface` — method dispatch, level conventions, geometry validation

**Files:**
- Modify: `src/phenotypic/enhance/_poly_surface_kernels.py`
- Modify: `tests/unit/enhance/test_poly_surface_kernels.py`

**Interfaces:**
- Consumes: Tasks 2–4.
- Produces (Task 6 and Task 7 call this):
```python
def flatten_surface(z: np.ndarray, *, method: str, order: int, independent: bool,
                    line_order: int, line_axis: str, fit: str, clip_sigma: float,
                    max_iter: int, max_fit_points: int = MAX_FIT_POINTS) -> np.ndarray: ...
    # float64 in -> new float64 array out; never mutates z. Spec §4.4-§4.7:
    # offset: lstsq -> z - z.mean(); robust -> z - robust constant (degree-0 surface, subsampled per §4.8).
    # plane: coef = fit_surface_coefficients(z, degree=1, independent=False, ...);
    #        term order ((0,0),(0,1),(1,0)) gives b_v = coef[1], b_u = coef[2];
    #        bx = 2*b_u/(W-1), by = 2*b_v/(H-1); out = z - bx*(j - W/2) - by*(i - H/2)  (D7: do NOT centre).
    # polynomial: z - evaluate_surface(fit_surface_coefficients(z, degree=order, independent=...)).
    # line: transpose if line_axis == "column", level_lines, transpose back.
    # ValueError (spec §4.7) for: H < 2 or W < 2; polynomial with H or W < order + 1;
    # line with line length < line_order + 1; unknown method / fit / line_axis strings.
```

- [ ] **Step 1: Append the failing tests**

Add `flatten_surface` to the import:
```python
DEFAULTS = dict(order=3, independent=True, line_order=1, line_axis="row",
                fit="lstsq", clip_sigma=3.0, max_iter=10)


def flatten(z, **overrides):
    return flatten_surface(z, **{**DEFAULTS, **overrides})


class TestOffset:
    def test_lstsq_subtracts_the_mean(self):
        z, _ = surface_plate(height=40, width=50, cover=0.1, seed=2040)
        out = flatten(z, method="offset")
        np.testing.assert_allclose(out, z - z.mean(), atol=1e-15)

    def test_robust_finds_the_agar_level_under_colonies(self):
        rng = np.random.default_rng(2028)
        from ._poly_surface_synth import colony_domes
        z = 0.4 + colony_domes(300, 450, 0.25, rng) + rng.normal(0, NOISE, (300, 450))
        level = z - flatten(z, method="offset", fit="robust")
        assert abs(float(level.flat[0]) - 0.4) < 0.5 * NOISE
        assert abs(z.mean() - 0.4) > 1.0 * NOISE


class TestPlane:
    def test_removes_tilt_about_gwyddions_pivot(self):
        """Drift D7 / L2: out = a + bx*W/2 + by*H/2 everywhere; NOT the true-centre value."""
        height, width, a, bx, by = 21, 34, 0.35, 0.002, 0.001
        i, j, _, _ = grid(height, width)
        z = a + bx * j + by * i
        out = flatten(z, method="plane")
        np.testing.assert_allclose(out, np.full_like(z, a + bx * width / 2 + by * height / 2), atol=1e-12)
        assert abs(out.mean() - (z.mean() + 0.5 * (bx + by))) < 1e-12

    def test_robust_plane_flattens_the_background_under_colonies(self):
        rng = np.random.default_rng(2041)
        from ._poly_surface_synth import colony_domes
        i, j, _, _ = grid(300, 450)
        background = 0.3 + 0.0004 * j - 0.0003 * i
        colonies = colony_domes(300, 450, 0.25, rng)
        z = background + colonies + rng.normal(0, NOISE, (300, 450))
        out = flatten(z, method="plane", fit="robust")
        flattened_background = (out - colonies)[colonies == 0]
        assert np.std(flattened_background) < 1.5 * NOISE


class TestPolynomial:
    def test_removes_its_own_form_exactly_and_lands_at_zero(self):
        _, _, u, v = grid(30, 41)
        z = 0.3 + 0.05 * u - 0.02 * v + 0.04 * u**2 * v + 0.01 * u**3 * v**3
        out = flatten(z, method="polynomial", order=3, independent=True)
        np.testing.assert_allclose(out, 0.0, atol=1e-10)

    def test_independent_keeps_the_u3v3_term_and_total_degree_does_not(self):
        _, _, u, v = grid(30, 41)
        z = 0.3 + 0.05 * u**3 * v**3
        assert np.max(np.abs(flatten(z, method="polynomial", order=3, independent=True))) < 1e-10
        assert rmse(flatten(z, method="polynomial", order=3, independent=False), 0.0) > 1e-3

    def test_robust_polynomial_recovers_the_background(self):
        z, background = surface_plate(cover=0.25, seed=2025)
        out = flatten(z, method="polynomial", order=3, independent=True, fit="robust")
        assert rmse(z - out, background) < 0.5 * NOISE


class TestLineDispatch:
    def test_column_axis_is_row_axis_on_the_transpose(self):
        z, _, _ = line_plate(height=40, width=55, cover=0.1, seed=2042)
        col = flatten(z, method="line", line_axis="column", line_order=2)
        row_of_t = flatten(z.T.copy(), method="line", line_axis="row", line_order=2)
        np.testing.assert_allclose(col, row_of_t.T, atol=1e-12)

    def test_line_levels_to_the_input_mean(self):
        z, _, _ = line_plate(height=40, width=55, cover=0.1, seed=2043)
        out = flatten(z, method="line", line_order=1)
        np.testing.assert_allclose(out.mean(axis=1), np.full(40, z.mean()), atol=1e-12)


class TestContract:
    @pytest.mark.parametrize("method", ["offset", "plane", "polynomial", "line"])
    def test_returns_new_float64_and_never_mutates_input(self, method):
        z, _ = surface_plate(height=30, width=40, cover=0.1, seed=2044)
        before = z.copy()
        out = flatten(z, method=method)
        assert out.dtype == np.float64 and out is not z
        np.testing.assert_array_equal(z, before)


class TestValidation:
    @pytest.mark.parametrize("shape", [(1, 10), (10, 1)])
    def test_single_row_or_column_is_rejected(self, shape):
        with pytest.raises(ValueError, match="at least 2"):
            flatten(np.zeros(shape), method="plane")

    def test_polynomial_needs_order_plus_one_samples_per_axis(self):
        flatten(np.random.default_rng(8).random((4, 4)), method="polynomial", order=3)
        with pytest.raises(ValueError, match="order"):
            flatten(np.random.default_rng(8).random((3, 4)), method="polynomial", order=3)

    def test_plane_works_on_two_by_two(self):
        assert np.isfinite(flatten(np.array([[0.1, 0.2], [0.3, 0.5]]), method="plane")).all()

    def test_column_lines_need_line_order_plus_one_rows(self):
        with pytest.raises(ValueError, match="line_order"):
            flatten(np.random.default_rng(9).random((2, 50)), method="line",
                    line_axis="column", line_order=2)

    @pytest.mark.parametrize("bad", [dict(method="huber"), dict(fit="ransac"), dict(line_axis="diag")])
    def test_unknown_strings_are_rejected(self, bad):
        with pytest.raises(ValueError):
            flatten(np.ones((5, 5)), **{"method": "plane", **bad})


class TestReviewFocus:
    @pytest.mark.parametrize("fit", ["lstsq", "robust"])
    @pytest.mark.parametrize("method, level", [("offset", 0.0), ("plane", 0.42),
                                               ("polynomial", 0.0), ("line", 0.42)])
    def test_constant_image_lands_at_its_documented_level(self, method, level, fit):
        """Review Focus 1: flat detect_mat; robust must stop on zero MAD (spec §4.6 levels)."""
        out = flatten(np.full((25, 35), 0.42), method=method, fit=fit)
        np.testing.assert_allclose(out, level, atol=1e-12)

    @pytest.mark.parametrize("method", ["offset", "plane", "polynomial", "line"])
    def test_majority_saturated_image_is_finite(self, method):
        """Review Focus 2: >= 50% pixels exactly 1.0 -- MAD over the kept set can be 0."""
        rng = np.random.default_rng(10)
        z = 0.3 + rng.normal(0, NOISE, (60, 80))
        z[:, :48] = 1.0
        assert np.isfinite(flatten(z, method=method, fit="robust")).all()

    def test_dark_colonies_are_clipped_too(self):
        """Review Focus 3 / drift D15: an un-inverted plate (dark colonies on bright agar)."""
        rng = np.random.default_rng(2027)
        from ._poly_surface_synth import colony_domes
        _, _, u, v = grid(300, 450)
        background = 0.70 + 0.05 * u - 0.04 * v + 0.03 * u * v
        z = background - colony_domes(300, 450, 0.25, rng) + rng.normal(0, NOISE, (300, 450))
        out = flatten(z, method="polynomial", order=2, independent=True, fit="robust")
        assert rmse(z - out, background) < 0.5 * NOISE
```

- [ ] **Step 2: Run to verify they fail**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/enhance/test_poly_surface_kernels.py -q --no-header -p no:randomly -o addopts= -m "not slow"`
Expected: ImportError for `flatten_surface`.

- [ ] **Step 3: Implement `flatten_surface`** per the Interfaces block and spec §4.4–§4.7. Geometry validation happens before any fitting, with messages that name the offending quantity (`"at least 2"`, `"order"`, `"line_order"`) so the tests' `match=` hold.

- [ ] **Step 4: Run to verify they pass**

Run: same as Step 2. Expected: all pass.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/enhance/_poly_surface_kernels.py tests/unit/enhance/test_poly_surface_kernels.py
git add src/phenotypic/enhance/_poly_surface_kernels.py tests/unit/enhance/test_poly_surface_kernels.py
git commit -m "feat(enhance): flatten_surface dispatch with Gwyddion per-method level conventions"
```

---

### Task 6: The operation, its types, and registration

**Files:**
- Modify: `src/phenotypic/sdk_/typing_.py` (add three aliases beside `BoundaryMode`, ~line 32)
- Create: `src/phenotypic/enhance/_subtract_poly_surface.py`
- Modify: `src/phenotypic/enhance/__init__.py` (import after `from ._subtract_rolling_ball import SubtractRollingBall`; `"SubtractPolySurface"` after `"SubtractRollingBall"` in `__all__`)
- Modify: `tests/unit/abc_/test_enhancer_taxonomy.py` (`BackgroundSubtraction` roster, ~line 55)
- Modify: `tests/unit/tune/test_enhance_annotations.py`
- Modify: `docs/source/explanation/what_enhancement_does.md` (~line 47)
- Create: `tests/unit/enhance/test_subtract_poly_surface.py`

**Interfaces:**
- Consumes: `flatten_surface` (Task 5).
- Produces:
```python
# phenotypic.sdk_.typing_
SurfaceMethod: TypeAlias = Literal["offset", "plane", "polynomial", "line"]
SurfaceFit: TypeAlias = Literal["lstsq", "robust"]
LineAxis: TypeAlias = Literal["row", "column"]

# phenotypic.enhance
class SubtractPolySurface(NormalizedOutputMixin, BackgroundSubtraction):
    method: SurfaceMethod = "plane"
    order: Annotated[int, TuneSpec(2, 5)] = Field(3, ge=2, le=11)
    independent: bool = True
    line_order: Annotated[int, TuneSpec(0, 3)] = Field(1, ge=0, le=5)
    line_axis: LineAxis = "row"
    fit: SurfaceFit = "lstsq"
    clip_sigma: Annotated[float, TuneSpec(2.0, 4.0)] = Field(3.0, gt=0.0)
    max_iter: Annotated[int, TuneSpec(tunable=False)] = Field(10, ge=1)
    def _operate(self, image: Image) -> Image: ...
```

- [ ] **Step 1: Write the failing operation tests**

`tests/unit/enhance/test_subtract_poly_surface.py`:
```python
"""Operation tests for SubtractPolySurface (design.md §3, §4.6, §4.7, §4.9, §7.2)."""

from __future__ import annotations

import json
from typing import get_args

import numpy as np
import pytest
from pydantic import ValidationError

from phenotypic import Image, ImagePipeline
from phenotypic.data import load_synth_yeast_plate
from phenotypic.enhance import SubtractPolySurface
from phenotypic.sdk_.typing_ import LineAxis, SurfaceFit, SurfaceMethod

from ._poly_surface_synth import NOISE, grid, rmse, surface_plate

FLOAT32_TOL = 1e-6  # ~8 * eps32 on [0, 1] data (design.md §5)


def _image(z: np.ndarray) -> Image:
    image = Image(arr=np.clip(z, 0.0, 1.0))
    image.detect_mat[:] = z
    return image


class TestFields:
    def test_field_order_ends_with_norm(self):
        assert list(SubtractPolySurface.model_fields) == [
            "method", "order", "independent", "line_order", "line_axis",
            "fit", "clip_sigma", "max_iter", "norm"]

    def test_defaults_match_the_spec(self):
        op = SubtractPolySurface()
        assert (op.method, op.order, op.independent, op.line_order, op.line_axis,
                op.fit, op.clip_sigma, op.max_iter, op.norm) == (
            "plane", 3, True, 1, "row", "lstsq", 3.0, 10, "clip")

    def test_literal_aliases_are_the_closed_sets(self):
        assert set(get_args(SurfaceMethod)) == {"offset", "plane", "polynomial", "line"}
        assert set(get_args(SurfaceFit)) == {"lstsq", "robust"}
        assert set(get_args(LineAxis)) == {"row", "column"}

    @pytest.mark.parametrize("bad", [
        dict(order=1), dict(order=12), dict(line_order=-1), dict(line_order=6),
        dict(clip_sigma=0.0), dict(max_iter=0), dict(method="median"),
        dict(fit="huber"), dict(line_axis="diagonal"), dict(clip=True)])
    def test_invalid_values_raise_validation_error(self, bad):
        with pytest.raises((ValidationError, ValueError)):
            SubtractPolySurface(**bad)

    def test_unused_fields_are_ignored_not_rejected(self):
        z, _ = surface_plate(height=40, width=60, cover=0.1, seed=2050)
        a = SubtractPolySurface(method="plane", norm=None).apply(_image(z)).detect_mat[:]
        b = SubtractPolySurface(method="plane", order=7, line_order=4, norm=None).apply(_image(z)).detect_mat[:]
        np.testing.assert_array_equal(a, b)


class TestOperationContract:
    def test_plane_through_the_operation(self):
        height, width, a, bx, by = 21, 34, 0.35, 0.002, 0.001
        i, j, _, _ = grid(height, width)
        out = SubtractPolySurface(method="plane", norm=None).apply(_image(a + bx * j + by * i)).detect_mat[:]
        np.testing.assert_allclose(out, a + bx * width / 2 + by * height / 2, atol=FLOAT32_TOL)

    def test_dtype_rgb_and_gray_are_preserved(self):
        z, _ = surface_plate(height=40, width=60, cover=0.1, seed=2051)
        image = _image(z)
        rgb, gray = image.rgb[:].copy(), image.gray[:].copy()
        out = SubtractPolySurface(method="polynomial").apply(image)
        assert out.detect_mat[:].dtype == np.float32
        np.testing.assert_array_equal(out.rgb[:], rgb)
        np.testing.assert_array_equal(out.gray[:], gray)

    def test_norm_policies(self):
        z, _ = surface_plate(height=60, width=80, cover=0.2, seed=2052)
        clip = SubtractPolySurface(method="polynomial", norm="clip").apply(_image(z)).detect_mat[:]
        none = SubtractPolySurface(method="polynomial", norm=None).apply(_image(z)).detect_mat[:]
        resc = SubtractPolySurface(method="polynomial", norm="rescale").apply(_image(z)).detect_mat[:]
        assert clip.min() >= 0.0 and clip.max() <= 1.0
        assert none.min() < 0.0                       # level 0 => negative half of the noise survives
        assert resc.min() == pytest.approx(0.0, abs=FLOAT32_TOL)
        assert resc.max() == pytest.approx(1.0, abs=FLOAT32_TOL)

    def test_robust_beats_lstsq_on_a_plate(self):
        z, background = surface_plate(cover=0.25, seed=2025)
        robust = SubtractPolySurface(method="polynomial", fit="robust", norm=None).apply(_image(z)).detect_mat[:]
        plain = SubtractPolySurface(method="polynomial", norm=None).apply(_image(z)).detect_mat[:]
        assert rmse(z - robust, background) < 0.5 * NOISE
        assert rmse(z - plain, background) > 10 * rmse(z - robust, background)

    def test_grid_image_only_detect_mat_changes(self):
        """Review Focus 4: rgb, gray and objmap of a detected GridImage are untouched."""
        image = load_synth_yeast_plate()
        rgb, gray, objmap = image.rgb[:].copy(), image.gray[:].copy(), image.objmap[:].copy()
        before = image.detect_mat[:].copy()
        out = SubtractPolySurface(method="polynomial", fit="robust").apply(image)
        assert not np.array_equal(out.detect_mat[:], before)
        assert 0.0 <= out.detect_mat[:].min() and out.detect_mat[:].max() <= 1.0
        np.testing.assert_array_equal(out.rgb[:], rgb)
        np.testing.assert_array_equal(out.gray[:], gray)
        np.testing.assert_array_equal(out.objmap[:], objmap)

    def test_apply_time_errors_keep_their_cause(self):
        """Spec §4.7: ImageOperation wraps twice; the root cause is our ValueError."""
        image = _image(np.random.default_rng(11).random((3, 40)))
        with pytest.raises(Exception, match="order") as excinfo:
            SubtractPolySurface(method="polynomial", order=3).apply(image)
        root = excinfo.value
        while root.__cause__ is not None:
            root = root.__cause__
        assert isinstance(root, ValueError)


class TestSerialization:
    def test_pipeline_json_round_trip(self):
        op = SubtractPolySurface(method="line", line_order=2, line_axis="column",
                                 fit="robust", clip_sigma=2.5, max_iter=7, norm="rescale")
        loaded = ImagePipeline.from_json(ImagePipeline(ops=[op]).to_json())
        (restored,) = list(loaded._ops.values())
        assert isinstance(restored, SubtractPolySurface)
        assert restored.model_dump() == op.model_dump()

    def test_schema_is_json_serializable(self):
        json.dumps(SubtractPolySurface.model_json_schema())
```

- [ ] **Step 2: Run to verify they fail**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/enhance/test_subtract_poly_surface.py -q --no-header -p no:randomly -o addopts= -m "not slow"`
Expected: `ImportError: cannot import name 'SubtractPolySurface'`.

- [ ] **Step 3: Add the three aliases to `src/phenotypic/sdk_/typing_.py`**

```python
SurfaceMethod: TypeAlias = Literal["offset", "plane", "polynomial", "line"]
"""``SubtractPolySurface.method``: offset (order 0), plane (order 1), polynomial surface, per-line."""

SurfaceFit: TypeAlias = Literal["lstsq", "robust"]
"""``SubtractPolySurface.fit``: plain least squares (Gwyddion) or sigma-clipped least squares."""

LineAxis: TypeAlias = Literal["row", "column"]
"""``SubtractPolySurface.line_axis``: which scan lines ``method="line"`` levels."""
```
(Place them beside `BoundaryMode`; match that block's docstring style if it differs.)

- [ ] **Step 4: Implement the operation**

`src/phenotypic/enhance/_subtract_poly_surface.py`: the class exactly as in **Interfaces**, each
search window marked `# TODO: review bound (unverified vs literature)` (spec §3). `_operate`
(spec §4.9): `flatten_surface(image.detect_mat[:].astype(np.float64), **fields)` → cast to the
original dtype → `self._apply_norm(...)` → assign `image.detect_mat[:]`; return `image`. The
docstring follows spec §5's checklist in full (methods, the §4.6 level table, that defaults
follow Gwyddion with the literature citation Nečas & Klapetek (2012), doi:10.2478/s11534-011-0096-2
— no source-file citations — the lstsq-bias warning with its measured 2.3–9.5 σ, the robust limits including the
**per-line** arrayed-plate limit, `line` ≠ Align Rows' Median, `Consider Also`), Google style per
`abc_/CLAUDE.md`, with this runnable example:
```python
>>> from phenotypic.data import load_synth_yeast_plate
>>> from phenotypic.enhance import SubtractPolySurface
>>> plate = load_synth_yeast_plate()
>>> flat = SubtractPolySurface(method="polynomial", fit="robust").apply(plate)
>>> bool(0.0 <= flat.detect_mat[:].min() and flat.detect_mat[:].max() <= 1.0)
True
```

- [ ] **Step 5: Register**

1. `src/phenotypic/enhance/__init__.py`: `from ._subtract_poly_surface import SubtractPolySurface` after the `SubtractRollingBall` import; `"SubtractPolySurface",` after `"SubtractRollingBall",` in `__all__`.
2. `tests/unit/abc_/test_enhancer_taxonomy.py`: add `"SubtractPolySurface",` to the `BackgroundSubtraction` tuple after `"SubtractRollingBall",`.
3. `tests/unit/tune/test_enhance_annotations.py`: import `SubtractPolySurface`; add to the A.1 Tier-1 parametrize list
   ```python
   (SubtractPolySurface(), "order", IntRange, (2, 5)),
   (SubtractPolySurface(), "line_order", IntRange, (0, 3)),
   (SubtractPolySurface(), "clip_sigma", FloatRange, (2.0, 4.0, False)),
   ```
   to `test_tune_spec_off_excludes`'s list `(SubtractPolySurface(), "max_iter")`; to the pure-metadata factories `lambda: SubtractPolySurface(order=11, line_order=5, clip_sigma=9.0)`; and to the A.2 rejection factories `lambda: SubtractPolySurface(order=1)`, `lambda: SubtractPolySurface(line_order=-1)`, `lambda: SubtractPolySurface(clip_sigma=0.0)`, `lambda: SubtractPolySurface(max_iter=0)`. (Match each list's exact tuple shape by reading its neighbours first.)
4. `docs/source/explanation/what_enhancement_does.md`, after the `SubtractRollingBall` bullet:
   ```markdown
   - **SubtractPolySurface** — fits and subtracts a constant, plane, polynomial
     surface, or per-scan-line polynomial (Gwyddion's leveling tools), with an
     optional robust fit that ignores colonies
   ```

- [ ] **Step 6: Run the operation tests and the doctest**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/enhance/test_subtract_poly_surface.py -q --no-header -p no:randomly -o addopts= -m "not slow"`
Run: `QT_QPA_PLATFORM=offscreen uv run pytest --doctest-modules src/phenotypic/enhance/_subtract_poly_surface.py -q --no-header -p no:randomly -o addopts=`
Expected: all pass.

- [ ] **Step 7: Run the registration surface**

Run:
```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/enhance/test_detect_mat_invariant.py \
  tests/unit/abc_/test_enhancer_taxonomy.py tests/unit/tune/test_enhance_annotations.py \
  tests/unit/tune/test_annotation_coverage.py tests/unit/tune/test_annotation_subset_invariant.py \
  tests/unit/ci/test_startup_imports.py tests/unit/ci/test_deferred_imports.py \
  -q --no-header -p no:randomly -o addopts= -m "not slow"
```
Expected: all pass, with the Task 0 baseline's pre-existing failures (if any) unchanged.
`test_detect_mat_invariant` now applies `SubtractPolySurface()` automatically — it must pass.

- [ ] **Step 8: Lint, type-check the new files, commit**

```bash
uv run ruff check --fix src/phenotypic/enhance/_subtract_poly_surface.py src/phenotypic/enhance/_poly_surface_kernels.py src/phenotypic/enhance/__init__.py src/phenotypic/sdk_/typing_.py tests/unit/enhance/test_subtract_poly_surface.py tests/unit/abc_/test_enhancer_taxonomy.py tests/unit/tune/test_enhance_annotations.py
uv run mypy src/phenotypic/enhance/_subtract_poly_surface.py src/phenotypic/enhance/_poly_surface_kernels.py
git add src/phenotypic/sdk_/typing_.py src/phenotypic/enhance/_subtract_poly_surface.py src/phenotypic/enhance/__init__.py \
  tests/unit/enhance/test_subtract_poly_surface.py tests/unit/abc_/test_enhancer_taxonomy.py \
  tests/unit/tune/test_enhance_annotations.py docs/source/explanation/what_enhancement_does.md
git commit -m "feat(enhance): SubtractPolySurface operation (Gwyddion-faithful leveling, opt-in robust fit)"
```

---

### Task 7: Golden-fixture test against the Gwyddion oracle

**Files:**
- Create: `tests/unit/enhance/test_subtract_poly_surface_gwyddion.py`

**Interfaces:**
- Consumes: Task 1's `.npz` / `.json` schema; Task 5's `flatten_surface`.

- [ ] **Step 1: Write the test**

```python
"""Golden fixture: the float64 kernel reproduces Gwyddion 2.71 (design.md §7.3).

The fixture pins TRANSCRIPTION; behavioural correctness is pinned by the kernel and operation
tests. Tolerances are derived from drift D1, not guessed: Gwyddion solves monomial normal
equations by Cholesky, so its own forward error scales with cond(Gram_monomial) * eps.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from phenotypic.enhance._poly_surface_kernels import flatten_surface, term_powers

FIXTURE = Path(__file__).parents[2] / "fixtures" / "enhance" / "subtract_poly_surface_gwyddion"
MANIFEST = json.loads(FIXTURE.with_suffix(".json").read_text())
ARRAYS = np.load(FIXTURE.with_suffix(".npz"))
EPS = np.finfo(np.float64).eps


def _monomial_gram_condition(case: dict, height: int, width: int) -> float:
    if case["method"] == "polynomial":
        v, u = np.meshgrid(2 * np.arange(height) / (height - 1.0) - 1, 2 * np.arange(width) / (width - 1.0) - 1,
                           indexing="ij")
        a = np.stack([u.ravel() ** p * v.ravel() ** q
                      for p, q in term_powers(case["order"], case["independent"])], axis=1)
    elif case["method"] == "line":
        n = width if case["line_axis"] == "row" else height
        x = np.arange(n) - (n - 1) / 2.0                     # Gwyddion's centred pixel coordinate
        a = np.vander(x, case["line_order"] + 1, increasing=True)
    else:
        return 1.0
    return float(np.linalg.cond(a.T @ a))


def _tolerance(case: dict, z: np.ndarray) -> float:
    scale = max(1.0, float(np.max(np.abs(z))))
    return max(1e-12, 10 * EPS * scale * _monomial_gram_condition(case, *z.shape))


@pytest.mark.parametrize("case", MANIFEST["cases"], ids=[c["name"] for c in MANIFEST["cases"]])
def test_kernel_reproduces_the_oracle(case):
    z = ARRAYS[f"input__{case['input']}"]
    expected = ARRAYS[f"output__{case['name']}"]
    out = flatten_surface(
        z, method=case["method"], order=case["order"] or 3,
        independent=True if case["independent"] is None else case["independent"],
        line_order=0 if case["line_order"] is None else case["line_order"],
        line_axis=case["line_axis"] or "row", fit="lstsq", clip_sigma=3.0, max_iter=10)
    np.testing.assert_allclose(out, expected, rtol=0, atol=_tolerance(case, z))


def test_fixture_provenance_is_recorded():
    assert MANIFEST["kind"] in {"gwyddion-oracle", "contract-rederivation"}
    assert MANIFEST["gwyddion_version"] == "2.71"
    assert len(MANIFEST["cases"]) == 30
```

- [ ] **Step 2: Run it**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/enhance/test_subtract_poly_surface_gwyddion.py -q --no-header -p no:randomly -o addopts= -m "not slow"`
Expected: 31 passed. **If any case fails, do not loosen the tolerance and do not open the Gwyddion
source.** Report the case name, max abs error and tolerance to the orchestrator; the oracle side
(Task 1's agent) diagnoses whether the contract (spec §4), the harness, or the kernel is wrong.
A contract fix is a spec change and goes to the user.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/enhance/test_subtract_poly_surface_gwyddion.py
git commit -m "test(enhance): SubtractPolySurface kernel reproduces the Gwyddion 2.71 oracle"
```

---

### Task 8: Load-bearing proof and mutation matrix

**Files:**
- Create: `docs/superpowers/plans/2026-10-08-subtract-poly-surface/run_mutations.py`
- Create: `docs/superpowers/reports/2026-10-08-subtract-poly-surface/mutation-matrix.md`

**Interfaces:**
- Consumes: the shipped kernel/op and the three test files.

- [ ] **Step 1: Write the harness** (restores from saved **bytes**, never `git checkout --` — a checkout once wiped a subagent's uncommitted fixes)

```python
"""Mutation harness for SubtractPolySurface. Drives shipped code, so it lives beside the plan.

Each mutant is ONE exact text replacement that must match exactly once. The original bytes are
saved before mutating and written back in `finally`, whatever happens.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
TESTS = ["tests/unit/enhance/test_poly_surface_kernels.py",
         "tests/unit/enhance/test_subtract_poly_surface.py",
         "tests/unit/enhance/test_subtract_poly_surface_gwyddion.py"]


def run_mutant(path: str, old: str, new: str) -> dict:
    target = ROOT / path
    original = target.read_bytes()
    text = original.decode()
    if text.count(old) != 1:
        return {"status": "INVALID", "detail": f"pattern matched {text.count(old)} times"}
    try:
        target.write_bytes(text.replace(old, new).encode())
        proc = subprocess.run(
            ["uv", "run", "pytest", *TESTS, "-q", "--no-header", "-p", "no:randomly",
             "-o", "addopts=", "-m", "not slow"],
            cwd=ROOT, capture_output=True, text=True,
            env={**os.environ, "QT_QPA_PLATFORM": "offscreen"})
        failed = [line for line in proc.stdout.splitlines() if line.startswith("FAILED")]
        return {"status": "KILLED" if proc.returncode != 0 else "SURVIVED", "killed_by": failed[:10]}
    finally:
        target.write_bytes(original)


def run_all_mutants(mutants_json: Path) -> None:
    mutants = json.loads(mutants_json.read_text())
    results = {m["name"]: run_mutant(m["path"], m["old"], m["new"]) for m in mutants}
    print(json.dumps(results, indent=1))
    assert all(r["status"] == "KILLED" for r in results.values()), "a mutant survived or was invalid"


if __name__ == "__main__":
    run_all_mutants(Path(sys.argv[1]))
```

- [ ] **Step 2: Write `mutants.json` against the real code** (beside the harness). One entry per spec §7.4 mutant, each `{"name", "path", "old", "new"}` with `old` copied verbatim from the shipped source so it matches exactly once:

| name | single change |
|---|---|
| `pivot_true_centre` | plane pivot `W/2, H/2` → `(W-1)/2, (H-1)/2` — **the load-bearing proof** |
| `plane_subtracts_constant` | plane also subtracts the fitted constant |
| `line_no_avg` | drop the `+ avg` in line leveling |
| `line_avg_after` | compute `avg` from the leveled array instead of the input |
| `swap_term_sets` | `independent` True/False term sets swapped |
| `normalize_by_n` | `n - 1` → `n` in `normalized_axis` |
| `skip_column_transpose` | `line_axis="column"` levels rows |
| `one_sided_clip` | `abs(r - center) <= k*s` → `(r - center) <= k*s` |
| `clipped_points_return` | `new = kept & (...)` → `new = (...)` |
| `no_convergence_stop` | remove the `count(new) == count(kept)` stop |
| `no_rank_check` | remove the rank-deficiency raise |
| `no_min_inliers_guard` | remove the `< n_terms` guard (drift D3) |

- [ ] **Step 3: Run the harness as a Slurm job** (12 mutants × ~1–2 min of tests; `slurm-job` skill, `--partition=short --cpus-per-task=4 --mem=16G --time=1:00:00`, log on `/bigdata`). The job must run in a **detached worktree at the commit under test** (root `CLAUDE.md`: a run measures one tree), created with `git worktree add --detach <path> <sha>` and removed by an `afterany` finalizer; submit with `sbatch --chdir=<path>` rather than `cd`-ing this session into it.
Expected: every mutant `KILLED`. `pivot_true_centre` must be killed by
`test_subtract_poly_surface_gwyddion.py` (proving the fixture load-bearing) **and** by
`TestPlane::test_removes_tilt_about_gwyddions_pivot`.

- [ ] **Step 4: For any `SURVIVED` mutant**, add the smallest test that kills it to the owning test file, re-run that mutant, and record the before/after in the report. An *equivalent* mutant (behaviourally identical on every reachable input) is recorded with evidence, as `2026-07-08-alt-phase-detection/drift-register.md` M1 does, not "fixed".

- [ ] **Step 5: Write `mutation-matrix.md`** — the mutant × killing-test table from the harness output, the load-bearing proof, any added tests, any equivalent mutants. Commit:

```bash
git add docs/superpowers/plans/2026-10-08-subtract-poly-surface/run_mutations.py \
  docs/superpowers/plans/2026-10-08-subtract-poly-surface/mutants.json \
  docs/superpowers/reports/2026-10-08-subtract-poly-surface/mutation-matrix.md tests/unit/enhance/
git commit -m "test(enhance): SubtractPolySurface mutation matrix; fixture proven load-bearing"
```

---

### Task 9: Final verification

**Files:**
- Create: `docs/superpowers/reports/2026-10-08-subtract-poly-surface/final-gate.md`

- [ ] **Step 1: Lint and type-check everything this change touched**

```bash
uv run ruff check src/phenotypic/enhance/_subtract_poly_surface.py src/phenotypic/enhance/_poly_surface_kernels.py src/phenotypic/enhance/__init__.py src/phenotypic/sdk_/typing_.py tests/unit/enhance/_poly_surface_synth.py tests/unit/enhance/test_poly_surface_kernels.py tests/unit/enhance/test_subtract_poly_surface.py tests/unit/enhance/test_subtract_poly_surface_gwyddion.py
uv run mypy src/phenotypic
```
Expected: clean (or mypy findings identical to `main`'s, recorded).

- [ ] **Step 2: Re-run the logic-validation scripts** (Task 0 Step 3 commands). Expected: 3 × `0 failure(s)`.

- [ ] **Step 3: Affected-surface run, once** — derived from importers: everything importing `phenotypic.enhance`, `phenotypic.sdk_.typing_`, or enumerating enhancers. At minimum `tests/unit/enhance tests/unit/abc_ tests/unit/tune tests/unit/ci tests/gui/builder`. Submit as one Slurm job (`slurm-job` skill; `--partition=short --cpus-per-task=16 --mem=32G --time=2:00:00`, `-n 16`, never `-n auto`), in a detached worktree at the final SHA as in Task 8 Step 3.

- [ ] **Step 4: Full sharded regression, once** — copy `docs/superpowers/plans/2026-08-18-ome-zarr-image-store/run_unit_suite.sbatch`, point its `WORKTREE=` at a detached worktree at the final SHA, submit per the `slurm-job` skill and the regression-baseline recipe (24-way array), with an `afterany` finalizer that removes the worktree. Compare against the latest `main` baseline; run every failure in isolation before attributing it.

- [ ] **Step 5: Write `final-gate.md`** — commands, job IDs, pass/fail counts copied from the logs (state no number you did not just read), and any failure with its isolated re-run. Commit it.

```bash
git add docs/superpowers/reports/2026-10-08-subtract-poly-surface/final-gate.md
git commit -m "docs(reports): SubtractPolySurface final gate"
```
