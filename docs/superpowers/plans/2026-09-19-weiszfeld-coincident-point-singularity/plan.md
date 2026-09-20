# Weiszfeld Coincident-Point Singularity — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `weiszfeld_median` return the geometric median when a data point lies on the running estimate, instead of returning that data point and reporting success.

**Architecture:** Replace the `np.maximum(distances, 1e-10)` distance floor with the Vardi & Zhang (2000) modified Weiszfeld step: split the points that sit *on* the estimate out of the reweighting, take a damped step toward the reweighted centroid of the rest, and treat the damping factor reaching 1 as an optimality certificate. With no coincident point the step reduces to today's arithmetic **bit-identically, provided no distance falls in the band `(radius, 1e-10]`** — see *Property 1*, which is where the old floor and the new rule genuinely differ.

**Tech Stack:** Python 3.11+, numpy, pytest, `uv` as the sole package manager and runner.

**Spec:** `docs/superpowers/specs/2026-09-19-weiszfeld-coincident-point-singularity/README.md` (added by `f18c59b6`). The plan argues from the spec; executors read both.

**Executable witness:** `docs/superpowers/logic_validation_scripts/2026-09-19-weiszfeld-coincident-point-singularity/weiszfeld_singularity.py` — 7 checks today, 9 after Task 4. Run with `uv run python <path>`; exits non-zero on failure; does not import `phenotypic`.

**Plan review:** `docs/superpowers/reports/2026-09-19-weiszfeld-coincident-point-singularity/plan-review.md` — 16 findings (2 BLOCKER, 5 MAJOR, 7 MINOR, 2 NIT). **All applied in this revision.** Read it for the reasoning behind anything below that looks over-specified; it also records what was *validated* (the Vardi–Zhang transcription, the `γ == 1` argument, and a hand-trace of every existing test through the new update), which is worth not re-deriving.

---

## Global Constraints

- **`uv` is the sole package manager and runner.** Never bare `python` or `pip`. Run commands as `uv run <cmd>`.
- **`geometric_median` is a public export** (`src/phenotypic/util/__init__.py:5,27`), as is `robust_color_center` (`:13,30`). `weiszfeld_median` itself is private. The public signature, defaults and return contract do **not** change (spec Non-goals).
- **The `method='cohen'` branch stays unimplemented and keeps raising.** The Cohen et al. routines in the same module (including their own `1e-10` clamps at `_geometric_median.py:69` and `:791`) are unreachable dead code and are **out of scope** — do not touch them (spec Non-goals).
- **`ruff` is already red on `_geometric_median.py`, pre-existing.** `F841` at `:680` (`n` assigned but unused in `line_search`) is in that same dead Cohen path, and `--fix` has no safe fix for it. **The bar for every lint step below is "no finding other than `F841` at `:680`", never "clean".** Do not fix it — that would be an out-of-scope edit to the Cohen path.
- **`GEOMEDIAN_TOL = 1e-6` / `GEOMEDIAN_MAX_ITER = 200`** (`_color_checker_profile.py:57-58`) are correct and pinned by `test_profile_geomedian_constants_are_tight_enough`. Do not re-tune them (spec Non-goals).
- **Initialization stays `np.mean`** (`_geometric_median.py:1129`). See *Decision 2* — this is not a free choice.
- **Lint with explicit paths only:** `uv run ruff check --fix <paths you changed>`. Bare `ruff check --fix` rewrites the whole tree.
- **Do not author `bio_desc` on schema members** and **do not edit anything under `docs/superpowers/**/refs`.** Neither applies to any file this plan touches; stated because they bind the whole repo.

---

## Decisions taken before this plan was written

Both were measured, not reasoned.

### Decision 1 — the coincidence test is a scale-relative radius, not `d == 0`

The spec's *Compare the distance floor* section says "Test coincidence explicitly (`d == 0`)". **That is not sufficient, and Task 4 amends the spec.**

Measured on the spec's own skewed swatch **with a mean-initialised solver — the initialization that ships** — planting the extra pixel at a small offset from the centroid rather than exactly on it:

| planted offset | smallest distance at init | error with `d == 0` | error with `d <= 1e-12·max(‖x‖,1)` |
|---|---|---|---|
| `0` | `0.00e+00` | 0.000 | 0.000 |
| `1e-16` | `1.11e-16` | **21.896** | 0.000 |
| `1e-15` | `9.99e-16` | 0.000 | 0.000 |
| `1e-13` | `1.00e-13` | 0.000 | 0.000 |
| `1e-11` | `1.00e-11` | 0.000 | 0.000 |

Errors are 8-bit code values against the converged median. **21.896 is the same failure magnitude as the original defect** (the spec quotes 21.9): a point at `1.11e-16` is not `== 0`, so an exact test lets it take weight `9e15` and capture the solve exactly as the `1e-10` floor did. The hole is not monotone in the offset, which is what makes it a lurking hazard rather than a clean boundary.

**The "mean-initialised" qualifier is load-bearing, not decoration.** Under a *median*-initialised solver the same experiment gives `0.0000` for both tests, because no iterate ever comes within `1e-12` of the planted pixel. Carrying this number into a median-initialised subject produces a check that cannot fail — that is exactly what plan-review PR-1 caught in the first draft of Task 4.

### Decision 2 — initialization stays `np.mean`, and this is load-bearing

The witness script's reference starts at `np.median`; the shipped solver starts at `np.mean`. Matching them looks tidy and is wrong. Measured on the clean swatch, varying **only** the start:

| init | iterations | error vs converged median | `test_patch_center_converges_past_a_loose_tolerance` (asserts `< 1e-2`) |
|---|---|---|---|
| `np.mean` (shipped) | 12 | `9.4e-05` code values | **PASSES** |
| `np.median` | 1 | `5.4e-01` code values | **FAILS** |

The stopping rule is `‖step‖ < eps`. A *better* start makes it fire **earlier** — the coordinate-wise median is already close enough that step 1 is under `1e-6`, so the loop exits after one iteration half a code value from the answer. Switching the start would break an existing, currently-green test that has nothing to do with this bug.

---

## The defect, in the code as it stands

`src/phenotypic/util/_geometric_median.py:1135-1141`:

```python
        # Compute weights: w_i = 1/||x - a^(i)||_2
        distances = np.linalg.norm(points - x, axis=1)
        distances = np.maximum(distances, 1e-10)   # <-- line 1137
        weights = 1.0 / distances

        # Weighted update
        x = np.sum(points * weights[:, np.newaxis], axis=0) / np.sum(weights)
```

A point on `x` gets weight `1e10`, holding 99.99953% of the total on the spec's 3,901-pixel swatch. The update is that point to within rounding; the next iteration sees `change == 0` and returns `converged: True` after **one** iteration, having answered with the **mean** — the exact non-robust statistic the geometric median was chosen to avoid.

## The replacement — Vardi & Zhang (2000)

Let `x` be the estimate, `η` the number of points counted as coincident with it, `R` the rest, and `W = Σ_{a∈R} 1/‖a−x‖`.

```
T    = Σ_{a∈R} a/‖a−x‖  /  W
r    = ‖ Σ_{a∈R} (a−x)/‖a−x‖ ‖
γ    = min(1, η / r)
x⁺   = (1 − γ)·T + γ·x
```

Three properties the tasks below depend on. **Properties 1 and 3 are stated more carefully than in the first draft** — both were wrong as originally written (plan-review PR-6, PR-7).

1. **`η == 0` reduces to the current update, bit-identically — outside one band.** With `η == 0` the masked arrays are contiguous copies with identical values in identical order, so the products and reductions are bit-for-bit the old ones. **But the old code floored distances at `1e-10` and the new code applies no floor**, so the two agree bit-for-bit only when no distance falls in
   `( 1e-12·max(‖x‖,1) , 1e-10 ]`.
   A point in that band is *not* coincident under the new rule (`η` stays 0, classical branch) but *was* floored under the old one. In sRGB (`‖x‖ < 1`) the band is `(1e-12, 1e-10]`, two orders of magnitude wide. Measured on the clean swatch: the nearest point to any iterate is `1.24e-03`, so the band is never entered there and bit-identity holds — verified over 60 iterations.
2. **`γ == 1` is an optimality certificate, not a stall.** `f(x) = Σ‖x−a‖` has subdifferential `Σ_{a∈R}(x−a)/‖x−a‖ + η·B` at a point of multiplicity `η`. `0 ∈ ∂f(x)` iff `r ≤ η`, i.e. iff `γ == 1`. So `γ == 1` gives `x⁺ = x`, `change == 0`, and `converged: True` — correctly, with no change needed to the `if change < eps:` branch.
3. **The stopping rule certifies `r ≤ η + eps·W`, not `r ≤ η`.** The classical step satisfies `‖T − x‖ = r/W`, so the damped step is `‖x⁺ − x‖ = (1−γ)·r/W = (r − η)/W` whenever `γ = η/r < 1`. Damping subtracts `η/W` from the step length regardless of how near `γ` is to 1, so whenever `eps < r/W < eps + η/W` the undamped rule keeps iterating and the damped rule stops. **Measured on the planted swatch: `W = 534,440`, so `η/W = 1.87e-06` against `eps = 1e-06` — the window is open, not theoretical.** The honest guarantee is therefore `r ≤ η + eps·W`: exact optimality when `γ = 1`, and an `eps`-scaled subgradient slack otherwise. That is the same slack the classical rule has carried at `η = 0` since 1937; this change neither introduces nor removes it.

   > The first draft claimed "a small step under damping is a near-optimal step, so damping cannot cause an early stop." That argument was wrong, and acceptance criterion 2 was reworded because of it (Task 4 Step 5).

---

## File structure

| File | Change | Responsibility |
|---|---|---|
| `src/phenotypic/util/_geometric_median.py` | Modify `:1097-1176` | The solver. Task 1 extracts the return-pair builder; Task 2 replaces the update. |
| `tests/unit/util/test_geometric_median.py` | **Create** | Solver-level guards: coincidence, near-coincidence, degeneracy, the subgradient certificate, bit-identity of the clean path. |
| `tests/unit/correction/test_color_checker_geometric_median.py` | Modify `:12-19`, `:31`, `:75-85` | Remove the `xfail` marker and correct the module docstring (Task 3). |
| `docs/superpowers/specs/2026-09-19-weiszfeld-coincident-point-singularity/README.md` | Modify | Amend the `d == 0` prescription, criterion 2, the Blast-radius row, and the status line (Task 4). |
| `docs/.../logic_validation_scripts/.../weiszfeld_singularity.py` | Modify | Add claims 8–9 against a **mean-initialised** subject (Task 4). |
| `docs/superpowers/plans/2026-09-19-weiszfeld-coincident-point-singularity/run_unit_suite.sbatch` | **Create** | The regression gate, pointed at a worktree detached at the SHA under test (Task 5). |

Nothing else changes. `_robust_color_stats.py`, `_measure_color.py` and `_color_checker_profile.py` are **callers only** — they reach the fix through `robust_color_center` without edits.

---

### Task 1: Extract the Weiszfeld return-pair builder

Near-pure refactor. Task 2 adds two more exit points to a function that already duplicates the `(x, info)` construction twice; doing this first keeps that diff about the algorithm.

**One deliberate behaviour change, despite the framing.** The helper guards the improvement print with `f_initial > 0.0`. Today a fully degenerate cloud (`f_initial == 0.0`) with `verbose=True` prints `Improvement: nan%` and raises a numpy `invalid value` RuntimeWarning; afterwards it prints nothing. Nothing in the suite runs the solver with `verbose=True`, so no test can see this — say it in the commit message rather than letting it ride as an unannounced change (plan-review PR-11).

**Files:**
- Modify: `src/phenotypic/util/_geometric_median.py:1145-1176`
- Test: `tests/unit/util/test_robust_color_stats.py` (existing, must stay green)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `_weiszfeld_result(x: np.ndarray, points: np.ndarray, iterations: int, f_initial: float, converged: bool, verbose: bool) -> Tuple[np.ndarray, Dict]` — private, module-level, used by every `return` in `weiszfeld_median`. `Tuple` and `Dict` are already imported at `:100`; no new import is needed.

- [ ] **Step 1: Capture the current behaviour as a baseline**

Run: `uv run pytest tests/unit/util/test_robust_color_stats.py tests/unit/correction/test_color_checker_geometric_median.py -q -p no:randomly`

Expected, measured on this branch before any change: **`20 passed, 1 xfailed`**. The xfailed one is `test_patch_center_is_not_short_circuited_by_a_coincident_pixel`; it is `strict=True`, so "xfailed" is its pass state until Task 3.

- [ ] **Step 2: Add the private helper**

Insert immediately **above** `def weiszfeld_median(` (currently `:1097`):

```python
def _weiszfeld_result(
    x: np.ndarray,
    points: np.ndarray,
    iterations: int,
    f_initial: float,
    converged: bool,
    verbose: bool,
) -> Tuple[np.ndarray, Dict]:
    """Build the ``(x, info)`` pair returned by :func:`weiszfeld_median`.

    Every exit from the solver reports the same fields, so they are built in
    one place. ``f_initial`` is only used for the printed improvement line and
    is skipped when it is zero (a cloud of identical points), where the ratio
    would be ``0/0`` -- today that prints ``Improvement: nan%`` and raises a
    numpy invalid-value warning.

    Args:
        x: The estimate being returned, shape (d,).
        points: Data points, shape (n, d).
        iterations: Number of iterations actually performed.
        f_initial: Objective at the starting estimate.
        converged: Whether the solver reached a fixed point.
        verbose: Whether to print the summary.

    Returns:
        ``(x, info)`` exactly as :func:`weiszfeld_median` documents it.
    """
    objective = compute_geometric_median_objective(x, points)
    if verbose:
        if converged:
            print(f"✓ Converged after {iterations} iterations")
        else:
            print("⚠ Maximum iterations reached")
        print(f"  Final: f(x) = {objective:.6f}")
        if converged and f_initial > 0.0:
            print(f"  Improvement: {((f_initial - objective) / f_initial) * 100:.2f}%")
    return x, {
        "iterations": iterations,
        "objective": objective,
        "initial_objective": f_initial,
        "converged": converged,
        "method": "weiszfeld",
    }
```

- [ ] **Step 3: Route both existing returns through it**

**The prose is the authority here; the line numbers are given only to locate the blocks.** The first draft of this plan had both ranges wrong in ways that pass the Step 4 check (plan-review PR-4), so match on the quoted text.

Replace the converged block — `:1145-1159`, everything from `if change < eps:` through the closing `}` of its `return` — with:

```python
        if change < eps:
            return _weiszfeld_result(
                x, points, iteration + 1, f_initial, True, verbose
            )
```

**Leave `:1161-1163` — the `if verbose and (iteration + 1) % 100 == 0:` progress print — exactly where it is.** It sits between the two blocks you are replacing.

Replace the exhausted-iterations tail — `:1165-1176`, from `objective = compute_geometric_median_objective(x, points)` through the closing `}` of the final `return`, which is the last line of the function — with:

```python
    return _weiszfeld_result(x, points, max_iter, f_initial, False, verbose)
```

**Do not delete the `# ===== Main Interface Function =====` banner at `:1179-1181`.** It belongs to the next function.

- [ ] **Step 4: Verify nothing moved**

Run: `uv run pytest tests/unit/util/test_robust_color_stats.py tests/unit/correction/test_color_checker_geometric_median.py -q -p no:randomly`

Expected: identical to Step 1 — `20 passed, 1 xfailed`.

This check is weak on purpose-built grounds: no test runs `weiszfeld_median` with `verbose=True` at iteration exhaustion, so a botched Step 3 can pass it. Before moving on, read the function top to bottom and confirm it has exactly two `return` statements, both calling `_weiszfeld_result`, and that the progress print and the section banner both survive.

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check --fix src/phenotypic/util/_geometric_median.py; uv run mypy src/phenotypic/util/_geometric_median.py`

Expected: `mypy` reports `Success: no issues found in 1 source file`. **`ruff` reports exactly one finding — the pre-existing `F841` at `:680`, "Local variable `n` is assigned to but never used", and exits 1.** That is the dead Cohen path and is out of scope; do not fix it. Any *other* finding is yours.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/util/_geometric_median.py
git commit -m "refactor(util): build the Weiszfeld return pair in one place

Also stops the 0/0 'Improvement: nan%' print (and its numpy invalid-value
warning) on a fully degenerate cloud under verbose=True. No test covers that
path, so it is recorded here rather than left silent."
```

---

### Task 2: Split coincident points out of the reweighting

The fix itself, plus the solver-level guards that prove it.

**Files:**
- Modify: `src/phenotypic/util/_geometric_median.py:1135-1141` (the update block) and the docstrings
- Create: `tests/unit/util/test_geometric_median.py`

**Interfaces:**
- Consumes: `_weiszfeld_result(...)` from Task 1.
- Produces: `_COINCIDENCE_RTOL: float` (module constant, `1e-12`) and `_coincidence_atol(x: np.ndarray) -> float`. `weiszfeld_median`'s signature and return contract are unchanged.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/util/test_geometric_median.py`:

```python
"""Solver-level guards for the Weiszfeld coincident-point singularity.

``weiszfeld_median`` used to floor the per-point distance at ``1e-10`` before
inverting it. A point sitting on the running estimate therefore took ~1e10 of
weight, captured the weighted update, and was returned as the answer with
``converged: True`` after a single iteration. Since the first estimate is the
*mean*, the returned value was the mean -- the non-robust statistic the
geometric median exists to avoid.

These guards sit at the solver, below ``robust_color_center``. The
colony-checker-level consequences are pinned separately in
``tests/unit/correction/test_color_checker_geometric_median.py``; both are
needed, because a fix at one level does not imply a fix at the other.

Spec: docs/superpowers/specs/2026-09-19-weiszfeld-coincident-point-singularity/
"""

from __future__ import annotations

import numpy as np

from phenotypic.util._geometric_median import (
    _coincidence_atol,
    weiszfeld_median,
)


def _reference_median(points: np.ndarray) -> np.ndarray:
    """Vardi-Zhang run to machine precision, transcribed independently.

    This is independent of the *implementation*, not of the *algorithm*: it
    uses the same split, the same radius and the same gamma, differing only in
    initialization and tolerance. A wrong radius rule or a wrong gamma would be
    invisible to it. That is why the guards below also cross-check against
    ``_clipped_classical`` (a genuinely different algorithm) and against the
    objective, which no transcription can fake.
    """
    pts = np.asarray(points, dtype=np.float64)
    y = np.median(pts, axis=0)
    for _ in range(100_000):
        dist = np.linalg.norm(pts - y, axis=1)
        on_estimate = dist <= 1e-12 * max(float(np.linalg.norm(y)), 1.0)
        eta = int(on_estimate.sum())
        far = pts[~on_estimate]
        far_dist = dist[~on_estimate]
        if far.size == 0:
            return y
        weights = 1.0 / far_dist
        reweighted = (far * weights[:, None]).sum(axis=0) / weights.sum()
        if eta == 0:
            nxt = reweighted
        else:
            r = float(np.linalg.norm(((far - y) / far_dist[:, None]).sum(axis=0)))
            if r == 0.0:
                return y
            gamma = min(1.0, eta / r)
            nxt = (1.0 - gamma) * reweighted + gamma * y
        if np.linalg.norm(nxt - y) < 1e-13:
            return nxt
        y = nxt
    raise AssertionError("reference geometric median did not converge")


def _clipped_classical(points: np.ndarray) -> np.ndarray:
    """Classical Weiszfeld from the coordinate-wise median, clipped at 1e-15.

    A different algorithm from ``_reference_median`` -- no split, no damping --
    so agreement between the two is evidence about the answer rather than about
    a shared transcription.
    """
    pts = np.asarray(points, dtype=np.float64)
    guess = np.median(pts, axis=0)
    for _ in range(100_000):
        dist = np.clip(np.linalg.norm(pts - guess, axis=1), 1e-15, None)
        weights = 1.0 / dist
        nxt = (pts * weights[:, None]).sum(axis=0) / weights.sum()
        if np.linalg.norm(nxt - guess) < 1e-15:
            return nxt
        guess = nxt
    raise AssertionError("clipped classical reference did not converge")


def _old_floored_update(points: np.ndarray, eps: float, max_iter: int) -> np.ndarray:
    """The rule this change replaces, transcribed locally.

    Kept so the clean path can be checked for bit-identity against the code it
    replaced, rather than against a remembered iteration count.
    """
    pts = np.asarray(points, dtype=np.float64)
    x = pts.mean(axis=0)
    for _ in range(max_iter):
        x_old = x.copy()
        dist = np.maximum(np.linalg.norm(pts - x, axis=1), 1e-10)
        weights = 1.0 / dist
        x = np.sum(pts * weights[:, None], axis=0) / np.sum(weights)
        if np.linalg.norm(x - x_old) < eps:
            return x
    return x


def _skewed_cloud(seed: int = 11) -> np.ndarray:
    """A checker-patch core lit unevenly: tight body plus a brightness ramp.

    The skew is the point: on a symmetric cloud the mean and the geometric
    median coincide and a broken solver looks correct.
    """
    rng = np.random.default_rng(seed)
    base = np.array([0.35, 0.37, 0.40])
    body = rng.normal(base, 0.004, size=(3000, 3))
    ramp = base + rng.random((900, 1)) * 0.45 + rng.normal(0, 0.004, size=(900, 3))
    pixels = np.clip(np.vstack([body, ramp]), 0.0, 1.0)
    return np.round(pixels * 255.0) / 255.0  # 8-bit, as real swatch pixels are


def _objective(x: np.ndarray, points: np.ndarray) -> float:
    """Sum of Euclidean distances -- the quantity the median minimises."""
    return float(np.linalg.norm(np.asarray(points) - x, axis=1).sum())


def test_a_point_exactly_on_the_estimate_does_not_capture_the_solve():
    """The classical singularity: one pixel at the centroid, out of 3901."""
    cloud = _skewed_cloud()
    points = np.vstack([cloud, cloud.mean(axis=0)])

    # The planted pixel is at distance 0 from the first estimate, so this
    # exercises the degenerate branch rather than depending on luck. Measured
    # exactly 0.0 today; if a numpy change ever makes it ~1e-16 the test still
    # exercises the intended branch, so only this precondition would need
    # relaxing to `<= _coincidence_atol(start)`.
    start = points.mean(axis=0)
    assert np.linalg.norm(points - start, axis=1).min() == 0.0

    got, info = weiszfeld_median(points, eps=1e-6, max_iter=200, verbose=False)
    reference = _reference_median(points)

    assert np.allclose(got, reference, atol=1e-5)
    # Cross-check against a different algorithm, and against the objective --
    # the one thing no transcription can fake.
    assert np.allclose(got, _clipped_classical(points), atol=1e-5)
    assert _objective(got, points) < _objective(start, points)

    # The bug returned after exactly one iteration, pinned to the planted pixel.
    assert info["iterations"] > 1
    assert info["converged"] is True
    assert not np.any(np.all(points == got, axis=1))


def test_a_point_just_off_the_estimate_does_not_capture_the_solve():
    """Why the coincidence test is a radius and not an ``== 0`` comparison.

    A pixel 1.11e-16 from the estimate is not equal to it, so an exact-zero
    test lets it through to ``1/d`` with weight ~9e15 and it captures the
    solve just as the old 1e-10 floor did. Measured, that answers 21.896
    8-bit code values from the median -- the same magnitude as the original
    defect.
    """
    cloud = _skewed_cloud()
    points = np.vstack([cloud, cloud.mean(axis=0) + np.array([1e-16, 0.0, 0.0])])

    smallest = np.linalg.norm(points - points.mean(axis=0), axis=1).min()
    assert 0.0 < smallest < 1e-15  # near, but genuinely not, coincident

    got, _info = weiszfeld_median(points, eps=1e-6, max_iter=200, verbose=False)
    assert np.allclose(got, _reference_median(points), atol=1e-5)


def test_every_point_identical_returns_that_point():
    """A fully degenerate cloud has no non-coincident points to reweight."""
    points = np.tile([7.0, 7.0, 7.0], (5, 1))

    got, info = weiszfeld_median(points, eps=1e-6, max_iter=200, verbose=False)

    assert np.allclose(got, [7.0, 7.0, 7.0], atol=1e-12)
    assert info["converged"] is True


def test_converged_carries_the_subgradient_certificate():
    """``converged: True`` must certify optimality to the accuracy asked for.

    The guarantee the stopping rule actually gives is ``r <= eta + eps*W``
    (see the plan's Property 3), not ``r <= eta``: damping shortens the step by
    ``eta/W``, so the loop can stop while gamma < 1.
    """
    cloud = _skewed_cloud()
    points = np.vstack([cloud, cloud.mean(axis=0)])

    # The shipped constants (GEOMEDIAN_TOL / GEOMEDIAN_MAX_ITER, which
    # ColorCheckerProfile passes). This is not incidental: at a tighter
    # tolerance with more iterations the OLD floored rule escapes the capture
    # after its first step and converges correctly, so this guard would pass
    # against the very bug it exists to catch. Measured: the old rule at
    # eps=1e-9/5000 lands 0.000 code values from the median; at eps=1e-6/200
    # it lands 21.896 away.
    eps = 1e-6
    got, info = weiszfeld_median(points, eps=eps, max_iter=200, verbose=False)
    assert info["converged"] is True

    dist = np.linalg.norm(points - got, axis=1)
    on_estimate = dist <= _coincidence_atol(got)
    eta = int(on_estimate.sum())
    far = points[~on_estimate]
    far_dist = dist[~on_estimate]
    r = float(np.linalg.norm(((far - got) / far_dist[:, None]).sum(axis=0)))
    w_total = float((1.0 / far_dist).sum())

    # Measured at these constants: eta=0, r=1.240e-01, W=5.344e+05,
    # bound=5.344e-01 -> passes with ~4.3x margin. The pre-fix answer fails it:
    # r=2317.70 against its own bound of 4.360 (W is larger there, because the
    # planted pixel sits just outside the radius of the captured estimate), so
    # this assertion is not vacuous.
    assert r <= eta + eps * w_total


def test_the_non_degenerate_path_is_bit_identical_to_the_old_rule():
    """With no point in the coincidence radius, the arithmetic is the old one.

    Asserted as bit-identity against a local transcription of the floored
    update rather than as an iteration count: a count is a function of numpy's
    summation blocking and would be "fixed" by editing the number if a numpy
    upgrade moved it.

    This holds because no distance on this cloud falls in the band
    ``(radius, 1e-10]`` where the removed floor and the new rule genuinely
    differ -- the nearest point to any iterate here is ~1.24e-03.
    """
    cloud = _skewed_cloud()

    got, info = weiszfeld_median(cloud, eps=1e-6, max_iter=200, verbose=False)

    assert np.array_equal(got, _old_floored_update(cloud, 1e-6, 200))
    assert np.allclose(got, _reference_median(cloud), atol=1e-5)
    assert info["iterations"] == 12  # secondary; the bit-identity is the guard
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/util/test_geometric_median.py -q -p no:randomly`

Expected: a **collection error**, not a failure list —
`ImportError: cannot import name '_coincidence_atol'`, `collected 0 items / 1 error`.

**That is a weak red, and Step 3a below exists because of it.** A collection error proves a name is missing and nothing more: the five guards never execute. On a naive step order they would go from never-run straight to green, and a test observed only passing is not yet known to be a test. Do **not** treat this as evidence the guards work.

- [ ] **Step 3: Add the coincidence radius**

Insert immediately **above** `def _weiszfeld_result(` (added in Task 1):

```python
_COINCIDENCE_RTOL = 1e-12
"""Relative radius inside which a data point counts as *on* the estimate.

The test is floating-point-relative, not statistical. ``1e-12 * ||x||`` is a
few thousand ULPs at ``x``, so it asks "is this point indistinguishable from
the iterate at double precision?" -- which is exactly the question ``1/d``
blows up on. That is also why it scales with ``||x||`` rather than with the
data's spread: the failure mode is a floating-point one.

Coincidence must be a radius rather than an equality test. A point at
``1.11e-16`` from the estimate is not equal to it, but ``1/d`` still gives it
~1e15 of weight and it captures the update exactly as the old ``1e-10``
distance floor did -- measured at 21.9 8-bit code values of error.

It is harmless at every scale this codebase works in: 8-bit pixels are
``1/255`` apart, and for L*a*b* near (50, 10, 20) the radius is ~5.5e-11
against a nearest-neighbour spacing many orders of magnitude larger.

The ``max(||x||, 1)`` floor means the solver has a **1e-12 absolute**
resolution floor near the origin: a caller working at a coordinate scale below
~1e-11 would see every point swallowed into the coincident set and get the
mean back. That is within the cloud's own diameter of the truth, so it is
harmless, but ``geometric_median`` is a public export and this is part of its
contract.
"""


def _coincidence_atol(x: np.ndarray) -> float:
    """Absolute radius for the coincidence test at estimate ``x``.

    Args:
        x: Current estimate, shape (d,).

    Returns:
        The distance at or below which a point is treated as lying on ``x``.
    """
    return _COINCIDENCE_RTOL * max(float(np.linalg.norm(x)), 1.0)
```

- [ ] **Step 3a: Watch the guards fail against the OLD solver**

**Do not skip this, and do not merge it into Step 4.** Step 3 adds only the two
new names; the update block is still the floored rule. So the import now
resolves and the five guards run against the code they are meant to catch —
the only moment in the whole plan when that is possible.

Run: `uv run pytest tests/unit/util/test_geometric_median.py -q -p no:randomly`

Expected: **`3 failed, 2 passed`**.

| Test | Expected | Why |
|---|---|---|
| `test_a_point_exactly_on_the_estimate_does_not_capture_the_solve` | **FAIL** | Old rule returns the mean, `0.4010/0.4212/0.4509`, against the median `0.3515/0.3716/0.4013` — 21.9 code values. |
| `test_a_point_just_off_the_estimate_does_not_capture_the_solve` | **FAIL** | Same magnitude, via the `1.11e-16` pixel. |
| `test_converged_carries_the_subgradient_certificate` | **FAIL** | Old answer has `r = 2317.70` against **its own** bound of `4.3595` (`W = 4.36e6` there, not the post-fix `5.34e5`) — 531×. Every quantity on this change has a different value at each configuration; pair them carefully. |
| `test_every_point_identical_returns_that_point` | **PASS** | Regression guard; the old rule already handles it. |
| `test_the_non_degenerate_path_is_bit_identical_to_the_old_rule` | **PASS**, `iterations == 12` included | Trivially — at this stage the solver *is* the old rule. This is what proves `_old_floored_update` is a faithful transcription; if it fails here the bit-identity guard is worthless after the fix. |

**Any deviation from that table is a defect in the test file, not in the solver — stop and say so.** This step has already caught one: the certificate test originally ran at `eps=1e-9, max_iter=5000` and **passed against the old solver**, because at a tight tolerance with enough iterations the floored rule escapes the capture after its first step and converges correctly (measured: 21 iterations, 0.000 code values of error). It only fails at the shipped `eps=1e-6, max_iter=200`. A guard aimed at the wrong configuration is indistinguishable from a guard that works, right up until it is needed.

- [ ] **Step 4: Replace the update rule**

In `weiszfeld_median`, replace the block from `# Compute weights: w_i = 1/||x - a^(i)||_2` through the `x = np.sum(...)` line (`:1135-1141`) with:

```python
        # Vardi & Zhang (2000): split the points lying *on* the estimate out of
        # the reweighting. 1/d is undefined there, and flooring d does not
        # define it -- it hands those points ~1e10 of weight and pins the
        # estimate to them, which is the defect this replaces.
        distances = np.linalg.norm(points - x, axis=1)
        on_estimate = distances <= _coincidence_atol(x)
        eta = int(on_estimate.sum())

        far = points[~on_estimate]
        far_distances = distances[~on_estimate]

        if far.size == 0:
            # Every point coincides with the estimate, so it is the median.
            return _weiszfeld_result(
                x, points, iteration + 1, f_initial, True, verbose
            )

        weights = 1.0 / far_distances
        reweighted = (
            np.sum(far * weights[:, np.newaxis], axis=0) / np.sum(weights)
        )

        if eta == 0:
            # No coincidence: exactly the classical Weiszfeld update.
            x = reweighted
        else:
            # r is the norm of the non-coincident part of the subgradient.
            # 0 lies in the subdifferential iff r <= eta, i.e. iff gamma == 1 --
            # so gamma == 1 gives a zero step that is an optimality
            # certificate, not a stall. (The `r == 0.0` guard is required:
            # `eta / r` raises ZeroDivisionError there, it does not yield inf.)
            r = float(
                np.linalg.norm(
                    np.sum((far - x) / far_distances[:, np.newaxis], axis=0)
                )
            )
            if r == 0.0:
                return _weiszfeld_result(
                    x, points, iteration + 1, f_initial, True, verbose
                )
            gamma = min(1.0, eta / r)
            x = (1.0 - gamma) * reweighted + gamma * x
```

Leave `x_old = x.copy()`, the `change = np.linalg.norm(x - x_old)` test and everything below it untouched.

- [ ] **Step 5: Update the docstrings**

Replace the `Reference:` and `Iterative reweighting:` lines in `weiszfeld_median`'s docstring with:

```
    Reference: Weiszfeld (1937), with the coincident-point handling of
    Vardi & Zhang (2000).

    Iterative reweighting: x^(k+1) = Σ w_i a^(i) / Σ w_i where
    w_i = 1/||x^(k) - a^(i)||_2. Points lying on x^(k) are excluded from that
    sum and the step is damped toward them by γ = min(1, η/r), which is what
    keeps a data point on the estimate from capturing the iteration. With no
    coincident point (η = 0) this is the classical update unchanged.

    Convergence reports ``r <= η + eps*W`` (W = Σ 1/||a - x|| over the
    non-coincident points), which is exact optimality when γ = 1 and an
    eps-scaled subgradient slack otherwise -- the same slack the classical
    rule has always carried.
```

Add to `geometric_median`'s public docstring, in the `Args:` entry for `eps` or immediately below it:

```
        Note: points within ``1e-12 * max(||x||, 1)`` of the running estimate
        are treated as coincident with it. For data at a coordinate scale below
        ~1e-11 this collapses the whole cloud into that set and returns the
        mean.
```

Add to the module docstring's `Reference:` block, after the Cohen et al. entry:

```
    Vardi, Y., & Zhang, C.-H. (2000). The multivariate L1-median and
    associated data depth. PNAS, 97(4), 1423-1426.
    https://doi.org/10.1073/pnas.97.4.1423
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/util/test_geometric_median.py -q -p no:randomly`

Expected: `5 passed`.

- [ ] **Step 7: Confirm the callers still agree**

Run: `uv run pytest tests/unit/util/test_robust_color_stats.py -q -p no:randomly`

Expected: all pass — acceptance criterion 4. The plan review hand-traced each case through the new update and expects no movement; `test_robust_center_resists_single_outlier` is the one worth watching, since its iterate approaches a 99-point cluster geometrically.

- [ ] **Step 8: Lint and type-check**

Run: `uv run ruff check --fix src/phenotypic/util/_geometric_median.py tests/unit/util/test_geometric_median.py; uv run mypy src/phenotypic/util/_geometric_median.py`

Expected: `mypy` clean; `ruff` reports **only** the pre-existing `F841` at `:680`.

- [ ] **Step 9: Do not commit yet**

The `xfail(strict=True)` marker in `tests/unit/correction/test_color_checker_geometric_median.py` turns into a **failure** the moment this change lands, so committing here would put a red tree in history and a bisect-hostile point in the log (plan-review PR-14). Task 3 removes the marker; the two are committed together at Task 3 Step 6.

Run `uv run pytest tests/unit/correction/test_color_checker_geometric_median.py -q -p no:randomly` now if you want to see it: expected `1 failed, 3 passed`, the failure reported as `[XPASS(strict)]`. Then go straight to Task 3.

---

### Task 3: Flip the colour-checker xfail to green

The marker is `strict=True`, so it became a failure the moment Task 2 landed. This task clears it, and it is the spec's acceptance criterion 1.

**Files:**
- Modify: `tests/unit/correction/test_color_checker_geometric_median.py:12-19` (docstring bullet), `:31` (import), `:75-85` (the marker)

**Interfaces:**
- Consumes: the fixed `weiszfeld_median` from Task 2, reached via `robust_color_center`.
- Produces: nothing downstream.

- [ ] **Step 1: Confirm the strict xfail is now a failure**

Run: `uv run pytest tests/unit/correction/test_color_checker_geometric_median.py -q -p no:randomly`

Expected: **`1 failed, 3 passed`** — the failure is `test_patch_center_is_not_short_circuited_by_a_coincident_pixel`, reported as `[XPASS(strict)]`. That is the marker doing its job: the test now passes, and `strict` refuses to let the stale marker survive.

- [ ] **Step 2: Remove the marker**

Delete the entire `@pytest.mark.xfail(...)` decorator above `def test_patch_center_is_not_short_circuited_by_a_coincident_pixel` (`:75-85`, from `@pytest.mark.xfail(` through its closing `)`), leaving the function and its docstring intact.

`pytest` is then unused — it appears in this file only at `:31` (the import) and `:75` (this marker). Change:

```python
import numpy as np
import pytest
```

to:

```python
import numpy as np
```

- [ ] **Step 3: Correct the module docstring**

In the module docstring, replace the bullet at `:12-19` — the whole bullet, beginning `* ``test_patch_center_is_not_short_circuited_by_a_coincident_pixel`` pins the` and ending `` in ``docs/superpowers/specs/2026-09-19-weiszfeld-coincident-point-singularity/``. `` — with:

```
* ``test_patch_center_is_not_short_circuited_by_a_coincident_pixel`` pins the
  degenerate branch -- a pixel sitting on the running estimate must not end
  the solve and be returned as the answer. Routing through
  ``robust_color_center`` did not close this branch on its own: the shared
  solver floored the distance at ``1e-10``, which handed that pixel
  ~99.9995% of the weight and pinned the estimate to it anyway. Closed by the
  Vardi-Zhang coincident-point split in ``weiszfeld_median``; see
  ``docs/superpowers/specs/2026-09-19-weiszfeld-coincident-point-singularity/``
  and the solver-level guards in ``tests/unit/util/test_geometric_median.py``.
```

- [ ] **Step 4: Run the file**

Run: `uv run pytest tests/unit/correction/test_color_checker_geometric_median.py -v -p no:randomly`

Expected: **4 passed**, no xfail, no xpass.

- [ ] **Step 5: Lint**

Run: `uv run ruff check --fix tests/unit/correction/test_color_checker_geometric_median.py`

Expected: clean (this file has no pre-existing findings; the check also catches the `pytest` import if Step 2 missed it).

- [ ] **Step 6: Commit Tasks 2 and 3 together**

One commit, so no point in history has a red suite.

```bash
git add src/phenotypic/util/_geometric_median.py \
        tests/unit/util/test_geometric_median.py \
        tests/unit/correction/test_color_checker_geometric_median.py
git commit -m "fix(util): handle coincident points in the Weiszfeld iteration

Split points lying on the running estimate out of the reweighting and take
the Vardi-Zhang damped step, instead of flooring the distance at 1e-10 and
letting such a point take ~1e10 of the weight. Coincidence is tested with a
scale-relative radius, not d == 0: a point 1.11e-16 from the estimate is not
equal to it and still captures the solve, at 21.9 code values of error.

The strict xfail guarding this branch is removed in the same commit, so no
point in history has a red suite."
```

---

### Task 4: Amend the spec and extend the witness

The spec prescribes `d == 0`; the implementation uses a radius, for the measured reason in *Decision 1*. The spec is the artifact a later reader trusts, so it gets corrected rather than quietly diverged from — and the witness grows the claims that justify the correction.

**Files:**
- Modify: `docs/superpowers/specs/2026-09-19-weiszfeld-coincident-point-singularity/README.md`
- Modify: `docs/superpowers/logic_validation_scripts/2026-09-19-weiszfeld-coincident-point-singularity/weiszfeld_singularity.py`

**Interfaces:**
- Consumes: the coincidence radius `1e-12 · max(‖x‖, 1)` from Task 2.
- Produces: nothing downstream. Documentation and the executable witness only.

> **Why the witness needs a new subject rather than a new mode on `vardi_zhang`.** The first draft added a `coincide=` switch to `vardi_zhang`, which is **median**-initialised. The defect only a **mean**-initialised solver can reach, so both modes would return the same answer and claim 8 would print PASS whether or not the effect existed — a check whose success message is identical to its no-op message. `vardi_zhang` must also stay exactly as it is because claims 1, 4, 5, 6 and 7 all consume its output as `truth`.

- [ ] **Step 1: Add a mean-initialised subject to the witness**

In `weiszfeld_singularity.py`, add after `vardi_zhang` (leave `vardi_zhang` itself **unchanged**):

```python
def vz_mean(points, coincide="radius", eps=1e-12, max_iter=100_000):
    """Vardi-Zhang from the *mean*, with a selectable coincidence test.

    The shipped solver starts at the mean, and the singularity is reachable
    only from there: the planted pixel sits at the centroid. ``vardi_zhang``
    above starts at the coordinate-wise median and is the ground truth claims
    1-7 consume; it must not be used as the subject here, because from a median
    start neither coincidence test ever fires and the comparison is vacuous.

    ``coincide="exact"`` is the ``d == 0`` rule the spec first prescribed;
    ``"radius"`` is the scale-relative rule that ships.
    """
    pts = np.asarray(points, dtype=np.float64)
    y = pts.mean(axis=0)
    for _ in range(max_iter):
        y_old = y.copy()
        d = np.linalg.norm(pts - y, axis=1)
        if coincide == "exact":
            on = d <= 0.0
        else:
            on = d <= 1e-12 * max(float(np.linalg.norm(y)), 1.0)
        eta = int(on.sum())
        far, d_far = pts[~on], d[~on]
        if far.size == 0:
            return y
        w = 1.0 / d_far
        T = (far * w[:, None]).sum(axis=0) / w.sum()
        if eta == 0:
            y = T
        else:
            r = np.linalg.norm(((far - y) / d_far[:, None]).sum(axis=0))
            if r == 0.0:
                return y
            gamma = min(1.0, eta / r)
            y = (1.0 - gamma) * T + gamma * y
        if np.linalg.norm(y - y_old) < eps:
            return y
    raise AssertionError("vz_mean did not converge")


def masked_step(points, y):
    """One no-floor masked update -- the shipped rule's non-degenerate branch."""
    pts = np.asarray(points, dtype=np.float64)
    d = np.linalg.norm(pts - y, axis=1)
    on = d <= 1e-12 * max(float(np.linalg.norm(y)), 1.0)
    far, d_far = pts[~on], d[~on]
    w = 1.0 / d_far
    return (far * w[:, None]).sum(axis=0) / w.sum()


def floored_step(points, y):
    """One floored update -- the rule being replaced."""
    pts = np.asarray(points, dtype=np.float64)
    w = 1.0 / np.maximum(np.linalg.norm(pts - y, axis=1), CLAMP)
    return (pts * w[:, None]).sum(axis=0) / w.sum()
```

- [ ] **Step 2: Add claims 8 and 9**

Append after claim 7, before the `print()` / exit block:

```python
# Claim 8: an exact `d == 0` coincidence test is NOT sufficient. A pixel
# placed 1.11e-16 from the estimate is not equal to it, so an exact test
# lets it through to 1/d with ~9e15 of weight and it captures the solve --
# the same failure, at the same magnitude, as the 1e-10 floor. The subject is
# mean-initialised on purpose; from a median start neither test ever fires.
nudged = np.vstack([swatch, mean + np.array([1e-16, 0.0, 0.0])])
d_min = np.linalg.norm(nudged - nudged.mean(axis=0), axis=1).min()
truth_nudged = vardi_zhang(nudged)
err_exact = np.linalg.norm(vz_mean(nudged, "exact") - truth_nudged) * 255.0
err_radius = np.linalg.norm(vz_mean(nudged, "radius") - truth_nudged) * 255.0
check(
    "8. an exact d==0 coincidence test still loses the solve",
    d_min > 0.0 and err_exact > 5.0 and err_radius < 1e-6,
    f"d_min = {d_min:.3e} (strictly positive, so `== 0` misses it); "
    f"exact-test error = {err_exact:.3f} code values, "
    f"radius-test error = {err_radius:.3e} code values",
)

# Claim 9: the radius costs nothing where nothing coincides, and does
# something where something does. Both legs are needed: leg A alone would also
# pass if the mask silently never fired, which is the failure mode that makes
# a green check meaningless.
y0 = swatch.mean(axis=0)
clean_masked, clean_floored = masked_step(swatch, y0), floored_step(swatch, y0)
leg_a = np.array_equal(clean_masked, clean_floored)
y1 = nudged.mean(axis=0)
nudged_masked, nudged_floored = masked_step(nudged, y1), floored_step(nudged, y1)
leg_b = not np.allclose(nudged_masked, nudged_floored, atol=1e-9)
check(
    "9. masked update == floored update iff nothing coincides",
    leg_a and leg_b,
    f"clean: bit-identical={leg_a} (first coord {clean_masked[0]!r}); "
    f"nudged: differ={leg_b} by "
    f"{np.linalg.norm(nudged_masked - nudged_floored) * 255.0:.3f} code values",
)
```

- [ ] **Step 3: Run the witness**

Run: `uv run python docs/superpowers/logic_validation_scripts/2026-09-19-weiszfeld-coincident-point-singularity/weiszfeld_singularity.py; echo "exit=$?"`

Expected: 9 `[PASS]` lines, `All claims verified.`, `exit=0`.

**Read claim 8's and claim 9's detail lines, do not just count PASSes.** Measured values to expect, so a vacuous pass is visible:

| Quantity | Expected |
|---|---|
| claim 8 `d_min` | `1.110e-16` — strictly positive, which is the whole point |
| claim 8 `exact-test error` | **`21.896`** code values. A near-zero here means the subject is not reaching the singularity and the claim has gone vacuous again. |
| claim 8 `radius-test error` | `4.009e-11` code values |
| claim 9 leg A | `bit-identical=True` |
| claim 9 leg B | `differ=True`, by **`12.649`** code values. `differ=False` means the mask never fired — the check would then be green while proving nothing. |

Claims 3 and 4 still describe the **pre-fix** rule via `shipped_weiszfeld`, a local transcription of the floored update. They stay green after the fix because the script never imports `phenotypic`. That is the point of the file rule; do not "update" them to the new code.

- [ ] **Step 4: Amend the spec's prescription**

In `README.md`, replace the whole `### Compare the distance floor` section with:

```markdown
### The distance floor is removed outright

The floor is gone, not relocated. Once coincident points are split out of the
reweighting, every remaining distance is strictly greater than the coincidence
radius by construction, so `1/d` is finite for every input and there is nothing
left to guard. Re-adding a floor would reintroduce a weaker form of the
original defect.

**Test coincidence with a scale-relative radius, not `d == 0`.** An exact
comparison is not sufficient: a point `1.11e-16` from the estimate is not
equal to it, so an exact test lets it through to `1/d` with ~9e15 of weight
and it captures the solve exactly as the `1e-10` floor did — measured at
**21.896** code values of error, the same magnitude as the original defect.
The hole is not monotone in the offset (`1e-15` and `1e-13` are both fine),
which makes it a lurking hazard rather than a boundary one can reason about.
That measurement is against a **mean-initialised** solver, which is what
ships; from a median start the singularity is unreachable and the comparison
says nothing.

The shipped radius is `1e-12 · max(‖x‖, 1)`. It is floating-point-relative:
`1e-12·‖x‖` is a few thousand ULPs at `x`, so the test asks whether a point is
indistinguishable from the iterate at double precision — which is exactly the
question `1/d` blows up on. That is why it scales with `‖x‖` rather than with
the data's spread. It is harmless at every scale here: 8-bit pixels are
`1/255` apart, and for L\*a\*b\* near (50, 10, 20) the radius is ~5.5e-11
against a far larger nearest-neighbour spacing. The `max(‖x‖, 1)` floor does
give the solver a **1e-12 absolute** resolution floor near the origin, which
is now part of the public contract of `geometric_median`.

One consequence worth stating: removing the floor means the new rule and the
old one differ not only at coincidence but anywhere a distance falls in
`(1e-12·max(‖x‖,1), 1e-10]` — floored before, unfloored now. No colorimetric
caller produces such a distance, but it is the honest scope of "unchanged".

Claims 8 and 9 of the witness re-derive both halves.
```

- [ ] **Step 5: Amend acceptance criterion 2**

Replace criterion 2 with:

```markdown
2. `weiszfeld_median` is no longer able to report `converged: True` while
   *pinned to* a data point: a coincident point is excluded from the
   reweighting, so it cannot dominate the update. Where the answer genuinely
   is a data point, `γ = 1` is an exact optimality certificate —
   `0 ∈ ∂f(x)` iff `r ≤ η` iff `γ = 1`.

   The stopping rule's guarantee in general is `r ≤ η + eps·W`, where
   `W = Σ_{a∈R} 1/‖a−x‖`, **not** `r ≤ η`. Damping shortens the step by `η/W`,
   so the loop can stop at `γ < 1`; measured on the planted swatch,
   `η/W = 1.87e-06` against `eps = 1e-06`, so this is reachable rather than
   theoretical. That slack is `eps`-scaled and is the same slack the classical
   rule has carried at `η = 0` since 1937 — this change neither introduces nor
   removes it. Pinned by
   `tests/unit/util/test_geometric_median.py::test_converged_carries_the_subgradient_certificate`,
   which asserts the true bound against the objective's own subgradient rather
   than trusting the solver's stopping test.
```

- [ ] **Step 6: Fix the stale Blast-radius threshold**

In the `## Blast radius` table, the `MeasureColor` row reads "Float Lab pixels rarely land within `1e-10` of the estimate, but nothing prevents it." `1e-10` is the floor being removed. Replace that sentence with:

```
Float Lab pixels rarely land within the coincidence radius
(`1e-12 · max(‖x‖, 1)`, ~5.5e-11 at a typical L*a*b* magnitude), but nothing
prevents it.
```

- [ ] **Step 7: Mark the spec implemented**

Change the `**Status:**` line at the top from:

```markdown
**Status:** **Specified, not implemented.** Deferred to its own PR.
```

to:

```markdown
**Status:** **Implemented** on `fix/geo-median-convergence`. Plan:
[`../../plans/2026-09-19-weiszfeld-coincident-point-singularity/plan.md`](../../plans/2026-09-19-weiszfeld-coincident-point-singularity/plan.md).
Plan review:
[`../../reports/2026-09-19-weiszfeld-coincident-point-singularity/plan-review.md`](../../reports/2026-09-19-weiszfeld-coincident-point-singularity/plan-review.md).
```

Also update the witness's check count. `7 checks` is on the **continuation line** (`README.md:8`), not on the `**Executable witness:**` line (`:7`) — change it to `9 checks`.

- [ ] **Step 8: Commit**

```bash
git add docs/superpowers/specs/2026-09-19-weiszfeld-coincident-point-singularity/README.md \
        docs/superpowers/logic_validation_scripts/2026-09-19-weiszfeld-coincident-point-singularity/weiszfeld_singularity.py
git commit -m "docs(spec): coincidence is a radius, not an equality test

An exact d==0 test misses a point 1.11e-16 from the estimate, which still
captures the solve at 21.9 code values of error. Witness claims 8-9 added
against a mean-initialised subject -- from the median start vardi_zhang uses,
neither coincidence test ever fires and the comparison would be vacuous.

Also records that the distance floor is removed outright rather than kept,
and states the stopping rule's real guarantee (r <= eta + eps*W)."
```

---

### Task 5: Regression over the affected surface

The surface is derived from importers, not directory names. `geometric_median` and `robust_color_center` are public exports; the modules that reach them are `_robust_color_stats.py`, `_measure_color.py` and `_color_checker_profile.py`.

**Files:**
- Create: `docs/superpowers/plans/2026-09-19-weiszfeld-coincident-point-singularity/run_unit_suite.sbatch`

**Interfaces:**
- Consumes: everything from Tasks 1–4.
- Produces: the evidence for acceptance criteria 3 and 6. **Not** criterion 5 — see the note under Step 1.

- [ ] **Step 1: Run the directly-affected tests**

Run:

```bash
uv run pytest \
  tests/unit/util/test_geometric_median.py \
  tests/unit/util/test_robust_color_stats.py \
  tests/unit/correction/test_color_checker_geometric_median.py \
  tests/unit/correction/test_color_corrector.py \
  tests/unit/correction/test_color_correction_report.py \
  tests/unit/measure/test_measure_color.py \
  -q -p no:randomly
```

Expected: all pass, zero xfail, zero xpass.

**On acceptance criterion 5, be accurate about what this does not do.** `test_measure_color.py` makes **no numeric assertion on any `ColorLab_*GeoMedian` value** — it checks header presence, hex format, ΔE non-negativity, opt-in XYZ/xy, a serialization round-trip and a numeric reduction. There is no numeric regression coverage for those columns anywhere in the repo (checked: `tests/unit/measure/_golden/` holds only orientation-zone artefacts). So this step cannot detect a shift in published measurement output, and a green run here is **not** evidence for criterion 5. That was a deliberate decision — see *Carried risk* below.

If a test does fail on a `ColorLab_*GeoMedian` value anyway, adjudicate between three cases, not two — the band from *Property 1* is a real third possibility (plan-review PR-6):

| Nearest fixture point to any iterate | Case | What it means |
|---|---|---|
| within `1e-12·max(‖x‖,1)` | (a) | The fixture triggered the singularity. The new value is correct and the old expectation was recording the bug. Say so in the commit message. |
| in `(1e-12·max(‖x‖,1), 1e-10]` | (b) | Benign and expected: the old floor was distorting the weight, the new code does not floor. Not a defect. |
| above `1e-10` | (c) | The non-degenerate path was not preserved — a real defect in Task 2. |

- [ ] **Step 2: Run the module doctest**

Run: `uv run pytest --doctest-modules src/phenotypic/util/_geometric_median.py -q -p no:randomly`

Expected: `1 passed`. The `geometric_median` example asserts `info['method'] == 'weiszfeld'` and `bool(info['converged']) is True`; it is green on this branch today, so a failure here is caused by this change.

- [ ] **Step 3: Re-run the witness**

Run: `uv run python docs/superpowers/logic_validation_scripts/2026-09-19-weiszfeld-coincident-point-singularity/weiszfeld_singularity.py; echo "exit=$?"`

Expected: `All claims verified.` and `exit=0` — acceptance criterion 6.

- [ ] **Step 4: Lint and type-check the whole change**

Run:

```bash
uv run ruff check --fix \
  src/phenotypic/util/_geometric_median.py \
  tests/unit/util/test_geometric_median.py \
  tests/unit/correction/test_color_checker_geometric_median.py
uv run mypy src/phenotypic/util
```

Expected: `mypy` clean; `ruff` reports **only** the pre-existing `F841` at `_geometric_median.py:680`. **Explicit paths only** — bare `ruff check --fix` rewrites the whole tree.

- [ ] **Step 5: Create the gate's batch script**

The committed script at `docs/superpowers/plans/2026-08-18-ome-zarr-image-store/run_unit_suite.sbatch` hard-codes `WORKTREE=.../.worktrees/worktree-ome-zarr-image-store`, which **no longer exists** — all 24 array tasks would die at its `cd` under `set -euo pipefail` (plan-review PR-2). It also must not be edited in place; it belongs to another change.

Create `docs/superpowers/plans/2026-09-19-weiszfeld-coincident-point-singularity/run_unit_suite.sbatch` as a copy with two changes: `WORKTREE` reads from the environment, and the job name identifies this gate.

```bash
cp docs/superpowers/plans/2026-08-18-ome-zarr-image-store/run_unit_suite.sbatch \
   docs/superpowers/plans/2026-09-19-weiszfeld-coincident-point-singularity/run_unit_suite.sbatch
```

Then in the new copy:

- Change `#SBATCH --job-name=phenotypic-tests` to `#SBATCH --job-name=geomedian-gate`.
- Replace the hard-coded `WORKTREE=...` line with:

  ```bash
  # Supplied by the submitting shell via --export. A gate measures ONE tree, so
  # this must be a worktree detached at the SHA under test -- never a live
  # branch checkout, which can be edited while the 24-task array is spread over
  # ~30 minutes, yielding a union across two trees that neither ever was.
  : "${WORKTREE:?WORKTREE must be exported (sbatch --export=ALL,WORKTREE=...)}"
  ```

- Leave everything else byte-identical, in particular `QT_QPA_PLATFORM=offscreen`, `-p no:randomly`, `-o addopts=`, `-m "not slow"`, the `unset SLURM_ARRAY_*` block, and the comment forbidding `-n auto`.

- [ ] **Step 6: Submit the gate against a detached worktree**

Create a worktree detached at the commit under test, so nothing can edit the tree mid-run:

```bash
SHA=$(git rev-parse HEAD)
GATE=/bigdata/exfab/anguy344/PhenoTypic/.claude/worktrees/gate-${SHA:0:8}
git worktree add --detach "$GATE" "$SHA"
mkdir -p /bigdata/exfab/anguy344/slurm_logs
( cd "$GATE" && uv sync --group dev --group test-qt --extra gui --extra napari )
```

Submit, capturing the array id — **`sbatch --parsable` returns an empty id on rejection while printing the error, so verify the id before using it**:

```bash
ARRAY_ID=$(sbatch --parsable --export=ALL,WORKTREE="$GATE" \
  docs/superpowers/plans/2026-09-19-weiszfeld-coincident-point-singularity/run_unit_suite.sbatch)
[[ "$ARRAY_ID" =~ ^[0-9]+$ ]] || { echo "SUBMIT FAILED: $ARRAY_ID"; exit 1; }
echo "array=$ARRAY_ID gate=$GATE"
```

Arm the cleanup as an `afterany` finalizer — a task that is OOM-killed never reaches its own cleanup line:

```bash
sbatch --dependency=afterany:"$ARRAY_ID" --partition=short --time=00:05:00 \
       --mem=2G --cpus-per-task=1 --job-name=geomedian-gate-cleanup \
       --output=/bigdata/exfab/anguy344/slurm_logs/%j.log \
       --wrap="git -C /bigdata/exfab/anguy344/PhenoTypic worktree remove --force $GATE"
```

Confirm it is queued, not merely submitted: `scontrol show job "$ARRAY_ID" | grep -E 'StartTime|Reason'`.

- [ ] **Step 7: Read the gate**

Logs land in `/bigdata/exfab/anguy344/slurm_logs/${ARRAY_ID}_*.log`.

Compare against the recorded baseline (11,106 tests, 81 pre-existing failures, all outside `sdk_`/`_cli`/`gui`). **The baseline was captured at 48 shards; this script uses 24.** The totals are comparable — it is the same file set — but the per-shard grouping differs, so cross-shard contamination lands differently. That makes the next rule load-bearing rather than advisory:

**Run every new failure in isolation before attributing it.** A red shard mixes this change with contamination from unrelated files sharing it. Read the failure *list*, never the count.

Expected: no new failures relative to the baseline; zero failures in `tests/unit/util/` and `tests/unit/correction/`.

- [ ] **Step 8: Commit the gate script and any fallout**

```bash
git add docs/superpowers/plans/2026-09-19-weiszfeld-coincident-point-singularity/run_unit_suite.sbatch
git commit -m "test: gate script for the geometric-median change

Copy of the ome-zarr runner with WORKTREE taken from the environment; the
original hard-codes a worktree that no longer exists. Submitted against a
worktree detached at the SHA under test, cleaned up by an afterany finalizer."
```

---

## Carried risk — acceptance criterion 5 is argued, not tested

**Decided deliberately, not overlooked.** Criterion 5 says the `ColorLab_*GeoMedian` columns must be unchanged on inputs that never triggered the singularity. Nothing in the repo tests that: `test_measure_color.py` makes no numeric assertion on those columns, and there is no golden corpus for them. The criterion rests on the argument in *Property 1* — the `η == 0` branch is arithmetically identical outside the `(radius, 1e-10]` band — and on nothing else.

The cheap way to close it was offered and declined: capture those columns from `load_synth_yeast_plate()` before the change and assert them after. If this ever needs to become evidence rather than argument, that is the task to add.

**Anyone reading the traceability table below should read this paragraph first.** The table says ARGUED for criterion 5 for exactly this reason.

---

## Self-review against the spec

| Spec item | Covered by |
|---|---|
| Objective — return the median when a point coincides | Task 2, Steps 3–4 |
| Non-goal: `method='cohen'` still raises | Global Constraints; no task touches it |
| Non-goal: public signature/defaults/return contract unchanged | Task 2 Interfaces; `_weiszfeld_result` is private |
| Non-goal: `GEOMEDIAN_*` not re-tuned | Global Constraints; pinned by an existing test |
| Proposed fix — Vardi & Zhang formulas | Task 2, Step 4 (verbatim `T`, `r`, `γ`, `x⁺`); transcription validated against a published restatement in the plan review (V-1) |
| `η = 0` reduces to the current update | Task 2 `test_the_non_degenerate_path_is_bit_identical_to_the_old_rule`; scope qualified by *Property 1* |
| `r == 0` is the optimality certificate | Task 2, Step 4 early return |
| *Compare the distance floor* (`d == 0`) | **Amended** — Task 4, Steps 1–4. See *Decision 1*. |
| Acceptance 1 — xfail removed, test passes | Task 3 |
| Acceptance 2 — honest `converged` flag | Task 2 `test_converged_carries_the_subgradient_certificate`; **criterion reworded** in Task 4 Step 5, because the original claim was false (*Property 3*) |
| Acceptance 3 — all-identical input returns that point | Task 2 `test_every_point_identical_returns_that_point` |
| Acceptance 4 — `test_robust_color_stats.py` unchanged | Task 2 Step 7; Task 5 Step 1 |
| Acceptance 5 — `ColorLab_*GeoMedian` unchanged | **ARGUED, NOT TESTED.** No numeric coverage exists anywhere in the repo. See *Carried risk*. |
| Acceptance 6 — witness exits zero | Task 4 Step 3; Task 5 Step 3 |

## References

Vardi, Y., & Zhang, C.-H. (2000). The multivariate L1-median and associated
data depth. *PNAS*, 97(4), 1423–1426. https://doi.org/10.1073/pnas.97.4.1423

Weiszfeld, E. (1937). Sur le point pour lequel la somme des distances de n
points donnés est minimum. *Tôhoku Mathematical Journal*, 43, 355–386.
