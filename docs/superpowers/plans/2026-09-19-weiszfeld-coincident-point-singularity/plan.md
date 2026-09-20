# Weiszfeld Coincident-Point Singularity — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `weiszfeld_median` return the geometric median when a data point lies on the running estimate, instead of returning that data point and reporting success.

**Architecture:** Replace the `np.maximum(distances, 1e-10)` distance floor with the Vardi & Zhang (2000) modified Weiszfeld step: split the points that sit *on* the estimate out of the reweighting, take a damped step toward the reweighted centroid of the rest, and treat the damping factor reaching 1 as an optimality certificate. With no coincident point the step reduces **exactly** to today's arithmetic, so the non-degenerate path is bit-identical.

**Tech Stack:** Python 3.11+, numpy, pytest, `uv` as the sole package manager and runner.

**Spec:** `docs/superpowers/specs/2026-09-19-weiszfeld-coincident-point-singularity/README.md` (committed on this branch at `09963840`). The plan argues from the spec; executors read both.

**Executable witness:** `docs/superpowers/logic_validation_scripts/2026-09-19-weiszfeld-coincident-point-singularity/weiszfeld_singularity.py` — 7 checks today, 9 after Task 4. Run with `uv run python <path>`; exits non-zero on failure; does not import `phenotypic`.

---

## Global Constraints

- **`uv` is the sole package manager and runner.** Never bare `python` or `pip`. Run commands as `uv run <cmd>`.
- **`geometric_median` is a public export** (`src/phenotypic/util/__init__.py:5,27`), as is `robust_color_center` (`:13,30`). `weiszfeld_median` itself is private. The public signature, defaults and return contract do **not** change (spec Non-goals).
- **The `method='cohen'` branch stays unimplemented and keeps raising.** The Cohen et al. routines in the same module (including their own `1e-10` clamps at `_geometric_median.py:69` and `:791`) are unreachable dead code and are **out of scope** — do not touch them (spec Non-goals).
- **`GEOMEDIAN_TOL = 1e-6` / `GEOMEDIAN_MAX_ITER = 200`** (`_color_checker_profile.py:57-58`) are correct and pinned by `test_profile_geomedian_constants_are_tight_enough`. Do not re-tune them (spec Non-goals).
- **Initialization stays `np.mean`** (`_geometric_median.py:1129`). See *Decision 2* below — this is not a free choice.
- **Never author `bio_desc` on schema members** and **never edit files under `docs/superpowers/**/refs`.** Neither applies to any file this plan touches; stated because they bind the whole repo.
- **Lint with explicit paths only:** `uv run ruff check --fix <paths you changed>`. Bare `ruff check --fix` rewrites the whole tree.

---

## Decisions taken before this plan was written

Both were measured, not reasoned. Reproduce any of it with the probe code inlined in the relevant task.

### Decision 1 — the coincidence test is a scale-relative radius, not `d == 0`

The spec's *Compare the distance floor* section says "Test coincidence explicitly (`d == 0`)". **That is not sufficient, and Task 4 amends the spec.**

Measured on the spec's own skewed swatch, planting the extra pixel at a small offset from the centroid rather than exactly on it:

| planted offset | smallest distance at init | error with `d == 0` | error with `d <= 1e-12·max(‖x‖,1)` |
|---|---|---|---|
| `0` | `0.00e+00` | 0.000 | 0.000 |
| `1e-16` | `1.11e-16` | **21.896** | 0.000 |
| `1e-15` | `9.99e-16` | 0.000 | 0.000 |
| `1e-13` | `1.00e-13` | 0.000 | 0.000 |
| `1e-11` | `1.00e-11` | 0.000 | 0.000 |

Errors are 8-bit code values against the converged median. **21.896 is the same failure magnitude as the original defect** (the spec quotes 21.9): a point at `1.11e-16` is not `== 0`, so an exact test lets it take weight `9e15` and capture the solve exactly as the `1e-10` floor did. The hole is not even monotone in the offset, which is what makes it a lurking hazard rather than a clean boundary.

The scale-relative test costs nothing on clean input: with no coincident point, exact and scale-relative agree to **0.0** code values.

### Decision 2 — initialization stays `np.mean`, and this is load-bearing

The witness script's reference starts at `np.median`; the shipped solver starts at `np.mean`. Matching them looks tidy and is wrong. Measured on the clean swatch (no coincident pixel), varying **only** the start:

| init | iterations | error vs converged median | `test_patch_center_converges_past_a_loose_tolerance` (asserts `< 1e-2`) |
|---|---|---|---|
| `np.mean` (shipped) | 12 | `9.4e-05` code values | **PASSES** |
| `np.median` | 1 | `5.4e-01` code values | **FAILS** |

The stopping rule is `‖step‖ < eps`. A *better* start makes it fire **earlier** — the coordinate-wise median is already close enough that step 1 is under `1e-6`, so the loop exits after one iteration half a code value from the answer. Switching the start would break an existing, currently-green test that has nothing to do with this bug. Leave `np.mean` alone.

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

Let `x` be the estimate, `η` the number of points counted as coincident with it, and `R` the rest.

```
T    = Σ_{a∈R} a/‖a−x‖  /  Σ_{a∈R} 1/‖a−x‖
r    = ‖ Σ_{a∈R} (a−x)/‖a−x‖ ‖
γ    = min(1, η / r)
x⁺   = (1 − γ)·T + γ·x
```

Three properties the tasks below depend on:

1. **`η == 0` reduces to the current update exactly.** `x⁺ = T`, and `T` is the classical reweighted centroid over all points. With the floor removed and no distance under it, the arithmetic is bit-identical — which is why Task 2 can pin the clean path's iteration count.
2. **`γ == 1` is an optimality certificate, not a stall.** `f(x) = Σ‖x−a‖` has subdifferential `Σ_{a∈R}(x−a)/‖x−a‖ + η·B` at a point of multiplicity `η`. `0 ∈ ∂f(x)` iff `r ≤ η`, i.e. iff `γ == 1`. So `γ == 1` gives `x⁺ = x`, `change == 0`, and `converged: True` — correctly. This is how acceptance criterion 2 is met.
3. **A small step under damping is a near-optimal step.** `γ` is only near 1 when `r` is near `η`, which is the optimality boundary. So the `‖step‖ < eps` rule cannot stop early *because* of the damping.

---

## File structure

| File | Change | Responsibility |
|---|---|---|
| `src/phenotypic/util/_geometric_median.py` | Modify `:1097-1182` | The solver. Task 1 extracts the return-pair builder; Task 2 replaces the update. |
| `tests/unit/util/test_geometric_median.py` | **Create** | Solver-level guards: coincidence, near-coincidence, degeneracy, optimality, clean-path non-regression. |
| `tests/unit/correction/test_color_checker_geometric_median.py` | Modify `:75-85` | Remove the `xfail` marker (Task 3). |
| `docs/superpowers/specs/2026-09-19-weiszfeld-coincident-point-singularity/README.md` | Modify | Amend the `d == 0` prescription and criterion 2 (Task 4). |
| `docs/.../logic_validation_scripts/.../weiszfeld_singularity.py` | Modify | Add claims 8–9 for the near-coincidence hole (Task 4). |

Nothing else changes. `_robust_color_stats.py`, `_measure_color.py` and `_color_checker_profile.py` are **callers only** — they reach the fix through `robust_color_center` without edits.

---

### Task 1: Extract the Weiszfeld return-pair builder

Pure refactor, no behaviour change. Task 2 adds two more exit points to a function that already duplicates the `(x, info)` construction twice; doing this first keeps that diff about the algorithm.

**Files:**
- Modify: `src/phenotypic/util/_geometric_median.py:1145-1182`
- Test: `tests/unit/util/test_robust_color_stats.py` (existing, must stay green)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `_weiszfeld_result(x: np.ndarray, points: np.ndarray, iterations: int, f_initial: float, converged: bool, verbose: bool) -> Tuple[np.ndarray, Dict]` — private, module-level, used by every `return` in `weiszfeld_median`.

- [ ] **Step 1: Capture the current behaviour as a baseline**

Run: `uv run pytest tests/unit/util/test_robust_color_stats.py tests/unit/correction/test_color_checker_geometric_median.py -q`

Expected, measured on this branch before any change: **`20 passed, 1 xfailed`**. The xfailed one is `test_patch_center_is_not_short_circuited_by_a_coincident_pixel`; it is `strict=True`, so "xfailed" is its pass state until Task 3.

- [ ] **Step 2: Add the private helper**

Insert immediately **above** `def weiszfeld_median(` (currently `_geometric_median.py:1097`):

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
    would be ``0/0``.

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

Replace the converged block (currently `:1145-1163`) — that is, everything from `if change < eps:` through the closing `}` of its `return` — with:

```python
        if change < eps:
            return _weiszfeld_result(
                x, points, iteration + 1, f_initial, True, verbose
            )
```

Replace the exhausted-iterations tail (currently `:1170-1182`, from `objective = compute_geometric_median_objective(x, points)` to the end of the function) with:

```python
    return _weiszfeld_result(x, points, max_iter, f_initial, False, verbose)
```

Leave the periodic progress print (`if verbose and (iteration + 1) % 100 == 0:`) exactly where it is.

- [ ] **Step 4: Verify nothing moved**

Run: `uv run pytest tests/unit/util/test_robust_color_stats.py tests/unit/correction/test_color_checker_geometric_median.py -q`

Expected: identical to Step 1 — `20 passed, 1 xfailed`.

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check --fix src/phenotypic/util/_geometric_median.py && uv run mypy src/phenotypic/util/_geometric_median.py`

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/util/_geometric_median.py
git commit -m "refactor(util): build the Weiszfeld return pair in one place"
```

---

### Task 2: Split coincident points out of the reweighting

The fix itself, plus the solver-level guards that prove it.

**Files:**
- Modify: `src/phenotypic/util/_geometric_median.py:1132-1145` (the loop body)
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
colour-checker-level consequences are pinned separately in
``tests/unit/correction/test_color_checker_geometric_median.py``; both are
needed, because a fix at one level does not imply a fix at the other.

Spec: docs/superpowers/specs/2026-09-19-weiszfeld-coincident-point-singularity/
"""

from __future__ import annotations

import numpy as np

from phenotypic.util._geometric_median import weiszfeld_median


def _reference_median(points: np.ndarray) -> np.ndarray:
    """Vardi-Zhang run to machine precision, independent of the shipped solver.

    Deliberately a second transcription rather than a call into the code under
    test: it is the witness these guards are checked against, so it must not be
    able to share a bug with its subject.
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


def _skewed_cloud(seed: int = 11) -> np.ndarray:
    """A checker-patch core lit unevenly: tight body plus a brightness ramp.

    The same generator the correction-level guards use, so the two test
    modules are talking about the same cloud. The skew is the point: on a
    symmetric cloud the mean and the geometric median coincide and a broken
    solver looks correct.
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

    # The planted pixel really is at distance 0 from the first estimate, so
    # this exercises the degenerate branch rather than depending on luck.
    start = points.mean(axis=0)
    assert np.linalg.norm(points - start, axis=1).min() == 0.0

    got, info = weiszfeld_median(points, eps=1e-6, max_iter=200, verbose=False)
    reference = _reference_median(points)

    assert np.allclose(got, reference, atol=1e-5)
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


def test_converged_means_the_answer_is_a_local_minimum():
    """``converged: True`` must certify optimality, not merely a small step.

    The old rule reported convergence while pinned to a data point, which is
    why this is asserted against the objective itself rather than against the
    solver's own stopping test.
    """
    cloud = _skewed_cloud()
    points = np.vstack([cloud, cloud.mean(axis=0)])

    got, info = weiszfeld_median(points, eps=1e-9, max_iter=5000, verbose=False)
    assert info["converged"] is True

    f_got = _objective(got, points)
    probe = 1e-7
    for axis in range(3):
        for sign in (1.0, -1.0):
            step = np.zeros(3)
            step[axis] = sign * probe
            assert _objective(got + step, points) >= f_got - 1e-12


def test_the_non_degenerate_path_is_arithmetically_unchanged():
    """With no point within the coincidence radius, the rule is the old one.

    The removed floor was a no-op on this cloud (the closest point to any
    iterate is ~3.3e-03, far above 1e-10), so the iteration count is a real
    pin: if it moves, the clean path stopped taking the classical branch.
    """
    cloud = _skewed_cloud()

    got, info = weiszfeld_median(cloud, eps=1e-6, max_iter=200, verbose=False)

    assert np.allclose(got, _reference_median(cloud), atol=1e-5)
    assert info["iterations"] == 12
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/util/test_geometric_median.py -q`

Expected: `test_a_point_exactly_on_the_estimate_does_not_capture_the_solve`, `test_a_point_just_off_the_estimate_does_not_capture_the_solve` and `test_converged_means_the_answer_is_a_local_minimum` **FAIL**. The other two pass already — `test_every_point_identical_returns_that_point` because the floored update happens to return the right answer for that input, and `test_the_non_degenerate_path_is_arithmetically_unchanged` because it is the baseline the fix must preserve. Both are guards against regression, not against the bug.

- [ ] **Step 3: Add the coincidence radius**

Insert immediately **above** `def _weiszfeld_result(` (added in Task 1):

```python
_COINCIDENCE_RTOL = 1e-12
"""Relative radius inside which a data point counts as *on* the estimate.

Coincidence is a radius, not an equality test. A point at ``1.11e-16`` from
the estimate is not equal to it, but ``1/d`` still gives it ~1e15 of weight
and it captures the update exactly as the old ``1e-10`` distance floor did --
measured at 21.9 8-bit code values of error. The radius is scale-relative
because the estimate's magnitude varies by orders of magnitude across callers
(sRGB in ``[0, 1]``; L*a*b* around 100), and it is far below any real data
spacing in either (8-bit pixels are ``1/255`` apart).
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

- [ ] **Step 4: Replace the update rule**

In `weiszfeld_median`, replace the block from `# Compute weights: w_i = 1/||x - a^(i)||_2` through the `x = np.sum(...)` line (currently `:1135-1141`) with:

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
            # certificate, not a stall.
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

- [ ] **Step 5: Update the function docstring**

Replace the `Reference:` and `Iterative reweighting:` lines in `weiszfeld_median`'s docstring with:

```
    Reference: Weiszfeld (1937), with the coincident-point handling of
    Vardi & Zhang (2000).

    Iterative reweighting: x^(k+1) = Σ w_i a^(i) / Σ w_i where
    w_i = 1/||x^(k) - a^(i)||_2. Points lying on x^(k) are excluded from that
    sum and the step is damped toward them by γ = min(1, η/r), which is what
    keeps a data point on the estimate from capturing the iteration. With no
    coincident point (η = 0) this is the classical update unchanged.
```

Add to the module docstring's `Reference:` block, after the Cohen et al. entry:

```
    Vardi, Y., & Zhang, C.-H. (2000). The multivariate L1-median and
    associated data depth. PNAS, 97(4), 1423-1426.
    https://doi.org/10.1073/pnas.97.4.1423
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/util/test_geometric_median.py -q`

Expected: 5 passed.

- [ ] **Step 7: Confirm the callers still agree**

Run: `uv run pytest tests/unit/util/test_robust_color_stats.py -q`

Expected: all pass. In particular `test_robust_center_symmetric_cloud` (exact solution at a point no input occupies) and `test_robust_center_identical_points` (fully degenerate) — acceptance criterion 4.

- [ ] **Step 8: Lint and type-check**

Run: `uv run ruff check --fix src/phenotypic/util/_geometric_median.py tests/unit/util/test_geometric_median.py && uv run mypy src/phenotypic/util/_geometric_median.py`

Expected: clean.

- [ ] **Step 9: Commit**

```bash
git add src/phenotypic/util/_geometric_median.py tests/unit/util/test_geometric_median.py
git commit -m "fix(util): handle coincident points in the Weiszfeld iteration

Split points lying on the running estimate out of the reweighting and take
the Vardi-Zhang damped step, instead of flooring the distance at 1e-10 and
letting such a point take ~1e10 of the weight."
```

---

### Task 3: Flip the colour-checker xfail to green

The marker is `strict=True`, so it becomes a **failure** the moment Task 2 lands. This task is what clears it, and it is the spec's acceptance criterion 1.

**Files:**
- Modify: `tests/unit/correction/test_color_checker_geometric_median.py:1-26` (module docstring), `:75-85` (the marker)

**Interfaces:**
- Consumes: the fixed `weiszfeld_median` from Task 2, reached via `robust_color_center`.
- Produces: nothing downstream.

- [ ] **Step 1: Observe the strict xfail turning into a failure**

Run: `uv run pytest tests/unit/correction/test_color_checker_geometric_median.py -q`

Expected: **1 failed** — `test_patch_center_is_not_short_circuited_by_a_coincident_pixel`, reported as `[XPASS(strict)]`. That is the marker doing its job: the test now passes, and `strict` refuses to let the stale marker survive.

- [ ] **Step 2: Remove the marker**

Delete the entire `@pytest.mark.xfail(...)` decorator above `def test_patch_center_is_not_short_circuited_by_a_coincident_pixel` (currently `:75-85`, from `@pytest.mark.xfail(` through its closing `)`), leaving the function and its docstring intact.

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

In the module docstring, replace the bullet that begins `* ``test_patch_center_is_not_short_circuited_by_a_coincident_pixel`` pins the` — the whole bullet, through `` in ``docs/superpowers/specs/2026-09-19-weiszfeld-coincident-point-singularity/``. `` — with:

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

Run: `uv run pytest tests/unit/correction/test_color_checker_geometric_median.py -v`

Expected: **4 passed**, no xfail, no xpass.

- [ ] **Step 5: Lint**

Run: `uv run ruff check --fix tests/unit/correction/test_color_checker_geometric_median.py`

Expected: clean (this also catches the `pytest` import if Step 2 missed it).

- [ ] **Step 6: Commit**

```bash
git add tests/unit/correction/test_color_checker_geometric_median.py
git commit -m "test(correction): drop the xfail now the coincident-pixel branch is fixed"
```

---

### Task 4: Amend the spec and extend the witness

The spec prescribes `d == 0`; the implementation uses a radius, for the measured reason in *Decision 1*. The spec is the artifact a later reader trusts, so it gets corrected rather than quietly diverged from — and the witness grows the claim that justifies the correction.

**Files:**
- Modify: `docs/superpowers/specs/2026-09-19-weiszfeld-coincident-point-singularity/README.md`
- Modify: `docs/superpowers/logic_validation_scripts/2026-09-19-weiszfeld-coincident-point-singularity/weiszfeld_singularity.py`

**Interfaces:**
- Consumes: the coincidence radius `1e-12 · max(‖x‖, 1)` from Task 2.
- Produces: nothing downstream. Documentation and the executable witness only.

- [ ] **Step 1: Extend the witness with the near-coincidence claim**

In `weiszfeld_singularity.py`, change the `vardi_zhang` signature and coincidence test so the radius is a parameter:

```python
def vardi_zhang(points, eps=1e-12, max_iter=100_000, coincide="radius"):
    """Vardi & Zhang (2000) modified Weiszfeld; handles coincident points.

    Splits the estimate's own coincident mass out of the reweighting and
    takes a damped step, so a point sitting on the estimate cannot capture it.

    ``coincide`` selects how "sitting on the estimate" is decided: ``"radius"``
    uses a scale-relative radius (what the fix ships), ``"exact"`` uses
    ``d == 0`` (what the spec first prescribed). Claim 8 shows why they differ.
    """
    pts = np.asarray(points, dtype=np.float64)
    y = np.median(pts, axis=0)
    for _ in range(max_iter):
        d = np.linalg.norm(pts - y, axis=1)
        if coincide == "exact":
            coincident = d <= 0.0
        else:
            coincident = d <= 1e-12 * max(float(np.linalg.norm(y)), 1.0)
        eta = int(coincident.sum())
```

Leave the rest of the function body unchanged.

- [ ] **Step 2: Add claims 8 and 9**

Append after claim 7, before the `print()` / exit block:

```python
# Claim 8: an exact `d == 0` coincidence test is NOT sufficient. A pixel
# placed 1.11e-16 from the estimate is not equal to it, so an exact test
# lets it through to 1/d with ~9e15 of weight and it captures the solve --
# the same failure, at the same magnitude, as the 1e-10 floor.
nudged = np.vstack([swatch, mean + np.array([1e-16, 0.0, 0.0])])
d_min = np.linalg.norm(nudged - nudged.mean(axis=0), axis=1).min()
truth_nudged = vardi_zhang(nudged, coincide="radius")
exact_answer = vardi_zhang(nudged, coincide="exact")
err_exact = np.linalg.norm(exact_answer - truth_nudged) * 255.0
check(
    "8. an exact d==0 coincidence test still loses the solve",
    d_min > 0.0 and err_exact > 5.0,
    f"d_min = {d_min:.2e} (not zero), exact-test error = {err_exact:.3f} "
    f"code values",
)

# Claim 9: the scale-relative radius costs nothing where nothing coincides.
# If it perturbed the ordinary path it would not be an acceptable fix.
plain_radius = vardi_zhang(swatch, coincide="radius")
plain_exact = vardi_zhang(swatch, coincide="exact")
delta = np.linalg.norm(plain_radius - plain_exact) * 255.0
check(
    "9. the radius changes nothing when no point coincides",
    delta < 1e-9,
    f"difference = {delta:.2e} code values",
)
```

- [ ] **Step 3: Run the witness**

Run: `uv run python docs/superpowers/logic_validation_scripts/2026-09-19-weiszfeld-coincident-point-singularity/weiszfeld_singularity.py`

Expected: 9 `[PASS]` lines, `All claims verified.`, exit code 0. Confirm with `echo $?`.

Note claims 3 and 4 still describe the **shipped-at-the-time** rule via `shipped_weiszfeld`, which is a local transcription of the floored update — they stay green after the fix because the script never imports `phenotypic`. That is the point of the file rule; do not "update" them to the new code.

- [ ] **Step 4: Amend the spec's prescription**

In `README.md`, replace the whole `### Compare the distance floor` section with:

```markdown
### Compare the distance floor

Keep a floor only where it guards division for points that are *near* but not
*on* the estimate. The floor must not be the mechanism that handles
coincidence — that is what failed.

**Test coincidence with a scale-relative radius, not `d == 0`.** An exact
comparison is not sufficient: a point `1.11e-16` from the estimate is not
equal to it, so an exact test lets it through to `1/d` with ~9e15 of weight
and it captures the solve exactly as the `1e-10` floor did — measured at
**21.896** code values of error, the same magnitude as the original defect.
The hole is not monotone in the offset (`1e-15` and `1e-13` are both fine),
which makes it a lurking hazard rather than a boundary one can reason about.

The shipped radius is `1e-12 · max(‖x‖, 1)`. It is scale-relative because
callers differ by orders of magnitude (sRGB in `[0,1]`; L*a*b* around 100),
and it sits far below any real data spacing in either — 8-bit pixels are
`1/255` apart. Where nothing coincides it changes the answer by `0.0` code
values, so the ordinary path is untouched. Claims 8 and 9 of the witness
re-derive both numbers.
```

- [ ] **Step 5: Amend acceptance criterion 2**

Replace criterion 2 with:

```markdown
2. `weiszfeld_median` reports `converged: True` only when the returned point is
   a genuine fixed point, not when it has been pinned to a data point. The
   certificate is Vardi–Zhang's: at a point of multiplicity `η`,
   `0 ∈ ∂f(x)` iff `r ≤ η` iff `γ = 1`, so the zero step taken at `γ = 1` is
   optimality rather than a stall. Pinned directly against the objective by
   `tests/unit/util/test_geometric_median.py::test_converged_means_the_answer_is_a_local_minimum`,
   not against the solver's own stopping test.
```

- [ ] **Step 6: Mark the spec implemented**

Change the `**Status:**` line at the top of `README.md` from:

```markdown
**Status:** **Specified, not implemented.** Deferred to its own PR.
```

to:

```markdown
**Status:** **Implemented** on `fix/geo-median-convergence`. Plan:
[`../../plans/2026-09-19-weiszfeld-coincident-point-singularity/plan.md`](../../plans/2026-09-19-weiszfeld-coincident-point-singularity/plan.md).
```

Also update the `**Executable witness:**` line's count from `7 checks` to `9 checks`.

- [ ] **Step 7: Commit**

```bash
git add docs/superpowers/specs/2026-09-19-weiszfeld-coincident-point-singularity/README.md \
        docs/superpowers/logic_validation_scripts/2026-09-19-weiszfeld-coincident-point-singularity/weiszfeld_singularity.py \
        docs/superpowers/plans/2026-09-19-weiszfeld-coincident-point-singularity/plan.md
git commit -m "docs(spec): coincidence is a radius, not an equality test

An exact d==0 test misses a point 1.11e-16 from the estimate, which still
captures the solve at 21.9 code values of error. Witness claims 8-9 added."
```

---

### Task 5: Regression over the affected surface

The surface is derived from importers, not from directory names. `geometric_median` and `robust_color_center` are public exports; the modules that reach them are `_robust_color_stats.py`, `_measure_color.py` and `_color_checker_profile.py`.

**Files:** none modified unless a failure demands it.

**Interfaces:**
- Consumes: everything from Tasks 1–4.
- Produces: the evidence for acceptance criteria 3, 5 and 6.

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
  -q
```

Expected: all pass, zero xfail, zero xpass.

If `test_measure_color.py` fails on a `ColorLab_*GeoMedian` value, **do not adjust the expected value before checking why.** Acceptance criterion 5 says those columns must be unchanged on inputs that never triggered the singularity. A moved value means either (a) that fixture *did* trigger it, in which case the new value is the correct one and the fixture's old expectation was recording the bug — say so explicitly in the commit message; or (b) the non-degenerate path was not preserved, which is a defect in Task 2. Distinguish the two by checking whether any fixture point lies within `1e-12·max(‖x‖,1)` of an iterate.

- [ ] **Step 2: Run the module doctest**

Run: `uv run pytest --doctest-modules src/phenotypic/util/_geometric_median.py -q`

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

Expected: clean. **Explicit paths only** — bare `ruff check --fix` rewrites the whole tree.

- [ ] **Step 5: Full regression, as a Slurm job**

This is the end of the implementation, so the full suite runs **once**, here. It is ~65 minutes and belongs to the scheduler, not to an interactive node.

Use the **`run-phenotypic-test`** skill for the invocation and the **`slurm-job`** skill for submission. The committed batch script is
`docs/superpowers/plans/2026-08-18-ome-zarr-image-store/run_unit_suite.sbatch`.

Compare against the recorded baseline (11,106 tests, 81 pre-existing failures, all outside `sdk_`/`_cli`/`gui`). **Run any new failure in isolation before attributing it** — a red shard mixes this change with contamination from unrelated files sharing it.

Expected: no new failures relative to the baseline; the failure count for `tests/unit/util/` and `tests/unit/correction/` is zero.

- [ ] **Step 6: Commit any fallout**

Only if Step 1 or 5 required a change. Otherwise skip — Tasks 1–4 already committed the work.

```bash
git add <files>
git commit -m "test: <what moved and why>"
```

---

## Self-review against the spec

| Spec item | Covered by |
|---|---|
| Objective — return the median when a point coincides | Task 2, Steps 3–4 |
| Non-goal: `method='cohen'` still raises | Global Constraints; no task touches it |
| Non-goal: public signature/defaults/return contract unchanged | Task 2 Interfaces; `_weiszfeld_result` is private |
| Non-goal: `GEOMEDIAN_*` not re-tuned | Global Constraints; pinned by an existing test |
| Proposed fix — Vardi & Zhang formulas | Task 2, Step 4 (verbatim `T`, `r`, `γ`, `x⁺`) |
| `η = 0` reduces to the current update | Task 2, Step 1 `test_the_non_degenerate_path_is_arithmetically_unchanged` |
| `r == 0` is the optimality certificate | Task 2, Step 4 early return |
| *Compare the distance floor* (`d == 0`) | **Amended** — Task 4, Steps 4 and 1–3. See *Decision 1*. |
| Acceptance 1 — xfail removed, test passes | Task 3 |
| Acceptance 2 — honest `converged` flag | Task 2 `test_converged_means_the_answer_is_a_local_minimum`; criterion reworded in Task 4 Step 5 |
| Acceptance 3 — all-identical input returns that point | Task 2 `test_every_point_identical_returns_that_point` |
| Acceptance 4 — `test_robust_color_stats.py` unchanged | Task 2 Step 7; Task 5 Step 1 |
| Acceptance 5 — `ColorLab_*GeoMedian` unchanged | Task 5 Step 1, with the adjudication rule |
| Acceptance 6 — witness exits zero | Task 4 Step 3; Task 5 Step 3 |

**Open risk, carried deliberately:** acceptance criterion 5 is checked by the existing `test_measure_color.py` fixtures, which are synthetic. No real-image `ColorLab_*GeoMedian` regression corpus exists in the repo, so "unchanged on inputs that never triggered the singularity" is established by argument (the `η == 0` branch is arithmetically identical) plus the synthetic fixtures, not by a corpus. If a real corpus is wanted before merge, that is a sixth task and should be said now rather than discovered at review.

## References

Vardi, Y., & Zhang, C.-H. (2000). The multivariate L1-median and associated
data depth. *PNAS*, 97(4), 1423–1426. https://doi.org/10.1073/pnas.97.4.1423

Weiszfeld, E. (1937). Sur le point pour lequel la somme des distances de n
points donnés est minimum. *Tôhoku Mathematical Journal*, 43, 355–386.
