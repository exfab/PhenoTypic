# Plan review — Weiszfeld coincident-point singularity

**Subject:** `docs/superpowers/plans/2026-09-19-weiszfeld-coincident-point-singularity/plan.md`
**Spec:** `docs/superpowers/specs/2026-09-19-weiszfeld-coincident-point-singularity/README.md`
**Witness:** `docs/superpowers/logic_validation_scripts/2026-09-19-weiszfeld-coincident-point-singularity/weiszfeld_singularity.py`
**Tree:** worktree `/bigdata/exfab/anguy344/PhenoTypic/.claude/worktrees/fix+geo-median-convergence`, branch `fix/geo-median-convergence` @ `57ce541a`
**Reviewer:** pre-dispatch plan-feasibility gate
**Date:** 2026-09-19

---

## Verdict

**BLOCKED — 3 blockers.**

The algorithm is right, and measurably so. The Vardi–Zhang transcription in Task 2
Step 4 is correct term by term against the published statement; the `γ == 1`
optimality argument matches the literature's own criterion; the `η == 0` path is
**measured bit-identical** to today's arithmetic over 60 iterations; and both
recorded decisions (scale-relative radius over `d == 0`; keep `np.mean`
initialization) are well-evidenced and load-bearing. The fix itself is sound and I
would dispatch it.

What blocks dispatch is three steps whose stated expected outcome is not what will
happen, each with a tempting wrong repair.

1. **Task 4's witness claim 8 cannot fire** — measured. The witness's `vardi_zhang`
   is median-initialised; at `init=median` the exact and radius coincidence tests
   agree to **0.0000** code values, and the plan's 21.896 only appears at
   `init=mean`. Claim 8 asserts `err_exact > 5.0`, so it fails, the witness exits
   non-zero, and Task 4 Step 3 and Task 5 Step 3 both go red. Claim 9 passes
   vacuously for the same reason.
2. **Task 5 Step 5's regression job cannot run** — the committed sbatch hard-codes
   a `WORKTREE` path that is not in `git worktree list` any more, and no task
   authorises editing it.
3. **Every `ruff check` step expects "clean" on a file that is already red** —
   `F841` at `_geometric_median.py:680`, in the dead Cohen `line_search`, which the
   plan's own Global Constraints forbid touching and for which `--fix` has no safe
   fix.

Beyond those, acceptance criterion 5 has no test behind it at all
(`test_measure_color.py` makes no numeric assertion on any `ColorLab_*GeoMedian`
value), two of Task 1's line ranges are wrong in ways that would corrupt the
refactor, and one property the plan relies on (damping cannot cause an early stop)
is **false**, now with a number attached: `η/W = 1.87e-6` against `eps = 1e-6`.

---

## Validated aspects

These are right. No change wanted.

**V-1 — The Vardi–Zhang transcription is correct.** Verified against a published
restatement (arXiv:2405.06965 §2.3, "1-st Power Weiszfeld Algorithm without
Singularity", read via ar5iv):

```
T̃(y)      = Σ_{x_i ≠ y} ‖y − x_i‖⁻¹ x_i  /  Σ_{x_i ≠ y} ‖y − x_i‖⁻¹
∇D₁(y)    = Σ_{x_i ≠ y} ‖y − x_i‖⁻¹ (y − x_i)
λ         = 0                     if y ∉ {x_i}
          = min{1, 1/‖∇D₁(y)‖}    if y ∈ {x_i}
y^(p+1)   = (1 − λ)·T̃(y^(p)) + λ·y^(p)
optimality: x_k is the minimum when ‖∇D₁(x_k)‖ ≤ η_k
```

Mapping to `plan.md:84-89` and to the code in Task 2 Step 4:

| Published | Plan | Verdict |
|---|---|---|
| `T̃(y)` | `T` / `reweighted` | identical |
| `‖∇D₁(y)‖` | `r = ‖Σ_{a∈R}(a−x)/‖a−x‖‖` | identical — published form is `(y − x_i)`, plan's is `(a − x)`; these differ by an overall sign and `r` is a norm, so the value is the same |
| `λ = min{1, 1/‖∇D₁‖}` (multiplicity 1) | `γ = min(1, η/r)` | correct generalisation to multiplicity `η`; reduces to the published form at `η = 1` |
| `λ = 0` off the data set | `if eta == 0: x = reweighted` | identical (`γ = 0` ⇒ `x⁺ = T`) |
| `y⁺ = (1−λ)T̃ + λy` | `x = (1.0 - gamma) * reweighted + gamma * x` | identical |
| `‖∇D₁(x_k)‖ ≤ η_k` | `0 ∈ ∂f(x) iff r ≤ η iff γ = 1` | identical |

No sign error, no wrong denominator.

**V-2 — The subdifferential argument for `γ == 1` is sound.** At a point of
multiplicity `η`, `∂f(x) = Σ_{a∈R}(x−a)/‖x−a‖ + η·B` with `B` the closed unit
ball; `0 ∈ ∂f(x) ⟺ r ≤ η ⟺ η/r ≥ 1 ⟺ γ = 1`. Same criterion as the published
source. `γ = 1` gives `x⁺ = x` exactly, so `change == 0 < eps` and the existing
`if change < eps:` branch reports `converged: True` correctly, with no change
needed to that branch. Property 2 (`plan.md:94`) holds as written.

**V-3 — The `η == 0` path is measured bit-identical.** Running the floored update
and the masked no-floor update side by side from the same start on the clean
swatch, `np.array_equal` holds at every one of 60 iterations. (Mechanically:
`points[~mask]` and `distances[~mask]` under an all-`True`-negated mask produce
C-contiguous copies with identical values in identical order, so the products and
`np.sum` reductions are bit-for-bit the same.) Property 1 is confirmed for this
input — subject to the band caveat in PR-6, which is **measured empty here**: the
closest point to any iterate is `1.24e-03`, and no iterate has a nearest point in
`(1e-12, 1e-10]`.

**V-4 — The `12`-iteration pin is real.** Measured: the shipped solver takes
exactly **12** iterations on the clean swatch, converged, at `9.40e-05` code
values from the converged median. Both numbers in *Decision 2* reproduce. See
PR-10 for why I would still not make it the only assertion.

**V-5 — Decision 1 (radius, not `d == 0`) is correct, and the number reproduces.**
At `init=mean` on the nudged swatch, the exact `d == 0` test answers **21.8963**
code values from the median while the radius test answers **0.0000**. The planted
pixel sits at `1.1102e-16` from the initial estimate against a radius of `1e-12`.
An equality test leaves exactly the hole the `1e-10` floor left.

**V-6 — Decision 2 (keep `np.mean` init) is right and non-obvious.** The
`‖step‖ < eps` rule makes a *better* start stop *earlier*; switching to `np.median`
would break the currently-green `test_patch_center_converges_past_a_loose_tolerance`
(`tests/unit/correction/test_color_checker_geometric_median.py:118-135`, asserts
`< 1e-2` code values). Correctly identified as load-bearing rather than cosmetic.

**V-7 — The `r == 0.0` early return is necessary.** With `eta` an `int` and `r` a
Python `float`, `eta / r` at `r == 0.0` raises `ZeroDivisionError`, not `inf`. The
guard is required. At a *denormal* `r` the quotient is `inf`, `min(1.0, inf) == 1.0`,
and the γ=1 path does the right thing — so there is no gap between the guard and
the damped branch.

**V-8 — Removing the floor introduces no division by zero.** `far_distances` is
`distances[~on_estimate]`, every element of which is strictly greater than
`_coincidence_atol(x) > 0`. The floor is genuinely redundant once the split exists.

**V-9 — The fix produces the right answer on the two planted clouds.** Measured
post-fix behaviour from `np.mean` at `eps=1e-6`: both the exact-plant and the
`1e-16`-nudge clouds converge in **12** iterations to **9.42e-05** code values of
the true median, i.e. `3.7e-07` in `[0,1]` units — comfortably inside the
`atol=1e-5` the new tests assert. The true median has multiplicity 0 on both
(nearest data point `2.13e-03` away), so `assert not np.any(np.all(points == got, axis=1))`
is satisfied honestly rather than by luck.

**V-10 — The new tests' preconditions hold exactly as drafted.** Measured:
`np.linalg.norm(points - points.mean(axis=0), axis=1).min()` is **exactly `0.0`**
for the exact plant (so `== 0.0` passes) and **`1.1102230246251565e-16`** for the
nudge (so `0.0 < smallest < 1e-15` passes). See PR-16 for why I would still
comment the first one.

**V-11 — `test_converged_means_the_answer_is_a_local_minimum` really does detect
the bug on this input.** Measured at the pre-fix answer: `η = 0` (so `f` is
differentiable there), `r = 2317.7`, `max|v_i| = 1341.5`. A `1e-7` axis probe
along the worst axis changes `f` by ≈`1.3e-04`, eight orders above the test's
`1e-12` slack and well above summation noise. The test fails pre-fix for the right
reason. PR-9 is about the docstring's generalisation, not this input.

**V-12 — No existing test in the affected surface regresses.** Traced each case in
`tests/unit/util/test_robust_color_stats.py` by hand:

- `test_robust_center_symmetric_cloud:13` — 4 points, `np.mean` is the origin which
  *is* the exact solution; no point within the radius; `η = 0`; converges at
  iteration 1 exactly as today.
- `test_robust_center_resists_single_outlier:18` — 99 identical points plus one far
  outlier. The iterate approaches the cluster with shrink factor ≈1/99 per
  iteration; at `tol = 1e-4` the stopping rule fires while the iterate is still
  ~1e-5 from the cluster, i.e. before the radius (≈5.5e-11) is entered. Unchanged.
  Even if entered, `η = 99` against `r = 1` gives `γ = 1` and the correct answer.
- `test_robust_center_identical_points:28` / `test_patch_center_handles_a_uniform_patch`
  — `far.size == 0`, returns the point, `converged: True`. Same as today.
- `test_robust_center_single_point_returns_it:24` / `test_robust_center_empty_returns_nan:32`
  — short-circuited inside `robust_color_center` before the solver is reached.

**V-13 — The "out of scope" claims about the Cohen path check out.** The `1e-10`
clamps at `_geometric_median.py:69` and `:791` are real, and `:69`'s only caller is
`:1037`, inside a Cohen routine that `geometric_median` cannot reach
(`method == "cohen"` raises). Leaving them alone is correct — see PR-17 for the
one consequence the plan did not anticipate.

**V-14 — mypy is clean on the file today**, so Task 1 Step 5 / Task 2 Step 8's
mypy half is a real check with a real baseline. (`Success: no issues found in 1
source file`, exit 0.) `Tuple` and `Dict` are already imported at `:100`, so
`_weiszfeld_result`'s annotation needs no new import.

**V-15 — Lint rule surface is as the plan assumes.** `[tool.ruff]`
(`pyproject.toml:271-287`) sets only `line-length` and `extend-exclude`; there is
no `[tool.ruff.lint]` section, so the default rule set (`E4`, `E7`, `E9`, `F`)
applies. No `FBT` (the boolean positional args in `_weiszfeld_result` calls are
fine) and no `E501` (the 79-column limit is formatter-only). The proposed code
blocks lint clean by inspection.

**V-16 — Line references that resolve correctly.** `:1135-1141` (the exact block
Task 2 Step 4 replaces), `:1129`, `:1097`, `_color_checker_profile.py:57-58`,
`util/__init__.py:5,13,27,30`, `test_color_checker_geometric_median.py:31` and
`:75-85` (the decorator does end at 85). Every file named in Task 5 Step 1 exists.
The docstring bullet Task 3 Step 3 rewrites is
`test_color_checker_geometric_median.py:12-19` and the quoted start and end text
match byte for byte.

**V-17 — No concurrency surface.** `weiszfeld_median` is a pure function over a
locally-owned `np.ndarray`; no shared state, no I/O, no module-level mutable
settings, no worker interaction. Called synchronously from `robust_color_center` →
`MeasureColor._robust_lab_row`/`_robust_hsv_row` (single-threaded per-object loop,
`_measure_color.py:123-127`) and from `ColorCheckerProfile._fit_from_rois`
(`_color_checker_profile.py:604`). The change adds no state, no caching, no I/O.
There is no shared-state inventory to take and no synchronization to assess. The
only scheduler interaction is Task 5 Step 5's regression array, whose defect
(PR-2) is a stale path, not a race; that script already avoids the known traps
(`-p no:randomly`, `-o addopts=`, explicit `-m "not slow"`, an in-script comment
forbidding `-n auto`).

---

## Critical issues

### PR-1 — BLOCKER — Witness claim 8 cannot fire; Task 4 Step 3 and Task 5 Step 3 go red

**Where:** plan Task 4 Step 1 and Step 2 (`plan.md:624-691`);
`weiszfeld_singularity.py:55-84` (`vardi_zhang`, initialised at `np.median`,
line 62).

**What's wrong.** Claim 8 is meant to demonstrate that an exact `d == 0`
coincidence test still loses the solve, by running the witness's `vardi_zhang`
over `nudged` (swatch plus a pixel at `mean + [1e-16, 0, 0]`) in both
`coincide="exact"` and `coincide="radius"` modes and asserting
`err_exact > 5.0` code values.

But `vardi_zhang` starts at `y = np.median(pts, axis=0)`, not at the mean, and the
planted pixel sits at the *mean*. For the two modes to disagree, some iterate must
land strictly inside `(0, 1e-12·max(‖y‖,1)]` of the planted pixel — and the
iterates converge toward the geometric median, which is `0.0859` in `[0,1]` units
(21.9 code values) from the mean. No iterate ever gets close. Both modes take the
`eta == 0` branch on every iteration and return bit-identical answers.

**Measured, not argued:**

```
Q4 nudged init=median  exact-mode err=0.0000 codes, radius-mode err=0.0000 codes
Q4 nudged init=mean    exact-mode err=21.8963 codes, radius-mode err=0.0000 codes
```

So claim 8 as drafted computes `err_exact = 0.0000`, `0.0000 > 5.0` is **False**,
the claim FAILS, the witness `sys.exit(1)`s, and Task 4 Step 3's "Expected: 9
`[PASS]` lines, `All claims verified.`, exit code 0" does not happen. Because the
witness is also acceptance criterion 6 and is re-run at Task 5 Step 3, this fails
twice.

**Claim 9 is vacuous for the same reason.** Measured:

```
Q4b witness claim9 delta (clean, median init): 0.0
```

It asserts the radius "changes nothing when no point coincides" by differencing
the two modes on the clean swatch — but neither mode ever flags a coincidence
there, so the difference is trivially `0.0` for *any* radius value, including an
absurd one. This is the "success message identical to the no-op message" trap: the
check cannot fail, so it certifies nothing.

The measurement in *Decision 1* (`plan.md:39-45`) is not wrong. It is a measurement
of a **mean**-initialised solver, which is what ships (`_geometric_median.py:1129`).
The plan carries that number into a median-initialised witness where it does not
apply.

**Why it matters.** Two gates depend on a green witness, and the likely repairs are
all wrong: loosening `> 5.0`, deleting the claim, or "fixing" `vardi_zhang` to
start at the mean — which would silently change claims 1, 4, 5, 6 and 7, every one
of which consumes `vardi_zhang`'s output as `truth`.

**Suggested fix.** Make claim 8's subject **mean**-initialised, because that is the
property being claimed, and leave `vardi_zhang` as the median-initialised ground
truth claims 1–7 already depend on. Concretely: put the `coincide=` parameter on a
mean-initialised transcription of the *shipped* loop (extending `shipped_weiszfeld`
with a coincidence mode is the natural home, since it is already the "what ships"
transcription), not on `vardi_zhang`. Claim 8 then compares that run against
`vardi_zhang(nudged)` as truth and will read `21.896`. Claim 9 must likewise compare
a mean-initialised radius run against a mean-initialised exact run on the *clean*
swatch, where both genuinely reach small distances, so "costs nothing" is actually
exercised; verify it is still `0.0` before Task 4 Step 4 writes that number into
the spec.

---

### PR-2 — BLOCKER — Task 5 Step 5's regression job points at a worktree that no longer exists

**Where:** plan Task 5 Step 5 (`plan.md:820-829`);
`docs/superpowers/plans/2026-08-18-ome-zarr-image-store/run_unit_suite.sbatch:14`.

**What's wrong.** The committed batch script hard-codes

```bash
WORKTREE=/bigdata/exfab/anguy344/PhenoTypic/.worktrees/worktree-ome-zarr-image-store
```

then `cd "$WORKTREE"` under `set -euo pipefail`. That path is **not** in
`git worktree list` — the live worktrees are `ci-split-full-suite`,
`fix+geo-median-convergence`, `nested-gpu-staging`, `import-laziness`,
`lazy-baseline`, `mcp-server`, `migrate-maresca`, `private-gui` and
`make-streamlit-app`. `worktree-ome-zarr-image-store` is gone. All 24 array tasks
die at the `cd`.

The plan says "The committed batch script is `…/run_unit_suite.sbatch`" and gives
no instruction to edit it, while the *File structure* table (`plan.md:101-109`)
says "Nothing else changes". A fresh agent therefore has no authorisation to change
the one line that must change, and the step is not executable.

Secondary: `--output=/bigdata/exfab/anguy344/slurm_logs/%A_%a.log` and
`PYTEST_GROUP_COUNT=24` are likewise fixed, while the baseline the plan asks to
compare against ("11,106 tests, 81 pre-existing failures") was captured with a
48-shard recipe. The *total* is comparable across shard counts but per-shard
contamination differs, so the mismatch should be stated rather than discovered.

**Why it matters.** This step produces the evidence for acceptance criteria 3, 5
and 6. A job array that dies at line 30 produces none. Worse, if someone repairs
the path to a *different* live worktree, the gate measures a tree that is not this
change — the "union across two trees" failure the repo's own Slurm guidance warns
about.

**Suggested fix.** Either (a) add a sub-step that copies the sbatch into this
change's own plan folder with `WORKTREE` set to
`/bigdata/exfab/anguy344/PhenoTypic/.claude/worktrees/fix+geo-median-convergence`,
and add that file to the File-structure table; or (b) better, follow the repo rule
and run the gate in a worktree detached at the SHA under test, so the tree cannot
move underneath the array. Either way, name the shard count the baseline was taken
at and the `--output` directory to use.

---

### PR-17 — BLOCKER — Every `ruff check` step expects "clean" on a file that is already red, and the only fix is forbidden

**Where:** plan Task 1 Step 5 (`plan.md:205-209`), Task 2 Step 8
(`plan.md:522-526`), Task 5 Step 4 (`plan.md:806-818`) — all three say
"Expected: clean".

**What's wrong.** Measured on this branch, before any change:

```
$ uv run ruff check src/phenotypic/util/_geometric_median.py
F841 Local variable `n` is assigned to but never used
   --> src/phenotypic/util/_geometric_median.py:680:5
Found 1 error.
No fixes available (1 hidden fix can be enabled with the `--unsafe-fixes` option).
ruff exit=1
```

Line 680 is inside `line_search` (`:649`, "Algorithm 4: LineSearch … Reference:
Page 7, Algorithm 4") — the Cohen et al. dead path, which the plan's own Global
Constraints (`plan.md:21`) say to "not touch". `--fix` has no safe fix; the hidden
fix needs `--unsafe-fixes`.

So a fresh agent reaches the last step of Task 1, sees a non-zero exit against an
"Expected: clean", and has no sanctioned move. The three repairs it might reach for
are all bad: deleting the line (violates the Global Constraint and edits a
transcription of a published algorithm), adding `# noqa: F841` to the Cohen path
(same), or re-running with `--unsafe-fixes` (same, and broader). The fourth — just
proceeding — is correct but the plan gives no permission for it.

**Why it matters.** This is a hard stop at the end of the very first task, and the
attractive repairs all edit code the plan declared out of scope and the repo treats
as a faithful transcription. It also recurs twice more, so an agent that guesses
wrong at Task 1 will have baked the guess in before anyone sees it.

**Suggested fix.** Make the expectation a *diff against a baseline* rather than
"clean", and capture that baseline where the plan already captures the pytest one.
In Task 1 Step 1, add: run `uv run ruff check src/phenotypic/util/_geometric_median.py`
and record the single pre-existing `F841` at `:680`. In Steps 5/8 and Task 5 Step 4,
change "Expected: clean" to "Expected: exactly the one pre-existing `F841` at
`:680` (`line_search`, the dead Cohen path) and nothing else. **Do not fix it** —
Global Constraints forbid touching the Cohen routines, and `--fix` has no safe fix
for it. A second finding, or a finding anywhere outside `line_search`, is caused by
this change." Note also that `ruff check --fix` will exit non-zero on every one of
those steps, so any `&&` chaining with `mypy` (as written in Task 1 Step 5 and Task
2 Step 8) will silently skip the mypy half — split them onto separate lines.

---

## Concerns

### PR-3 — MAJOR — Acceptance criterion 5 has no test behind it; the self-review table says it does

**Where:** plan self-review row "Acceptance 5 — `ColorLab_*GeoMedian` unchanged |
Task 5 Step 1, with the adjudication rule" (`plan.md:858`); the "Open risk"
paragraph (`plan.md:861`); `tests/unit/measure/test_measure_color.py`.

**What's wrong.** `test_measure_color.py` is 64 lines and contains **no numeric
assertion on any `ColorLab_*GeoMedian` value**. It checks that the robust headers
are present (`:20-21`), that the hex column is a string starting with `#`
(`:31-33`), that the ΔE scalars are non-negative (`:36-40`), that XYZ/xy are
opt-in (`:43-46`), a serialization round-trip (`:49-53`), and that a numeric
reduction skips the hex column (`:56-64`). Nothing pins a geometric-median value.

So Task 5 Step 1's carefully written adjudication rule — "If `test_measure_color.py`
fails on a `ColorLab_*GeoMedian` value, do not adjust the expected value before
checking why" — describes a failure that cannot occur. And the Open-risk
paragraph's claim that criterion 5 is "established by argument … plus the synthetic
fixtures" overstates: the fixtures contribute nothing to it. There is no corpus
elsewhere either — `tests/unit/measure/_golden/` holds only orientation-zone and
symmetric-zone artefacts, and the only other files matching `ColorLab_` are GUI/CLI
tests using the strings as synthetic column names.

**Why it matters.** Criterion 5 is what protects *published measurement output*
from an unintended shift — the reason the spec says this fix needs its own PR
(`README.md:71`). Merging while the traceability table asserts it is covered turns
an open risk into a believed-closed one.

**Suggested fix.** Cheap to close, and the plan is already shaped for it. In Task 1
Step 1 (before any source change) capture the `ColorLab_*GeoMedian` and
`ColorHSV_*RobustMean` columns from
`MeasureColor().measure(OtsuDetector().apply(load_synth_yeast_plate()))` and commit
them as a small characterization fixture; in Task 5 Step 1, assert against it. On
the same plate, measure whether any colony's `lab_px` mean lands within
`1e-12·max(‖x‖,1)` of one of its pixels — that settles whether the plate is an
`η == 0` case (values must be byte-identical) or an `η > 0` case (values are
*expected* to move, the fixture records the bug's old output, and the commit
message must say so). If the team would rather carry the risk, then the self-review
row and the Open-risk paragraph both need rewording to say criterion 5 is argued,
not tested.

---

### PR-4 — MAJOR — Task 1 Step 3's two line ranges are both wrong, and the first contradicts its own prose

**Where:** plan Task 1 Step 3 (`plan.md:180-197`); also the Task 1 header
"Modify: `…_geometric_median.py:1145-1182`" (`plan.md:118`).

**What's wrong.** Ground truth in the file:

- `1145-1159` — `if change < eps:` through the closing `}` of its `return`.
- `1161-1163` — the periodic progress print.
- `1165-1168` — `objective = compute_geometric_median_objective(x, points)` plus
  the two `⚠ Maximum iterations reached` / `Final: f(x)` prints.
- `1170-1176` — the exhausted-iterations `return x, { … }`; `1176` is the last line
  of `weiszfeld_median`. `1179-1181` is the `# ===== Main Interface Function =====`
  banner for the *next* function.

The plan says:

1. "Replace the converged block (currently `:1145-1163`) — that is, everything from
   `if change < eps:` through the closing `}` of its `return`". The prose is right
   (1145–1159); the range is not — it swallows 1161–1163, the progress print that
   the *same step* then says to "leave exactly where it is". The step contradicts
   itself.
2. "Replace the exhausted-iterations tail (currently `:1170-1182`, from
   `objective = compute_geometric_median_objective(x, points)` to the end of the
   function)". The `objective = …` line is **1165**, not 1170, and the function
   ends at **1176**, not 1182. Trusting the numbers leaves 1165–1168 in place — so
   the function prints `⚠ Maximum iterations reached` and `Final: f(x) = …` and
   then `_weiszfeld_result` prints both again — and deletes the section banner at
   1179–1181.

The surviving `objective` local would be caught by ruff `F841`, so the
duplicate-print half is *partly* self-correcting — except that after PR-17 the
agent is now primed to treat an `F841` in this file as pre-existing noise.

**Why it matters.** Task 1 is explicitly framed as "Pure refactor, no behaviour
change", and its verification (Step 4) is only "identical to Step 1 — 20 passed,
1 xfailed". Every corruption above passes that check, because nothing in the suite
runs `weiszfeld_median` with `verbose=True` at `max_iter` exhaustion. It would
ship.

**Suggested fix.** Correct both ranges to `1145-1159` and `1165-1176`, keep the
prose as the authority (it is right in both cases), and fix the task header's
`:1145-1182` to `:1145-1176`.

---

### PR-5 — MAJOR — The amended spec still mandates a floor the implementation removes

**Where:** plan Task 4 Step 4's replacement text (`plan.md:697-718`), first
paragraph; against Task 2 Step 4 (`plan.md:441-483`).

**What's wrong.** The replacement section opens by preserving the spec's existing
sentence verbatim:

> Keep a floor only where it guards division for points that are *near* but not
> *on* the estimate. The floor must not be the mechanism that handles
> coincidence — that is what failed.

But Task 2 Step 4 deletes `distances = np.maximum(distances, 1e-10)` and never
reintroduces any floor. There is none left in the shipped code — correctly so,
because `far_distances` is by construction strictly greater than the coincidence
radius, so `1/d` is always finite (V-8). The amended spec would prescribe a
component the implementation does not contain, in the *same section* whose purpose
is to record what the implementation actually does.

**Why it matters.** The spec is what a later reader trusts. A maintainer reading
"keep a floor" and finding none will either re-add one — reintroducing a weaker
form of the original defect — or lose confidence in the section.

**Suggested fix.** Rewrite that paragraph to say the floor is removed outright, and
why that is safe: the coincidence split guarantees every reweighted distance
exceeds the radius, so no division needs guarding. That is a stronger and more
accurate statement than the one being preserved.

---

### PR-7 — MAJOR — Property 3 is false, with a number: `η/W = 1.87e-6` against `eps = 1e-6`

**Where:** plan property 3 (`plan.md:95`); Task 4 Step 5's replacement for
acceptance criterion 2 (`plan.md:724-732`).

**What's wrong.** The plan argues: "A small step under damping is a near-optimal
step. `γ` is only near 1 when `r` is near `η`, which is the optimality boundary. So
the `‖step‖ < eps` rule cannot stop early *because* of the damping."

The step is not small only when `γ` is near 1. Writing `W = Σ_{a∈R} 1/‖a−x‖`, the
classical step satisfies `‖T − x‖ = r/W`, so the damped step is

```
‖x⁺ − x‖ = (1 − γ)·r/W = (r − η)/W      when γ = η/r < 1
```

against `r/W` undamped. Damping subtracts `η/W` from the step length regardless of
how far `γ` is from 1. Whenever `eps < r/W < eps + η/W`, the undamped rule keeps
iterating and the damped rule stops.

**Measured on the spec's own swatch:**

```
Q8 W = sum(1/d) at truth: 534440.38     eps*W at eps=1e-6: 0.53444
```

So `η/W = 1/534440 = 1.871e-6`, which is **1.9× `eps`**. The window
`r ∈ (eps·W, eps·W + η)` is `r ∈ (0.534, 1.534)`. Optimality at `η = 1` requires
`r ≤ 1`, so the upper part of that window — `r ∈ (1, 1.534)` — is genuinely
non-optimal territory in which the damped solver stops and reports
`converged: True` at `γ = 1/r < 1`, i.e. with **no certificate**. This is not a
corner case bounded away by a large margin; it is a window the same size as the
quantity being tested.

What actually holds at the stop is the weaker `r ≤ η + eps·W`. (Note this is the
*same* class of slack the classical `η = 0` rule has always carried — the stopping
rule is on step size, not on the subgradient — so this is not a regression. It is a
mis-stated justification.)

Task 4 Step 5's replacement text says:

> The certificate is Vardi–Zhang's: at a point of multiplicity `η`, `0 ∈ ∂f(x)`
> iff `r ≤ η` iff `γ = 1`, so the zero step taken at `γ = 1` is optimality rather
> than a stall.

That sentence is true about the `γ = 1` case, but it is offered as the justification
for criterion 2, and criterion 2 is a statement about *every* `converged: True`. As
written the criterion is not met.

**Why it matters.** The team is about to write this into the spec as the reason
criterion 2 is satisfied. A reader who later finds a `converged: True` at `γ < 1`
will reasonably conclude the fix is broken, when in fact the criterion was
mis-stated — and the numbers say that reader will exist.

**Suggested fix.** Replace property 3's argument with the algebra above and quote
`W ≈ 5.3e5`, `η/W ≈ 1.9e-6`, `eps·W ≈ 0.53` so the slack is on the record. State
the honest guarantee: the stopping rule certifies `r ≤ η + eps·W`, which is exact
optimality when `γ = 1` and an `eps`-scaled subgradient slack otherwise — the same
slack the classical rule has always carried at `η = 0`. Then reword criterion 2 to
claim what is true: `converged: True` is no longer compatible with being *pinned
to* a data point, because a coincident point no longer dominates the update, and
where the answer *is* a data point, `γ = 1` is a genuine optimality certificate.
If a tighter guarantee is wanted, the cheap version is PR-9's direct assertion.

---

## Suggestions for improvement

### PR-6 — MINOR — "bit-identical" is true only outside a band; Task 5's adjudication rule does not cover it

**Where:** plan Architecture (`plan.md:7`, "reduces **exactly** to today's
arithmetic, so the non-degenerate path is bit-identical"); property 1
(`plan.md:93`); Task 5 Step 1's adjudication rule (`plan.md:792`).

The old code applied `np.maximum(distances, 1e-10)`; the new code applies no floor.
They agree bit for bit only when no distance falls in the band
`(1e-12·max(‖x‖,1), 1e-10]` — two orders of magnitude wide in sRGB. A point there
is *not* coincident under the new rule (so `η` stays 0 and the classical branch is
taken) but *was* floored under the old one.

**Measured, this band is empty on the reviewed fixture**: the minimum over all
iterates of the nearest-point distance on the clean swatch is `1.24e-03`, and no
iterate has a nearest point in `(1e-12, 1e-10]`. So V-3's bit-identity holds and
nothing is broken today. That is why this is MINOR rather than MAJOR.

It is still a hole in Task 5 Step 1, which tells the executor to distinguish "the
fixture triggered the singularity" from "the non-degenerate path was not preserved"
by "checking whether any fixture point lies within `1e-12·max(‖x‖,1)` of an
iterate". A fixture whose closest point sits at, say, `5e-11` fails that check and
would be adjudicated as "a defect in Task 2" when it is in fact a legitimate,
intended change (the old floor was distorting the weight; the new code is not). The
rule becomes operative the moment PR-3's characterization pin is added.

**Suggested direction.** State the band explicitly in property 1 and soften the
Architecture sentence to "bit-identical whenever no distance falls in
`(radius, 1e-10]`", noting the measurement above. In Task 5 Step 1, make the
adjudication test the *band*, not the radius: case (b) requires no fixture point
within `1e-10` of any iterate; a point inside the radius is case (a); a point in
the band is a third, benign case that should be named.

### PR-8 — MINOR — The new test file's reference shares an algorithm (and any algorithm bug) with its subject

**Where:** plan Task 2 Step 1, `_reference_median` (`plan.md:261-291`).

Its docstring says: "Deliberately a second transcription rather than a call into
the code under test: it is the witness these guards are checked against, so it must
not be able to share a bug with its subject." But it transcribes the *same
algorithm with the same coincidence radius* (`1e-12 * max(‖y‖, 1)`), the same
split, and the same `γ`, differing only in initialization (`np.median`) and
tolerance (`1e-13`). It is independent of the *implementation*, not of the
*algorithm* — a wrong radius rule or a wrong `γ` is invisible to it.

The existing `test_color_checker_geometric_median.py:40-56` is a genuinely
different algorithm (clipped classical Weiszfeld from `np.median`) and is the
stronger witness for the non-degenerate cases.

**Suggested direction.** Either soften the docstring to say what it actually is (an
independent transcription, not an independent algorithm), or strengthen the guards:
cross-check the two non-degenerate assertions against a clipped classical run, and
for the coincident case assert the *objective* is no worse than the
clipped-classical answer's and no worse than the mean's. The objective is the one
thing no transcription can fake.

### PR-9 — MINOR — The 6-axis probe establishes optimality only where `f` is differentiable

**Where:** plan Task 2 Step 1, `test_converged_means_the_answer_is_a_local_minimum`
(`plan.md:364-383`).

For `f(x) = Σ‖x − a‖` away from every data point, `f` is smooth and a nonzero
gradient always has a negative component along some `±e_i`, so the axis probe is
sufficient. **This holds for this input** — measured `η = 0` at both the pre-fix
answer (`r = 2317.7`, `max|v_i| = 1341.5`) and the post-fix answer (nearest data
point `2.13e-03` away), so the test is valid and detects the bug by a factor of
~1e8 (V-11).

The docstring's *general* claim is not valid. At a data point of multiplicity `η`,
`∂f(x) = v + η·B` with `v = Σ_{a∈R}(x−a)/‖x−a‖`, and `f'(x; ±e_i) = ±v_i + η`. All
six probes are non-decreasing iff `|v_i| ≤ η` for every `i`, which can hold while
`r = ‖v‖ > η`. In 3-D with `η = 1` and `v = (0.6, 0.6, 0.6)`, every axis probe
passes and `r = 1.039 > 1` — not optimal. So an axis probe is not in general a test
of "converged means local minimum" for this objective, and the degenerate case is
exactly the one this PR is about.

**Suggested direction.** Keep the probe, and add the certificate directly: at the
returned point recompute `η` (points within `_coincidence_atol`) and
`r = ‖Σ_{far}(x − a)/‖x − a‖‖`, then assert `r <= eta + eps * W`. That is the
property criterion 2 actually wants, it is three lines, and it is the assertion
PR-7 asks for. Probing along `−v/‖v‖` instead of the axes is a cheaper partial
improvement.

### PR-10 — MINOR — `info["iterations"] == 12` is a fragile proxy for the property it stands in for

**Where:** plan Task 2 Step 1,
`test_the_non_degenerate_path_is_arithmetically_unchanged` (`plan.md:398`).

The number is real (V-4: measured 12) and the docstring justifies it well. But the
literal `12` is a function of numpy's summation blocking and the linear convergence
rate against a fixed `eps`; a numpy or BLAS change can move it without anything
being wrong, and the test would then be "fixed" by editing the number — at which
point it guards nothing.

**Suggested direction.** Assert the thing itself. Transcribe the *old* floored
update locally in the test file (six lines) and assert
`np.array_equal(got, old_transcription_result)` — that is exactly the check I ran
for V-3, it drifts with numpy rather than against it, and it fails loudly if the
clean path ever stops taking the classical branch. Keep `iterations == 12` as a
secondary assertion if wanted, but not as the only one.

### PR-11 — MINOR — Task 1 is labelled "no behaviour change" but changes one

**Where:** plan Task 1 preamble (`plan.md:115`) and Step 2's helper (`plan.md:169`).

The helper guards the improvement print with `if converged and f_initial > 0.0:`.
Today a fully degenerate cloud (`f_initial == 0.0`) with `verbose=True` prints
`Improvement: nan%` and emits a numpy `invalid value encountered` RuntimeWarning;
after the refactor it prints nothing. That is an improvement, but it is a behaviour
change inside a task that declares it makes none, and Step 4's verification cannot
see it — nothing in the suite runs the solver with `verbose=True`.

**Suggested direction.** Say so in the task preamble and the commit message ("also
stops the `0/0` improvement print on a degenerate cloud"), or move the guard to
Task 2 where behaviour changes are expected.

### PR-12 — MINOR — Stale claims the amendment does not reach

**Where:** spec `README.md:67`; plan `plan.md:11`.

- `README.md:67` (Blast radius, `MeasureColor` row) says "Float Lab pixels rarely
  land within `1e-10` of the estimate, but nothing prevents it." After the fix the
  operative threshold is `1e-12·max(‖x‖,1)`. Task 4 amends the *Compare the
  distance floor* section and criterion 2, but not this row, so the spec keeps a
  number the code no longer uses.
- `plan.md:11` says the spec was "committed on this branch at `09963840`". Both the
  spec and the witness were added by **`f18c59b6`** (`git log` on each file, and
  the orchestrator's own run confirms it). `09963840` is the preceding
  `fix(color): converge the colour-checker geometric median` commit.

**Suggested direction.** Add the Blast-radius row to Task 4's edit list; correct the
SHA.

### PR-13 — MINOR — The radius docstring gives the wrong reason for a right choice

**Where:** plan Task 2 Step 3, `_COINCIDENCE_RTOL` docstring (`plan.md:412-422`).

The docstring justifies `1e-12` by "it is far below any real data spacing in either
(8-bit pixels are `1/255` apart)". True, but that is what makes the radius
*harmless*, not what makes it *correct*. What makes it correct is that it is
floating-point-relative: `1e-12·‖x‖` is roughly 4,500 ULPs at `x`, so the test asks
"is this point indistinguishable from the iterate at double precision, to within a
few thousand ULPs" — exactly the question `1/d` blows up on. That framing also
answers the scale objection head-on: it is right to scale with `‖x‖` rather than
with the data's spread, because the failure mode is a floating-point one, not a
statistical one.

On the specific case raised — a tight L\*a\*b\* cloud within 0.01 of (50, 10, 20) —
**measured**: radius `5.477e-11`, nearest-point distance `1.198e-03`. Seven orders
of separation. The radius is fine there, and it would only bite on a cloud whose
genuine point spacing is below ~1e-11 relative to its distance from the origin,
which no colorimetric caller produces.

One real consequence worth documenting: the `max(‖x‖, 1)` floor means the solver
has a **1e-12 absolute** resolution floor for data near the origin. A caller working
at scale ~1e-11 would see every point swallowed into `η`, `far.size == 0`, and the
mean returned. The answer is still within the cloud's diameter of the truth, so it
is harmless — but `geometric_median` is a public export and this is now part of its
contract.

**Suggested direction.** Lead the docstring with the floating-point reason, keep the
data-spacing sentence as the "and it is harmless here" corollary, and note the
`1e-12` absolute floor in `geometric_median`'s public docstring.

### PR-14 — MINOR — Task 2's commit leaves the tree red until Task 3

**Where:** plan Task 2 Step 9 (`plan.md:528-537`), Task 3 preamble (`plan.md:543`).

The `xfail(strict=True)` marker turns into a failure the moment Task 2 lands, so the
commit `fix(util): handle coincident points in the Weiszfeld iteration` is a commit
at which `tests/unit/correction/test_color_checker_geometric_median.py` fails. The
plan is aware and Task 3 fixes it one commit later, but it leaves a bisect-hostile
point in history and a CI-red intermediate.

**Suggested direction.** Either fold Task 3's marker removal into Task 2's commit
(three lines, same change), or note explicitly in Task 2 Step 9 that the commit is
expected red and must not be pushed alone.

### PR-15 — NIT — Three copies of the swatch generator, one claiming to be the others

**Where:** `weiszfeld_singularity.py:92`, `test_color_checker_geometric_median.py:59`,
plan Task 2 Step 1 `_skewed_cloud` (`plan.md:294-307`).

`_skewed_cloud`'s docstring says "The same generator the correction-level guards
use, so the two test modules are talking about the same cloud" — but it is a third
hand copy and nothing enforces the claim. The witness's copy must stay separate (the
file rule forbids importing `phenotypic`), but the two *test* modules could share
one helper.

**Suggested direction.** Put the generator in a shared test helper the two test
modules import, or drop the "same generator" sentence so the docstring does not
assert an invariant nothing checks.

### PR-16 — NIT — Small step-level inaccuracies

- Task 3 Step 1 (`plan.md:556`) expects "**1 failed**"; the file has four tests, so
  it is `1 failed, 3 passed`.
- Task 4 Step 6 (`plan.md:749`) says to update "the `**Executable witness:**`
  line's count from `7 checks` to `9 checks`", but `7 checks` is on the continuation
  line (`README.md:8`), not on the `**Executable witness:**` line (`README.md:7`).
- Task 2's header (`plan.md:225`) says "Modify `:1132-1145` (the loop body)"; the
  block actually replaced is `:1135-1141`, which Step 4 states correctly.
- `test_a_point_exactly_on_the_estimate_does_not_capture_the_solve` (`plan.md:323`)
  asserts `np.linalg.norm(points - start, axis=1).min() == 0.0` — an exact float
  equality on a quantity derived from two different reductions (`np.mean` over
  3,900 rows vs over 3,901). **Measured exactly `0.0` today** (V-10), so it passes;
  and if it ever flipped to ~1.1e-16 the test would still exercise the intended
  branch, so only the precondition assert would break. Worth a one-line comment
  saying that, so a future reader does not treat the equality as the point of the
  test.

---

## Concurrency analysis

Not applicable, and deliberately so — see **V-17**. `weiszfeld_median` is a pure
function over a locally-owned array with no shared state, no I/O, no module-level
mutable settings and no worker interaction; the change adds none of those. There is
no shared-state inventory to take and no synchronization to assess. The only
scheduler interaction in the plan is Task 5 Step 5's regression array, whose defect
(PR-2) is a stale hard-coded path rather than a race.

---

## Verification results

### Measured (probe run by the orchestrator; no `phenotypic` import)

```
Q1 exact-plant min dist at init      : 0.0
Q1 nudged  min dist at init          : 1.1102230246251565e-16
Q1 nudged atol at init               : 1e-12
Q2 shipped clean-swatch iterations   : 12 converged True
Q2 shipped clean err vs truth (codes): 9.395432443264606e-05
Q2 min over iterates of min-dist     : 0.0012424719803765367  any in (atol,1e-10]? False
Q3 eta==0 bit-identical over 60 iters: True
Q4 nudged init=median exact-mode err=0.0000 codes, radius-mode err=0.0000 codes
Q4 nudged init=mean   exact-mode err=21.8963 codes, radius-mode err=0.0000 codes
Q4b witness claim9 delta (clean, median init): 0.0
Q5 pre-fix answer: eta= 0  r= 2317.700420537894  max|v_i|= 1341.520364248574
Q5 shipped iters/converged on planted: 1 True
Q6 planted: min dist at truth=2.134e-03  fixed(mean,eps=1e-6): iters=12 conv=True err=9.42e-05 codes
Q6 nudged: min dist at truth=2.134e-03  fixed(mean,eps=1e-6): iters=12 conv=True err=9.42e-05 codes
Q7 lab atol= 5.477230868355318e-11  min pairwise-ish nn dist= 0.0011982547806355616
Q8 W=sum(1/d) at truth: 534440.3804962522  eps*W at eps=1e-6: 0.5344403804962522
```

```
$ uv run mypy src/phenotypic/util/_geometric_median.py
Success: no issues found in 1 source file          (exit 0)

$ uv run ruff check src/phenotypic/util/_geometric_median.py
F841 Local variable `n` is assigned to but never used
   --> src/phenotypic/util/_geometric_median.py:680:5
Found 1 error.                                      (exit 1)

$ git log --oneline -1 -- .../specs/.../README.md
f18c59b6 test(color): guard the patch geometric median; drop the dead wrapper
```

Which finding each measurement settles: Q4 → **PR-1** (blocker, confirmed);
Q4b → PR-1's claim-9 half; Q8 → **PR-7** (the `η/W = 1.87e-6` vs `eps = 1e-6`
number); ruff → **PR-17** (blocker); Q2 (band) → PR-6 downgraded to MINOR;
Q3 → V-3; Q2 (iterations) → V-4; Q1 → V-10; Q5 → V-11 and PR-9's scope;
Q6 → V-9; Q7 → PR-13; mypy → V-14; `git log` → PR-12.

### Read and checked against the tree (@ `57ce541a`)

- `src/phenotypic/util/_geometric_median.py` — full `weiszfeld_median`
  (`:1097-1176`), the defect block (`:1135-1141`), both return sites
  (`:1145-1159`, `:1170-1176`), `geometric_median` (`:1184-1263`), the module
  docstring `Reference:` block (`:5-9`), `compute_geometric_median_objective`
  (`:34`), the Cohen clamps (`:69`, `:791`) and `:69`'s only caller (`:1037`),
  `line_search` (`:649-…`, home of the pre-existing `F841` at `:680`).
- `tests/unit/correction/test_color_checker_geometric_median.py` — all 150 lines;
  `pytest` import at `:31`, decorator `:75-85`, docstring bullet `:12-19` with the
  quoted start and end text matching byte for byte.
- `tests/unit/util/test_robust_color_stats.py` — all 120 lines; every
  `robust_color_center` case traced through the proposed update by hand (V-12).
- `tests/unit/measure/test_measure_color.py` — all 64 lines. **No numeric
  `ColorLab_*GeoMedian` assertion exists** (PR-3).
- `src/phenotypic/util/_robust_color_stats.py`,
  `src/phenotypic/measure/_measure_color.py:123-170`,
  `src/phenotypic/correction/_color_correction/_color_checker_profile.py:57-58,596-615`.
- `weiszfeld_singularity.py` — all 188 lines; `vardi_zhang` starts at `np.median`
  (`:62`) and tests coincidence with `d <= 0.0` (`:65`) — the basis of PR-1.
- `pyproject.toml` — `testpaths` includes `tests/unit` (`:219`); `[tool.ruff]` has
  no `lint.select`, so defaults apply (`:271-287`).
- `run_unit_suite.sbatch` — all 75 lines; `WORKTREE` at `:14` (PR-2).
- `git worktree list` — `worktree-ome-zarr-image-store` absent (PR-2).
- Searched `docs/` for other `Weiszfeld` references: only the two 2026-06-10
  robust-Lab documents and this change's own artefacts. Nothing under `docs/source`
  (user docs) goes stale. No golden fixture anywhere carries a
  `ColorLab_*GeoMedian` value.

### Verified against the literature

- arXiv:2405.06965 §2.3 (via ar5iv) — the Vardi–Zhang de-singularity transform
  `T̃(y)`, `∇D₁(y)`, `λ = min{1, 1/‖∇D₁(y)‖}`, the update
  `y⁽ᵖ⁺¹⁾ = (1−λ)T̃(y⁽ᵖ⁾) + λy⁽ᵖ⁾`, and the optimality criterion
  `‖∇D₁(x_k)‖ ≤ η_k`. This is what V-1 and V-2 are checked against.

### Not verified

- **The PNAS original (doi:10.1073/pnas.97.4.1423) is unread.** It returns HTTP
  403; the author's Rutgers copy 301-redirects to a landing page; the
  Fritz–Filzmoser–Croux comparison paper downloaded as a PDF but neither
  `pdftotext` nor `poppler-utils` is installed on this node, so its text could not
  be extracted. V-1 and V-2 therefore rest on a secondary restatement. It is
  unambiguous and matches the plan term for term, so I am confident in them, but
  the primary citation has not been read against the code.
- **Whether `load_synth_yeast_plate()` contains a colony whose pixel mean coincides
  with a pixel** — this decides how PR-3's suggested characterization pin should be
  adjudicated. Not measured; it requires importing `phenotypic`.
- **The full suite.** No regression run was attempted (PR-2 makes the committed
  recipe unrunnable as written).

---

## Questions for clarification

1. **PR-17 (blocker):** confirm the preferred shape — baseline-diff wording in the
   plan's own steps (my recommendation, one sentence per step), or a repo-level
   `per-file-ignores` entry? The latter is a `pyproject.toml` change and therefore
   scope creep for this PR.
2. **PR-3:** is criterion 5 meant to be *tested* before merge, or carried as the
   stated open risk? The plan's Open-risk paragraph offers "a sixth task … should
   be said now rather than discovered at review" — this is that moment, and the
   cheap version (characterize the synth plate in Task 1, assert in Task 5) is well
   under a task's worth of work.
3. **PR-2:** should the gate run in this worktree, or in a fresh worktree detached
   at the SHA under test? The latter is what the repo's Slurm guidance prescribes
   and costs one extra command.
4. **PR-7:** given `η/W = 1.87e-6` against `eps = 1e-6`, is reporting
   `converged: True` at `γ < 1` acceptable (it matches the pre-existing `η = 0`
   behaviour and is a property of the step-size stopping rule, not of this fix), or
   should criterion 2 be tightened to the subgradient test in PR-9?

---

**BLOCKED — 3 blockers.**
