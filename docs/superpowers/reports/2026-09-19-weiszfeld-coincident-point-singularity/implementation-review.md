# Implementation review — Weiszfeld coincident-point singularity (Tasks 1–3)

**Subject:** `066553e9` (refactor) + `2560479b` (fix), reviewed together.
**Tree:** worktree `/bigdata/exfab/anguy344/PhenoTypic/.claude/worktrees/fix+geo-median-convergence`, branch `fix/geo-median-convergence`, clean at review time and still clean at write time (this file is the only addition).
**Files under review:** `src/phenotypic/util/_geometric_median.py`, `tests/unit/util/test_geometric_median.py` (new), `tests/unit/correction/test_color_checker_geometric_median.py`.
**Out of scope:** Task 4 (spec amendment, witness claims 8–9) and Task 5 (regression gate, sbatch). Neither was executed; nothing below faults them for that.
**Read:** plan, spec, `plan-review.md` (V-1…V-17, PR-1…PR-17), `cluster-a-report.md`.
**Date:** 2026-09-19

---

## Verdict in one paragraph

**The implementation is correct.** Every term of the shipped update block maps onto Vardi & Zhang as the plan and the published restatement state it; there is no sign error, no wrong denominator, no reachable division by zero, no non-terminating input, and `converged` is never reported wrongly at any of the four exits. Everything runnable is green: **101 passed** across the surface derived from importers, the module doctest `1 passed`, the witness **7/7**, `mypy` clean, `ruff` at its one tolerated pre-existing finding.

**The tests do not prove it.** Of thirteen mutations applied to the shipped fix, **nine leave all five new guards green**, and three of those nine leave the entire 26-test affected surface green — including deleting the damping term outright, which is the half of Vardi & Zhang that the whole change exists to introduce. The suite as committed proves that coincident points are *split out of the reweighting*. It proves nothing about what is done with them afterwards.

---

## Method

Three things were done, in this order.

1. **The shipped update block was checked term by term** against the plan's formula block, the spec's, and the published restatement V-1 cites (arXiv:2405.06965 §2.3) — by re-deriving the mapping from the code, not by re-reading V-1's conclusion. Termination, division by zero and oscillation were analysed separately.
2. **`_old_floored_update` was diffed line by line** against `git show 066553e9:src/phenotypic/util/_geometric_median.py`, since the bit-identity assertion is worthless if that transcription has drifted.
3. **The five new guards were mutation-tested.** A probe (`weiszfeld_mutation_probe.py`) loads the shipped test module, rebinds the two names it imported (`weiszfeld_median`, `_coincidence_atol`) to locally-transcribed mutant solvers, and re-runs each of the five test bodies verbatim against each mutant. A companion pytest plugin (`mutantplug.py`) does the same swap at session scope, so the same mutants run against the whole affected surface. **No source file was edited at any point**; both probes rebind attributes in a running interpreter and write nothing.

`MEASURED` below means the number came back from one of those runs. `DERIVED` means it was reasoned from the code and is labelled as such. Probe outputs are at `/scratch/anguy344/28928169/claude-5188/-bigdata-exfab-anguy344-PhenoTypic/d0ae59d0-2d93-45cd-a05c-7c8232733bc3/scratchpad/{probe1.txt,probe2.txt,candidates.txt}` (`candidates.txt` and `probe_recommendations.txt` are the same run).

**The read-only claim was independently verified rather than taken on trust.** The orchestrator audited both scripts before running them (no `open`/`write`/`Path`/`shutil`/`subprocess`/`os.remove`/`rename`) and took sha256 over all 1,660 `src`/`tests` Python files before and after every run. Identical, `git` clean, across three probe runs and five mutant pytest sessions. That check belongs in the record: a mutation harness that edits the tree and cleans up imperfectly is the standard way this kind of exercise poisons a later result.

---

## Findings

| ID | Severity | Title |
|---|---|---|
| F-1 | **BLOCKER** | The `eta > 0` arithmetic — γ, its direction, and the count itself — survives deletion and corruption against the entire repo |
| F-2 | MAJOR | The `r == 0.0` guard is untested, and deleting it crashes on a three-point input |
| F-3 | MAJOR | `converged: False` on iteration exhaustion is asserted nowhere |
| F-4 | MINOR | Nothing bounds `_COINCIDENCE_RTOL` from above; a 10⁸× radius passes everything |
| F-5 | MINOR | `test_every_point_identical_returns_that_point` kills no mutation, and the branch it names is redundant |
| F-6 | MINOR | "The classical update unchanged" is stated without the `(radius, 1e-10]` caveat, in the docstring and the commit message |
| F-7 | MINOR | The affected surface is green, but acceptance criterion 5 still has no witness |
| F-12 | MINOR | `test_measure_color.py` costs ~56s per test and is the file criterion 5 names |
| F-8 | MINOR | `verbose=True` is uncovered, including the one behaviour Task 1 deliberately changed |
| F-9 | MINOR | `_reference_median` never runs its own `eta > 0` branch on any input in the file |
| F-10 | NIT | `_weiszfeld_result` takes two adjacent positional booleans at four call sites |
| F-11 | NIT | The witness script's `shipped_weiszfeld` no longer transcribes what ships |

---

## Part 1 — Correctness of the shipped code

### 1.1 The update block is a correct Vardi–Zhang step

`src/phenotypic/util/_geometric_median.py:1232-1273`. Mapping re-derived from the code:

| Published (arXiv:2405.06965 §2.3) | Shipped | Line | Verdict |
|---|---|---|---|
| `T̃(y) = Σ_{xᵢ≠y}‖y−xᵢ‖⁻¹xᵢ / Σ_{xᵢ≠y}‖y−xᵢ‖⁻¹` | `reweighted`, over `far` only | `:1249-1252` | identical |
| `‖∇D₁(y)‖ = ‖Σ_{xᵢ≠y}‖y−xᵢ‖⁻¹(y−xᵢ)‖` | `r = ‖Σ (far−x)/far_distances‖` | `:1263-1267` | identical — the sign is inverted and `r` is a norm, so the value is the same |
| `λ = min{1, 1/‖∇D₁‖}` at multiplicity 1 | `gamma = min(1.0, eta / r)` | `:1272` | correct generalisation; reduces to the published form at `eta == 1` |
| `λ = 0` off the data set | `if eta == 0: x = reweighted` | `:1254-1256` | identical |
| `y⁺ = (1−λ)T̃ + λy` | `x = (1.0 - gamma) * reweighted + gamma * x` | `:1273` | identical, operand order included |
| optimality iff `‖∇D₁‖ ≤ η` | `gamma == 1 ⟹ x⁺ == x ⟹ change == 0 < eps` | `:1273`, `:1276-1280` | identical |

**`eta` cannot be miscounted.** `on_estimate` and `far` come from the same boolean mask and its negation (`:1237`, `:1240-1241`), so `eta + len(far) == len(points)` by construction. Vardi & Zhang define `η` as the summed weight of the points at `y`; this data is unweighted, so a count is the right quantity.

**`γ == 1` gives a bit-exact zero step.** `(1.0 - 1.0) * reweighted + 1.0 * x` is `0.0 * finite + x`. `far_distances` is strictly above `_coincidence_atol(x) > 0` by construction, so `reweighted` is always finite and `0.0 * reweighted` is exactly `0.0`. `change` is then exactly `0.0` and the existing `change < eps` branch reports `converged: True` — correct per Property 2, not a stall. **MEASURED** on the cloud `[[0,0],[0,0],[2,0],[-1,0],[-1,0]]`: `γ = 1.0` fires at iteration 0 and the solver returns exactly `[0. 0.]`, `converged=True`, `iterations=1`, objective `4.000000000000` — the exact geometric median.

**The two early returns are both necessary and both correct.**

- `far.size == 0` (`:1243-1247`): every point is inside the radius, so `x` is the median to within `atol`. Returning `converged: True` is right.
- `r == 0.0` (`:1268-1271`): `r = 0 ≤ η`, so `0 ∈ ∂f(x)` and `x` is optimal. `converged: True` is right. The guard is load-bearing, not decorative: `eta` is an `int` and `r` a Python `float`, so `eta / 0.0` raises `ZeroDivisionError` rather than yielding `inf`. **MEASURED** on `[[1,0],[-1,0],[0,0]]`: shipped returns `[0,0] converged iterations=1`; with the guard removed the same call raises `ZeroDivisionError: float division by zero`. V-7's claim is confirmed by execution, not just by reading.

**No division by zero elsewhere.** `1.0 / far_distances` (`:1249`) operates on values strictly above a strictly positive radius. V-8 holds.

**The loop cannot fail to terminate.** `for iteration in range(max_iter)` with two early returns and one exhaustion return; every path exits. One worst case is worth naming (**DERIVED**): an iterate that lands *near* but not inside a data point's radius can be pulled back onto it by the ordinary `1/d` weight, step away under damping, and chatter. That chatter is bounded — the step away is `(r − η)/W`; if it is under `eps` the `change < eps` branch fires immediately, and if it is over `eps` the point rejoins `far` at an ordinary distance and the pull-back weight is ordinary. Either way `max_iter` bounds it and the failure mode is `converged: False`, never a hang. The old floored rule had the same hazard, worse.

**One residual approximation, real and benign (DERIVED).** Vardi & Zhang's `0 ∈ ∂f` argument assumes the `η` points are *exactly* at `x`; here they are within `1e-12·max(‖x‖,1)`. `f` is therefore differentiable at `x` and the certificate holds only to within `η·atol` in objective — at most ~`1e-12·n·‖x‖`. That is below the slack the stopping rule already concedes and orders below anything this codebase publishes. Worth one sentence in the spec at Task 4; not a defect.

### 1.2 `_old_floored_update` is a faithful transcription

`tests/unit/util/test_geometric_median.py:83-99` against `git show 066553e9:src/phenotypic/util/_geometric_median.py`:

| Old solver | Transcription | Same? |
|---|---|---|
| `points = np.asarray(points, dtype=np.float64)` | `pts = np.asarray(points, dtype=np.float64)` | yes |
| `x = np.mean(points, axis=0)` | `x = pts.mean(axis=0)` | yes — same reduction |
| `for iteration in range(max_iter)` | `for _ in range(max_iter)` | yes |
| `x_old = x.copy()` | `x_old = x.copy()` | yes |
| `distances = np.linalg.norm(...)`; `np.maximum(distances, 1e-10)` | the two fused into one expression | yes |
| `weights = 1.0 / distances` | `weights = 1.0 / dist` | yes |
| `np.sum(points * weights[:, np.newaxis], axis=0) / np.sum(weights)` | `np.sum(pts * weights[:, None], axis=0) / np.sum(weights)` | yes — `np.newaxis is None` |
| `change = np.linalg.norm(x - x_old); if change < eps` | `if np.linalg.norm(x - x_old) < eps` | yes |
| returns `x` at both exits | returns `x` at both exits | yes |

The only divergence is that the old solver also built an `info` dict; `_old_floored_update` returns the array alone and the assertion compares arrays only, so nothing rests on it.

**The brief undersold the evidence here.** At plan Step 3a the radius helpers existed but the update rule was still the floored one, so `test_the_non_degenerate_path_is_bit_identical_to_the_old_rule` ran against the *real* old solver and passed (`cluster-a-report.md`: `FF.F.`, `3 failed, 2 passed`). That is a direct check of the transcription against the code it transcribes, at the only moment in the plan when such a check was possible. It is not "never observed failing" — it was observed *confirming*, which is the stronger of the two things that test can do.

### 1.3 The `_weiszfeld_result` refactor is correct at all four call sites

| Line | Exit | `iterations` | `converged` | Correct? |
|---|---|---|---|---|
| `:1245-1247` | every point coincides | `iteration + 1` | `True` | yes |
| `:1269-1271` | `r == 0.0` | `iteration + 1` | `True` | yes |
| `:1278-1280` | `change < eps` | `iteration + 1` | `True` | yes, unchanged from the old code |
| `:1286` | loop exhausted | `max_iter` | `False` | yes, unchanged from the old code |

`converged` is never `True` where the loop has not reached a fixed point or an optimality certificate, and never `False` where it has. `max_iter=0` still returns `iterations: 0, converged: False`. The two new exits report `iteration + 1` for an iteration that performed no *update*, which matches what the old code did on a degenerate cloud (it performed one no-op update and reported `1`). **MEASURED**: `eps=0.0, max_iter∈{1,3,5}` on the skewed cloud returns `iterations` equal to `max_iter` and `converged=False` in every case.

The one deliberate behaviour change — suppressing `Improvement: nan%` and its numpy invalid-value warning when `f_initial == 0.0` — is implemented as planned (`:1174-1175`), recorded in `066553e9`'s message as plan-review PR-11 required, and **MEASURED** working: `verbose=True` on `np.tile([7.,7.,7.], (5,1))` prints `✓ Converged after 1 iterations` / `Final: f(x) = 0.000000` and emits no warning.

### 1.4 Nothing out of scope was touched

`git diff --stat 066553e9^..2560479b` is four files; the solver diff is four hunks, at `:9` (module docstring reference), `:1094+` (the new module-level helpers), `:1132+` (the update block) and `:1201+` (the `geometric_median` docstring). No Cohen-path line appears in any hunk.

- The ten surviving `1e-10` occurrences are at `:73, :327, :512, :517, :568, :579, :627, :795, :952, :1007` — all Cohen-path — plus one inside the new `_COINCIDENCE_RTOL` docstring at `:1112`. The two the brief names, `:73` and `:795`, are **untouched**. (The brief's numbers are the post-change ones; the plan and plan-review call the same two lines `:69` and `:791`, which were correct pre-change. The +4 shift is the Vardi–Zhang citation added to the module docstring — the same shift that moved the tolerated `F841` from `:680` to `:684`, already recorded as `cluster-a-report.md` Finding 4.)
- `GEOMEDIAN_TOL = 1e-6` / `GEOMEDIAN_MAX_ITER = 200` are unchanged at `_color_checker_profile.py:57-58`, still pinned by `test_profile_geomedian_constants_are_tight_enough`.
- `method='cohen'` still raises; the public `geometric_median` signature, defaults and return contract are unchanged.
- Task 3 is complete and clean: decorator gone, `import pytest` gone (no `pytest` reference survives in the file), module-docstring bullet replaced with the plan's text verbatim.
- Task 2 Step 9 was honoured — the fix and the `xfail` removal are one commit, so no point in history is red.

---

## Part 2 — Per-test mutation results

Thirteen mutants (M0 is a control: the shipped rule re-transcribed, and it correctly survives). Columns are the five guards in `tests/unit/util/test_geometric_median.py`.

```
                                                   1=on  2=off 3=ident 4=cert 5=bitid
M0  baseline: shipped rule, re-transcribed         PASS  PASS  PASS    PASS   PASS    (control)
M1  gamma := 0.0  (damping deleted, split kept)    PASS  PASS  PASS    PASS   PASS    >>> SURVIVES <<<
M2  gamma := 1.0  (always full damping, eta>0)     FAIL  FAIL  PASS    FAIL   PASS    killed by 3
M3  drop the `r == 0.0` early return               PASS  PASS  PASS    PASS   PASS    >>> SURVIVES <<<
M4  drop the `far.size == 0` early return          PASS  PASS  PASS    PASS   PASS    >>> SURVIVES <<<
M5  RTOL := 0.0  (coincidence becomes d == 0)      PASS  FAIL  PASS    PASS   PASS    killed by 1
M6  RTOL := 1e-9  (radius 1e3x too large)          PASS  PASS  PASS    PASS   PASS    >>> SURVIVES <<<
M7  RTOL := 1e-4  (radius 1e8x too large)          PASS  PASS  PASS    PASS   PASS    >>> SURVIVES <<<
M8  restore the OLD 1e-10 floored rule (the bug)   FAIL  FAIL  PASS    FAIL   PASS    killed by 3
M9  init := np.median instead of np.mean           PASS  PASS  PASS    PASS   FAIL    killed by 1
M10 `<` instead of `<=` in coincidence test        PASS  PASS  PASS    PASS   PASS    >>> SURVIVES <<<
M11 eta := 1 whenever any point coincides          PASS  PASS  PASS    PASS   PASS    >>> SURVIVES <<<
M12 swap the damping: x = g*T + (1-g)*x            PASS  PASS  PASS    PASS   PASS    >>> SURVIVES <<<
M13 exhaustion reports converged=True              PASS  PASS  PASS    PASS   PASS    >>> SURVIVES <<<
```

**Nine of thirteen survive.** Mutation score against the code this file was written for: 4/13 ≈ 31%.

Four of the survivors were re-run against the **whole affected surface** — the five new guards plus `test_robust_color_stats.py` plus `test_color_checker_geometric_median.py`, 26 tests — by swapping the module attribute at pytest session scope:

| Mutant | Whole-surface result |
|---|---|
| control (no mutation) | 26 passed |
| `gamma := 0.0` | **26 passed** |
| `gamma := 1.0` | 4 failed, 22 passed (adds `test_patch_center_is_not_short_circuited_by_a_coincident_pixel`) |
| drop the `r == 0.0` guard | **26 passed** |
| drop the `far.size == 0` guard | **26 passed** |

### What kills each test — the answer to the brief's question

| Test | Killed by | Notes |
|---|---|---|
| `test_a_point_exactly_on_the_estimate_does_not_capture_the_solve` | M8 (the original bug), M2 (`γ := 1`) | Detects the defect it was written for. |
| `test_a_point_just_off_the_estimate_does_not_capture_the_solve` | M8, M2, **M5 (`RTOL := 0`)** | The only test that proves the radius must not be an equality test. Decision 1's guard, and it works. |
| `test_every_point_identical_returns_that_point` | **none found** | See F-5. |
| `test_converged_carries_the_subgradient_certificate` | M8, M2 | Retargeting to `eps=1e-6/200` did its job: this now fails against the old solver. |
| `test_the_non_degenerate_path_is_bit_identical_to_the_old_rule` | **M9 (`median` init)** — and only M9 | Note it is *not* killed by M8: restoring the bug makes the solver agree with `_old_floored_update` again, by design. It guards the clean path's arithmetic and the initialisation, nothing else. |
| `test_patch_center_is_not_short_circuited_by_a_coincident_pixel` (correction file) | M2 only, of the four run against it | Not killed by `gamma := 0` or by either guard deletion. |

**On the brief's question 2 — are the two "regression guards" guarding anything?** Different answers for the two.

- `test_the_non_degenerate_path_is_bit_identical_to_the_old_rule` **is** guarding something: it is the *only* test in the repo that fails if the initialisation changes (M9), and Decision 2 establishes that the initialisation is load-bearing for a currently-green colour-checker test. It also carries the positive Step 3a evidence in §1.2. Keep it as is.
- `test_every_point_identical_returns_that_point` is not, within the branch it names. See F-5.

---

## Part 3 — Findings in detail

### F-1 — BLOCKER — The `eta > 0` arithmetic survives deletion and corruption against the entire repo

**Where:** `src/phenotypic/util/_geometric_median.py:1257-1273`; `tests/unit/util/test_geometric_median.py` (absent).

**What is wrong.** Three independent mutations to the damped branch leave all five new guards green, and the first leaves all 26 tests on the affected surface green:

| Mutation | Five guards | 26-test surface |
|---|---|---|
| `gamma = 0.0` — delete the damping entirely, keep only the coincident-point split | 5 passed | **26 passed** |
| `eta = 1` whenever any point coincides — discard the multiplicity count | 5 passed | not run |
| `x = gamma * reweighted + (1 - gamma) * x` — the convex combination written backwards | 5 passed | not run |

**Why.** The gamma trace explains it in one line. **MEASURED**, on both planted clouds:

```
exactly-on cloud   loop passes=12   nonzero gammas=[0.0004312759236850096]
just-off  cloud    loop passes=12   nonzero gammas=[0.0004312759236850096]
clean     cloud    loop passes=12   nonzero gammas=[]
all-identical      loop passes=0    nonzero gammas=[]
```

There is exactly **one** non-zero γ in twelve passes, it occurs at iteration 0, and it is `4.3e-4` — because `η = 1` against `r ≈ 2318`. A damping factor of `4.3e-4` applied once out of twelve steps is numerically indistinguishable from no damping at all, which is precisely why `gamma := 0.0` is invisible. `γ == 1` — the branch on which the entire Property 2 argument rests, and the only branch that makes `γ` an *optimality certificate* rather than a rounding detail — **is reached by no input anywhere in the repo.**

Nor is it reached by the obvious candidate. **MEASURED** on `[[0,0],[0,0],[0,0],[5,0],[0,5]]`, whose true median *is* the triple point: the solver stops at `[6.17e-07, 6.17e-07]` after 19 iterations, `converged=True`, because `change < eps` fires six orders of magnitude before the iterate enters the `1e-12` radius. `gamma := 0.0` returns the identical value. The stopping rule outruns the radius on any cloud approached geometrically.

**Why it matters.** The change's stated architecture is "split the coincident points out **and** take the Vardi–Zhang damped step". Tests 1, 2 and 4 prove the split. Nothing proves the step. If a later refactor simplified `:1257-1273` to `x = reweighted` — a natural-looking simplification, since the split alone fixes both planted clouds — the suite would stay green, the spec would still say Vardi–Zhang, and the optimality certificate would silently not exist. The colour-checker case where this bites is not exotic: a patch core is 8-bit quantised, so a modal pixel value with real multiplicity is ordinary, and a median sitting exactly on it is exactly the `γ == 1` case.

**Suggested fix — one test, values already measured.** The construction puts the multiplicity mass exactly at the mean, so `η > 0` at iteration 0 and the stopping rule cannot outrun it. Far points sum to zero (so the mean is the origin) but their unit vectors do not (so `r ≠ 0`, and the `r == 0.0` return is not what answers):

```python
def test_gamma_reaching_one_is_an_optimality_certificate():
    """A data point of multiplicity 2 that IS the median: r = 1 <= eta = 2.

    The multiplicity mass sits exactly at the mean, so eta > 0 at iteration 0.
    On a cloud merely approached geometrically the `change < eps` rule fires
    ~6 orders of magnitude before the 1e-12 radius is entered and this branch
    is never reached -- measured.
    """
    points = np.array([[0.0, 0.0], [0.0, 0.0], [2.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]])
    # 1-D median of {-1,-1,0,0,2} is 0, so [0,0] is the geometric median.
    got, info = weiszfeld_median(points, eps=1e-6, max_iter=200, verbose=False)

    assert np.array_equal(got, np.zeros(2))   # gamma == 1 -> a bit-exact zero step
    assert info["iterations"] == 1
    assert info["converged"] is True
```

**MEASURED** against each mutant on this input:

| Solver | Result |
|---|---|
| shipped | `[0. 0.]`, `converged=True`, `iterations=1`, `f=4.000000000000`; γ trace `[1.0]` |
| `gamma := 0.0` | `[-5.823796e-07, 0.]`, `iterations=22` → **killed** |
| swapped damping | `[-5.823796e-07, 0.]`, `iterations=22` → **killed** |
| OLD floored rule | `[-5.0e-11, 0.]`, `iterations=1` → **killed** by `array_equal` |
| `gamma := 1.0`, `eta := 1`, either guard dropped, any `RTOL`, `<` vs `<=` | `[0. 0.]`, `iterations=1` → not killed by this test (all are covered elsewhere except `eta := 1`) |

**`eta := 1` needs a second cloud, because the primary cannot catch it** — with `η := 1` against `r = 1`, `γ` is still `min(1, 1/1) = 1` and the answer is unchanged. The count only matters when `1 < r ≤ η`. Add:

```python
def test_the_coincident_multiplicity_is_counted_not_merely_detected():
    """eta = 3 against r = 2: gamma is 1 only if the count is right.

    Collapsing the count to a boolean gives gamma = min(1, 1/2) = 0.5 -- a
    half-step off a point that is already the answer.
    """
    points = np.array(
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
         [4.0, 0.0], [-1.0, 0.0], [-1.0, 0.0], [-2.0, 0.0]]
    )
    # 1-D median of {-2,-1,-1,0,0,0,4} is 0.
    got, info = weiszfeld_median(points, eps=1e-6, max_iter=200, verbose=False)

    assert np.array_equal(got, np.zeros(2))
    assert info["iterations"] == 1
    assert info["converged"] is True
```

**MEASURED** (this was DERIVED in the first draft of this report and has since been run; the derivation held in every particular — `η = 3`, `r = 2.0`, `γ = 1.0`):

| Solver | Result on this cloud |
|---|---|
| shipped | `[0. 0.]`, `array_equal(zeros)=True`, `iterations=1`, `converged=True`; γ trace `[1.0]` |
| `eta := 1` | `[-1.51448939e-06, 0.]`, `iterations=33`, γ trace `[0.5]` → **killed** |
| `gamma := 0.0` | `[-1.59888268e-06, 0.]`, `iterations=37`, γ trace `[0.0]` → **killed** |

**Keep both clouds, and know what each is for.** This second one is strictly stronger on the two mutants it was run against — it kills `eta := 1`, which the primary cannot. The primary is still worth having: it is the minimal `γ == 1` case (`η = 2`, `r = 1`), and it is the one measured against all eleven mutants, including the swapped damping and the restored floored rule, which this one was not.

### F-2 — MAJOR — The `r == 0.0` guard is untested, and deleting it crashes on a three-point input

**Where:** `src/phenotypic/util/_geometric_median.py:1268-1271`; `tests/unit/util/test_geometric_median.py` (absent).

**What is wrong.** No input anywhere in the suite reaches this return. **MEASURED**: deleting it leaves the five guards green *and* the whole 26-test surface green (`no_r_guard: 26 passed`), yet on `np.array([[1.,0.],[-1.,0.],[0.,0.]])` the shipped solver returns `[0,0] converged iterations=1` and the guard-less version raises `ZeroDivisionError: float division by zero`.

**Why it matters.** This is the one branch in the change whose absence is a *crash* rather than a wrong number, on a three-point input a user could plausibly supply — a symmetric pair with a point between them. `geometric_median` is a public export. V-7 established the guard is necessary by reading; nothing established it by running, and nothing will notice if it is removed.

**Suggested fix — values measured:**

```python
def test_a_zero_subgradient_at_a_data_point_is_an_early_certificate():
    """r == 0 with eta > 0: x is optimal. Without the guard, eta / r raises."""
    points = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]])
    got, info = weiszfeld_median(points, eps=1e-6, max_iter=200, verbose=False)
    assert np.array_equal(got, np.zeros(2))
    assert info["iterations"] == 1
    assert info["converged"] is True
```

**MEASURED**: shipped `[0. 0.]`, `converged=True`, `iterations=1`, `f=2.000000000000`; `no r==0 guard` → `ZeroDivisionError`. No other mutant is killed by it, which is fine — this one branch is its job.

### F-3 — MAJOR — `converged: False` on iteration exhaustion is asserted nowhere

**Where:** `src/phenotypic/util/_geometric_median.py:1286`; no test anywhere.

**What is wrong.** **MEASURED**: a mutant whose exhaustion return reports `converged=True` passes all five new guards. No test in the repo runs the solver to exhaustion.

**Why it matters.** Spec acceptance criterion 2 is, verbatim, "`weiszfeld_median` reports `converged: True` only when the returned point is a genuine fixed point". The criterion has two halves and only the first is tested. The exhaustion return is also the line the Task 1 refactor rewrote most aggressively — the plan's own Step 4 warned that "no test runs `weiszfeld_median` with `verbose=True` at iteration exhaustion, so a botched Step 3 can pass it", and closed that hole by asking the executor to re-read the function. A re-read is not a regression guard.

**Suggested fix — values measured** (`eps=0.0` guarantees exhaustion; `iterations` equals `max_iter` at 1, 3 and 5):

```python
def test_exhausting_the_iteration_budget_reports_not_converged():
    got, info = weiszfeld_median(_skewed_cloud(), eps=0.0, max_iter=3, verbose=False)
    assert info["iterations"] == 3
    assert info["converged"] is False
    assert got.shape == (3,)
```

### F-4 — MINOR — Nothing bounds `_COINCIDENCE_RTOL` from above

**Where:** `src/phenotypic/util/_geometric_median.py:1101`; `tests/unit/util/test_geometric_median.py` (absent).

**What is wrong.** `test_a_point_just_off_the_estimate_does_not_capture_the_solve` is an excellent *lower* bound: at `RTOL = 0` it fails, which is Decision 1's whole point, and it is the only mutation that test kills. Nothing tests the other side. **MEASURED**: `RTOL := 1e-9` (10³× too large) and `RTOL := 1e-4` (10⁸× too large) both pass all five guards, as does weakening `<=` to `<`. The only resistance to an oversized radius is accidental — `test_the_non_degenerate_path_is_bit_identical_to_the_old_rule` breaks once the radius exceeds the clean cloud's nearest-iterate distance (~`1.24e-03` at `‖x‖ ≈ 0.65`, so around `RTOL ≈ 1.9e-03`), nine orders above the shipped value, and as a side-effect of a test written for something else.

**Why it matters.** `_COINCIDENCE_RTOL` is the one number this change invents. Its docstring makes three quantitative claims and all three are correct — **MEASURED**: `_coincidence_atol([50,10,20]) = 5.477226e-11` against the docstring's "~5.5e-11"; `_coincidence_atol([0,0,0]) = 1e-12` and `_coincidence_atol([0.5,0,0]) = 1e-12`, confirming the absolute floor; `1e-12 / 2.2e-16 ≈ 4500`, confirming "a few thousand ULPs". None of the three is executable. A future edit that widened the radius "to be safe" would swallow real data points into the coincident set and return something closer to the mean, which is the original defect wearing a different hat.

**And one thing worth saying plainly, which no document currently says.** **MEASURED**: `_coincidence_atol` returns exactly `1e-12` for `[0,0,0]`, `[0.5,0,0]` and `[0.35,0.37,0.40]` alike, because `max(‖x‖, 1)` clamps. So **across the whole sRGB working range the radius is a flat `1e-12` absolute and is not scale-relative at all** — the relative half of the rule only begins to bite in L\*a\*b\* (`1.732051e-12` at `[1,1,1]`, `5.477226e-11` at `[50,10,20]`). The constant's docstring leads with the relative framing ("`1e-12 * ||x||` is a few thousand ULPs at `x`") and treats the clamp as a footnote about "a caller working at a coordinate scale below ~1e-11". For `ColorCheckerProfile` — the caller the spec's Blast-radius table rates **highest** exposure — the footnote is the whole story and the headline never applies. Reversing that emphasis costs a sentence and stops the next reader reasoning about a scale-relative radius that, for their caller, does not exist.

**This is time-critical in a way the rest of F-4 is not.** The constant is correct; only its justification is misleading. But Task 4 Step 4 is about to copy that justification into the spec, where it becomes the durable account of why `1e-12` was chosen. Fix the docstring before Task 4 runs, not after, or the wrong reasoning propagates to the document that outlives the code comment.

**Suggested fix — every value measured:**

```python
def test_the_coincidence_radius_is_absolute_below_unit_scale_and_relative_above():
    assert _coincidence_atol(np.zeros(3)) == 1e-12
    assert _coincidence_atol(np.array([0.5, 0.0, 0.0])) == 1e-12       # sRGB
    assert _coincidence_atol(np.array([0.35, 0.37, 0.40])) == 1e-12    # the swatch base
    assert _coincidence_atol(np.array([1.0, 1.0, 1.0])) == pytest.approx(1.732051e-12)
    assert _coincidence_atol(np.array([50.0, 10.0, 20.0])) == pytest.approx(5.477226e-11)
    # Orders below an 8-bit step (1/255 ~= 3.9e-3) at every scale this code works in.
    assert _coincidence_atol(np.array([100.0, 100.0, 100.0])) < 1e-6
```

### F-5 — MINOR — `test_every_point_identical_returns_that_point` kills no mutation, and the branch it names is redundant

**Where:** `tests/unit/util/test_geometric_median.py:167-174`; `src/phenotypic/util/_geometric_median.py:1243-1247`.

**What is wrong.** Two things, and the second explains the first.

1. **MEASURED**: none of M1–M13 turns this test red. It is the one "none found" row in the table.
2. The branch it exists for is **provably redundant** with the `r == 0.0` guard. **MEASURED**: deleting `if far.size == 0` leaves this test green — because control falls through to `weights = 1.0 / far_distances` on an empty array, `reweighted` becomes `nan` from `0/0`, `eta = n > 0` sends control to the `else`, `r = ‖Σ over an empty set‖ = 0.0`, and the `r == 0.0` guard returns the same `x` with the same `converged: True`. The answer and the iteration count are identical.

**What would kill it (DERIVED, not measured — these mutations were not in the probe set):** returning `reweighted` instead of `x` from `:1245-1247`, or reporting `converged=False` there. Both are trivially caught by the assertions as written. So the test is not vacuous — it pins the *contract* of a degenerate cloud, which is spec acceptance criterion 3. It simply does not discriminate the branch its name and docstring point at.

**One caveat on my own evidence.** The probe's mutant solvers wrap the reweighting in `np.errstate(invalid="ignore")`, so the `0/0` that the guard-less path produces was suppressed inside the harness. A faithful guard-less solver would emit `RuntimeWarning: invalid value encountered in divide`, and `pyproject.toml`'s `filterwarnings` has no `error` entry, so nothing in the suite would fail on it either. **I did not measure that**, and the claim above is limited to the returned value and flags, which I did.

> **Correction appended after this report was accepted — the caveat above is now closed, and the report is left otherwise as written.** The warning *was* subsequently measured, twice, both times without `errstate`. (1) A transcription of the real code shape on `np.tile([7.0,7.0,7.0],(5,1))`, inside `warnings.catch_warnings(record=True)` with `simplefilter("always")`: with the guard, `got=[7. 7. 7.] iters=1 conv=True warnings=[]`; with it deleted, identical value, iterations and flag but `warnings=['RuntimeWarning']`. (2) The mutation harness, once the same `errstate` suppression was removed from it — `no_far_guard` flipped from SURVIVES to `1 failed, 31 passed, 3 warnings`. So the guard's entire observable contribution is suppressing that `0/0`, and the spec's acceptance criterion 3 states it as measured. This caveat is retained rather than rewritten because it was true when written, and because the mechanism it names — *a harness supplying a safety net the shipped code does not have* — went on to cause two more false SURVIVES results in the remediation run.

**Suggested fix.** Keep the test — criterion 3 needs it. Amend its docstring to say it pins the contract rather than the branch, add `assert info["iterations"] == 1`, and if the `far.size == 0` guard is to be kept as defence in depth (it should be — it is clearer and avoids a `nan` intermediate), say in the code comment that it is redundant with the `r == 0.0` return and is there for legibility.

### F-6 — MINOR — "The classical update unchanged" is stated without the `(radius, 1e-10]` caveat

**Where:** `src/phenotypic/util/_geometric_median.py:1197-1198` (the `weiszfeld_median` docstring) and `2560479b`'s commit message ("The non-degenerate path is bit-identical to the old rule").

**What is wrong.** The docstring says "With no coincident point (η = 0) this is the classical update unchanged." Plan Property 1, as rewritten in response to plan-review PR-6, is careful that this holds only when no distance falls in `(1e-12·max(‖x‖,1), 1e-10]`. Inside that band the old code floored and the new code does not, so the arithmetic genuinely differs. Both the user-facing docstring and the commit message drop the qualification the plan went to the trouble of adding.

**Why it matters.** `tests/unit/util/test_geometric_median.py:230-232` gets this right — the test's own docstring names the band and states why this cloud never enters it. The shipped docstring is what a reader of the module sees, and it now asserts more than the plan is willing to. It is also the first thing anyone will quote when adjudicating a future `ColorLab_*GeoMedian` movement, where plan Task 5 Step 1's *three*-case table turns on exactly that band existing.

**Suggested fix.** One clause: "…is the classical update unchanged, except that no distance floor is applied — the two differ only for a point in `(1e-12·max(‖x‖,1), 1e-10]`." Fix the same overstatement in the PR description.

### F-7 — MINOR — The affected surface is green, but acceptance criterion 5 still has no witness

**Where:** `tests/unit/measure/test_measure_color.py`; spec acceptance criterion 5.

**The surface was run and it is clean.** The phase's own evidence covered three files; the importer set (`grep -rln "robust_color_center\|geometric_median\|MeasureColor\|ColorCheckerProfile" tests/unit`) is fourteen, of which four are numeric. Those four were run for this review and pass:

| Run | Result |
|---|---|
| `test_geometric_median.py` + `test_color_corrector.py` + `test_color_correction_report.py` + `test_measure_color.py` + `test_measurement_outputs.py` | **101 passed** in 320.13s, exit 0 |
| `pytest --doctest-modules src/phenotypic/util/_geometric_median.py` (plan Task 5 Step 2) | **1 passed** |
| `weiszfeld_singularity.py` (spec criterion 6) | **7/7 PASS**, exit 0 |

So nothing in the `MeasureColor` / `ColorChecker` surface regressed, and this is no longer an open question about whether the change broke something.

**What remains is narrower and is the plan's own finding, not mine.** A green `test_measure_color.py` is **not** evidence for criterion 5. That file makes no numeric assertion on any `ColorLab_*GeoMedian` value — confirmed independently here: the only occurrence of those headers anywhere under `tests/` is a name list at `tests/unit/schema/test_color_schema.py:8`. The plan records this as *Carried risk* and as **ARGUED, NOT TESTED** in its traceability table, and records that the cheap closure (capture the columns from `load_synth_yeast_plate()` before and after) was offered and declined. That decision is not re-litigated here. It does mean the measured `0.014` code-value movement on the `MeasureColor` configuration (`cluster-a-report.md` Finding 1) has no guard anywhere in the repo, and that 101 green tests do not change that by one bit.

**Suggested fix.** None required by this phase. If criterion 5 is ever to become evidence, the declined task is the one to add.

### F-12 — MINOR — `test_measure_color.py` costs ~56s per test, and it is the file criterion 5 names

**Where:** `tests/unit/measure/test_measure_color.py`.

**What is wrong.** **MEASURED** (`--durations=15` over the five-file run):

```
58.69s  test_measure_color.py::test_opt_in_xyz_and_xy
56.16s  test_measure_color.py::test_deltae_scalars_nonnegative
56.01s  test_measure_color.py::test_default_output_is_robust_only
55.80s  test_measure_color.py::test_hex_column_is_string
55.59s  test_measure_color.py::test_hex_column_survives_numeric_aggregation
 4.18s  test_color_correction_report.py::...test_image_report_handles_multiple_rois
```

Five tests account for ~282s of the run's 320s. The next entry is an order of magnitude cheaper.

**Why it matters.** Pre-existing and entirely unrelated to this change — no finding against the diff. It is recorded because of *which* file it is: `test_measure_color.py` is the surface spec criterion 5 names, and F-7's declined closure would add numeric `ColorLab_*GeoMedian` assertions to exactly this file. A 56s-per-test file is one nobody runs casually while iterating, so a guard added there is a guard that will be skipped. Anyone taking up the declined task should budget for making these cheaper, or site the new assertions somewhere else, rather than discovering the cost afterwards.

### F-8 — MINOR — `verbose=True` is uncovered, including the one behaviour Task 1 deliberately changed

**Where:** `src/phenotypic/util/_geometric_median.py:1161-1175`; no test calls either entry point with `verbose=True`.

**What is wrong.** `_weiszfeld_result`'s entire printing half is dead to the suite, including the `f_initial > 0.0` guard at `:1174` that Task 1 added on purpose. The plan flagged this as "no test can see this, so say it in the commit message rather than letting it ride" — which was done — but the consequence is that the behaviour is asserted nowhere and any future edit to `:1174` would undo it silently. `pyproject.toml`'s `filterwarnings` has no `error` entry, so a returned `nan%` plus a `RuntimeWarning` would fail nothing on its own.

**Suggested fix — MEASURED output** (`'Weiszfeld Algorithm (1937)\n…\n✓ Converged after 1 iterations\n  Final: f(x) = 0.000000\n'`, zero warnings):

```python
def test_verbose_on_a_degenerate_cloud_neither_prints_nan_nor_warns(capsys):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        weiszfeld_median(np.tile([7.0, 7.0, 7.0], (5, 1)), eps=1e-6, max_iter=200, verbose=True)
    out = capsys.readouterr().out
    assert "nan" not in out.lower()
    assert "Improvement" not in out
    assert "Converged after 1 iterations" in out
```

### F-9 — MINOR — `_reference_median` never runs its own `eta > 0` branch on any input in the file

**Where:** `tests/unit/util/test_geometric_median.py:28-62`.

**What is wrong.** The helper's docstring is admirably honest — "A wrong radius rule or a wrong gamma would be invisible to it" — and says the compensation is `_clipped_classical` plus the objective. The compensation is weaker than stated, for a second reason the docstring does not give: `_reference_median` starts at `np.median`, and Decision 1 measured that under median initialisation no iterate comes within `1e-12` of the planted pixel on either cloud. So on every input this file supplies, `_reference_median` runs its `eta == 0` branch exclusively and is, in effect, a second copy of `_clipped_classical` with a different clip and a different tolerance. The "two genuinely different algorithms agree" argument at `:136-139` is one algorithm checked twice, plus a monotonicity check on the objective at `:140`.

**Why it matters.** It does not make the current tests wrong — the answer they converge to is right and the objective check is real. It makes the *stated* strength of the cross-check higher than the actual strength, which is the class of thing this gate exists to catch. It also hard-codes `1e-12` at `:42` rather than importing `_coincidence_atol`, which is the right call (an independent witness should not import the constant under test) but means the helper cannot corroborate a radius change even in principle.

**Suggested fix.** Amend the docstring at `:29-37` to say that on the inputs in this file the helper's split never fires, so the cross-check bears on the answer and not on the split. Close the real gap with F-1 and F-2's tests, not with a reference rewrite.

### F-10 — NIT — `_weiszfeld_result` takes two adjacent positional booleans at four call sites

**Where:** `src/phenotypic/util/_geometric_median.py:1245-1247, 1269-1271, 1278-1280, 1286`.

Each call ends `..., f_initial, True, verbose)` or `..., f_initial, False, verbose)`. All four are correct as shipped (checked individually, §1.3). But `converged` and `verbose` are adjacent, same-typed and positional, and swapping them is a silent change that only the untested printing path would reveal (F-8). Ruff's default rule set has no `FBT` (V-15), so nothing will complain. Making the last two keyword-only (`*, converged: bool, verbose: bool`) costs one line and removes the failure mode.

### F-11 — NIT — The witness script's `shipped_weiszfeld` no longer transcribes what ships

**Where:** `docs/superpowers/logic_validation_scripts/2026-09-19-weiszfeld-coincident-point-singularity/weiszfeld_singularity.py:40-41`.

The function is named `shipped_weiszfeld` and its docstring reads "The shipped update rule, transcribed: clamp the distance, reweight." As of `2560479b` that is the *former* rule. Its `vardi_zhang` reference (`:55-84`) uses exact `d <= 0.0` coincidence with `np.median` initialisation — precisely the combination Decision 1 shows cannot see the `1.11e-16` hole.

Both are Task 4's job (Step 1 adds a mean-initialised subject, Step 2 adds claims 8–9), so this is not a defect in the phase under review. It is recorded because the witness currently describes the code inaccurately while sitting in a directory whose entire contract is independence, and Task 4 should run before anyone reads it as current.

---

## One hypothesis checked and refuted

I asked for `--durations=15` because I suspected the new test file was quietly expensive: `_clipped_classical` iterates to a `1e-15` step tolerance under a 100,000-iteration cap on a 3,901-point cloud, `_reference_median` to `1e-13`, and `test_a_point_exactly_on_the_estimate_does_not_capture_the_solve` calls both. **That was wrong.** No test from `tests/unit/util/test_geometric_median.py` reaches the slowest-15 cut-off of 0.85s; both reference loops converge fast enough on this cloud not to register at all. It is recorded rather than dropped because the worry is a reasonable one to have about those two helpers, and the next reader will have it too — the answer is that they are cheap, measured, not argued.

The 5m20s the run did cost is `test_measure_color.py`, and that is F-12.

## What I still did not run

One item. Everything else asked for in the course of this review was run and is folded into the findings above; no *finding* now rests on an unmeasured number. Three analytical arguments remain labelled **DERIVED** — the termination bound and the residual-approximation bound in §1.1, and the two mutations named under F-5 that were not in the probe set. They are reasoning, not measurements, and are marked as such where they appear.

- `M11` (`eta := 1`) and `M12` (swapped damping) against the **26-test surface** — they were run against the five new guards, where both survive, and `M11` was additionally run against F-1's second cloud, where it dies. Whether the wider surface catches either is unknown, though `gamma := 0.0` surviving all 26 makes it unlikely. This does not affect any finding: F-1 already rests on `gamma := 0.0` surviving all 26.

## Note on the mutant count

Two counts are circulating and they describe the same table. `weiszfeld_mutation_probe.py` prints **fourteen** rows, of which `M0` is a control — the shipped rule re-transcribed, which correctly survives and is not a mutation. Of the **thirteen real mutants, nine survive and four are killed** (M2, M5, M8, M9). Counting the control as a survivor gives ten of fourteen. "Eight of fourteen" undercounts by two; the row-by-row table in Part 2 is the authority.

---

## Summary

Everything that could be run is green: 101 tests across the surface derived from importers, the module doctest, the seven-claim witness, `mypy`, and `ruff` at its one tolerated pre-existing `F841`. Nothing is wrong with the shipped algorithm, and I could not construct an input on which it misbehaves.

What is wrong is that the suite can no longer tell you that. Four tests must be added before this is a change anyone can safely refactor around — F-1 (γ), F-2 (`r == 0.0`), F-3 (exhaustion) and F-4 (the radius constant) — and the expected values for all four are measured and recorded above, so writing them is mechanical rather than exploratory. F-1 alone is the blocker: deleting the Vardi–Zhang damping, the mechanism this entire change exists to introduce, leaves all 26 tests on the affected surface green, and `γ == 1` — the branch the plan's Property 2 argument is built on — is executed by no input anywhere in the repo.

The rejection is not a claim that the code is wrong. It is a claim that the phase has not yet produced what it was asked for, which was a fix *and* the evidence that the fix is what is running. The gap is roughly forty lines of test with every number already in hand.

**PHASE REJECTED — 1 blocker**
