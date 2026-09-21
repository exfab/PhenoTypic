# Cluster A execution report — Weiszfeld coincident-point singularity (Tasks 1–3)

**Plan:** [`../../plans/2026-09-19-weiszfeld-coincident-point-singularity/plan.md`](../../plans/2026-09-19-weiszfeld-coincident-point-singularity/plan.md)
**Spec:** [`../../specs/2026-09-19-weiszfeld-coincident-point-singularity/README.md`](../../specs/2026-09-19-weiszfeld-coincident-point-singularity/README.md)
**Plan review:** [`plan-review.md`](plan-review.md)
**Tree:** worktree `/bigdata/exfab/anguy344/PhenoTypic/.claude/worktrees/fix+geo-median-convergence`, branch `fix/geo-median-convergence`
**Scope executed:** Tasks 1, 2 and 3. Tasks 4 (spec + witness) and 5 (regression + sbatch) were **not** executed.
**Date:** 2026-09-19

---

## Outcome

Implemented as planned. Two commits:

| Commit | Contents |
|---|---|
| `066553e9` | Task 1 — `_weiszfeld_result` extracted; both returns routed through it. |
| `2560479b` | Tasks 2 + 3 — the Vardi–Zhang coincident-point split, the new solver-level guard file, and the removal of the now-stale strict `xfail`. |

Tasks 2 and 3 are one commit deliberately (plan Task 2 Step 9 / plan-review PR-14), so no
point in history has a red suite.

### Files changed

| File | Change |
|---|---|
| `src/phenotypic/util/_geometric_median.py` | `_weiszfeld_result` (Task 1); `_COINCIDENCE_RTOL` + `_coincidence_atol` (Task 2 Step 3); the update block replaced with the Vardi–Zhang split (Step 4); three docstring edits (Step 5). |
| `tests/unit/util/test_geometric_median.py` | **New.** Five solver-level guards. |
| `tests/unit/correction/test_color_checker_geometric_median.py` | `@pytest.mark.xfail` decorator removed, `import pytest` removed, module-docstring bullet rewritten. |

Nothing outside these three files was touched. The Cohen et al. routines, including their
own `1e-10` clamps, are untouched and `method='cohen'` still raises.

---

## Verbatim result of the last run of each verification step

**Task 1 Step 1 (pre-change baseline)** and **Task 1 Step 4 (post-refactor)** — identical output:

```
tests/unit/util/test_robust_color_stats.py .................
tests/unit/correction/test_color_checker_geometric_median.py x...

======================== 20 passed, 1 xfailed in 1.52s =========================
```

**Task 1 Step 5** — ruff exit 1, mypy exit 0:

```
F841 Local variable `n` is assigned to but never used
   --> src/phenotypic/util/_geometric_median.py:680:5
Found 1 error.
No fixes available (1 hidden fix can be enabled with the `--unsafe-fixes` option).

Success: no issues found in 1 source file
```

**Task 2 Step 2 (tests before the fix)** — exit 2:

```
collected 0 items / 1 error
E   ImportError: cannot import name '_coincidence_atol' from 'phenotypic.util._geometric_median'
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
=============================== 1 error in 1.37s ===============================
```

**Task 2 Step 3a (added — see Finding 2)** — the new tests against the *old* floored solver:

```
tests/unit/util/test_geometric_median.py FF.F.

>       assert r <= eta + eps * w_total
E       assert 2317.700420537894 <= (0 + (1e-06 * 4359522.249222453))

========================= 3 failed, 2 passed in 0.69s ==========================
```

**Task 2 Step 6** — `5 passed in 0.16s`.
**Task 2 Step 7** — `17 passed in 0.89s`.
**Task 2 Step 8** — ruff exit 1, mypy exit 0:

```
F841 Local variable `n` is assigned to but never used
   --> src/phenotypic/util/_geometric_median.py:684:5
Found 1 error.

Success: no issues found in 1 source file
```

**Task 3 Step 1 (the strict marker flipping on its own)** — exit 1:

```
________ test_patch_center_is_not_short_circuited_by_a_coincident_pixel ________
[XPASS(strict)] Known, specified defect: weiszfeld_median floors the distance at 1e-10, so a pixel on the running estimate takes ~99.9995% of the weight and captures the solve. See docs/superpowers/specs/2026-09-19-weiszfeld-coincident-point-singularity/. strict=True, so this flips to a failure the moment the fix lands and the marker cannot be left behind.
========================= 1 failed, 3 passed in 0.16s ==========================
```

**Task 3 Step 4** — exit 0:

```
tests/unit/correction/test_color_checker_geometric_median.py::test_patch_center_is_not_short_circuited_by_a_coincident_pixel PASSED
tests/unit/correction/test_color_checker_geometric_median.py::test_patch_center_converges_past_a_loose_tolerance PASSED
tests/unit/correction/test_color_checker_geometric_median.py::test_profile_geomedian_constants_are_tight_enough PASSED
tests/unit/correction/test_color_checker_geometric_median.py::test_patch_center_handles_a_uniform_patch PASSED

============================== 4 passed in 0.17s ===============================
```

**Task 3 Step 5** — `All checks passed!` (exit 0).

### Acceptance criterion 1 is discharged by observation, not argument

Task 3 Step 1's `[XPASS(strict)]` is the only direct evidence that the marker did what it
was written to do: fail before the fix, and force its own cleanup after. It quotes the
marker's own reason text, which ends *"this flips to a failure the moment the fix lands and
the marker cannot be left behind."* Once the decorator is deleted that state cannot be
reproduced, so it was captured deliberately rather than asserted later.

---

## Findings for Task 4 and Task 5

### Finding 1 — the escape is driven by `eps`, not `max_iter`, and both callers were exposed

**This supersedes the account in the spec, and an earlier draft of this report.**

The spec attributes the defect's survival to "absent a coincident point the shipped rule is
already correct". True, but incomplete. Measured on the planted swatch, at every
configuration a caller actually uses (errors in 8-bit code values against the converged
median):

| Caller | `eps` | `max_iter` | OLD iters | OLD err | NEW iters | NEW err |
|---|---|---|---|---|---|---|
| `ColorCheckerProfile` | `1e-06` | 200 | 1 | **21.896** | 12 | 0.000 |
| `MeasureColor` | `1e-04` | 50 | 1 | **21.896** | 7 | 0.014 |
| `geometric_median` default | `1e-06` | 1000 | 1 | **21.896** | 12 | 0.000 |
| (tight probe) | `1e-09` | 5000 | 21 | 0.000 | 19 | 0.000 |

**The mechanism.** The old floor binds only while the estimate is *exactly* on the pixel.
After the first step it no longer is, the floor stops binding, and the classical iteration
would recover — but the post-capture step is smaller than `eps`, so `change < eps` fires and
the loop reports `converged: True` on the second iteration. Only tightening `eps` to `1e-9`
makes that step register as movement and lets the iteration continue.

**`max_iter` is not the lever.** At `eps=1e-6` with a *thousand* iterations the old rule
still fails after one; the loop never uses its budget. This matters for how Task 4 writes it
up: the wrong version of this story ("200 iterations stopped it before it could recover")
implies raising `max_iter` would have masked the bug, and it would not have. The defect is
that the stopping rule mistakes post-capture stillness for convergence.

**Both callers were exposed, equally.** The spec's Blast-radius table rates `MeasureColor`
as "lower" exposure because float Lab pixels rarely land on the estimate. That is a claim
about the *likelihood* of a coincident pixel and it is fine — but it reads as a claim about
*severity if one occurs*, and severity is identical: 21.896 code values, the same as the
colour checker. Task 4 should separate the two.

**A real, small change to published output.** `MeasureColor`'s post-fix error is `0.014`
rather than `0.000`, because 50 iterations at `eps=1e-4` is a loose budget. Correct, but it
means `ColorLab_*GeoMedian` moves on any input that triggered the singularity. That is
acceptance criterion 5, which the plan carries as argued-not-tested (see *Carried risk* in
the plan). This measurement is the first concrete instance of the thing that criterion is
about.

> **Correction to an earlier draft of this report.** It claimed `geometric_median` "never
> forwards `max_iter`", so the `MeasureColor` path ran at the 1000 default. That is wrong.
> `geometric_median` takes `**kwargs` and forwards them
> (`_geometric_median.py`, the `method == "weiszfeld"` branch), and `robust_color_center`
> passes `max_iter=max_iter` explicitly (`_robust_color_stats.py:42-43`, defaults
> `max_iter=50, tol=1e-4`). Every caller runs at its own configuration; none runs at the
> library default. The underlying question — whether the `MeasureColor` configuration had
> ever been checked — was sound, and the table above is the answer.

### Finding 2 — step-order defect in the plan (Task 2, Steps 1–4)

**A defect in the plan, not a step added by choice.** Task 2 Step 2 states its expected
outcome as "the first two tests and `test_converged_carries_the_subgradient_certificate`
FAIL", then parenthetically concedes the import fails first. Those are not the same signal.
On the plan's literal step order the three bug-reproduction tests **never execute in a
failing state at all** — they go from `collected 0 items / 1 error` straight to green after
Step 4. A test observed only passing is not yet known to be a test.

The repair is free, because Steps 3 and 4 are separable: Step 3 adds only the constant and
the radius helper, Step 4 replaces the update rule. Running the suite between them puts the
new tests against the old floored solver. That run — recorded above as **Step 3a** — is what
caught Finding 3, which nothing else would have.

**Action:** the plan should carry Step 3a explicitly, so a later execution does not lose it.

### Finding 3 — the certificate guard, as the plan wrote it, passed against the broken solver

The plan specified `eps=1e-9, max_iter=5000` for
`test_converged_carries_the_subgradient_certificate`. At that configuration the old rule
converges correctly (Finding 1), so the test the plan describes as "not vacuous" was
**green pre-fix**. It guarded nothing.

Retargeted to the shipped constants `eps=1e-6, max_iter=200`, it fails pre-fix by ~531×
(`r = 2317.70` against a bound of `4.3595`) and passes post-fix with ~4.3× margin
(`r = 0.1240` against `0.5344`). The committed test carries a comment explaining why the
tighter configuration hides the bug, so it cannot be "simplified" back.

**Every measured number the plan attaches to this test is also wrong**, for one reason: the
numbers were taken at mixed configurations. The plan's `r=1.260e-04, W=5.344e+05,
bound=5.344e-04` are `eps=1e-9` values; its "six orders of magnitude" (docstring) and "seven
orders" (comment) pair `r=2317.70` — an `eps=1e-6` quantity — with an `eps=1e-9` bound. The
correct pre-fix comparison is `r=2317.70` against **its own** bound of `4.3595`: `W` at the
captured estimate is `4.36e6`, not `5.34e5`, because the planted pixel sits just outside the
coincidence radius of that estimate and `1/d` is enormous there.

**This error class — a number carried across a configuration boundary — is the recurring
hazard on this change.** The plan review caught one instance as PR-1; it then recurred twice
more during execution, once in the proposed correction to the first recurrence. Every
numeric claim in this spec and plan should be read as under-specified unless it names its
`eps` and `max_iter`. Task 4 is about to write several of these numbers into the spec.

### Finding 4 — `ruff`'s pre-existing `F841` has moved to `:684`; Task 5 Step 4 will mis-read it

The plan's Global Constraints and Task 5 Step 4 both pin the known-acceptable finding at
`_geometric_median.py:680`. Task 2 Step 5 appends the Vardi & Zhang citation to the
**module** docstring at the top of the file, shifting everything below it by +4. Post-change
the finding is at **`:684`** — visible in the Step 5 vs Step 8 outputs above.

Same finding, same dead Cohen `line_search`, still out of scope. But Task 5's executor is
told to accept `:680` and nothing else, and will see a line number that does not match.
**Action:** correct the line number in the plan before Task 5 runs.

---

## Out of scope, but worth someone's attention

- **The `F841` at `:684` itself.** Correctly left alone (it is in the unreachable Cohen
  path, and `--fix` has no safe fix). But it is live lint debt on a file that two tasks now
  touch, and every lint step spends attention re-establishing that it is expected.
- **Acceptance criterion 5 remains argued, not tested.** Finding 1 sharpens why that
  matters: the argument rests on the `η == 0` path being arithmetically identical, which is
  measured on one cloud, and we now have a measured `0.014` code-value movement on the
  `MeasureColor` configuration when the singularity *is* triggered. The cheap closure the
  plan offered and declined — capture `ColorLab_*GeoMedian` from `load_synth_yeast_plate()`
  before and after — is still the right task if anyone wants evidence rather than argument.
- **The plan's Task 1 Step 3 warning was justified.** Both original line ranges were wrong
  in ways that pass Step 4's verification; matching on the quoted prose is what made the
  refactor safe. Keep that instruction if this plan is re-run.

## What was not executed

Task 4 (amend the spec, extend the witness to 9 claims) and Task 5 (regression gate,
`run_unit_suite.sbatch`) were out of this cluster's scope and were not started. No spec
file, witness file, or batch script was created or modified.

Nothing in the plan proved un-executable.
