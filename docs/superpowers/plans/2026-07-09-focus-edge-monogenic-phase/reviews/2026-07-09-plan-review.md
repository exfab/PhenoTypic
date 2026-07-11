# Plan review — `FocusEdgeMonogenicPhase`

**Date:** 2026-07-09
**Reviewer:** independent `plan-reviewer` agent (Opus, adversarial brief), run against `plan.md` @ `198702a1f`
**Triage:** orchestrator. **Every finding below was independently reproduced before being recorded.** Nothing here is accepted on the reviewer's assertion alone, and one of its numbers is corrected.

Nothing has been fixed yet. This document is the issue list.

---

## Summary

| # | Severity | Where | Issue |
|---|---|---|---|
| 1 | **BLOCKER** | `plan.md` Task 2 Step 3 | Removing `List` from the imports breaks `mypy` and `ruff` at Step 7 |
| 2 | **BLOCKER** | `plan.md` Task 2 Step 3 | "delete roughly line 379 to the end of the class" deletes `_compute_angular_spread`, which Step 3 says to keep |
| 3 | **MAJOR** | `plan.md` Task 1, `TestPeriodicFft2` | Tautological assertion: passes for `return np.zeros_like(img)` |
| 4 | **MAJOR** | plan + shipped code | `rayleigh_mode` diverges from **all three** references; undeclared; **no test kills a broken version** |
| 5 | **MAJOR** | plan + `verify_claims.py` | Nothing pins the shipped `periodic=False` default. Flipping it is invisible to every test |
| 6 | **MAJOR** | spec §3.1 | `riesz_multiplier` reassociates the reference's single division; no drift row |
| 7 | MINOR | `plan.md` Task 4 Step 8 | The GUI-dropdown check is decorative — acceptance criterion 6 rests on an assumption |
| 8 | MINOR | `plan.md` Task 3 Step 5 | `noise_method` dispatch uses `1e-9`; both references use `epsilon` |
| 9 | MINOR | plan | `noise_method` has no `Field` bound; `-1.5` silently degrades to `T = ε` |
| 10 | MINOR | `plan.md` ×6 | Stale "measured" numbers, incl. one self-contradiction |
| 11 | NIT | `drift-register.md` M3 | "Changelog entry required" — no `CHANGELOG` exists anywhere in the repo |

---

## The mutation audit

The reviewer's brief asked "could a test pass while the code is wrong?" Rather than reason about it, each plausible bug was injected into the plan's kernel and the plan's own assertions were evaluated against it. **A mutant that no test kills is a hole.**

```
mutant             golden  load_bear  step2line  noiseonf  axis_pair  starsine  affine  clamp  T_floor
------------------------------------------------------------------------------------------------------
None               pass    pass       pass       pass      pass       pass      pass    pass   pass
swap_axes          FAIL    pass       pass       pass      FAIL       FAIL      pass    pass   pass
flip_h2_sign       pass    pass       pass       pass      pass       FAIL      pass    pass   pass
riesz_sign         FAIL    pass       pass       pass      FAIL       FAIL      pass    pass   pass
eps_1e-5           FAIL    pass       pass       pass      pass       pass      pass    pass   FAIL
no_lowpass         FAIL    pass       pass       pass      pass       FAIL      pass    pass   pass
tau_at_scale_1     FAIL    pass       pass       pass      pass       pass      pass    pass   pass
geom_off_by_one    FAIL    pass       pass       pass      pass       pass      pass    pass   pass
no_energy_clamp    FAIL    pass       pass       pass      pass       pass      FAIL    pass   pass
no_deviation_max   FAIL    pass       pass       pass      pass       pass      pass    pass   pass
no_T_floor         pass    pass       pass       pass      pass       pass      pass    pass   FAIL
rayleigh_broken    pass    pass       pass       pass      pass       pass      pass    pass   pass   <-- SURVIVES
```

Three things fall out, none of which inspection would have produced:

**The golden fixture does almost all the work.** Seven of twelve mutants are killed by `golden` and nothing else — `tau_at_scale_1`, `geom_off_by_one`, `no_deviation_max` among them. The behavioural controls (`step2line`, `noiseonf`) kill **zero** mutants. They are not useless — they answer a different question, "does this behave like a phase-congruency operator" — but they must never be mistaken for a correctness net. This restates drift lesson S6 with numbers.

**`starsine` is the sole test that catches the `−sum_h2` sign flip.** With the mutant correctly constructed (orientation only; flipping the Riesz multiplier as well makes the two errors partially cancel), `flip_h2_sign` passes `golden`, passes `axis_pair`, and dies only on `starsine`. The fixture cannot see it because the `.npz` stores `pc`, `ft` and `T` — **not** `orientation`. Exactly as spec §7 claims, now measured.

**`rayleigh_broken` survives everything** — see finding 4.

> The first run of this audit was itself wrong: `flip_h2_sign` was implemented as flipping *both* the Riesz multiplier and the orientation formula, and the errors partially cancelled, making `starsine` pass. That is the third too-strict-or-too-loose test written in this session. The audit is only trustworthy because its baseline mutant (`None`) passes every test and each mutant was checked to be the single intended change.

---

## Findings

### 1. BLOCKER — `List` is still used after the deletion

`plan.md` Task 2 Step 3 says: *"(`ifftshift` and `List` are no longer used here — remove them.)"*

`ifftshift` is genuinely dead. `List` is not:

```
$ grep -n "List\[" src/phenotypic/enhance/_focus_edge_phase.py
435:    def _construct_log_gabor_filters(self, radius: np.ndarray) -> List[np.ndarray]:   # deleted
473:    ) -> List[np.ndarray]:                                                            # KEPT
```

Line 473 is `_compute_angular_spread`'s return annotation, and Step 3 explicitly keeps that method. `from __future__ import annotations` defers evaluation, so pytest passes — but Step 7 runs `mypy` and `ruff`, which do not:

```
error: Name "List" is not defined  [name-defined]
F821 Undefined name `List`
```

`F821` is not auto-fixable, so `ruff check --fix` will not rescue it.

**Fix:** keep `List` in the import, or change line 473 to `list[np.ndarray]` in the same edit.

### 2. BLOCKER — the stated line range deletes a method the same step says to keep

`plan.md` Task 2 Step 3: *"Delete the methods … (they run from roughly line 379 to the end of the class)"*, then six lines later: *"`_compute_angular_spread` stays exactly where it is."*

Measured spans:

| Method | Lines | Fate |
|---|---|---|
| `_construct_filter_grids` | 379–433 | delete |
| `_construct_log_gabor_filters` | 435–469 | delete |
| **`_compute_angular_spread`** | **471–510** | **KEEP** |
| `_rayleigh_mode` | 512–542 | delete |

The file ends at 542, so "379 to the end of the class" swallows the kept method. An agent obeying the range breaks `_phasecong3:218` and `detect/_filamentous_fungi_detector.py:424`.

**Fix:** drop the range; name the three spans explicitly.

### 3. MAJOR — a tautological assertion

`plan.md` Task 1, `TestPeriodicFft2::test_it_removes_the_border_discontinuity_of_a_step_edge`:

```python
periodic = np.real(ifft2(periodic_fft2(img)))
smooth = img - periodic
assert np.abs(smooth).max() > 0.1
assert np.abs(periodic + smooth - img).max() < 1e-12   # <-- |img - img|
```

`smooth` is *defined* as `img - periodic`, so the second assertion is `|img − img| < 1e-12`. It is identically `0.0`, and it passes for `def periodic_fft2(img): return np.zeros_like(img)`. The first assertion only says "a smooth part exists", not that it is the border jump.

**Fix:** assert what `perfft2` exists for — that the periodic component's opposite-border jump is far smaller than the input's: `|p[:,0] − p[:,-1]|.max()` versus `|img[:,0] − img[:,-1]|.max()`.

### 4. MAJOR — `rayleigh_mode` diverges from all three references, and nothing tests it

Our port (`_focus_edge_phase.py:512`, lifted verbatim into the plan) drops zeros and lets `np.histogram` place its edges at `data.min()`. Kovesi anchors the bins at zero and retains zeros:

```matlab
% refs/phasecongmono.m:460-471
mx = max(data(:));
edges = 0:mx/nbins:mx;
n = histc(data(:),edges);
```

`phasepack` (`tools.py:86`) does `np.histogram(data, nbins)` — no zero-drop either.

Measured on the finest-scale amplitude of `load_synth_yeast_plate` (480 000 px, **zero** exact zeros, so the divergence is the *edge placement*, not the dropping):

```
port   (drops zeros, edges at min) = 0.00273765
Kovesi (keeps zeros, edges at 0)   = 0.00273729
relative difference                = 0.0130%
T(port) = 0.015957   T(Kovesi) = 0.015955     -> 0.0130% apart
```

Small — but it is an **undeclared deviation** in a register that says "Nothing else", and the plan newly consumes it from the monogenic path. Worse, the mutation audit shows the `noise_method = -2` branch is **completely untested**: doubling `rayleigh_mode`'s return value kills no test. The golden fixture cannot help — the `.npz`'s `_params` records `noiseMethod = -1` (median).

The plan's only `-2` test asserts `threshold > 0.0`, which any transcription error satisfies.

**Decision needed** (see below). Fixing it changes shipped `FocusEdgePhase` behaviour at `noise_method=-2`.

### 5. MAJOR — nothing pins the shipped `periodic=False` default

`_operate` calls `monogenic_phase_congruency(...)` without `periodic`. `TestGoldenFixture` passes `periodic=True` explicitly; `test_the_fixture_is_load_bearing` passes `False` explicitly. Every other test uses the default. Measured — flip the kernel default to `True` and re-run each test that relies on it:

```
step2line    default=False -> pass    default=True -> pass    blind
noiseonf     default=False -> pass    default=True -> pass    blind
axis_pair    default=False -> pass    default=True -> pass    blind
starsine     default=False -> pass    default=True -> pass    blind
affine       default=False -> pass    default=True -> pass    blind
```

So a one-character change to the kernel signature silently switches the operation onto the MATLAB branch — a `0.67`-absolute change in `pc` — and the whole suite stays green. This is the S6 failure mode, in the very feature that taught us S6.

**Fix:** one test that asserts the *operation's* output equals the kernel at `periodic=False` and differs from it at `periodic=True`.

### 6. MAJOR — `riesz_multiplier` reassociates the reference's single division

Both references divide **once**, by the radius array:

```matlab
refs/phasecongmono.m:170,183    radius(1,1) = 1;  ...  H = (1i*u1 - u2)./radius;
```
```julia
refs/frequencyfilt.jl:234-241   f[1,1] = 1  ...  H = (im.*fx .- fy)./f  ...  H[1,1] = 0
```

The spec (`monogenic-phase-congruency.md` §3.1) specifies `riesz_multiplier(sintheta, costheta) -> 1j*sintheta - costheta`, i.e. `1j*(fx/radius) - (fy/radius)` — **two** divisions, then a subtract — while its own prose two lines later calls this "Kovesi's `packedmonogenicfilters` (`H = (i·fx − fy)/f`)". The plan is faithful to the spec; the **spec** departs from the reference.

This is the source of the 1 ulp (`1.6e-16`/bin) and of the golden agreement being `5.324e-14` rather than the prototype's `3.52e-14`. Harmless numerically. But the governing principle is *faithfulness to validated reference logic beats a convenient shortcut*, and this is a shortcut with no drift row.

### 7. MINOR — the GUI check is decorative

`plan.md` Task 4 Step 8 constructs `OperationRegistry`, calls `.discover()`, builds an unused `names` list, then asserts only `'FocusEdgeMonogenicPhase' in phenotypic.enhance.__all__`. The registry is never queried, so acceptance criterion 6 rests on an assumption. A real check exists:

```
$ uv run python -c "from phenotypic.gui import OperationRegistry; r=OperationRegistry(); r.discover(); print(r.get_by_category('Enhancer'))"
Enhancer: ['FocusEdgeFrangi','FocusEdgeHessian','FocusEdgeLaplace','FocusEdgeMeijering','FocusEdgePhase','FocusEdgeSato','FocusEdgeSobel']
```

### 8. MINOR — the `noise_method` dispatch epsilon

`plan.md` uses `abs(noise_method + 1.0) < 1e-9`. Both references compare against `epsilon` (`1e-4` for `phasecongmono`): `refs/phasecongruency.jl:512`, `refs/phasecongmono.m:224`. `_phasecong3:273` already does the faithful thing. `1e-9` is arguably better; it is undeclared.

### 9. MINOR — `noise_method` has no bound

`noise_method = -1.5` matches neither branch, leaves `tau = 0.0`, and silently returns `threshold = ε`. Kovesi's MATLAB errors on the undefined `tau`. A `field_validator` restricting to `{-1, -2} ∪ [0, ∞)` would fail loudly.

### 10. MINOR — stale "measured" numbers (drift lesson S4, again)

| Where | Says | Actual |
|---|---|---|
| `plan.md` Task 2 Step 1 | `pc_sum.max` "a float around 0.33" | **0.805411** |
| `plan.md` Task 1 Step 4 | "all pass (26 tests)" | the file defines **19** |
| `plan.md` ×2 (`load_bearing`) | "Measured drift: 0.67 absolute" | **0.5342** — `0.67` is check_18's `n=256` figure; the test runs at `n=64` |
| `plan.md` `TestAxisConvention` | diagonal `45.18°` | **45.0075°** — the `abs=0.5` tolerance is carrying it, not the anchor |
| `plan.md` Task 3 Step 6 | "eleven orders" inside `rtol=1e-6` | **7.27 orders** (`1e-6 / 5.324e-14 = 1.9e7`) |
| `plan.md` commit text vs body | `3.5e-14` vs `5.3e-14` | contradicts itself two paragraphs apart |

**The reviewer said "eight orders"; that is also wrong.** `log10(1e-6 / 5.324e-14) = 7.27`.

### 11. NIT — `drift-register.md` M3 requires a changelog entry

`ls CHANGELOG* docs/CHANGELOG*` → nothing. No `CHANGELOG` exists in this repo. Either M3's requirement is stale, or the repo is missing one.

---

## Verified sound (so it is clear what was actually checked)

- Spec §6 tests 1–12: all mapped to concrete steps; all reproduce.
- Drift rows M1–M5: all implemented, none contradicted. `n_clamped = 0` on all three plates (600×800, 770×1644, 862×1696). `T == 1e-4` exactly on a constant image; smallest fixture `T = 3.7025e-3`, so "37× the floor" is exact.
- The four extracted helpers are **bit-identical** to the methods they replace (`np.array_equal`, all outputs). The Task 2 refactor is bit-identical by substitution. Full `_phasecong3` substitution on `load_synth_yeast_plate`: `max|Δ| = 0.0` on all six outputs.
- `ε = 1e-5` is genuinely what `_phasecong3:320` uses; no other epsilon crosses the `spread_weight` boundary. Passing `1e-4` instead shifts the weight by `max|Δ| = 0.094`.
- `log_gabor_scale` zeroes DC *before* the lowpass multiply, matching the shipped code. MATLAB does the opposite (`phasecongmono.m:201-203`), which is inert because `0 × finite = 0`.
- Ordering: **no hazards.** Each cluster leaves the repo green alone. Task 1's tests do not import `_kovesi_synthetic`. The relative import resolves (`tests/unit/enhance/__init__.py` exists; `tests/__init__.py` does not, so pytest anchors at `tests/`). The `.npz` is tracked; `git mv` works; `check_19`'s parent-walk resolves; `n_scale >= 2` breaks nothing in `src/`.

  > **Correction (2026-07-09, B+C review).** The parenthetical above originally read
  > "`n_scale=1` appears nowhere in `tests/` or `src/`." It appears in `tests/`:
  > `tests/fixtures/tune/back_compat_pipelines/enhance_features_edges.json` serializes
  > `"n_scale": 1`, and `test_annotation_back_compat.py` deserializes it. The `ge=2`
  > narrowing gives `1 failed, 14 passed` on that file. This was caught before cluster B ran
  > and is handled in the plan body; the claim here was simply false. Drift `M3`.
- Repo conventions: keyword-only construction; every `TuneSpec` window is a subset of its `Field` bound; the coverage gate will not regress (`output` is a `Literal`, excluded); `detect_mat ∈ [0,1]`; `rgb`/`gray` unmutated; rotation equivariance exactly `0.0` through the real `Image` path.
- Every attribution claim in the plan checks out against `refs/`, **including this morning's correction**: `filtergrid.m:49` and `frequencyfilt.jl:73` both divide odd axes by `N`; `phasepack` divides by `N−1` in *both* `filtergrid.py` and `tools.lowpassfilter`. Also confirmed: `T = max(…, ε)` appears **only** at `phasepack_phasecongmono.py:269`; Kovesi does not floor it in `phasecongmono` (he does in `phasesymmono`, a different function).

**One place the plan is *more* faithful than `verify_claims.py`, and must not be "fixed" back.** `phasecongmono.m:292` and `phasecongruency.jl:580` both use **single-argument** `atan(-sumh2/sumh1)`, range `(-π/2, π/2)`. The plan's fold to `(-π/2, π/2]` reproduces that. The **prototype's** `% np.pi → [0, π)` is the deviation.

## Not verified

- Whether Julia's `phasecong3` feeds `An` or `sumAn_ThisOrient` to `rayleighmode`. Irrelevant to Task 2 (the argument is preserved), but it means we cannot say whether shipped `FocusEdgePhase` is faithful there.
- Whether `_rayleigh_mode`'s zero-drop was ever deliberate. No comment or drift row explains it.
- `phasepack` end-to-end (not reinstalled, by constraint). The fixture's agreement is strong indirect evidence for the `noiseMethod = -1` path **only**.
- Full-suite runtime (the new test files do not exist yet).

---

## Decisions required before cluster A dispatches

1. **`riesz_multiplier` (finding 6)** — reassociate to `(1j*fx - fy)/radius` and match the reference exactly, or keep the current form and add a drift row?
2. **`rayleigh_mode` (finding 4)** — correct it to Kovesi's zero-anchored bins? That changes shipped `FocusEdgePhase` behaviour at `noise_method=-2`, so it is not a free fix.
3. **`noise_method` (finding 9)** — add a `field_validator` rejecting values outside `{-1, -2} ∪ [0, ∞)`?

Findings 1, 2, 3, 5, 7, 8, 10 have no design content and can be applied mechanically once approved.
