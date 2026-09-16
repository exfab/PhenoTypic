# Phase 0–2 implementation review — nested `GpuDetector` staging

**Scope:** commits `5aaeeb77` (Task 1), `6fd41645` (Task 3), `014d8cf9` (Task 4).
Diff reviewed: `git diff a806c4f9..014d8cf9 -- src/ tests/`.
**Reference tree:** `/bigdata/exfab/anguy344/gate-trees/014d8cf9` (clean, at the
reviewed SHA). The live worktree is **not** the reviewed state — Task 5/6 were
landing in it during this review (`_cli_pipeline_split.py`,
`tests/unit/cli/test_cli_pipeline_split.py`, `tune/_search_space/_infer.py`
modified; `_cli_replay_detector.py`, `test_pipeline_split_nested.py`,
`test_replay_detector.py`, `test_container_child_contracts.py` untracked).
Every `file:line` below resolves against `014d8cf9`.

**Verdict:** the change does what it says — the nesting blind spot is closed, and
the narrowing that was lost once before is genuinely on the production path this
time, structurally rather than by care. Three things keep it from being clean.
One is a blocker, **confirmed by execution**: `_CHILD_CONTRACT`'s lazy population
races, and because the GUI already swallows `ValueError`, the losing thread's
symptom is not an error but an intermittent silent route to CPU — a race whose
failure mode is silence, inside the one path that converts refusals into silence.
The second is that same GUI `except`, unfixed since it was raised in review. The
third is that Task 4 left `TwoKFilamentousDetector` writing step paths that
resolve to `KeyError`, which is worse than the coarse paths it replaced and
falsifies a spec invariant.

None of the three was introduced by this change. All three are load-bearing
because of it, which is the difference between a pre-existing wart and a defect.

**The tests are real.** 13 mutations, 11 killed; the two survivors are both in
`substitute_at_path` and are the two protections the implementer added by hand
rather than taking from the plan (**F6**). In particular, reverting
`pipeline_requires_gpu` to the old top-level-only scan kills 8 tests, and three
independent mutants kill the "composition primitives only" narrowing — so the
failure mode that lost that narrowing once before, a rule living in a table and
in prose but in no executed code path, is now excluded by execution rather than
by care.

---

## Summary

| # | Finding | Severity | Verified |
|---|---|---|---|
| F1 | `_CHILD_CONTRACT` lazy population races; the losing thread's refusal is then swallowed into a silent CPU route | **BLOCKER** | **CONFIRMED by execution** (probe P1) |
| F2 | The GUI swallows every new refusal; one shape is a regression | **MAJOR** | CONFIRMED by reading |
| F3 | Tasks 1–4 are not shippable alone: nested-GPU runs die, on SLURM after the whole array is submitted | **MAJOR** | CONFIRMED by reading |
| F4 | `TwoKFilamentousDetector` records step paths that resolve against nothing; spec §5.3's invariant is false | **MAJOR** | **CONFIRMED by execution** (probe P4) |
| F5 | ~~The CPU-only-slot scan cannot see a detector that *is* the slot entry~~ | **REFUTED** | **REFUTED by execution** (probe P2) |
| F6 | `substitute_at_path`'s two hand-added protections are undefended — both surviving mutants land here | MINOR | **CONFIRMED by execution** (M9, M10 SURVIVED) |
| F7 | `substitute_at_path` aliasing | MINOR | **CONFIRMED by execution** (probe P3) |
| F8 | The `reset` retraction | **correct** | CONFIRMED by reading |
| F9 | Blast radius of deeper `pipeline_step_path`s | **clean** | CONFIRMED by search |
| F10 | Is the narrowing on the path `pipeline_requires_gpu` takes? | **yes** | **CONFIRMED by execution** (M3, M5, M11, M13) |
| F11 | The two recorded gaps | **adequate**, one wording fix | CONFIRMED by reading |
| F12 | Can the new tests fail? | **11/13 killed** | **CONFIRMED by execution** |
| F13 | Gate coverage: the one shard that could exercise `NapariPipelineViewer` could not run | note | CONFIRMED from gate output |

---

## F1 — `_CHILD_CONTRACT`'s lazy population is not thread-safe, and the losing thread gets the wrong answer, not an error

**BLOCKER. CONFIRMED by execution**, in the frozen `014d8cf9` tree, two threads,
0.05 s apart:

```
=== P1: live race on the lazily-populated _CHILD_CONTRACT ===
  T2-CompositeEnhance -> ('RAISED', 'UnstageableGpuDetectorError',
                          'a GpuDetector cannot be nested inside CompositeEnhance:
                           only composition primitives (ImagePipeline, CompositeD')
  T1-CompositeDetector -> ('OK', 'parallel')
  restored table: {'CompositeDetector': 'parallel', 'CompositeEnhance': 'parallel'}
```

The probe widened the window with a `time.sleep` in the dict's `__setitem__`; it
did not *create* the window, which is the gap between the two literal insertions
at `:181-182` with a lazy `phenotypic.enhance` import sitting in it.

*(This is the lead's question 3.)*

**The defect in one sentence.** `_populate_child_contract` guards on "is the dict
non-empty?" and then fills it one key at a time, so a second thread entering
between the two insertions sees a truthy dict, returns immediately, and
`_child_contract` refuses a `CompositeEnhance` that the table is supposed to
allow.

```python
# src/phenotypic/_cli/_cli_validation.py:170-183
def _populate_child_contract() -> None:
    if _CHILD_CONTRACT:            # <-- truthy after the FIRST insert
        return
    from phenotypic.detect import CompositeDetector
    from phenotypic.enhance import CompositeEnhance

    _CHILD_CONTRACT[CompositeDetector] = "parallel"
    _CHILD_CONTRACT[CompositeEnhance] = "parallel"   # <-- window closes here
```

The window is not theoretical-only: the two names come from
`phenotypic.detect` and `phenotypic.enhance`, deliberately imported lazily
because they are "heavy subpackages" (the function's own docstring, `:176-178`).
A cold first import of `phenotypic.detect` is exactly where a thread gets
descheduled, and it sits *inside* the guarded region, so a concurrent caller is
most likely to arrive precisely while the table is half-built.

**Failure scenario.** Dash serves callbacks on a threaded Werkzeug server. Two
run-console callbacks fire close together for a pipeline whose GPU detector sits
in a `CompositeEnhance`. Thread A enters `_populate_child_contract`, sets the
`CompositeDetector` key, and blocks on the `phenotypic.enhance` import. Thread B
calls `_child_contract(<CompositeEnhance>)`, `_populate_child_contract()` returns
at the guard, `type(container)` is not in the one-key table, and B raises

> `UnstageableGpuDetectorError: a GpuDetector cannot be nested inside CompositeEnhance: only composition primitives ... may carry one.`

for a placement the design explicitly permits. Chained with **F2**, the GUI
catches it as a `ValueError` and reports "not a GPU pipeline".

**The two findings compound into something worse than either.** The
user-visible symptom of this race is not an error message and not a crash — it
is a GPU pipeline quietly reading as a CPU pipeline, intermittently, depending
on thread interleaving. A race whose failure mode is silence, living inside the
one code path that already converts refusals into silence. That is why it is the
blocker and **F2** is not: **F2** alone produces a wrong UI for a pipeline the CLI
will refuse anyway, whereas **F1 + F2** produces a wrong answer for a pipeline
that is *correct*, and does so non-deterministically, so it will not reproduce
when someone goes looking.

**Reachability.** `pipeline_requires_gpu` has four callers
(`_cli_execution_strategies.py:344`, `:906`, `:1341`;
`gui/run_console/_callbacks.py:255`). The three CLI sites are single-threaded
per process and SLURM workers are separate processes, so **the SLURM workers are
not at risk** — the answer to "is the laziness thread-safe enough for the SLURM
workers?" is yes, for the uninteresting reason that they never share an
interpreter. The GUI is the exposure.

**The second half of the question — can a reader see an empty dict and get a
wrong answer rather than an exception?** Single-threaded, no: every read inside
this module goes through `_child_contract`, which calls `_populate_child_contract()`
first (`:196`). But `_CHILD_CONTRACT` is a module-level name that
`tests/unit/cli/test_gpu_detection_tree_wide.py:21` already imports directly, and
the only thing stopping a future reader from doing a bare `cls in _CHILD_CONTRACT`
is a comment — `"Populated lazily by _populate_child_contract -- read it through
that, never directly, or a first reader sees an empty dict"` (`:165-166`). That is
a convention where a mechanism is cheap.

**Fix.** Publish the table in one binding rather than filling it incrementally.
Either

```python
_CHILD_CONTRACT: dict[type, str] = {}

def _populate_child_contract() -> None:
    if _CHILD_CONTRACT:
        return
    from phenotypic.detect import CompositeDetector
    from phenotypic.enhance import CompositeEnhance
    _CHILD_CONTRACT.update({CompositeDetector: "parallel",
                            CompositeEnhance: "parallel"})
```

(one `dict.update` from a fully-built literal — atomic under the GIL, and the
guard can then never observe a partial table), or better, replace the mutable
global with an `@lru_cache`d accessor returning a `MappingProxyType`, which also
removes the "never read it directly" convention by removing the thing to read.

Note the `update` form still leaves a narrower race — two threads can both pass
the guard and both do the work — but that is idempotent and harmless, which the
current form is not.

**Status:** the lead has handed the one-line `update()` fix to the Task 5/6
agent, which is already editing this function for two other reasons, rather than
opening a second writer on the file. Correct call — this file is the contended
one in the whole change.

---

## F2 — The GUI turns every new refusal back into silence, and for one shape that is a regression

**MAJOR.** Raised as `m19` in `plan-a-review-2.md`; **documented, not fixed**.

```python
# src/phenotypic/gui/run_console/_callbacks.py:248-256
def _pipeline_uses_staged_gpu(path_value: object) -> bool:
    ...
    try:
        from phenotypic._cli._cli_validation import pipeline_requires_gpu
        return pipeline_requires_gpu(Path(path_value))
    except (OSError, ValueError, TypeError):
        return False
```

`UnstageableGpuDetectorError` subclasses `ValueError` (`_cli_validation.py:135`),
so every placement refusal this change adds is caught here and reported as
"not a GPU pipeline".

**The regression.** Consider `ops={"Sam2": <GpuDetector>}` plus
`meas={"MZ": MeasureSymZones(center_detector=<GpuDetector>)}`.

- Before Task 3: `any(isinstance(op, GpuDetector) for op in pipeline.get_ops().values())` → `True`.
- After Task 3: `hits == [("Sam2",)]`, then the unconditional `_CPU_ONLY_SLOTS`
  loop (`:307-336`) finds the measurer's detector and raises → GUI returns `False`.

So a pipeline the GUI previously identified as a GPU run now reads as a CPU run.

**Consequence.** The flag's only consumer is `_callbacks.py:1744` —
`visible = mode == "slurm" and _pipeline_uses_staged_gpu(pipeline_path)` — the
visibility of the GPU SLURM options panel. The user loses the ability to set the
GPU partition/account for that pipeline. The CLI does **not** catch the error
(`_cli_execution_strategies.py:1341` and `phenotypicCLI.py:594`, `:2608`, `:2962`
are all uncaught — I checked each for a surrounding broad `except`), so the run
itself still fails loudly. **This is a UI-correctness defect, not a wrong-numbers
defect**, which is why it is MAJOR and not a blocker.

**Why it still matters more than its blast radius suggests.** The change's whole
premise is "silent wrong → loud fail". On the one surface where a human is
looking at the answer, the failure is still silent, and the code records this in
a docstring *inside the function being swallowed*
(`_cli_validation.py:352-360`): *"the GUI silently reports 'not a GPU pipeline'
instead of surfacing the message ... Both still want deciding."* A defect
documented at the site that cannot observe it is not mitigated. The minimum fix
is to narrow the GUI's `except` to exclude `UnstageableGpuDetectorError` and
surface its message as a toast — the callback already has a warning-toast path
(`_looks_like_pipeline_json`, `:230-243`, exists for exactly this).

---

## F3 — Tasks 1–4 are not shippable alone: every nested-GPU run dies, and on SLURM only after the full array is submitted

**MAJOR.** CONFIRMED by reading the committed source; not executed, because the
live splitter is mid-edit by the Task-5 agent.

Task 3 makes `pipeline_requires_gpu` return `True` for a `GpuDetector` inside a
`CompositeDetector`. `uses_staged_gpu_strategy`
(`_cli_execution_strategies.py:1334-1346`) therefore routes it to a staged
strategy (`:1411`, `:1423`). But at `014d8cf9` the splitter still scans only the
top level:

```python
# src/phenotypic/_cli/_cli_pipeline_split.py:33-38  (at 014d8cf9)
ops = pipeline.get_ops()
gpu_keys = [k for k, op in ops.items() if isinstance(op, GpuDetector)]
if len(gpu_keys) == 0:
    raise ValueError(
        "no GpuDetector in pipeline; staged execution requires exactly one"
    )
```

- **Local** (`StagedGpuStrategy.execute`, `_cli_staged_strategy.py:75`): splits in
  the driving process, so the run dies immediately. Acceptable.
- **SLURM** (`StagedSlurmStrategy.execute`, `_cli_staged_slurm.py:521`): does
  **not** split. `split_pipeline_at_gpu` appears only in the workers
  (`_cli_staged_slurm_worker.py:182`, `:296`, `:448`). So the controller, the
  manifest, the ledger and the Stage-1 array are all created and submitted, and
  then every array task raises `ValueError: no GpuDetector in pipeline`.

On the 33,923-image driver that is a complete submission cycle burned before the
first useful byte. It is loud — which is the intended direction, and strictly
better than the silent CPU run it replaces — but:

1. **no test pins it**, so nobody would notice if a future change made this shape
   silently do something else instead; and
2. the commit's "two gaps recorded rather than papered over" does not mention it,
   while it is by some distance the most consequential thing about running from
   this SHA.

Task 5 closes the window (the in-flight `_cli_pipeline_split.py` already calls
`find_gpu_detectors(pipeline, strict=True)`), so this is a **commit-boundary**
finding, not a design one. It matters for anyone who runs from, or bisects to,
`014d8cf9`.

**Recommendation, and it should outlive Task 5 rather than vanish with it.**
`StagedSlurmStrategy.execute` should split once in the submitting process before
creating *any* scheduler state — the orchestration UUID, the manifest, the
ledger, Controller 0. That is where `StagedGpuStrategy` already does it
(`_cli_staged_strategy.py:75`), and it is the right shape independently of this
bug: the splitter is the only component that can reject a pipeline on structural
grounds, and every reason it rejects one is knowable before a single `sbatch`.
Doing it only in the workers means any future structural refusal — not just this
one — is discovered n_images times, on the far side of a submission, per array
task. Once Task 5 lands, this stops being a live failure and becomes an
unexercised asymmetry between the two staged strategies, which is exactly the
kind of thing that is cheap now and expensive later.

---

## F4 — `TwoKFilamentousDetector` records step paths that resolve against nothing, and spec §5.3's invariant is false

**MAJOR. CONFIRMED by execution** (probe P4).

`TwoKFilamentousDetector` has three operation-valued fields
(`_two_k_filamentous_detector.py:61,70,71`) and drives all three. Task 4 routed
exactly one of them through `apply_child`:

| field | call site | routed? |
|---|---|---|
| `branch_base` | `:172-174` | **yes** |
| `center_detector` | `:148-151` | no |
| `background_subtractor` | `:154` | no |

**Measured** — walker paths against what the journal actually recorded:

```
  walker:    TwoK/center_detector/InoculumDetector
             TwoK/center_detector/KeepSectionLargest
  recorded:  ['TwoK', 'InoculumDetector']
             ['TwoK', 'KeepSectionLargest']
```

**This is worse than the collision I predicted, and differently shaped — I was
wrong about the mechanism.** I expected `center_detector`'s children to record
`['TwoK']`: coarse, colliding with the container's own entry, but at least a path
that *resolves*. What actually happens is that `center_detector` is itself an
`ImagePipeline` (built at `:107-108` via `__build_center_pipe`), so its own
`_run_operations` faithfully pushes a segment per child — and the **middle
segment is dropped**, because nothing pushed `center_detector`. The result is a
well-formed, plausible-looking path that addresses nothing:
`get_at_path(pipeline, ["TwoK", "InoculumDetector"])` raises `KeyError`.

So the journal does not record "I don't know where this ran". It records
something false about where it ran, in the same vocabulary as the entries that
are correct, with nothing to tell them apart. `branch_base` matches the walker
exactly, three lines away in the same class. Before this change the whole class
was uniformly coarse; now part of it is right, part is wrong, and distinguishing
them requires re-deriving the tree — which is the thing the step path exists to
save you from.

**The justification does not cover the case.** Spec §5.2 names four classes that
must push a branch step and states exactly one exclusion:

> **Explicitly excluded: the zone measurers' `center_detector`.** Per
> `measure/CLAUDE.md`, a nested operation run by a *measurement* is a private
> probe…

That carve-out is about `MeasureSymZones.center_detector` / a
`MeasureFeatures`' nested probe. `TwoKFilamentousDetector.center_detector` is a
**detector's** field. The commit message merges the two —

> Scope held: center_detector records no step path (a measurement's nested
> operation is a private probe, measure/CLAUDE.md)

— and there are two different `center_detector`s in this codebase. The one the
rule protects is excluded correctly and is pinned by
`test_a_measurement_probe_records_no_step_path`; the one in the detector is
excluded by a rule that does not reach it.

**The plan does authorize it**, on a different ground: *"Only `branch_base` is
listed. `center_detector` (`:149,151`) and `background_subtractor` (`:154`) are
**not** descended, because the class is refused for staging anyway (Task 5 step
3c) and descending them buys nothing."* That is a **staging** argument applied to
a **provenance** change. The journal is written on every `TwoKFilamentousDetector`
run, staged or not; Task 4's stated purpose is journal correctness, and it is the
one task in this phase whose blast radius is "every pipeline using a container
op". Being refused for GPU staging says nothing about whether a user can read
their own provenance.

And "descending them buys nothing" is now measurably false: not descending them
does not leave the journal unchanged, it *writes an unresolvable path into it*.
The choice was never between "addressed" and "not addressed" — it was between
"addressed" and "misaddressed", and the plan's sentence reads as though it were
the former.

**Consequence for the spec.** §5.3 says *"the identity `gpu_path ==
pipeline_step_path` is a **testable invariant** (§10) rather than a coincidence
maintained by hand."* It is not an invariant of the codebase — it holds for
`ImagePipeline`, `CompositeDetector`, `CompositeEnhance` and
`FilamentousFungiDetector`, and fails for `TwoKFilamentousDetector`.
`test_the_recorded_paths_are_the_walker_paths`
(`test_provenance_step_descent.py:106-123`) asserts set equality of walker paths
and recorded paths, and would fail on a TwoK pipeline; it only ever builds
composites, so the invariant is asserted exactly where it happens to hold. That
is the shape of test that gets promoted to "proven" in a later summary.

**Recommendation — asked for explicitly, so: descend it.** Route
`center_detector` and `background_subtractor` through `apply_child`. Three lines,
the same helper already imported in the same method, and `apply_child` supplies
the `reset=False` that `:149` currently passes by hand — so `:148-151` collapses
from a four-line `isinstance` branch to one call, exactly as it did in the other
four adopters.

I am not recommending the alternative (scope §5.3 to composition primitives and
leave the code alone), and the measurement is why. Scoping the invariant would be
the right answer if the untouched fields recorded *nothing* — an absent path is
honestly incomplete, and a narrower invariant would describe it truthfully. They
record a **wrong** path instead. No wording of §5.3 makes
`['TwoK', 'InoculumDetector']` correct, so scoping the spec would leave a
documented invariant that is true of the classes it names while the journal
quietly carries entries that resolve to `KeyError` for a class it does not name.
That is the failure mode this whole change exists to remove, one layer down.

**Do both, in fact** — descending is the fix, but §5.3 should still say which
classes it binds, because "container operations" is not a closed set and the next
one added will not be covered by anything. Concretely, one change to Task 9 makes
the invariant self-enforcing for classes nobody has written yet:

> assert every recorded `pipeline_step_path` **resolves** — `get_at_path(pipeline, p)`
> does not raise — rather than asserting set equality with the walker's paths.

Resolvability is the property that is actually wanted, it is strictly cheaper,
and it does not break the moment a branch is skipped or short-circuits (set
equality demands every walker path also be *recorded*, which is false for any
pipeline with an unfilled `None` slot or a conditional branch). The current
`test_the_recorded_paths_are_the_walker_paths`
(`test_provenance_step_descent.py:106-123`) asserts set equality and only ever
builds composites — so it is asserted exactly where it happens to hold, which is
the shape of test that gets promoted to "proven" in a later summary. Under the
resolvability form it would have caught this on a TwoK fixture.

---

## F5 — REFUTED: the CPU-only-slot scan's blind spot for a direct slot entry is unreachable

**REFUTED by execution** (probe P2). I raised this as a live concern; it is not
one. Recording it here as refuted rather than deleting it, because the reasoning
is what makes the `_CPU_ONLY_SLOTS` loop's shape correct rather than lucky.

```python
# src/phenotypic/_cli/_cli_validation.py:307-336 (abridged)
for slot in _CPU_ONLY_SLOTS:
    accessor = getattr(pipeline, f"get_{slot}", None)
    if accessor is None:
        continue
    container = accessor()
    ...
    for name, op in entries:
        for sub_path, sub_op in walk_operations(op):      # <-- skips `op` itself
            if isinstance(sub_op, GpuDetector):
                raise UnstageableGpuDetectorError(...)
```

`walk_operations` deliberately does not yield its own root
(`_operation_tree.py:62-64`), so a `GpuDetector` that *is* the value of a `meas` /
`post` / `filters` / `model` entry is not seen — only ones nested inside such an
entry. `plan-review.md:365-372` argues this is unreachable because those fields
are typed `Dict[str, MeasureFeatures]`, `Dict[str, PostMeasurement]`,
`Dict[str, SetAnalyzer]` and `Optional[ModelFitter]`. I did not take that on
trust. Measured, all four:

```
  meas:    rejected by construction (ValidationError)
  post:    rejected by construction (ValidationError)
  filters: rejected by construction (ValidationError)
  model:   rejected by construction (ValidationError)
```

**So the direct-entry case is unconstructible**, and `walk_operations` skipping
its own root costs the slot scan nothing. The loop remains load-bearing for the
*nested* case, which **is** constructible and **is** refused —
`meas/MeasureSymZones/center_detector`, pinned by
`test_a_gpu_detector_in_the_ROOT_meas_slot_is_refused` and by
`test_a_ROOT_meas_slot_gpu_detector_does_not_route_to_the_cpu_strategy`.

Two smaller things in the same loop, both the same shape as the bug the loop
exists to prevent:

- `if accessor is None: continue` (`:318-319`) is unreachable today — all four
  accessors exist (`_image_pipeline_core.py:561, 572, 599, 634`) — and
  `_CPU_ONLY_SLOTS` is a fixed literal. A silent `continue` on a missing accessor
  in a function whose job is to stop silence should be an assertion; if a slot is
  ever renamed, the refusal disappears with nothing failing.
- The slot name is reconstructed as an f-string (`f"get_{slot}"`), so the link
  between `_CPU_ONLY_SLOTS` and the accessors is invisible to every tool. Storing
  the bound accessor names, or asserting all four resolve at module import, costs
  one line.

---

## F6 — `substitute_at_path`'s two hand-added protections are undefended: both surviving mutants land here

**MINOR in consequence, but it is the only place in the diff with no test
coverage at all — and it is confirmed, not predicted.** Of 13 mutations, 11 were
killed; **the 2 survivors are both in `substitute_at_path`, and both are
protections added by hand during implementation rather than specified by the
plan.**

```
M9  substitute drops _provenance_pipeline           30 passed, 1 xfailed   SURVIVED
M10 substitute drops the non-ImagePipeline guard    30 passed, 1 xfailed   SURVIVED
```

**(a) The `TypeError` guard** (`_operation_tree.py:154-158`) — one of the "two
deviations from the plan text" the Task-1 commit records. Replacing its condition
with `if False:` changes no test result. Under the mutant the call falls through
to the generic branch, `_INDEXED` does not match an ops key, `hasattr` fails, and
the caller gets `KeyError('<operation name>')` — which is precisely the confusing
failure the guard's own comment says it exists to prevent ("failing with a bare
`KeyError` naming the operation instead of the unsupported container").

**(b) The `qc`/`plots`/`name`/`_provenance_pipeline` carry**
(`_operation_tree.py:175-188`), added in response to `plan-review.md` MAJOR #6
("`substitute_at_path` loses six fields and a private attr"). The test file is
**byte-for-byte the plan's Step-1 block**, which predates that fix, so nothing
asserts the carry. The code defends itself with a comment — *"An earlier draft
dropped `qc`, `plots`, `name` and `_provenance_pipeline` — a booby trap for
whoever next reads plots off a substituted pipeline"* — and a comment does not
fail. A reader who simplifies the rebuild back toward the plan's version
reintroduces the exact defect #6 named, with a green suite. That is not
hypothetical: the plan's version is still in the plan, and it is the version a
future executor would reach for.

### These are the same finding as F13, and that is the argument

M10's guard exists to refuse an `ImagePipelineCore` that is not an
`ImagePipeline` — today, exactly `NapariPipelineViewer`
(`_napari_pipeline_viewer.py:72`, the only such subclass). So the one deviation
in Task 1 with nothing behind it is unwitnessed **twice over**: no test in the
shard that could not run (**F13**), and no test that a mutation can kill
(**F6**). Either observation alone is weak — a missing extra, or a thin test.
Together they say the `ImagePipelineCore`-vs-`ImagePipeline` distinction, which
is the load-bearing type relation in Task 1, rests entirely on the implementer
having reasoned correctly at the time.

### Recommended tests

Two, both in `tests/unit/sdk_/test_operation_tree.py`, both cheap.

**1. The carry — and it has a trap that would make the obvious version useless.**

```python
def test_substitute_carries_the_slots_a_stage3_pipeline_reads():
    pipe = _pipeline_with_composite()
    pipe._provenance_pipeline = {"name": "sentinel", "version": "0"}
    original = pipe._provenance_pipeline

    out = substitute_at_path(
        pipe, ("CompositeDetector", "ops[1]"), OtsuDetector(ignore_zeros=True)
    )

    assert out._provenance_pipeline is original
    assert out.name == pipe.name
    assert out.get_qc() == pipe.get_qc()
    assert out.get_plots() == pipe.get_plots()
    assert (out.nrows, out.ncols) == (pipe.nrows, pipe.ncols)
```

**The assignment on the second line is load-bearing.**
`_provenance_pipeline` is `PrivateAttr(default=None)`
(`_image_pipeline_core.py:231`), so on a freshly constructed pipeline the
natural assertion `out._provenance_pipeline is pipe._provenance_pipeline`
compares `None is None` and **passes under M9**. A test written the obvious way
would leave this mutant alive and look like coverage. Set a non-`None` sentinel
first, or assert against a captured value, or the test is decorative.

For `qc`/`plots` the fixture needs at least one entry each — both default to
empty lists, and `[] == []` has the same problem.

**2. The guard — and do *not* import `NapariPipelineViewer` to write it.**

```python
def test_substitute_refuses_a_non_ImagePipeline_core_by_name():
    """A walk can hand a path THROUGH any ImagePipelineCore, but the rebuild is
    ImagePipeline-specific. Refuse by name, not with a bare KeyError.

    Uses a local subclass rather than NapariPipelineViewer: the guard keys on
    the TYPE RELATION (ImagePipelineCore and not ImagePipeline), not on napari,
    and tying a core invariant to an optional extra is what left this untested.
    """
    class _CoreButNotAPipeline(ImagePipelineCore):
        pass

    node = _CoreButNotAPipeline(ops={"OtsuDetector": OtsuDetector()})
    with pytest.raises(TypeError, match="_CoreButNotAPipeline"):
        substitute_at_path(node, ("OtsuDetector",), OtsuDetector(ignore_zeros=True))
```

`ImagePipelineCore` is a plain pydantic model with no abstract methods
(`_image_pipeline_core.py:143`, `class ImagePipelineCore(BaseOperation,
LazyWidgetMixin)`), so the local subclass costs two lines and needs no optional
dependency. Under M10 this fails with `KeyError` instead of `TypeError`, killing
the mutant. Match on the class name, not on a fixed string, so the assertion also
pins the "refuse **by name**" half — which is the entire point of the guard, and
what a `pytest.raises(TypeError)` with no `match` would miss.

**(c) `_CHILD_CONTRACT`'s values have no consumer at this SHA.**
`_child_contract`'s return value (`"parallel"` / `"sequence"`) is read only by
`_branch_prefix`, which is Task 5. At `014d8cf9` the table functions purely as a
membership set. So the comment's central safety claim — *"this restates a type
contract; it does not cache an observation about today's `_operate`"* — is
currently unverified in either direction, and the plan's own value-verifying
probe tests (`tests/unit/detect/test_container_child_contracts.py`, plan
`:1379-1418`) belong to Task 5 and are untracked in the worktree as I write.
That placement is correct; it just means the table's *values* are not evidence
yet, and should not be described as proven until Task 5 lands them.

---

## F7 — `substitute_at_path` aliasing

*(Lead's question 7.)* **CONFIRMED by execution** (probe P3). Measured on
`ImagePipeline(ops={"C": CompositeDetector(ops=[Otsu, Manual])}, meas={"MZ": …})`
substituting at `("C", "ops[0]")`:

```
  root object identical:       False
  meas dict shared:            False
  meas VALUE shared:           True
  composite node copied:       True
  composite ops LIST shared:   False
  untouched sibling ops[1]:    True   (shared)
  __pydantic_private__ shared: False
  _provenance_pipeline shared: True
  name carried:                True
```

**Which objects end up shared:** every untouched operation, in both directions —
the sibling `ops[1]`, every `meas`/`post`/`filters`/`model` value, and
`_provenance_pipeline`. The *containers* are all fresh (`ops` dict, `meas` dict,
the composite's `ops` list, the composite node itself), so rebinding anything on
the substituted pipeline cannot reach the original. That is the right split, and
it matches what the original pipeline already does with those objects.

**Can Stage 3 mutate one through the other?** Not by any route Stage 3 takes
today. Operations are applied to an image, not mutated; the one thing that does
write to a shared operation is the benchmark toggle in `_run_operations`
(`_image_pipeline_core.py:892-895`, `operation._benchmark = …` on a shared nested
pipeline), and that is pre-existing, restored in the same block, and not
introduced here.

One measured result corrects something I expected. `__pydantic_private__` is
**not** the same object after `model_copy(deep=False)`, so rebinding a
`PrivateAttr` on the copy is isolated — my concern that private state is shared
wholesale was wrong. What I did **not** measure, and what pydantic's shallow-copy
semantics imply, is that the *values inside* that mapping are still shared
references: a `PrivateAttr` holding a mutable object (a cache dict, a loaded
model handle) would be the same object in both trees, and mutating it in place
would be visible through either. No current code does that, and I am flagging it
as inferred rather than observed.

`substitute_at_path` has two branches with different copy semantics.

**`ImagePipeline` branch (`:160-188`)** — not a `model_copy` at all; it rebuilds.
`ops` is a fresh dict but its *values* are the original operation objects except
the one replaced. `meas`, `post`, `filters`, `model`, `qc`, `plots` are passed
straight through from the original's accessors, and `_provenance_pipeline` is
assigned by reference (`:187`). Pydantic does not revalidate model instances by
default here (no `revalidate_instances` on `ImagePipelineCore.model_config`,
`:186-187`), which the round-trip test relies on — `get_at_path(out, path) is
replacement` only passes because construction does not copy.

**Generic branch (`:199-222`)** — `model_copy(deep=False)`. Shares every field
value with the original; the private-attr *mapping* is a fresh object (measured),
so rebinding is isolated, but its values are shared references (inferred — see
above). The indexed path immediately rebinds the list (`sequence = list(...)`;
`setattr`, `:204-212`), so the original's list is safe, which the
`composite ops LIST shared: False` measurement confirms; a non-indexed attribute
path rebinds only that attribute.

**The one sharing nobody has written down** is `_provenance_pipeline` (`:187`,
measured shared). A substituted pipeline therefore records the **original**
pipeline's identity in the journal. For Stage 3 that is almost certainly what you
want — the journal should name the user's pipeline, not the throwaway stub-bearing
copy — but it is a deliberate-looking line with no comment and no test, and the
comment block immediately above it (`:190-198`) discusses only the deep-copy
*cost*, not the aliasing. One sentence there would keep the next reader from
"fixing" it.

Deliberately not carried, and documented: `benchmark`, `verbose`, `reset`,
`desc_value` (`:172-174`). `reset` is the one with teeth — a substituted Stage-3
pipeline built from a `reset=True` root silently loses the reset. The comment
says so and calls it harmless for a throwaway; that is honest and I agree it is
harmless **today**, and it is exactly the kind of thing that stops being harmless
when someone reuses the helper.

---

## F8 — The `reset` retraction is correct. Both citations verified.

*(Lead's question 5.)* **CONFIRMED by reading.**

- `_image_pipeline_core.py:193` → `reset: bool = False` ✓
- `_image_pipeline_core.py:966` → `effective_reset = reset if reset is not None else self._reset` ✓

The claim also depends on a third fact the commit does not cite:
`self._reset` is a property returning `self.reset`
(`_image_pipeline_core.py:459-462`), so the field default really is what
`reset=None` resolves to. It does.

`TwoKFilamentousDetector`'s default `branch_base` is built at `:103` as
`ImagePipeline(ops=[...])` with `reset` unset, hence `False`. So the previous
`self.branch_base.apply(enhanced, inplace=True)` already ran with
`effective_reset=False`, and `apply_child`'s `reset=False` is not a behaviour
change **on the default path** — which is precisely what the commit claims. It
*is* a change for a user-supplied `branch_base=ImagePipeline(..., reset=True)`,
and that case is pinned by
`test_apply_child_sends_reset_false_to_a_pipeline_child`
(`test_provenance_step_descent.py:163-182`), whose docstring states the
distinction correctly.

**Nothing was shipped as a no-op that isn't one.** The retraction stands.

---

## F9 — Deeper `pipeline_step_path`s break no production consumer

*(Lead's question 4: "search rather than assume".)* **CONFIRMED by search, not by
trusting the implementer's grep.**

`grep -rn "step_path" src/` over the whole package, at `014d8cf9`:

- **One production writer** — `_cli_staged_workers.py:497`,
  `pipeline_step_path=[plan.gpu_key]`. Writes a single segment from a top-level
  key. That is Task 5's to change and is already wrong for a nested detector, but
  it is unreachable for one at this SHA because of **F3**.
- **One validator** — `_provenance.py:277-282`. It accepts `None`, or a non-empty
  list of non-empty strings. Deeper paths pass unchanged; there is no length or
  shape constraint to violate.
- **Zero readers** in `_cli/`, `sdk_/` or `gui/`. The only other hit in `_cli/` is
  a docstring line in the in-flight `_cli_pipeline_split.py:37`.

So the specific consumers the lead named:

- **Continuation predicates** — key on store validity, the Stage-2 signal, and
  atomic completion markers (`_cli/CLAUDE.md`, *Staged GPU engine*), never on
  journal content. `processing_configuration_digest` is an explicit allowlist
  (`_cli_failure_tracker.py`) and the journal is not in it.
- **Migration** — `_cli_migrate_provenance.py` converts schema versions; no
  step-path handling.
- **Store journals** — written and validated, not compared.
- **GUI** — no reader.

The one structural consumer is `_carry_logical_image_state`'s entry-equality
match (`_provenance.py:~945-955`), which compares whole operation dicts including
`pipeline_step_path`. Both sides are produced in the same process by the same
code, so they move together.

**Residual exposure, and it is cosmetic.** A store written before this commit
keeps the old shallow paths; one written after has deep ones. Nothing compares
journals across stores, so a tree processed half-before and half-after is
internally inconsistent in a field nothing reads. Worth one sentence in the
Task-14 documentation, not a migration.

---

## F10 — The narrowing *is* on the path `pipeline_requires_gpu` takes

*(Lead's question 2 — "verify by execution".)* **CONFIRMED by execution.** Four
mutants, four independent ways to sever the narrowing, all killed:

| Mutant | What it severs | Killed |
|---|---|---|
| M3 | the ancestor-contract call disappears from `find_gpu_detectors` | 3 tests |
| M5 | `_child_contract` returns `"parallel"` for everything, never refuses | 3 tests |
| M11 | `_child_contract` stops populating the table | 3 tests |
| M13 | `pipeline_requires_gpu` reverts to the old top-level-only scan | 8 tests |

The chain those mutants cut, all in `_cli_validation.py`:

```
pipeline_requires_gpu               :378   return bool(find_gpu_detectors(pipeline))
  find_gpu_detectors                :299   for path, _ in hits:
    validate_ancestor_contracts     :300       validate_ancestor_contracts(pipeline, path)
      _child_contract               :236       _child_contract(get_at_path(pipeline, path[:depth]))
        _populate_child_contract    :196       _populate_child_contract()
```

and `_CPU_ONLY_SLOTS` is scanned unconditionally at `:307-336`, **not** gated on
`strict`, which is what makes the root-`meas` refusal fire from production. The
placement loop runs on `hits` and the slot loop runs regardless of whether `hits`
is empty — that ordering is correct and non-obvious, and it is what makes
`test_a_gpu_detector_in_the_ROOT_meas_slot_is_refused` a real test rather than a
restatement.

**Nothing reaches a staged run around it.** `uses_staged_gpu_strategy`
(`_cli_execution_strategies.py:1334-1346`) is the sole gate into both staged
strategies (`:1411` local, `:1423` SLURM) and its first act after the
`measure_only` short-circuit is `pipeline_requires_gpu`. I checked all four
callers for a surrounding swallow: three are uncaught, one is **F2**.

This is the thing the change most clearly gets right. The refusal lives in
`find_gpu_detectors`, which is what `pipeline_requires_gpu` calls, rather than in
a prefix builder that runs after routing. The earlier loss mode — the rule
existing in prose, in a lookup table and in tests, but in no executed code path —
is **structurally excluded rather than merely avoided**, and the four mutants say
so by execution rather than by my reading of the call chain.

The one caveat is **F1**: the chain is correct, and its last link
(`_populate_child_contract`) is not thread-safe, so under concurrency it can
return the *wrong* answer while every link is intact. Reachability and
correctness are different questions and only the first is settled here.

---

## F11 — The two recorded gaps: recording is adequate; one docstring sentence overclaims

*(Lead's question 6.)*

**`strict=True` has no production caller.** Adequate to record. The branch is
exercised by `test_two_gpu_detectors_anywhere_are_refused`, the docstring says it
is pending and names its future caller (`_cli_validation.py:255-264`), and Task 5
— landing now — is that caller. Not blocking. The real cost is stated honestly:
until Task 5, two detectors nested in a composite are *catchable but not caught*,
because `split_pipeline_at_gpu` keeps its own top-level scan.

**A GpuDetector in a nested pipeline's `meas`/`post`/`filters`/`model`.**
Recording is adequate, and the instrument chosen is right: a non-strict `xfail`
asserting the *desired* refusal
(`test_gpu_detection_tree_wide.py:151-191`) XPASSes when someone fixes it,
whereas a characterisation test would fail on the fix and teach people to delete
tests. The reasoning in that marker is the best-argued thing in the diff.

**One sentence overclaims.** `_operation_tree.py:68-70`:

> That shape is not reachable from the GUI builder and **has no known user**, so
> it is out of scope here

"Has no known user" is a claim about the user population, and nothing in the
change establishes it — what was established is that the reviewers did not
construct one. Say "we know of no pipeline in this repository or its fixtures
that has this shape", which is checkable, or drop the clause; the
"not reachable from the GUI builder" half is the load-bearing part and is fine.

---

## F12 — Can the new tests fail? Mutation results

*(Lead's question 1 — the primary question.)*

**Answer: yes, with two exceptions, and both exceptions are `substitute_at_path`
(F6).** 13 mutations, run against the three new test files in the frozen
`014d8cf9` tree. Baseline: `30 passed, 1 xfailed`.

| Mutation | File | Result | Verdict |
|---|---|---|---|
| M1 walker list index → always `0` | `_operation_tree.py` | 4 failed, 26 passed | KILLED |
| M2 walker does not extend the path | `_operation_tree.py` | 14 failed, 16 passed | KILLED |
| M3 ancestor-contract check removed | `_cli_validation.py` | 3 failed, 27 passed | KILLED |
| M4 CPU-only-slot scan removed | `_cli_validation.py` | 2 failed, 28 passed | KILLED |
| M5 `_child_contract` never refuses | `_cli_validation.py` | 3 failed, 27 passed | KILLED |
| M6 `apply_child` pushes a constant segment | `_provenance.py` | 7 failed, 23 passed | KILLED |
| M7 composite enumerates only non-`None` slots | `_composite_detector.py` | 1 failed, 29 passed | KILLED |
| M8 `apply_child` forwards `reset=None` verbatim | `_provenance.py` | 1 failed, 29 passed | KILLED |
| **M9 substitute drops `_provenance_pipeline`** | `_operation_tree.py` | **30 passed, 1 xfailed** | **SURVIVED** |
| **M10 substitute drops the non-`ImagePipeline` guard** | `_operation_tree.py` | **30 passed, 1 xfailed** | **SURVIVED** |
| M11 `_child_contract` skips lazy population | `_cli_validation.py` | 3 failed, 27 passed | KILLED |
| M12 strict multi-detector threshold off by one | `_cli_validation.py` | 1 failed, 29 passed | KILLED |
| M13 `pipeline_requires_gpu` back to top-level-only | `_cli_validation.py` | 8 failed, 22 passed | KILLED |

**11/13 killed.** Restore verified twice — by the harness (`ALL RESTORED`) and by
the lead's own sha256 snapshot taken before the run, with `git status --porcelain`
empty afterwards.

**The four that matter most, and what each proves.**

- **M13** is the regression test for the bug this change exists to fix: reverting
  `pipeline_requires_gpu` to the old top-level-only scan kills 8 tests. The
  nesting blind spot cannot silently return.
- **M3, M5 and M11** together answer the lead's question 2 by execution, not by
  reading the call chain. M3 removes the ancestor check, M5 makes
  `_child_contract` never refuse, M11 stops it populating the table — each kills
  3 tests. **The "composition primitives only" narrowing is genuinely on the path
  `pipeline_requires_gpu` takes.** The failure mode that lost it last time — a
  rule living in a table and in prose but in no executed code path — is now
  excluded by three independent mutants.
- **M4** kills 2: the CPU-only-slot refusal fires from production, not only under
  `strict=True`. That was `plan-review.md` BLOCKER #5, and it is fixed.
- **M7** kills exactly 1 — `test_an_empty_slot_does_not_shift_its_siblings_branch_index`,
  and nothing else. That test is the sole witness for the decision to `enumerate`
  over the whole `ops` list including skipped `None` slots. Single-witness
  coverage of a real decision is fine; it is worth knowing it is single.

**One note on the harness itself**, because it bears on how much the table is
worth. The first attempt aborted with `FileNotFoundError` creating a backup
directory under a bare `/scratch/<user>` path, which is per-job on this cluster
and not writable at the user level. It failed **at the backup step, before any
mutation was applied** — the precondition check doing its job rather than a near
miss. The lead verified no file had been touched, repointed the backup root, and
re-ran unchanged.

### Tests that still pass with their subject broken

None of these is a defect — a suite needs negative controls and type-contract
assertions. They are listed because each is the kind of test that gets counted
as coverage of something it does not cover.

- **`test_substitute_replaces_only_the_addressed_node`** (`:62-72`) and
  **`test_substitute_at_depth_two`** (`:75-87`). **Confirmed by M9 and M10
  surviving:** neither asserts anything about `qc`, `plots`, `name`,
  `_provenance_pipeline`, or the `ImagePipelineCore` guard. See **F6** for the
  two tests that would fix this.
- **`test_every_path_segment_is_a_non_empty_string`**
  (`test_operation_tree.py:40-46`). Necessarily survives M1: `"ops[0]"` is still a
  non-empty string when every list entry is indexed `0`. The arithmetic supports
  this — M1 killed exactly 4, and exactly four tests *must* fail under it
  (`test_walk_yields_list_entries_as_bracket_indexed_strings`,
  `test_find_operations_locates_a_nested_type`,
  `test_get_at_path_round_trips_with_walk`, and
  `test_the_recorded_paths_are_the_walker_paths`; the other provenance tests read
  the container's own f-string segment, which M1 does not touch). It asserts the
  type contract it says it asserts and adds nothing to index correctness.
- **`test_the_child_contract_table_holds_exactly_the_two_composites`**
  (`test_gpu_detection_tree_wide.py:222-233`). Necessarily survives M11, because
  it calls `_populate_child_contract()` itself before asserting. M11 still killed
  3 — the tests that reach the table *through production* — which is the right
  outcome and shows where the real coverage lives. This test is honestly labelled
  ("Coverage is asserted on the TABLE, not by enumerating the tree"); it just
  proves nothing about reachability, and the module docstring names
  unreachability as the defect the file exists to prevent.
- **`test_a_cpu_only_pipeline_is_still_false`** (`:82-90`) and
  **`test_a_domain_detector_with_no_gpu_detector_is_untouched`** (`:309-321`).
  Negative controls: they pass under any breakage that returns *fewer* hits,
  including a walker that returns none at all — they survive M13, which killed 8
  others. Correct to have; not evidence.

### Tests that carry real weight

`test_a_gpu_detector_in_the_ROOT_meas_slot_is_refused`,
`test_a_ROOT_meas_slot_gpu_detector_does_not_route_to_the_cpu_strategy`,
`test_a_gpu_detector_inside_a_domain_detector_is_refused` and
`test_the_domain_detector_refusal_names_the_offending_class` all drive
`pipeline_requires_gpu` / `uses_staged_gpu_strategy` — the production entry
points — rather than `find_gpu_detectors(strict=True)`. That is the single
correction that distinguishes this suite from the one `plan-review.md` #5 and
`plan-a-review-2.md` B1 rejected, and it is applied consistently.

`test_an_empty_slot_does_not_shift_its_siblings_branch_index`
(`test_provenance_step_descent.py:86-103`) is the sharpest test in the diff: it
pins the one decision (`enumerate` over the whole `ops` list including skipped
`None` slots) where the container and the walker could silently disagree, and
nothing else would catch it.

### Provenance of the test files

`test_operation_tree.py` and the first three tests of
`test_provenance_step_descent.py` are **verbatim the plan's Step-1 blocks**. The
tests that are *not* from the plan — the Task-3 production-entry-point tests, the
`apply_child` contract tests (`:148-206`), the empty-slot test, and the
walker/journal identity test — are the strong ones. So the suite's weaknesses are
inherited from the plan and its strengths were added during implementation, which
is the right direction but means the plan's test blocks should not be treated as
a floor next time.

---

## F13 — Gate coverage: the one shard that could exercise `NapariPipelineViewer` could not run

**Note, not a defect.** Gate results from the frozen `014d8cf9` array, supplied
by the lead:

| Shards | Result |
|---|---|
| `tests/unit/cli` + `tests/integration/cli`, 6 shards | all COMPLETED, all green |
| `tests/unit/core` + `detect` + `enhance` + `sdk_` + `abc_`, 6 shards | 5 green, **1 red: 14 failed, 647 passed** |
| the seven `pipeline_step_path` files | 89 passed |
| `tests/unit/tune` | 21 failed / 927 passed / 116 skipped — identical name-for-name to baseline after Task 2 (`ba0db2d9`) |
| `mypy src/phenotypic` | **434 errors in 127 files** vs. a **435 in 127** baseline at `5aaeeb77` — one fewer |

**The 14 reds are pre-existing and proven so**, not assumed: the same two files
(`test_napari_pipeline_viewer.py`, `test_label_editor_widget.py`) give
`14 failed, 25 passed` at `5aaeeb77`, same count and same names, before any of
this change. Neither venv has the `napari` extra installed, which root
`CLAUDE.md` requires for those tests (`--group test-qt --extra napari`). So
nothing in Phase 0–2 regressed there — **and that shard must not be recorded as
green either.**

**Why this is worth a numbered entry rather than a footnote.** Task 1's single
most consequential deviation from the plan is keying `iter_child_operations` on
`ImagePipelineCore` rather than `ImagePipeline`, and the commit justifies it by
name:

> `_child` keys on `ImagePipelineCore`, not `ImagePipeline` … `ops` is typed over
> `ImagePipelineCore` and **`NapariPipelineViewer` is a second concrete
> subclass**.

`NapariPipelineViewer` is the reason the deviation exists, and it lives in
precisely the part of the suite the gate could not execute. So the change's
handling of the second `ImagePipelineCore` subclass is currently supported by
static reasoning alone. Related: `substitute_at_path`'s `TypeError` guard
(**F6(a)**) exists to refuse exactly that class, and has no test either. Neither
is likely to be wrong; both are unwitnessed, and the two gaps are the same gap.

If the napari extra can be installed in a gate tree, those two files are the
cheapest coverage available for the one deviation with no other evidence behind
it.

---

## Sections that are clean

- **Question 4 (blast radius).** Clean — see **F9**. No production reader of
  `pipeline_step_path` exists to break.
- **Question 5 (the `reset` retraction).** Correct, both citations verified — see
  **F8**.
- **Commit messages vs. code.** The lead asked for decisions recorded in a commit
  message but contradicted by the code. I found one, and it is **F4**'s
  "Scope held: center_detector records no step path (a measurement's nested
  operation is a private probe)" — true of the measurer, not of the detector
  field the sentence is actually about. Everything else I checked holds: the
  tree-wide scan, the refusal living in `find_gpu_detectors`, the four (not two)
  callers of `pipeline_requires_gpu`, the lazy table, the `ImagePipelineCore`
  keying, and the `reset` retraction are all as described.
- **Question 1 (can the tests fail).** Substantially yes — 11 of 13 mutants
  killed, and the two survivors are confined to one function. See **F12**.

---

## Recommended actions, in order

1. **F1** — make `_populate_child_contract` publish the table in one binding
   (`dict.update` from a literal, or an `lru_cache`d accessor returning a
   `MappingProxyType`). One line. Blocker, confirmed by execution, and the only
   finding here that can produce a wrong answer for a *correct* pipeline.
2. **F4** — route `center_detector` and `background_subtractor` through
   `apply_child`. Three lines, and it makes `_two_k_filamentous_detector.py:148-151`
   shorter. Correct the Task-4 commit message's `center_detector` conflation while
   you are there.
3. **F4 (spec/plan)** — rewrite Task 9's invariant as *resolvability*
   (`get_at_path` does not raise for any recorded path) rather than set equality
   with the walker. Set equality is already false for any pipeline with an
   unfilled `None` slot, so it is the wrong invariant today, not merely fragile
   later.
4. **F6** — add the two tests in the shapes given. Note the `_provenance_pipeline`
   trap: the obvious version of that test compares `None is None` and leaves M9
   alive.
5. **F2** — narrow the GUI's `except (OSError, ValueError, TypeError)` to let
   `UnstageableGpuDetectorError` through, and surface its message on the existing
   warning-toast path.
6. **F3** — split once in `StagedSlurmStrategy.execute` before creating any
   scheduler state, as `StagedGpuStrategy` already does. Task 5 closes the live
   failure; this keeps the asymmetry from outliving it.
7. **F13** — try the `napari` extra in a gate tree and re-run that shard. It is
   the only evidence available for the `ImagePipelineCore` deviation that
   **F6**'s local-subclass test cannot supply.
8. **F11** — replace "has no known user" in `_operation_tree.py:68-70` with a
   checkable claim.

---

## What I could not verify, and why

Every probe and every test run below was executed by the lead against the frozen
`/bigdata/exfab/anguy344/gate-trees/014d8cf9` checkout, not by me: subagent
commands with side effects go through the lead by protocol, and every form of
these probes mutates process state or writes `__pycache__`. I read the output;
I did not produce it.

**Everything is now resolved by execution except the two items below.** F1
(CONFIRMED, P1), F4 (CONFIRMED and its mechanism corrected, P4), F5 (**REFUTED**,
P2), F6 (CONFIRMED, M9+M10 survived), F7 (CONFIRMED, P3), F10 (CONFIRMED, M3/M5/
M11/M13), F12 (CONFIRMED, 11/13 killed), F13 (gate output).

**Where I was wrong, stated plainly:**

- **F5** was my finding and the probe refuted it. A `GpuDetector` cannot be a
  direct entry of any CPU-only slot; pydantic rejects all four.
- **F4's mechanism** was wrong in my first draft. I predicted a path that
  collided with the parent's; the actual recorded path drops a middle segment and
  resolves to `KeyError`, which is a worse defect and changed my recommendation
  from "either/or" to "descend it".

**Still not independently verified:**

- **The gate figures and the mutation table are reported output, not my own
  measurement.** The lead executed every probe, the harness and the gate array;
  I read the results. That is the correct division of labour under the
  command round-trip protocol, but it means my confidence in those numbers is the
  lead's confidence, not a second independent measurement, and this report should
  not be read as supplying one. Where I could derive a result from the numbers
  rather than accept it — M1's kill count against the four tests that must fail
  under it — I have shown the derivation.
- **No real output tree was inspected.** **F9** establishes that nothing reads
  `pipeline_step_path` in production, so a tree holding stores written on both
  sides of this commit is inert. I confirmed that from the source, not from a
  tree on disk.
- **`NapariPipelineViewer`'s own behaviour** under the `ImagePipelineCore`
  deviation is untested in both senses (**F6**, **F13**) and I could not test it
  either — the shard is unrunnable without the `napari` extra. The recommended
  test in **F6** deliberately sidesteps this with a local subclass, which pins the
  type relation but not the real class.

**Not verified, and out of scope by choice:** whether any *existing* OME-Zarr
store in a live output tree carries journals written before this commit. **F9**
establishes that nothing reads `pipeline_step_path` in production, so a mixed
tree is inert, but I inspected no real tree to confirm the mix exists.
