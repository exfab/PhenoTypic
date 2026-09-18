# Plan B review — staging via a declared phase protocol

**Subject:** `docs/superpowers/plans/2026-09-15-nested-gpu-staging/plan-b-phase-protocol.md`
**Reviewer:** plan-reviewer (first review; Plan B has no spec)
**Date:** 2026-09-15
**Worktree:** `/bigdata/exfab/anguy344/PhenoTypic/.claude/worktrees/nested-gpu-staging`
**Probe:** all pydantic findings below are confirmed by an executed probe
(pydantic 2.12.5 / Python 3.12.10); output reproduced in **Verification
results**. One blocker in the pre-probe draft was withdrawn on that evidence —
see V1b.

---

## Summary verdict

**NEEDS REVISION.** The central idea is sound and the single most load-bearing
assumption survives intact: an underscore-prefixed `ClassVar` on a pydantic v2
`BaseModel` is legal, works with `from __future__ import annotations`, survives
an *un-annotated* override in a subclass, and the codebase already uses the
pattern. `PrivateAttr` really is excluded from `model_dump(mode="json")` and
from `model_fields`, so the argument that phase state stays out of the
provenance `parameters` and out of the `OperationField` walker is correct.
`GpuDetector`'s hook surface really does split three ways, and the staged
workers already run exactly that split across two processes today. The
declaration syntax the plan proposes needs no change.

But the plan is incomplete at precisely the shape it exists to serve. The
production driver is a `Sam2` inside `CompositeDetector.ops`, and
`CompositeDetector._operate` does not merely run its children — it runs each on
a *copy* and then **combines** all their objmaps. `_child_order` describes how
siblings relate to each other; it says nothing about the parent's own work that
consumes them. Plan A never had to answer this because it re-executes the
parent with a stub substituted at the GPU child's path. Plan B forbids
re-execution, so it must answer it, and does not. Three further holes compound
that: the device tag is a class constant while for every container the device
is an instance property; driving phases directly bypasses `apply()`, which is
where provenance, the step path and the error wrapper live, with nothing
proposed to replace them; and Task B4's three-file scope omits the four modules
that hard-code a three-stage state machine on disk.

Seven tasks is not a smaller plan than Plan A's sixteen for a harder problem —
it is the same problem with the hard half unwritten. The approach remains
worth pursuing; it needs a spec that settles the nine questions at the end
before any code is written.

**Severity map after the probe:** three blockers (B-1, B-2, B-3 below — the
pre-probe draft's fourth, on un-annotated declarations, is withdrawn), eight
concerns, eight suggestions.

---

## Validated aspects ✓

**V1 — Underscore `ClassVar` on a pydantic v2 model is legal, including under
stringified annotations.** `pydantic/_internal/_model_construction.py:463-466`
skips private-attribute collection for a sunder name whose annotation is a
ClassVar, and `_typing_extra.py:100-130` (`is_classvar_annotation`) matches the
**string** form `'ClassVar[...]'` via `_classvar_re`, which is what
`from __future__ import annotations` produces. The codebase already relies on
this: `FilamentousFungiDetector` carries eight of them
(`src/phenotypic/detect/_filamentous_fungi_detector.py:254-270`,
e.g. `_GAUSS_SIGMA_PER_R: ClassVar[float] = 1.2`), with a comment explaining
that `ClassVar` keeps them out of `model_fields`. Confirmed by probe Q1/Q2/Q4:
both the fully-subscripted and the bare `ClassVar` forms work, on the class and
on an instance. Probe Q7 additionally shows pydantic **refuses** an instance
assignment to a ClassVar name (`AttributeError: '_phases' is a ClassVar ... and
cannot be set on an instance`), so an engine bug cannot silently clobber an
operation's declaration.

**V1b — An *un-annotated* override in a subclass is repaired, provided a base
declares the ClassVar. The pre-probe draft called this a blocker; it is not.**
Read in isolation, `_model_construction.py:463-466` says a leading-underscore
attribute without a ClassVar annotation becomes a `PrivateAttr` and is deleted
from the namespace, and `base_class_vars` is consulted only afterwards at
`:467`. That reading is incomplete. `set_model_fields`
(`_model_construction.py:571-581`) **undoes** it:

```python
for k in class_vars:
    # Class vars should not be private attributes
    #     We remove them _here_ and not earlier because we rely on inspecting the class ...
    value = cls.__private_attributes__.pop(k, None)
    if value is not None and value.default is not PydanticUndefined:
        setattr(cls, k, value.default)
```

and `class_vars` comes from `collect_model_fields`, which reads
`get_model_type_hints` — annotations **from the whole MRO**
(`_typing_extra.py:308-313`, `_fields.py:265, 280`). So an inherited
`_phases: ClassVar` annotation makes the subclass's bare `_phases = (...)` a
class var again, and the raw tuple is restored.

Probe Q3 confirms it directly: `SloppyChild(Base)` writing
`_phases = (("_prep","cpu"), ("_typo_here","gpu"))` with no annotation yields
`type(...) == 'tuple'`, `'_phases' in __class_vars__ == True`,
`'_phases' in __private_attributes__ == False`. **Task B1's tests are written
correctly as they stand**, and an author who forgets `: ClassVar` is not
silently punished.

**The repair also holds at depth**, which matters because operations here are
three and four levels deep (`GridObjectDetector` → `ObjectDetector` →
`ImageOperation` → `BaseOperation`). A targeted follow-up probe on the real
hierarchy — `PhasedBase(ObjectDetector)` declaring `_phases: ClassVar[...]`,
then `Child` and `GrandChild` overriding it with **no** annotation at one and
two levels, against a `NoBase` control with no declaring ancestor:

```
PhasedBase         type=tuple            classvar=1  priv=0   (('_operate','cpu'),)
Child              type=tuple            classvar=1  priv=0   (('_prep','cpu'),('_infer','gpu'))
GrandChild         type=tuple            classvar=1  priv=0   (('_a','cpu'),)
NoBase (control)   type=ModelPrivateAttr classvar=0  priv=1   ModelPrivateAttr(default=...)
```

The control is what makes this readable: the mechanism is inheritance of the
*annotation*, not the depth of the override.

The one caveat is that the repair is *entirely* contingent on some ancestor
declaring the annotation — probe Q8's `RealSloppy(ObjectDetector)` and the
`NoBase` control both come back as `'ModelPrivateAttr'`. Since Plan B puts the
declaration on the base, that is the protected case, and the residual is only
an operation that somehow reaches the scan without a declaring ancestor. Kept
as **m-9**, not as a blocker, and M-1's placement decision is what keeps it
true.

**V2 — `PrivateAttr` exclusion is real, and the argument built on it is
correct.** Private attributes live in `__pydantic_private__`
(`pydantic/main.py:999-1010`) and are never part of the core schema, so they
appear in neither `model_dump(mode="json")` nor `model_fields`. `OperationField`
is a genuine pydantic field annotation (`src/phenotypic/sdk_/typing_.py:250-284,
400`), so the walker inspects fields and cannot reach a private attr. The plan's
claim that this avoids the bug class `ReplayDetector`'s `NdArrayField` would have
reintroduced — a per-image array landing in the provenance `parameters` — holds.
The codebase already uses private attrs for exactly this kind of non-serializable
state (`_image_operation.py:389-394`, `_base_operation.py:180-181`).

**V3 — `GpuDetector` really does map to three phases without losing
behaviour.** `src/phenotypic/abc_/_gpu_detector.py:249-262`:

```
array = getattr(image, self.input_layer)[:]
sample  = self._preprocess(array)
batch   = self._collate([sample])
results = self._infer_batch(batch)
self._write_object_output(image, results[0])
```

and `_cli_staged_workers.py:334-342` / `:461` already run that exact split
across two processes today (Stage 2 does `_preprocess` → `_collate` →
`_infer_batch`; Stage 3 does `_write_object_output`). The
`(_phase_prep, cpu), (_phase_infer, gpu), (_phase_write, cpu)` mapping is
accurate for the current implementation. One caveat in m-4.

**V4 — The "no re-execution" argument and its arithmetic.** 71 s × 33,923
images = 2,408,533 s = 669 CPU-hours. Correct, and a real cost Plan A carries.

**V5 — The pipeline-caching collision is correctly identified.** Stage 2 today
is a serial loop over one shared `plan.gpu_detector` instance
(`_cli_staged_strategy.py:222-250`), so per-image `PrivateAttr` state on that
instance would carry from image to image. `clear_phase_state` at the start of
every image (B4 step 3) is necessary and the plan has it.

**V6 — `TwoKFilamentousDetector`'s refusal is correctly reasoned.** It carries
three independent operation fields — `branch_base`, `center_detector`,
`background_subtractor` (`_two_k_filamentous_detector.py:61,70,71`) — with
genuinely different input relationships, so no single `_child_order` is true.
Refusing it for that reason is better than Plan A's class-kind rule.

**V7 — Task B5's instance-state poisoning step (rebuild the op between phases
carrying only declared `_phase_state`) is the right gate** and is the strongest
idea in the plan. A phase split that silently loses undeclared state is exactly
the failure mode, and this is the only construction that catches it.

---

## Critical issues 🚨

### B-1 — No rule for how a parent's own phases compose with, and consume, its children's

**What is wrong.** The driver pipeline is a `Sam2` inside
`CompositeDetector.ops` (spec `design.md:5-7`). `CompositeDetector._operate`
(`src/phenotypic/detect/_composite_detector.py:107-160`) is not "run the
children and you are done":

```python
for detector in self.ops:
    ...
    detected_image = detector.apply(image, inplace=False)   # :132 / :139 — a COPY
    objmaps.append(detected_image.objmap[:].astype(bool))
# then union / intersection / overlap over ALL objmaps  (:142-160)
```

Three facts the declaration cannot express:

1. Each branch runs against a **copy** of the parent's input, so N branch
   images are live, not one.
2. There is a **combine** step that is the parent's own CPU work, that runs
   after every branch, and that consumes *all* of their outputs
   simultaneously.
3. `CompositeDetector` would declare the default
   `_phases = (("_operate","cpu"),)` — and `_operate` is the thing that runs
   the children. Declaring `_child_order = "parallel"` alongside it does not
   say which part of `_operate` is the combine.

`FilamentousFungiDetector` has the same shape: `_operate:393-408` runs the
child, and `:410-545` consumes its two outputs (`inoculum_objmask` at :400,
`inoculum_img.objmap[:]` at :464).

**Why it matters.** Plan A never had to answer this. Its Stage 3 **re-executes**
the parent's `_operate` with a `ReplayDetector` stub substituted at the GPU
child's path, so the copy semantics and the combine happen by themselves. Plan
B's headline is that re-execution is gone — which makes the parent's internal
consumption a question it must answer, and it does not.

This also corrects a claim in the plan's own comparison table.
`_child_order` is **not** carrying less information than `_CHILD_CONTRACT`:
Plan A's table is `{CompositeDetector: "same", CompositeEnhance: "same"}`
(`plan-a:1110-1113`) — the same `{parallel, sequence}` distinction, just stored
externally. The problem is that Plan B needs *strictly more* information than
Plan A did, and proposes the same amount.

**Direction — what the rule has to say.** `_child_order` answers one question
(how do siblings relate to each other?). A parent that does its own work needs
three more answered, and any workable design has to name all four:

1. **Where do the children run?** `_child_order` is not a position. The
   declaration must place the child block relative to the parent's own phases —
   e.g. `_phases = (("_prep","cpu"), CHILDREN, ("_combine","cpu"))`, with a
   sentinel in the phase tuple rather than a separate attribute, so the
   position is unambiguous and a parent with children but no sentinel is a
   definition-time error rather than a silent "children run first".
2. **What does each child receive?** This is the part `_child_order` already
   covers (`"parallel"` = the parent's own input; `"sequence"` = the previous
   sibling's output), and it is enough — but only because it is answered
   *per slot*. `FilamentousFungiDetector` and `TwoKFilamentousDetector` have
   named slots with different answers, so the natural shape is per-field, not
   per-class: `_child_order = {"inoculum_detector": "own_input"}`. A class-wide
   scalar is why `TwoKFilamentousDetector` has to be refused wholesale rather
   than partly supported.
3. **What does the parent get back, and in what container?** This is the
   genuinely new question and the reason B-1 is a blocker. Today each child is
   applied with `inplace=False` and the parent reads one array off the returned
   image (`_composite_detector.py:132-140`: `detected_image.objmap[:]`). A
   phase-driven engine has to hand `_combine` the same thing across a process
   boundary, so the spec must fix (a) the **type** of the handoff — a list of
   images, or a list of named arrays, and which layers of each are live; (b)
   the **arity** — `"parallel"` yields N, `"sequence"` yields one; and (c) the
   **signature** the parent's consuming phase takes, which is no longer
   `_operate(self, image)`. Something like
   `_combine(self, image, children: Sequence[ChildResult]) -> Image` — at which
   point "phases are just method names with a device tag" is no longer true,
   and the protocol has two phase kinds.
4. **Where does the handoff live between rounds?** N branch results are exactly
   the "large intermediates" M-4 says must be layers rather than ferried state:
   two branches of a 3140×5094 plate are ~32 MB of objmap alone, before
   `detect_mat`. So the answer to (3) probably has to be *layer names in the
   store*, not in-memory arrays — which then collides with the rule that Stage 2
   never writes the store, and needs a per-branch namespace so two branches
   cannot overwrite each other.

The cheapest honest alternative is worth costing against all of that: **keep
Plan A's re-execution for the parent only.** Let a container declare that its
own `_operate` is replayed with its GPU descendants stubbed, while its children
are still phase-split. That gives up "each phase runs exactly once" for the
parent's own body — which for `CompositeDetector` is a mask combine, not a
`FocusEdgePhase` — while keeping the 669 CPU-hour win, which comes from not
re-running the *branch prefix*, not from not re-running the combine. It also
lets Plan B ship without answering (3) and (4) at all. If the spec can show the
parent bodies in scope are all cheap, this is a much smaller change than a
two-kind phase protocol.

---

### B-2 — The device tag is a `ClassVar`, but for every container the device is an instance property

**What is wrong.** `FilamentousFungiDetector.inoculum_detector` is
`Union[OperationField, None] = None` (`:276`), defaulting to a pure-CPU
`InoculumDetector` + `KeepSectionLargest` pipeline (`:238-250`). Task B6
declares, at class level:

```python
_phases: ClassVar = (("_prep", "cpu"), ("_gpu_detection", "gpu"), ("_post_gpu_work", "cpu"))
```

So **every** `FilamentousFungiDetector` — including every existing all-CPU one
— advertises a GPU phase. `CompositeDetector` is identical: `ops` is a list
field (`_composite_detector.py:86-88`) whose contents decide whether any GPU
work exists.

**Why it matters.** This is not open question 3's "near-empty array submission".
`pipeline_requires_gpu` (`src/phenotypic/_cli/_cli_validation.py:135-147`) is
what selects the partition and `--gres`. On this cluster a falsely GPU-tagged
pipeline routes an entire 33,923-image run to `exfab` (one node; account cap 32
CPU / 256 GB) or `short_gpu` (2 h walltime cap) instead of `batch`/`intel`/
`epyc`. That is a queueing decision for the whole run, made from a class
constant that cannot see the instance.

It also makes B6's stated rationale incoherent. B6 says the split is natural
"because the inoculum detection is essentially the first thing it does, so the
CPU work before the GPU phase is nearly nothing." But in the nested case the
inoculum detection **is** the GPU work — it is the child. And a `gpu`-tagged
parent phase that merely calls `child.apply()` runs the child's `_preprocess`,
`_collate` **and** `_write_object_output` on the GPU node, which is the cost
Plan B claims to abolish. Conversely, if the split really is FFD's own
PHASE 1 / PHASE 2 boundary, there is no GPU work in it at all —
`_operate` is entirely CPU.

**Direction.** Make `phases_of(op)` an instance-level function permitted to
consult fields (the plan already spells it `phases_of(op)`, so let the ClassVar
be the default and let a container override a `_phases_for_instance()`), or
forbid containers from tagging their own phases and derive the device purely
from flattened children. Either way, drop the claim that the phase structure is
"readable from the pipeline JSON without constructing anything" — see m-5.

---

### B-3 — Phases bypass `apply()`, which is where provenance, the step path and the error wrapper live

**What is wrong.** Provenance is emitted at the `apply()` boundary, not at
`_operate`. `ImageOperation.__pydantic_init_subclass__`
(`src/phenotypic/abc_/_image_operation.py:396-416`) rewraps every subclass's
resolved `apply` with `wrap_image_operation_apply`, and
`src/phenotypic/_core/_provenance.py:549-629` is what appends one journal entry
per operation, carrying its duration and its `pipeline_step_path` read from the
`_pipeline_step_path` ContextVar (`:623`). Plan A's Task 4 pushes
`pipeline_step` around each **child `.apply()`** call, and Task 9 asserts that
the walker's `gpu_path` equals the path that shows up in the journal from a
monolithic `pipe.apply()` (`plan-a:1917-1931`).

A phase driver calling `getattr(op, "_phase_infer")(image)` enters none of it —
no wrapper, no ContextVar push, no `_apply_to_single_image` inplace/copy
handling (`_image_operation.py:466-496`), no `RuntimeError` context wrapper.
The current Stage 3 hand-rolls the missing record precisely because the
detector's `apply` never runs (`_cli_staged_workers.py:461-469`):

```python
append_operation_provenance(
    image, plan.gpu_detector,
    duration_seconds=float(token["detector_duration_seconds"]) + merge_duration,
    pipeline_step_path=[plan.gpu_key],
)
```

Plan B multiplies that one hand-rolled record to every phased operation, across
N processes, and is silent on it.

**Why it matters.** This directly falsifies two carry-over claims. Task 4
("container ops push a per-branch `pipeline_step`") is listed as "**reused
unchanged**" and Task 9 ("`gpu_path == pipeline_step_path`") as "reused as
phase-path == step-path". Both mechanisms live inside `apply`. Under Plan B
neither fires for a flattened phase, so both are superseded, not reused.

There is a second layer. `_application_owner_depth` is also a **ContextVar**,
and `src/phenotypic/measure/CLAUDE.md:19-30` documents that its value decides
whether a nested `apply` *joins* or *appends* an application — with the depth-0
path (what `stage3_merge_measure_core` establishes) raising
`ValueError: cannot start a new provenance application before the last ends`
when it is wrong. With K rounds in K processes, which round opens the
application, which closes it, and what `truncate_provenance_to_retry_base`
means are all open. Task B5 step 4 files this as "owner-depth-0 coverage",
i.e. as testing. It is a design decision first.

**Direction.** The spec must state: who enters the `pipeline_step` contexts
along a flattened phase's path; who emits the single journal entry for an
operation whose phases ran in three processes; how durations are summed; and
how ownership depth is established per round. Until that exists, "the round
chain is the flattened phase list" describes half an engine.

---

## Concerns ⚠️

### M-1 — `_phases` cannot live on `ImageOperation`, and the default `("_operate","cpu")` is a type lie across the hierarchy

Task B1 says *Modify: `src/phenotypic/abc_/_image_operation.py`*. But the
things the scan must see are not all `ImageOperation`s:

| class | base | file:line |
|---|---|---|
| `ImagePipeline` → `ImagePipelineCore` | `BaseOperation, LazyWidgetMixin` | `_core/_pipeline_parts/_image_pipeline_core.py:143` |
| `MeasureFeatures` | `BaseOperation` | `abc_/_measure_features.py:49` |
| `PostMeasurement` | `BaseOperation` | `abc_/_post_measurement.py:10` |
| `ImageOperation` | `BaseOperation, LazyWidgetMixin` | `abc_/_image_operation.py:19` |

Pipelines and measurers would carry no `_phases` at all — and Plan A's Task 3
explicitly refuses a `GpuDetector` in the `meas` slot, so the scan has to see
measurers. Probe Q8 confirms `issubclass(ImagePipeline, ImageOperation)` is
`False` while `issubclass(ImagePipeline, BaseOperation)` is `True`.

This placement is also what V1b's repair depends on. A hierarchy that does not
inherit the ClassVar annotation gets the un-repaired behaviour: probe Q8's
`RealSloppy(ObjectDetector)` with a bare `_phases = (("_operate","cpu"),)`
comes back as `'ModelPrivateAttr'`. So if `_phases` lands only on
`ImageOperation`, a `MeasureFeatures` or `PostMeasurement` subclass that writes
an un-annotated `_phases` is silently misdeclared — the exact footgun that V1b
withdraws for the `ImageOperation` branch.

Moving the ClassVars to `_base_operation.py` exposes the second half. `_operate`
means three incompatible things:

- `ImageOperation._operate(self, image) -> Image` (`_image_operation.py:455`)
- `MeasureFeatures._operate(image) -> pd.DataFrame` — **no `self`**
  (`_measure_features.py:471-472`)
- `PostMeasurement._operate(df) -> df` (`_post_measurement.py:53`)

A generic `run_phase_round` doing `getattr(op, method)(image)` hands an `Image`
to a `PostMeasurement`. So "the default keeps every existing operation working
unchanged" is true only while the default is never *driven* — and deciding
which ops are drivable reintroduces the kind-based special-casing the plan
claims to delete.

Worse: probe Q8 shows `hasattr(ImagePipeline, "_operate")` is **`False`** —
`ImagePipelineCore` has `apply` and `apply_with_intermediates`, never
`_operate`. So under the inherited default `_phases = (("_operate","cpu"),)`,
Task B1's own guard *"every phase method exists on the class"* would **reject
`ImagePipeline` itself** at class-definition time. The default is not a no-op
for the most important container in the codebase.

**Risk:** high. **Direction:** either scope `_phases` to `ImageOperation` and
give the scan a separate rule for `meas`/`post` slots, or give the three
hierarchies distinct default phase names.

### M-2 — Task B4's scope is three files; a K-round chain touches at least eight, including on-disk schemas

B4 lists `_cli_staged_workers.py`, `_cli_staged_slurm_worker.py`,
`_cli_staged_strategy.py`, plus `_gpu_detector.py`. The three-stage shape is
hard-coded in four modules it does not list:

- `src/phenotypic/_cli/_stages.py:6-15` — `StageTag = Literal["stage1","stage2","stage3"]`,
  `VALID_STAGE_TAGS`, and `validate_stage_tag` which **raises** on an unknown
  tag. Every event-log row carries one. A K-round chain needs a new tag scheme
  *and* back-compat for logs already on disk.
- `src/phenotypic/_cli/_cli_staged_resume.py:35-40` —
  `ResumeStage = Literal["stage1","stage2","stage3","complete"]` with a
  hardcoded rank map, plus `stage3_completion_exists`,
  `write_stage3_completion_marker`, `stages.stage3` in the per-image record,
  and an existing `migrate_legacy_stage3_markers`. That is a second on-disk
  schema needing a second migration.
- `src/phenotypic/_cli/_cli_staged_slurm.py:347-438, 616-657` — a three-script
  state machine (`stage1` chunks → one `stage2` script → `stage3` chunks →
  finalizer), `staged_resume_phase`, `stage1_index`/`stage3_index`, the
  `phenotypic-stage2` console entry point, and the epoch-fenced recovery ledger.
- `src/phenotypic/_cli/_cli_staged_controller.py` (389 lines) and
  `_cli_staged_orchestration.py` (743 lines) — unmentioned.

**Risk:** high — this is most of the plan's real cost, unbudgeted.

### M-3 — The three "cores" are not three instances of one function

B4 step 1 says the three cores "become the CPU/GPU/CPU instances of"
`run_phase_round`. They are not symmetric:

- `stage1_preprocess_core` — imread, provenance init/resume, `_retain_original`,
  **two** store publications, `set_retry_base_length`, `valid_staged_store`
  assertions.
- `stage2_detect_core` — read-only store load, raw `.npy` + token write,
  explicitly **never** writes the store.
- `stage3_merge_measure_core` — truncate provenance to the retry base, replay
  the raw array, post-ops, `measure`, overlays, `PlotCoordinator.emit_image`,
  re-promote the store *with measurements*, completion marker, consume token
  then raw.

A generic driver needs first-round / middle-round / last-round behaviours plus
a per-boundary persistence policy. And with a five-round chain — which B2's own
`test_parallel_children_get_one_round_each` asserts — there are **three**
intermediate persistences of a full image per image that do not exist today.
Where those land, in what format, how they are versioned and when they are
cleaned up is unstated, and at 33,923 images on GPFS it is not a rounding error.

**Risk:** high. **Direction:** cost the intermediate-persistence story in the
spec, and say which rounds write the store vs. a `branch/` artifact.

### M-4 — "Large intermediates are layers, not phase state" contradicts Task B6

Global Constraints: *"The engine caps the ferried payload; anything above it
must be written as an image layer."* B6 then declares `_inoculum_objmask`
(bool) and `_inoculum_objmap` (uint16) as phase state. At the production size
this change quotes (3140 × 5094 = 16.0 Mpx) that is 16 MB + 32 MB = **48 MB per
image per boundary**. The only concrete number written down anywhere is B3's
test refusing a 122 MB payload. The cap is undefined and the worked example is
within a factor of three of the one refusal on record.

**Risk:** medium. **Direction:** name the cap in the spec, and say explicitly
whether B6's two arrays are phase state or layers. If they are layers, B6's
`_phase_state` declaration is wrong as written.

### M-5 — B3's `test_unset_is_distinguishable_from_none` cannot pass against B6's declaration

B3 requires `capture_phase_state` to distinguish unset from explicit `None`.
B6 declares `_inoculum_objmask: np.ndarray | None = PrivateAttr(default=None)`.
A `PrivateAttr` **with** a default is populated into `__pydantic_private__` at
init, so a fresh instance and an explicitly-`None` instance are identical; only
`PrivateAttr()` with no default leaves the key absent (`pydantic/main.py:1006-1010`
raises `AttributeError` for a missing key). The two tasks contradict each other.

**Confirmed by probe Q5/Q6.** A fresh `Phased()` already has
`__pydantic_private__` keys `['_carried', '_not_carried']` (the two with
defaults) and `op._carried` reads `None`; `_carried_nodefault`, declared
`PrivateAttr()` with no default, is **absent** from the dict and raises
`AttributeError: 'Phased' object has no attribute '_carried_nodefault'`. So
"unset" is representable, but only by declaring the attr **without** a default —
which is the opposite of what B6 writes.

**Risk:** medium — it is a one-line fix, but it decides the `capture`/`restore`
API shape, and it propagates to every `_phase_state` declaration authors write.

### M-6 — Task 12 is not "reused unchanged"

Plan A's Task 12 is deliberately scoped to *process-mode layer semantics*, and
its second test `test_a_full_run_digest_is_UNCHANGED_by_the_bump`
(`plan-a:2158-2178`) exists specifically to avoid cold-starting in-flight
full/measure continuations on the cluster. Plan B changes **full-run** staging
semantics — signal layout, round count, per-image record schema. It therefore
needs an additional and *opposite* decision (invalidate full-run continuation),
which that test as written forbids.

**Risk:** high if missed — it is the same wrong-answer-on-resume class the task
was written to prevent, one level up.

### M-7 — The plan's own table and its own test disagree about N > 1

README: *"GPU detectors per pipeline: 1, with N > 1 additive."* B2's
`test_parallel_children_get_one_round_each` asserts
`["cpu","gpu","cpu","gpu","cpu"]` from two branches each containing a
`FakeGpuDetector` — that is N = 2, with two resident-model loads. Either the
test is wrong or the table is; as written the plan's flagship flattening test
exercises a case the plan says it does not support.

**Risk:** medium. Also note the model-residency story changes: two GPU rounds
means two model loads, which open question 4 acknowledges only in passing.

### M-8 — `object.__setattr__` works, but writes phase state to the wrong place

The brief asks whether the engine can read/write private attrs generically.
It can — but not the way the plan spells it.
`BaseModel._setattr_handler` (`pydantic/main.py:1050-1058`) routes a **declared**
private name into `__pydantic_private__`; plain `setattr(op, name, value)` is
correct and complete. Probe Q5 confirms: after `setattr`, the value is in
`__pydantic_private__` and **not** in `op.__dict__`.

`object.__setattr__` writes into the instance `__dict__` instead. Probe Q5
shows the value then lands in `__dict__` **as well as** the key already being
present in `__pydantic_private__` (it was, from the `default=None`) — i.e. the
private dict is never updated by the write, so it retains `None` while
`op._carried` reads the array through `__dict__`. Pydantic defines `__getattr__`
(not `__getattribute__`), so the `__dict__` entry wins on every read and the
divergence is invisible.

To be precise about what the probe did and did not show: it printed key
*presence*, not the value held in `__pydantic_private__`, so the stale-`None`
claim is inferred from `object.__setattr__`'s semantics rather than observed.
The probe did confirm the benign parts — `model_dump(mode="json")` stays clean
(`{'x': 1}`) and both `deepcopy` and `pickle` preserve the array either way.

**Risk:** low-to-medium (silent divergence between two storage locations, with
no observed failure yet). **Direction:** there is no reason to take the risk —
`restore_phase_state` uses `setattr`; `capture_phase_state` reads
`op.__pydantic_private__` directly (which also gives the unset-vs-`None`
distinction M-5 needs); `clear_phase_state` pops the keys.

---

## Suggestions for improvement 💡

- **m-1.** Payload dicts holding ndarrays cannot be compared with `==` / `!=`.
  Confirmed by probe Q9: `{'a': arr} == {'a': arr.copy()}` raises
  `ValueError: The truth value of an array with more than one element is
  ambiguous`. Two of B3's four tests
  (`test_unset_is_distinguishable_from_none`, `test_clear_resets_between_images`)
  compare payload dicts directly; they would pass today only because their
  values happen to be `None`, and would start raising the moment a real array is
  in the payload — which is the case the tests exist for. Give the payload a
  comparison helper, or compare keys and arrays separately.
- **m-2.** The README carry-over table accounts for 15 of 16 Plan A tasks.
  **Task 2** ("Consolidate `tune/`'s traversal onto the shared walker",
  `plan-a:404`) appears in neither column.
- **m-3.** "Roughly 70% of Plan A survives" is 10/15 = 67% by task count and
  ~64% by plan volume — the superseded set (5, 6, 7, 8, 13) is 842 of 2,340
  plan lines, and contains Task 5, the single largest task in the plan. The
  claim is fair; the framing understates which half is discarded.
- **m-4 — cross-image batching.** `supports_batching` is a declared capability
  and the `GpuDetector` docstring promises a batchable subclass can override
  `_infer_batch` with a true `(N, C, H, W)` forward "no engine changes needed".
  A per-image `_phase_infer` cannot batch across images. Today's Stage 2 also
  runs at batch size 1 (`_cli_staged_workers.py:339`), so this is not a
  regression — but B4 replaces the one function that could have grown a batch
  loop with a per-image generic driver. Say in the spec whether batching is
  being given up or deferred.
- **m-5.** "the phase structure is readable from the pipeline JSON without
  constructing anything" is false twice over. The JSON stores class names and
  parameters, so the class must be imported and resolved through the
  `phenotypic` registry (`_serializable_pipeline.py:627-678`, as Plan A Task 3's
  own note records), and `pipeline_requires_gpu` already does
  `ImagePipeline.from_json`. Per B-2, the device is instance-dependent anyway.
- **m-6.** "the bug this whole effort exists to fix stops being *possible*
  rather than being fixed" overstates, though less than the pre-probe draft
  claimed. Plan A Task 3's `isinstance` scan is already tree-wide and equally
  blind-spot-free, so both are tree scans and neither has the nesting blind
  spot. The difference is that `isinstance` derives the answer from the class
  hierarchy while the phase scan derives it from something an author writes
  down — and V1b shows that declaration is robust to the obvious slip, so the
  fragility gap is small. It is not zero: a wrong *device* tag, a phase method
  that exists but does the wrong work, and B-2's instance-dependent device are
  all misdeclarations `isinstance` cannot make. "More expressive, and about as
  reliable" is the accurate claim.
- **m-7.** Make the "one operation instance per image" rule an assertion, not a
  sentence (see the concurrency section).
- **m-8.** Consider whether `_phase_state` for a `GpuDetector` is redundant with
  the existing Stage-2 raw `.npy`. In the nested-FFD case the ferried
  `_inoculum_objmap` *is* the detector's output — the same array the
  `stage2_raw/` mechanism already persists. Two persistence paths for the same
  bytes is worth avoiding.
- **m-9 — the residual of the withdrawn blocker (see V1b).** pydantic's repair
  of an un-annotated override depends on an ancestor carrying the annotation.
  That is Plan B's own arrangement, so the case is protected — but it is
  protected by a mechanism two layers down in pydantic's internals
  (`_model_construction.py:571-581` undoing `:463-466`), not by anything in
  this codebase, and it is the kind of thing a pydantic minor release could
  reasonably change without it looking like a breaking change. Have
  `validate_phase_declaration` open with
  `isinstance(cls._phases, tuple)` (and the same for `_phase_state`), raising
  a message that names the missing `: ClassVar`. One line, runs once per class
  at definition time, and converts a silent misdeclaration into an import-time
  error in the one configuration where the repair does not fire. Cheap
  insurance; not a design change.

---

## Concurrency analysis 🔄

**Shared state inventory.**

| State | Where | Sharing |
|---|---|---|
| `plan.gpu_detector` instance | `_cli_staged_strategy.py:222-250` | **one instance, serial loop over all images in a shard** |
| Stage 1 / Stage 3 op instances | `_cli_staged_strategy.py:206, 341` — `joblib.Parallel(n_jobs=cfg.n_jobs)` | copied per worker (process backend) |
| `_pipeline_step_path` | `_provenance.py:74-75` | **ContextVar** — per-thread, does not cross a process |
| `_application_owner_depth` | `_provenance.py` / `measure/CLAUDE.md:22-27` | **ContextVar** — same |
| Stage-2 raw + token | `.phenotypic/progress/` | on disk; raw written before token, so a crash between them is recoverable |

**Assessment.** The plan's structural claim — per-image `PrivateAttr` state
rules out caching a pipeline instance across images — is correct and is
currently satisfied only by accident: Stage 2 is serial, and joblib's default
process backend copies the op for Stage 1/3 so nothing is shared there. If any
round is ever run with a threading backend, or if a GPU round is ever
parallelised locally, two images write `_phase_state` onto the same instance and
the corruption is silent (the wrong array, not an exception). Prefer an
assertion — e.g. `capture_phase_state` recording the image identity it was
captured for, and `restore_phase_state` refusing a mismatch — over a documented
constraint.

**Cross-process state.** Both provenance ContextVars are per-process. With a
K-round chain in K processes the engine must re-establish them at the top of
every round, and the plan does not say to what values. This is the mechanism
half of B-3.

**Conditional GPU phases and the dataset-wide barrier.** Open question 3 is
narrower than it looks. Deriving a round's task list from disk does keep the
barrier honest — the round runs over whatever images produced work. What it does
*not* keep honest is the **submission-time** decision: `pipeline_requires_gpu`
picks the partition and `--gres` before any image is touched (B-2). So a
conditional GPU phase costs a GPU allocation for the whole run, not an empty
array task. Additionally, a round whose task list is empty must be
distinguishable from a round whose predecessor failed, or the continuation logic
will read "no work" as "done".

---

## Verification results 🧪

### Executed probe

`<scratchpad>/probe_plan_b_pydantic.py`, run by the orchestrator with
`uv run python ...` in the worktree. **pydantic 2.12.5, pydantic-core 2.41.5,
Python 3.12.10**, exit 0. Verbatim output:

```
Q1 underscore ClassVar accepted at class build:
  Base._phases            : (('_operate', 'cpu'),)
  Base()._phases          : (('_operate', 'cpu'),)
  '_phases' in class_vars : True
  '_phases' in privattrs  : False
  '_phases' in model_fields: False

Q2 annotated override:
  GoodChild._phases       : (('_prep', 'cpu'), ('_infer', 'gpu'), ('_write', 'cpu'))
  GoodChild()._phases     : (('_prep', 'cpu'), ('_infer', 'gpu'), ('_write', 'cpu'))

Q3 UN-annotated override (Plan B Task B1 test style):
  type(SloppyChild._phases): 'tuple'
  SloppyChild._phases      : (('_prep', 'cpu'), ('_typo_here', 'gpu'))
  in __private_attributes__: False
  in __class_vars__        : True
  instance._phases         : (('_prep', 'cpu'), ('_typo_here', 'gpu'))
  isinstance(..., tuple)   : True

Q4 bare `ClassVar` annotation:
  BareChild._phases       : (('_operate', 'cpu'),)
  BareChild._child_order  : 'parallel'
  in __class_vars__       : True

Q5 PrivateAttr mechanics:
  fresh __pydantic_private__ keys: ['_carried', '_not_carried']
  fresh op._carried              : None
  fresh op._carried_nodefault    : !! AttributeError: 'Phased' object has no attribute '_carried_nodefault'
  after setattr(op,'_carried',arr):
    op._carried is arr           : True
    in __pydantic_private__      : True
    in op.__dict__               : False
  after object.__setattr__(op2,'_carried',arr):
    op2._carried is arr          : True
    in __pydantic_private__      : True
    in op2.__dict__              : True
    op2.model_dump(mode='json')  : {'x': 1}
    op2.model_dump()             : {'x': 1}
    deepcopy keeps it            : array([0, 1, 2, 3, 4, 5], dtype=uint16)
    pickle keeps it              : 'ndarray'

Q5b serialization exclusion (plain setattr instance):
  model_dump(mode='json')        : {'x': 1}
  model_fields                   : ['x']
  deepcopy keeps _carried        : 'ndarray'
  pickle  keeps _carried         : 'ndarray'

Q6 unset vs explicit None:
  fresh    in private dict       : True
  explicit in private dict       : True
  nodefault in fresh private dict: False
  getattr fresh _carried_nodefault: !! AttributeError: 'Phased' object has no attribute '_carried_nodefault'

Q7 assigning to a ClassVar name on an instance:
  op._phases = ()                : !! AttributeError: '_phases' is a ClassVar of `Phased` and cannot be set on an instance. If you want to set a value on the class, use `Phased._phases = value`.

Q8 against the real phenotypic ABCs:
  RealPhased._phases           : (('_prep', 'cpu'), ('_infer', 'gpu'), ('_write', 'cpu'))
  model_dump(mode='json')      : {}
  ImagePipeline is ImageOperation?: False
  ImagePipeline is BaseOperation? : True
  GpuDetector is ImageOperation?  : True
  ImagePipeline has _operate?     : False
  RealSloppy._phases type      : 'ModelPrivateAttr'

Q9 payload dict equality with ndarray values (B3 test style):
  {'a': arr} == {'a': arr.copy()}: !! ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
```

### Pydantic claims

| Claim | Verdict | Evidence |
|---|---|---|
| Underscore `ClassVar` allowed on a `BaseModel` | ✅ true | probe Q1; `_model_construction.py:463-466` |
| ...under `from __future__ import annotations` | ✅ true | probe Q1 (the probe file uses it); `_typing_extra.py:100-130`, `_classvar_re` |
| ...in the bare `ClassVar` form the plan writes | ✅ true | probe Q4 |
| Un-annotated `_x = ...` becomes a `PrivateAttr` | ⚠️ **only if no base declares the ClassVar** | probe Q3 (repaired) vs Q8 `RealSloppy` (not repaired); repair at `_model_construction.py:571-581` + `_fields.py:265,280` |
| ...so Task B1's un-annotated test style is broken | ❌ **withdrawn** — it works | probe Q3 |
| A ClassVar name cannot be set on an instance | ✅ true (a safety property) | probe Q7 |
| Engine can read/write private attrs generically via `setattr` | ✅ true | probe Q5; `main.py:1050-1058` |
| ...via `object.__setattr__` | ⚠️ works, but writes to `__dict__` instead of `__pydantic_private__` | probe Q5 |
| `PrivateAttr` excluded from `model_dump(mode="json")` | ✅ true | probe Q5b, Q8 (`{}` for a real op holding an ndarray) |
| `PrivateAttr` excluded from `model_fields` | ✅ true | probe Q1, Q5b |
| Private attrs survive `deepcopy` and `pickle` | ✅ true | probe Q5b (matters for joblib workers) |
| `PrivateAttr(default=None)` makes unset indistinguishable from `None` | ✅ true | probe Q5/Q6 |
| `PrivateAttr()` with no default leaves the key absent | ✅ true | probe Q6 |
| Payload dicts with ndarrays compare with `==` | ❌ raises | probe Q9 |

### Repository claims

| Claim | Verdict | Evidence |
|---|---|---|
| `GpuDetector` hooks map to cpu/gpu/cpu | ✅ true | `abc_/_gpu_detector.py:249-262`; `_cli_staged_workers.py:334-342`, `:461` |
| The `ClassVar`-on-an-operation pattern is already in use | ✅ true | `detect/_filamentous_fungi_detector.py:254-270` |
| `_operate` has one meaning across the hierarchy | ❌ false | `_image_operation.py:455`; `_measure_features.py:471-472` (no `self`, returns a DataFrame); `_post_measurement.py:53` |
| `ImagePipeline` is an `ImageOperation` | ❌ false — it is a `BaseOperation` | probe Q8; `_image_pipeline_core.py:143` |
| `ImagePipeline` has an `_operate` | ❌ **false** | probe Q8 — so the inherited default phase names a method it does not have |
| Plan A Task 4 / Task 9 survive unchanged | ❌ false | both live inside `apply`; `_image_operation.py:396-416`, `_provenance.py:549-629` |
| Plan A Task 12 "reused unchanged" | ❌ false | `plan-a:2158-2178` deliberately excludes full runs |
| Plan A Task 3 substantially survives | ⚠️ partly | the error type, the `meas`-slot refusal and the tests survive; `pipeline_requires_gpu` itself is rewritten |
| FFD PHASE 1 / PHASE 2 is the right cut | ❌ see B-2 | in the nested case PHASE 1 *is* the GPU work |
| What must cross the FFD boundary | ✅ the plan names them correctly | `inoculum_objmask` (`:400`), `inoculum_img.objmap[:]` (`:464`) |
| README carry-over arithmetic | ⚠️ Task 2 unaccounted | `plan-a:404` |
| 669 CPU-hours | ✅ 71 × 33,923 = 2,408,533 s = 669 h | arithmetic |
| Three-stage shape is hard-coded on disk | ✅ true | `_stages.py:6-15`; `_cli_staged_resume.py:35-40`; `_cli_staged_slurm.py:347-438` |

**Not verified:** anything requiring a real GPU run; the actual per-image size
of FFD's ferried arrays on production data (calculated from the 3140×5094 figure
the spec quotes, not measured); whether `joblib`'s backend is ever threading
here; and the *value* (as opposed to key presence) left in
`__pydantic_private__` after an `object.__setattr__` write — see M-8, where the
stale-`None` claim is inferred, not observed.

---

## Questions for clarification ❓

These are what the spec must settle before execution. Each maps to a finding.

1. **How does a parent's own phase receive its children's outputs?** N branch
   images, N objmaps, one combine. Four sub-questions — child-block position,
   per-slot input rule, handoff type/arity/signature, and where the handoff
   lives between rounds — are sketched under B-1, along with the cheaper
   fallback of keeping Plan A's re-execution for the parent body only.
   (B-1 — this is the blocking one.)
2. **Is `phases_of` allowed to consult instance fields?** If not, how does a
   container whose device depends on a configured child avoid claiming a GPU it
   never uses? (B-2)
3. **Who emits provenance for a phased operation** — with what duration, what
   `pipeline_step_path`, and what `_application_owner_depth`, across K
   processes? (B-3)
4. **What is the ferried-payload cap, and are B6's two arrays state or layers?**
   They are ~48 MB per image per boundary. (M-4)
5. **What is the on-disk tag and per-image record schema for a K-round chain,**
   and what is the migration for runs already recorded as three stages? (M-2)
6. **Is full-run continuation invalidated by this change?** (M-6)
7. **Is N > 1 GPU round supported or refused?** The table and B2's test
   disagree. (M-7)
8. **Is cross-image batching abandoned, deferred, or preserved?** (m-4)
9. **Does a phase return a new image or mutate in place** — and if in place,
   what happens to `apply(inplace=False)` semantics for a phased op invoked
   from a notebook? (the plan's own open question 2, unanswered for the
   notebook path)

---

## Is the approach viable at all?

**Yes — but not as seven tasks, and not without the spec.**

The declaration idea is genuinely better than inference for the thing it is
good at: a `GpuDetector` declaring its own cpu/gpu/cpu split is more honest than
an external table describing it, and removing the branch-prefix re-execution is
worth 669 CPU-hours on the driver run. The pydantic mechanics hold. Task B5's
instance-poisoning gate is the right instrument for the failure mode.

What the plan has not reckoned with is that removing re-execution *adds* a
requirement rather than removing one. Plan A can treat a container's internals
as opaque because it re-runs them. Plan B must model them — the copy semantics,
the combine, where each child sits among the parent's own phases, and what the
parent gets back. `_child_order` is one bit where at minimum three are needed,
and every one of B-1, B-2 and B6's confused rationale is a symptom of that one
gap.

Realistically this is closer to 16–20 tasks than 7, and the honest comparison is
not "7 vs 16" but "7 written vs 16 written, for a strictly larger change." The
sequencing in the README — land Plan A, then spec and review Plan B — is the
right call, and this review does not change it. When Plan B's spec is written,
B-1 is the first thing it has to answer; everything else follows from it.
