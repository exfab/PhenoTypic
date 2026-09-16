# Nested `GpuDetector` staging — design

**Date:** 2026-09-15
**Status:** draft, pending review
**Driver:** `ucr_033_e_d_Linzer_Ganoderma/config/F1gfd5.json.pht-pipe` — a 33,923-image
run whose `Sam2` lives inside `CompositeDetector.ops`, where the CLI silently
declines to stage it.

---

## 1. Objective & non-goals

**Objective.** Let the staged GPU engine run a pipeline whose single
`GpuDetector` is *nested* inside another operation, rather than a top-level
element of `ImagePipeline.get_ops()`.

**Non-goals.**

- More than one `GpuDetector` per pipeline. The single-detector rule stands,
  extended tree-wide — but it is now a **deferred feature, not a limit of the
  design**: §13 records the intended `N > 1` execution model, and §4.4 lands the
  slot-keyed Stage-2 signal *now* so that work is additive rather than a change
  to the on-disk layout of a signal a 33,923-image run depends on.
- Any change to the Stage-2 raw/token contract, the epoch-fenced controller, or
  SLURM chaining. All are untouched. *Continuation is untouched in mechanism,
  but §8.3 deliberately invalidates it once, at the upgrade, via the work-id
  digest.*
- Model residency or pipeline caching in the *non-staged* per-image path
  (`_cli_process_single.py:260`). That was option B in the scoping discussion and
  is explicitly not this design.

**In scope, by decision.** Two behaviour changes reaching beyond GPU staging:

1. **(§8)** `--mode process --layer objmap` changes meaning for **both**
   top-level and nested GPU detectors — it comes to mean "the objmap your
   pipeline produces" rather than "whatever the detector emitted".
2. **(§5.2)** Container operations push a per-branch `pipeline_step`, so
   `pipeline_step_path` descends uniformly. This changes recorded journals for
   **every** pipeline using a container op, including pipelines with no GPU
   detector at all.

Both are changes to shipped, documented behaviour and are the widest-reaching
parts of this design. Each creates a version boundary (§8.3, §12).

---

## 2. Current behaviour, and why it is a bug

`pipeline_requires_gpu` scans only the top level:

```python
# src/phenotypic/_cli/_cli_validation.py:147
return any(isinstance(op, GpuDetector) for op in pipeline.get_ops().values())
```

Measured against the real pipeline:

```
top-level ops: [CropImage, DenoiseBlockMatch, CompositeEnhance, CompositeDetector,
                MaskFill, MaskDilation, SmallObjectRemover, KeepSectionLargest]
pipeline_requires_gpu -> False
composite.ops types  : [Sam2, ManualPointDetector]
split_pipeline_at_gpu -> ValueError: no GpuDetector in pipeline
```

Three consequences, none of which raises:

| Consequence | Where |
|---|---|
| Staged engine never engages | `_cli_execution_strategies.py:1341` |
| SLURM never adds `slurm_gpus_per_node=1`; the array lands on a CPU partition and `device="auto"` resolves to CPU | `:906` |
| Locally `effective_n_jobs` stays at `--njobs` (default `-1`), so N processes each build a SAM2 model on one GPU | `:341`, `:361` |

This is a **wrong-answer bug**, not a slow path: the run completes on CPU and
produces different numbers. Fixing detection is worth landing independently of
the rest of this design (§9, step 1).

---

## 3. Why the split is the hard part

`split_pipeline_at_gpu` (`_cli_pipeline_split.py:34`) assumes the GPU op is one
element of a *linear* top-level sequence, cuts at its index, and returns
`pre_pipeline` / `gpu_detector` / `post_pipeline`. A nested detector is a leaf in
a tree: there is no "ops before it" inside the composite, and its sibling
branches must run and be merged inside the same `_operate` call.

The Stage-2/Stage-3 contract, however, is narrower than it looks and survives
nesting intact. Stage 2 produces exactly one `(H, W)` array per image
(`_cli_staged_workers.py:398-412`); Stage 3 replays it with
`_write_object_output` (`:490`) and then runs `post_pipeline`. Nothing in that
requires the detector to be top-level — only the *split* and the *replay
injection point* do.

---

## 4. Design

Cut at the GPU op's **top-level ancestor**, and in Stage 3 substitute a replay
stub for the nested detector so the enclosing operation runs normally with the
recorded mask standing in for live inference.

For the driver pipeline this yields:

| Stage | Runs on | Ops |
|---|---|---|
| 1 | CPU | `CropImage`, `DenoiseBlockMatch`, `CompositeEnhance` |
| 2 | GPU, model resident per shard | `Sam2` over the store's `detect_mat` |
| 3 | CPU | `CompositeDetector`(stub, `ManualPointDetector`), `MaskFill`, `MaskDilation`, `SmallObjectRemover`, `KeepSectionLargest`, then 7 measurers |

### 4.1 Tree walker

A recursive walk over operation-bearing children: an `ImagePipeline` yields its
`get_ops()` entries; any pydantic operation yields its `OperationField`-typed
fields, including list entries, addressed as `field[i]`.

**Decision (2026-09-15): consolidate.** Two walkers over this same structure
already exist — the marker scan at `gui/_operation_registry.py:33` and the
recursion in `tune/_search_space/_infer.py:429,546,703`. A third private copy
would be the single largest design risk in this change, so the walker lives in
**one shared module** and `gui/` and `tune/` are migrated onto it rather than
left duplicated.

This widens the blast radius into two modules this change otherwise does not
touch, which the plan must account for: `gui/` and `tune/` each need their own
regression pass, and the shared walker must satisfy all three callers'
needs — the CLI wants *paths to GpuDetectors*, `gui/` wants *marker presence on
an annotation*, `tune/` wants *one-level list recursion with its own depth
rule*. If a single signature cannot serve all three without contortion, the
correct outcome is one shared traversal primitive with thin per-caller
adapters, not three traversals.

### 4.2 `StagePlan` becomes path-shaped

```
gpu_key:  str                    ->  gpu_path: tuple[str, ...]
                                     e.g. ("CompositeDetector", "ops[0]")
+ stage2_prefix: list[ImageOperation]
```

`pre_pipeline` is cut at `gpu_path[0]`; **the node at `gpu_path[0]` heads
`post_pipeline`**, and Stage 3 substitutes the stub at `gpu_path` within it.

This is uniform across depths, and that uniformity changes the **top-level** case
too. When the detector is itself top-level, `gpu_path == ("Sam2",)`, so `Sam2`'s
own slot now sits in `post_pipeline` and the stub lands in it — where previously
`post_pipeline` began *after* the detector and Stage 3 called
`_write_object_output` explicitly before applying it. Both paths therefore become
one path. Two consequences the plan must carry:

- Omitting the ancestor for a top-level detector would make Stage 3 **re-run the
  real detector on a CPU node**. The existing
  `tests/unit/cli/test_cli_pipeline_split.py:24` asserts the old exclusion and
  must be updated — an intended change, not a regression.
- `_branch_prefix` must return `[]` when `len(gpu_path) == 1`. The spike's
  version does not (it lacks the root guard on its final block) and returns every
  preceding top-level op — ops Stage 1 has already applied and written to the
  store, which Stage 2 would then re-run on top of themselves. The spike never
  exercises a top-level detector, so this was latent. **Pin it with a test.**

### 4.3 Stage-2 branch prefix

If the GPU op sits behind CPU operations *inside its own branch* (a nested
`ImagePipeline` such as `[ContrastStretching, Sam2]`), Stage 2 cannot read the
store layer directly. It must apply that prefix first.

The rule walks the ancestor chain and asks one question of each container:
**what image does it hand its children?** Two answers, and a closed table:

**Only composition primitives may carry a staged GPU detector.** Three
classes, and nothing else — ever, by rule rather than by survey:

| Contract | Meaning | Prefix contribution | Class |
|---|---|---|---|
| `"sequence"` | each child receives the previous child's output | the ops preceding the branch | `ImagePipeline` |
| `"same"` | every child receives the container's own input | **nothing** — branches are parallel, none runs "before" another | `CompositeDetector`, `CompositeEnhance` |

Every other `OperationField`-bearing class is **refused**, naming the class.

**The line is "composition primitive", not "currently classifiable".** These
three exist *to compose other operations*; their child-input semantics is part
of what they are. A `CompositeDetector` whose branches chained would not be a
composite — it would be an `ImagePipeline`, which already exists for exactly
that. So the table restates a type contract rather than caching an observation,
which is what makes a class-keyed table in the splitter safe rather than
drift-prone.

A **domain detector** is on the other side of that line even when its current
code would classify cleanly. `FilamentousFungiDetector` passes
`inoculum_detector` the container's own image today (`:395,398`), so it *reads*
as `"same"` — but that is incidental to an algorithm whose `_operate` also runs
an inline `ContrastStretching()` (`:413`), a destructive `_subtract_background`,
and a `del enhanced_work`. Nothing about being a fungus detector constrains it to
keep feeding its child the raw image. Admitting it would mean the table's safety
argument no longer holds uniformly, for one class nobody has asked to stage.

An earlier draft admitted it, and separately argued that
`TwoKFilamentousDetector` was *inexpressible*. Drawing its call flow showed that
was too strong — a `"same"` / `"after(<field>)"` vocabulary describes all three
of its fields exactly (`branch_base` is `"same"`, `background_subtractor` is
`"after(branch_base)"`). The refusal is therefore a **scope** decision, not an
impossibility, and scoping it by *kind of class* rather than by *expressibility*
is both simpler and more stable.

**The contract is tested, not merely asserted.** Each `"same"` entry carries a
behavioural test that puts two recording probe operations in the container's
children and asserts the second did **not** observe the first's output. A
declaration can lie and still pass; a probe cannot. This is the reason a
lookup table beats a `_child_input` ClassVar on each operation — the ClassVar
would add API surface *and* still need the probe test to be trustworthy.

**Coverage is enforced.** A guard test enumerates every
`OperationField`-bearing class (7 today) and requires each to be in the table or
on an explicit unsupported list with a reason. Adding a container fails the
suite until someone decides, so the failure lands at authoring time rather than
in a 33,923-image run.

`TwoKFilamentousDetector` remains the worked illustration of *why* domain
detectors are excluded: `center_detector` receives the original image (`:149`),
`background_subtractor` a derived `enhanced.copy()` (`:154`), and `branch_base`
**mutates** `enhanced` in place (`:164`). Three fields, three inputs, inside one
algorithm — and the shape is invisible from the outside.

The prefix is applied to an **in-memory copy** inside Stage 2 and is never
written to the store, preserving the existing "Stage 2 does NOT write into the
store" invariant. It is then re-executed in Stage 3 as part of the normal branch
run. See §7 for the cost of that double execution.

**The copy must be provenance-detached, not a bare `image.copy()`.**
`stage2_detect_core` runs at `_application_owner_depth == 0` against a store whose
trailing application is `"staged"`, and `Image.copy()` carries the journal across.
A bare copy therefore makes the first prefix op raise `cannot start a new
provenance application before the last ends` (`_provenance.py:361-363`) — the same
trap as §8.2. Use the established in-tree pattern from
`measure/_canonical_zone_measure.py:279-295`: deep-copy the journal, mark every
non-terminal application `"complete"`, attach it to the copy, then apply. The
copy is discarded, so its journal is meaningless — and detaching (rather than
`continuing_provenance_application`) is the right choice here precisely because
the prefix's records must reach **nothing**: Stage 3 re-runs the same ops and
records them for real.

**For the driver pipeline the prefix is empty** — `Sam2` is a bare leaf, so
Stage 2 reads `detect_mat` from the store exactly as it does today.

### 4.4 The Stage-2 signal is keyed by detector slot

Today the Stage-2 signal is keyed by **image alone**:

```python
stage2_raw_path(output_dir, dataset, image_stem)
#   -> <output>/.phenotypic/progress/stage2_raw/<dataset>/<stem>.npy
```

One array per image, structurally. That is the single thing standing between
this design and supporting **more than one GPU detector**, and it is changed
**now**, while `N` is still 1:

```python
stage2_raw_path(output_dir, dataset, image_stem, slot)
#   -> <output>/.phenotypic/progress/stage2_raw/<dataset>/<slot>/<stem>.npy
```

`slot` is derived from the detector's `gpu_path`: each segment sanitised to
`[A-Za-z0-9-]`, joined with `__`, plus an 8-character hash of the exact path so
two different paths can never collide after sanitisation. Debuggable by eye,
collision-proof by construction —
`CompositeDetector__ops-0__3f9a1c02`. The token path changes the same way.

**Why now rather than with the feature.** At `N == 1` this is invisible: one
extra directory level and no behavioural difference. Deferred, it becomes a
change to the on-disk layout of a signal that a 33,923-image run depends on,
and every in-flight run would have to recompute Stage 2 — GPU time, on the
scarcest resource in the system. Landing it here costs one path helper and a
test, and makes `N > 1` purely additive.

**Cost at the boundary.** A staged run interrupted mid-Stage-2 *before* this
change and resumed *after* it will not find its signals at the new paths and
will recompute them. That is a recompute, not a wrong answer — Stage 2 is
content-defined and idempotent. To avoid paying it at all, resume performs a
one-time relocation: when the legacy per-image path exists, the slot path does
not, and the plan has exactly one slot, move the file. Roughly ten lines, and
worth it against a 33,923-image GPU sweep.

### 4.5 The replay stub

```python
class ReplayDetector(ObjectDetector):
    """Stage-3 stand-in: writes a PRE-RECORDED Stage-2 result."""
    detector: OperationField
    result: NdArrayField

    def _operate(self, image):
        self.detector._write_object_output(image, self.result)
        return image
```

`CompositeDetector` calls `detector.apply(image, inplace=False)` and reads
`detected_image.objmap[:]`, so the stub satisfies the existing contract with no
change to the composite. Substitution builds a deep copy addressed by
`gpu_path`; the loaded pipeline is never mutated.

---

## 5. Provenance

Nested operations already receive their own journal entries —
`_run_operations` wraps each apply in `with pipeline_step(key)`
(`_image_pipeline_core.py:897-899`). But the step path does not *descend*
uniformly, and this design fixes that (§5.2).

**Constraint throughout:** `pipeline_step_path` is validated as a list of
**non-empty strings** (`_provenance.py:277-283`). An integer branch index is
illegal; the `field[i]` string form is used everywhere.

**Required of the replay stub:** `append_operation_provenance`
(`_provenance.py:872-912`) derives **four** fields from the operation, and the
stub must override all four — not two:

| Field | Default source | Why the stub must override it |
|---|---|---|
| `operation_name` | `type(operation).__name__` | else the journal reads `ReplayDetector` beside an `operation_class` of `Sam2` — internally inconsistent, and pinned by `tests/integration/cli/test_staged_store_stages.py:115` |
| `operation_class` | `module.qualname` | staged/single-pass parity |
| `parameters` | `operation.model_dump(mode="json")` | parity — **and** the stub holds an `NdArrayField`, so the default would serialise the entire recorded objmap into the journal |
| `duration_seconds` | measured wall time of the apply | the merge alone is not the cost; the total is the Stage-2 token's `detector_duration_seconds` + the merge, pinned by `test_staged_store_stages.py:128` |

Keep the existing JSON round-trip (`json.loads(json.dumps(...))`) around whichever
source supplies `parameters`; it is what guarantees the value is JSON-native before
`validate_provenance_journal` sees it. Stage 3 currently appends
the GPU op's entry explicitly, *outside* the apply
(`_cli_staged_workers.py:490-500`); under this design the write happens inside
the enclosing operation's `_operate`, so that explicit append is replaced by the
stub's identity delegation.

### 5.1 The step-path asymmetry (measured)

Measured via `probe_step_paths.py`:

```
B (nested ImagePipeline branch):
   ContrastStretching   ['CompositeDetector', 'ContrastStretching']   <- descends
   FakeGpuDetector      ['CompositeDetector', 'FakeGpuDetector']      <- descends

A (bare leaf in a composite):
   FakeGpuDetector      ['CompositeDetector']                         <- does NOT
```

A nested `ImagePipeline` descends because its own `_run_operations` pushes a
`pipeline_step`; a `CompositeDetector` does not, because it calls `.apply()`
directly. The consequence, shape C:

```
FakeGpuDetector      ['CompositeDetector']
ManualPointDetector  ['CompositeDetector']
CompositeDetector    ['CompositeDetector']   <- the INNER composite (mode=union)
ManualPointDetector  ['CompositeDetector']
CompositeDetector    ['CompositeDetector']   <- the OUTER composite (mode=overlap)
```

Five entries, one path. Two different `CompositeDetector` instances with
different `mode` values, and two different `ManualPointDetector` calls, are
indistinguishable by path — only by ordinal position and the `parameters` dict.

The asymmetry is also a trap in its own right: a reader who observes shape B
descending would reasonably infer shape A does too, and write code against it.

### 5.2 Fix: container operations push their branch step

Container operations that drive children wrap each child apply:

```python
for i, detector in enumerate(self.ops):
    with pipeline_step(f"ops[{i}]"):
        detected = detector.apply(image, inplace=False)
```

Four classes drive children this way — `CompositeDetector`, `CompositeEnhance`,
`FilamentousFungiDetector`, `TwoKFilamentousDetector`. The convention belongs in
a **shared helper**, not copied four times. `pipeline_step`
(`_provenance.py:527`) already exists and has exactly one caller today
(`_image_pipeline_core.py:900`), so this adds a caller rather than a mechanism.

**Explicitly excluded: the zone measurers' `center_detector`.** Per
`measure/CLAUDE.md`, a nested operation run by a *measurement* is a private probe
whose steps deliberately do not belong in the plate's provenance. It continues to
receive no step path. This exclusion is a design decision and must be asserted by
a test, or a later reader will "complete" the change by adding it.

### 5.3 Consequence: one addressing scheme, not two

With §5.2 in place, the walker's `gpu_path` and the recorded
`pipeline_step_path` become **the same value**. Measured via
`probe_step_unification.py`, which simulates §5.2 by subclassing
`CompositeDetector`:

| Shape | walker `gpu_path` | journal step path | identical |
|---|---|---|---|
| A | `['CompositeDetector', 'ops[0]']` | `['CompositeDetector', 'ops[0]']` | yes |
| B | `['CompositeDetector', 'ops[0]', 'FakeGpuDetector']` | same | yes |
| C | `['CompositeDetector', 'ops[0]', 'ops[0]']` | same | yes |

This is why §5.2 belongs in *this* spec rather than a follow-up: the staged
engine needs a path to address the GPU op for substitution, and provenance needs
a path to record it. Without §5.2 those are two schemes that happen to look
alike; with it they are one value, derived once by the walker and used for both.
The stub's journal entry then names the exact branch it stood in for, and
**the identity `gpu_path == pipeline_step_path` is a testable invariant** (§10)
rather than a coincidence maintained by hand.

Staged/single-pass journal parity is unaffected: substitution happens at the
same path, so both runs record the same step path, and the stub's identity
delegation supplies the same class and parameters.

---

## 6. Evidence: the spike

A throwaway spike substitutes a deterministic CPU `FakeGpuDetector` with the same
`GpuDetector` hook surface for SAM2, so the mechanism is exercised without a GPU.
Scripts are committed beside this spec:

- `spike_nested_gpu.py` — walker, prefix rule, substitution, three nesting shapes
- `spike_controls.py` — negative control + provenance comparison
- `probe_real.py` — the real `F1gfd5` pipeline through the prototype splitter
- `probe_process_provenance.py` — the process-mode provenance trap (§8)

Run: `uv run python spike_nested_gpu.py && uv run python spike_controls.py`

| Shape | Path found | Stage-2 prefix | objmap | measurements |
|---|---|---|---|---|
| A — GPU leaf in composite (*the real shape*) | `CompositeDetector/ops[0]` | — | identical | identical (9×48) |
| B — GPU behind a CPU prefix in a nested pipeline | `CompositeDetector/ops[0]/FakeGpuDetector` | `[ContrastStretching]` | identical | identical (9×48) |
| C — composite inside a composite | `CompositeDetector/ops[0]/ops[0]` | — | identical | identical (9×48) |

Negative control — shifting the recorded Stage-2 array by 7 px:

```
clean replay matches reference : True   (want True)
corrupted replay matches ref   : False  (want False)
corrupted objects=9 vs ref=9     <- an object-count check alone would NOT catch it
```

**Note on placement.** These scripts import `phenotypic`, so per the repo rule
they cannot live under `docs/superpowers/logic_validation_scripts/` — that
directory's contract is that nothing in it imports the code under test. A check
that must drive the shipped code belongs beside its plan, which is where these
are. The invariant here is an *equivalence* property of the shipped code, not a
numeric one derivable from first principles.

---

## 7. Limits & bounds

### Supported (proven, §6)

- A `GpuDetector` as a bare leaf in a container's operation list.
- A `GpuDetector` behind a deterministic CPU prefix inside a nested
  `ImagePipeline` branch.
- Arbitrary nesting depth.

### Refused, loudly

| Case | Reason |
|---|---|
| More than one `GpuDetector` anywhere in the tree | Deferred, not impossible — see §13. The refusal must name **every** offending path, not just the count, so the message tells a user which branches to split |
| A `GpuDetector` in the `meas` / `post` / `filters` / `model` slots | Stage 3 runs these on a CPU node. The walker must scan these slots in order to reject them — the driver pipeline does carry nested ops there (`MeasureSymZones.center_detector`, `MeasureOrientationZones.center_detector`, both `ManualPointDetector`) |
| A `GpuDetector` anywhere other than `ImagePipeline` / `CompositeDetector` / `CompositeEnhance` | Only composition primitives may carry one (§4.3). Domain detectors are refused by rule, not by survey |
| A `GpuDetector` whose immediate container is an **enhancer** container | An independent gate from the one above, on *output kind* rather than child input: `_write_object_output` writes an objmap, while an enhancer branch must yield a layer. `CompositeEnhance` therefore has a child-input contract (it is a primitive) and still refuses a detector as a direct child |

### Accepted costs

- **The branch prefix runs twice** — once in Stage 2 in-memory, once in Stage 3.
  It must therefore be deterministic. An expensive op placed inside a branch is
  paid twice. **Zero cost for the driver pipeline** (empty prefix); this only
  bites if `DenoiseBlockMatch` or `FocusEdgePhase` is later moved inside the
  composite branch.
- **Stage-2 memory** rises by one image copy when a prefix exists (none today).
- **Stage-3 retry idempotency** is inherited unchanged, including the known
  `FLOW-21` caveat that a post-op touching `detect_mat`/`gray` is applied twice
  on retry. A retry now also re-runs the composite's CPU branches; these are
  deterministic, so the objmap is unaffected.

---

## 8. `--mode process --layer objmap`

**Decision: run the post-detector op chain, for both top-level and nested GPU
detectors.** The flag comes to mean *the objmap your pipeline produces*.

### 8.1 Rationale

The governing principle is that **the operations a user supplies as pipeline
parameters should be the operations that actually run.** A pipeline declaring
`MaskFill`, `MaskDilation`, `SmallObjectRemover` and `KeepSectionLargest` after
its detector is asking for those to shape the objmap; exporting the detector's
raw mask and calling it the objmap silently discards four operations the user
configured.

Today `_export_objmap_layer` (`_cli_staged_strategy.py:397`) does
`load_zarr → _write_object_output → write`, applying no post-detector ops. For a
*nested* detector that would be worse still — the SAM2-only mask, with the
composite's `overlap` merge against `ManualPointDetector` never run. Rather than
fork the flag's meaning by where the detector sits in the tree, both cases run
the chain.

**The post-detector op chain** is the Stage-3 operation sequence *without*
measurement: for the driver pipeline, `CompositeDetector`(stub,
`ManualPointDetector`) → `MaskFill` → `MaskDilation` → `SmallObjectRemover` →
`KeepSectionLargest`. `ImagePipeline.apply()` runs only `_run_operations`
(`_image_pipeline_core.py:966-974`), so measurement, filters and model are not
triggered by this call.

### 8.2 Consequence 1 — the depth-0 provenance trap

Stage 1 leaves the trailing application `"staged"`, which is not terminal
(`_provenance.py:29,363-364`), so `apply()` at CLI owner-depth 0 appends and
raises. Reproduced in `probe_process_provenance.py`:

```
Stage-1 trailing application status: 'staged'
post-detector op chain at depth 0: ValueError: cannot start a new provenance
application before the last ends
```

**Mitigation.** Wrap the apply in `continuing_provenance_application(image)` and
install **no** `provenance_success_sink`.

> **Corrected 2026-09-15 after plan review.** An earlier draft of this section
> prescribed `truncate_provenance_to_retry_base` + `set_provenance_status(image,
> "in_progress")` instead. **That does not work.** `"in_progress"` is not in
> `_append_application`'s terminal set either — the set is `{"complete",
> "failed"}` (`_provenance.py:362-363`), while `_APPLICATION_STATUSES` is
> `{"complete", "failed", "in_progress", "staged"}` (`:28`). Setting
> `"in_progress"` therefore raises the very error it was written to avoid. The
> draft was presented as settled on the strength of
> `probe_process_provenance.py`, which reproduces the trap but never exercises
> the mitigation — a probe that tests the disease and not the cure.

What actually makes Stage 3's apply legal is `continuing_provenance_application`
(`_provenance.py:454-465`): it accepts a `"staged"` application and increments
`_application_owner_depth`, so `provenance_application` **joins** the open
application instead of appending a new one. Process mode uses the same shape Stage
3 uses (`_cli_staged_workers.py:503-511`) **minus the success sink** — the sink is
what would write to the store, so omitting it is what preserves FLOW-16/FLOW-30/
FLOW-6, not omitting a checkpoint call. Neither `truncate_provenance_to_retry_base`
nor `set_provenance_status` is needed: the image is freshly loaded on every export,
so there is nothing stale to truncate, and `continuing_provenance_application`
accepts `"staged"` directly.

The absent sink is intentional and must carry a comment saying so, or a later
reader will "fix" it by adding one back.

### 8.3 Consequence 2 — continuation across the upgrade (the dangerous one)

`work_id_for_image` (`_cli_failure_tracker.py:310`) hashes
`pipeline_fingerprint`, `input_sha256`, `processing_configuration_digest` and
`mode`. `processing_configuration_digest` (`:191-236`) contains only
*user-supplied configuration* — `image_type`, `nrows`, `ncols`, `bit_depth`,
`detect_mode`, `drop_originals`, and for process mode `process_only_layer`,
`ext`, `process_format`.

**Nothing in the work id tracks a change in PhenoTypic's own output semantics.**
A process run interrupted before this change and resumed after it would treat
old-semantics PNGs as complete, and publish a tree mixing raw-detector and
pipeline objmaps with no indication which is which.

**Mitigation: add an explicit output-semantics revision to the digest** — a
module-level integer constant, bumped whenever per-image output semantics change,
folded into `processing_configuration_digest_from_values`.

**It goes in the `process_only_layer is not None` branch, beside
`process_format` — NOT in the base payload.** There is a documented local
precedent three lines from the insertion point
(`_cli_failure_tracker.py:218-223`):

> `# Beside `ext` and NOT in the base payload: a full or measure run has no`
> `# process format, and folding it into the base would change every existing`
> `# run's digest and cold-start every continuation in flight.`

The behaviour change in §8 is scoped to `--mode process --layer objmap`. A
base-payload placement would cold-start every in-flight `full` and `measure`
continuation on the cluster — including the 33,923-image run this spec exists
for — buying no correctness. Fold in `f"{process_only_layer}:{revision}"` so a
`gray` export is not invalidated by an `objmap` semantics change either.

The constant's docstring must name **which outputs** the revision governs; read
broadly, "per-image output semantics" is exactly what argues for the wrong
placement.

Two alternatives were rejected. Folding in `phenotypic.__version__` invalidates
continuation on **every** release including patches, breaking legitimate resume.
Documenting "re-run from scratch after upgrading" is silent by default, which is
the failure mode being mitigated.

### 8.4 Consequence 3 — the change would otherwise land unobserved

No test in the suite currently pins the old behaviour. The closest,
`tests/integration/cli/test_staged_gpu_local.py:1039`
(`test_process_objmap_export_writes_the_detected_labels_not_zeros`), builds
`ImagePipeline(ops=[FakeGpuDetector(threshold=0.3)])` — a pipeline with **no
post-detector ops**, so old and new semantics coincide. Its assertions
(`labels.max() > 0`, `stored.max() == 0`) both still hold after the change.

That test remains valid — it guards FLOW-16, that the export never reads the
store — but its docstring becomes stale and must be updated. A **new** test is
required to pin the new semantics (§10).

---

## 9. Change inventory

| # | Where | Change |
|---|---|---|
| 1 | `_cli_validation.py:147` | `pipeline_requires_gpu` → recursive walk. **The CPU-only-slot refusal must fire on the production path**, not only under a `strict=` argument that only `split_pipeline_at_gpu` passes — see the risk table |
| 2 | new shared module | The tree walker, consolidated; **migrate** `gui/_operation_registry.py:33` and `tune/_search_space/_infer.py:429,546,703` onto it (§4.1) |
| 2a | `gui/`, `tune/` | Regression pass for each, as consumers of the consolidated walker |
| 3 | `_cli_pipeline_split.py` | `gpu_key` → `gpu_path`; add `stage2_prefix`; cut at `gpu_path[0]`, ancestor heads `post_pipeline` |
| 4 | *new* | `ReplayDetector` + path-addressed `substitute()` |
| 5 | `_cli_staged_workers.py` `stage2_detect_core` | Apply `stage2_prefix` to a provenance-detached copy before reading `input_layer` (§4.3) |
| 5a | `_cli_stage2_token.py:54,145` | **Key the raw array and the token by detector slot** (§4.4). Add `slot` to `stage2_raw_path` / `stage2_token_path` and every reader/writer/predicate |
| 5b | `_cli_stage2_token.py` | One-time resume relocation: legacy per-image path → slot path, when the plan has exactly one slot (§4.4) |
| 6 | `_cli_staged_workers.py` `stage3_merge_measure_core:490` | Replace the explicit `_write_object_output` + `append_operation_provenance` pair with stub substitution; stub carries identity + duration |
| 7 | `_cli_staged_strategy.py` `_export_objmap_layer:397` | Run the post-detector op chain (top-level **and** nested) via the same stub substitution, with in-memory-only provenance handling and no checkpoint write (§8.1, §8.2) |
| 7a | `_cli_failure_tracker.py:191-236` | Add an output-semantics revision constant to `processing_configuration_digest_from_values`, bumped by this change (§8.3) |
| 7b | root `CLAUDE.md` | The sentence "`--mode process --layer objmap` exports objmaps after Stages 1–2" becomes wrong; also the `--mode process` bullet describing the objmap export |
| 7c | `docs/source/how_to/` | Any page documenting the objmap export semantics |
| 7d | `tests/integration/cli/test_staged_gpu_local.py:1039` | Docstring is stale — the export now replays *and refines*. Assertions stand (§8.4) |
| 7e | *new shared helper* | Per-branch `pipeline_step` push for container ops (§5.2) |
| 7f | `detect/_composite_detector.py`, `enhance/_composite_enhance.py`, `detect/_filamentous_fungi_detector.py`, `detect/_two_k_filamentous_detector.py` | Adopt the helper; children applied under `ops[i]` |
| 7g | `measure/_canonical_zone_measure.py` | **No change** — `center_detector` stays step-pathless by design (§5.2); pin with a test |
| 8 | `_cli_pipeline_split.py:56-68` | Plot-reference guard: the ancestor key moves from pre to post, so `ref.key in pre_ops` must be re-derived |
| 9 | `_cli_staged_slurm_worker.py:182,296,448` | Three `split_pipeline_at_gpu` call sites — inherit the new plan |
| 10 | tests | §10 |

---

## 10. Testing

- **Unit — walker:** finds a GPU detector at depth 1, 2, and inside a list;
  returns `field[i]` string paths; finds detectors in `meas` so they can be
  refused.
- **Unit — split:** `gpu_path`, `pre`/`post` membership, and `stage2_prefix`
  for both the composite (empty) and the nested-pipeline (non-empty) case.
- **Unit — refusals:** each row of the §7 refusal table raises with a message
  naming the offending path.
- **Equivalence (the gate):** staged vs single-pass produce an identical objmap
  and identical measurements, for all three shapes of §6.
- **Mutation control:** the equivalence test must **fail** when the recorded
  Stage-2 array is perturbed. Without this the gate is vacuous — note the
  object count is unchanged under the perturbation, so the assertion must
  compare the objmap, not a count.
- **Provenance:** the staged journal matches the single-pass journal in class,
  parameters, order, and `pipeline_step_path`.
- **Step-path descent (§5.2):** each of the four container classes records
  `ops[i]` for its children; shape C yields five *distinct* paths where it
  previously yielded five identical ones.
- **Step-path exclusion (§5.2):** a zone measurer's `center_detector` records
  **no** step path. Without this the exclusion is indistinguishable from an
  oversight and will be "fixed" away.
- **The addressing invariant (§5.3):** for every supported shape, the walker's
  `gpu_path` equals the `pipeline_step_path` recorded for that detector. This is
  the assertion that keeps the two schemes from silently diverging later.
- **Owner-depth-0:** Stage 3 exercised with `_application_owner_depth` forced to
  0, per the `measure/CLAUDE.md` rule. A test at the default programmatic depth
  passes on broken code.
- **Process-mode semantics (new, §8.4):** a pipeline with a GPU detector **and**
  post-detector ops that measurably change the objmap (e.g. `SmallObjectRemover`
  that removes a known blob). Assert the exported PNG equals the pipeline's
  objmap and **differs** from the detector's raw output. The existing test at
  `:1039` cannot catch this — its pipeline has no post-detector ops — so this
  test must be written with a pipeline that does. Run it for both a top-level
  and a nested detector.
- **Process-mode provenance:** the export path exercised at
  `_application_owner_depth == 0` against a `"staged"` store, asserting it does
  not raise **and** that the store on disk is byte-unchanged afterwards (the
  no-checkpoint invariant of §8.2).
- **Continuation across the semantics bump (§8.3):** two runs whose configs are
  identical except for the output-semantics revision produce **different**
  work ids, so the second does not reuse the first's outputs.
- **Regression scope:** `tests/unit/cli/test_staged_*.py` and
  `tests/integration/cli/test_staged_gpu_local.py` are the directly touched
  surface; `tests/unit/cli/test_cli_process_only.py` and
  `test_process_format_option.py` are reached through the digest change. Derive
  the wider surface from importers of any shared helper changed, per the repo
  testing rule. Full sharded regression once, at the end.

---

## 11. Open questions

1. ~~**§8** — refuse vs. post-detector op chain.~~ **Resolved 2026-09-15:** run
   the chain for both, on the principle that the ops a user supplies as
   parameters should be the ops that actually run. Consequences and mitigations
   are §8.2–§8.4.
2. ~~**Change 2** — consolidate the walker or duplicate it?~~ **Resolved
   2026-09-15: consolidate**, migrating `gui/` and `tune/` onto the shared
   traversal. See §4.1 for the blast-radius consequences the plan must carry.
3. ~~**Step-path descent** — separate spec or folded in?~~ **Resolved
   2026-09-15: folded in** (§5.2, §5.3). The deciding argument was §5.3: the
   staged engine needs a path to address the GPU op, and provenance needs a path
   to record it. Kept apart, those are two schemes maintained by hand; folded
   together, they are one measured invariant.

**No open questions remain.** The spec is ready for an implementation plan.

---

## 12. Risks

| Risk | Mitigation |
|---|---|
| A third tree walker diverges from the two that exist | Change 2 — consolidate; this is the primary design risk |
| Equivalence test passes vacuously | Mandatory mutation control (§10) |
| Provenance regression invisible at the default owner depth | Mandatory owner-depth-0 test (§10) |
| A non-deterministic branch prefix makes Stage 2 and Stage 3 disagree | Documented bound (§7); the repo's fixed-seed convention applies |
| Silent CPU fallback persists for any shape not covered | Every unsupported shape refuses loudly rather than falling through (§7) |
| A process run resumed across the upgrade mixes old- and new-semantics objmaps | Output-semantics revision in the work-id digest (§8.3). **Highest-severity risk in this design** — it is silent by default and produces a tree that looks complete |
| The deliberate no-checkpoint omission in process-mode provenance is "fixed" by a later reader | Comment at the site explaining FLOW-16/FLOW-30/FLOW-6 (§8.2), plus the byte-unchanged store test (§10) |
| The objmap semantics change lands unobserved | New pinning test with a pipeline that actually has post-detector ops (§8.4, §10) |
| **Journals change for every composite pipeline, GPU or not (§5.2)** | A second version boundary, wider than §8.3's: the repo's "two identical runs write byte-identical stores" guarantee holds only *within* a version across this change, since the journal lives in the store. Must be called out in release notes — there is no digest to bump here, because the change is to recorded output, not to work identity |
| The deliberate `center_detector` step-path exclusion is "completed" by a later reader | Test asserting it records no step path (§5.2, §10) |
| `gpu_path` and `pipeline_step_path` drift apart after the fact | The §5.3 identity is asserted as a test, not left as a convention (§10) |
| **A refusal that production never reaches.** If `pipeline_requires_gpu` scans non-strictly, a GpuDetector in `meas`/`post`/`filters`/`model` yields `False`, routes to the non-staged strategy, and silently runs on CPU — the exact bug this change exists to kill — while a unit test calling the refusal directly still passes | The refusal must be reachable from `pipeline_requires_gpu` itself. Because that function is also called from `gui/run_console/_callbacks.py:253-255`, where an exception is unwelcome, prefer: always scan every slot, return the ops-slot hits, and raise for a CPU-only-slot hit regardless of any `strict` flag. Test through `pipeline_requires_gpu`, never through the helper |
| A mitigation documented but never exercised | §8.2 shipped a wrong fix because its probe reproduced the trap and stopped. Every provenance mitigation in this spec must have a test that runs **the mitigation**, at `_application_owner_depth == 0` |

---

## 13. Planned extension: more than one GPU detector

This spec ships `N = 1`. The shape below is **not implemented here**, but §4.4's
slot keying exists to make it additive rather than structural, so it is recorded
now while the reasoning is fresh.

### 13.1 The two axes are not equally expensive

| Axis | Meaning | Where it lands |
|---|---|---|
| **N** — detectors per round | Detectors reading the **same** upstream state, e.g. `CompositeDetector(ops=[Sam2, Dino, ManualPointDetector])`. Composite branches each get `apply(image, inplace=False)` on the same input, so no branch depends on another. | worker + storage layer |
| **R** — rounds | Detectors where one's input depends on another's output, e.g. `CompositeA(Sam2, …) → MaskDilation → CompositeB(Dino, …)`. | scheduler orchestration |

Generalised, a pipeline is segments:

```
CPU(S0) → GPU(round 1) → CPU(S1) → GPU(round 2) → … → CPU(S_R) → measure
```

and this spec is the `R = 1, N = 1` corner. **`N > 1` and `R > 1` are separate
pieces of work and should stay separate** — `N` touches the worker and the
signal layout, `R` touches the epoch-fenced controller, which is the component
where a mistake yields artifacts interleaved from two attempts rather than a
clean failure.

### 13.2 The intended execution model for `N > 1`

**One detector at a time across the whole dataset; parallel across images
within each sweep.** Stage 2 becomes `N` sub-sweeps:

```
Stage 2a   load Sam2 once per worker   → sweep every image → stage2_raw/<sam2-slot>/…
Stage 2b   load Dino once per worker   → sweep every image → stage2_raw/<dino-slot>/…
Stage 3    replay both slots, composite merges, both raws consumed and deleted
```

This is deliberately **not** "hold N resident models per worker". Two
foundation models resident on one ada6000 is a memory gamble that fails late,
after all the CPU preprocessing, in the same way a Pascal placement does. One
model per worker keeps the existing residency invariant exactly and buys
parallelism from array width instead of GPU memory.

Consequences, all additive on top of §4.4:

- `StagePlan` carries a **list** of slots rather than one.
- Stage 3 folds `substitute_at_path` over the slots — it already composes.
- The continuation predicate becomes *every* slot present for an image, not
  *the* file present.
- Intermediates are deleted per slot, as today: token consumed, then raw.

### 13.3 What `R > 1` would additionally need

Recorded for completeness; no user requires it today.

- The controller chains `2R + 1` job groups instead of 3; the append-only
  ledger, the recovery derivation and the epoch fence all scale with `R`.
- `submit_capacity = max_submit - 2` (`_cli_staged_slurm.py:557`) reserves two
  slots for the running controller and its pre-armed recovery controller. Both
  that reservation and the
  `ceil(n_images / min(MaxArraySize, MaxSubmitJobs - 2))` chunking change with
  `R`, and at 33,923 images these limits are already close.
- **No new artifact type is needed**, which is the one cheap part: each CPU
  segment re-promotes the per-image store, so the store already carries state
  between rounds.

### 13.4 Binding constraint: every GPU round is a full-dataset sweep

**A GPU round processes every image with the model resident, then hands off.
It is never interleaved per image, however many rounds there are.**

The tempting implementation of `R > 1` is to follow one image all the way
through — GPU round 1, CPU segment, GPU round 2 — and then move to the next
image. It is tempting because it needs no barrier and no extra job group, which
is exactly the cost §13.3 is trying to avoid. It is wrong, and it forfeits the
entire reason the staged engine exists:

- **Model residency is destroyed.** Each image would rebuild the model, which is
  precisely the cost the non-staged path already pays at
  `_cli_process_single.py:260` and the defect that motivated staging in the first
  place. With `R` rounds it is paid `R × n_images` times instead of `R` times per
  worker — on 33,923 images, catastrophically.
- **Batching becomes impossible.** A detector that declares
  `supports_batching = True` can only fill a batch if the worker holds several
  images at once. Per-image interleaving guarantees it never sees more than one.

So the arrows in `CPU(S0) → GPU(round 1) → CPU(S1) → GPU(round 2) → …` are
**dataset-wide barriers**, not per-image transitions. Round `r` sweeps every
image; the CPU segment then runs across every image; round `r+1` sweeps again.
This is what makes `R > 1` expensive in **scheduling** — `2R + 1` job groups —
while costing nothing in **per-image GPU efficiency**, which stays identical to
`R = 1`. That trade is the right way round, and an implementation that reverses
it has optimised the cheap axis.

The same property holds within a round for `N > 1`: the `N` sub-sweeps of §13.2
are each a full pass over the dataset, one model resident per pass.

**Batching headroom, currently unused.** Today's Stage 2 obtains residency but
not batching — it calls `detector._collate([sample])` with a **one-element**
list, per image (`_cli_staged_workers.py:389`), and the only other caller is the
single-image notebook path (`abc_/_gpu_detector.py:261`). Nothing in the engine
ever collates across images, so `supports_batching` buys nothing yet. That is a
gap, not a decision, and it is worth naming here because the sweep structure is
its **precondition**: a worker streaming a shard of a full-dataset sweep can
accumulate samples and issue one `(N, C, H, W)` forward, whereas a per-image
interleave forecloses it permanently. Preserve the sweep and the optimisation
stays available; abandon it and no amount of later work recovers it.
