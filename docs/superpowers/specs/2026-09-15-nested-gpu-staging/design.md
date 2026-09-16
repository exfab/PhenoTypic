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

- More than one `GpuDetector` per pipeline. The existing single-detector rule
  stands, extended tree-wide.
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

`pre_pipeline` is cut at `gpu_path[0]`; the ancestor itself heads
`post_pipeline`.

### 4.3 Stage-2 branch prefix

If the GPU op sits behind CPU operations *inside its own branch* (a nested
`ImagePipeline` such as `[ContrastStretching, Sam2]`), Stage 2 cannot read the
store layer directly. It must apply that prefix first.

The rule, derived from how each container drives its children:

> Walk the ancestor chain. A nested **`ImagePipeline`** contributes the ops that
> precede the branch. A **`CompositeDetector`** contributes **nothing** — its
> `ops` are parallel branches, each applied to the same input image via
> `inplace=False` (`_composite_detector.py:131-138`).

The prefix is applied to an **in-memory copy** inside Stage 2 and is never
written to the store, preserving the existing "Stage 2 does NOT write into the
store" invariant. It is then re-executed in Stage 3 as part of the normal branch
run. See §7 for the cost of that double execution.

**For the driver pipeline the prefix is empty** — `Sam2` is a bare leaf, so
Stage 2 reads `detect_mat` from the store exactly as it does today.

### 4.4 The replay stub

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

**Required of the replay stub:** it must report the wrapped detector's
`operation_class` and `parameters`, and carry the Stage-2 token's
`detector_duration_seconds` plus the merge duration. Stage 3 currently appends
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
| More than one `GpuDetector` anywhere in the tree | Extends the existing single-detector rule; Stage 2 emits one array per image |
| A `GpuDetector` in the `meas` / `post` / `filters` / `model` slots | Stage 3 runs these on a CPU node. The walker must scan these slots in order to reject them — the driver pipeline does carry nested ops there (`MeasureSymZones.center_detector`, `MeasureOrientationZones.center_detector`, both `ManualPointDetector`) |
| A `GpuDetector` inside `CompositeEnhance` or any enhancer container | `_write_object_output` writes an objmap; an enhancer branch must produce a layer. `OperationField` will not stop this being constructed |

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

**Mitigation.** Process mode performs Stage 3's `truncate_provenance_to_retry_base`
+ `set_provenance_status(image, "in_progress")` handling **in memory only**,
deliberately omitting `write_provenance_checkpoint`. `_export_objmap_layer`'s
invariant (FLOW-16/FLOW-30/FLOW-6) — that it never writes to the store, because a
write there invalidates the descriptor the success marker just recorded — is
preserved. This omission is intentional and must carry a comment saying so, or a
later reader will "fix" it by adding the checkpoint back.

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

**Mitigation: add an explicit output-semantics revision to the digest payload** —
a module-level integer constant, bumped whenever per-image output semantics
change, folded into `processing_configuration_digest_from_values`. Resuming
across the upgrade then re-derives every image exactly once, and the constant
documents *why* at the site that depends on it.

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
| 1 | `_cli_validation.py:147` | `pipeline_requires_gpu` → recursive walk, scanning `meas`/`post`/`filters`/`model` in order to refuse them |
| 2 | new shared module | The tree walker, consolidated; **migrate** `gui/_operation_registry.py:33` and `tune/_search_space/_infer.py:429,546,703` onto it (§4.1) |
| 2a | `gui/`, `tune/` | Regression pass for each, as consumers of the consolidated walker |
| 3 | `_cli_pipeline_split.py` | `gpu_key` → `gpu_path`; add `stage2_prefix`; cut at `gpu_path[0]`, ancestor heads `post_pipeline` |
| 4 | *new* | `ReplayDetector` + path-addressed `substitute()` |
| 5 | `_cli_staged_workers.py` `stage2_detect_core` | Apply `stage2_prefix` to an in-memory copy before reading `input_layer` |
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
