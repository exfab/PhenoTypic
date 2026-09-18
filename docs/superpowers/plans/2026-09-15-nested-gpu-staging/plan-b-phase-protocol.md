# Plan B — Staging via a Declared Phase Protocol

> **Second of two plans, in sequence — Plan A lands first.** See `README.md`.
> Plan B redesigns how an operation declares its execution so the scheduler can
> split it, instead of inferring the split from outside. It supersedes Plan A's
> splitter, replay stub and branch-prefix machinery, and reuses the rest — see
> *What carries over from Plan A* for which parts, per task.
>
> **Status (2026-09-17): blocked on design decisions.** Plan A is implemented
> (PR #224, `1fcab4bf`), reviewed twice plus an implementation review, and passed
> a real-GPU smoke run (`../../reports/2026-09-15-nested-gpu-staging/gpu-smoke.md`).
> This plan was re-checked against the built code in `../../reports/2026-09-15-nested-gpu-staging/plan-b-staleness-review.md`; its §4 lists the
> forks that must be decided before a spec can be written. **For `F1gfd5` the
> built Plan A already runs every operation exactly once** (the in-branch prefix
> is empty), so this plan's headline saving is hypothetical for today's
> pipelines. Plan B still has no spec of its own and must get one, plus a review,
> before execution.

**Goal:** Let any operation declare its execution as an ordered list of phases with a device tag, so the staged engine derives the CPU/GPU round chain statically from the pipeline JSON rather than inferring where a GPU detector sits.

**Architecture:** Every `ImageOperation` carries `_phases` — a tuple of `(method_name, device)`. The default, `(("_operate", "cpu"),)`, leaves every existing operation working unchanged. An operation that wants to be split across stages declares more phases and names the `PrivateAttr`s that carry state between them. Containers additionally declare `_child_order` so their children's phases can be flattened in the right order. The round chain is the flattened device sequence with adjacent same-device phases coalesced.

**Tech Stack:** Python 3.11+, pydantic v2 (`PrivateAttr`, `ClassVar`), zarr v3 / OME-Zarr 0.5, numpy, pytest, `uv`.

**Spec:** none yet — see the banner above. The design below is the contract until one exists.

---

## Design

### The declaration

```python
class ImageOperation(BaseModel):
    #: Ordered execution phases as (method_name, device). The default keeps every
    #: existing operation working unchanged and unstageable.
    _phases: ClassVar[tuple[tuple[str, str], ...]] = (("_operate", "cpu"),)

    #: PrivateAttr names that carry state ACROSS a phase boundary. Every entry
    #: must resolve to a declared PrivateAttr on this class.
    _phase_state: ClassVar[tuple[str, ...]] = ()

    #: For operations with OperationField children. "sequence" = each child
    #: receives the previous child's output; "parallel" = every child receives
    #: this operation's own input. Absent = children cannot be flattened, so a
    #: GpuDetector nested inside is refused.
    _child_order: ClassVar[Literal["sequence", "parallel"] | None] = None
```

Method **names**, not bound methods — Stage 3 rebuilds the operation from JSON in a fresh process, so a bound method cannot travel. `ClassVar`, so the phase structure is a property of the class. Single underscore, because `__phases` would name-mangle and break override in subclasses.

### What this replaces

| Plan A mechanism | Plan B |
|---|---|
| `pipeline_requires_gpu` walks the tree (slots included) for `isinstance(op, GpuDetector)` | `any(device == "gpu" for _, device in flatten_phases(pipeline))`. Both are tree scans; the difference is that the answer comes from a declaration, not `isinstance` |
| `_CHILD_CONTRACT` table in `_cli_validation.py`, "composition primitives only" rule, guard test over class kinds | `_child_order` on the class. Declaring `_phases` + `_child_order` **is** the opt-in; an operation that has not declared is unstageable, by construction rather than by rule |
| `StagePlan` with `gpu_path` / `stage2_prefix` | the flattened phase chain |
| `ReplayDetector` + `substitute_at_path` (consumed by Stage 3, the process-mode objmap export, and the provenance hook protocol) | **gone** only if parents stop being replayed (review fork F-2). The GPU phase persists its result; the next CPU phase reads it |
| Stage-2 branch prefix, re-executed in Stage 3 (empty for `F1gfd5`) | **gone.** Each phase runs exactly once |

`GpuDetector` stops being special-cased. Its existing hooks already have this shape:

```python
_phases = (("_phase_prep", "cpu"), ("_phase_infer", "gpu"), ("_phase_write", "cpu"))
#            _preprocess/_collate      _infer_batch          _write_object_output
```

### Cross-phase state

Declared `PrivateAttr`s, ferried by the engine:

```python
class FilamentousFungiDetector(ObjectDetector):
    _phases: ClassVar = (("_prep", "cpu"), ("_gpu_detection", "gpu"), ("_post_gpu_work", "cpu"))
    _child_order: ClassVar = "parallel"
    _phase_state: ClassVar = ("_inoculum_objmask", "_inoculum_objmap")

    _inoculum_objmask: np.ndarray | None = PrivateAttr()   # no default: see B3
    _inoculum_objmap:  np.ndarray | None = PrivateAttr()
```

In-process (notebook, CPU-only run) nothing is persisted — the attributes simply survive on the instance. Across stages the engine reads them after a phase, persists them, and restores them onto the rebuilt instance before the next. The author writes the same code either way.

`PrivateAttr` is load-bearing, not stylistic: it keeps per-image state out of the constructor (a user cannot pass a corrupted mask), out of `model_dump(mode="json")` — so it never reaches the pipeline JSON *or* the provenance `parameters`. Plan A avoids that leak for `ReplayDetector`'s `NdArrayField` with an explicit `provenance_parameters` hook (`_cli/_cli_replay_detector.py:87-89`); a `PrivateAttr` avoids it structurally — and out of `model_fields`, so the `OperationField` walker cannot mistake it for a child.

### Round chain

Flatten the phase tree using `_child_order`, coalesce adjacent same-device phases:

```
flattened:    cpu cpu cpu │ gpu │ cpu cpu
round chain:  [CPU job]  →  [GPU job]  →  [CPU job]
```

Static, identical for every image, computable before any image is touched. **One round per sub-pipeline** — independent rounds for `"parallel"` children, so each branch's intermediates live and die inside its own round and peak storage stays flat in branch count.

---

## Global Constraints

Inherit every constraint from Plan A (`uv` only, keyword-only pydantic construction, `pipeline_step_path` as non-empty strings, Stage 2 never writes the store, `_export_objmap_layer` never writes the store, ruff with explicit paths, the `run-phenotypic-test` skill, no `-n auto`), plus:

- **A GPU round is a full-dataset sweep, never a per-image interleave** (Plan A's constraint, unchanged — breaking it destroys residency and forecloses batching).
- **`self` is a channel only for declared `_phase_state` names.** Anything else set on the instance in one phase is absent in the next, and absent usually reads as a default rather than an error.
- **Per-image `PrivateAttr` state rules out caching a pipeline instance across images.** Two images sharing one object would clobber each other's phase state. This is now structural.
- **Slot placements stay refused** (Plan A revision entries 14–17): a `GpuDetector` in any pipeline's `meas`/`post`/`filters`/`model` slot, at any depth, is refused by `refuse_cpu_only_slot`, independent of the phase scan. Slot entries are spelled `meas:<key>`, `post:<key>`, `filters:<key>`, `model:<ClassName>`. The refusal fires before `--overwrite` and `--dry-run`, and the GUI run console shows it in the `rc-staged-gpu-refusal` alert.
- **The model loads once per GPU round, and only when the round has pending images** (`_cli_staged_strategy.py:228-230`); inference goes through `_infer_batch`, pinned by `tests/unit/abc_/test_infer_batch_loads_the_model.py`.
- **Large intermediates are layers, not phase state.** The engine caps the ferried payload; anything above it must be written as an image layer, where the store already handles persistence, versioning and cleanup.

---

## What carries over from Plan A

Do these first; they are foundation under either plan and are already reviewed.

Corrected against the built code; evidence per row in `../../reports/2026-09-15-nested-gpu-staging/plan-b-staleness-review.md` §2.

| Plan A task | Status under B |
|---|---|
| 1 — shared operation-tree traversal | **reused** (now also walks slots; `substitute_at_path` dies only if replay is dropped) |
| 2 — tune consolidation | **reused**, independent of staging |
| 3 — tree-wide detection + `UnstageableGpuDetectorError` | **partly reused**: `find_gpu_detectors`, `refuse_cpu_only_slot`, the error, the CLI preflight and the GUI alert stay; `_CHILD_CONTRACT`, `_child_contract` and `validate_ancestor_contracts` are replaced, and their tests retire or invert |
| 4 — container ops push a per-branch `pipeline_step` | segment scheme **reused**; the mechanism is reused only if phases run inside `apply()` (review F-7) |
| 6a — slot-keyed Stage-2 signal | **reused**; `staged_detector_slot` and every `plan.gpu_path` call site are repointed from `StagePlan` |
| 9 — recorded paths resolve | the resolvability test is **reused** as is; the gpu-path test is rewritten against the phase chain |
| 10 — equivalence + mutation control | mutation-control and owner-depth patterns **reused**; shapes rewritten; prefix-specific tests retired |
| 11 — process mode post-detector chain | **rewritten**: the export substitutes a `ReplayDetector` today; the semantics and the no-store-write / no-success-sink constraints carry over |
| 12 — continuation digest revision | **reused for process mode**; full-run invalidation is a new decision (review F-8) |
| 14 — docs | **rewritten**, including the doc-pin tests (see B7) |
| 15 — regression | **reused** (procedure) |
| 5 — path-shaped `StagePlan` | superseded; 8 consumers to repoint |
| 6 — `ReplayDetector` · 8 — Stage-3 stub substitution | superseded only under review F-2 option (a) |
| 7 — Stage-2 branch prefix · 13 — forward `stage2_prefix` | superseded |

---

## Task B1: The phase declaration and its guards

**Files:**
- Modify: `src/phenotypic/abc_/_image_operation.py` (the three `ClassVar`s)
- Create: `src/phenotypic/sdk_/_phases.py` (`Phase`, `phases_of`, `validate_phase_declaration`)
- Test: `tests/unit/sdk_/test_phase_declaration.py`

**Interfaces:**
- Produces: `phases_of(op) -> tuple[Phase, ...]`, `validate_phase_declaration(cls) -> None`

- [ ] **Step 1: Write the failing test**

```python
def test_the_default_keeps_an_ordinary_operation_unchanged():
    from phenotypic.detect import OtsuDetector
    assert phases_of(OtsuDetector()) == (Phase("_operate", "cpu"),)


def test_every_phase_method_exists_on_the_class():
    class Broken(ObjectDetector):
        _phases = (("_prep", "cpu"), ("_typo_here", "gpu"))
        def _prep(self, image): return image
    with pytest.raises(ValueError, match="_typo_here"):
        validate_phase_declaration(Broken)


def test_every_phase_state_name_is_a_declared_private_attr():
    """A typo here ferries nothing and restores nothing, silently."""
    class Broken(ObjectDetector):
        _phases = (("_operate", "cpu"),)
        _phase_state = ("_not_declared",)
    with pytest.raises(ValueError, match="_not_declared"):
        validate_phase_declaration(Broken)


def test_a_device_must_be_cpu_or_gpu():
    class Broken(ObjectDetector):
        _phases = (("_operate", "tpu"),)
    with pytest.raises(ValueError, match="tpu"):
        validate_phase_declaration(Broken)
```

- [ ] **Step 2–4:** run (fails on import), implement, run (passes).

- [ ] **Step 5: Wire validation into the pydantic model build**

Wire validation into `ImageOperation.__pydantic_init_subclass__`, calling `super().__pydantic_init_subclass__(**kwargs)` first (`abc_/_image_operation.py:396-416`), so a bad declaration fails at class definition. A `model_validator` runs per instance and cannot. `validate_phase_declaration` opens with an `isinstance(cls._phases, tuple)` check.

- [ ] **Step 6: Commit**

---

## Task B2: Flatten a pipeline tree into a phase chain

**Files:**
- Create: `src/phenotypic/_cli/_cli_phase_chain.py`
- Test: `tests/unit/cli/test_phase_chain.py`

**Interfaces:**
- Consumes: Task 1's traversal, `phases_of`.
- Produces: `flatten_phases(pipeline) -> list[PhaseStep]` where `PhaseStep` carries `(path, method, device)`; `round_chain(steps) -> list[Round]`.

- [ ] **Step 1: Write the failing test**

```python
def test_a_cpu_only_pipeline_is_one_round():
    # FakeGpuDetector: tests._fakes.fake_gpu_detector, registered as in
    # tests/unit/cli/test_gpu_detection_tree_wide.py for any from_json path.
    chain = round_chain(flatten_phases(ImagePipeline(ops=[BlurGauss(), OtsuDetector()])))
    assert [r.device for r in chain] == ["cpu"]


def test_a_gpu_detector_yields_cpu_gpu_cpu():
    chain = round_chain(flatten_phases(ImagePipeline(ops=[BlurGauss(), FakeGpuDetector()])))
    assert [r.device for r in chain] == ["cpu", "gpu", "cpu"]


# BLOCKED on review fork F-5: two GPU detectors are refused today
# (find_gpu_detectors(strict=True)), and "one round per sub-pipeline"
# contradicts spec §13.2's N-sub-sweeps model.
def test_parallel_children_get_one_round_each():
    """The per-sub-pipeline round decision: branch intermediates stay scoped."""
    pipe = ImagePipeline(ops={"C": CompositeDetector(ops=[
        ImagePipeline(ops=[ContrastStretching(), FakeGpuDetector()]),
        ImagePipeline(ops=[BlurGauss(),          FakeGpuDetector()]),
    ], mode="overlap")})
    assert [r.device for r in round_chain(flatten_phases(pipe))] == [
        "cpu", "gpu", "cpu", "gpu", "cpu"]


def test_a_container_without_child_order_is_refused():
    pipe = ImagePipeline(ops={"TwoK": TwoKFilamentousDetector(branch_base=FakeGpuDetector())})
    with pytest.raises(UnstageableGpuDetectorError, match="_child_order"):
        flatten_phases(pipe)


def test_the_chain_is_identical_across_images():
    """Static scheduling: the chain depends on the pipeline, never on data."""
    pipe = ImagePipeline(ops=[BlurGauss(), FakeGpuDetector()])
    assert round_chain(flatten_phases(pipe)) == round_chain(flatten_phases(pipe))
```

- [ ] **Steps 2–5:** implement, verify, commit.

---

## Task B3: Ferry `_phase_state` across a boundary

**Files:**
- Create: `src/phenotypic/_cli/_cli_phase_state.py`
- Test: `tests/unit/cli/test_phase_state_ferry.py`

**Interfaces:**
- Produces: `capture_phase_state(op) -> dict`, `restore_phase_state(op, payload) -> None`, `clear_phase_state(op) -> None`.

- [ ] **Step 1: Write the failing test**

```python
def test_round_trips_declared_state_only():
    op = _Phased()
    op._carried = np.arange(9, dtype=np.uint16)
    op._not_carried = "should not survive"
    payload = capture_phase_state(op)

    fresh = _Phased()
    restore_phase_state(fresh, payload)
    assert np.array_equal(fresh._carried, op._carried)
    assert fresh._not_carried is None


def test_unset_is_distinguishable_from_none():
    """A conditional phase that skips an assignment must not restore a default
    that reads as a real value."""
    op = _Phased()
    op._carried = None
    payload_explicit_none = capture_phase_state(op)
    payload_unset = capture_phase_state(_Phased())
    # Compare by key set, then np.array_equal per value -- never ==/!= on
    # dicts holding ndarrays, which raises.
    assert payload_explicit_none.keys() != payload_unset.keys()


def test_clear_resets_between_images():
    """Stale state from image A must not leak into image B when a conditional
    phase on B skips the assignment."""
    op = _Phased()
    op._carried = np.arange(4)
    clear_phase_state(op)
    assert capture_phase_state(op).keys() == capture_phase_state(_Phased()).keys()


def test_an_oversized_payload_is_refused_with_a_pointer_to_layers():
    op = _Phased()
    op._carried = np.zeros((3140, 5094), dtype=np.float64)   # 122 MB
    with pytest.raises(ValueError, match="layer"):
        capture_phase_state(op)
```

- [ ] **Steps 2–5:** implement (ndarray → compressed payload at the location decided by review fork F-6, JSON-able → token; there is no `branch/` schema today), verify, commit.

---

## Task B4: Drive the phase chain in the staged workers

**Files:**
- Modify: `src/phenotypic/_cli/_cli_staged_workers.py`, `_cli_staged_slurm_worker.py`, `_cli_staged_strategy.py`
- Modify: `src/phenotypic/abc_/_gpu_detector.py` (declare its three phases)
- Modify (consumers of the superseded splitter/replay, per review E-9): `_cli/_cli_replay_detector.py`, `_cli/_cli_pipeline_split.py`, `_cli/_cli_validation.py`, `_cli/_cli_stage2_token.py` (`staged_detector_slot`), `phenotypicCLI.py` (`staged_detector_slot` callers), `_cli/_cli_checkpoint_handler.py`, `_cli/_cli_staged_slurm.py`, `_cli/_cli_staged_controller.py`, `_cli/_cli_staged_resume.py`, `_cli/_stages.py`, `sdk_/_image_record.py` (`STAGE_STAGE1..3`), `_cli/_cli_image_record.py`, `_core/_provenance.py` (hook docstring), `_gui/run_console/_callbacks.py` (refusal text)

- [ ] **Step 1:** Replace `stage1_preprocess_core` / `stage2_detect_core` / `stage3_merge_measure_core` with a single `run_phase_round(round, image, ...)` driven by the chain. The three existing functions become the CPU/GPU/CPU instances of it.
- [ ] **Step 2:** `GpuDetector` declares `(_phase_prep, cpu), (_phase_infer, gpu), (_phase_write, cpu)`, with the inference result carried by the mechanism chosen in review fork F-4 (the as-built Stage-2 raw file + token, or `_phase_state`). The `ReplayDetector` path has three consumers (Stage 3, the process-mode export, the provenance hooks); replace each, do not just delete. Load the model once per GPU round, only when it has pending images; `_phase_infer` calls `_infer_batch`, never `_infer_one`.
- [ ] **Step 3:** `clear_phase_state` at the start of every image's phase sequence.
- [ ] **Step 4:** Give each staged suite a disposition (keep / rewrite / retire), decided once forks F-2..F-5 are:
  `test_staged_resume.py`, `test_staged_resume_equivalence.py`, `test_staged_store_stages.py`, and the suites Plan A added — `tests/unit/cli/`: `test_replay_detector.py`, `test_pipeline_split_nested.py`, `test_staged_stage2_prefix.py`, `test_staged_nested_equivalence.py`, `test_cli_pipeline_split.py`, `test_gpu_detection_tree_wide.py`, `test_stage2_slot_keying.py`, `test_cli_gpu_refusal.py`; `tests/unit/detect/test_container_child_contracts.py`; `tests/integration/cli/test_process_objmap_semantics.py`; `tests/integration/gui/test_run_console_callbacks.py`; `tests/unit/test_docs_staged_cli.py`.
  Keep unconditionally: `test_recorded_paths_resolve.py` (resolvability), the owner-depth test in `test_staged_nested_equivalence.py`, `test_stage2_slot_keying.py`, `test_cli_gpu_refusal.py`, `test_infer_batch_loads_the_model.py`.
- [ ] **Step 5:** Commit.

---

## Task B5: The equivalence gate, generalised

**Files:**
- Test: `tests/unit/cli/test_phased_equals_monolithic.py`

**This is the gate for the whole plan.** A wrong phase split is *silently* wrong: state that should cross doesn't, and the missing value reads as a default rather than raising.

- [ ] **Step 1:** For every class declaring more than one phase, assert the phased path produces an identical objmap and identical measurements to running its phases back-to-back in one process.
- [ ] **Step 2: Mutation control.** Perturb the ferried payload and assert the comparison **fails**. Without this the gate is vacuous.
- [ ] **Step 3: Instance-state poisoning.** Between phases, replace the operation instance with a freshly-rebuilt one carrying only the declared `_phase_state`. Any phase that secretly depended on an undeclared attribute now fails — which is the point. This is the test that catches the `self`-is-not-a-channel trap.
- [ ] **Step 4:** Owner-depth-0 coverage, per `measure/CLAUDE.md`.
- [ ] **Step 5:** Commit.

---

## Task B6: Make `FilamentousFungiDetector` stageable

The worked example of the whole plan. Plan A refuses it by policy, not by mechanism (`_cli_validation.py:204-213`): one `_CHILD_CONTRACT` entry plus a behavioural probe would admit it on the replay path (review fork F-9).

- [ ] **Step 1:** Split its `_operate` at the PHASE 1 / PHASE 2 boundary — the inoculum detection is essentially the first thing it does, so the CPU work before the GPU phase is nearly nothing and the split is natural.
- [ ] **Step 2:** Declare `_phases`, `_child_order = "parallel"`, `_phase_state = ("_inoculum_objmask", "_inoculum_objmap")` as `PrivateAttr()` **with no default** — the only declaration under which B3's unset-vs-None test holds. Read state through `op.__pydantic_private__`; write it with `setattr`, never `object.__setattr__`.
- [ ] **Step 3:** Task B5's equivalence gate must pass for it.
- [ ] **Step 4:** Commit.

`TwoKFilamentousDetector` stays unstageable, now for an obvious reason rather than an argued one: its children have **mixed** order — `center_detector` reads the original image, `background_subtractor` reads a sibling's output — so no single `_child_order` is true for it, and it declares none.

---

## Task B7: Documentation

- [ ] Rewrite `docs/source/contrib_guide/gpu_detectors.md`'s container-contract section around `_phases` / `_child_order` / `_phase_state`, replacing the composition-primitives rule.
- [ ] Record the pipeline-caching collision as a constraint.
- [ ] Root `CLAUDE.md` and `src/phenotypic/_cli/CLAUDE.md`.
- [ ] Rewrite `docs/source/contrib_guide/gpu_detectors.md` from *"Nesting: the container contract"* through *"What Stage 3 actually runs: `ReplayDetector`"*, and update the doc-pin tests in `tests/unit/test_docs_staged_cli.py` that quote the nesting refusal and name the `_CHILD_CONTRACT` entries.

---

## Open questions for Plan B

1. **No spec.** The committed spec describes Plan A. Plan B needs its own, and an independent review, before execution.
2. **Does a phase return a new image or mutate in place?** In-place matches `_operate`'s convention and is the recommendation; the visibility concern that argued for return-style turned out to be unfounded — the real hazard is instance state, which Task B5 step 3 covers.
3. **Conditional GPU phases** are supported and cost a near-empty array submission when no image requests inference, because the round's task list derives from disk. Confirm that is acceptable rather than worth a static "this pipeline never infers" check.
4. **Flattening order for `"parallel"` children** is currently one round per sub-pipeline (storage-scoped). The alternative — coalescing all branches' CPU phases, then all their GPU phases — is fewer rounds and one model load, at the cost of every branch's intermediates being live simultaneously. Revisit if round count becomes the binding constraint.

---

## Measurements taken while designing this (keep — they are perishable)

All at production size, `3140×5094` (a Linzer `.CR3` after `F1gfd5`'s crop),
33,923 images in the run.

| What | Time | Peak RSS | Across the run |
|---|---|---|---|
| `FocusEdgePhase._phasecong3` — a **hypothetical in-branch prefix** (in `F1gfd5` it is a Stage-1 op and runs once under Plan A) | **71.1 s** | **7.36 GB** | 669 CPU-hours per pass |
| `CompositeDetector._filter_mask_by_overlap_bidirectional` — a **parent body** | **4.0 s** (4.5 s at `min_overlap_ratio=0.3`) | 0.84 GB | 38 CPU-hours per pass |
| `np.savez_compressed` on a 240-label objmap | 1.2 s | — | 30.5 MB → **0.23 MB** (133×) |

Why in-branch prefix determinism matters concretely: `gpu-smoke.md` §4 found
`DenoiseBlockMatch` (BM3D) output depends on the CPU model and thread count, so
under Plan A such an op *inside* the GPU branch would feed the detector pixels
that Stage 3 never reconstructs.

Three things follow, and none is obvious without the numbers:

1. **The OOM risk is the ops, not the branch masks.** A retained composite branch
   mask is 15 MB; `FocusEdgePhase` peaks at 7.36 GB for one op on one image. Run
   32 concurrently under the `exfab` cap and you need 235 GB against a 256 GB
   ceiling. **Stage 1 needs `--mem` ≈ 8 GB per task** on this pipeline;
   `DefMemPerCPU` is 1 GB, so an array that does not set it is OOM-killed.
2. **Compression changes the storage story.** The dataset-wide barrier means
   every Stage-2 output is live before Stage 3 starts: uncompressed that is
   ~1.03 TB for one detector, ~8 GB compressed. (133× is an idealised synthetic
   — clean disks, 240 labels. Real SAM2 output with noisier boundaries will do
   worse, perhaps 10–30×. Still an order of magnitude.)
3. **The hybrid buys less than it first appears.** "Phase-split the children,
   re-execute the parent body" avoids re-running the 71 s prefix but **relocates
   it onto the GPU node**, where it costs ~19 h wall with both GPUs idle rather
   than ~3.5 h on the 384-CPU pool. Getting it onto a CPU partition needs Stage 1
   to descend into branches and persist per-branch layers — which is new
   machinery and ~1 TB of transient storage. The hybrid is still strictly better
   than Plan A on this axis (one prefix run instead of two, and a simpler stub),
   but it is not the near-free capture of Plan B's value it looks like.

**Open design questions Plan B must answer** (from its review, three blockers):

- **How a parent's own phases consume its children's output.** `CompositeDetector`
  runs each child on a *copy* and then **combines** their objmaps — the production
  shape — and `_child_order` cannot express the combine. The rule needs four
  things, not one: where the child block sits among the parent's phases (a
  sentinel, so a container with children and no sentinel is a definition-time
  error); what each child receives, **per-slot** rather than per-class; what the
  parent gets back, including a second phase signature; and where that handoff
  lives between rounds, which collides with "Stage 2 never writes the store".
- **The device tag is a `ClassVar`, but for a container it is instance state.** A
  `FilamentousFungiDetector` with a CPU `inoculum_detector` has no GPU work; one
  with `Sam2` does. As written, every default all-CPU instance advertises a GPU
  phase and routes a whole run to `exfab`.
- **Driving phases bypasses `apply()`**, which is where provenance,
  `pipeline_step_path` and the error wrapper live — falsifying the "Task 4 reused
  unchanged" and "Task 9 reused" carry-over claims.

**Reassessed 2026-09-17** in `../../reports/2026-09-15-nested-gpu-staging/plan-b-staleness-review.md` §3–4: all three blockers are still open,
blocker 3 now has a working pattern in the built code (the final phase runs
inside `apply()` with the `provenance_*` hooks and a forwarded duration), and
open question 4 conflicts with spec §13.2. Eleven forks (F-1 … F-11) need a
decision before a spec; F-1 asks whether this plan is still worth doing.
