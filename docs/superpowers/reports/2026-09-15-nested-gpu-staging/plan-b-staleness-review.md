# Plan B staleness review — against Plan A as built

**Subject:** `docs/superpowers/plans/2026-09-15-nested-gpu-staging/plan-b-phase-protocol.md` (393 lines)
**Checked against:** worktree `nested-gpu-staging` at `1fcab4bf` (53 commits ahead of `origin/main`), Plan A's revision entries 1–17, `gpu-smoke.md`, and the earlier review `plan-b-review.md` (written before Plan A was executed).
**Reviewer:** plan-b staleness reviewer, 2026-09-17. Analysis only; this file is the only write.
**Commands run:** read-only (`grep`, `sed -n`, `git log`, `git diff --stat`). No probes were needed. The pydantic findings in `plan-b-review.md` (V1, V1b, M-5, M-8, m-1) came from an executed probe on pydantic 2.12.5 / pydantic-core 2.41.5, and `uv.lock:3813-3814, 3840-3841` still locks those versions, so they are carried over here without re-running.

Notation: `B:<n>` is a line of Plan B. `A:<n>` is a line of Plan A. Every code citation is a path under `src/phenotypic/` unless it says otherwise.

---

## 0. The finding that changes the framing

**For the driver pipeline, the Plan A that was actually built already runs each operation once.** Plan B's main argument is that Plan A runs a branch prefix twice, measured at *"71 s and 7.4 GB peak per image … 669 CPU-hours"* (README:51-53, B:352, B:5-7). For `F1gfd5`, that cost is **zero**:

- `Sam2` is a bare leaf at `("CompositeDetector", "ops[0]")` (`gpu-smoke.md` §1; spec `design.md:59-61`).
- `_branch_prefix` adds nothing for the root, which is guarded at `_cli/_cli_pipeline_split.py:87-90`. It also adds nothing for a `"parallel"` container (`:85-86`), and `CompositeDetector` is `"parallel"` (`_cli/_cli_validation.py:196-198`). So `stage2_prefix == []`.
- `FocusEdgePhase` is inside the top-level `CompositeEnhance`, which is in Stage 1 (spec `design.md:103-107`) and runs once.
- The spec already says so: *"Zero cost for the driver pipeline (empty prefix)"* (`design.md:248, 477-481`).
- The parent's combine (`CompositeDetector._operate`, `detect/_composite_detector.py:141-172`) runs once, in Stage 3.
- `GpuDetector`'s three hooks already run exactly once each. `_preprocess`/`_collate`/`_infer_batch` run in Stage 2 (`_cli/_cli_staged_workers.py:437-439`). `_write_object_output` runs in Stage 3 through `ReplayDetector._operate` (`_cli/_cli_replay_detector.py:99-102`). That is Plan B's proposed `(_phase_prep, cpu), (_phase_infer, gpu), (_phase_write, cpu)` split (B:60-62, B:293), and it is already in production.
- It is also proven exact on real hardware: *"From ONE preprocessed image: live post-pipeline vs a `ReplayDetector` … objmap identical; measurements identical"* (`gpu-smoke.md` §3, bold row).

What Plan B still offers over the as-built code is narrower than its preamble says:

1. **A non-empty in-branch prefix would run once instead of twice.** It is a hypothetical for today's pipelines. The smoke run makes it concrete, though: `DenoiseBlockMatch` gives different output on different CPU models and thread counts (`gpu-smoke.md` §4). Under Plan A, such an op placed *inside* the GPU branch would run on the GPU node in Stage 2 and on a CPU node in Stage 3. The detector would see pixels that the enclosing operation never reconstructs. The docs say only that prefix ops *"must be deterministic"* (`_cli/CLAUDE.md`, *Nested detectors*), and nothing enforces it.
2. **`FilamentousFungiDetector` would become stageable.** Plan A refuses it by policy, not because the mechanism can't handle it (see F-9).
3. **The class-kind rule would be replaced by a declaration.**

Every fork in §4 should be read with that in mind.

---

## 1. Stale claims

| # | Plan B says | Code as built | Correction |
|---|---|---|---|
| S-1 | *"roughly 70% of A stands under B"* (B:6-7, B:127; README:4-5) | Per §2, only Tasks 1, 2 and 15 are fully reused. Tasks 3, 4, 6a, 9, 10 and 12 are partly reused, and Tasks 11 and 14 need rewriting. | Replace the percentage with the per-task table in §2. |
| S-2 | Carry-over table omits **Task 2** (B:110-125; README:39) | Task 2 landed as `ba0db2d9`, `tune/_search_space/_infer.py` | Add a row: *reused, independent of staging*. |
| S-3 | *"`_CHILD_CONTRACT` table **in the splitter**"* (B:52) | `_CHILD_CONTRACT` and `_child_contract` live in `_cli/_cli_validation.py:165, 201-260`. The splitter only imports them (`_cli_pipeline_split.py:26`). Plan A revision entry 3 moved them there on purpose. | *"in `_cli_validation.py`"*. |
| S-4 | *"`pipeline_requires_gpu` walks the tree … **no nesting blind spot by construction**"*, presented as a Plan B advantage (B:51; README:54-56) | As built, `find_gpu_detectors` is tree-wide (`_cli_validation.py:376`). It also walks every pipeline's `meas`/`post`/`filters`/`model` slots at any depth (`sdk_/_operation_tree.py:58-71, 87-90`; revision entry 14). Plan A has no blind spot either. | State that both approaches are tree scans. Plan B's difference is where the answer comes from (a declaration instead of `isinstance`), not coverage. |
| S-5 | *"Stage-2 branch prefix, re-executed in Stage 3 → **gone**. Each phase runs exactly once"* (B:55), and the 669 CPU-hour framing (B:352, B:368-375) | The prefix is empty for the driver (§0). The 71 s `FocusEdgePhase` is a Stage-1 op, not a branch prefix. | Label B:352's row *"hypothetical: if moved inside the GPU branch"*. Drop *"measured … 669 CPU-hours on a 33,923-image run"* as a claim about Plan A's cost today. |
| S-6 | *"`ReplayDetector` + `substitute_at_path` → **gone**"* (B:54) and *"Delete the `ReplayDetector` path"* (B:293) | `ReplayDetector` has three consumers, not one: Stage 3 (`_cli_staged_workers.py:549`), the process-mode objmap export (`_cli_staged_strategy.py:492`, via `build_replay_pipeline`), and the provenance hook protocol (`_core/_provenance.py:952-987`, whose docstring names it *"the one implementer"*). | List all three in B4's scope. The process-mode export needs a replacement, not just a deletion (see S-9). |
| S-7 | Task 6a *"reused, generalised to the `branch/` schema"* (B:115). B3 also writes *"ndarray → compressed `branch/` payload"* (B:282). | **No `branch/` schema exists.** The signal layout is `.phenotypic/progress/stage2_{raw,done}/<ds>/<slot>/<stem>.{npy,json}` (`_cli/_cli_stage2_token.py:313-323`). It is written uncompressed with `np.save` (`:362`). | Remove the `branch/` name, or define it. Where cross-round state lives is fork F-6. |
| S-8 | Task 6a *"reused"* | `detector_slot(path)` is path-based and does survive (`_cli_stage2_token.py:74-95`). **`staged_detector_slot` does not survive as written.** It derives the slot through `split_pipeline_at_gpu` → `StagePlan` (`:139-154`). Its callers are `phenotypicCLI.py:2753, 3207`, `_cli/_cli_checkpoint_handler.py:263-275` and `_cli/_cli_staged_slurm.py:592`. Five more sites call `detector_slot(plan.gpu_path)` (`_cli_staged_strategy.py:85, 450`; `_cli_staged_slurm_worker.py:183, 248, 451`; `_cli_staged_workers.py:516`). | *"Reused. `staged_detector_slot` and every `plan.gpu_path` call site are repointed from `StagePlan` to `find_gpu_detectors`."* |
| S-9 | Task 11 *"reused **unchanged**"* (B:118) | `_export_objmap_layer` works by substituting a `ReplayDetector` into `post_pipeline` (`_cli_staged_strategy.py:486-503`). Tasks 5 and 6 are both superseded under B. | *"Rewritten. Under B the export runs the phases remaining after the GPU round, then writes the layer. The semantics (§8 of the spec) and the no-store-write / no-success-sink constraints carry over."* |
| S-10 | Task 4 *"reused unchanged"* (B:114) | `apply_child` is `with pipeline_step(segment): operation.apply(...)` (`_core/_provenance.py:604-605`). The step path is read inside the `apply` wrapper (`:694-700`). A phase driver that calls a method directly never enters either. | *"The segment scheme is reused. The mechanism is reused only if phases run inside `apply()`."* This is fork F-7. (It is the same point as the earlier review's B-3, and it still holds.) |
| S-11 | Task 9 *"reused as phase-path == step-path"* (B:116) | Task 9 was **rewritten at execution time**. The invariant is now *every recorded `pipeline_step_path` resolves* (`tests/unit/cli/test_recorded_paths_resolve.py:222`), because set equality was false for `TwoKFilamentousDetector` (A:2122-2161). The only `gpu_path` check (`:283`) calls `split_pipeline_at_gpu`. | *"The resolvability test is reused as is (it does not depend on staging). The gpu-path test is rewritten against the phase chain."* |
| S-12 | Task 10 *"reused, generalised"* (B:117) | `test_staged_nested_equivalence.py` drives `split_pipeline_at_gpu`, `ReplayDetector` and `stage2_prefix`. Its tests include `:374` and `:392`, which are prefix-specific. The owner-depth test (`:590`) and the corrupted-replay control pattern (`:346`) do not depend on any of that. | *"The mutation-control and owner-depth patterns are reused. The shapes are rewritten. The prefix-specific tests are retired."* |
| S-13 | Task 12 *"reused unchanged"* (B:119) | The revision sits only in the process-only branch (`_cli/_cli_failure_tracker.py:205, 256-258`). `test_a_full_run_digest_is_UNCHANGED_by_the_bump` (`tests/unit/cli/test_work_id_semantics_revision.py:81`) pins full runs as *not* invalidated. | *"Reused for process mode. Full-run invalidation is a new decision (F-8) that this test currently forbids."* (Same as the earlier review's M-6, still open.) |
| S-14 | Task 14 *"reused"* (B:120) | The docs as built describe `ReplayDetector`, `_CHILD_CONTRACT` and the composition-primitives rule (`docs/source/contrib_guide/gpu_detectors.md:143-412`; `_cli/CLAUDE.md`; root `CLAUDE.md`). Tests pin that text: `tests/unit/test_docs_staged_cli.py:187` (quotes the nesting refusal) and `:314` (names the `_CHILD_CONTRACT` entries). | *"Rewritten, and B7 must update the two doc-pin tests."* |
| S-15 | *"the phase structure is readable from the pipeline JSON without constructing anything"* (B:45) | `pipeline_requires_gpu` constructs the pipeline (`_cli_validation.py:451`), and class resolution goes through the registry. | Delete the clause. (Earlier review m-5, still true.) |
| S-16 | The `NdArrayField` argument (B:80) is framed as a bug `ReplayDetector` *would* reintroduce | As built, the leak is prevented by the `provenance_parameters` hook (`_cli_replay_detector.py:87-89`; `_core/_provenance.py:974-978`). | Reword: Plan A avoids the leak with an explicit hook, and `PrivateAttr` would avoid it structurally. |
| S-17 | B4 step 4: *"existing staged suites pass unchanged — `test_staged_resume.py`, `test_staged_resume_equivalence.py`, `test_staged_store_stages.py`"* (B:295) | Plan A added suites that pin the mechanisms B deletes. `tests/unit/cli/`: `test_replay_detector.py`, `test_pipeline_split_nested.py`, `test_staged_stage2_prefix.py`, `test_staged_nested_equivalence.py`, `test_cli_pipeline_split.py`, `test_gpu_detection_tree_wide.py` (e.g. `:400`, `:430`, `:453`), `test_stage2_slot_keying.py`, `test_cli_gpu_refusal.py`. Also `tests/unit/detect/test_container_child_contracts.py`, `tests/integration/cli/test_process_objmap_semantics.py`, `tests/integration/gui/test_run_console_callbacks.py` and `tests/unit/test_docs_staged_cli.py`. | Add these to B4/B5 with a disposition (keep / rewrite / retire) that depends on the forks. See §5 E-9. |
| S-18 | *"Exactly one `GpuDetector` per pipeline"* is inherited (B:97), but B2's flagship test has two (B:206-213) | `find_gpu_detectors(strict=True)` refuses two detectors (`_cli_validation.py:386-398`). It is pinned by `test_gpu_detection_tree_wide.py:98`. | The test contradicts the inherited constraint. See F-5. |
| S-19 | B2 test fixtures `ImagePipeline([ContrastStretching(), FakeGpuDetector()])` (B:209-210) | Operations are keyword-only pydantic models. `ops` is the field (`_core/_pipeline_parts/_image_pipeline_core.py:202-205`), so a positional list raises. `FakeGpuDetector` also has to be registered for any `from_json` path (Plan A review finding 4; `tests/_fakes/register_fake_gpu.py`). | Mechanical, see §5 E-5. |
| S-20 | B1 step 5: *"`__init_subclass__` or a `model_validator`"* so a bad declaration fails *at class definition* (B:177) | A `model_validator` runs per instance, not per class. The existing class-definition hook is `ImageOperation.__pydantic_init_subclass__` (`abc_/_image_operation.py:396-416`), which already rewraps `apply`. | *"`__pydantic_init_subclass__`, calling `super()` first."* |
| S-21 | *"`GpuDetector` stops being special-cased"* (B:57) | The as-built engine adds two `GpuDetector`-specific obligations that Plan B does not mention: `_ensure_model_loaded` is called once per sweep before any image (`_cli_staged_strategy.py:230`; `_cli_staged_slurm_worker.py:310`), and every `_infer_batch` override must call it (AST guard `tests/unit/abc_/test_infer_batch_loads_the_model.py`). | B4 must keep both: the model loads once per GPU round, and `_phase_infer` goes through `_infer_batch`. |
| S-22 | Blocker text: *"falsifying the 'Task 4 reused unchanged' and 'Task 9 reused' carry-over claims"* (B:392-393) sits under a table that still claims them (B:114, B:116) | — | Make the table agree with the blocker text (see §2). |
| S-23 | *"Plan A has been through an independent plan review (20 findings, all applied)"* (B:9-10) | Plan A has since had a second review (`plan-a-review-2.md`), a Phase 0-2 implementation review, and revision entries 1–17 recorded during execution. It is now implemented, and the GPU smoke run passed (`gpu-smoke.md`). | Update the banner to say Plan A is implemented (PR #224, `1fcab4bf`) and to cite the smoke verdict. |

---

## 2. Carry-over table: corrected rows

| Plan A task | Plan B says | Actual status under B | Evidence |
|---|---|---|---|
| 1 shared traversal | reused | **Reused.** It now also walks pipeline slots and spells them `meas:<key>` etc. `substitute_at_path` (`:233-342`) is part of this module and becomes dead only if F-2 drops replay. | `sdk_/_operation_tree.py:58-136` |
| 2 tune consolidation | *(missing)* | **Reused**, unaffected by B | `ba0db2d9` |
| 3 tree-wide detection + refusal | reused, predicate swapped | **Partly reused.** Surviving: `find_gpu_detectors`, `refuse_cpu_only_slot` (independent of the predicate and still needed), `UnstageableGpuDetectorError`, the CLI preflight that runs before `--overwrite`/`--dry-run` (`phenotypicCLI.py:2183`), and the GUI alert (`_gui/run_console/_callbacks.py:258-266`). Replaced: `_CHILD_CONTRACT`, `_child_contract` (including the `_operate`-override refusal) and `validate_ancestor_contracts`. Their tests retire or invert, e.g. `test_gpu_detection_tree_wide.py:400, 430, 453` and all of `test_container_child_contracts.py`. | `_cli_validation.py:135-452` |
| 4 per-branch `pipeline_step` | reused unchanged | **Segment scheme reused. Mechanism reused only if phases run inside `apply()` (F-7).** | `_core/_provenance.py:537-605, 694-700` |
| 5 path-shaped `StagePlan` | superseded | **Superseded**, but it has 8 consumers to repoint (S-8) plus `test_recorded_paths_resolve.py:283` | `_cli_pipeline_split.py:29-165` |
| 6 `ReplayDetector` | superseded | **Superseded only under F-2 option (a).** Under F-2 option (b) it stays. It also has two consumers outside Stage 3 (S-6). | `_cli_replay_detector.py`; `_cli_staged_strategy.py:492` |
| 6a slot-keyed signal | reused, `branch/` schema | **Reused.** `staged_detector_slot` gets repointed (S-8), and there is no `branch/` schema (S-7). The legacy relocation (`_cli_stage2_token.py:452`) is unaffected. | `_cli_stage2_token.py:74-154, 313-368` |
| 7 Stage-2 branch prefix | superseded | **Superseded.** Empty for the driver (§0). | `_cli_staged_workers.py:146-194, 432-433` |
| 8 Stage-3 stub substitution | superseded | **Superseded under F-2 option (a) only** | `_cli_staged_workers.py:540-567` |
| 9 invariant | reused as phase-path == step-path | **Invariant changed to resolvability (reused as is). The gpu-path test is rewritten.** | S-11 |
| 10 equivalence + mutation | reused, generalised | **Patterns reused, shapes rewritten, prefix tests retired** | S-12 |
| 11 process-mode post-detector chain | reused unchanged | **Rewritten** (the semantics carry over, the mechanism does not) | S-9 |
| 12 continuation digest | reused unchanged | **Reused for process mode. Full-run invalidation is a new decision (F-8).** | S-13 |
| 13 forward `stage2_prefix` | superseded | **Superseded** (landed in `1204cb76`) | `_cli_staged_strategy.py:261`; `_cli_staged_slurm_worker.py:332` |
| 14 docs | reused | **Rewritten, including the two doc-pin tests** | S-14 |
| 15 regression | reused | **Reused** (procedure only) | — |
| *(not tasks in Plan A's list, but landed)* GUI refusal surface, CLI preflight ordering, `_infer_batch` AST guard, the GPU-smoke harness | — | **Reused.** B has to keep the refusal surfaces for whatever B still refuses (e.g. `TwoKFilamentousDetector`). | `629b6112`, `dc6eb838`, `fc05e066` |

---

## 3. Design problems against the real code, and the blockers and open questions reassessed

### 3.1 The three blockers (B:377-393)

**Blocker 1: how a parent consumes its children. Still open, and now constrained by a working alternative.**
`CompositeDetector` still runs each branch on a copy through `apply_child(..., inplace=False)` and combines the resulting objmaps (`detect/_composite_detector.py:128-172`). `CompositeEnhance` does the same with `detect_mat` (`enhance/_composite_enhance.py:152-167`). `FilamentousFungiDetector` reads `objmask` and `objmap` off its child's returned image (`detect/_filamentous_fungi_detector.py:399-407, 470`).

What has changed is the cost of the alternative. The as-built replay has been checked on real hardware (§0), the parent's combine is 4 s (B:353), and for the driver no work is duplicated. So the *"keep re-execution for the parent body"* fallback suggested in the earlier review (`plan-b-review.md` B-1, last paragraph) is not a proposal any more: it is what the code does. This is fork F-2.

One constraint the earlier review did not name is **provenance ordering**. A container's journal entry is appended *after* its children's entries, through the in-process `_operation_apply_stack` and `parent_frame.nested_records` (`_core/_provenance.py:629-640, 709-717`). A phase protocol that runs the child's write phase in one round and the parent's combine in another has to reproduce that post-order and the `sequence` numbering (`:990`) across processes. Replay gets both for free, because Stage 3 re-enters the parent's `apply`.

**Blocker 2: the device tag is a `ClassVar`, but for a container it depends on the instance. Still open, and Plan A has answered the detection half.**
As built, "does this pipeline need a GPU" is `isinstance(op, GpuDetector)` over the walked tree (`_cli_validation.py:376`), which derives the answer from the leaves. A container never declares a device. The default `FilamentousFungiDetector` gets its `inoculum_detector` from a validator (`detect/_filamentous_fungi_detector.py:230-240, 268, 309`) as an all-CPU pipeline, so a class-level GPU tag would still route every default instance to a GPU partition (`_cli/_cli_execution_strategies.py:1334-1345`). This is fork F-3. Recommendation there: keep leaf-derived detection.

**Blocker 3: driving phases bypasses `apply()`. Still open, and Plan A supplies a working pattern.**
The earlier review cited a hand-rolled provenance append in Stage 3 (`plan-b-review.md` B-3, `_cli_staged_workers.py:461-469`). **That code is gone.** As built:

- Stage 2 bypasses `apply()` and records nothing (`_cli_staged_workers.py:434-468`).
- Its duration travels in the token (`:466`).
- Stage 3 runs the write phase inside the real `apply()` chain as a leaf. The identity-delegation hooks (`_cli_replay_detector.py:78-97`) and `provenance_duration_offset` (`_core/_provenance.py:984-987`) produce one journal entry at the correct `pipeline_step_path`, in the correct order.
- Owner depth is handled by `continuing_provenance_application` (`_cli_staged_workers.py:559-567`) and pinned by `test_staged_nested_equivalence.py:590`.

That is a general answer to *"who emits the single journal entry for an operation whose phases ran in three processes"*: the last CPU phase does, inside `apply()`, and earlier phases forward their durations. Plan B should adopt it or explicitly reject it (fork F-7).

### 3.2 The four open questions (B:336-341)

1. **No spec.** Still open. The spec and README still describe B as unwritten.
2. **Return a new image or mutate in place?** Still open, and constrained. `ImageOperation.apply` handles copy vs in-place in `_apply_to_single_image` (`abc_/_image_operation.py:471-496`), and the provenance wrapper carries journal state across a returned copy (`_core/_provenance.py:684-689, 720-722`). If F-7 picks "last phase runs inside `apply()`", the question only applies to non-final phases, where "in place" is the natural answer.
3. **Conditional GPU phases.** Partly answered. The model loads only when a round has pending images (`_cli_staged_strategy.py:228-230`; `_cli_staged_slurm_worker.py:303-310`), so an empty GPU round already costs no model load. What remains, as the earlier review said, is that routing to a GPU partition is decided at submission (`_cli_execution_strategies.py:1411, 1423`).
4. **Flattening order for parallel children.** Still open, and **now in conflict with the committed spec.** Spec §13.1-13.2 (`design.md:739-781`) classifies `CompositeDetector(ops=[Sam2, Dino, …])` as **N > 1 inside one round**: *"Stage 2 becomes N sub-sweeps"*, with the store carrying state between rounds (`:793-795`). Plan B's default (*"one round per sub-pipeline"*, B:91, B:341) turns that into **R > 1**, which the spec calls the expensive axis, touching the epoch-fenced controller (`design.md:751-756, 785-791`). This is fork F-5.

### 3.3 Other design problems against the real code

- **D-1. `_phases` on `ImageOperation` only (B:134).** The earlier review's M-1 worried that the scan must see measurers because Plan A refuses GPU detectors in the `meas` slot. That is now handled independently of any predicate: `refuse_cpu_only_slot` refuses every slot placement at any depth (`_cli_validation.py:290-333, 383-384`). So scoping `_phases` to `ImageOperation` is safe **provided B keeps `refuse_cpu_only_slot`**. The other half of M-1 still holds: `ImagePipelineCore` has no `_operate` (grep: no `def _operate` under `_core/`), so the inherited default would fail B1's *"every phase method exists"* guard for pipelines. They must be handled as `"sequence"` containers by the engine, as `_child_contract` does today (`_cli_validation.py:228-229`). This is fork F-11.
- **D-2. The GPU result would have two persistence paths.** B4 step 2 carries the inference result in `_phase_state` (B:293). The as-built engine already persists exactly that array as the slot-keyed Stage-2 raw file, with a token, atomic writes, epoch fencing, legacy relocation and a replayability predicate used by six call sites (`_cli/CLAUDE.md`, *Stage-2 signals*). A second path for the same bytes would duplicate all of that. This is fork F-4.
- **D-3. Stage tags are fixed at three in more places than the earlier review listed.** Its M-2 named `_cli/_stages.py:6-15` and `_cli/_cli_staged_resume.py:35-40`. The per-image record also carries `STAGE_STAGE1..3` (`sdk_/_image_record.py:52-54`), written by `_cli/_cli_image_record.py:184, 231`. Anything with **R > 1** needs a schema change and a migration in all three. Any design that keeps R = 1 (F-5 option b) needs none.
- **D-4. "Stage 2 never writes the store" (inherited, B:97) conflicts with B3's persistence (B:282) and with blocker 1's handoff.** The constraint is real (`_cli_staged_workers.py:446-450`). The spec's R > 1 model relies on *CPU* rounds re-promoting the store (`design.md:793-795`), which is allowed. Branch copies are not the image, so they have no home in the store. This is fork F-6.
- **D-5. `FilamentousFungiDetector` is refused by policy, not by mechanism.** `_child_contract`'s docstring says FFD *"reads as `"parallel"`"* and is refused only because that property is incidental (`_cli_validation.py:204-213`). The replay mechanism would stage it correctly. B6 is presented as *"the thing Plan A cannot do"* (B:317). Plan A could do it with one table entry and a behavioural probe. This is fork F-9.
- **D-6. B6's rationale is still inconsistent** (earlier review B-2, unchanged by Plan A). In the nested case, the inoculum detection *is* the GPU work (`_filamentous_fungi_detector.py:394-403`).
- **D-7. The M-5 contradiction still holds.** `test_unset_is_distinguishable_from_none` (B:256-263) cannot pass against `PrivateAttr(default=None)` (B:74-75). B3's dict equality on ndarrays raises (m-1). Both are carried into §5.

---

## 4. Design forks requiring a human decision

These are listed, not resolved. Each changes behaviour or the public surface.

### F-1. Is Plan B still worth doing, and for what?

- **(a) Proceed with the full phase protocol.** Gains: in-branch prefixes run once, FFD is admitted, and the class-kind rule goes away. Cost: blockers 1–3 plus D-3, for a change whose main measured payoff (669 CPU-hours) does not apply to the driver pipeline.
- **(b) Shelve Plan B and make targeted Plan A extensions:** admit FFD (F-9), and add a guard for non-empty prefixes (refuse them, or warn when the prefix contains an op known to be non-deterministic across CPU models, such as `DenoiseBlockMatch`). Cheap, keeps the proven replay path, and leaves the double execution in place for non-empty prefixes.
- **(c) Narrow Plan B to "phase-split leaves only".** `GpuDetector` declares its three phases, and containers stay on replay. That is mostly a renaming of what is built (§0).

**Recommendation: (b)**, unless someone has a real pipeline with an expensive or non-deterministic op *inside* the GPU branch. If so, choose (a) with F-2 option (b), which keeps replay for the parent.

### F-2. How a parent consumes its children (blocker 1)

- **(a) A two-kind protocol:** a CHILDREN sentinel in `_phases`, a `_combine(image, children)` phase signature, and a defined handoff type, arity and storage. It removes all re-execution, needs the most new machinery, and requires provenance post-order to be reconstructed across processes (§3.1).
- **(b) Keep container replay.** Children are phase-split, and the parent's `_operate` re-runs in the final CPU round with stubs at the child write phases. This is proven exact (`gpu-smoke.md` §3), and the combine is cheap (4 s). It gives up "each phase exactly once" for parent bodies only.

**Recommendation: (b).**

### F-3. Where the device comes from (blocker 2)

- **(a) Containers never declare a device.** GPU-ness comes from leaves, as `find_gpu_detectors` does today.
- **(b) An instance-level `phases_of(op)` that may consult fields.**
- **(c) A class-level `ClassVar`, as written.** This mis-routes default FFD instances.

**Recommendation: (a).**

### F-4. How a `GpuDetector`'s result crosses the GPU→CPU boundary (D-2)

- **(a) Keep the slot-keyed Stage-2 raw file and token as the ferry.** `GpuDetector` needs no `_phase_state`.
- **(b) A generic `_phase_state` ferry.** This means one persistence path for all phased ops, and a migration of the Stage-2 signal, which spec §4.4 exists to avoid.

**Recommendation: (a).**

### F-5. Parallel GPU siblings: N sub-sweeps or R rounds (open question 4; S-18)

- **(a) One round per sub-pipeline**, as in B2's test: R = 2 for two siblings, `2R+1` job groups, controller changes, and the D-3 schema migration.
- **(b) The spec's §13.2 model:** one Stage 2 with N sub-sweeps, R = 1, and the same stage schema.
- **(c) Keep refusing N > 1** (today's `strict=True`) and drop B2's test until N > 1 is in scope.

**Recommendation: (c) now, (b) when N > 1 is needed.** Option (a) contradicts the committed spec and should be adopted only by amending §13.

### F-6. Where cross-round intermediates live (D-4, S-7)

- **(a) CPU rounds re-promote the store, GPU rounds write only `.phenotypic/progress/` signals** (spec §13.3). No new artifact type.
- **(b) A new per-branch artifact under `.phenotypic/progress/`**, with compression, a size cap, and cleanup and fencing rules. Only needed under F-2 (a).
- **(c) Image layers in the store.** This conflicts with "Stage 2 never writes the store" whenever the producer is a GPU round.

**Recommendation: (a).** The ferried-payload cap (B:102) matters only under (b).

### F-7. Provenance for a phased operation (blocker 3)

- **(a) Earlier phases record nothing and forward their durations. The final CPU phase runs inside `apply()`** with the `provenance_*` hooks (the as-built pattern).
- **(b) The engine writes the records itself** with `append_operation_provenance` and an explicit step path. Post-order and `sequence` have to be rebuilt by hand, and Task 4's mechanism is bypassed.

**Recommendation: (a).**

### F-8. Continuation across the Plan B upgrade (S-13)

- **(a) Add a staged-full-run revision to the work-id digest.** This invalidates in-flight staged full runs and inverts `test_a_full_run_digest_is_UNCHANGED_by_the_bump`.
- **(b) No invalidation**, because B keeps the on-disk signal and stage schema unchanged. That is only true under F-4 (a) plus F-5 (b) or (c).
- **(c) Migrate** old signals in place.

**Recommendation:** decide after F-4 and F-5. Under the recommended options, (b) is honest and nothing needs invalidating. Otherwise choose (a).

### F-9. Admitting `FilamentousFungiDetector`

- **(a) B6's phase split.**
- **(b) Add `FilamentousFungiDetector: "parallel"` to `_CHILD_CONTRACT`**, with a behavioural probe in `test_container_child_contracts.py`, and invert `test_gpu_detection_tree_wide.py:430`. This drops the "composition primitives only" policy (spec §4.3) for this one class.
- **(c) Keep refusing it.**

**Recommendation:** (b) if FFD staging is actually wanted. It reuses the proven path, and the policy's worry, that FFD's `_operate` might change what it hands its child, is covered by the probe test.

### F-10. Per-slot child order (the reason for the TwoK refusal)

- **(a) A class-wide `_child_order`** (B as written). TwoK is refused wholesale.
- **(b) Per-field order.** TwoK's `center_detector` receives the container's own input (`detect/_two_k_filamentous_detector.py:160-162`) and would be admissible, while `background_subtractor` receives a sibling's output (`:165-170`) and would stay refused.

**Recommendation: (a)** unless a TwoK GPU use case exists. This fork affects public surface only if B proceeds.

### F-11. Where `_phases` is declared (D-1)

- **(a) On `ImageOperation` only.** Pipelines are sequence containers by rule, and slots stay refused by `refuse_cpu_only_slot`.
- **(b) On `BaseOperation`**, with a different default per hierarchy (earlier review M-1).

**Recommendation: (a).** Now safe, because the slot refusal is independent.

---

## 5. Mechanical edits (no decision needed)

- **E-1 (banner, B:3-12).** Say Plan A is implemented (PR #224, `1fcab4bf`), reviewed twice, and GPU-smoke verified (`gpu-smoke.md`). Replace *"roughly 70% of A stands"* with a pointer to the corrected carry-over table.
- **E-2 (B:49-55, the replacement table).** Row 1: drop *"no nesting blind spot by construction"*, or add *"(Plan A as built is also tree-wide, slots included)"*. Row 2: *"`_CHILD_CONTRACT` table in `_cli_validation.py`"*. Row 4: add *"(also consumed by process-mode export and the provenance hook protocol)"*. Row 5: add *"(empty for the driver pipeline; see §0 of the staleness review)"*.
- **E-3 (B:45).** Delete *"so the phase structure is readable from the pipeline JSON without constructing anything"*.
- **E-4 (B:80).** Reword the `NdArrayField` sentence to say Plan A prevents the leak with `provenance_parameters` (`_cli_replay_detector.py:87-89`).
- **E-5 (B2 tests, B:197-225).** Use `ImagePipeline(ops=[...])` everywhere. Import `FakeGpuDetector` from `tests._fakes.fake_gpu_detector` and use the registration pattern from `tests/unit/cli/test_gpu_detection_tree_wide.py`. Mark `test_parallel_children_get_one_round_each` as blocked on F-5.
- **E-6 (B1 step 5, B:177).** *"Wire validation into `__pydantic_init_subclass__`, calling `super().__pydantic_init_subclass__(**kwargs)` first (see `abc_/_image_operation.py:396-416`). A `model_validator` runs per instance and cannot fail at class definition."* Also open `validate_phase_declaration` with an `isinstance(cls._phases, tuple)` check (earlier review m-9).
- **E-7 (B3, B:244-280).** Compare payloads by key set plus `np.array_equal` per value, never with `==` or `!=` (m-1). Replace *"compressed `branch/` payload"* with *"payload at the location decided by F-6"*.
- **E-8 (B6, B:320).** Declare the state as `PrivateAttr()` **with no default**, because that is the only declaration under which B3's unset-vs-None requirement holds (M-5). Name the read path: `op.__pydantic_private__`. Name the write path: `setattr`, never `object.__setattr__` (M-8).
- **E-9 (B4 file list, B:289-290).** Add: `_cli/_cli_replay_detector.py`, `_cli/_cli_pipeline_split.py`, `_cli/_cli_validation.py`, `_cli/_cli_stage2_token.py` (`staged_detector_slot`), `phenotypicCLI.py:2753, 3207`, `_cli/_cli_checkpoint_handler.py:263-275`, `_cli/_cli_staged_slurm.py:592`, `_cli/_cli_staged_controller.py`, `_cli/_cli_staged_resume.py`, `_cli/_stages.py`, `sdk_/_image_record.py:52-54`, `_cli/_cli_image_record.py`, `_core/_provenance.py:943-987` (hook docstring), `_gui/run_console/_callbacks.py` (refusal text).
- **E-10 (B4 step 4, B:295).** Replace *"pass unchanged"* with the list in S-17 and a disposition column (keep / rewrite / retire), to be filled in once F-2 through F-5 are decided. Unconditionally keep the tests that don't depend on the replay mechanism: `test_recorded_paths_resolve.py:222, 260`, the owner-depth test `test_staged_nested_equivalence.py:590`, `test_stage2_slot_keying.py`, `test_cli_gpu_refusal.py`, and `test_infer_batch_loads_the_model.py`.
- **E-11 (B4 step 2, B:293).** Add: *"Load the model once per GPU round, only when the round has pending images (`_cli_staged_strategy.py:228-230`); `_phase_infer` calls `_infer_batch`, never `_infer_one` (AST guard `tests/unit/abc_/test_infer_batch_loads_the_model.py`)."*
- **E-12 (carry-over table, B:110-127, and README:37-44).** Replace both with the table in §2, including the missing Task 2 row.
- **E-13 (B7, B:330-332).** Add `tests/unit/test_docs_staged_cli.py:187, 314` (the doc-pin tests for the refusal text and the `_CHILD_CONTRACT` entries). Add `docs/source/contrib_guide/gpu_detectors.md` §§ *"Nesting: the container contract"* through *"What Stage 3 actually runs: `ReplayDetector`"* (`:143-412`) as the sections to rewrite.
- **E-14 (measurements, B:352 and B:368-375).** Label the `FocusEdgePhase` row *"hypothetical in-branch placement; in `F1gfd5` it is a Stage-1 op and runs once under Plan A"*. Add the `gpu-smoke.md` §4 finding (BM3D output depends on the CPU) as the concrete reason in-branch prefix determinism matters.
- **E-15 (B:127, B:317).** Delete *"the thing Plan A cannot do"*, or qualify it as *"cannot do under its current policy"* (D-5).
- **E-16 (B:97, constraints).** Add the constraints Plan A added since B was written: slot segments are spelled `meas:<key>`/`post:<key>`/`filters:<key>`/`model:<ClassName>`; every slot placement is refused at any depth; the refusal fires before `--overwrite` and `--dry-run` and is shown in the GUI alert.

---

## 6. Verdict

**Blocked on the forks in §4. It is not executable as is, and the mechanical edits alone do not make it executable.**

The mechanical edits (§5) fix the stale citations and the broken test fixtures, but the three blockers remain design questions, and Plan A's implementation has changed what they cost. F-1 comes first. The case for Plan B rested on removing a double execution that the driver pipeline doesn't have, and on admitting a detector that the as-built mechanism could admit with a table entry. If the answer to F-1 is still "proceed", then F-2, F-3, F-5 and F-7 need decisions before a spec can be written, and the recommended options (replay for parents, leaf-derived device, R = 1, provenance recorded in the last phase) shrink Plan B to something close to option (c) of F-1.

---

## Separately reported: risks in the as-built code (not Plan B defects)

- **R-1. Transient storage at the dataset-wide barrier.** The Stage-2 raw file is written uncompressed (`_cli/_cli_stage2_token.py:362`, `np.save`). At 3140×5094 uint16 that is about 32 MB per image. The staged controller runs Stage 3 only after the Stage-2 round, so on a 33,923-image run about 1.0 TiB of `.phenotypic/progress/stage2_raw/` is live on `/bigdata` at once. Plan B's measurement (B:363-367) applies to Plan A as built. `np.savez_compressed` measured 133× smaller on a synthetic case, likely 10–30× on real SAM2 output. I haven't checked the actual free quota on this filesystem.
- **R-2. No enforcement of prefix determinism.** Under Plan A, any non-empty `stage2_prefix` runs twice, on different nodes. `gpu-smoke.md` §4 shows that `DenoiseBlockMatch` depends on the CPU model and thread count, so such an op inside a GPU branch would silently give the detector input that Stage 3 never reconstructs. Only the documentation says prefix ops must be deterministic. No shipped pipeline is affected today, because the driver's prefix is empty.

No new bugs were found in the code paths reviewed.
