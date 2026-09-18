# Plan A — second review (post-edit deltas + internal consistency)

**Subject:** `docs/superpowers/plans/2026-09-15-nested-gpu-staging/plan-a-composite-path.md` (2425 lines)
**Spec:** `docs/superpowers/specs/2026-09-15-nested-gpu-staging/design.md`
**Prior review (applied):** `docs/superpowers/reports/2026-09-15-nested-gpu-staging/plan-review.md`
**Worktree:** `/bigdata/exfab/anguy344/PhenoTypic/.claude/worktrees/nested-gpu-staging`
**Reviewed:** 2026-09-15. Analysis only — no plan, spec or source file was edited.

## Verdict

**NEEDS REVISION.** The architecture is sound and the prior review's blockers all
landed. But the narrowing to "only composition primitives" — the largest change
since that review — **is not wired into any code path**: `_child_contract` is
defined and never called, so every refusal test in Task 5 steps 3b/3c fails and a
`GpuDetector` inside `FilamentousFungiDetector` or `TwoKFilamentousDetector`
silently *succeeds* instead of being refused. Separately, Task 6a (also written
after the review) understates its blast radius by three source modules and ~10
test files, specifies a relocation guard (`len(plan.slots) == 1`) against an
attribute `StagePlan` does not have, and relocates only half the two-part signal.
Tasks 7, 8 and 11 still call the un-slotted path helpers that 6a removes. Two
test message-match strings cannot match the messages the plan specifies.

None of this is architectural. It is all fixable inside the existing task
structure, but it must be fixed before dispatch: at least four of the plan's own
tests cannot pass as written, and the headline refusal does not exist.

---

# BLOCKER

## B1. `_child_contract` is never called — the "composition primitives only" rule does not exist

**What is wrong.** `_branch_prefix` (plan `:1142-1167`) is the only consumer the
plan gives `_child_contract` (`:1116-1140`), and it does not call it. Its body
tests `isinstance(container, ImagePipeline)` and `continue`s past everything else:

```
:1156        if not isinstance(container, ImagePipeline) or container is pipeline:
:1157            continue
...
:1163    if isinstance(parent, ImagePipeline) and parent is not pipeline:
```

Its own docstring (`:1145-1149`) says the opposite — *"Walks the ancestor chain,
asking `_child_contract` what each container hands its children… A container with
no entry raises — see `_child_contract`."* `_CHILD_CONTRACT` and `_child_contract`
are therefore dead code, and `grep` over the plan and spec confirms no other
call site.

**Consequence.** Trace `ImagePipeline(ops={"TwoK": TwoKFilamentousDetector(branch_base=FakeGpuDetector())})`:

- `gpu_path == ("TwoK", "branch_base")`, `len(path) == 2`.
- Loop `for depth in range(1, 2)`: `depth == 1` → `container is pipeline` → `continue`.
- Final block: `parent = get_at_path(pipeline, ("TwoK",))` = the `TwoKFilamentousDetector`,
  which is not an `ImagePipeline` → skipped.
- `_branch_prefix` returns `[]`. **`split_pipeline_at_gpu` returns a plan.**

So the split succeeds, cuts at `"TwoK"`, and Stage 3 substitutes the stub into
`branch_base`. For `TwoKFilamentousDetector` that is wrong on the merits —
spec §4.3 documents that `branch_base` mutates `enhanced` in place
(`_two_k_filamentous_detector.py:164`) while `background_subtractor` gets a
derived `enhanced.copy()` (`:154`), so Stage 2 would infer on the wrong array with
no error anywhere. Same for `FilamentousFungiDetector.inoculum_detector`.

These fail (all in Task 5 steps 3b/3c):
`test_a_domain_detector_is_refused_even_though_it_would_classify`,
`test_a_gpu_detector_inside_a_domain_detector_is_refused`,
`test_the_refusal_names_the_supported_containers`.

**Fix direction.** This is not a one-line call insertion: both guards that would
be the natural call sites (`container is pipeline`, `isinstance(parent, ImagePipeline)`)
skip exactly the nodes that must be interrogated. Restructure `_branch_prefix`
(or a sibling `_validate_ancestors`) to walk *every* ancestor node from
`gpu_path[0]`'s container down to the GPU op's parent, call `_child_contract` on
each (root pipeline included; it returns `"sequence"` and contributes nothing
because Stage 1 already ran it), and dispatch on the returned string rather than
on `isinstance`. Keep the existing root guard — it is correct and is the finding
the previous review overturned (see V4 below).

Consider also raising the refusal from `find_gpu_detectors` rather than from the
prefix builder: the refusal is a property of *placement*, not of prefix
computation, and `pipeline_requires_gpu` is the production entry point that the
prior review's Blocker 5 established must carry placement refusals.

## B2. Two `pytest.raises(..., match=...)` strings cannot match the message the plan specifies

**What is wrong.** Task 3's implementation (plan `:668-671`) raises:

```
"staged execution currently supports one GpuDetector per pipeline; "
f"found {len(hits)} at: {paths}"
```

The substring "more than one" does not occur in it. Two tests assert it does:

- Plan `:574` — the plan's own new test
  `test_two_gpu_detectors_anywhere_are_refused`, `match="more than one"`. **Fails.**
- `tests/unit/cli/test_cli_pipeline_split.py:33` —
  `pytest.raises(ValueError, match="more than one GpuDetector")`. **Fails**
  (`UnstageableGpuDetectorError` is a `ValueError`, so the type matches; the
  message does not).

Task 5 step 5 flags the second one only as *"a message assertion may not [pass]"*.
It definitely will not, and an executor reading a hedge will not act on it.

**Fix direction.** Either put "more than one GpuDetector" back into the message —
it is the wording the existing suite already pins, and the path list can follow it
— or state both test edits explicitly as required, with the new expected string.

---

# MAJOR

## M3. Task 6a's caller inventory misses three source modules and ~10 test files

Task 6a step 4 lists the callers to update as `_cli_staged_workers.py`,
`_cli_staged_strategy.py`, `_cli_staged_slurm_worker.py`. A tree-wide grep for
the eight path helpers and `DIR_STAGE2_DONE` turns up three more source modules:

| Module | Sites | Why it matters |
|---|---|---|
| `_cli_staged_resume.py` | `:256, :309, :420, :423, :478, :479` | `classify_staged_image` and `clear_downstream_artifacts_for_stage1` — the resume classifier. Its signature is `(image, dataset, output_dir, input_root, process_only_layer, markers_required, expected_work_id)`; it has **no plan and no slot**. Threading a required keyword-only `slot` means changing `build_staged_resume_plan` and every caller above it. |
| `_cli_staged_controller.py:84` | `stage2_result_replayable(output_dir, entry.dataset, entry.stem)` | the recovery controller's already-done skip; also has no plan in scope. |
| `_cli_migrate_state.py:184` | builds the token path **by hand** from `progress_dir / DIR_STAGE2_DONE / dataset / f"{stem}.json"` | with a slot level inserted, `_stage2_entry` silently returns `None` for every modern token, so `--mode migrate` stops recording interrupted Stage-2 state. The function's own docstring calls that a real interrupted state worth recording. |

Test files touching these helpers positionally (all break under
"keyword-only and required"): `tests/unit/cli/conftest.py:287`,
`tests/integration/cli/conftest.py:209,210,235`,
`test_staged_controller.py:455,847,851,869,876,877`,
`test_provenance_fencing.py:102,103`,
`test_lifecycle_publication_races.py:453,471,493,496,521,538,560,563`,
`test_staged_store_stages.py:71,76,77,252,262,278,284,352,360`,
`test_staged_gpu_local.py:120,121,128,129,841,1105,1144,1220,1249,1348,1349,1373,1374,1404,1405`,
`test_staged_resume.py` (~14 sites), `test_staged_resume_equivalence.py:1443`,
`test_schema_gate.py:623`, `test_migrate_state.py:110`.

Task 6a step 6 runs only four of those files.

**Fix direction.** Either (a) expand Task 6a's inventory and verification set to
the full list, and say explicitly how `slot` reaches `classify_staged_image` and
`_partition_entries` (both currently receive no plan); or (b) reconsider "required
keyword-only". A module-level default derived from the run's single slot is the
structural hole 6a exists to close, but a *plan-derived* slot passed down one
level from the strategies, with the resume/controller layer taking it as an
ordinary parameter, closes it just as well at a fraction of the churn.

Note the scale claim in Task 6a's rationale — *"At `N == 1` this is one extra
directory level and no behavioural difference"* — is true of the on-disk layout
and not of the code change.

## M4. `relocate_legacy_stage2_signal`'s guard references an attribute `StagePlan` does not have, and only half the signal is relocated

Two separate defects in the same helper.

**(a) The guard is unimplementable as written.** Plan `:1679`: *"Call it from the
resume path **only when `len(plan.slots) == 1`**"*. Task 5's `StagePlan` has
`pre_pipeline`, `gpu_path`, `gpu_detector`, `stage2_prefix`, `post_pipeline` — no
`slots`. Spec §13 (`:771`) lists *"`StagePlan` carries a **list** of slots rather
than one"* as part of the **future** `N > 1` work. Under Plan A the plan has
exactly one slot by construction, so the condition is vacuously true and the guard
protects nothing. It is also not stated *where* on the resume path the call goes —
`classify_staged_image` is a pure classifier with no slot (see M3).

**(b) Only the raw is relocated.** This module's vocabulary defines the Stage-2
*signal* as two files (`_cli_stage2_token.py:1-10`: the retained raw plus the
consumable token), and `stage2_result_replayable` (`:186-210`) requires **both**.
The helper is named `relocate_legacy_stage2_signal`, but its docstring says
"move a pre-slot-keying signal into its slot directory" (singular file) and both
tests assert only `stage2_raw_path(...)`. If only the raw moves, the slot-keyed
`stage2_token_exists` stays `False`, `stage2_result_replayable` stays `False`,
Stage 2 recomputes anyway — the exact GPU cost the task exists to avoid — and the
tree is now half-migrated.

**(c) The ordering invariant is not stated.** The module's rule is raw-before-token
on write (`_cli_stage2_token.py:172-174`) and token-before-raw on delete
(`_cli_staged_workers.py:559-572`, `_cli_staged_resume.py:418-424`,
`_cli_staged_strategy.py:483-487`). A relocation must move the **raw first, then
the token**, so an interrupted relocation never leaves a slot-keyed token with no
slot-keyed raw. (Both orders happen to recover, because `stage2_result_replayable`
checks both halves — but say so rather than rely on it.)

**Fix direction.** Relocate both files, raw first; assert both in both tests;
replace `len(plan.slots) == 1` with something expressible today that survives
N > 1 (e.g. the caller passes the single slot explicitly and the helper refuses
to run when the caller declares more than one). Name the actual call site.

## M5. Tasks 7, 8 and 11 still call the un-slotted path helpers that Task 6a deletes

Task 6a sits at position 6a in the dependency order (`1 → {2,3,5} → 6 → 6a → {7,8} → 9 …`),
so every later snippet must pass `slot=`. Three do not:

- Plan `:1735` (Task 7 test) — `load_stage2_raw(staged_store_fixture.output_dir, "ds", "img")`.
  The same test's `stage2_detect_core(...)` call (`:1723-1730`) also passes no slot.
- Plan `:1828-1829` (Task 8) — `load_stage2_raw(output_dir, dataset_name, image_stem)`
  and `read_stage2_token(output_dir, dataset_name, image_stem)`.
- Plan `:2069` (Task 11) — `load_stage2_raw(output_dir, ds.name, source_image_stem(img))`.

Under 6a's "keyword-only and required" these are `TypeError` at call time, which
is at least loud. But it means three tasks' reference snippets are wrong, and
Task 8 is the one whose snippet the executor is told to paste.

**Fix direction.** Sweep every post-6a snippet for `slot=`, and say once, in 6a,
where the slot value comes from in each stage (Stage 2 and Stage 3 both have
`plan` in scope; the process-mode export does too, via `_export_objmap_layer`'s
`plan` parameter at `_cli_staged_strategy.py:398`).

## M6. Task 13's SLURM snippet drops `active_check` and `commit_guard`

Plan step 1 gives, for `_cli_staged_slurm_worker.py:310`:

```python
stage2_detect_core(
    plan.gpu_detector, output_dir, ds_name, stem, cfg.image_type,
    stage2_prefix=plan.stage2_prefix,
)
```

The real call there is:

```python
stage2_detect_core(
    plan.gpu_detector, output_dir, item.dataset, item.stem, image_type,
    active_check=check, commit_guard=commit_guard,
)
```

The snippet uses the *local* strategy's variable names (`ds_name`/`stem`/`cfg.image_type`
instead of `item.dataset`/`item.stem`/`image_type`) and silently drops both
keywords. `active_check` is the SLURM epoch fence and `commit_guard` gates durable
writes; pasting this snippet disables both on the GPU stage, and nothing fails to
say so.

Otherwise **Task 13's re-aim is correct and verified**: `gpu_key` appears nowhere
outside `_cli_pipeline_split.py`, `:182/:296/:448` really are plain
`split_pipeline_at_gpu(ImagePipeline.from_json(pipeline_path))` calls, and
`_cli_staged_strategy.py:246` / `_cli_staged_slurm_worker.py:310` really are the
only two `stage2_detect_core` call sites.

## M7. The plot guard's stated reduction is wrong for a top-level detector

Task 5's note says *"the ancestor key now lives in `post_ops`, so the old
`ref.key == gpu_key or ref.key in pre_ops` check reduces to `ref.key in pre_ops`."*

That holds when `gpu_path[0]` is a *container*. When the detector is itself
top-level, `gpu_path[0]` **is the detector**, and the dropped disjunct was the
guard that refused a plot bound to the detector — deliberately, because Stage 3
never runs the real detector. Under the new code such a binding passes the split,
and then Task 8 calls `substitute_at_path(post_pipeline, ("Sam2",), stub)`. The
rebuilt pipeline re-runs `_resolve_plot_bindings`
(`_image_pipeline_core.py:328-344`) against a registry in which the `"Sam2"` slot
now holds a `ReplayDetector`, which is not plot-capable. Best case that raises at
Stage 3; worst case `PlotCoordinator(plan.post_pipeline, …)` (untouched by Task 8)
emits against the *original* detector object while the image came from the
substituted copy.

Narrow in practice (it needs a `GpuDetector` that also subclasses `PlotImage`),
but the plan asserts the reduction is sound and pins it with a test
(`test_a_plot_referencing_the_ancestor_is_now_allowed`) that only exercises the
nested case.

**Fix direction.** Keep a guard for `gpu_path` of length 1 — i.e. refuse
`ref.key == gpu_path[0] and len(gpu_path) == 1` — and add a test for it, or state
explicitly that plot bindings on the substituted node are unsupported and refuse
them by path rather than by key.

## M8. Task 11 drops the `PerImageScientificError` wrapper

Current code (`_cli_staged_strategy.py:464-471`) wraps the merge:

```python
try:
    plan.gpu_detector._write_object_output(image, raw)
except MemoryError:
    raise
except Exception as exc:
    raise PerImageScientificError(STAGE_MEASURE, exc) from exc
```

Task 11's replacement snippet (plan `:2060-2085`) has no wrapper at all. The
replacement does strictly *more* work than the call it replaces (the whole
post-detector op chain, not one array write), so it is strictly more likely to
raise — and an unwrapped exception changes how `_record_local_terminal_failure`
classifies the image.

**Fix direction.** Keep the `MemoryError` re-raise and the
`PerImageScientificError(STAGE_MEASURE, exc)` wrapper around `residual.apply(...)`.

## M9. Spec §4.3 mandates an enumerating coverage gate; Task 5 replaces it with a closed-set assertion, and the plan contains both

Three-way inconsistency introduced by the narrowing:

- **Spec** `design.md:203-207`: *"A guard test enumerates every `OperationField`-bearing
  class (7 today) and requires each to be in the table **or on an explicit
  unsupported list with a reason**."*
- **Plan `_CHILD_CONTRACT` comment** `:1107-1108`: *"coverage of this table is
  enforced by a guard test over every `OperationField`-bearing class."*
- **Plan Task 5 step 3b** `:1252-1259`: the opposite —
  *"The table is closed by RULE, not by survey… this asserts the list itself
  rather than enumerating the tree"* — `assert set(_CHILD_CONTRACT) == {CompositeDetector, CompositeEnhance}`.

Downstream, Task 14 step 2 (`:2337`) requires the contributor guide's class list
to *"match `_CHILD_CONTRACT` / `_UNSUPPORTED_CONTAINERS` exactly"*.
`_UNSUPPORTED_CONTAINERS` is the spec's "explicit unsupported list", which the
plan deleted; it exists nowhere in the plan or the tree.

**Fix direction.** Pick one. The closed-set assertion is the better design and
follows from the narrowing — but then the spec §4.3 paragraph, the
`_CHILD_CONTRACT` comment, and the Task 14 reference all need updating. (Spec
edits are the lead's call.)

## M10. Task 10's equivalence gate does not run the production entry points

Task 10 is *"the gate for the whole change"*, but step 1 says to drive it *"through
the real `split_pipeline_at_gpu` + `ReplayDetector` + `substitute_at_path`"* —
i.e. to reimplement the stage sequence in the test, exactly as the spike does
(`spike_controls.py:staged_run`). It does not call `stage2_detect_core` or
`stage3_merge_measure_core`. A defect in Task 7's provenance-detached probe, in
Task 8's substitution wiring, in the token/duration plumbing, or in Task 13's
forwarding would not be caught by the gate. Only step 4's owner-depth test
mentions `stage3_merge_measure_core`, and only for one shape.

The shape names also drift: the spike defines `shape_a/b/c`
(`spike_nested_gpu.py:171,175,184`); Task 10 step 2 refers to `shape_leaf`.

**Fix direction.** Keep the port as the *algorithmic* equivalence check, and add
at least one shape driven end-to-end through `stage2_detect_core` +
`stage3_merge_measure_core` against a real staged store — the fixtures in
`tests/unit/cli/conftest.py` (`write_stage2_raw` at `:287`) and
`tests/integration/cli/conftest.py:200-240` already build one.

---

# MINOR

## m11. Task 4's file table misdescribes `FilamentousFungiDetector`

Step 5's table (plan `:958-962`) lists `_filamentous_fungi_detector.py` as
*"list of `ops`"* with segment `f"ops[{i}]"`. That class has no `ops` list; its
operation-valued field is `inoculum_detector: Union[OperationField, None]`
(`:276`), applied at `:395/:398`. The correct segment is `"inoculum_detector"`.

The same table covers only `branch_base` for `_two_k_filamentous_detector.py`,
leaving `center_detector` (`:149,151`) and `background_subtractor` (`:154`)
un-descended — a partial descent whose rationale is not recorded.

Also: `apply_child` injects `reset=False` for an `ImagePipeline` child, while the
current `:164` call passes only `inplace=True`. `branch_base` *defaults* to an
`ImagePipeline` (`:103`), so this is a live behaviour change on the default path;
`ImagePipeline.apply`'s `reset` default is `None`, not `False`
(`_image_pipeline_core.py:947-949`).

## m12. Task 5 step 3a asks for a probe test for a class the plan refuses

Plan `:1246`: *"Write the equivalent for `CompositeEnhance` … and
`FilamentousFungiDetector`."* `FilamentousFungiDetector` is not in
`_CHILD_CONTRACT` and is explicitly refused by Task 5 step 3b/3c. Leftover from
the pre-narrowing draft.

## m13. Scope preamble uses a vocabulary word that does not exist

Plan `:22`: *"`FilamentousFungiDetector` would classify as `"parallel"` today"*.
The contract vocabulary is `"same"` / `"sequence"` (spec §4.3, plan `:1110-1140`).
`"parallel"` appears nowhere else.

## m14. Self-Review's spec-coverage map predates Task 6a

Plan `:2417` maps *"1→T3, 2/2a→T1/T2, 3→T5, 4→T6, 5→T7, 6→T8, 7→T11, 7a→T12, …"*
and omits spec inventory rows **5a** and **5b** (`design.md:619-620` — slot keying
and the legacy relocation), which are Task 6a. The claim "every row maps to a
task" is therefore false as written, even though the coverage is real.

## m15. `_with_plot_on` is not implementable the way the plan points

Task 5's two plot tests call an undefined `_with_plot_on(pipe, key)` and say the
construction *"follows the existing pattern in `tests/unit/cli/` for plot
bindings"*. That pattern (`test_cli_pipeline_split.py:53-54`) is
`class _PreGpuPlot(BlurGauss, PlotImage)` — the plot capability is a **mixin on
the op class**. A plot cannot be retrofitted onto an existing plain
`CompositeDetector` instance; the test needs a `CompositeDetector`+`PlotImage`
subclass. Worth spelling out, since `normalize_plot_bindings` raises on a
non-plot-capable entry.

## m16. Task 10 step 4's `_application_owner_depth.set(0)` adds no coverage

`_application_owner_depth` is a `ContextVar` with `default=0`
(`_provenance.py:80-81`), so setting it to 0 in a test that is not already inside
a `provenance_application` is a no-op. The rationale quoted from
`measure/CLAUDE.md:59-66` comes from the *measurement* case, where the enclosing
`with provenance_application(image, kind="programmatic"):` is what raises the
depth — and the plan's snippet omits that wrapper.

The test is still worth having: the depth-0 exposure in Stage 3 comes from the
store's trailing `"staged"` application, not from the context var. But the stated
mechanism is wrong and an executor will believe they have added a guard they have
not.

## m17. `substitute_at_path`'s "EVERY slot" comment is not accurate

The rebuild (plan `:340-357`) carries `ops/meas/post/filters/model/qc/plots/
nrows/ncols/name/_provenance_pipeline` but drops `benchmark`, `verbose`, `reset`
and `desc_value` (`_image_pipeline_core.py:190-221`). Harmless for a throwaway
Stage-3 pipeline; the comment claiming completeness is the thing that will mislead
the next reader.

## m18. `ImagePipeline` vs `ImagePipelineCore` in the traversal

`iter_child_operations` / `_child` / `substitute_at_path` key on
`phenotypic._core._image_pipeline.ImagePipeline`, but `ops` is typed
`Dict[str, Union[ImageOperation, "ImagePipelineCore"]]`
(`_image_pipeline_core.py:202`). `ImagePipelineCore` has a second concrete
subclass (`NapariPipelineViewer`, `_napari_pipeline_viewer.py:72`) that is not an
`ImagePipeline`. Keying on `ImagePipelineCore` costs nothing and closes the gap.

## m19. The GUI swallows the new refusal

`gui/run_console/_callbacks.py:246-255` wraps `pipeline_requires_gpu` in
`except (OSError, ValueError, TypeError): return False`. `UnstageableGpuDetectorError`
is a `ValueError`, so a `meas`-slot or unstageable-container placement silently
reports "not a GPU pipeline" in the GUI rather than surfacing the message. The CLI
path (`_cli_execution_strategies.py:1341`) does *not* catch it, so there the user
gets a raw traceback. Neither is wrong, but neither is designed, and Task 3's
docstring claims the GUI *"calls the non-strict path"* as if that mattered for the
unconditional refusal.

## m20. Unused imports in two new test files

Plan `:492-494` (Task 3) imports `numpy as np` and `MeasureShape`, neither used.
Plan `:900` (Task 5) imports `numpy as np`, unused. `ruff` will flag both at the
task's own lint step.

---

# Verified correct — do not re-check

- **Prior-review blockers landed.** Spot-checked four of the five: the
  provenance-detached Stage-2 probe (Task 7 step 3, matching
  `measure/_canonical_zone_measure.py:279-295` verbatim); the process-mode
  `continuing_provenance_application` with **no** success sink and the explicit
  "do not use `set_provenance_status('in_progress')`" note (Task 11 step 3 —
  and `_provenance.py:458` really does accept `"staged"` while `:362` really
  does not); all **four** provenance fields hooked (Task 6 step 4); the
  CPU-only-slot refusal unconditional on `pipeline_requires_gpu` (Task 3 step 3).
  The `tests/_fakes/fake_gpu_detector.py` + `test_staged_routing.py:21-24`
  monkeypatch pattern is correctly quoted.
- **Task 5's cut rule is coherent with Task 8 and with the existing suite.**
  `post_ops = keys[cut:]` matches the proven spike (`spike_nested_gpu.py:156`).
  For a top-level detector the detector's own slot lands in `post_pipeline` and
  `substitute_at_path(post_pipeline, ("Sam2",), stub)` resolves through the
  `ImagePipeline` branch of `_child`/`substitute_at_path` — coherent.
  `tests/unit/cli/test_cli_pipeline_split.py:24` does assert the old exclusion
  (`== ["SmallObjectRemover"]`) and does need the stated update.
- **The spike's `branch_prefix` really does lack a root guard.** With
  `path == ("FakeGpuDetector",)` its loop body is skipped and its final block
  (`spike_nested_gpu.py:117-120`) runs against the root pipeline, returning every
  preceding top-level op. The plan's `_branch_prefix` root guards
  (`container is pipeline`, `parent is not pipeline`) are correct, and its
  ancestor coverage is exact: the loop covers containers `path[:0] … path[:-2]`
  and the final block covers `path[:-1]`, with no gap and no double-count.
  Hand-traced for depths 1, 2, 3 and 4.
- **Deferability of Tasks 2, 4 and 9 — the lead's claim is accurate.**
  `grep -rn pipeline_step_path --include=*.py src/` outside `_provenance.py`
  returns exactly one hit: the *write* at `_cli_staged_workers.py:497`, which
  Task 8 deletes. Nothing in the staging path reads a step path, so step-path
  descent is not a dependency of the run. Without Task 4 the replayed detector
  records the container's path (e.g. `["CompositeDetector"]`) — which is exactly
  what a single-pass run records for the real detector, so staged/single-pass
  provenance equivalence still holds. Only Task 9 breaks, and the plan already
  states "Task 4 … must land before 9". Task 2 is `tune/`-only with no consumer.
  Task 9 is a test. The "wide blast radius" characterisation of Task 4 is also
  right: it changes the journal for every pipeline using a container op.
- **`ReplayDetector`'s provenance hook reaches the pipeline path.** The apply
  decorator calls `append_operation_provenance` at `_provenance.py:624`, so
  hooking that one function covers the substituted stub. `OperationField` is
  `Annotated[Any, BeforeValidator, AfterValidator(_RequireValue), PlainSerializer]`
  and `_RequireValue.__call__` returns the value unchanged
  (`sdk_/typing_.py:370-378`), so identity survives `setattr` under
  `validate_assignment=True` — Task 1's `is replacement` assertions hold.
- **Task 8's removal of `_checkpoint_successful_operation` is safe.** The
  installed `provenance_success_sink` fires per successful leaf
  (`_provenance.py:629-636`) with the same roll-back-on-sink-failure semantics as
  `_checkpoint_successful_operation` (`_cli_staged_workers.py:144-158`).
- **`test_staged_store_stages.py:115-128` should still pass** under Task 8:
  the stub runs as `post_pipeline`'s first op under `pipeline_step("gpu-detect")`,
  giving `operation_name == "_FixedBlobDetector"`, `pipeline_step_path ==
  ["gpu-detect"]`, `sequence == 2`, and `duration_seconds >= compute_duration` via
  `provenance_duration_offset`.
- **The schema gate is not affected by slot keying.** `_schema_shape.py:282,364`
  and `test_schema_gate.py:613-627` deliberately do *not* fire on `stage2_done/`,
  so an extra directory level under it is invisible to the conversion verdict.
  `progress_dir` is `<output>/.phenotypic/progress` (`_io_constants.py:1032-1046`),
  so Task 6a's hand-built legacy path in its test is correct.
- **Task 11's `continuing_provenance_application` precondition holds.** Stage 1
  sets the application to `"staged"` (`_cli_staged_workers.py:337`), which
  `_provenance.py:458` accepts.
- **Task 12 is sound.** `processing_configuration_digest_from_values`
  (`_cli_failure_tracker.py:191-236`) has exactly the keyword-only signature the
  test uses, `drop_originals` defaults, and the `process_only_layer is not None`
  branch is the right home for `layer_semantics` — the existing comment at
  `:222-226` makes the same argument for `process_format`.
- **Cited `file:line` references spot-checked and correct:**
  `_cli_validation.py:135`, `_cli_pipeline_split.py:22,34-57`,
  `_cli_staged_workers.py:337,368,451,487-500`, `_cli_staged_strategy.py:246,397`,
  `_cli_staged_slurm_worker.py:182,296,310,448`, `_provenance.py:277-283,361-363,
  458,527,624,872`, `_cli_execution_strategies.py:1341`,
  `gui/_operation_registry.py:32` (plan says 33 — the docstring line),
  `gui/run_console/_callbacks.py:246-255` (plan says 253 — the call is at 254),
  `test_staged_gpu_local.py:1039`, `test_cli_pipeline_split.py:24`,
  `_composite_detector.py:126-140`, `_filamentous_fungi_detector.py:395,398,413`,
  `_two_k_filamentous_detector.py:149,154,164`,
  `measure/_canonical_zone_measure.py:279-295`, `docs/source/contrib_guide/gpu_detectors.md`
  (exists; **not referenced from any toctree** — `grep -rn gpu_detectors docs/source/`
  returns only the file itself, so Task 14 step 2's "registered in the toctree on
  this branch" is wrong and `sphinx-build` will emit a "not included in any
  toctree" warning).
- **Task 2's rescoping rationale is correct.** `gui/_operation_registry.py:32`
  walks a type-annotation tree for `_OperationFieldMarker`, not a live operation
  graph; there is nothing there for `iter_child_operations` to replace.

---

# Questions

1. **M9:** does the spec's enumerating coverage gate stay (and the plan change),
   or does the plan's closed-set rule stay (and the spec §4.3 paragraph plus
   Task 14's `_UNSUPPORTED_CONTAINERS` reference get updated)?
2. **M3/M4:** is `slot` genuinely required keyword-only everywhere — including
   `classify_staged_image` and the recovery controller, neither of which has a
   plan in scope — or does the resume/controller layer take it as an ordinary
   parameter threaded from the strategy?
3. **Task 6a scope:** the stated justification is "land it now so `N > 1` stays
   additive". Given M3's real cost (3 extra modules, ~10 test files, a signature
   change to the resume classifier), is 6a still the right call *before* the
   `F1gfd5` run, or after it?
