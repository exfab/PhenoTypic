# Plan review — nested `GpuDetector` staging

**Reviewed:** `docs/superpowers/plans/2026-09-15-nested-gpu-staging/plan.md` (1694 lines)
against `docs/superpowers/specs/2026-09-15-nested-gpu-staging/design.md` and the
worktree at `/bigdata/exfab/anguy344/PhenoTypic/.claude/worktrees/nested-gpu-staging`.

**Method:** every `file:line` and API the plan cites was opened and read. No commands
with side effects were run; every finding below is sourced from the tree as committed
at `9876ef1c`. Nothing was executed, so anything labelled "would fail" is derived by
reading the code path, not by running it — where that distinction matters I say so.

**Verdict: NEEDS REVISION.** The architecture is sound and the spike proves the
mechanism, but three defects will stop the plan on contact (Stage-2 and process-mode
provenance both raise; the provenance identity hook is incomplete in a way an *existing*
integration test already pins), and three tasks are mis-targeted at line numbers /
call sites that do not contain what the plan says they contain.

---

## Summary table

| # | Severity | Finding |
|---|---|---|
| 1 | BLOCKER | Stage-2 branch prefix applies at owner-depth 0 on a `"staged"` journal → `ValueError` |
| 2 | BLOCKER | Task 11's process-mode provenance mitigation does not work; `set_provenance_status("in_progress")` is still non-terminal |
| 3 | BLOCKER | Provenance hook covers `operation_class`/`parameters` but not `operation_name` or `duration_seconds`; breaks `test_staged_store_stages.py:115,128` |
| 4 | BLOCKER | Task 3 tests round-trip a test-local `_FakeGpu` through `from_json`, which resolves classes by bare name in the `phenotypic` namespace only |
| 5 | BLOCKER | The meas/post/filters/model refusal is unreachable in production — `pipeline_requires_gpu` calls `find_gpu_detectors` non-strict |
| 6 | MAJOR | `substitute_at_path` drops `qc`, `plots`, `name`, `benchmark`, `verbose`, `reset` and `_provenance_pipeline` when it rebuilds an `ImagePipeline` |
| 7 | MAJOR | `_CPU_ONLY_SLOTS` loop calls `.items()` on `get_model()`, which returns `Optional[ModelFitter]`, not a dict |
| 8 | MAJOR | Nobody updates `_cli_staged_strategy.py:246` — the local Stage-2 call site never receives `stage2_prefix` |
| 9 | MAJOR | Task 13 targets `plan.gpu_key` at three lines that never mention it; the single real `gpu_key` use is in a different file |
| 10 | MAJOR | `tests/unit/cli/test_cli_pipeline_split.py:26` will fail and is not in the plan |
| 11 | MAJOR | Task 3's `test_a_gpu_detector_in_the_meas_slot_is_refused` is incoherent (confirmed) |
| 12 | MAJOR | Task 2's `tune/` migration snippet does not correspond to the real code; `gui/` migration is a non-starter |
| 13 | MAJOR | Task 4 step 5 "apply the identical pattern" is wrong for `TwoKFilamentousDetector.branch_base` (`inplace=True`) |
| 14 | MAJOR | Task 12 puts the revision in the **base** digest payload, cold-starting every in-flight full/measure run |
| 15 | MINOR | `walk_operations(include_root_children=...)` is incoherent and unused (confirmed) |
| 16 | MINOR | `ReplayDetector.result` is an `NdArrayField` — an accidental `model_dump` is a memory bomb |
| 17 | MINOR | Task 9's `ids=` callable returns a non-string |
| 18 | MINOR | The loosened plot guard is asserted, never tested |
| 19 | MINOR | Nested pipelines' own `meas`/`post`/`filters`/`model` are never walked |
| 20 | MINOR | `model_copy(deep=True)` on every node of the path; spelling of `input_layer` in test fakes; unused import after Task 8 |

---

## Your six specific questions, answered first

### Q1 — Task 5 `_branch_prefix` vs the spike: **your suspicion is wrong; the plan's version is correct, and strictly better**

I traced both by hand over four shapes. They are equivalent for every nested shape,
and the plan's version *fixes* a real bug in the spike.

`spike_nested_gpu.py:101-115`:

```python
for step in path[:-1]:
    if isinstance(cursor, ImagePipeline):
        keys = list(cursor.get_ops())
        if cursor is not pipeline:          # top level split by the caller
            for k in keys[:keys.index(step)]: prefix.append(...)
    cursor = _get_step(cursor, step)
if isinstance(cursor, ImagePipeline):       # <-- NO `is not pipeline` guard
    keys = list(cursor.get_ops())
    for k in keys[:keys.index(path[-1])]: prefix.append(...)
```

The final block has **no** `cursor is not pipeline` guard. For a **top-level** GPU
detector (`path == ("Sam2",)`, `path[:-1]` empty, `cursor is pipeline`), the spike
returns every top-level op preceding `Sam2` as the Stage-2 prefix — ops Stage 1 has
already applied and written into the store. Stage 2 would re-run them on top of
themselves. The spike never exercises that shape (all three of its shapes nest), so
the bug is latent there.

The plan's version guards both loops with `parent is not pipeline`, so `len(path) == 1`
yields `[]`. Verified equivalent on:

| path | plan result | spike result |
|---|---|---|
| `("CompositeDetector","ops[0]")` | `[]` | `[]` |
| `("CompositeDetector","ops[0]","FakeGpuDetector")` | `[ContrastStretching]` | `[ContrastStretching]` |
| `("Branch","Sub","Gpu")` (pipeline in pipeline) | `[x, y]` | `[x, y]` |
| `("Sam2",)` (top level) | `[]` ✅ | all preceding top-level ops ❌ |

Two notes, both cosmetic: the `or container is pipeline` clause inside the loop is
dead (at `depth == 1` the ternary already yields `pipeline`; at `depth >= 2`,
`path[:depth-1]` is non-empty so `get_at_path` can never return the root), and the
loop is O(depth²) because it re-resolves from the root each iteration. Neither
matters at these depths.

**Fix:** none needed to the code. Add `test_a_top_level_detector_needs_no_prefix`
to Task 5 — it is the case the spike got wrong, and nothing in the plan pins it.

### Q2 — Task 6 Step 4: the site exists, but the hook as written is **incomplete**

The real site is `append_operation_provenance` at
`src/phenotypic/_core/_provenance.py:872-912`, called from
`wrap_image_operation_apply` at `:625-630`. It derives **four** things, not two:

```python
# _provenance.py:881-902
parameters = json.loads(json.dumps(operation.model_dump(mode="json"), ensure_ascii=False))
operations.append({
    "operation_name": type(operation).__name__,                                  # <-- not hooked
    "operation_class": f"{type(operation).__module__}.{type(operation).__qualname__}",
    "parameters": parameters,
    "duration_seconds": float(duration_seconds),                                 # <-- not hooked
    ...
})
```

The hook approach *does* fit — that is the one place these fields are built, and it
has exactly one caller. But the plan only hooks two of the four, and the two it
misses are pinned by an existing test. See finding 3.

Also: the plan's snippet replaces `parameters = json.loads(json.dumps(...))` with a
bare `operation.provenance_parameters()`, dropping the JSON round-trip. Keep the
round-trip around whichever source is chosen — it is what guarantees the value is
JSON-native before it reaches `validate_provenance_journal`.

### Q3 — Task 3's `test_a_gpu_detector_in_the_meas_slot_is_refused`: **yes, it is junk** (confirmed)

The first six lines build `pipe`, serialise it, deserialise it as `loaded`, then run
`object.__setattr__(x, "__dict__", x.__dict__)` — a self-assignment that does nothing.
`loaded` is never referenced again. The test's actual subject is `offending`, built
fresh three lines later. `MeasureShape` is imported solely for the dead half.

**Fix:** delete everything from `pipe = ImagePipeline(` through the `object.__setattr__`
line and the `MeasureShape` import; move `from phenotypic.measure import MeasureSymZones`
to the module header. (The test has a second, worse problem — see finding 5 — it
exercises a code path production never reaches.)

### Q4 — `substitute_at_path` and `qc`/`plots`: **yes, it silently drops them**

Confirmed. `ImagePipelineCore` declares `qc: List[QcRecipeEntry]` and
`plots: List[Any]` (`_image_pipeline_core.py:216,225`) plus `name`, `desc_value`,
`benchmark`, `verbose`, `reset`, and the `_provenance_pipeline` **PrivateAttr**
(`:231`). `split_pipeline_at_gpu` passes `qc=` and `plots=` today
(`_cli_pipeline_split.py:73-74`); Task 1's `substitute_at_path` passes neither, so
every substitution strips them. `name` is also regenerated as a fresh UUID4 by the
`default_factory`. See finding 6 for why it does not blow up *today* and why it is
still wrong.

### Q5 — `walk_operations(include_root_children=...)`: **vestigial and incoherent** (confirmed)

```python
if not include_root_children:
    return          # generator yields NOTHING, not "root children excluded"
yield from visit(pipeline, ())
```

`False` makes the generator empty. No caller in the plan passes it. Delete the
parameter.

### Q6 — Task 4 `apply_child` and the `ImagePipeline` branch: **correct**

`_composite_detector.py:126-140` passes `reset=False` for `ImagePipeline` only and
omits it otherwise; `apply_child`'s `if isinstance(operation, ImagePipeline):
kwargs["reset"] = False if reset is None else reset` reproduces that exactly. The
`if detector is None: continue` skip is preserved and `enumerate` keeps indices
stable across skipped slots. ✅

Two caveats: `PrefabPipeline(ImagePipeline)` (`abc_/_prefab_pipeline.py:9`) is covered,
but `NapariPipelineViewer(ImagePipelineCore)` is not an `ImagePipeline` — irrelevant in
practice. And the pattern is **not** transferable to the other three containers
unchanged (finding 13).

### Q7 — Task 12, does the digest actually change: **yes** ✅

`canonical_digest` (`sdk_/_digests.py:46-71`) is `json.dumps(value, sort_keys=True,
separators=(",",":"), ensure_ascii=False)` hashed. Keys are sorted, never filtered,
never allowlisted. Adding `output_semantics_revision` changes the hash. The test's
module-attribute rebinding also works, because the implementation reads the global
inside the function body. The *placement* of the key is the problem — finding 14.

---

## BLOCKERs

### 1. Stage 2's branch prefix will raise on a `"staged"` journal

**What is wrong.** Task 7 Step 3:

```python
image = image_cls.load_zarr(store)
if stage2_prefix:
    image = image.copy()
    for operation in stage2_prefix:
        operation.apply(image, inplace=True)
```

`stage2_detect_core` (`_cli_staged_workers.py:368-412`) runs at
`_application_owner_depth == 0` and has no `continuing_provenance_application`.
`ImageOperation.apply` is wrapped by `wrap_image_operation_apply`
(`abc_/_image_operation.py:411-415`), which at `:588` computes
`owns_application = parent_frame is None and _application_owner_depth.get() == 0`
→ `True`, and at `:594` calls `_append_application(...)`. That function
(`_provenance.py:361-363`) is:

```python
if applications and applications[-1]["status"] not in {"complete", "failed"}:
    raise ValueError("cannot start a new provenance application before the last ends")
```

Stage 1 leaves the trailing application `"staged"` (confirmed by
`tests/integration/cli/test_staged_store_stages.py:86` — `assert staged["status"] ==
"staged"`), and `image.copy()` carries the journal. So the first prefix op raises.

**Why it matters.** Shape B — the whole reason `stage2_prefix` exists — is dead on
arrival. The plan's own Task 7 test would fail with a `ValueError` about provenance,
not the `TypeError` its Step 2 predicts, and an executor is likely to misread that
as "the keyword isn't wired yet".

**Concrete fix.** There is a precedent in-tree for exactly this:
`measure/_canonical_zone_measure.py:280-293` detaches a throwaway copy's journal
(marking every non-terminal application `"complete"`) before running a nested op on
it, precisely so the probe's records never reach the plate. Do the same for the
Stage-2 in-memory copy — the copy is discarded, so its journal is meaningless.
`continuing_provenance_application(image)` also works (it accepts `"staged"`,
`_provenance.py:458`) but leaves the prefix's records on the copy, which is noise.
Either way the plan must say which, and Task 7's test must assert the prefix records
land nowhere.

### 2. Task 11's process-mode provenance mitigation does not work

**What is wrong.** Spec §8.2 and Task 11 Step 3 mitigate the depth-0 trap with:

```python
truncate_provenance_to_retry_base(image)
set_provenance_status(image, "in_progress")
...
residual.apply(image, inplace=True)
```

`set_provenance_status` → `_set_journal_status` (`_provenance.py:394`) sets the status
to `"in_progress"`. But `"in_progress"` is **not** in `_append_application`'s terminal
set `{"complete", "failed"}` (`_provenance.py:362`). `ImagePipelineCore.apply`
(`:966-974`) opens `provenance_application(img, ...)`, which at depth 0 takes
`owns_application = True` and calls `_append_application` — which raises the *same*
`ValueError` the mitigation was meant to avoid.

**Evidence that this was never checked:** `probe_process_provenance.py` reproduces
the trap but stops there; it never runs the proposed mitigation. The spec presents
the mitigation as settled ("Mitigation. Process mode performs Stage 3's ... handling
**in memory only**") on the strength of a probe that does not test it.

**Concrete fix.** Wrap the residual apply in `continuing_provenance_application(image)`
and install **no** `provenance_success_sink` — that is precisely the shape Stage 3
uses (`_cli_staged_workers.py:503-511`) minus the sink, and it is what keeps FLOW-16
intact. With that, `truncate_provenance_to_retry_base` + `set_provenance_status` are
not needed at all (`continuing_provenance_application` accepts `"staged"` directly,
and the image is freshly loaded on every export so there is nothing stale to truncate).
The spec's §8.2 text and the "deliberately omit `write_provenance_checkpoint`" comment
both need rewording to describe the sink, not the checkpoint call.

### 3. The provenance hook is half a hook, and an existing test already catches it

**What is wrong.** `tests/integration/cli/test_staged_store_stages.py:115-128`:

```python
expected_names = ["CropImage", "_FixedBlobDetector", "SmallObjectRemover"]
assert [entry["operation_name"] for entry in _application(completed)["operations"]] == expected_names
...
assert _application(completed)["operations"][1]["duration_seconds"] >= compute_duration
```

where `compute_duration = token["detector_duration_seconds"]` (`:108`).

Under Task 8, the detector's journal entry is produced by the wrapper for
`ReplayDetector`, so:

- `operation_name` becomes `"ReplayDetector"` — `append_operation_provenance:884`
  uses `type(operation).__name__` and the plan hooks only `operation_class`.
  **This assertion fails.**
- `duration_seconds` becomes the wrapper's measured wall time for
  `_write_object_output` alone. The current code deliberately records
  `token["detector_duration_seconds"] + merge_duration`
  (`_cli_staged_workers.py:493-497`). The stub carries
  `detector_duration_seconds` as a field but **nothing consumes it** anywhere in the
  plan. **This assertion fails** whenever GPU inference took longer than the merge —
  i.e. always, in production.

Spec §5 states the requirement ("must ... carry the Stage-2 token's
`detector_duration_seconds` plus the merge duration") and spec §10 asks for
staged/single-pass journal parity, so this is a spec↔plan gap, not a spec gap.

**Concrete fix.** Extend the hook set at `_provenance.py:881-902` to cover all four
derived fields — `provenance_operation_name()`, `provenance_operation_class()`,
`provenance_parameters()`, and a duration adjustment (either
`provenance_duration_seconds(measured)` returning `measured + self.detector_duration_seconds`,
or an additive `provenance_duration_offset()`). Add `tests/unit/cli/test_replay_detector.py`
assertions for name and duration alongside the class/parameters ones. And add
`tests/integration/cli/test_staged_store_stages.py` to Task 8's "verify existing
tests" step — the plan currently names only `test_staged_resume.py` and
`test_staged_resume_equivalence.py`, neither of which pins the journal.

### 4. Task 3's tests cannot deserialise their own fixture

**What is wrong.** Three of the five tests do
`ImagePipeline.from_json(_write(tmp_path, pipe))` with a module-local
`class _FakeGpu(GpuDetector)`. Class resolution on load is
`SerializablePipeline._find_class_in_phenotypic` (`_serializable_pipeline.py:627-678`):
it looks up the **bare class name** on the `phenotypic` module and then a hard-coded
list of thirteen `phenotypic.*` submodules. A test module's class is in neither.

This is exactly why the existing suite has a shared fake plus a registration fixture —
`tests/unit/cli/test_staged_routing.py:17-25`:

```python
from tests._fakes.fake_gpu_detector import FakeGpuDetector

@pytest.fixture(autouse=True)
def _register_fake_gpu_detector(monkeypatch):
    monkeypatch.setattr(phenotypic, "FakeGpuDetector", FakeGpuDetector, raising=False)
```

**Concrete fix.** Use `tests/_fakes/fake_gpu_detector.FakeGpuDetector` and the
autouse registration fixture in every test that round-trips through JSON. The plan
currently defines four separate near-identical `_FakeGpu` classes (Tasks 3, 5, 6, 9)
where one shared fake already exists; Tasks 5, 6 and 9 do not serialise, so they
*can* use a local class, but there is no reason to. Note `FakeGpuDetector` sets
`drop_frame_background = False` by default, which Task 6's second test needs to
override — that is fine, it is a field.

### 5. The CPU-only-slot refusal is unreachable from production

**What is wrong.** The plan's `pipeline_requires_gpu` ends with
`return bool(find_gpu_detectors(pipeline))` — i.e. `strict=False`. In non-strict mode
`find_gpu_detectors` returns `find_operations(pipeline, ...)`, and `walk_operations`
on an `ImagePipeline` root yields **only** `get_ops()` children:

```python
# plan, iter_child_operations
if isinstance(obj, ImagePipeline):
    yield from obj.get_ops().items()
    return
```

So a pipeline whose only `GpuDetector` sits in `meas`/`post`/`filters`/`model`
returns `False`, routes to `LocalParallelStrategy`
(`_cli_execution_strategies.py:1341`), and **silently runs the GPU op on CPU** — the
exact wrong-answer bug this change exists to kill. `strict=True` only ever runs
inside `split_pipeline_at_gpu`, which is only reached once `pipeline_requires_gpu`
has already said `True`.

Spec §9 row 1 explicitly requires `pipeline_requires_gpu` to "scan
`meas`/`post`/`filters`/`model` in order to refuse them". The plan does not.

**Why the plan's test does not catch it.** `test_a_gpu_detector_in_the_meas_slot_is_refused`
calls `find_gpu_detectors(offending, strict=True)` directly. It passes on a
production path that never reaches that argument. This is the "test that passes on
broken code" case you asked me to look for.

**Concrete fix.** Either have `pipeline_requires_gpu` call with `strict=True`, or
(better, because `pipeline_requires_gpu` is also called from the GUI at
`gui/run_console/_callbacks.py:253-255` where an exception is unwelcome) give
`find_gpu_detectors` a third behaviour: always scan all slots, return the ops-slot
hits, and raise for a CPU-only-slot hit regardless of `strict`. Then add a test that
drives `create_execution_strategy` / `pipeline_requires_gpu` — not
`find_gpu_detectors` — with a meas-slot GPU detector.

*Non-issue for completeness:* a `GpuDetector` cannot be a **direct** value of
`meas`/`post`/`filters`/`model`, because those fields are typed
`Dict[str, MeasureFeatures]`, `Dict[str, PostMeasurement]`,
`Dict[str, SetAnalyzer]`, `Optional[ModelFitter]`. Only *nested* ones matter, and
`walk_operations(op)` finds those (it does not yield its own root, but the root can
never be the detector here). The driver pipeline's
`MeasureSymZones.center_detector` / `MeasureOrientationZones.center_detector` are
`ManualPointDetector`s, so it passes.

---

## MAJOR

### 6. `substitute_at_path` loses six fields and a private attr

Confirmed in Q4 above. Impact assessment, so you can size it:

- **`plots`** — harmless *as the plan is written*, because Task 8 leaves
  `PlotCoordinator(plan.post_pipeline, ...)` (`_cli_staged_workers.py:522-524`)
  pointing at the un-substituted pipeline. It becomes a silent plot-loss bug the
  moment anyone "tidies" that to `replay_pipeline`. That is a booby trap.
- **`qc`** — `measure()` (`:1101`) does not read `qc`, so no immediate effect.
- **`name`** — regenerated as a fresh UUID4 per substitution
  (`_image_pipeline_core.py:190` `default_factory`).
- **`_provenance_pipeline`** — set only by `from_json`
  (`_serializable_pipeline.py:290`) and read only by `apply`/`measure` at
  `_image_pipeline_core.py:970,1094`, where it is consumed **only** when
  `owns_application` is true. Stage 3 wraps the apply in
  `continuing_provenance_application`, so depth > 0 and the value is never read.
  Verified harmless on the staged path — but only by accident, and it *is* read on
  the process-mode path once finding 2 is fixed with a `continuing_` wrapper (it
  won't be read there either, same reason). Still worth preserving.

**Concrete fix.** Rebuild with `model_copy(update={"ops": ops})` rather than
`ImagePipeline(...)`. `model_copy` carries `__pydantic_private__`, every unlisted
field, and does not re-run `_resolve_plot_bindings` — which matters, because
re-resolving plot bindings against a rebuilt op dict is its own question nobody has
asked. If the constructor form is kept for validation reasons, pass all of
`qc=`, `plots=`, `name=`, `benchmark=`, `verbose=`, `reset=` and restore
`_provenance_pipeline` explicitly, and add a Task 1 test asserting a pipeline with
`qc` and `plots` survives substitution with both intact.

### 7. `get_model()` is not a dict

```python
for slot in _CPU_ONLY_SLOTS:                       # ("meas","post","filters","model")
    container = getattr(pipeline, f"get_{slot}", None)
    for name, op in (container() or {}).items():   # <-- get_model() -> Optional[ModelFitter]
```

`get_model` is declared `-> Optional["ModelFitter"]` (`_image_pipeline_core.py:634`).
Any pipeline with a fitted model raises `AttributeError: 'XxxFitter' object has no
attribute 'items'` at split time.

**Fix:** special-case `model` as a single optional value, or normalise with
`{"model": m} if m is not None else {}`.

### 8. Nobody passes `stage2_prefix` at the local Stage-2 call site

There are exactly two callers of `stage2_detect_core`:

- `src/phenotypic/_cli/_cli_staged_strategy.py:246` (local staged strategy)
- `src/phenotypic/_cli/_cli_staged_slurm_worker.py:310` (SLURM Stage-2 shard worker)

Task 7 changes the signature and Task 13 vaguely covers the SLURM one ("Pass
`plan.stage2_prefix` into `stage2_detect_core` at the Stage-2 site"). **Nothing in
the plan touches `_cli_staged_strategy.py:246.`** So a local run of shape B silently
skips the prefix and Stage 2 infers on the wrong array. Because Task 10's equivalence
test drives the stage cores by hand rather than through the strategy, it would not
notice.

**Fix:** add the call-site edit to Task 7 (it owns the signature change) and name
both files explicitly. Add an assertion to Task 7 or Task 10 that drives
`StagedGpuStrategy` end-to-end on shape B.

### 9. Task 13 is aimed at the wrong thing

Task 13 says "Modify `_cli_staged_slurm_worker.py:182,296,448`" and "Step 1: Replace
every `plan.gpu_key` use". Verified with `grep -rn gpu_key src/`:

- `_cli_staged_slurm_worker.py:182,296,448` are three `split_pipeline_at_gpu(...)`
  calls. **None of them reads `gpu_key`.** They need no change at all, except line
  310 (Stage 2) which needs `stage2_prefix=`.
- The one and only `plan.gpu_key` read in the codebase is
  `src/phenotypic/_cli/_cli_staged_workers.py:497` — which Task 8 already deletes.
- `grep -rn gpu_key tests/` returns nothing, so no test depends on it.

**Fix:** retitle Task 13 as "pass `stage2_prefix` at the SLURM Stage-2 site
(`_cli_staged_slurm_worker.py:310`)" and delete the `gpu_key` framing, which will
otherwise send an executor hunting for edits that do not exist.

### 10. An existing split test will fail, and it is not in the plan

`tests/unit/cli/test_cli_pipeline_split.py:26`:

```python
assert list(plan.post_pipeline.get_ops().keys()) == ["SmallObjectRemover"]
```

Today `post_ops = {k: ops[k] for k in keys[cut + 1:]}` (`_cli_pipeline_split.py:51`) —
the detector is **excluded**. The plan's `keys[cut:]` **includes** it, which is
correct and necessary (the stub is substituted inside `post_pipeline`), so this
assertion must become `["FakeGpuDetector", "SmallObjectRemover"]`.

That file is not in the plan's File Structure table and not mentioned in Task 5.
`tests/unit/cli/test_cli_pipeline_split.py` is also the natural home for the new
split tests — the plan creates a parallel `test_pipeline_split_nested.py` instead,
leaving two files testing one function.

**Fix:** add the file to Task 5 with the expectation update called out as intended,
and consider folding `test_pipeline_split_nested.py` into it.

### 11. Task 3's meas-slot test is incoherent

Confirmed in Q3. See also finding 5 — the deeper problem is what it tests, not just
how it is written.

### 12. Task 2 does not match either migration target

**`gui/` cannot be migrated at all.** Spec §4.1 calls
`gui/_operation_registry.py:33` one of "two walkers over this same structure". It is
not. `_has_operation_field_marker` (`:32-58`) walks a **type annotation** with
`get_args` / `__metadata__` looking for `_OperationFieldMarker`; it never touches a
live operation. The instance-facing code at `:498-560`
(`_detect_operation_types`) is also pure annotation analysis. There is no instance
traversal in that file to share. The plan's Task 2 Step 3 hedges this correctly —
but the File Structure table, the Dependency order, and spec §9 rows 2/2a still
carry a `gui/` migration and a `gui/` regression pass that will produce nothing.

**`tune/`'s traversal is not the same traversal.** `_infer_nested_field`
(`tune/_search_space/_infer.py:617-660`) differs from `iter_child_operations` on
three axes simultaneously:

| | `iter_child_operations` | tune |
|---|---|---|
| segment | `f"{field}[{i}]"` | `f"{position}.{field}[{i}]"` (positional, not name-keyed) |
| single-valued op fields | yielded | **skipped** (`if isinstance(value, list):` only) |
| nested `ImagePipeline` | yielded | **skipped** (`_is_recursable_op:553` excludes anything with `get_ops`) |

Reproducing tune's behaviour on top of the shared primitive is three `continue`s
around a loop — which is exactly what tune already has. And the plan's snippet is
not runnable pseudo-code: `_exceeds_depth` does not exist, `depth` is not in scope,
and the call site is inside a `for field_name, field_info in type(op).model_fields.items()`
loop that needs `field_info` for `_field_holds_operation`, which
`iter_child_operations` does not yield.

**Fix — recommended:** drop Task 2 and record the finding. The spec already grants
the escape hatch ("If a single signature cannot serve all three without contortion,
the correct outcome is one shared traversal primitive with thin per-caller
adapters"). Here it serves *one* caller without contortion. Forcing the other two
buys a regression pass across `gui/` and `tune/` in exchange for no shared code.
If you disagree, the spec §4.1/§9 rows need rewriting to say what specifically is
shared, because "two walkers already exist" is not true as stated.

### 13. Task 4 step 5 is wrong for two of the three remaining containers

"Apply the identical pattern in `_composite_enhance.py`,
`_filamentous_fungi_detector.py`, and `_two_k_filamentous_detector.py`."

Read against the real call shapes:

| Site | Real call | `apply_child` default |
|---|---|---|
| `_composite_enhance.py:155,157` | `enhancer.apply(image, inplace=False[, reset=False])` | matches ✅ |
| `_filamentous_fungi_detector.py:395,398` | `self.inoculum_detector.apply(image, inplace=False[, reset=False])` | matches ✅ |
| `_two_k_filamentous_detector.py:149,151` | `self.center_detector.apply(image, inplace=False[, reset=False])` | matches ✅ |
| `_two_k_filamentous_detector.py:154` | `self.background_subtractor.apply(enhanced.copy(), inplace=False)` | matches, but the image is `enhanced.copy()`, not `image` |
| `_two_k_filamentous_detector.py:164` | `self.branch_base.apply(enhanced, inplace=True)` | **`inplace=True` — the default is wrong** |

Also, three sites in `_filamentous_fungi_detector.py` (`:413`, `:441`, `:552`)
construct and apply **inline, non-configured** operations
(`ContrastStretching().apply(enhanced_work, inplace=True)` etc.). Those are algorithm
internals, not addressable branches, and must **not** get a step segment — there is
no field name to address them by, so any segment would be un-round-trippable by the
walker and would break the §5.3 invariant Task 9 pins.

And a naming trap worth a comment in the code: `TwoKFilamentousDetector.center_detector`
**does** get a step path (it is a detector's configured branch), while
`MeasureSymZones.center_detector` **does not** (it is a measurement's private probe).
Same field name, opposite rule. Task 4's test only covers the measurer.

**Fix:** enumerate the five sites and their exact kwargs in Task 4 step 5 rather than
saying "identical pattern"; state that inline ops are excluded and why; add a test
that `TwoKFilamentousDetector.branch_base` records `["...","branch_base"]` *and* that
the inline `ContrastStretching` in `FilamentousFungiDetector` records none.

### 14. The semantics revision is in the wrong half of the digest payload

Task 12 puts `"output_semantics_revision"` in the **base** payload — the one every
mode hashes. Three lines below the insertion point is a comment explaining, at
length, why the last person to face this question chose the other branch
(`_cli_failure_tracker.py:218-223`):

> `# Beside `ext` and NOT in the base payload: a full or measure run has no process`
> `# format, and folding it into the base would change every existing run's digest`
> `# and cold-start every continuation in flight.`

The behaviour change in spec §8 is scoped to `--mode process --layer objmap`. Putting
its revision in the base payload cold-starts every in-flight `full` and `measure`
continuation on the cluster — including, by your own framing, a 33,923-image run.
That is a real cost, incurred for no correctness gain, against a documented local
precedent.

**Fix:** fold the revision into the `process_only_layer is not None` branch, beside
`process_format`. If you want per-layer precision, fold in
`f"{process_only_layer}:{OUTPUT_SEMANTICS_REVISION}"` so a `gray` export is not
invalidated by an `objmap` semantics change either. Either way the constant's
docstring should say which outputs the revision governs, because "per-image output
semantics" read broadly is what produced the base-payload placement.

Two existing tests were checked and survive a bump either way:
`tests/unit/cli/test_process_format_cli.py:141-143` and
`tests/unit/cli/test_cli_provenance_original.py:113` both assert digest
*relationships*, not literals. `tests/unit/cli/test_migrate_state.py:742`
(`test_work_ids_are_untouched`) compares planted ids against planted ids, so it is
also safe — but its docstring ("D-C keeps `processing_configuration_digest`
unchanged") becomes misleading under a base-payload bump.

---

## MINOR

### 15. `walk_operations(include_root_children=...)`
Covered in Q5. Delete it.

### 16. `ReplayDetector.result` is a serialisable ndarray field

`NdArrayField` carries `PlainSerializer(_ndarray_to_list, return_type=list)`
(`sdk_/typing_.py:250-255`). Any `model_dump(mode="json")` on a `ReplayDetector`
holding a 4000×6000 objmap materialises a 24-million-element Python list. The
provenance hook (finding 3) is what keeps that out of the journal — so that hook is
load-bearing for *memory*, not just for cosmetics, and the plan should say so at the
site. Consider `Field(exclude=True)` on `result` as a second line of defence, and
note that `ReplayDetector` is not resolvable by `_find_class_in_phenotypic` either,
so a substituted pipeline must never be `to_json()`'d (it isn't, today).

### 17. Task 9's `ids=` callable

```python
@pytest.mark.parametrize("name,detector", list(_shapes()), ids=lambda v: getattr(v, "__name__", v))
```

For the `detector` argument this returns a `CompositeDetector` **instance**, not a
string. pytest passes a non-`None` id result to `ascii_escaped`, which expects
`str`/`bytes`. Use `ids=[n for n, _ in _shapes()]` or drop `ids` entirely (the `name`
parameter already carries it).

### 18. The loosened plot guard is asserted but untested

Task 5 changes `ref.key == gpu_key or ref.key in pre_ops` → `ref.key in pre_ops`
(`_cli_pipeline_split.py:57`) and notes "A plot referencing the ancestor is now
legal, because the ancestor runs in Stage 3."

For a **top-level** detector that also newly legalises a plot bound to the GPU
detector's own key — and the object at that key in `plan.post_pipeline` is the
*real* detector, which never runs; the stub in `replay_pipeline` is what runs.
`PlotCoordinator` is constructed from `plan.post_pipeline`
(`_cli_staged_workers.py:522`), so what such a plot would emit is unexamined.
The existing `test_pre_gpu_plot_reference_is_rejected`
(`tests/unit/cli/test_cli_pipeline_split.py:56-61`) still passes because its plot op
is in `pre_ops`, so nothing notices.

**Fix:** either keep refusing a plot bound to `gpu_path[0]` when
`len(gpu_path) == 1`, or add a test that pins what it emits. Do not leave it as a
one-line note.

### 19. A nested pipeline's own slots are never walked

`iter_child_operations` early-returns after `get_ops()` for an `ImagePipeline`, so a
`GpuDetector` inside a *nested* pipeline's `meas`/`post`/`filters`/`model` is invisible
to both detection and refusal. Narrow, but it is the same class of silent-CPU-fallback
bug this change exists to close; worth one sentence in the docstring saying it is
out of scope, or one extra branch.

### 20. Smaller items

- **`model_copy(deep=True)` per node** (`substitute_at_path`). Deep-copies the
  addressed node's whole subtree — including the original `GpuDetector` and any
  `PrivateAttr` it holds. Harmless in Stage 3 (the model is not loaded), but
  `_cli_staged_slurm_worker.py:297` does call `plan.gpu_detector._ensure_model_loaded()`
  on the Stage-2 worker, so the pattern is one refactor away from trying to
  `deepcopy` a live torch model. A shallow per-node copy is sufficient for the
  "root is never mutated" contract.
- **`input_layer: str = "detect_mat"`** in the plan's test fakes widens a
  `Literal["rgb","gray","detect_mat"]` (`sdk_/typing_.py:110`) to `str`. Works, but
  just construct with `input_layer="detect_mat"` — `FakeGpuDetector` already leaves
  the field alone for exactly this reason.
- **`detector_duration_seconds: float = 0.0`** does *not* trip the tune
  annotation-coverage gate: that gate's denominator is `detect/` + `enhance/`
  `__all__` (`tests/unit/tune/test_annotation_coverage.py:3-8`), and
  `ReplayDetector` lives in `_cli/`. Verified — no `TuneSpec` needed. Adding
  `TuneSpec(tunable=False)` anyway would match the `adding-an-operation` convention.
- **Unused import after Task 8.** `append_operation_provenance` is imported at
  `_cli_staged_workers.py:31` for its single use at `:490`. Task 8 deletes that use;
  ruff will flag F401. Add the removal to Task 8 step 1.
- **Docs targets are unnamed.** Task 14 says "`docs/source/how_to/` pages describing
  the objmap export". The page is `docs/source/how_to/pages/gpu_detection_setup.md:485`
  ("runs Stages 1-2, then writes one objmap PNG per image"). `src/phenotypic/_cli/CLAUDE.md:28`
  also spells the plan shape as `StagePlan{pre_pipeline, gpu_detector, post_pipeline}`
  and needs `gpu_path` / `stage2_prefix` added. Name both.
- **Line-number drift in the spec**, all within ±5 and all resolvable, but worth a
  sweep before anyone cites them: spec cites `_image_pipeline_core.py` which is
  really `_core/_pipeline_parts/_image_pipeline_core.py`; `_cli_pipeline_split.py:56-68`
  is really `:53-62`; `_cli_execution_strategies.py:341/906/1341` are `:337/903/1339`;
  `_composite_detector.py:131-138` is `:126-140`.

---

## What I verified and found correct

These are claims I checked and can confirm, so you know what does not need re-checking:

- **The bug is real, exactly as spec §2 describes it.** `_cli_validation.py:147`
  scans `pipeline.get_ops().values()` only. All three downstream consequences
  resolve: `_cli_execution_strategies.py:1339-1341` (staged routing),
  `:903-910` (`slurm_gpus_per_node`), `:337-344` + `:361` (`effective_n_jobs = 1`).
- **`pipeline_step` has exactly one caller today** — `_image_pipeline_core.py:900`
  (spec §5.2). `grep -rn "pipeline_step("` across `src/` returns that line and the
  definition. ✅
- **`pipeline_step_path` is validated as non-empty strings** (`_provenance.py:277-283`),
  so the `field[i]` string form is mandatory. ✅
- **`_write_object_output` is on `GpuDetector`**, not `ObjectDetector`
  (`abc_/_gpu_detector.py:227-250`), and owns `drop_frame_background` +
  `split_disconnected_labels` in that order. Task 6's delegation rationale and its
  second test (`num_objects == 1` after a background-spanning label is dropped) are
  correct.
- **`CompositeDetector` calls `detector.apply(image, inplace=False)` and reads
  `detected_image.objmap[:]`** (`_composite_detector.py:126-140`), so the stub
  satisfies the contract with no change to the composite. ✅
- **`center_detector` already runs on a provenance-detached copy**
  (`measure/_canonical_zone_measure.py:279-295`, from `10f4606b`), so Task 4's third
  test does pin existing behaviour and should pass before the change, as the plan
  predicts. ✅
- **Task 1's identity assertions hold.** `_deserialize_operation_value`
  (`sdk_/typing_.py:338-340`) passes live instances through by reference,
  `_make_require_value` is an isinstance guard, `BaseOperation.model_config` sets no
  `revalidate_instances`, and `_normalize_operation_collection`
  (`_image_pipeline_core.py:77-79`) passes a dict through unchanged preserving key
  order. So `get_at_path(out, path) is replacement` and the sibling/original
  assertions are sound.
- **`canonical_digest` does not filter or allowlist keys** — adding one changes the
  hash (Q7). ✅
- **Stage 1 sets the journal's pipeline identity explicitly**
  (`_cli_staged_workers.py:296-299` → `_initialize_stage1_provenance`), not from the
  pipeline object, so the `_provenance_pipeline` loss in finding 6 does not corrupt
  the staged journal. ✅
- **`stage2_detect_core`'s current body matches the plan's Step 3 splice point** —
  `array = getattr(image, detector.input_layer)[:]` at `_cli_staged_workers.py:384`
  is exactly the line the snippet replaces. ✅
- **Task 3's remaining four tests are logically sound** (given finding 4's fixture fix).
- **Every row of spec §9 has a task.** The mapping in the plan's Self-Review Notes
  checks out row-by-row. The gaps are *inside* tasks (findings 3, 5, 8, 9, 12), not
  missing rows. I found no scope creep — nothing in the plan is absent from the spec.

---

## Suggested revision order

1. Findings 1, 2, 3, 5 — the four correctness holes. Each is a small edit but each
   changes what a task's tests must assert.
2. Findings 8, 9, 10 — the three mis-targeted tasks. Cheap; pure re-aiming.
3. Finding 12 — decide on Task 2 before dispatching, because it changes the
   dependency graph (`1 → {2,3,5}` becomes `1 → {3,5}`) and removes two regression
   passes.
4. Findings 6, 7, 13, 14 — implementation corrections inside tasks that are
   otherwise well-specified.
5. The MINORs can ride along with whichever task owns the file.
