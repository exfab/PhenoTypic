# Plan A — Nested `GpuDetector` Staging via the Composite Path

> **First of two plans, in sequence — this one lands first.** See `README.md`.
> Plan A repairs the composite path with no changes to the operation interface,
> and is the shortest route to a working run. Plan B
> (`plan-b-phase-protocol.md`) follows it and supersedes parts of it; nothing
> here needs undoing to start B.
>
> **Plan A is the reviewed one.** It has been through an independent plan review
> (`docs/superpowers/reports/2026-09-15-nested-gpu-staging/plan-review.md`, 20
> findings, all applied). Plan B has not.

## Scope, and what it deliberately excludes

**In:** a single `GpuDetector` nested inside `ImagePipeline`, `CompositeDetector`
or `CompositeEnhance`, at any depth.

**Out, by decision:**

| Excluded | Why |
|---|---|
| `FilamentousFungiDetector`, `TwoKFilamentousDetector`, any domain detector | Only composition primitives may carry a staged detector (spec §4.3). `FilamentousFungiDetector` *would* classify as `"parallel"` today, and is still refused — that behaviour is incidental to its algorithm, not part of what the class is |
| More than one `GpuDetector` | Deferred; spec §13 records the intended `N > 1` model, and Task 6a lands the slot-keyed signal now so it stays additive |
| Sub-phase decomposition of an operation | That is Plan B |


> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the staged GPU engine run a pipeline whose single `GpuDetector` is nested inside another operation (e.g. inside `CompositeDetector.ops`) instead of being a top-level element of `ImagePipeline.get_ops()`.

**Architecture:** Address the GPU op by a *tree path* rather than a top-level key. Cut the pipeline at the GPU op's top-level ancestor; Stage 2 runs the detector alone (after any CPU prefix inside its own branch); Stage 3 substitutes a replay stub for the nested detector so the enclosing operation runs normally with the recorded mask standing in for live inference. Container operations additionally push a per-branch `pipeline_step`, which makes the walker's `gpu_path` and the recorded `pipeline_step_path` the same value.

**Tech Stack:** Python 3.11+, pydantic v2, zarr v3 / OME-Zarr 0.5, numpy, pytest, `uv` as the sole package manager and runner.

**Spec:** `docs/superpowers/specs/2026-09-15-nested-gpu-staging/design.md` (committed at `83a8680f`). The plan argues from the spec; executors read both.

## Global Constraints

- **`uv` is the sole package manager and runner.** Never bare `python` or `pip`. Run commands as `uv run <cmd>`.
- **Operations are pydantic v2 models, constructed keyword-only.** Parameters are annotated class-level fields; no hand-written `__init__`. Guards go in `field_validator`, never `__init__`.
- **`pipeline_step_path` is a list of non-empty strings** (`_provenance.py:277-283`). An integer branch index is illegal — always the `field[i]` string form.
- **Exactly one `GpuDetector` per pipeline**, anywhere in the tree.
- **Stage 2 never writes into the per-image store.** Its outputs are the retained raw `.npy` and the token under `.phenotypic/progress/`.
- **A GPU round is a full-dataset sweep with the model resident — never a per-image interleave** (spec §13.4). This binds every task that touches Stage 2, including the slot keying in Task 6a. Breaking it destroys model residency (paying `_cli_process_single.py:260`'s per-image rebuild cost, the defect staging exists to fix) and permanently forecloses cross-image batching. Note the engine does not batch *yet* — `_cli_staged_workers.py:389` collates a one-element list per image — so the sweep is protecting headroom, not just current behaviour.
- **`_export_objmap_layer` never writes into the store** (ledger FLOW-16 / FLOW-30 / FLOW-6). A store write after the success marker invalidates the descriptor the marker just recorded.
- **A measurement's nested operation is a private probe.** Its steps deliberately do not enter the plate's provenance (`measure/CLAUDE.md`). `center_detector` must keep recording no step path.
- **Vendored reference sources under `docs/superpowers/specs/*/refs/` are read-only.** Never lint or reformat them.
- **`ruff` is always given explicit paths.** `uv run ruff check --fix <paths you changed>` — never bare.
- **Test running:** use the `run-phenotypic-test` skill for any non-trivial pytest invocation. Never `-n auto` (it reads node cores, not the allocation). Always `QT_QPA_PLATFORM=offscreen` for GUI tests. The full suite is a Slurm job (~65 min), run **once** at the end — not between tasks.
- **Per-task testing:** run only the directly-touched test files (~1 minute). Per-phase, run the affected surface once.

---

## Phase gates

Tasks group into phases with one gate each. **Match the instrument to the
stage** — the full suite is the *last* check, never a step-level one.

| Phase | Tasks | Gate scope | Where | Excludes |
|---|---|---|---|---|
| 0 Foundation | 1 | `tests/unit/sdk_/test_operation_tree.py` | local, seconds | everything else |
| 0b *(deferrable)* | 2 | `tests/unit/tune` — 102 files | **Slurm**, 1 task | — |
| 1 **Detection** | 3 | new file + `test_staged_routing.py` | local, ~1 min | the whole staged surface |
| 2 *(deferrable)* | 4 | `tests/unit/core -k provenance`, `tests/unit/detect`, `tests/unit/enhance` — 67 files | **Slurm**, 1 task | `cli`, `gui` |
| 3 Split & replay | 5, 6, 6a, 7, 8, 13 | `tests/unit/cli` + `tests/integration/cli` — 102 files | **Slurm**, 4 shards | `gui`, `tune` |
| 4 Equivalence | 9, 10 | the invariant + equivalence files | **Slurm**, 1 task | everything else |
| 5 Process mode | 11, 12 | `tests/integration/cli` + the process/format unit files | **Slurm**, 2 shards | `gui` |
| 6 Regression | 14, 15 | all 734 files in `testpaths` | **Slurm array**, 24 shards | nothing |

**Every phase gate runs against a frozen checkout, never the live worktree.**
A parallel gate measures ONE tree: a file edited while an array's shards are
spread across nodes yields a union across two trees that no single tree ever
produced — void, not stale, with nothing in the output saying so. This is not
hypothetical here; three clusters ran concurrently in one worktree during
Phase 0-2 and every focused check taken in that window had to be labelled
"mixed tree" and re-taken.

`make_gate_tree.sh` builds the checkout and refuses to return a path unless the
tree is clean AND `import phenotypic` resolves to that tree — the editable
install is a bare `.pth` path entry, so a mis-synced gate tree silently imports
the live worktree's source and attributes every number to the wrong commit.
A `uv sync` there costs ~50s.

```bash
TREE=$(docs/superpowers/plans/2026-09-15-nested-gpu-staging/make_gate_tree.sh HEAD)
PHENO_GATE_TREE=$TREE PHENO_GATE_PATHS="tests/unit/cli tests/integration/cli" \
  sbatch --array=0-3%4 docs/superpowers/plans/2026-09-15-nested-gpu-staging/run_phase_gate.sbatch
```

The array prints its commit, its dirty-file count and its resolved
`phenotypic.__file__` in every task's log, so a contaminated run is visible in
the artifact rather than only in the submitter's intent.

**Two corrections to an earlier draft of this table, made at execution time:**

- **Task 9 moved from Phase 2 to Phase 4.** Its Interfaces block consumes Tasks
  4 *and 5*, and Task 5 is in Phase 3 — so Phase 2 could not contain it. Worse,
  Phase 2's gate *excludes* `cli`, and Task 9's only deliverable is
  `tests/unit/cli/test_gpu_path_is_the_step_path.py`: the gate structurally
  could not run the phase's own test. Phase 2 is now Task 4 alone.
- **Task 13 moved from Phase 5 to Phase 3.** Task 7 adds a `stage2_prefix`
  parameter defaulting to `None`; until Task 13 forwards it at both call sites,
  shape B is silently broken. Landing 13 in the same phase that introduces the
  parameter closes that window instead of holding it open across two gates.

**A narrow gate can be green while the default lane is red.** `testpaths` covers
`tests/unit`, `tests/smoke`, `tests/integration` **and** `tests/gui`, so naming
one path narrows the run — which is why every row above carries an *Excludes*
column. Phase 3 being green says nothing about `tests/gui`.

**`mypy` and `ruff` are already red at baseline.** Measured on a frozen
checkout at `5aaeeb77` (Task 1 landed, nothing else), `uv run mypy
src/phenotypic` reports **435 errors in 127 files** and `uv run ruff check
src/phenotypic` reports **25**. A gate that runs them compares against those
counts; it never reports "passes".

An earlier draft of this line said 417 / 124. That figure was stale and is
retracted — it was carried forward from an older measurement rather than
re-taken, which is exactly the mistake the frozen-tree rule below exists to
prevent.

**Never `-x` for a number you intend to record.** It stops at the first failure
and `tests/unit/cli` sorts early, so a run that looks like a clean sweep may have
covered a third of the set.

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `src/phenotypic/sdk_/_operation_tree.py` | **new** — the single shared traversal over operation-bearing children; path addressing; `get`/`substitute` | 1 |
| `src/phenotypic/tune/_search_space/_infer.py` | migrate its recursion onto the shared traversal | 2 |
| `src/phenotypic/_cli/_cli_validation.py` | `pipeline_requires_gpu` → tree-wide; refusals for unstageable placements | 3 |
| `src/phenotypic/_core/_provenance.py` | `apply_child()` helper wrapping `pipeline_step` | 4 |
| `src/phenotypic/detect/_composite_detector.py` | adopt `apply_child` | 4 |
| `src/phenotypic/enhance/_composite_enhance.py` | adopt `apply_child` | 4 |
| `src/phenotypic/detect/_filamentous_fungi_detector.py` | adopt `apply_child` | 4 |
| `src/phenotypic/detect/_two_k_filamentous_detector.py` | adopt `apply_child` | 4 |
| `src/phenotypic/_cli/_cli_pipeline_split.py` | `StagePlan.gpu_path`, `stage2_prefix`, ancestor-based cut, plot guard | 5 |
| `src/phenotypic/_cli/_cli_replay_detector.py` | **new** — `ReplayDetector` | 6 |
| `src/phenotypic/_cli/_cli_stage2_token.py` | slot-keyed raw/token paths + legacy relocation | 6a |
| `src/phenotypic/_cli/_cli_staged_workers.py` | Stage-2 prefix; Stage-3 stub substitution | 7, 8 |
| `src/phenotypic/_cli/_cli_staged_strategy.py` | process-mode post-detector op chain (11); forward `stage2_prefix` at the local Stage-2 call site `:246` (13) | 11, 13 |
| `src/phenotypic/_cli/_cli_failure_tracker.py` | output-semantics revision in the work-id digest | 12 |
| `src/phenotypic/_cli/_cli_staged_slurm_worker.py` | forward `stage2_prefix` at the SLURM Stage-2 call site (`:310`) | 13 |

**Dependency order:** 1 → {2, 3, 5} → 6 → 6a → {7, 8, 13} → {9, 10} → 11 → 12 → 14 → 15.
Task 9 consumes Tasks 4 and 5 only — **not** 6/6a/7/8, which an earlier draft of
this line implied by placing it after {7, 8}.
Task 4 is independent of the staging chain and may run in parallel with 2/3/5,
but must land before 9. Task 2 is `tune/`-only (see its note) and is on no
critical path — it can be deferred without blocking anything.

**Task 13 is not optional cleanup.** Task 7 adds a `stage2_prefix` parameter that
**defaults to `None`**, so until Task 13 forwards it at both call sites, shape B
is silently broken with no error anywhere. Do not stop after Task 12.

---

## Task 1: Shared operation-tree traversal

**Files:**
- Create: `src/phenotypic/sdk_/_operation_tree.py`
- Test: `tests/unit/sdk_/test_operation_tree.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `iter_child_operations(obj) -> Iterator[tuple[str, object]]`
  - `walk_operations(pipeline) -> Iterator[tuple[tuple[str, ...], object]]`
  - `find_operations(pipeline, predicate) -> list[tuple[tuple[str, ...], object]]`
  - `get_at_path(root, path: Sequence[str]) -> object`
  - `substitute_at_path(root, path: Sequence[str], replacement) -> object`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/sdk_/test_operation_tree.py
import pytest

from phenotypic import ImagePipeline
from phenotypic.detect import CompositeDetector, ManualPointDetector, OtsuDetector
from phenotypic.enhance import BlurGauss
from phenotypic.sdk_._operation_tree import (
    find_operations,
    get_at_path,
    substitute_at_path,
    walk_operations,
)

CENTERS = [[10.0, 10.0], [10.0, 40.0]]


def _pipeline_with_composite():
    return ImagePipeline(
        ops={
            "BlurGauss": BlurGauss(sigma=2.0),
            "CompositeDetector": CompositeDetector(
                ops=[OtsuDetector(),
                     ManualPointDetector(centers=CENTERS, shape="disk", width=11)],
                mode="overlap",
            ),
        }
    )


def test_walk_yields_list_entries_as_bracket_indexed_strings():
    pipe = _pipeline_with_composite()
    paths = {"/".join(p) for p, _ in walk_operations(pipe)}
    assert "BlurGauss" in paths
    assert "CompositeDetector" in paths
    assert "CompositeDetector/ops[0]" in paths
    assert "CompositeDetector/ops[1]" in paths


def test_every_path_segment_is_a_non_empty_string():
    """pipeline_step_path validation rejects integers and empty strings."""
    pipe = _pipeline_with_composite()
    for path, _ in walk_operations(pipe):
        assert path, "empty path"
        for segment in path:
            assert isinstance(segment, str) and segment


def test_find_operations_locates_a_nested_type():
    pipe = _pipeline_with_composite()
    hits = find_operations(pipe, lambda op: isinstance(op, ManualPointDetector))
    assert len(hits) == 1
    assert hits[0][0] == ("CompositeDetector", "ops[1]")


def test_get_at_path_round_trips_with_walk():
    pipe = _pipeline_with_composite()
    for path, op in walk_operations(pipe):
        assert get_at_path(pipe, path) is op


def test_substitute_replaces_only_the_addressed_node():
    pipe = _pipeline_with_composite()
    replacement = OtsuDetector(ignore_zeros=True)
    out = substitute_at_path(pipe, ("CompositeDetector", "ops[1]"), replacement)

    assert get_at_path(out, ("CompositeDetector", "ops[1]")) is replacement
    # sibling untouched, and the ORIGINAL pipeline is not mutated
    assert isinstance(get_at_path(out, ("CompositeDetector", "ops[0]")), OtsuDetector)
    assert isinstance(
        get_at_path(pipe, ("CompositeDetector", "ops[1]")), ManualPointDetector
    )


def test_substitute_at_depth_two():
    inner = CompositeDetector(ops=[OtsuDetector(), OtsuDetector()], mode="union")
    pipe = ImagePipeline(
        ops={"CompositeDetector": CompositeDetector(ops=[inner, OtsuDetector()],
                                                    mode="overlap")}
    )
    replacement = OtsuDetector(ignore_zeros=True)
    path = ("CompositeDetector", "ops[0]", "ops[0]")
    out = substitute_at_path(pipe, path, replacement)
    assert get_at_path(out, path) is replacement


def test_get_at_path_raises_on_unknown_path():
    pipe = _pipeline_with_composite()
    with pytest.raises(KeyError):
        get_at_path(pipe, ("CompositeDetector", "ops[9]"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/sdk_/test_operation_tree.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'phenotypic.sdk_._operation_tree'`

- [ ] **Step 3: Write the implementation**

```python
# src/phenotypic/sdk_/_operation_tree.py
"""The single traversal over operation-bearing children of a pipeline.

Three callers share this module and each wants something slightly different
(see the spec §4.1): the CLI wants *paths to GpuDetectors*, ``gui/`` wants
*marker presence on an annotation*, ``tune/`` wants *one-level list recursion
with its own depth rule*. The primitive here is the traversal; each caller
adapts it rather than reimplementing it.

Path segments are always **non-empty strings**, because a path is also a
``pipeline_step_path``, which ``_provenance.validate_provenance_journal``
rejects unless every segment is a non-empty string. A list entry is therefore
addressed ``"ops[0]"``, never ``("ops", 0)``.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterator, Sequence

_INDEXED = re.compile(r"^(?P<field>[^\[\]]+)\[(?P<index>\d+)\]$")


def _is_operation(value: Any) -> bool:
    """True for anything that can carry further operations."""
    from phenotypic._core._image_pipeline import ImagePipeline
    from phenotypic.abc_ import ImageOperation, MeasureFeatures

    return isinstance(value, (ImageOperation, MeasureFeatures, ImagePipeline))


def iter_child_operations(obj: Any) -> Iterator[tuple[str, Any]]:
    """Yield ``(segment, child)`` for each operation-bearing child of *obj*."""
    # Key on ImagePipelineCore, not ImagePipeline: `ops` is typed
    # Dict[str, Union[ImageOperation, "ImagePipelineCore"]]
    # (`_image_pipeline_core.py:202`), and ImagePipelineCore has a second
    # concrete subclass (`NapariPipelineViewer`) that is not an ImagePipeline.
    from phenotypic._core._pipeline_parts._image_pipeline_core import (
        ImagePipelineCore,
    )

    if isinstance(obj, ImagePipelineCore):
        yield from obj.get_ops().items()
        return

    model_fields = getattr(type(obj), "model_fields", None)
    if not model_fields:
        return

    for field_name in model_fields:
        value = getattr(obj, field_name, None)
        if isinstance(value, list):
            for index, item in enumerate(value):
                if _is_operation(item):
                    yield f"{field_name}[{index}]", item
        elif _is_operation(value):
            yield field_name, value


def walk_operations(pipeline: Any) -> Iterator[tuple[tuple[str, ...], Any]]:
    """Depth-first walk yielding ``(path, operation)`` for every node.

    Does not yield the root itself (its path would be empty, and an empty
    ``pipeline_step_path`` is invalid).

    KNOWN LIMIT: for a **nested** ``ImagePipeline`` this descends only its
    ``ops``, not its own ``meas``/``post``/``filters``/``model``. A GpuDetector
    hidden in a nested pipeline's ``meas`` is therefore neither staged nor
    refused. That shape is not reachable from the GUI builder and has no known
    user, so it is out of scope here -- but it is a gap, not an invariant, and
    the CPU-only-slot refusal covers only the ROOT pipeline's slots.
    """

    def visit(node: Any, path: tuple[str, ...]) -> Iterator[tuple[tuple[str, ...], Any]]:
        if path:
            yield path, node
        for segment, child in iter_child_operations(node):
            yield from visit(child, path + (segment,))

    yield from visit(pipeline, ())


def find_operations(
    pipeline: Any, predicate: Callable[[Any], bool]
) -> list[tuple[tuple[str, ...], Any]]:
    """Every ``(path, operation)`` in *pipeline* satisfying *predicate*."""
    return [(path, op) for path, op in walk_operations(pipeline) if predicate(op)]


def _child(node: Any, segment: str) -> Any:
    from phenotypic._core._image_pipeline import ImagePipeline

    matched = _INDEXED.match(segment)
    if matched is not None:
        field = matched.group("field")
        index = int(matched.group("index"))
        sequence = getattr(node, field, None)
        if not isinstance(sequence, list) or index >= len(sequence):
            raise KeyError(segment)
        return sequence[index]
    if isinstance(node, ImagePipeline):
        ops = node.get_ops()
        if segment not in ops:
            raise KeyError(segment)
        return ops[segment]
    if not hasattr(node, segment):
        raise KeyError(segment)
    return getattr(node, segment)


def get_at_path(root: Any, path: Sequence[str]) -> Any:
    """Resolve *path* against *root*; raise ``KeyError`` if absent."""
    node = root
    for segment in path:
        node = _child(node, segment)
    return node


def substitute_at_path(root: Any, path: Sequence[str], replacement: Any) -> Any:
    """Return a copy of *root* with the node at *path* replaced.

    *root* is never mutated: each node on the path is copied on the way down.
    """
    from phenotypic._core._image_pipeline import ImagePipeline

    if not path:
        return replacement

    head, rest = path[0], tuple(path[1:])

    if isinstance(root, ImagePipeline):
        ops = dict(root.get_ops())
        if head not in ops:
            raise KeyError(head)
        ops[head] = (
            replacement if not rest
            else substitute_at_path(ops[head], rest, replacement)
        )
        # Rebuild carrying every slot a Stage-3 pipeline READS. An earlier
        # draft dropped `qc`, `plots`, `name` and `_provenance_pipeline` -- a
        # booby trap for whoever next reads plots off a substituted pipeline.
        # NOT carried, deliberately: `benchmark`, `verbose`, `reset`,
        # `desc_value` (`_image_pipeline_core.py:190-221`). Harmless for a
        # throwaway Stage-3 pipeline, but do not describe this as complete.
        rebuilt = ImagePipeline(
            ops=ops,
            meas=root.get_meas(),
            post=root.get_post(),
            filters=root.get_filters(),
            model=root.get_model(),
            qc=root.get_qc(),
            plots=root.get_plots(),
            nrows=root.nrows,
            ncols=root.ncols,
        )
        rebuilt.name = root.name
        rebuilt._provenance_pipeline = root._provenance_pipeline
        return rebuilt

    # SHALLOW, deliberately. model_copy(deep=True) would copy the entire
    # subtree -- including the real GpuDetector and whatever its PrivateAttr
    # holds -- only to overwrite one child of it. Measured: a deep copy does
    # carry PrivateAttr through and allocates a new child object, so a loaded
    # torch model in that subtree would be deep-copied and discarded. Latent
    # today (Stage 3 never loads a model) and free to avoid.
    #
    # Untouched siblings stay shared by reference, which is exactly what the
    # original pipeline does with them.
    node = root.model_copy(deep=False)
    matched = _INDEXED.match(head)
    if matched is not None:
        field = matched.group("field")
        index = int(matched.group("index"))
        sequence = list(getattr(node, field))
        if index >= len(sequence):
            raise KeyError(head)
        sequence[index] = (
            replacement if not rest
            else substitute_at_path(sequence[index], rest, replacement)
        )
        setattr(node, field, sequence)
        return node

    if not hasattr(node, head):
        raise KeyError(head)
    current = getattr(node, head)
    setattr(
        node,
        head,
        replacement if not rest else substitute_at_path(current, rest, replacement),
    )
    return node
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/sdk_/test_operation_tree.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/sdk_/_operation_tree.py tests/unit/sdk_/test_operation_tree.py
uv run mypy src/phenotypic/sdk_/_operation_tree.py
git add src/phenotypic/sdk_/_operation_tree.py tests/unit/sdk_/test_operation_tree.py
git commit -m "feat(sdk_): add the shared operation-tree traversal"
```

---

## Task 2: Consolidate `tune/`'s traversal onto the shared walker

**Files:**
- Modify: `src/phenotypic/tune/_search_space/_infer.py`
- Test: existing `tests/unit/tune/` suite (no new behaviour)

> **Rescoped after plan review.** The spec (§4.1) decided to consolidate and
> migrate both `gui/` and `tune/`. On inspection **`gui/` has no instance
> traversal to share**: `gui/_operation_registry.py:33` (`_has_operation_field_marker`)
> walks a **type annotation tree** looking for `_OperationFieldMarker`, never a
> live operation graph. There is nothing there for `iter_child_operations` to
> replace. The spec anticipated this outcome explicitly — "If a single signature
> cannot serve all three without contortion, the correct outcome is one shared
> traversal primitive with thin per-caller adapters" — so `gui/` is left alone
> **deliberately**, and this note is the recorded reason.
>
> Consequence for the plan's dependency graph: `1 → {2, 3, 5}` stands, but Task 2
> no longer needs a `gui/` regression pass, and inventory row 2a narrows to
> `tune/` only.

**Interfaces:**
- Consumes: `iter_child_operations` from Task 1.
- Produces: no new public surface. Behaviour-preserving.

- [ ] **Step 1: Capture the pre-migration baseline**

```bash
uv run pytest tests/unit/tune -q 2>&1 | tail -5
```

Record the count. The migration must reproduce it exactly.

- [ ] **Step 2: Read the real recursion before editing**

Do **not** work from a snippet. Open `_infer.py` around `:429`, `:546` and `:703`
and identify where it descends into a nested operation value. The plan
deliberately shows no replacement code here: an earlier draft's snippet did not
correspond to the actual control flow, and a wrong snippet is worse than none.

- [ ] **Step 3: Replace only the child-enumeration, keep `tune`'s own policy**

`tune/` recurses a list-valued `OperationField` **one level by default**. That
depth rule is the caller's policy and must stay in `tune/` — do not move it into
`_operation_tree.py`. Only the "what are this operation's operation-valued
children" question moves.

- [ ] **Step 4: Verify the baseline is reproduced exactly**

```bash
uv run pytest tests/unit/tune -q 2>&1 | tail -5
```

Expected: identical pass/fail counts to Step 1. **Any change is a regression** —
this task adds no behaviour.

- [ ] **Step 5: Commit**

```bash
uv run ruff check --fix src/phenotypic/tune/_search_space/_infer.py
git add src/phenotypic/tune/_search_space/_infer.py
git commit -m "refactor(tune): use the shared operation-tree traversal"
```

---

## Task 3: Detect GPU detectors tree-wide, and refuse unstageable placements

**Files:**
- Modify: `src/phenotypic/_cli/_cli_validation.py:135-147`
- Test: `tests/unit/cli/test_gpu_detection_tree_wide.py` (create)

**Interfaces:**
- Consumes: `find_operations` from Task 1.
- Produces:
  - `pipeline_requires_gpu(pipeline_path: Path) -> bool` (unchanged signature, tree-wide behaviour)
  - `find_gpu_detectors(pipeline) -> list[tuple[tuple[str, ...], GpuDetector]]`
  - `UnstageableGpuDetectorError(ValueError)`

**This task alone fixes a wrong-answer bug** and is worth landing independently of the rest (spec §2).

> **Use the existing fake and the existing registration fixture.**
> `pipeline_requires_gpu` takes a **path**, so these tests must round-trip
> through JSON — and `ImagePipeline.from_json` resolves op classes **by bare
> name against the `phenotypic` namespace** (`_serializable_pipeline.py:627-678`).
> A test-local `class _FakeGpu` therefore raises `AttributeError: Class
> '_FakeGpu' not found in phenotypic namespace`.
>
> The established in-process pattern is the autouse monkeypatch fixture at
> `tests/unit/cli/test_staged_routing.py:21-23` — **not**
> `tests/_fakes/register_fake_gpu.py`, whose own docstring says it exists for
> the live SLURM dispatch test, where worker processes the fixture cannot reach
> need the preload. Copy the fixture:
>
> ```python
> @pytest.fixture(autouse=True)
> def _register_fake_gpu_detector(monkeypatch):
>     monkeypatch.setattr(phenotypic, "FakeGpuDetector", FakeGpuDetector,
>                         raising=False)
> ```

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/cli/test_gpu_detection_tree_wide.py
import pytest

from phenotypic import ImagePipeline
from phenotypic.detect import CompositeDetector, ManualPointDetector
from phenotypic._cli._cli_validation import (
    UnstageableGpuDetectorError,
    find_gpu_detectors,
    pipeline_requires_gpu,
)
import phenotypic
from tests._fakes.fake_gpu_detector import FakeGpuDetector


@pytest.fixture(autouse=True)
def _register_fake_gpu_detector(monkeypatch):
    """from_json resolves classes by bare name in the phenotypic namespace."""
    monkeypatch.setattr(phenotypic, "FakeGpuDetector", FakeGpuDetector,
                        raising=False)

CENTERS = [[10.0, 10.0], [10.0, 40.0]]


def _write(tmp_path, pipeline):
    path = tmp_path / "pipeline.json"
    path.write_text(pipeline.to_json(), encoding="utf-8")
    return path


def test_a_nested_gpu_detector_is_detected(tmp_path):
    pipe = ImagePipeline(
        ops={"CompositeDetector": CompositeDetector(
            ops=[FakeGpuDetector(),
                 ManualPointDetector(centers=CENTERS, shape="disk", width=11)],
            mode="overlap")}
    )
    assert pipeline_requires_gpu(_write(tmp_path, pipe)) is True


def test_the_detected_path_addresses_the_branch(tmp_path):
    pipe = ImagePipeline(
        ops={"CompositeDetector": CompositeDetector(
            ops=[FakeGpuDetector(),
                 ManualPointDetector(centers=CENTERS, shape="disk", width=11)],
            mode="overlap")}
    )
    hits = find_gpu_detectors(ImagePipeline.from_json(_write(tmp_path, pipe)))
    assert [p for p, _ in hits] == [("CompositeDetector", "ops[0]")]


def test_a_cpu_only_pipeline_is_still_false(tmp_path):
    pipe = ImagePipeline(
        ops={"CompositeDetector": CompositeDetector(
            ops=[ManualPointDetector(centers=CENTERS, shape="disk", width=11)],
            mode="union")}
    )
    assert pipeline_requires_gpu(_write(tmp_path, pipe)) is False


def test_two_gpu_detectors_anywhere_are_refused(tmp_path):
    pipe = ImagePipeline(
        ops={"CompositeDetector": CompositeDetector(
            ops=[FakeGpuDetector(), FakeGpuDetector()], mode="union")}
    )
    with pytest.raises(UnstageableGpuDetectorError, match="more than one"):
        find_gpu_detectors(ImagePipeline.from_json(_write(tmp_path, pipe)),
                           strict=True)


def test_a_gpu_detector_in_the_meas_slot_is_refused(tmp_path):
    """Stage 3 runs measurers on a CPU node, so a GPU op there cannot stage.

    Drives `pipeline_requires_gpu` -- the PRODUCTION entry point -- not
    `find_gpu_detectors(strict=True)`. A test that calls the helper directly
    passes even when the refusal is unreachable from production, which is
    exactly the bug this test exists to prevent.
    """
    from phenotypic.measure import MeasureSymZones

    offending = ImagePipeline(
        ops={"ManualPointDetector":
             ManualPointDetector(centers=CENTERS, shape="disk", width=11)},
        meas={"MeasureSymZones": MeasureSymZones(
            center_detector=FakeGpuDetector())},
    )
    with pytest.raises(UnstageableGpuDetectorError, match="meas"):
        pipeline_requires_gpu(_write(tmp_path, offending))


def test_a_meas_slot_gpu_detector_does_not_route_to_the_cpu_strategy(tmp_path):
    """The refusal must fire BEFORE strategy selection.

    Without this, pipeline_requires_gpu returns False, the run routes to
    LocalParallelStrategy (_cli_execution_strategies.py:1341) and the GPU op
    runs on CPU -- the exact wrong-answer bug this change exists to kill.
    """
    from phenotypic.measure import MeasureSymZones

    offending = ImagePipeline(
        ops={"ManualPointDetector":
             ManualPointDetector(centers=CENTERS, shape="disk", width=11)},
        meas={"MeasureSymZones": MeasureSymZones(
            center_detector=FakeGpuDetector())},
    )
    path = _write(tmp_path, offending)
    with pytest.raises(UnstageableGpuDetectorError):
        pipeline_requires_gpu(path)


```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/cli/test_gpu_detection_tree_wide.py -v`
Expected: FAIL — `ImportError: cannot import name 'UnstageableGpuDetectorError'`

- [ ] **Step 3: Write the implementation**

```python
# src/phenotypic/_cli/_cli_validation.py  (replacing lines 135-147)

class UnstageableGpuDetectorError(ValueError):
    """A GpuDetector sits somewhere the staged engine cannot drive it."""


#: Pipeline slots the staged engine runs on a CPU node in Stage 3. A
#: GpuDetector in any of them cannot be staged, so it is refused rather than
#: silently run on CPU.
_CPU_ONLY_SLOTS = ("meas", "post", "filters", "model")


def find_gpu_detectors(pipeline, *, strict: bool = False):
    """Every ``(path, detector)`` GpuDetector in *pipeline*, tree-wide.

    A CPU-only-slot detector raises **regardless of** ``strict``. That refusal
    has to fire on the production path (``pipeline_requires_gpu``), because the
    only other caller -- ``split_pipeline_at_gpu`` -- is reached only once
    ``pipeline_requires_gpu`` has already returned True. A refusal reachable
    only under ``strict=True`` is a refusal production never performs, while a
    unit test calling the helper directly still passes.

    Args:
        pipeline: The pipeline to scan.
        strict: When True, additionally raise for MORE THAN ONE detector. The
            GUI (``gui/run_console/_callbacks.py:253``) calls the non-strict
            path, where a multi-detector pipeline should report True rather
            than raise.
    """
    from phenotypic.abc_ import GpuDetector
    from phenotypic.sdk_._operation_tree import find_operations, walk_operations

    hits = find_operations(pipeline, lambda op: isinstance(op, GpuDetector))

    if strict and len(hits) > 1:
        paths = ", ".join("/".join(p) for p, _ in hits)
        # Name EVERY offending path, not just the count: the message's job is to
        # tell the user which branches to split. Deferred feature, not a limit of
        # the design -- see spec §13 for the intended N>1 execution model.
        # Keep the literal "more than one GpuDetector": it is the wording the
        # EXISTING suite already pins (`test_cli_pipeline_split.py:33` matches
        # it), and the plan's own test at Task 3 matches it too. The path list
        # follows it rather than replacing it.
        raise UnstageableGpuDetectorError(
            "staged execution does not support more than one GpuDetector "
            f"per pipeline; found {len(hits)} at: {paths}"
        )

    # Placement refusals, ALL of them, live here -- not in the splitter. This is
    # the function `pipeline_requires_gpu` calls, so a refusal placed anywhere
    # else fires only after the run has already been routed (prior review, B5).
    for path, _ in hits:
        validate_ancestor_contracts(pipeline, path)

    # Unconditional -- deliberately NOT gated on `strict`; see the docstring.
    for slot in _CPU_ONLY_SLOTS:
        accessor = getattr(pipeline, f"get_{slot}", None)
        if accessor is None:
            continue
        container = accessor()
        if container is None:
            continue
        # get_model() returns Optional[ModelFitter], NOT a dict -- calling
        # .items() on it raises AttributeError.
        entries = (
            container.items() if isinstance(container, dict)
            else [(slot, container)]
        )
        for name, op in entries:
            for sub_path, sub_op in walk_operations(op):
                if isinstance(sub_op, GpuDetector):
                    raise UnstageableGpuDetectorError(
                        f"GpuDetector at {slot}/{name}/"
                        f"{'/'.join(sub_path)} cannot be staged: Stage 3 "
                        f"runs the {slot!r} slot on a CPU node"
                    )
    return hits


def pipeline_requires_gpu(pipeline_path: Path) -> bool:
    """Check whether a pipeline JSON contains any GpuDetector, at any depth.

    Scans the whole operation tree, not just the top level: a ``GpuDetector``
    nested inside a ``CompositeDetector`` is still a GPU pipeline, and missing
    it means the run silently completes on CPU with different numbers.

    NOTE the two callers handle this differently, and neither was designed:
    `gui/run_console/_callbacks.py:246-255` wraps this in
    `except (OSError, ValueError, TypeError): return False`, and
    UnstageableGpuDetectorError IS a ValueError -- so the GUI silently reports
    "not a GPU pipeline" instead of surfacing the message. The CLI path
    (`_cli_execution_strategies.py:1341`) does not catch it, so there the user
    gets a raw traceback. Decide both: the GUI should surface the reason, and
    the CLI should print it rather than a traceback.

    Raises:
        UnstageableGpuDetectorError: a GpuDetector sits in a CPU-only slot.
            This MUST raise from here, not only from ``split_pipeline_at_gpu``:
            that function is reached only after this one returns True, so a
            refusal gated behind it can never fire in production -- the op
            would route to the non-staged strategy and silently run on CPU.
    """
    pipeline = ImagePipeline.from_json(pipeline_path)
    return bool(find_gpu_detectors(pipeline))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/cli/test_gpu_detection_tree_wide.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Verify the routing tests still pass**

Run: `uv run pytest tests/unit/cli/test_staged_routing.py -v`
Expected: PASS, unchanged count.

- [ ] **Step 6: Commit**

```bash
uv run ruff check --fix src/phenotypic/_cli/_cli_validation.py tests/unit/cli/test_gpu_detection_tree_wide.py
git add src/phenotypic/_cli/_cli_validation.py tests/unit/cli/test_gpu_detection_tree_wide.py
git commit -m "fix(cli): detect GpuDetectors nested inside container operations"
```

---

## Task 4: Container operations push a per-branch `pipeline_step`

**Files:**
- Modify: `src/phenotypic/_core/_provenance.py` (add `apply_child`)
- Modify: `src/phenotypic/detect/_composite_detector.py:126-140`
- Modify: `src/phenotypic/enhance/_composite_enhance.py`
- Modify: `src/phenotypic/detect/_filamentous_fungi_detector.py`
- Modify: `src/phenotypic/detect/_two_k_filamentous_detector.py`
- Test: `tests/unit/core/test_provenance_step_descent.py` (create)

**Interfaces:**
- Consumes: `pipeline_step` (`_provenance.py:527`).
- Produces: `apply_child(operation, image, *, segment, inplace=False, reset=None)` → the applied image.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/core/test_provenance_step_descent.py
from phenotypic import ImagePipeline
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import CompositeDetector, ManualPointDetector, OtsuDetector
from phenotypic.measure import MeasureSymZones

CENTERS = [[150.0, 200.0], [300.0, 400.0]]


def _step_paths(image):
    journal = image._metadata.provenance_journal
    return [
        (op["operation_class"].rsplit(".", 1)[-1], op.get("pipeline_step_path"))
        for app in journal.get("applications", [])
        for op in app.get("operations", [])
    ]


def test_composite_children_record_their_branch_index():
    image = load_synth_yeast_plate()
    ImagePipeline(
        ops={"CompositeDetector": CompositeDetector(
            ops=[OtsuDetector(),
                 ManualPointDetector(centers=CENTERS, shape="disk", width=41)],
            mode="union")}
    ).apply(image, inplace=True)

    recorded = dict(_step_paths(image))
    assert recorded["OtsuDetector"] == ["CompositeDetector", "ops[0]"]
    assert recorded["ManualPointDetector"] == ["CompositeDetector", "ops[1]"]


def test_nested_composites_produce_distinct_paths():
    """Before this change all five entries shared ['CompositeDetector']."""
    image = load_synth_yeast_plate()
    inner = CompositeDetector(
        ops=[OtsuDetector(),
             ManualPointDetector(centers=CENTERS, shape="disk", width=41)],
        mode="union")
    ImagePipeline(
        ops={"CompositeDetector": CompositeDetector(
            ops=[inner,
                 ManualPointDetector(centers=CENTERS, shape="disk", width=41)],
            mode="overlap")}
    ).apply(image, inplace=True)

    paths = [tuple(p) for _, p in _step_paths(image)]
    assert len(paths) == len(set(paths)), f"duplicate step paths: {paths}"


def test_a_measurement_probe_records_no_step_path():
    """A nested op run by a MEASUREMENT is a private probe (measure/CLAUDE.md).

    This exclusion is deliberate. Without this test it is indistinguishable
    from an oversight and will be 'completed' by a later reader.
    """
    image = load_synth_yeast_plate()
    pipe = ImagePipeline(
        ops={"OtsuDetector": OtsuDetector()},
        meas={"MeasureSymZones": MeasureSymZones(
            center_detector=ManualPointDetector(
                centers=CENTERS, shape="disk", width=41))},
    )
    pipe.apply(image, inplace=True)
    pipe.measure(image, apply_post=False)

    classes = [cls for cls, _ in _step_paths(image)]
    assert "ManualPointDetector" not in classes, (
        "a measurement's center_detector must not enter the plate journal"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/core/test_provenance_step_descent.py -v`
Expected: FAIL — first two tests assert `['CompositeDetector', 'ops[0]']` but get `['CompositeDetector']`. The third should already PASS (it pins existing, correct behaviour).

- [ ] **Step 3: Add the shared helper**

```python
# src/phenotypic/_core/_provenance.py  (append near pipeline_step)

def apply_child(
    operation: Any,
    image: "Image",
    *,
    segment: str,
    inplace: bool = False,
    reset: bool | None = None,
) -> "Image":
    """Apply a nested *operation* under its own ``pipeline_step`` segment.

    Container operations (``CompositeDetector``, ``CompositeEnhance``, ...)
    drive children by calling ``.apply()`` directly, which — unlike
    ``ImagePipeline._run_operations`` — pushes no step segment. Every child
    therefore inherited the container's own path, so N children of a composite
    were indistinguishable in the journal. Routing child applies through here
    fixes that, and makes a walker path and a ``pipeline_step_path`` the same
    value (spec §5.3).

    ``segment`` must be a non-empty string, normally ``f"ops[{i}]"``;
    ``validate_provenance_journal`` rejects a path containing anything else.

    NOT for measurements: a nested operation run by a ``MeasureFeatures`` is a
    private probe whose steps deliberately stay out of the plate's provenance
    (see ``measure/CLAUDE.md``). Those keep calling ``.apply()`` directly.
    """
    from phenotypic._core._image_pipeline import ImagePipeline

    kwargs: dict[str, Any] = {"inplace": inplace}
    if isinstance(operation, ImagePipeline):
        kwargs["reset"] = False if reset is None else reset

    with pipeline_step(segment):
        return operation.apply(image, **kwargs)
```

- [ ] **Step 4: Adopt it in `CompositeDetector`**

```python
# src/phenotypic/detect/_composite_detector.py, inside _operate
from phenotypic._core._provenance import apply_child

objmaps = []
for index, detector in enumerate(self.ops):
    if detector is None:
        continue
    detected_image = apply_child(
        detector, image, segment=f"ops[{index}]", inplace=False
    )
    objmaps.append(detected_image.objmap[:].astype(bool))
```

Note this also removes the `isinstance(detector, ImagePipeline)` branch — `apply_child` handles the `reset=False` difference.

- [ ] **Step 5: Adopt it in the other three containers**

The pattern is **not** identical everywhere — check each call's existing
`inplace` before changing it:

| File | Call site | `inplace` | Segment |
|---|---|---|---|
| `_composite_enhance.py` | list of `ops` | `False` | `f"ops[{i}]"` |
| `_filamentous_fungi_detector.py:395,398` | `inoculum_detector` — **not** a list; the class has no `ops` | `False` | `"inoculum_detector"` |
| `_two_k_filamentous_detector.py:164` | `self.branch_base.apply(enhanced, inplace=True)` | **`True`** | `"branch_base"` |

`apply_child` takes `inplace` as a keyword and defaults it to `False`, so the
`branch_base` call must pass `inplace=True` explicitly. Flipping it to `False`
silently discards the enhancement — the op would run and its result be thrown
away, with nothing failing to say so.

Two more things about `TwoKFilamentousDetector`, both deliberate and both worth
stating so the partial descent is not read as an oversight:

- Only `branch_base` is listed. `center_detector` (`:149,151`) and
  `background_subtractor` (`:154`) are **not** descended, because the class is
  refused for staging anyway (Task 5 step 3c) and descending them buys nothing.
- `apply_child` injects `reset=False` for an `ImagePipeline` child, while the
  current `:164` call passes only `inplace=True` and `ImagePipeline.apply`'s
  `reset` default is `None`, not `False` (`_image_pipeline_core.py:947-949`).
  `branch_base` **defaults to an `ImagePipeline`** (`:103`), so this is a live
  behaviour change on the default path. Either preserve `reset=None` or record
  the change with a test.

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/unit/core/test_provenance_step_descent.py -v`
Expected: PASS (3 tests)

- [ ] **Step 7: Check the blast radius on existing provenance tests**

```bash
uv run pytest tests/unit/core -k provenance -q 2>&1 | tail -20
uv run pytest tests/unit/detect tests/unit/enhance -q 2>&1 | tail -10
```

Any failure asserting a literal `['CompositeDetector']` step path is an **expected** update, not a regression — fix the expectation. Any other failure is a real regression: stop and investigate with systematic-debugging.

- [ ] **Step 8: Commit**

```bash
uv run ruff check --fix src/phenotypic/_core/_provenance.py src/phenotypic/detect src/phenotypic/enhance tests/unit/core/test_provenance_step_descent.py
git add -A
git commit -m "feat(provenance): descend step paths through container operations"
```

---

## Task 5: Path-shaped `StagePlan` with a Stage-2 branch prefix

**Files:**
- Modify: `src/phenotypic/_cli/_cli_pipeline_split.py` (whole file)
- Test: `tests/unit/cli/test_pipeline_split_nested.py` (create)

**Interfaces:**
- Consumes: `find_gpu_detectors` (Task 3), `get_at_path` (Task 1).
- Produces: `StagePlan(pre_pipeline, gpu_path: tuple[str, ...], gpu_detector, stage2_prefix: list, post_pipeline)`; `split_pipeline_at_gpu(pipeline) -> StagePlan`.

**`gpu_key` is removed.** Task 13 updates the three SLURM call sites.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/cli/test_pipeline_split_nested.py
import numpy as np
import pytest

from phenotypic import ImagePipeline
from phenotypic.detect import CompositeDetector, ManualPointDetector
from phenotypic.enhance import BlurGauss, ContrastStretching, SubtractGaussian
from phenotypic._cli._cli_pipeline_split import split_pipeline_at_gpu
# These tests build pipelines IN MEMORY (no from_json), so a plain import is
# enough -- no namespace registration needed. Same as test_cli_pipeline_split.py:14.
from tests._fakes.fake_gpu_detector import FakeGpuDetector

CENTERS = [[10.0, 10.0], [10.0, 40.0]]


def _manual():
    return ManualPointDetector(centers=CENTERS, shape="disk", width=11)


def test_split_cuts_at_the_top_level_ancestor():
    pipe = ImagePipeline(ops={
        "BlurGauss": BlurGauss(sigma=2.0),
        "SubtractGaussian": SubtractGaussian(sigma=50.0),
        "CompositeDetector": CompositeDetector(ops=[FakeGpuDetector(), _manual()],
                                               mode="overlap"),
        "ContrastStretching": ContrastStretching(input_layer="detect_mat"),
    })
    plan = split_pipeline_at_gpu(pipe)

    assert plan.gpu_path == ("CompositeDetector", "ops[0]")
    assert list(plan.pre_pipeline.get_ops()) == ["BlurGauss", "SubtractGaussian"]
    # the ANCESTOR heads the post pipeline -- it has not run yet
    assert list(plan.post_pipeline.get_ops()) == [
        "CompositeDetector", "ContrastStretching"]


def test_a_bare_leaf_needs_no_stage2_prefix():
    pipe = ImagePipeline(ops={
        "CompositeDetector": CompositeDetector(ops=[FakeGpuDetector(), _manual()],
                                               mode="overlap")})
    assert split_pipeline_at_gpu(pipe).stage2_prefix == []


def test_a_branch_pipeline_contributes_its_preceding_ops():
    branch = ImagePipeline(ops={
        "ContrastStretching": ContrastStretching(input_layer="detect_mat"),
        "FakeGpu": FakeGpuDetector()})
    pipe = ImagePipeline(ops={
        "CompositeDetector": CompositeDetector(ops=[branch, _manual()],
                                               mode="overlap")})
    plan = split_pipeline_at_gpu(pipe)

    assert plan.gpu_path == ("CompositeDetector", "ops[0]", "FakeGpu")
    assert [type(op).__name__ for op in plan.stage2_prefix] == ["ContrastStretching"]


def test_composite_siblings_contribute_nothing_to_the_prefix():
    """Composite ops are PARALLEL branches applied to the same input."""
    pipe = ImagePipeline(ops={
        "CompositeDetector": CompositeDetector(ops=[_manual(), FakeGpuDetector()],
                                               mode="overlap")})
    plan = split_pipeline_at_gpu(pipe)
    assert plan.gpu_path == ("CompositeDetector", "ops[1]")
    assert plan.stage2_prefix == []


def test_a_top_level_detector_needs_no_prefix():
    """The case the spike got wrong (spec §4.2).

    The spike's branch_prefix lacks a root guard on its final block, so for a
    top-level detector it returns every preceding top-level op -- ops Stage 1
    has ALREADY applied and written to the store, which Stage 2 would then
    re-run on top of themselves. Nothing pinned this because all three spike
    shapes nest.
    """
    pipe = ImagePipeline(ops={
        "BlurGauss": BlurGauss(sigma=2.0),
        "SubtractGaussian": SubtractGaussian(sigma=50.0),
        "FakeGpuDetector": FakeGpuDetector(),
    })
    plan = split_pipeline_at_gpu(pipe)

    assert plan.gpu_path == ("FakeGpuDetector",)
    assert plan.stage2_prefix == []


def test_a_top_level_detector_stays_in_the_post_pipeline():
    """Its slot must survive so Stage 3's stub can land in it.

    Dropping it (the pre-change behaviour) would make Stage 3 re-run the REAL
    detector on a CPU node.
    """
    pipe = ImagePipeline(ops={
        "BlurGauss": BlurGauss(sigma=2.0),
        "FakeGpuDetector": FakeGpuDetector(),
        "ContrastStretching": ContrastStretching(input_layer="detect_mat"),
    })
    plan = split_pipeline_at_gpu(pipe)

    assert list(plan.pre_pipeline.get_ops()) == ["BlurGauss"]
    assert list(plan.post_pipeline.get_ops()) == [
        "FakeGpuDetector", "ContrastStretching"]


def test_a_plot_referencing_the_ancestor_is_now_allowed():
    """The ancestor runs in Stage 3, so a plot may reference it.

    The old guard refused `ref.key == gpu_key`; under the new cut that key is
    the ANCESTOR and lives in post_pipeline. Loosening the guard without a test
    would leave the behaviour asserted only in a code comment.
    """
    pipe = ImagePipeline(ops={
        "CompositeDetector": CompositeDetector(ops=[FakeGpuDetector(), _manual()],
                                               mode="overlap")})
    # Plot capability is a MIXIN ON THE OP CLASS, not something retrofitted onto
    # an instance -- see tests/unit/cli/test_cli_pipeline_split.py:53-54
    # (`class _PreGpuPlot(BlurGauss, PlotImage)`). `normalize_plot_bindings`
    # raises on a non-plot-capable entry, so `_with_plot_on` needs a
    # CompositeDetector+PlotImage subclass, not a plain CompositeDetector.
    plan = split_pipeline_at_gpu(_with_plot_on(pipe, "CompositeDetector"))
    assert "CompositeDetector" in plan.post_pipeline.get_ops()


def test_a_plot_referencing_a_pre_gpu_op_is_still_refused():
    pipe = ImagePipeline(ops={
        "BlurGauss": BlurGauss(sigma=2.0),
        "CompositeDetector": CompositeDetector(ops=[FakeGpuDetector(), _manual()],
                                               mode="overlap")})
    with pytest.raises(ValueError, match="pre-GPU"):
        split_pipeline_at_gpu(_with_plot_on(pipe, "BlurGauss"))


def test_no_gpu_detector_still_raises():
    pipe = ImagePipeline(ops={"CompositeDetector":
                              CompositeDetector(ops=[_manual()], mode="union")})
    with pytest.raises(ValueError, match="no GpuDetector"):
        split_pipeline_at_gpu(pipe)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/cli/test_pipeline_split_nested.py -v`
Expected: FAIL — `AttributeError: 'StagePlan' object has no attribute 'gpu_path'`

- [ ] **Step 3: Write the implementation**

```python
# src/phenotypic/_cli/_cli_pipeline_split.py

@dataclass
class StagePlan:
    """Result of splitting a pipeline at its (single) GpuDetector."""

    pre_pipeline: ImagePipeline       # ops before the detector's ancestor (Stage 1)
    gpu_path: tuple[str, ...]         # tree path to the detector (Stage 2 + provenance)
    gpu_detector: GpuDetector         # the detector itself (Stage 2)
    stage2_prefix: list               # ops Stage 2 must apply before inferring
    post_pipeline: ImagePipeline      # ancestor onward + meas/post/filters/model/qc


#: What each container hands its children. Keyed by class, and NOT a
#: declaration on the operations themselves: for these types the semantics is
#: definitional rather than incidental. A CompositeDetector whose branches
#: chained would not be a composite -- it would be an ImagePipeline, which
#: already exists for that. So this table restates a type contract; it does not
#: cache an observation about today's `_operate`.
#:
#: Every entry is backed by a behavioural probe test (Task 5 step 3a). Coverage
#: is asserted on the TABLE ITSELF (step 3b), not by enumerating the tree: the
#: set is closed by rule, so a new container needs no entry and no decision --
#: it is refused by default, which is the correct answer for it. There is no
#: `_UNSUPPORTED_CONTAINERS` map; an earlier draft had one and the narrowing
#: removed the need for it.
#: LIVES IN `_cli_validation.py`, not in the splitter. `_cli_pipeline_split`
#: already imports from `_cli_validation`, so the dependency cannot run the
#: other way -- and placement refusals belong with the other placement refusals,
#: where `pipeline_requires_gpu` (the production entry point) reaches them.
_CHILD_CONTRACT: dict[type, str] = {
    CompositeDetector: "parallel",
    CompositeEnhance: "parallel",
}


def _child_contract(container) -> str:
    """``"parallel"`` or ``"sequence"``; raise for anything else.

    ONLY composition primitives may carry a staged GpuDetector. A domain
    detector is refused even when its current code would classify cleanly --
    `FilamentousFungiDetector` feeds `inoculum_detector` the container's own
    image today (:395,398) and so reads as "parallel", but that is incidental to an
    algorithm that also runs an inline ContrastStretching (:413) and a
    destructive _subtract_background. Nothing about being a fungus detector
    constrains it to keep doing that, so the table's safety argument -- "this
    restates a type contract, it does not cache an observation" -- would not
    hold uniformly if it were admitted.
    """
    if isinstance(container, ImagePipeline):
        return "sequence"
    cls = type(container)
    if cls in _CHILD_CONTRACT:
        return _CHILD_CONTRACT[cls]
    raise UnstageableGpuDetectorError(
        f"a GpuDetector cannot be nested inside {cls.__name__}: only "
        "composition primitives (ImagePipeline, CompositeDetector, "
        "CompositeEnhance) may carry one. Lift the detector into a "
        "CompositeDetector branch, or into the top-level pipeline."
    )


def validate_ancestor_contracts(pipeline: ImagePipeline, path: tuple[str, ...]) -> None:
    """Every container on the ancestor chain must declare a child contract.

    Call this from ``find_gpu_detectors`` -- NOT from the prefix builder. The
    refusal is a property of *placement*, not of prefix computation, and
    ``pipeline_requires_gpu`` is the production entry point that must carry
    placement refusals (prior review, Blocker 5). A refusal reachable only from
    `_branch_prefix` fires only after the run has already been routed.
    """
    from phenotypic.sdk_._operation_tree import get_at_path

    for depth in range(len(path)):          # path[:0] is the root pipeline
        _child_contract(get_at_path(pipeline, path[:depth]))


def _branch_prefix(pipeline: ImagePipeline, path: tuple[str, ...]) -> list:
    """Ops between the Stage-1 store and the GPU op's model input.

    Dispatches on ``_child_contract``, never on ``isinstance``. A ``"sequence"``
    container contributes the ops preceding the step taken from it; a
    ``"parallel"`` container contributes nothing, because its children are
    parallel branches and none runs "before" another.

    An earlier draft tested ``isinstance(container, ImagePipeline)`` and
    ``continue``d past everything else, which meant ``_child_contract`` was
    never called and the "composition primitives only" rule existed nowhere in
    the code. Dispatch on the contract string.
    """
    from phenotypic.sdk_._operation_tree import get_at_path

    prefix: list = []
    for depth in range(len(path)):
        container = get_at_path(pipeline, path[:depth])     # path[:0] -> the root
        if _child_contract(container) == "parallel":
            continue
        if container is pipeline:
            # The root's own preceding ops are Stage 1's; they already ran and
            # are already in the store. This is the guard the spike lacked --
            # without it a TOP-LEVEL detector's prefix is every op Stage 1 just
            # applied, re-run on top of itself.
            continue
        keys = list(container.get_ops())
        for key in keys[: keys.index(path[depth])]:
            prefix.append(container.get_ops()[key])
    return prefix


def split_pipeline_at_gpu(pipeline: ImagePipeline) -> StagePlan:
    """Partition *pipeline* around its single GpuDetector, at any depth.

    Raises:
        ValueError: zero GpuDetectors in the pipeline.
        UnstageableGpuDetectorError: more than one, or one in a CPU-only slot.
    """
    from ._cli_validation import find_gpu_detectors

    hits = find_gpu_detectors(pipeline, strict=True)
    if not hits:
        raise ValueError(
            "no GpuDetector in pipeline; staged execution requires exactly one"
        )
    gpu_path, gpu_detector = hits[0]

    ops = pipeline.get_ops()
    keys = list(ops)
    cut = keys.index(gpu_path[0])
    pre_ops = {k: ops[k] for k in keys[:cut]}
    post_ops = {k: ops[k] for k in keys[cut:]}   # ANCESTOR INCLUDED

    for binding in pipeline.get_plots():
        ref = binding.ref
        if ref is None or ref.slot != "ops":
            continue
        if ref.key in pre_ops:
            raise ValueError(
                f"plot {binding.id!r} references pre-GPU operation {ref.key!r}; "
                "staged plotting supports only post-GPU operations, measurers, "
                "aggregate slots, and inline plots"
            )

    return StagePlan(
        pre_pipeline=ImagePipeline(ops=pre_ops, nrows=pipeline.nrows,
                                   ncols=pipeline.ncols),
        gpu_path=gpu_path,
        gpu_detector=gpu_detector,
        stage2_prefix=_branch_prefix(pipeline, gpu_path),
        post_pipeline=ImagePipeline(
            ops=post_ops, meas=pipeline.get_meas(), post=pipeline.get_post(),
            filters=pipeline.get_filters(), model=pipeline.get_model(),
            qc=pipeline.get_qc(), plots=pipeline.get_plots(),
            nrows=pipeline.nrows, ncols=pipeline.ncols),
    )
```

**Note on the plot guard.** The ancestor key now lives in `post_ops`, so a plot
referencing a *container* ancestor is legal — it runs in Stage 3. But the
reduction to `ref.key in pre_ops` is **wrong when the detector is itself
top-level**: there `gpu_path[0]` IS the detector, and the dropped disjunct was
what refused a plot bound to it. Stage 3 never runs the real detector, and the
substituted `ReplayDetector` is not plot-capable, so `_resolve_plot_bindings`
(`_image_pipeline_core.py:328-344`) would raise — or worse, `PlotCoordinator`
(untouched by Task 8) would emit against the *original* detector while the image
came from the substituted copy.

```python
if ref.key in pre_ops or (len(gpu_path) == 1 and ref.key == gpu_path[0]):
    raise ValueError(...)
```

Narrow in practice — it needs a `GpuDetector` that also subclasses `PlotImage` —
but pin it with a test for the top-level case, not only the nested one.

- [ ] **Step 3a: Verify each `"parallel"` contract behaviourally, not by assertion**

Create `tests/unit/detect/test_container_child_contracts.py`. For **each** class
in `_CHILD_CONTRACT`, put two recording probes in its children and assert the
second did not observe the first's output. A table entry that is merely stated
can be wrong and still pass everything; a probe cannot.

```python
def test_composite_branches_each_receive_the_composites_own_input():
    seen = []

    class _Probe(ObjectDetector):
        tag: str

        def _operate(self, image):
            seen.append((self.tag, int(image.objmap[:].max())))
            image.objmask[:] = image.gray[:] > image.gray[:].mean()
            return image

    CompositeDetector(ops=[_Probe(tag="a"), _Probe(tag="b")],
                      mode="union").apply(load_synth_yeast_plate())

    # Sequential branches would have "b" observing "a"'s objmap.
    assert seen == [("a", 0), ("b", 0)]
```

Write the equivalent for `CompositeEnhance` (probe `detect_mat` rather than
`objmap`). **Not** `FilamentousFungiDetector` — it is not in `_CHILD_CONTRACT`
and Task 5 step 3b/3c explicitly refuses it; a probe test for a refused class is
a leftover from the pre-narrowing draft.

- [ ] **Step 3b: Enforce coverage of the table**

```python
def test_the_contract_table_holds_only_composition_primitives():
    """The table is closed by RULE, not by survey.

    Anything not listed is refused, so this asserts the list itself rather than
    enumerating the tree. A new domain detector needs no entry and no decision:
    it is refused by default, which is the correct answer for it.
    """
    assert set(_CHILD_CONTRACT) == {CompositeDetector, CompositeEnhance}


def test_a_domain_detector_is_refused_even_though_it_would_classify():
    """FilamentousFungiDetector passes its child the container's own image
    today, so it would read as "parallel". It is still refused: that behaviour is
    incidental to its algorithm, not part of what the class IS."""
    pipe = ImagePipeline(ops={"Fungi": FilamentousFungiDetector(
        inoculum_detector=FakeGpuDetector())})
    with pytest.raises(UnstageableGpuDetectorError,
                       match="only composition primitives"):
        split_pipeline_at_gpu(pipe)
```

- [ ] **Step 3c: Refusal test**

```python
def test_a_gpu_detector_inside_a_domain_detector_is_refused():
    pipe = ImagePipeline(ops={"TwoK": TwoKFilamentousDetector(
        branch_base=FakeGpuDetector())})
    with pytest.raises(UnstageableGpuDetectorError,
                       match="only composition primitives"):
        split_pipeline_at_gpu(pipe)


def test_the_refusal_names_the_supported_containers():
    """The message's job is to tell the user what to do instead."""
    pipe = ImagePipeline(ops={"TwoK": TwoKFilamentousDetector(
        branch_base=FakeGpuDetector())})
    with pytest.raises(UnstageableGpuDetectorError) as exc:
        split_pipeline_at_gpu(pipe)
    assert "CompositeDetector" in str(exc.value)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/cli/test_pipeline_split_nested.py -v
uv run pytest tests/unit/detect/test_container_child_contracts.py -v
```
Expected: PASS

- [ ] **Step 5: Update the existing split test — it WILL fail, and that is correct**

`tests/unit/cli/test_cli_pipeline_split.py:24` asserts:

```python
assert list(plan.post_pipeline.get_ops().keys()) == ["SmallObjectRemover"]
```

i.e. the top-level detector is **excluded** from `post_pipeline`. Under the new
uniform rule its slot is **included**, so the stub can be substituted into it.
Update the expectation to `["FakeGpuDetector", "SmallObjectRemover"]`.

Also check `test_rejects_more_than_one_gpu_detector` (`:31`) still passes —
`split_pipeline_at_gpu` now raises `UnstageableGpuDetectorError`, which is a
`ValueError` subclass, so a `pytest.raises(ValueError)` still matches but a
message assertion may not.

Run: `uv run pytest tests/unit/cli/test_cli_pipeline_split.py -v`

- [ ] **Step 6: Commit**

```bash
uv run ruff check --fix src/phenotypic/_cli/_cli_pipeline_split.py tests/unit/cli/test_pipeline_split_nested.py tests/unit/cli/test_cli_pipeline_split.py
git add src/phenotypic/_cli/_cli_pipeline_split.py tests/unit/cli/test_pipeline_split_nested.py tests/unit/cli/test_cli_pipeline_split.py
git commit -m "feat(cli): address the staged GPU detector by tree path"
```

---

## Task 6: `ReplayDetector`

**Files:**
- Create: `src/phenotypic/_cli/_cli_replay_detector.py`
- Test: `tests/unit/cli/test_replay_detector.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `ReplayDetector(detector=..., result=..., detector_duration_seconds=0.0)` — an `ObjectDetector` whose `_operate` writes a pre-recorded array, and which reports the **wrapped detector's** identity to provenance.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/cli/test_replay_detector.py
import numpy as np

from phenotypic.abc_ import GpuDetector
from phenotypic.data import load_synth_yeast_plate
from phenotypic._cli._cli_replay_detector import ReplayDetector


from tests._fakes.fake_gpu_detector import FakeGpuDetector

# Built in memory, never round-tripped through JSON, so no registration fixture
# is needed here. `threshold` is a real field on the shared fake and is what the
# parameter-delegation assertion below keys on.


def test_replay_writes_the_recorded_array():
    image = load_synth_yeast_plate()
    recorded = np.zeros(image.gray[:].shape, dtype=np.uint16)
    recorded[20:60, 20:60] = 1
    recorded[120:160, 120:160] = 2

    detector = FakeGpuDetector(drop_frame_background=False, split_disconnected_labels=False)
    ReplayDetector(detector=detector, result=recorded).apply(image, inplace=True)

    assert image.num_objects == 2


def test_replay_applies_the_detectors_post_inference_cleanup():
    """_write_object_output owns drop_frame_background / relabel; the stub must
    delegate to it rather than assigning objmap itself."""
    image = load_synth_yeast_plate()
    recorded = np.zeros(image.gray[:].shape, dtype=np.uint16)
    recorded[:] = 9                      # a background-spanning label
    recorded[20:60, 20:60] = 1

    detector = FakeGpuDetector(drop_frame_background=True, split_disconnected_labels=True)
    ReplayDetector(detector=detector, result=recorded).apply(image, inplace=True)

    assert image.num_objects == 1, "frame background was not dropped"


def test_provenance_identity_is_the_wrapped_detector():
    """The journal must name Sam2, not ReplayDetector, or a staged run's
    provenance stops matching a single-pass run's."""
    detector = FakeGpuDetector(threshold=0.37)
    stub = ReplayDetector(detector=detector, result=np.zeros((4, 4), np.uint16))

    assert stub.provenance_operation_class().endswith("FakeGpuDetector")
    assert stub.provenance_operation_name() == "FakeGpuDetector"
    assert stub.provenance_parameters()["threshold"] == 0.37


def test_the_stub_carries_the_stage2_duration():
    """The stub's own wall time is the MERGE only; GPU cost lives in the token.

    tests/integration/cli/test_staged_store_stages.py:128 asserts the recorded
    duration is >= the token's detector_duration_seconds.
    """
    stub = ReplayDetector(detector=FakeGpuDetector(), result=np.zeros((4, 4), np.uint16),
                          detector_duration_seconds=12.5)
    assert stub.provenance_duration_offset() == 12.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/cli/test_replay_detector.py -v`
Expected: FAIL — `ModuleNotFoundError: phenotypic._cli._cli_replay_detector`

- [ ] **Step 3: Write the implementation**

```python
# src/phenotypic/_cli/_cli_replay_detector.py
"""Stage-3 stand-in for a GpuDetector whose inference already happened."""

from __future__ import annotations

from typing import TYPE_CHECKING

from phenotypic.abc_ import ObjectDetector
from phenotypic.sdk_.typing_ import NdArrayField, OperationField

if TYPE_CHECKING:
    from phenotypic._core._image import Image


class ReplayDetector(ObjectDetector):
    """Write a PRE-RECORDED Stage-2 result in place of running the model.

    Stage 2 ran the real detector on a GPU node and retained its raw output.
    Stage 3 substitutes this stub at the detector's tree path so the enclosing
    operation -- a ``CompositeDetector``, say -- runs exactly as it would in a
    single-pass run, merging this branch's mask with its CPU siblings'.

    The write delegates to the real detector's ``_write_object_output``, which
    owns ``drop_frame_background`` and ``split_disconnected_labels``. Assigning
    ``objmap`` directly here would skip both and silently bridge every colony
    the background label touches.
    """

    detector: OperationField
    result: NdArrayField
    detector_duration_seconds: float = 0.0

    def provenance_operation_class(self) -> str:
        """Report the WRAPPED detector's class to the journal."""
        cls = type(self.detector)
        return f"{cls.__module__}.{cls.__qualname__}"

    def provenance_parameters(self) -> dict:
        """Report the WRAPPED detector's parameters to the journal."""
        return self.detector.model_dump(mode="json")

    def _operate(self, image: "Image") -> "Image":
        self.detector._write_object_output(image, self.result)
        return image
```

- [ ] **Step 4: Wire the provenance hooks**

In `_provenance.py`, where an operation's `operation_class` and `parameters` are derived for a journal record, prefer the operation's own `provenance_operation_class()` / `provenance_parameters()` when present:

`append_operation_provenance` (`_provenance.py:872-912`) derives **four**
fields from the operation. Hook **all four** — the two the first draft of this
plan missed are each pinned by an existing integration test
(`tests/integration/cli/test_staged_store_stages.py:115` and `:128`).

```python
# _provenance.py, inside append_operation_provenance
operation_name = (
    operation.provenance_operation_name()
    if hasattr(operation, "provenance_operation_name")
    else type(operation).__name__
)
operation_class = (
    operation.provenance_operation_class()
    if hasattr(operation, "provenance_operation_class")
    else f"{type(operation).__module__}.{type(operation).__qualname__}"
)
source_parameters = (
    operation.provenance_parameters()
    if hasattr(operation, "provenance_parameters")
    else operation.model_dump(mode="json")
)
# KEEP the JSON round-trip around whichever source supplied the value -- it is
# what guarantees the payload is JSON-native before validate_provenance_journal
# sees it.
parameters = json.loads(json.dumps(source_parameters, ensure_ascii=False))
duration = float(duration_seconds)
if hasattr(operation, "provenance_duration_offset"):
    # The stub's own wall time covers the MERGE only; GPU inference happened in
    # Stage 2 and its cost lives in the token. Current behaviour records
    # token["detector_duration_seconds"] + merge (_cli_staged_workers.py:493-497)
    # and test_staged_store_stages.py:128 asserts it.
    duration += float(operation.provenance_duration_offset())
```

and on the stub:

```python
    def provenance_operation_name(self) -> str:
        return type(self.detector).__name__

    def provenance_duration_offset(self) -> float:
        return self.detector_duration_seconds
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/cli/test_replay_detector.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
uv run ruff check --fix src/phenotypic/_cli/_cli_replay_detector.py src/phenotypic/_core/_provenance.py tests/unit/cli/test_replay_detector.py
git add -A
git commit -m "feat(cli): add ReplayDetector for Stage-3 replay of a staged detector"
```

---

## Task 6a: Key the Stage-2 signal by detector slot

**Files:**
- Modify: `src/phenotypic/_cli/_cli_stage2_token.py` (`stage2_token_path:54`, `stage2_raw_path:145`, and every reader / writer / predicate)
- Test: `tests/unit/cli/test_stage2_slot_keying.py` (create)

**Interfaces:**
- Consumes: `StagePlan.gpu_path` (Task 5).
- Produces:
  - `detector_slot(gpu_path: Sequence[str]) -> str`
  - `stage2_raw_path(output_dir, dataset, image_stem, slot)`
  - `stage2_token_path(output_dir, dataset, image_stem, slot)`
  - `relocate_legacy_stage2_signal(output_dir, dataset, image_stem, slot) -> bool`

> **Why this task exists now, at N = 1.** The signal is currently keyed by image
> alone — one array per image, structurally — and that is the only thing making
> `N > 1` a change to the on-disk layout rather than an additive feature
> (spec §4.4, §13). At `N == 1` this is one extra directory level and no
> behavioural difference. Deferred, it becomes a layout migration on a signal a
> 33,923-image run depends on. Land it here.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/cli/test_stage2_slot_keying.py
import numpy as np

from phenotypic._cli._cli_stage2_token import (
    detector_slot,
    load_stage2_raw,
    relocate_legacy_stage2_signal,
    stage2_raw_path,
    write_stage2_raw,
)


def test_slot_is_readable_and_collision_proof():
    a = detector_slot(("CompositeDetector", "ops[0]"))
    b = detector_slot(("CompositeDetector", "ops[1]"))

    assert a != b
    assert "CompositeDetector" in a and "ops-0" in a   # debuggable by eye
    assert a.replace("-", "").replace("_", "").isalnum()  # filesystem-safe


def test_paths_that_sanitise_alike_still_differ():
    """`ops[0]` and `ops-0` both sanitise to `ops-0`; the hash must separate them."""
    assert detector_slot(("A", "ops[0]")) != detector_slot(("A", "ops-0"))


def test_two_slots_round_trip_independently(tmp_path):
    left = detector_slot(("CompositeDetector", "ops[0]"))
    right = detector_slot(("CompositeDetector", "ops[1]"))
    a = np.full((4, 4), 1, dtype=np.uint16)
    b = np.full((4, 4), 2, dtype=np.uint16)

    write_stage2_raw(tmp_path, "ds", "img", a, slot=left)
    write_stage2_raw(tmp_path, "ds", "img", b, slot=right)

    assert np.array_equal(load_stage2_raw(tmp_path, "ds", "img", slot=left), a)
    assert np.array_equal(load_stage2_raw(tmp_path, "ds", "img", slot=right), b)


def test_a_legacy_signal_is_relocated_rather_than_recomputed(tmp_path):
    """Avoids re-running a GPU sweep for images an interrupted run finished."""
    slot = detector_slot(("CompositeDetector", "ops[0]"))
    legacy = tmp_path / ".phenotypic" / "progress" / "stage2_raw" / "ds"
    legacy.mkdir(parents=True)
    np.save(legacy / "img.npy", np.full((4, 4), 7, dtype=np.uint16))

    assert relocate_legacy_stage2_signal(tmp_path, "ds", "img", slot) is True
    assert stage2_raw_path(tmp_path, "ds", "img", slot).is_file()
    assert not (legacy / "img.npy").exists()
    assert load_stage2_raw(tmp_path, "ds", "img", slot=slot).max() == 7


def test_relocation_never_overwrites_a_current_signal(tmp_path):
    slot = detector_slot(("CompositeDetector", "ops[0]"))
    write_stage2_raw(tmp_path, "ds", "img", np.full((4, 4), 3, np.uint16), slot=slot)
    legacy = tmp_path / ".phenotypic" / "progress" / "stage2_raw" / "ds"
    legacy.mkdir(parents=True, exist_ok=True)
    np.save(legacy / "img.npy", np.full((4, 4), 9, dtype=np.uint16))

    assert relocate_legacy_stage2_signal(tmp_path, "ds", "img", slot) is False
    assert load_stage2_raw(tmp_path, "ds", "img", slot=slot).max() == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/cli/test_stage2_slot_keying.py -v`
Expected: FAIL — `ImportError: cannot import name 'detector_slot'`

- [ ] **Step 3: Implement the slot id**

```python
# src/phenotypic/_cli/_cli_stage2_token.py
import hashlib
import re
from typing import Sequence

_UNSAFE = re.compile(r"[^A-Za-z0-9]+")


def detector_slot(gpu_path: Sequence[str]) -> str:
    """A filesystem-safe, collision-proof id for one staged detector.

    Readable half: each path segment with runs of non-alphanumerics collapsed to
    ``-``, joined by ``__``. Safe half: 8 hex characters of the EXACT path, so
    two paths that sanitise alike (``ops[0]`` and ``ops-0``) never collide.

        ("CompositeDetector", "ops[0]") -> "CompositeDetector__ops-0__3f9a1c02"
    """
    exact = "/".join(gpu_path)
    readable = "__".join(_UNSAFE.sub("-", part).strip("-") for part in gpu_path)
    digest = hashlib.sha256(exact.encode("utf-8")).hexdigest()[:8]
    return f"{readable}__{digest}"
```

- [ ] **Step 4: Thread `slot` through every path helper and its callers**

`stage2_raw_path`, `stage2_token_path`, `write_stage2_raw`, `load_stage2_raw`,
`write_stage2_token`, `read_stage2_token`, and the `stage2_result_replayable`
predicate all take `slot`.

**Do NOT make it required keyword-only everywhere.** Three of the six calling
modules have no `plan` in scope to derive a slot from, and forcing one through
them is most of this task's cost:

| Module | Sites | Problem |
|---|---|---|
| `_cli_staged_resume.py` | `:256, :309, :420, :423, :478, :479` | `classify_staged_image` is a pure classifier — signature `(image, dataset, output_dir, input_root, process_only_layer, markers_required, expected_work_id)`, no plan, no slot. Threading one means changing `build_staged_resume_plan` and every caller above it |
| `_cli_staged_controller.py` | `:84` | the recovery controller's already-done skip; no plan in scope |
| `_cli_migrate_state.py` | `:184` | builds the token path **by hand** from `progress_dir / DIR_STAGE2_DONE / dataset / f"{stem}.json"`. With a slot level inserted, `_stage2_entry` silently returns `None` for every modern token and `--mode migrate` stops recording interrupted Stage-2 state |

Instead: **the strategies derive the slot from the plan and pass it down one
level**, and the resume/controller layer takes it as an ordinary parameter. That
closes the structural hole just as well — no caller can reach a shared path
without being handed a slot — at a fraction of the churn.

**Test files touching these helpers positionally**, all of which break:
`tests/unit/cli/conftest.py:287`, `tests/integration/cli/conftest.py:209,210,235`,
`test_staged_controller.py:455,847,851,869,876,877`,
`test_provenance_fencing.py:102,103`,
`test_lifecycle_publication_races.py:453,471,493,496,521,538,560,563`,
`test_staged_store_stages.py:71,76,77,252,262,278,284,352,360`,
`test_staged_gpu_local.py` (~15 sites), `test_staged_resume.py` (~14 sites),
`test_staged_resume_equivalence.py:1443`, `test_schema_gate.py:623`,
`test_migrate_state.py:110`.

**Where the slot comes from, per stage** — say it once, here: Stage 2 and Stage 3
both have `plan` in scope (`detector_slot(plan.gpu_path)`); the process-mode
export does too, via `_export_objmap_layer`'s `plan` parameter
(`_cli_staged_strategy.py:398`).

The rationale's *"at `N == 1` this is one extra directory level and no
behavioural difference"* is true of the **on-disk layout**, not of the code
change. The code change is the widest in Plan A.

New layout:

```
<output>/.phenotypic/progress/stage2_raw/<dataset>/<slot>/<stem>.npy
<output>/.phenotypic/progress/stage2_done/<dataset>/<slot>/<stem>.json
```

Callers to update: `_cli_staged_workers.py` (Stage 2 write, Stage 3 read and
consume), `_cli_staged_strategy.py` (the objmap export and the replayable
probe), `_cli_staged_slurm_worker.py`.

- [ ] **Step 5: Implement the legacy relocation**

```python
def relocate_legacy_stage2_signal(
    output_dir: Path, dataset: str, image_stem: str, slot: str
) -> bool:
    """Move a pre-slot-keying signal into its slot directory. Returns moved?

    A staged run interrupted before slot keying and resumed after it would
    otherwise not find its signals and recompute Stage 2 -- correct, but paid in
    GPU time on the scarcest resource in the cluster.

    Moves BOTH halves. The Stage-2 *signal* is two files -- the retained raw and
    the consumable token (`_cli_stage2_token.py:1-10`) -- and
    `stage2_result_replayable` (`:186-210`) requires both. Moving only the raw
    leaves the slot-keyed token absent, so `stage2_result_replayable` stays
    False, Stage 2 recomputes anyway, and the tree is half-migrated: the exact
    cost this helper exists to avoid, plus a mess.

    Moves the **raw first, then the token**, mirroring the write order
    (`_cli_stage2_token.py:172-174`) so an interrupted relocation can never
    leave a slot-keyed token with no slot-keyed raw.

    Never overwrites a current signal: if the slot path already exists, the
    legacy file is stale and the current one wins.
    """
```

**Call site and guard.** `StagePlan` has no `slots` attribute in Plan A — that
is Spec §13's future `N > 1` shape, and an earlier draft gated on it, which is
unimplementable. Under Plan A a plan has exactly one slot by construction, so a
count check would be vacuous anyway.

Instead the caller passes the single slot explicitly and the helper refuses to
run when the caller declares more than one. Call it from
`clear_downstream_artifacts_for_stage1`'s sibling on the resume path
(`_cli_staged_resume.py`), where the slot arrives as the ordinary parameter M3
introduces — **not** from `classify_staged_image`, which is a pure classifier
with no slot.

Both tests must assert **both** files moved, not just the raw.

- [ ] **Step 6: Run the tests**

```bash
uv run pytest tests/unit/cli/test_stage2_slot_keying.py -v
uv run pytest tests/unit/cli/test_staged_resume.py tests/unit/cli/test_staged_resume_equivalence.py -v
uv run pytest tests/integration/cli/test_staged_gpu_local.py -v
```

Expected: PASS. The resume suites are the ones that touch these paths most —
a failure there means a caller was missed in step 4.

- [ ] **Step 7: Commit**

```bash
uv run ruff check --fix src/phenotypic/_cli/_cli_stage2_token.py src/phenotypic/_cli/_cli_staged_workers.py src/phenotypic/_cli/_cli_staged_strategy.py src/phenotypic/_cli/_cli_staged_slurm_worker.py tests/unit/cli/test_stage2_slot_keying.py
git add -A
git commit -m "refactor(cli): key the Stage-2 signal by detector slot"
```

---

## Task 7: Stage 2 applies the branch prefix

**Files:**
- Modify: `src/phenotypic/_cli/_cli_staged_workers.py` `stage2_detect_core` (~`:369-412`)
- Test: `tests/unit/cli/test_staged_stage2_prefix.py`

**Interfaces:**
- Consumes: `StagePlan.stage2_prefix` (Task 5).
- Produces: `stage2_detect_core(..., stage2_prefix: list | None = None)`.

- [ ] **Step 1: Write the failing test**

Assert that when `stage2_prefix` is non-empty, (a) the detector sees the prefixed array, and (b) the **store on disk is byte-unchanged** — Stage 2 must never write.

```python
# tests/unit/cli/test_staged_stage2_prefix.py
# (build a Stage-1 store via the existing fixtures in tests/unit/cli/, then:)

def test_prefix_is_applied_in_memory_and_never_written(staged_store_fixture):
    before = _store_digest(staged_store_fixture.store_path)

    stage2_detect_core(
        detector=staged_store_fixture.detector,
        output_dir=staged_store_fixture.output_dir,
        dataset_name="ds",
        image_stem="img",
        stage2_prefix=[ContrastStretching(input_layer="detect_mat")],
    )

    assert _store_digest(staged_store_fixture.store_path) == before, (
        "Stage 2 wrote into the store"
    )
    raw = load_stage2_raw(staged_store_fixture.output_dir, "ds", "img",
                          slot=staged_store_fixture.slot)
    assert raw.shape == staged_store_fixture.expected_shape
```

Reuse the existing store-construction helpers in `tests/unit/cli/` rather than inventing new ones; `test_staged_resume.py` shows the established pattern.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/cli/test_staged_stage2_prefix.py -v`
Expected: FAIL — `TypeError: stage2_detect_core() got an unexpected keyword argument 'stage2_prefix'`

If you instead see `ValueError: cannot start a new provenance application before
the last ends`, you have wired the keyword but used a bare `image.copy()`. See
step 3 — that error is the finding this task was revised for.

- [ ] **Step 3: Implement**

```python
# in stage2_detect_core, replacing the input-layer read at :384
image = image_cls.load_zarr(store)  # read-only use; never re-promoted here

if stage2_prefix:
    # The GPU op sits behind CPU ops inside its own branch. Run them on a
    # PROVENANCE-DETACHED copy.
    #
    # A bare image.copy() does NOT work: copy() carries the journal, Stage 1
    # left the trailing application "staged", stage2_detect_core runs at
    # _application_owner_depth == 0, and _append_application raises unless the
    # last application is "complete"/"failed" (_provenance.py:361-363). The
    # first prefix op would raise
    #     ValueError: cannot start a new provenance application before the last ends
    #
    # Detaching (rather than continuing_provenance_application) is right here
    # because the prefix's records must reach NOTHING: this copy is discarded,
    # and Stage 3 re-runs these same ops and records them for real. Pattern
    # copied from measure/_canonical_zone_measure.py:279-295.
    from copy import deepcopy

    probe = image.copy()
    probe_journal = deepcopy(image._metadata.provenance_journal)
    for application in probe_journal.get("applications", []):
        if application.get("status") not in {"complete", "failed"}:
            application["status"] = "complete"
    probe_journal["status"] = "complete"
    probe._metadata.provenance_journal = probe_journal

    for operation in stage2_prefix:
        operation.apply(probe, inplace=True)
    image = probe

array = getattr(image, detector.input_layer)[:]
```

**Also add to the Task 7 test:** assert the prefix's records land **nowhere** —
neither in the store (already asserted) nor in any journal Stage 3 later reads.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/cli/test_staged_stage2_prefix.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
uv run ruff check --fix src/phenotypic/_cli/_cli_staged_workers.py tests/unit/cli/test_staged_stage2_prefix.py
git add -A
git commit -m "feat(cli): apply the Stage-2 branch prefix in memory"
```

---

## Task 8: Stage 3 substitutes the stub instead of writing directly

**Files:**
- Modify: `src/phenotypic/_cli/_cli_staged_workers.py` `stage3_merge_measure_core:487-505`
- Test: covered by Task 10's equivalence test; add no new file here.

**Interfaces:**
- Consumes: `substitute_at_path` (Task 1), `ReplayDetector` (Task 6), `StagePlan.gpu_path` (Task 5).
- Produces: no new signature.

- [ ] **Step 1: Replace the explicit write + provenance append**

The current code writes the objmap and appends the journal entry by hand, *outside* the pipeline apply. Both now happen inside the enclosing operation's `_operate`:

```python
# BEFORE (delete):
#   plan.gpu_detector._write_object_output(image, result)
#   append_operation_provenance(image, plan.gpu_detector, ...)
#   _checkpoint_successful_operation(...)
#   with continuing_provenance_application(image), provenance_success_sink(...):
#       plan.post_pipeline.apply(image, inplace=True)

slot = detector_slot(plan.gpu_path)          # Task 6a: every helper takes it
result = load_stage2_raw(output_dir, dataset_name, image_stem, slot=slot)
token = read_stage2_token(output_dir, dataset_name, image_stem, slot=slot)
_check_active(active_check)

stub = ReplayDetector(
    detector=plan.gpu_detector,
    result=result,
    detector_duration_seconds=float(token.get("detector_duration_seconds", 0.0)),
)
replay_pipeline = substitute_at_path(plan.post_pipeline, plan.gpu_path, stub)

with continuing_provenance_application(image), provenance_success_sink(
    lambda updated: _write_provenance_checkpoint_fenced(
        store, updated, active_check, commit_guard=commit_guard)
):
    replay_pipeline.apply(image, inplace=True)
measurements = replay_pipeline.measure(image, apply_post=False)
```

**Keep the FLOW-21 comment** about replaying from the retained raw rather than the store's objmap — it still applies.

- [ ] **Step 2: Verify the existing staged tests still pass**

```bash
uv run pytest tests/unit/cli/test_staged_resume.py tests/unit/cli/test_staged_resume_equivalence.py -v
uv run pytest tests/integration/cli/test_staged_store_stages.py -v
```

Expected: PASS. `test_staged_store_stages.py` is the one that pins the journal —
it asserts `operation_name` at `:115` and the Stage-2-inclusive `duration_seconds`
at `:128`, and it is what catches a half-wired provenance hook. A failure in
either means the substitution changed ordering or provenance — investigate before
continuing.

- [ ] **Step 3: Commit**

```bash
uv run ruff check --fix src/phenotypic/_cli/_cli_staged_workers.py
git add src/phenotypic/_cli/_cli_staged_workers.py
git commit -m "feat(cli): replay the staged detector via path substitution"
```

---

## Task 9: Pin the addressing invariant

**Files:**
- Test: `tests/unit/cli/test_gpu_path_is_the_step_path.py` (create)

**Interfaces:** consumes Tasks 4 and 5. Produces no source change — this task is a test only.

**Why:** spec §5.3. Without this assertion, `gpu_path` and `pipeline_step_path` are two schemes kept aligned by hand, and they will drift.

- [ ] **Step 1: Write the test**

```python
# tests/unit/cli/test_gpu_path_is_the_step_path.py
import numpy as np
import pytest

from phenotypic import ImagePipeline
from phenotypic.abc_ import GpuDetector
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import CompositeDetector, ManualPointDetector
from phenotypic.enhance import ContrastStretching
from phenotypic._cli._cli_pipeline_split import split_pipeline_at_gpu

CENTERS = [[150.0, 200.0], [300.0, 400.0]]


from tests._fakes.fake_gpu_detector import FakeGpuDetector


def _manual():
    return ManualPointDetector(centers=CENTERS, shape="disk", width=41)


def _shapes():
    yield "leaf", CompositeDetector(ops=[FakeGpuDetector(), _manual()], mode="overlap")
    branch = ImagePipeline(ops={
        "ContrastStretching": ContrastStretching(input_layer="detect_mat"),
        "FakeGpu": FakeGpuDetector()})
    yield "branch", CompositeDetector(ops=[branch, _manual()], mode="overlap")
    yield "nested", CompositeDetector(
        ops=[CompositeDetector(ops=[FakeGpuDetector(), _manual()], mode="union"),
             _manual()], mode="overlap")


@pytest.mark.parametrize("name,detector", list(_shapes()),
                         ids=[n for n, _ in _shapes()])
def test_gpu_path_equals_the_recorded_step_path(name, detector):
    pipe = ImagePipeline(ops={"CompositeDetector": detector})
    plan = split_pipeline_at_gpu(pipe)

    image = load_synth_yeast_plate()
    pipe.apply(image, inplace=True)

    recorded = [
        op["pipeline_step_path"]
        for app in image._metadata.provenance_journal.get("applications", [])
        for op in app.get("operations", [])
        if op["operation_class"].endswith("FakeGpuDetector")
    ]
    assert recorded == [list(plan.gpu_path)], (
        f"{name}: walker path {plan.gpu_path} != journal path {recorded}"
    )
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/unit/cli/test_gpu_path_is_the_step_path.py -v`
Expected: PASS (3 parametrisations). A failure means Task 4's segments and Task 1's path segments disagree — fix the *segment* construction, not the test.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/cli/test_gpu_path_is_the_step_path.py
git commit -m "test(cli): pin gpu_path == pipeline_step_path"
```

---

## Task 10: Staged/single-pass equivalence, with a mutation control

**Files:**
- Test: `tests/unit/cli/test_staged_nested_equivalence.py` (create)

**Interfaces:** consumes Tasks 5–8. No source change.

**This is the gate for the whole change.** The spike at `docs/superpowers/specs/2026-09-15-nested-gpu-staging/spike_nested_gpu.py` is the reference; port it, do not re-derive it.

- [ ] **Step 1: Write the equivalence test**

Port `spike_nested_gpu.py`'s three shapes. For each: run the pipeline normally; then run Stage 1 / Stage 2 / Stage 3 through the real `split_pipeline_at_gpu` + `ReplayDetector` + `substitute_at_path`; assert the objmap and the measurement frame are identical.

- [ ] **Step 2: Write the mutation control**

```python
def test_a_corrupted_replay_is_detected():
    """Without this the equivalence assertion is vacuous.

    NOTE: the perturbation below leaves the OBJECT COUNT unchanged, so a test
    that compares num_objects passes on broken code. Compare the objmap.
    """
    reference, staged_clean = _run_both(shape_leaf)
    staged_dirty = _run_staged(shape_leaf, corrupt=lambda raw: np.roll(raw, 7, axis=0))

    assert np.array_equal(reference, staged_clean)
    assert not np.array_equal(reference, staged_dirty)
```

- [ ] **Step 2a: Drive at least one shape through the PRODUCTION entry points**

Steps 1-2 port the spike, which reimplements the stage sequence in-test. That
checks the *algorithm* and would miss a defect in Task 7's provenance-detached
probe, Task 8's substitution wiring, the token/duration plumbing, or Task 13's
forwarding — none of which the port touches.

Add one shape driven end-to-end through `stage2_detect_core` +
`stage3_merge_measure_core` against a real staged store. The fixtures exist:
`tests/unit/cli/conftest.py:287` (`write_stage2_raw`) and
`tests/integration/cli/conftest.py:200-240` already build one.

- [ ] **Step 3: Run**

Run: `uv run pytest tests/unit/cli/test_staged_nested_equivalence.py -v`
Expected: PASS — all three shapes equivalent, corrupted replay detected.

**Shape names:** the spike defines `shape_a` / `shape_b` / `shape_c`
(`spike_nested_gpu.py:171,175,184`). Use those, not `shape_leaf`.

- [ ] **Step 4: Add the owner-depth-0 test**

**The stated mechanism in an earlier draft was wrong.** `_application_owner_depth`
is a `ContextVar` with `default=0` (`_provenance.py:80-81`), so setting it to 0
outside an enclosing `provenance_application` is a **no-op**. The
`measure/CLAUDE.md` rationale comes from the *measurement* case, where a
surrounding `with provenance_application(image, kind="programmatic"):` is what
raises the depth in the first place.

The test is still worth having — but for the real reason: Stage 3's depth-0
exposure comes from the **store's trailing `"staged"` application**, not from the
context var. Either exercise it against a real staged store, or reproduce the
measurement pattern in full (wrap in `provenance_application`, *then* force the
depth back to 0):

```python
from phenotypic._core._provenance import _application_owner_depth

def test_stage3_at_cli_owner_depth():
    token = _application_owner_depth.set(0)
    try:
        ...  # run stage3_merge_measure_core against a staged store
    finally:
        _application_owner_depth.reset(token)
```

- [ ] **Step 5: Commit**

```bash
git add tests/unit/cli/test_staged_nested_equivalence.py
git commit -m "test(cli): staged/single-pass equivalence for nested GPU detectors"
```

---

## Task 11: Process mode runs the post-detector op chain

**Files:**
- Modify: `src/phenotypic/_cli/_cli_staged_strategy.py` `_export_objmap_layer:397-480`
- Test: `tests/integration/cli/test_process_objmap_semantics.py` (create)
- Modify: `tests/integration/cli/test_staged_gpu_local.py:1039` (docstring only)

**Interfaces:** consumes Tasks 1, 5, 6. No new signature.

**Behaviour change (spec §8):** the export now means *the objmap your pipeline produces*, for top-level and nested detectors alike.

- [ ] **Step 1: Write the failing test**

The existing test at `:1039` uses a pipeline with **no post-detector ops**, so old and new semantics coincide and it cannot catch this. The new test must use a pipeline whose post-detector ops measurably change the objmap.

```python
# tests/integration/cli/test_process_objmap_semantics.py

def test_export_applies_post_detector_ops(tmp_path):
    """--layer objmap exports the PIPELINE's objmap, not the detector's raw output."""
    # pipeline: FakeGpu (emits a large blob AND a 9-px speck)
    #           -> SmallObjectRemover(min_size=100)
    pipe = ImagePipeline(ops={
        "FakeGpu": FakeGpuTwoBlobs(),
        "SmallObjectRemover": SmallObjectRemover(min_size=100),
    })
    ... run StagedGpuStrategy with process_only_layer="objmap" ...

    exported = cv2.imread(str(out_path), cv2.IMREAD_UNCHANGED)
    assert len(np.unique(exported)) - 1 == 1, (
        "the speck survived -- post-detector ops were not applied"
    )


def test_export_applies_the_composite_merge_for_a_nested_detector(tmp_path):
    """For a nested detector the raw array is one BRANCH, not the objmap."""
    ...


def test_the_store_is_byte_unchanged_by_the_export(tmp_path):
    """FLOW-16/FLOW-30/FLOW-6: this path must not write into the store."""
    before = _store_digest(store_path)
    ... run the export ...
    assert _store_digest(store_path) == before
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/integration/cli/test_process_objmap_semantics.py -v`
Expected: FAIL — the speck survives; only 2 labels present.

- [ ] **Step 3: Implement**

```python
# in _export_objmap_layer, replacing the direct _write_object_output call
from phenotypic._core._provenance import continuing_provenance_application
from phenotypic.sdk_._operation_tree import substitute_at_path

from ._cli_replay_detector import ReplayDetector

image = image_cls.load_zarr(store)
raw = load_stage2_raw(output_dir, ds.name, source_image_stem(img),
                      slot=detector_slot(plan.gpu_path))

stub = ReplayDetector(detector=plan.gpu_detector, result=raw)
residual = substitute_at_path(plan.post_pipeline, plan.gpu_path, stub)

# Stage 1 left the application "staged", which is NOT terminal, so an apply at
# CLI owner-depth 0 would raise "cannot start a new provenance application
# before the last ends". continuing_provenance_application accepts "staged"
# (_provenance.py:458) and raises _application_owner_depth, so the apply JOINS
# the open application instead of appending a new one.
#
# Do NOT try to fix this with set_provenance_status(image, "in_progress"):
# "in_progress" is ALSO outside _append_application's terminal set
# {"complete", "failed"} (_provenance.py:362), so it raises the very error it
# looks like it prevents. An earlier draft of this plan did exactly that.
#
# NOTE the absent provenance_success_sink. Stage 3 installs one; this path must
# NOT, because the sink is what writes to the store, and a store write after the
# success marker invalidates the descriptor the marker just recorded (ledger
# FLOW-16/FLOW-30/FLOW-6). The absence is deliberate -- do not "restore" it.
# KEEP the wrapper the replaced call had (_cli_staged_strategy.py:464-471).
# This does strictly MORE work than the single array write it replaces -- the
# whole post-detector op chain -- so it is strictly more likely to raise, and an
# unwrapped exception changes how _record_local_terminal_failure classifies the
# image.
try:
    with continuing_provenance_application(image):
        residual.apply(image, inplace=True)   # ops only; never .measure()
except MemoryError:
    raise
except Exception as exc:
    raise PerImageScientificError(STAGE_MEASURE, exc) from exc

write_process_only_layer(image, "objmap", out_path)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/integration/cli/test_process_objmap_semantics.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Update the stale docstring at `test_staged_gpu_local.py:1039`**

Its assertions still hold (it guards FLOW-16), but "replays Stage 2's raw array" is now "replays Stage 2's raw array **and applies the post-detector ops**". Update the prose; change no assertion.

- [ ] **Step 6: Commit**

```bash
uv run ruff check --fix src/phenotypic/_cli/_cli_staged_strategy.py tests/integration/cli/test_process_objmap_semantics.py
git add -A
git commit -m "feat(cli)!: process objmap export applies post-detector ops"
```

---

## Task 12: Invalidate continuation across the semantics change

**Files:**
- Modify: `src/phenotypic/_cli/_cli_failure_tracker.py:191-236`
- Test: `tests/unit/cli/test_work_id_semantics_revision.py`

**Interfaces:** produces `PROCESS_LAYER_SEMANTICS_REVISION: int`.

**This is the highest-severity risk in the spec (§8.3, §12).** Without it, a process run interrupted before the upgrade and resumed after reuses old-semantics PNGs and publishes a tree that *looks* complete while mixing two meanings.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/cli/test_work_id_semantics_revision.py
from phenotypic._cli import _cli_failure_tracker as tracker


def test_bumping_the_revision_changes_the_digest():
    kwargs = dict(image_type="Image", nrows=8, ncols=12, bit_depth=None,
                  detect_mode="gray", process_only_layer="objmap", ext=".png",
                  process_format="tiff", include_dataset_column=True,
                  overlay_alpha=0.5, save_overlays=False)
    before = tracker.processing_configuration_digest_from_values(**kwargs)

    original = tracker.PROCESS_LAYER_SEMANTICS_REVISION
    try:
        tracker.PROCESS_LAYER_SEMANTICS_REVISION = original + 1
        after = tracker.processing_configuration_digest_from_values(**kwargs)
    finally:
        tracker.PROCESS_LAYER_SEMANTICS_REVISION = original

    assert before != after, (
        "the digest ignores the semantics revision, so a process run resumed "
        "across an output-semantics change would reuse stale outputs"
    )


def test_a_full_run_digest_is_UNCHANGED_by_the_bump():
    """The change is scoped to process mode; full/measure must not cold-start.

    Putting the revision in the base payload would invalidate every in-flight
    full and measure continuation on the cluster for no correctness gain -- see
    the precedent comment at _cli_failure_tracker.py:218-223.
    """
    kwargs = dict(image_type="Image", nrows=8, ncols=12, bit_depth=None,
                  detect_mode="gray", process_only_layer=None, ext=".png",
                  process_format="tiff", include_dataset_column=True,
                  overlay_alpha=0.5, save_overlays=False)
    before = tracker.processing_configuration_digest_from_values(**kwargs)

    original = tracker.PROCESS_LAYER_SEMANTICS_REVISION
    try:
        tracker.PROCESS_LAYER_SEMANTICS_REVISION = original + 1
        after = tracker.processing_configuration_digest_from_values(**kwargs)
    finally:
        tracker.PROCESS_LAYER_SEMANTICS_REVISION = original

    assert before == after, "a full-run digest must not depend on process-layer semantics"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/cli/test_work_id_semantics_revision.py -v`
Expected: FAIL — `AttributeError: module has no attribute 'PROCESS_LAYER_SEMANTICS_REVISION'`

- [ ] **Step 3: Implement**

```python
#: Bumped when the semantics of a PROCESS-MODE EXPORTED LAYER change, so a
#: process run resumed across the upgrade re-derives its images instead of
#: reusing outputs that mean something different.
#:
#: Scope is deliberately narrow: this governs `--mode process --layer <L>`
#: outputs ONLY. It is NOT the package version (that would invalidate
#: continuation on every patch release, breaking legitimate resume), and it is
#: NOT a general "output semantics" dial -- read broadly, that is what argues
#: for the base payload, which is wrong. See the placement note below.
#:
#: 1 -> 2: `--layer objmap` now applies the post-detector op chain, so the
#:         export is the pipeline's objmap rather than the detector's raw
#:         output (spec 2026-09-15-nested-gpu-staging §8).
PROCESS_LAYER_SEMANTICS_REVISION = 2


def processing_configuration_digest_from_values(...) -> str:
    payload: dict[str, object] = {
        "image_type": image_type,
        ...
    }
    if process_only_layer is not None:
        payload.update(
            {
                "process_only_layer": process_only_layer,
                "ext": ext,
                "process_format": process_format,
                # Beside `process_format` and NOT in the base payload, for the
                # reason the comment above it already gives: folding a
                # process-only concern into the base changes every existing
                # run's digest and cold-starts every continuation in flight --
                # including full and measure runs this change does not touch.
                # Keyed by layer so a `gray` export is not invalidated by an
                # `objmap` semantics change.
                "layer_semantics": (
                    f"{process_only_layer}:{PROCESS_LAYER_SEMANTICS_REVISION}"
                ),
            }
        )
```

- [ ] **Step 4: Run and check the blast radius**

```bash
uv run pytest tests/unit/cli/test_work_id_semantics_revision.py -v
uv run pytest tests/unit/cli/test_cli_process_only.py tests/unit/cli/test_process_format_option.py -v
```

Any test asserting a **literal** digest string needs its expectation regenerated — that is the intended effect. A test asserting digest *relationships* must still pass.

- [ ] **Step 5: Commit**

```bash
uv run ruff check --fix src/phenotypic/_cli/_cli_failure_tracker.py tests/unit/cli/test_work_id_semantics_revision.py
git add -A
git commit -m "fix(cli): invalidate continuation across an output-semantics change"
```

---

## Task 13: Pass `stage2_prefix` at both Stage-2 call sites

**Files:**
- Modify: `src/phenotypic/_cli/_cli_staged_slurm_worker.py:310` (SLURM Stage 2)
- Modify: `src/phenotypic/_cli/_cli_staged_strategy.py:246` (local Stage 2)
- Test: `tests/unit/cli/test_staged_slurm_scripts.py`, `test_staged_controller.py` (existing)

> **This task was re-aimed after plan review.** An earlier draft targeted
> `plan.gpu_key` at `_cli_staged_slurm_worker.py:182,296,448`. Those three lines
> call `split_pipeline_at_gpu` and never mention `gpu_key`; the *only* `gpu_key`
> uses in the tree are inside `_cli_pipeline_split.py` itself (`:22,34-57`),
> which Task 5 rewrites wholesale. Verified with
> `grep -rn "gpu_key" --include=*.py src/ tests/`.
>
> The real gap is the opposite one: `stage2_detect_core` grew a `stage2_prefix`
> parameter in Task 7 and **nobody passes it**. Both call sites pass positionally
> today, so the new parameter silently defaults to `None` and shape B — the whole
> reason the prefix exists — stays broken with no error.

**Interfaces:**
- Consumes: `StagePlan.stage2_prefix` (Task 5), `stage2_detect_core(..., stage2_prefix=)` (Task 7).
- Produces: no new signature.

- [ ] **Step 1: Pass the prefix at the SLURM Stage-2 site**

`_cli_staged_slurm_worker.py:310` currently calls `stage2_detect_core` with
positional arguments. Add the keyword:

The real call there passes `active_check` and `commit_guard`. **Keep both** —
`active_check` is the SLURM epoch fence and `commit_guard` gates durable writes;
dropping them disables both on the GPU stage and nothing fails to say so. Note
the variable names are the SLURM worker's (`item.dataset` / `item.stem` /
`image_type`), not the local strategy's:

```python
stage2_detect_core(
    plan.gpu_detector,
    output_dir,
    item.dataset,
    item.stem,
    image_type,
    active_check=check,
    commit_guard=commit_guard,
    slot=detector_slot(plan.gpu_path),
    stage2_prefix=plan.stage2_prefix,
)
```

- [ ] **Step 2: Pass the prefix at the local Stage-2 site**

`_cli_staged_strategy.py:246`, same change:

```python
stage2_detect_core(
    plan.gpu_detector,
    output_dir,
    ds.name,
    source_image_stem(img),
    cfg.image_type,
    stage2_prefix=plan.stage2_prefix,
)
```

- [ ] **Step 3: Guard against the silent-default failure mode**

A test that only checks "shape B works locally" would pass while the SLURM path
stays broken, because the default is `None` rather than an error. Assert both
call sites forward it — e.g. monkeypatch `stage2_detect_core` and assert the
received `stage2_prefix` is non-empty for a shape-B plan, once per call site.

- [ ] **Step 4: Verify**

```bash
uv run pytest tests/unit/cli/test_staged_slurm_scripts.py tests/unit/cli/test_staged_controller.py -v
uv run python -c "import phenotypic._cli._cli_staged_slurm_worker"
```

- [ ] **Step 5: Commit**

```bash
uv run ruff check --fix src/phenotypic/_cli/_cli_staged_slurm_worker.py src/phenotypic/_cli/_cli_staged_strategy.py
git add src/phenotypic/_cli/_cli_staged_slurm_worker.py src/phenotypic/_cli/_cli_staged_strategy.py
git commit -m "fix(cli): forward stage2_prefix at both Stage-2 call sites"
```

---

## Task 14: Documentation

**Files:**
- Modify: root `CLAUDE.md` (the `--mode process` bullet and the staged-GPU bullet)
- Modify: `src/phenotypic/_cli/CLAUDE.md`
- Verify: `docs/source/contrib_guide/gpu_detectors.md` (written on this branch; check against the implementation as landed)
- Modify: `docs/source/how_to/` pages describing the objmap export

- [ ] **Step 1: Fix the now-false sentence in root `CLAUDE.md`**

"`--mode process --layer objmap` exports objmaps after Stages 1–2" is wrong. Replace with wording that says the export applies the post-detector op chain and therefore yields the pipeline's objmap.

- [ ] **Step 2: Land the contributor guide**

`docs/source/contrib_guide/gpu_detectors.md` is written and registered in the
toctree on this branch. Re-read it against the implementation as landed and fix
any drift — in particular the container-contract section, whose refusal message
and class list must match `_CHILD_CONTRACT` exactly. (There is no
`_UNSUPPORTED_CONTAINERS`; the narrowing removed it -- see Task 5 step 3b.)

Build the docs to confirm the page renders and the toctree resolves:

```bash
uv run sphinx-build -b html docs/source docs/_build/html -q
```

- [ ] **Step 3: Document nested support in `src/phenotypic/_cli/CLAUDE.md`**

Record: GPU detectors are found tree-wide; the split cuts at the top-level ancestor; the Stage-2 branch prefix runs twice and must be deterministic; unstageable placements are refused.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md src/phenotypic/_cli/CLAUDE.md docs/source
git commit -m "docs: nested GpuDetector staging and the new objmap export semantics"
```

---

## Task 15: Full regression

- [ ] **Step 1: Type check**

```bash
uv run mypy src/phenotypic
```

- [ ] **Step 2: Run the full suite as a Slurm job**

Use the **`run-phenotypic-test`** skill and the committed batch script at `docs/superpowers/plans/2026-08-18-ome-zarr-image-store/run_unit_suite.sbatch`. **Never `-n auto`.** Always `QT_QPA_PLATFORM=offscreen`. This is ~65 minutes — run it **once**, here, not between tasks.

- [ ] **Step 3: Compare against the recorded baseline**

The captured baseline is 11,106 tests / 81 failed, all outside `sdk_`/`_cli`/`gui`. Any **new** failure inside `sdk_`, `_cli`, `gui`, `detect`, `enhance`, or `core` is attributable to this change. Run each failing test in isolation before attributing it — most pass alone.

- [ ] **Step 4: Report**

State the counts measured, not counts expected. If anything regressed, stop and report rather than proceeding.

---

## Execution: cluster assignment

Derived from the per-task `Files`/`Interfaces` blocks by the
`execute-plan-orchestration` procedure. Shapes: **K**eystone (novel
interdependent core logic), **S**weep (broad and mechanical), **Se**am (one
risky wiring point), **L**eaf (small and independent).

| # | Tasks | Shape | Model | Files touched | Gate |
|---|---|---|---|---|---|
| C1 | 1 | K | Opus, high | `sdk_/_operation_tree.py` | local (seconds) |
| C2 | 3 | Se | Opus, high | `_cli/_cli_validation.py` | Slurm, 1 task |
| C3 | 2 | S | Sonnet, med | `tune/_search_space/_infer.py` | Slurm, 2 shards |
| C4 | 4 | S + K head | Opus, high | `_core/_provenance.py`, 4 container classes | Slurm, 2 shards |
| C5 | 5, 6 | K | Opus, high | `_cli_pipeline_split.py`, `_cli_replay_detector.py` | Slurm, 2 shards |
| C6 | 6a | S | Opus, high | `_cli_stage2_token.py` + 5 callers + ~15 test files | Slurm, 4 shards |
| C7 | 7, 8, 13 | Se | Opus, high | `_cli_staged_workers.py`, `_cli_staged_slurm_worker.py`, `_cli_staged_strategy.py:246` | Slurm, 4 shards |
| C8 | 9, 10 | L (tests) | Opus, high | two new test files | Slurm, 1 task |
| C9 | 11, 12 | K + L | Opus, high | `_cli_staged_strategy.py`, `_cli_failure_tracker.py` | Slurm, 2 shards |
| C10 | 14 | L | Sonnet, med | docs only | none (docs) |
| C11 | 15 | gate | — | — | **Slurm array, 24 shards** |

**Parallel fan-out:** C2, C3 and C4 all depend only on C1 and touch provably
disjoint files, so they run concurrently. Everything else is sequential.

**Why C6 stays on the frontier model** despite being a Sweep: it re-keys
`stage2_result_replayable`, the predicate the whole continuation contract rests
on, across six modules with no compiler to catch a missed call site.

**Why C7 absorbs Task 13:** see the phase-table correction above.

**Reviews.** A fresh `implementation-test-reviewer` runs over every cluster's
combined diff before the next cluster is dispatched — not a lighter reviewer,
and never a weaker model than the implementer.

---

## Revision history

**Revised 2026-09-15 after an independent plan review**
(`docs/superpowers/reports/2026-09-15-nested-gpu-staging/plan-review.md`).
Twenty findings; all applied. The five that would have stopped execution:

1. **Stage-2 branch prefix raised on a `"staged"` journal.** A bare
   `image.copy()` carries the journal; at owner-depth 0 the first prefix op hits
   `_append_application`'s terminal-status guard. Now uses the
   provenance-detached probe pattern from `measure/_canonical_zone_measure.py`.
2. **Process-mode provenance mitigation did not work.** `set_provenance_status(
   "in_progress")` is *also* non-terminal, so it raised the very error it was
   written to prevent. Now `continuing_provenance_application`, no success sink.
   The spec was corrected too (§8.2) — it had shipped an untested mitigation.
3. **The provenance hook was half a hook.** It covered `operation_class` and
   `parameters` but not `operation_name` or `duration_seconds`, both pinned by
   `tests/integration/cli/test_staged_store_stages.py:115,128`.
4. **Tests could not deserialise their own fixture.** `from_json` resolves classes
   by bare name in the `phenotypic` namespace; the plan used a module-local class.
   Now uses `tests/_fakes/fake_gpu_detector.py` with the monkeypatch fixture
   pattern from `test_staged_routing.py:21`.
5. **The CPU-only-slot refusal was unreachable from production.** It fired only
   under `strict=True`, which only `split_pipeline_at_gpu` passes — and that is
   reached only after `pipeline_requires_gpu` has already returned `True`. A
   meas-slot GPU detector would have returned `False` and run silently on CPU,
   while the plan's own test passed by calling the helper directly.

Also corrected: Task 2 rescoped to `tune/` only (`gui/` walks type annotations,
not instances — nothing to share); Task 13 re-aimed from three non-existent
`gpu_key` uses to the two Stage-2 call sites that never received
`stage2_prefix`; the digest revision moved out of the base payload, which would
have cold-started every in-flight `full` and `measure` continuation.

One reviewer finding **overturned my own analysis**: I believed the plan's
`_branch_prefix` was wrong and the spike's was right. The reverse is true — the
spike lacks a root guard on its final block and, for a top-level detector,
returns every preceding top-level op as the Stage-2 prefix. All three spike
shapes nest, so nothing caught it. Task 5 now pins it.

---

## Self-Review Notes

**Spec coverage:** every row of the spec's §9 inventory maps to a task — 1→T3, 2/2a→T1/T2, 3→T5, 4→T6, 5→T7, **5a/5b→T6a** (slot keying and the legacy relocation), 6→T8, 7→T11, 7a→T12, 7b/7c→T14, 7d→T11 step 5, 7e/7f/7g→T4, 8→T5 step 3, 9→T13, 10→T9/T10/T15.

**Known gap, deliberately left to the executor:** Task 7 and Task 11 reference existing store-construction fixtures in `tests/unit/cli/` and `tests/integration/cli/` without reproducing them. Those fixtures are long and already established (`test_staged_resume.py`, `test_staged_gpu_local.py`); copying them here would drift. The executor must read the existing pattern and follow it.
