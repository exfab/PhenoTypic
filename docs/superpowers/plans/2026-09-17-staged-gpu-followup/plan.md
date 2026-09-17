# Staged-GPU follow-up (Stage-2 storage + parameter-shape coverage) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the two defects an independent review found in the staged GPU engine as shipped in PR #224 — the retained Stage-2 output is stored uncompressed (~1 TiB live for a production run), and a `GpuDetector` held in a dict- or tuple-valued parameter of a CPU operation is invisible to detection rather than refused.

**Architecture:** Both are local changes to code PR #224 already landed, and neither changes the staged engine's shape. Compression changes only the *bytes* at an unchanged path, so every `.is_file()` predicate, the write-raw-then-token ordering, the legacy relocation and the deletion order keep working, and the reader accepts both forms so an interrupted run replays instead of re-inferring. The walker change adds two field shapes to one traversal, after which the existing ancestor contract refuses them with no new refusal code.

**Tech Stack:** Python 3.11+, pydantic v2, numpy (`np.savez_compressed` / `np.load`), zarr v3 / OME-Zarr 0.5, pytest, `uv`, Slurm.

**Spec:** `docs/superpowers/specs/2026-09-15-nested-gpu-staging/design.md` (the staged engine this extends; §4.4 for the Stage-2 signal, §4.3 for the container contract). The findings this plan closes are R-1 and R-2 of `docs/superpowers/reports/2026-09-15-nested-gpu-staging/plan-b-staleness-review.md`.

## Global Constraints

- `uv` is the sole runner: `uv run <cmd>`, never bare `python` or `pip`.
- Operations are keyword-only pydantic models; construction with a positional argument raises.
- **Stage 2 never writes the store.** Its result goes under `.phenotypic/progress/` only.
- **Write raw then token; delete token then raw.** The only reachable intermediate state is "no token, orphan raw".
- Stage-2 signal paths keep the layout `stage2_raw/<ds>/<slot>/<stem>.npy` and `stage2_done/<ds>/<slot>/<stem>.json`; `<slot>` is `detector_slot(path)`.
- Path segments are **non-empty strings**, because a path doubles as a `pipeline_step_path`. A list or tuple entry is `"field[0]"`; a dict entry is `"field:<key>"`.
- `uv run ruff check <explicit paths>` — never bare, which rewrites unrelated files.
- Tests: `QT_QPA_PLATFORM=offscreen`, explicit `-n`, `-o addopts=` plus `-m "not slow"`, never `-x` for a baseline. Anything over ~10 minutes is a Slurm job on a frozen checkout built by `docs/superpowers/plans/2026-09-15-nested-gpu-staging/make_gate_tree.sh`, which syncs `--all-extras` as CI does.

## Out of scope, by decision (2026-09-17)

| Item | Decision | Why |
|---|---|---|
| Stage `FilamentousFungiDetector` (a CPU op holding a GPU detector) | **Keep refusing.** GPU detectors stay prohibited as a parameter of a CPU operation. | The user's call. It is a policy question, not a missing mechanism: one `_CHILD_CONTRACT` entry plus a behavioural probe would admit it on the proven replay path (review F-9). Task 2 below *enforces* the prohibition on the two parameter shapes that previously escaped it. |
| A guard on non-deterministic CPU ops inside a GPU branch | **Leave as documented.** | The user's call: prefix ops are not required to be deterministic, and no shipped pipeline has an in-branch prefix at all (`F1gfd5`'s is empty). The hazard is recorded in `gpu-smoke.md` §4 and in `_cli/CLAUDE.md`. |
| Plan B (declared phase protocol) | **Blocked, not scheduled.** | `plan-b-phase-protocol.md` is updated and marked blocked on 11 design forks. Its headline saving does not apply to `F1gfd5`, whose operations already each run once. |

## File Structure

| File | Responsibility in this change |
|---|---|
| `src/phenotypic/_cli/_cli_stage2_token.py` | Owns the Stage-2 signal. Gains `_STAGE2_RAW_KEY`; `write_stage2_raw` compresses; `load_stage2_raw` reads both forms. No path helper changes. |
| `tests/unit/cli/test_cli_stage2_token.py` | Unit coverage for that module: compression, backward compatibility, and the repaired failed-write control. |
| `src/phenotypic/sdk_/_operation_tree.py` | The single traversal. `iter_child_operations` yields dict and tuple entries; `_child` resolves both spellings. |
| `tests/unit/sdk_/test_operation_tree.py` | Path spelling, resolution, and the value-not-type control. |
| `tests/unit/cli/test_gpu_detection_tree_wide.py` | The refusal, driven through detection — the property the walker change exists for. |
| `CLAUDE.md`, `src/phenotypic/_cli/CLAUDE.md`, `docs/source/how_to/pages/gpu_detection_setup.md` | Record that the file is compressed and that readers accept the older form. |

---

## Task 1: Compress the retained Stage-2 raw output

**Files:**
- Modify: `src/phenotypic/_cli/_cli_stage2_token.py` (`_STAGE2_RAW_DIR` block, `write_stage2_raw`, `load_stage2_raw`)
- Test: `tests/unit/cli/test_cli_stage2_token.py`
- Modify: `CLAUDE.md`, `src/phenotypic/_cli/CLAUDE.md`, `docs/source/how_to/pages/gpu_detection_setup.md`

**Interfaces:**
- Consumes: `stage2_raw_path(output_dir, dataset, image_stem, slot) -> Path`, `atomic_write_with_writer(final, writer, *, commit_guard)`.
- Produces: unchanged signatures — `write_stage2_raw(output_dir, dataset, image_stem, array, slot, *, commit_guard=None) -> Path` and `load_stage2_raw(output_dir, dataset, image_stem, slot) -> np.ndarray`. Only the bytes at that path change, so `stage2_result_replayable`, `relocate_legacy_stage2_signal`, `delete_stage2_raw` and all five probe sites need no edit.

**Why:** the controller starts Stage 3 only after the whole Stage-2 round, so every image's raw output is live at once. At 3140×5094 uint16 that is ~32 MB per image and ~1 TiB for a 33,923-image run.

- [x] **Step 1: Write the failing tests**

```python
def test_raw_array_is_stored_compressed(tmp_path: Path) -> None:
    array = np.full((1024, 2048), 7, dtype=np.uint16)
    written = write_stage2_raw(tmp_path, "ds", "img", array, _SLOT)

    assert written.stat().st_size < array.nbytes // 100
    np.testing.assert_array_equal(
        load_stage2_raw(tmp_path, "ds", "img", _SLOT), array
    )


def test_a_precompression_raw_array_still_replays(tmp_path: Path) -> None:
    """Writing the old form BY HAND is the only way to test the reader: a file
    written by write_stage2_raw cannot fail this whatever the reader does."""
    array = np.arange(12, dtype=np.uint16).reshape(3, 4)
    legacy = stage2_raw_path(tmp_path, "ds", "img", _SLOT)
    legacy.parent.mkdir(parents=True, exist_ok=True)
    with open(legacy, "wb") as handle:
        np.save(handle, array)

    loaded = load_stage2_raw(tmp_path, "ds", "img", _SLOT)

    np.testing.assert_array_equal(loaded, array)
    assert loaded.dtype == array.dtype
    assert stage2_result_replayable(tmp_path, "ds", "img", _SLOT) is False
```

Add `stage2_result_replayable` to the module's import list in the test file.

- [x] **Step 2: Run them and watch the first one fail**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/cli/test_cli_stage2_token.py -q --no-header -p no:randomly -o addopts= -k compressed`
Expected: FAIL — `np.save` writes 4 MB for a 4 MB array, so `st_size < nbytes // 100` is false.

- [x] **Step 3: Compress on write**

```python
#: Array name inside the compressed raw file. Reading tolerates any single
#: name, so a file written before this key existed still loads.
_STAGE2_RAW_KEY = "result"
```

and in `write_stage2_raw`:

```python
    def _write(path: str) -> None:
        with open(path, "wb") as handle:
            np.savez_compressed(handle, **{_STAGE2_RAW_KEY: array})
```

- [x] **Step 4: Accept both forms on read**

```python
    loaded = np.load(
        stage2_raw_path(output_dir, dataset, image_stem, slot),
        allow_pickle=False,
    )
    if isinstance(loaded, np.ndarray):  # pre-compression file
        return loaded
    with loaded as archive:
        name = (
            _STAGE2_RAW_KEY
            if _STAGE2_RAW_KEY in archive.files
            else archive.files[0]
        )
        return archive[name]
```

- [x] **Step 5: Repair the failed-write control**

`test_a_failed_raw_write_never_replaces_a_good_one` patched `np.save`. The writer no longer calls it, so the patch left the write **succeeding** and the assertion compared the good array against itself — a test that could only report success. Patch what the writer calls:

```python
    # The writer compresses, so this patches savez_compressed, not save. A
    # patch on the wrong function makes the write SUCCEED and the assertion
    # below then compares the good array against itself.
    monkeypatch.setattr(module.np, "savez_compressed", _raise_boom)
```

- [x] **Step 6: Run the module's tests**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/cli/test_cli_stage2_token.py tests/unit/cli/test_stage2_slot_keying.py -q --no-header -p no:randomly -o addopts=`
Expected: PASS (24 + 17).

- [x] **Step 7: Record it in the docs**

Root `CLAUDE.md`, `_cli/CLAUDE.md` and `how_to/pages/gpu_detection_setup.md` each describe the `.npy`; say it is compressed, that the name is unchanged, and that readers accept a pre-compression bare array. Then confirm the doc pins still hold:

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/test_docs_staged_cli.py tests/unit/test_docs_myst_fences.py -q --no-header -p no:randomly -o addopts=`
Expected: PASS (13).

- [x] **Step 8: Commit** — landed as `1046030f`.

```bash
git add src/phenotypic/_cli/_cli_stage2_token.py tests/unit/cli/test_cli_stage2_token.py \
        CLAUDE.md src/phenotypic/_cli/CLAUDE.md docs/source/how_to/pages/gpu_detection_setup.md
git commit -F <message file>
```

**Measured, not assumed:** a plate-like objmap (16 disks of r=180 on 3132×5086 uint16) went 31.9 MB → 0.06 MB (525×) for 0.68 s of CPU, round-tripping bit-exact. Real detector output has noisier boundaries; the design note estimates 10–30×. Either way the write cost is far below one image's inference.

---

## Task 2: Walk dict- and tuple-valued operation fields

**Files:**
- Modify: `src/phenotypic/sdk_/_operation_tree.py` (`iter_child_operations`, `_child`)
- Test: `tests/unit/sdk_/test_operation_tree.py`, `tests/unit/cli/test_gpu_detection_tree_wide.py`

**Interfaces:**
- Consumes: `_is_operation(value) -> bool`, `_SLOT_SEPARATOR = ":"`, `_INDEXED` (integer-index bracket form).
- Produces: unchanged signatures. `iter_child_operations(obj)` additionally yields `("field:<key>", child)` for a dict-valued field and `("field[<i>]", child)` for a tuple-valued one; `get_at_path` resolves both. `find_gpu_detectors`, `pipeline_requires_gpu` and `refuse_cpu_only_slot` gain the coverage without any change of their own.

**Why:** every operation-valued parameter shipped today is a single operation or a list, both already walked. A user's own class may hold one in a dict or tuple field, and an unwalked field is worse than an unsupported one — the detector is invisible, `pipeline_requires_gpu` answers `False`, and the run goes to the CPU strategy and infers per image. That is the wrong-answer bug tree-wide detection exists to prevent, one field shape further out.

- [x] **Step 1: Write the failing tests**

In `tests/unit/sdk_/test_operation_tree.py`, a carrier whose two fields accept anything:

```python
class _DictCarrier(OtsuDetector):
    keyed: Any = None
    fixed: Any = None


def _carrier(**children: Any) -> _DictCarrier:
    return _DictCarrier(**children)


def test_walk_yields_dict_field_entries_with_a_colon_namespace():
    inner = ManualPointDetector(centers=CENTERS, shape="disk", width=11)
    pipe = ImagePipeline(ops={"Carrier": _carrier(keyed={"inoculum": inner})})

    paths = {"/".join(path) for path, _ in walk_operations(pipe)}

    assert "Carrier/keyed:inoculum" in paths
    assert get_at_path(pipe, ("Carrier", "keyed:inoculum")) is inner


def test_walk_yields_tuple_field_entries_as_bracket_indexed_strings():
    inner = ManualPointDetector(centers=CENTERS, shape="disk", width=11)
    pipe = ImagePipeline(ops={"Carrier": _carrier(fixed=(OtsuDetector(), inner))})

    paths = {"/".join(path) for path, _ in walk_operations(pipe)}

    assert "Carrier/fixed[1]" in paths
    assert get_at_path(pipe, ("Carrier", "fixed[1]")) is inner


def test_a_dict_field_of_plain_values_yields_nothing():
    """Control: the walker filters on the value, not on the field's type."""
    pipe = ImagePipeline(ops={"Carrier": _carrier(keyed={"width": 11})})

    assert [path for path, _ in walk_operations(pipe)] == [("Carrier",)]
```

and in `tests/unit/cli/test_gpu_detection_tree_wide.py` the property itself, driven through detection:

```python
def test_a_gpu_detector_in_a_dict_valued_parameter_is_refused():
    from phenotypic.detect import OtsuDetector

    class _KeyedCarrier(OtsuDetector):
        keyed: Union[dict, None] = None

    pipe = ImagePipeline(
        ops={"Carrier": _KeyedCarrier(keyed={"inoculum": FakeGpuDetector()})}
    )

    with pytest.raises(
        UnstageableGpuDetectorError, match="only composition primitives"
    ) as caught:
        find_gpu_detectors(pipe)

    # The ancestor refusal names the carrier class, not the path. What this
    # test pins is that the detector is SEEN at all.
    assert "_KeyedCarrier" in str(caught.value)


def test_a_gpu_detector_in_a_tuple_valued_parameter_is_refused():
    """Control on the other new shape, indexed rather than keyed."""
    from phenotypic.detect import OtsuDetector

    class _FixedCarrier(OtsuDetector):
        fixed: Union[tuple, None] = None

    pipe = ImagePipeline(
        ops={"Carrier": _FixedCarrier(fixed=(ManualPointDetector(), FakeGpuDetector()))}
    )

    with pytest.raises(
        UnstageableGpuDetectorError, match="only composition primitives"
    ) as caught:
        find_gpu_detectors(pipe)

    assert "_FixedCarrier" in str(caught.value)
```

- [x] **Step 2: Run them and watch four fail**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/sdk_/test_operation_tree.py tests/unit/cli/test_gpu_detection_tree_wide.py -q --no-header -p no:randomly -o addopts= -k "dict_field or tuple_field or dict_valued or tuple_valued"`
Expected: FAIL — the walk yields only `{"Carrier"}`, and `find_gpu_detectors` returns `[]` instead of raising. The plain-values control passes already.

- [x] **Step 3: Yield the two shapes**

In `iter_child_operations`, replacing the `isinstance(value, list)` branch:

```python
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                if _is_operation(item):
                    yield f"{field_name}[{index}]", item
        elif isinstance(value, dict):
            for key, item in value.items():
                if _is_operation(item) and isinstance(key, str) and key:
                    yield f"{field_name}{_SLOT_SEPARATOR}{key}", item
        elif _is_operation(value):
            yield field_name, value
```

Comment why: an unwalked field makes a detector invisible; a set is deliberately excluded because no stable segment can name an entry, so a path into one could not round-trip.

- [x] **Step 4: Resolve the two spellings**

In `_child`, widen the index branch to `isinstance(sequence, (list, tuple))`, and add the keyed lookup **before** the plain attribute lookup, so a field whose name contains a colon cannot shadow it:

```python
    field, separator, key = segment.partition(_SLOT_SEPARATOR)
    if separator:
        mapping = getattr(node, field, None)
        if isinstance(mapping, dict) and key in mapping:
            return mapping[key]
```

- [x] **Step 5: Run the two suites whole**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/sdk_/test_operation_tree.py tests/unit/cli/test_gpu_detection_tree_wide.py tests/unit/detect/test_container_child_contracts.py -q --no-header -p no:randomly -o addopts=`
Expected: PASS (61 for the first two; 68 with the contracts suite).

- [x] **Step 6: Commit** — landed as `6e2a7c06`.

```bash
git add src/phenotypic/sdk_/_operation_tree.py tests/unit/sdk_/test_operation_tree.py \
        tests/unit/cli/test_gpu_detection_tree_wide.py
git commit -F <message file>
```

Use a message **file**: backticks in a `-m` string are command substitution, and a message containing `` `keyed:inoculum` `` silently loses that text.

---

## Task 3: Regression gate and the follow-up PR

**Files:** none — verification and publication only.

**Interfaces:**
- Consumes: `docs/superpowers/plans/2026-09-15-nested-gpu-staging/make_gate_tree.sh`, `run_phase_gate.sbatch`.

- [x] **Step 1: Build a frozen checkout at HEAD**

Run: `docs/superpowers/plans/2026-09-15-nested-gpu-staging/make_gate_tree.sh HEAD followup`
It refuses a dirty tree, proves the checkout imports its own `src/`, and syncs `--all-extras` as CI does. Landed as `/bigdata/exfab/anguy344/gate-trees/6e2a7c06-followup`.

- [x] **Step 2: Submit the full suite as a 24-shard array**

```bash
PHENO_GATE_TREE=/bigdata/exfab/anguy344/gate-trees/6e2a7c06-followup \
PHENO_GATE_PATHS="tests/unit tests/smoke tests/integration tests/gui" \
  sbatch --array=0-23%24 docs/superpowers/plans/2026-09-15-nested-gpu-staging/run_phase_gate.sbatch
```

Submitted as job 28705414.

- [ ] **Step 3: Read the result by NAME, not by count**

Every shard must print `Commit: <HEAD>` and `Dirty: 0`. Compare the failure names against the previous green run at `64de95b9` (12,933 passed, 0 outstanding). A count moves with shard packing; a name does not. Re-run any failure alone before attributing it — contention failures pass in isolation.

- [ ] **Step 4: Push and open the PR against #224's branch**

```bash
GIT_SSH_COMMAND="ssh -i /rhome/anguy344/.ssh/github_agent -o IdentitiesOnly=yes -o BatchMode=yes" \
  git push origin worktree-nested-gpu-staging
gh pr create --base worktree-nested-gpu-staging --head <follow-up branch> --title ... --body-file ...
```

The base is #224's branch, not `main`, so the diff shows only this follow-up. The body states the measured compression ratio, the backward-compatible reader, the parameter shapes now refused, and the three out-of-scope decisions above.

- [ ] **Step 5: Remove the frozen checkout**

```bash
git worktree remove --force /bigdata/exfab/anguy344/gate-trees/6e2a7c06-followup
git worktree prune
```

---

## Self-review

**Coverage.** R-1 (uncompressed Stage-2 storage) → Task 1. R-2 (an unenforceable prohibition, in the shape the user asked to enforce) → Task 2. The two risks the review raised against the as-built code are the only in-scope items; everything else it raised is listed under *Out of scope, by decision* with the reason.

**Placeholders.** None: every code step carries the code as landed, and every run step carries its command and expected result.

**Type consistency.** No signature changes in either task. Task 1 keeps `write_stage2_raw` / `load_stage2_raw` and touches no path helper. Task 2 keeps `iter_child_operations` / `get_at_path` and adds only segment shapes, which `_child` resolves. The only new name is the module-private `_STAGE2_RAW_KEY`.

**Status.** Tasks 1 and 2 are implemented and committed (`1046030f`, `6e2a7c06`); Task 3 is at Step 3, waiting on job 28705414.
