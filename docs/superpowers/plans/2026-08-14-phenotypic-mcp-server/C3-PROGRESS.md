# Cluster C3 — Task 9: extract a pure sbatch-spec builder

Branch `feat/mcp-server`, started from `5ad5530b9`.

## What was built

`src/phenotypic/_cli/_cli_slurm_array_scripts.py` now splits in two:

```python
def build_array_script_spec(
    dataset: Dataset,
    array_indices: Tuple[int, int],
    config: ExecutionConfig,
    output_dir: Path,
    chunk_id: int = 0,
    checkpoint_interval: Optional[int] = None,
    is_last_chunk: bool = False,
) -> SlurmArrayScriptSpec:
```

Identical shape to `generate_array_job_script`, so every argument works either
positionally or by keyword. `output_dir` is read as a *value* only — embedded in
the worker command line and in the `#SBATCH --output` log path, never created.

`generate_array_job_script` keeps its original positional signature and its
return type (`Path`), and is now the only side-effecting half: it calls the
builder, does `slurm_scripts_dir(...).mkdir()` + `logs_dir(...)/"slurm"/name`
`.mkdir()`, and writes the script.

A third helper, `_array_script_names(dataset, array_indices, chunk_id) ->
(job_name, script_name)`, holds the single-chunk-vs-chunked naming rule. The
builder needs `job_name` (it goes in the spec) and the generator needs
`script_name` (it does not); without the helper the rule would have been
duplicated across the split — exactly the drift the agreement test exists to
catch.

Tests: `tests/unit/cli/test_build_array_script_spec_is_pure.py` (3 cases) plus a
new `array_script_kwargs` fixture in `tests/unit/cli/conftest.py`.

## Deviation from the plan sketch

The plan's Step-3 sketch ends with `write_slurm_array_script(script_dir / name,
spec.render())`. The real signature is `write_slurm_array_script(path: Path, spec:
SlurmArrayScriptSpec) -> Path` (`sdk_/slurm/_script_rendering.py:133`) — it takes
the **spec**, not the rendered text, and it returns the path. The implementation
passes `spec` and returns the call's result directly.

Also, `git add -A` from the plan's Step 6 was **not** used; paths were staged
explicitly.

## Mutation runs — every one actually executed

| # | Mutation | Expected | Observed |
|---|---|---|---|
| 1 | Revert the module to `5ad5530b9` (pre-extraction) — the plan's Step 2 | ImportError | **2 failed** — `ImportError: cannot import name 'build_array_script_spec'` |
| 2 | `(output_dir / "scratch").mkdir(...)` at the top of `build_array_script_spec` | purity test fails | **FAILED** `test_build_array_script_spec_writes_nothing` — "the builder touched the output dir", digest `e3b0c442…` → `5a9cb6b5…` |
| 3a | `job_name += "-MUTATED"` in the **builder** (changes the rendered `#SBATCH --job-name`) | (brief predicted the agreement test fails) | **all 3 passed** — see finding 1 |
| 3b | Generator drifts: `spec.model_copy(update={"job_name": ... + "-DRIFT"})` before the write | agreement test fails | **FAILED** `test_generator_and_builder_agree` |
| 4 | Generator returns the path without mkdir/write | writer-side guard fails | **FAILED** `test_generator_still_writes_the_script` (and `test_generator_and_builder_agree`) |

Module restored byte-for-byte from a saved copy after each mutation; the final
`git diff --stat` showed only the two intended files.

## Findings

**1. The brief's prescribed agreement mutation cannot fail, and that is correct.**
"Change one rendered `#SBATCH` line in the builder (the agreement test must
fail)" holds only while the generator carries its *own* copy of the spec. After
the extraction the generator delegates, so any builder-side change moves both
sides of the comparison together and they still match (mutation 3a: 3 passed).
`test_generator_and_builder_agree` is a *duplication* detector, not a rendering
detector — the mutation that exercises it is one that makes the generator diverge
from the builder (3b), which does fail. Reported rather than worked around; the
test is unchanged.

**2. Pre-existing weak assertions in `tests/unit/cli/test_cli_slurm_array.py`.**
Under mutation 3a the whole 35-case file still passed with every job name
silently renamed to `pht-test_dataset-chunk0-MUTATED`. The assertions are
substring checks (`assert "#SBATCH --job-name=pht-test_dataset-chunk0" in
content`), so an appended suffix slips through. Out of scope for Task 9 and left
alone, but it means that file is not a guard on job naming.

**3. Every call site of `generate_array_job_script` already uses keyword
arguments** — `_cli_slurm_array_scripts.py:484` and all ten test call sites in
`test_cli_slurm_array.py`, `test_slurm_process_only_scripts.py`,
`test_cli_v2.py`. The B8 instruction to keep the positional signature was
followed anyway (it is the existing signature and cheapest to keep), but the
breakage it guards against would not have occurred.

## Verification

- `uv run --no-sync pytest tests/unit/cli -q` — **452 passed** (4:25).
- `uv run --no-sync pytest tests/unit/gui/run_console/test_slurm_live_harness.py -q` — **30 passed**.
- `uv run --no-sync ruff check src/phenotypic/_cli/_cli_slurm_array_scripts.py tests/unit/cli/conftest.py tests/unit/cli/test_build_array_script_spec_is_pure.py` — **All checks passed**, before committing.
- `uv run --no-sync mypy src/phenotypic` — 420 errors / 124 files both with and
  without the change. Verified by **diff**, not by count: `git stash push` on the
  two modified files, rerun, `git stash pop`, then compare the two `src/`-prefixed
  error sets sorted and with `` `N` `` typevar ids normalized. Diff is empty.

---

# C3 re-apply, on top of the main merge (`6007f5a0c`)

The first pass (`7471ae701`) was displaced, not rejected: main independently
rewrote `_cli_slurm_array_scripts.py` (+56/-3, the identity-verification
mechanism) and the conflict was resolved by taking main's file whole. Same task,
same design, redone against the new file. The `array_script_kwargs` fixture in
`tests/unit/cli/conftest.py` survived the merge untouched and is unchanged here.

## What moved where

`build_array_script_spec` keeps the signature from the first pass. Everything
main added is spec-building and lives in the **builder**: the
`CURRENT_WORK_ID` / `CURRENT_INPUT_SHA256` / `CURRENT_ATTEMPT_ID` assignments
prepended to `dispatch_block`, the `identity_rows` loop and the three
`EXPECTED_*` / `ATTEMPT_IDS` prelude arrays, `EXPECTED_PIPELINE_SHA256`, the four
new dispatch args, the relocated `--input-root`, and the `SLURM_GENERATION_ENV_VAR`
prelude line. The writer keeps exactly three things: `script_dir.mkdir()`,
`log_dir.mkdir()`, `write_slurm_array_script(script_dir / script_name, spec)`.

The ~230 moved lines were **transformed programmatically from the merged file**,
not retyped — five anchored replacements (drop `script_dir` + its mkdir, swap the
naming if-block for `_array_script_names`, drop the `log_dir` mkdir but keep
`log_path`, drop `script_path`, turn the `write_slurm_array_script(script_path,
SlurmArrayScriptSpec(...))` call into `return SlurmArrayScriptSpec(...)` dedented
one level). Every anchor asserted `count == 1`. Hand-distributing main's identity
code was the risk the team lead flagged, so no line of it was hand-edited.

## Two properties of main's identity mechanism — the answer to the extra check

**1. The builder is pure with respect to `output_dir`, but it is NOT I/O-free: it
reads every input image.** `work_id_for_image` (`_cli_failure_tracker.py:177`)
calls `file_sha256(config.pipeline_json)` **and** `file_sha256(image_path)`, and
the `identity_rows` loop then calls `file_sha256(image_path)` a second time. Per
chunk of N images that is 2N image reads plus N pipeline-JSON reads. Nothing is
written and nothing under `output_dir` is touched, so the preview guarantee holds
— but a `deploy_plan` preview inherits a full read of the chunk's images, which
on a real plate dataset is not free. Pre-existing in main; not worked around.
Pinned by `test_building_a_spec_reads_every_input_image`.

**2. The spec is nondeterministic: each task's `ATTEMPT_IDS` entry is a fresh
`uuid4().hex`.** Two calls with identical arguments render scripts that differ.
This is a genuine blocker for the agreement test as originally written — byte
equality of two independent calls is unsatisfiable by *any* correct
implementation, not just by a wrong one. Confirmed empirically before changing
anything: the restored test failed with the diff isolated to the `ATTEMPT_IDS`
array and nothing else.

Rather than delete the test or weaken it to a substring check, it was split into
three guards:

- `test_generator_and_builder_agree` — byte equality with **only** the
  `ATTEMPT_IDS=( ... )` block masked. Every other byte must match.
- `test_attempt_ids_are_the_only_drift` — builds the same spec twice and asserts
  the renders differ *and* are equal once masked. This is what keeps the mask
  honest: if a second field ever became per-call random, the masked comparison
  above would keep passing while this fails. Mutation M6 proves it fires.
- `test_generator_consumes_the_builder` — monkeypatches the module's
  `build_array_script_spec` to stamp a job name nothing else produces, then
  asserts the written file equals that spec's render. A structural proof of
  consumption, immune to the drift entirely.

**Implication for Phase 2C worth deciding before `deploy_plan` is built:** a
preview cannot be byte-identical to the script that eventually gets submitted,
because the attempt ids are regenerated at submit time. Either the preview is
presented as "modulo attempt ids", or the attempt ids have to be threaded in
rather than generated inside the builder. Flagging, not deciding.

## Mutation runs — all seven executed against the post-merge file

| # | Mutation | Expected | Observed |
|---|---|---|---|
| 1 | Revert the module to `6007f5a0c` (pre-extraction) | ImportError | **5 failed, 1 passed** — `ImportError: cannot import name 'build_array_script_spec'` |
| 2 | `(output_dir / "scratch").mkdir(...)` at the top of the builder | purity fails | **FAILED** `test_build_array_script_spec_writes_nothing` |
| 3a | `job_name += "-MUTATED"` in the **builder** | (brief predicted agreement fails) | **all 6 passed** — see finding 1 below; still true post-merge |
| 3b | Generator drifts: `dataclasses.replace(spec, job_name=... + "-DRIFT")` before the write | agreement fails | **FAILED** `test_generator_and_builder_agree` **and** `test_generator_consumes_the_builder` |
| 4 | Generator returns the path without mkdir/write | writer guard fails | **FAILED** `test_generator_still_writes_the_script` (+ the two above) |
| 5 | Attempt ids made constant (`"deadbeef"` for `uuid4().hex`) | drift guard fails | **FAILED** `test_attempt_ids_are_the_only_drift` on `first != second` |
| 6 | A **second** field made per-call random (`job_name = f"{job_name}-{uuid4().hex}"`) | mask must not hide it | **FAILED** `test_attempt_ids_are_the_only_drift` **and** `test_generator_and_builder_agree` |
| 7 | Builder stops hashing inputs (stub work id / sha / pipeline sha) | read guard fails | **FAILED** `test_building_a_spec_reads_every_input_image` |

Module restored from a saved copy after each mutation; the final restore was
verified with `diff -q` (identical), not assumed.

## Findings

**1. The prescribed agreement mutation still cannot fail, for the same reason.**
Re-confirmed on the merged file (M3a: 6 passed). Acknowledged by the team lead;
M3b is now the agreement test's mutation of record.

**2. The substring false green in `tests/unit/cli/test_cli_slurm_array.py`
survived the merge.** Under M3a all **36** cases passed with every job name
renamed to `pht-test_dataset-chunk0-MUTATED`. Still out of scope; still recorded.

**3. New — the builder reads the inputs and is nondeterministic.** Detailed
above. Neither was worked around; both are pinned by tests and reported.

## Verification

- `uv run --no-sync pytest tests/unit/cli -q` — **552 passed** (5:11) =
  the 546 post-merge baseline + the 6 new cases.
- `uv run --no-sync pytest tests/unit/gui/run_console/test_slurm_live_harness.py -q` — **30 passed**.
- `uv run --no-sync ruff check src/phenotypic/_cli/_cli_slurm_array_scripts.py tests/unit/cli/conftest.py tests/unit/cli/test_build_array_script_spec_is_pure.py` — **All checks passed**, before committing.
- `uv run --no-sync mypy src/phenotypic` — by **diff**: `git stash push` the one
  modified source file, rerun, `git stash pop`, compare `src/`-prefixed error sets
  sorted with `` `N` `` typevar ids normalized. **Empty diff.** (Absolute count
  moved 420/124 → 416/123 across the merge, which is exactly why the count is not
  the signal.)
