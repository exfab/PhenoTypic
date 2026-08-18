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
