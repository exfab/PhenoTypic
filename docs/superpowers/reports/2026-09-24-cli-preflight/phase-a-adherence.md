# Phase A adherence review: CLI preflight ordering (F1, F2, F3, F27)

- **Reviewer:** independent (did not write the code)
- **Date:** 2026-09-24
- **Reviewed:** `git diff 145157e..e13247d -- src tests` (implementation commits `425eb66`
  test, `e13247d` fix) against spec `docs/superpowers/specs/2026-09-24-cli-preflight/design.md`
  §0, §1 and plan `docs/superpowers/plans/2026-09-24-cli-preflight/plan.md` Phase A (Tasks 1, 2)
- **Worktree HEAD:** `e13247d` (detached from `origin/claude/modest-mccarthy-jz0ylw`)
- **Environment note:** this container has no `libEGL.so.1`, so the `pytest-qt` plugin
  aborts collection. Every pytest command below therefore adds `-p no:pytest-qt`
  (and `-p no:cacheprovider`). No test in the reviewed surface uses Qt.

## Verdict

**Pass with changes.**

The reorder is correct: every step above the `--dry-run` exit is read-only (confirmed by
reading and by an `strace` of three dry-run variants against a real prior run), every
mutation sits below it in the original relative order, the four findings are closed for the
cases the plan names, and each fix is pinned by a test that fails when the fix is reverted.
No regressions in the spot-checked surface.

One Major gap remains in the new overlap refusal (A1: a run input that is a **symlink**
stored under `--output` is not refused, so `--overwrite` still deletes the previous run and
then fails). One Minor behavior change reaches a non-dry-run path the task asked about
(A2: `--mode recompile` now refuses a `--metadata` under `--output` when a no-op
`--overwrite` is passed; it exited 0 at `145157e`). The rest are test-coverage and wording
items.

## Findings

### Blocking

None.

### Major

#### A1. The overlap refusal misses a symlinked run input stored under `--output`

`_refuse_run_inputs_the_run_would_delete` (`src/phenotypic/phenotypicCLI.py:578-635`)
compares only the **resolved** input path (`:618`, `Path(path).resolve(strict=False)`)
with the resolved output. A symlink that *lives* under `--output` but points outside it
resolves outside, so it is not refused. `shutil.rmtree` does not follow symlinks; it removes
the link itself (`:2646-2650`). The run then reads the link path again after the delete
and fails, having already deleted the previous run. That is the F1/F27 harm pattern
(destroy, then fail), for a file that is literally "stored under `--output`" (F27's wording).

Evidence (probe `edge_cases.py`, CliRunner, HEAD `e13247d`, prior real run in `out/`):

```
[symlink --pipeline inside --output -> outside target, --overwrite] exit=1 link_exists_after=False target_exists=True
    | Overwriting existing output directory: /tmp/tmp26efiwa0/out
    | Error: Cannot prepare incremental startup state: [Errno 2] No such file or directory: '/tmp/tmp26efiwa0/out/deliverables/linked_pipeline.json'
[symlink --input dir inside --output -> outside target, --overwrite] exit=1 link_exists_after=False target_exists=True
    | Overwriting existing output directory: /tmp/tmpcdjstk8q/out
    | Error: Cannot prepare incremental startup state: [Errno 2] No such file or directory: '/tmp/tmpcdjstk8q/out/deliverables/linked_images/plate1/img001.tiff'
[symlink --metadata inside --output -> outside target, --overwrite] exit=1 link_exists_after=False
    | Overwriting existing output directory: /tmp/tmp5z__ul3_/out
    | Error: Cannot prepare incremental startup state: [Errno 2] No such file or directory: '/tmp/tmp5z__ul3_/out/deliverables/linked_meta.csv'
```

The link *targets* survive, so no user input is lost, but the previous run is deleted and
the new run fails, with an error that does not name the cause. The same gap exists one level
down for `--restart`: a symlink nested inside a real child of `.phenotypic/` (for example
`.phenotypic/progress/p.json -> /elsewhere/p.json`) is not refused, because the restart
targets are resolved at child granularity (`:612-616`) and the nested link resolves outside
them, yet `rmtree(.phenotypic/progress)` removes it. (A symlink that *is* a direct child of
`.phenotypic/` is refused, because the target resolves to the same place.)

The spec and plan prescribe `resolve(strict=False)` "as the existing process-mode rule
does", so the implementation follows the letter of the plan; the gap is in the spec. The
process-mode rule is about tree containment, where resolving is right; this rule is about
what a delete will remove, where the *lexical* location is what matters.

**Fix.** Refuse when either spelling lies inside the delete: compare both
`Path(os.path.abspath(path))` (lexical, normalized, links not followed) and
`Path(path).resolve(strict=False)` against both `Path(os.path.abspath(output_dir))` and the
resolved output; for the restart half, compare the lexical input path against the
**unresolved** targets as well as the resolved pair. Add a test that places a symlink under
`--output` pointing at a pipeline outside it and asserts refusal, exit != 0, and that
`previous-run.txt` survives. Record the spec deviation in §1.

### Minor

#### A2. The refusal now fires in `--mode recompile`, where `--overwrite` and `--restart` do nothing

The refusal is gated only on `not migrate_only` (`phenotypicCLI.py:1992`). `--mode recompile`
exits at its own branch (`sys.exit(0)` after `_handle_recompile`) long before the restart
clear and the `rmtree`, so neither flag deletes anything there, yet a recompile that names
a `--metadata` under `--output` together with a stray `--overwrite` is now refused.

Evidence (same probe, `case_recompile_overwrite_metadata_inside`):

```
# 145157e (base):
[--mode recompile --overwrite --metadata inside --output] exit=0 meta_exists_after=True
    | Recompilation complete: /tmp/tmp1xefsp0n/out
# e13247d (HEAD):
[--mode recompile --overwrite --metadata inside --output] exit=2 meta_exists_after=True
    | Error: --metadata /tmp/tmpz_m20ru0/out/deliverables/my_meta.csv lies inside --output /tmp/tmpz_m20ru0/out, which --overwrite deletes before the run reads it. ...
```

The message is also false for that mode: `--overwrite` does not delete in recompile. The same
placement changes error precedence in `--mode measure` (the overlap message pre-empts
"`--mode measure cannot be combined with --overwrite`" at `:2225`) and for
`--restart --overwrite` together (pre-empts "mutually exclusive" at `:2119`). Both still
refuse, so those two are cosmetic.

**Fix.** Run the refusal only in the modes that reach the delete:
`if cli_mode in {"full", "process"}:` (or `not (migrate_only or recompile_only or measure_only)`),
and move it below the `restart and overwrite` mutual-exclusion check so the more specific
usage errors win. Add a recompile case to the test file as the out-of-scope control that
spec §0 asks for ("an option is checked only in the modes that read it").

#### A3. The new tests carry no negative controls for the refusal

`tests/unit/cli/test_cli_preflight_ordering.py` pins every positive case, and each one fails
under revert (table below). It has no case proving the refusal *admits* what it should:

- a `--pipeline` inside a **preserved** `.phenotypic/` entry under `--restart` (probed
  manually: `.phenotypic/legacy-v2/p.json` with `--restart` exits 0 and the file survives);
- a relative `--pipeline`/`--output` spelling (probed manually: correctly refused);
- `--restart` when `.phenotypic/` is absent;
- the migrate exemption.

A mutation that dropped the `_PRESERVED_ON_RESTART` filter from
`machine_state_restart_targets` is caught only indirectly, by
`tests/unit/sdk_/test_io_constants.py::TestClearMachineState::test_restart_preserves_terminal_failure_journal`
and `tests/unit/cli/test_run_identity.py::test_restart_epoch_survives_clear_machine_state`
(verified: both fail under that mutant; none of the nine new tests does). A refusal-local
bug (for example resolving over unfiltered `cache.iterdir()` in the refusal only) would pass
the whole Phase A surface.

**Fix.** Add a parametrized negative-control test: preserved-entry input under `--restart`
(exit 0, file intact), relative spellings (refused), recompile (not refused, after A2), and
the A1 symlink case.

### Nit

#### A4. Stale guard description in the `mint_run_identity` comment

`phenotypicCLI.py:2674-2677`: "the overwrite branch above is guarded by `not config.resume
and not restart and not measure_only`". The delete is now `if will_overwrite:` at `:2646`,
and that guard lives on the computation of `will_overwrite` ~160 lines higher (`:2482`).
Still true transitively, but a reader looking "above" finds a one-line `if will_overwrite:`.
**Fix:** "the overwrite delete above runs only when `will_overwrite`, which is computed
under `not config.resume and not restart and not measure_only`".

#### A5. `clear_machine_state` is not strictly behavior-identical in two pathological legacy layouts

Old code (`145157e:src/phenotypic/sdk_/_io_constants.py:1356-1370`) called
`shutil.rmtree(legacy_progress)` unconditionally and `unlink()` on legacy files. New code
(`_io_constants.py:1349-1353`) dispatches on `is_dir() and not is_symlink()` for every
target. So a root-level `progress` that is a symlink (old: `OSError` from `rmtree`; new:
unlink) or a regular file (old: `NotADirectoryError`; new: unlink), and a root-level
`processing_state.json` or `processing_events.log` that is a directory (old:
`IsADirectoryError`; new: `rmtree`), now succeed where they raised. For every real layout the
set of deleted paths, the return value, the empty-`.phenotypic/` `rmdir` and the preserved
set are identical (deletion order is now sorted, which is unobservable). This is an
improvement; it only needs the sentence "behavior-identical" qualified wherever it is
claimed. **Fix:** a one-line note in the commit record or the `clear_machine_state` comment.

#### A6. `--mode process` refuses a `--metadata` it ignores

Process mode warns that `--metadata` is ignored (`phenotypicCLI.py:2043`, "is ignored in --mode
process"), yet the refusal checks it under `--overwrite` (probe:
`--mode process --overwrite --metadata <out>/ignored_meta.csv` exits 2; at `145157e` the
dry run exited 0 and **deleted** the file). Refusing is defensible, since it protects a user
file, but it contradicts spec §0's "an option is checked only in the modes that read it".
**Fix:** either drop `--metadata` from `run_inputs` in process mode, or keep it and say why
in the helper's docstring.

#### A7. Docstring and message wording

- `preserved_on_restart_names` (`_io_constants.py:1395`) has a one-line docstring with no
  `Returns:` section (Google style is the repo convention).
- `_refuse_run_inputs_the_run_would_delete` says the delete happens "before the run reads
  its pipeline, metadata, image manifest or images". After this phase the pipeline is
  loaded and the manifest snapshotted *before* the delete; what the delete breaks is the
  later reads (metadata snapshot, startup preparation, workers). Suggest "before the run
  has finished reading it".
- The dry-run preview (`_print_dry_run_mutation_preview`) prints above the
  "DRY-RUN MODE: Verbose Preview" banner that frames the preview. Consider printing it
  after the banner, or inside `execute_dry_run`'s output-structure section.

## Revert-proof table

Command for every row (run from the worktree root):

```
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/cli/test_cli_preflight_ordering.py \
  -o addopts= -m "not slow" -q -p no:cacheprovider -p no:pytest-qt -rfE
```

Each revert was applied to `src/phenotypic/phenotypicCLI.py` in the worktree only, then
restored with `git checkout -- src/phenotypic/phenotypicCLI.py`. After every restore the
file ran green: **9 passed** (0.7 to 1.4 s). Nothing was committed.

| Finding | Hunk reverted | Test(s) expected to catch it | Result with revert | Result restored |
|---|---|---|---|---|
| F1 | `shutil.rmtree(output_dir)` re-inlined into the `if overwrite:` branch of the freshness check (the `145157e` placement), in place of `will_overwrite = True` | `test_overwrite_with_a_corrupt_pipeline_keeps_the_previous_run`, `test_overwrite_dry_run_previews_and_deletes_nothing` | **2 failed, 7 passed.** Corrupt pipeline: `FileNotFoundError: ... out/previous-run.txt` (exit != 0 and "Pipeline loading failed" held; the previous run was gone). Dry run: `AssertionError: Overwriting existing output directory: .../out` ("would delete" absent, output deleted) | 9 passed |
| F2 (clear) | `if restart and output_dir.exists(): clear_machine_state(output_dir)` reinstated above `config.output_dir = output_dir` (the `145157e` placement) | `test_restart_dry_run_leaves_machine_state_untouched` | **1 failed, 8 passed.** `AssertionError: assert {} == {'.aggregate_...43f2646', ...}` | 9 passed |
| F2 (epoch) | `mint_run_identity(config, restart=restart)` moved to just above `if config.dry_run:` (lower call replaced by the minted value) | `test_restart_dry_run_leaves_machine_state_untouched` | **1 failed, 8 passed.** `Left contains 1 more item: {'restart_epoch.json': '1ace375d...'}` | 9 passed |
| F3 + F27 (whole refusal) | `if not migrate_only:` before `_refuse_run_inputs_the_run_would_delete(...)` changed to `if False:` (`:1992`) | the 4 `test_overwrite_refuses_each_run_input_inside_the_output[...]`, `test_restart_refuses_a_run_input_inside_machine_state`, `test_overlap_refusal_survives_skip_validation` | **6 failed, 3 passed.** `--input`/`--pipeline`/`--metadata`/restart/skip-validation: `assert '<option>' in "... No such file or directory ..."`. `--image-manifest`: `assert 0 != 0`: the run **succeeded** with the manifest deleted (silent loss, as the helper's docstring predicts) | 9 passed |
| F3 + F27 (overwrite half) | `if overwrite and (` changed to `if False and (` (`:619`) | the 4 parametrized cases, skip-validation case | **5 failed, 4 passed** (exactly those five) | 9 passed |
| F27 (restart half) | `if restart and output_dir.exists()` changed to `if False` (`:612`), so no restart targets | `test_restart_refuses_a_run_input_inside_machine_state` | **1 failed, 8 passed** (exactly that test) | 9 passed |
| §0 not skippable | refusal gated `if not migrate_only and not skip_validation:` | `test_overlap_refusal_survives_skip_validation` | **1 failed, 8 passed.** `assert '--input' in "... No such file or directory ..."` | 9 passed |
| Task 1 Step 3 | all of `src/` checked out from `425eb66` (tests present, fix absent) | all 9 | **9 failed in 4.07 s** | 9 passed |

Every test fails when its fix is reverted, and each partial revert fails exactly the tests
aimed at it. No finding.

## Regression spot-check

| Command (all with `QT_QPA_PLATFORM=offscreen uv run pytest -o addopts= -m "not slow" -q -p no:cacheprovider -p no:pytest-qt -n 4`) | Result |
|---|---|
| `tests/unit/cli/test_cli_preflight_ordering.py tests/unit/cli/test_cli_v2.py tests/unit/cli/test_cli_gpu_refusal.py tests/unit/cli/test_run_identity.py tests/unit/cli/test_cli_metadata_startup.py tests/unit/sdk_/test_io_constants.py tests/unit/cli/test_cli_image_manifest.py tests/unit/cli/test_cli_mode_contract.py tests/unit/cli/test_scanner_stores.py tests/unit/cli/test_process_format_cli.py tests/unit/cli/test_cli_store_options.py tests/unit/cli/test_cli_provenance_original.py tests/unit/cli/test_schema_gate.py tests/unit/cli/test_embedded_measurement_migration.py tests/unit/plotting/test_backends.py` (the five required files plus the rest of plan Task 2 Step 6's list) | **515 passed, 1 skipped, 6 xfailed** in 69 s |
| `tests/unit/sdk_/test_verification_cache_disk.py tests/unit/cli/test_staged_controller.py tests/integration/cli/test_cli_store_output.py tests/unit/tune/ -k "task2 or not tune"` | 115 passed, 2 skipped, **17 failed**, all in `tests/unit/tune/test_distributed_finalize_task2.py` (16) and `test_distributed_lifecycle_task2.py` (1) |
| same two tune files with `--tb=line` | all 17: `ModuleNotFoundError: No module named 'optuna'` (one surfaces as `ImportError: Optuna is required for this strategy`) |
| same two tune files with `src/` checked out from `425eb66` | **17 failed, 46 passed, 1 skipped**: identical count at the pre-fix commit |
| `tests/unit/ci/test_startup_imports.py tests/unit/ci/test_deferred_imports.py` (lazy-entry guards; `sdk_/__init__.py` gained two exports) | **236 passed** |
| `uv run ruff check` (no `--fix`) on the six changed `src`/`tests` files | All checks passed |
| `uv run mypy` on `phenotypicCLI.py`, `sdk_/_io_constants.py`, `_cli/_cli_interactive.py`, `_cli/_cli_validation.py` | Success: no issues found |

The implementer's optuna claim holds: the 17 failures are environmental and present before
the fix. **UNVERIFIED:** the full 1422-test / 60-file affected surface and the other 30
importer files were not re-run in full; the files above are a spot check.

## Verified as correct

1. **Everything above the dry-run exit is read-only.** Read every call between the
   freshness refusal and `if config.dry_run:` (`phenotypicCLI.py:2480-2620`):
   `_is_ignorable_output_entry`/`iterdir` (read), `scan_store_outputs`,
   `scan_directory_structure`, `apply_image_manifest`, `organize_by_dataset` (no write,
   mkdir, unlink or rename anywhere in `_cli/_cli_directory_scanner.py`; `Dataset` is a plain
   dataclass), `validate_execution_config` and `validate_pipeline` (`exists()`, `from_json`,
   `preflight_plot_backends`), `_display_execution_config`, `_print_dry_run_mutation_preview`,
   `_print_process_only_dry_run_plan`, `execute_dry_run` (echo only).
   **Empirically:** `strace -f -e trace=openat,mkdir,mkdirat,unlink,unlinkat,rename,renameat,renameat2,rmdir,link,symlink,truncate,ftruncate`
   of `python -m phenotypic ... --restart --dry-run`, `--overwrite --dry-run` and plain
   `--dry-run` against a real prior run: **zero** write-mode opens, mkdirs, unlinks or renames
   under `--output` (only `O_RDONLY` opens of `.phenotypic`, `results` and a missing
   `slurm_lifecycle.json`), and a `find -printf '%p %s %T@'` snapshot of the tree is
   identical before and after each. `--restart --dry-run` on a nonexistent output leaves it
   nonexistent.
2. **Every mutation is below the exit, in the original relative order:** restart clear
   (`:2626`), `rmtree` guarded by `will_overwrite` (`:2646-2650`), `mint_run_identity` (`:2678`,
   still below the `rmtree` as its comment requires), `_run_initiation`, sampling,
   `output_dir.mkdir`, `_prepare_incremental_startup`, strategy creation.
3. **Nothing the moved blocks need was computed below them,** and nothing below needs a
   value only the old position produced: `config.run_initiation` is read only at `:3171` and
   `:3228` (after the exit); no dry-run printer reads the minted identity.
4. **Non-dry-run paths are unchanged in effect.** `continuing` is false under `--overwrite`,
   so `will_overwrite` has the old guard. Resume: no mutation moved relative to the resume
   sites. Restart: clear still precedes the mint (counter preserved). Measure: rejects
   `--restart`/`--overwrite`/`--dry-run` before this code, `scan_store_outputs` moved
   above an unchanged mint. Process: dry-run plan unchanged, preview prepended. Recompile and
   migrate exit before the reordered region (see A2 for the one change that reaches
   recompile). `--sample` and `--image-manifest` selection are unchanged (sampling still
   after the mint; manifest snapshot, subset-change refusal and scan order unchanged). SLURM:
   no submission, `sinfo` or `sbatch` call moved above the exit. The "output inside input"
   layout scans the same set before and after the delete, because the scanner reads only one
   level of subdirectories and the CLI writes no image file at the output root. **UNVERIFIED
   on a real Slurm cluster.**
5. **`_refuse_run_inputs_the_run_would_delete`:** `None` paths skipped; relative spellings
   resolved against the cwd and refused (probed); a nonexistent output yields no targets;
   `--restart` with `.phenotypic/` absent yields no targets and no refusal, and the preview
   reports 0 entries; preserved entries (`legacy-v2/`, `restart_epoch.json`,
   `terminal_failures.jsonl`) are admitted (probed: `--restart` with `--pipeline` in
   `.phenotypic/legacy-v2/` exits 0 and the file survives); migrate is exempt; the call sits
   outside every `skip_validation` branch (revert-proven). Symlinks: see A1.
6. **`machine_state_restart_targets` matches the old deletion set** for every real layout
   (same children minus `_PRESERVED_ON_RESTART`, broken symlinks included via `iterdir`,
   same three legacy paths gated on `exists()`, same `removed` value, same empty-cache
   `rmdir`); see A5 for the pathological exceptions. Mutating its preserved filter fails
   two existing tests.
7. **Plan Task 2 Steps 2, 4, 5:** the preview prints count and paths ("would delete 3
   entries", "would clear 8 machine-state entries ... (kept: legacy-v2, restart_epoch.json,
   terminal_failures.jsonl)"); the kept names come from the new public accessor rather than
   the private constant; the "gate finding F8" paragraph is rewritten to say a dry run never
   reaches the mint; `full_validation` is deleted and no caller remains in `src`, `tests` or
   `docs/source`; the `_ANNOUNCED_PLOT_WARNINGS` comment and the `test_backends.py:307`
   docstring are updated.
8. **Existing docs stay true:** `src/phenotypic/_cli/CLAUDE.md:194-205` (GPU refusal above
   the clearing and the dry-run exit) and `docs/source/contrib_guide/tracked_state.md`
   (`clear_machine_state` semantics) need no change for Phase A.
