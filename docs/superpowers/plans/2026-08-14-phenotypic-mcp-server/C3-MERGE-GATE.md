# C3 + main-merge gate review

Reviewed at `8587bbf5f` on `feat/mcp-server`, tree clean. Analysis only; no source
was edited. Every mutation ran against an isolated copy of `src/` + `tests/` in
scratch, reached through `PYTHONPATH` (verified in effect: `phenotypic.__file__`
resolved to the scratch copy, and mutations M1/M2/M3 produced failures).

**Verdict: no blockers.** The merge is faithful — every semantic change of main's
is present in the merged tree, and I can account for the provenance of every byte
that differs from an automatic three-way merge. Task 9's re-apply is a clean
refactor on top of main's file. Six improvements follow, the first two of which
are false greens I proved by mutation.

---

## 1. Merge fidelity (area 1) — clean, with a machine-checkable argument

Rather than eyeball 307 changed files, I computed the tree git *would* have
produced unaided and diffed the human's merge against it:

```
git merge-tree --write-tree 7471ae701 9e4159a39   ->  43b5a5e002409f0b37c7a89dc1152844bcf23b46
git diff 43b5a5e00 6007f5a0c^{tree}
```

The two trees differ in **exactly 9 paths**:

| Path | Why it differs from auto-merge |
|---|---|
| `src/phenotypic/_cli/_cli_slurm_array_scripts.py` | conflict — took main's file whole |
| `src/phenotypic/gui/run_console/_state.py` | conflict — kept our shim |
| `src/phenotypic/gui/shell/_runs_registry.py` | conflict — kept our shim |
| `src/phenotypic/gui/tune/_setup_authoring.py` | conflict — kept our shim |
| `src/phenotypic/gui/tune/_space.py` | conflict — kept our shim |
| `src/phenotypic/_services/argv.py` | port target |
| `src/phenotypic/_services/runs.py` | port target |
| `src/phenotypic/_services/tune_spec.py` | port target |
| `tests/unit/cli/test_build_array_script_spec_is_pure.py` | Task 9's test, reverted here and re-added by `81c4a7ae6` |

Everything else in the merge — all of `tests/`, `docs/`, `.github/`, `CLAUDE.md`,
`src/` outside the five conflicts — is **byte-identical to the automatic merge**,
so main's content is present wherever ours did not conflict. That reduces the
audit to those 9 files, all of which I checked line by line.

### The overlap is exactly 6 source files, not more

```
comm -12 <(git diff --name-only c847373c8..9e4159a39 -- 'src/**/*.py')
         <(git diff --name-only c847373c8..7471ae701 -- 'src/**/*.py')
```
returns the 5 conflicted files plus `src/phenotypic/sdk_/_io_constants.py`.

I also checked the subtler failure mode — main edits a file whose body Phase 1
had *copied* into `_services/` (so git sees no conflict but the promoted copy
silently keeps the old semantics). Phase 1's promotion sources are
`gui/shell/_sandbox.py`, `gui/_operation_registry.py`, `gui/run_console/_runner.py`,
`gui/tune/{_command,_export,_run_argv,_validation,_domain_editor}.py` — **main
changed none of them**. The run-console and shell files main *did* change
(`_callbacks.py`, `_form.py`, `_ids.py`, `_request_safety.py`, `_slurm_observer.py`,
`shell/_metadata_context.py`) were never promoted. So the six above are the
complete surface.

### Each port verified against main's intent

- **`resume` → `retry_failures` (7 sites).** Main's diff on
  `gui/run_console/_state.py` is 7 hunks. Five landed in `_services/argv.py`
  (docstring `:70`, field `:92`, `run_state_to_json` `:125`, `run_state_from_json`
  `:172`, `to_argv` `:384`); the two in `state_from_controls` stayed in the shim
  and are present at `gui/run_console/_state.py:118` and `:218`. Corroborated
  globally: `grep -rn -- '--resume' src/` returns only a `CLAUDE.md` sentence
  saying the flag no longer exists, and no branch-new file (`_services/*`,
  `gui/tune/_space_view.py`) carries a stale `resume` identifier.
- **Completion-marker check.** `_services/runs.py:610-640` is a **byte-identical**
  copy of main's addition to `_runs_registry.py:592`. Both names it needs are in
  scope: `import json` at `:33`, `run_completion_marker_path` at `:70`.
- **`METADATA.IMAGE_NAME` → `IMAGE.IMAGE_NAME`.** Main changed 2 sites in
  `_setup_authoring.py` and 2 in `_space.py`; the port covers all four in
  `_services/tune_spec.py` (`:113`, `:305`, `:308`, `:1512`), including
  `_default_qc_scorer`, which is a branch-only copy main could never have reached.
  `grep -rn '\bMETADATA\.' src/` now hits only prose (`FEATURES.md`, one comment);
  `schema/__init__.py:148` keeps a deprecating alias regardless.
- **`_normalize_setup_metadata_groupby`.** Present at `_services/tune_spec.py:1446`,
  re-exported through `gui/tune/_setup_authoring.py:18`. Body matches main's.
- **`_cli_slurm_array_scripts.py`.** `git diff 9e4159a39..HEAD` on this file is
  163+/59−, and every hunk is Task 9's split. Main's identity mechanism is intact:
  the `CURRENT_WORK_ID`/`CURRENT_INPUT_SHA256`/`CURRENT_ATTEMPT_ID` prelude, the
  `identity_rows` loop, the three `EXPECTED_*`/`ATTEMPT_IDS` arrays and
  `EXPECTED_PIPELINE_SHA256` all moved into the builder unmodified.

### `sdk_/_io_constants.py` — nothing lost (the file the brief flagged)

Main's +49/−9 and Phase 1's +58 touch **disjoint regions** (main: markers and
`clear_machine_state` near lines 560–1090 and 1518; branch: `IMAGE_EXTS` and the
sandbox-dir block at ~118), which is why git merged them silently. Both sets are
present at HEAD: `AGGREGATE_PUBLICATION_JSON`, `TERMINAL_FAILURES_JSONL`,
`DIR_IMAGE_COMPLETE`, `terminal_failures_jsonl_path`,
`aggregate_publication_marker_path`, `image_completion_marker_path`, and the
journal-preserving `clear_machine_state` branch (`:1138`), alongside `IMAGE_EXTS`,
`SANDBOX_GUI_DIRNAME`, `SANDBOX_PRESETS_SUBDIR`, `SANDBOX_TUNE_PRESETS_SUBDIR`,
`tune_presets_dir`.

---

## 2. Shim re-export coverage (area 6) — clean

AST-parsed **1512** `.py` files under `src/`, `tests/`, `scripts/` (0 parse
errors), collected every `from phenotypic.* import name` — including multi-line
parenthesised forms and relative imports — and resolved each name against the
live module. **711 modules referenced, 0 import failures, 3 unresolved names**:

| Name | Site | Verdict |
|---|---|---|
| `phenotypic.sdk_.ClipControlMixin` | `tests/unit/sdk_/mixin/test_norm_control_mixin.py:81` | inside `pytest.raises(ImportError)` — deliberate |
| `phenotypic.detect.nn.Sam2Detector` ×2 | `scripts/accuracy_gate_gpu_detectors.py:164,176` | pre-existing at the merge base (`c847373c8` already had it); optional GPU dep |

None is a shim gap. Script: `scratchpad/check_reexports.py`.

---

## 3. Purity of `build_array_script_spec` (area 3) — genuinely write-free

- **Static.** AST walk of the function body (`_cli_slurm_array_scripts.py:149-443`)
  yields calls `{Path, SlurmArrayScriptSpec, ValueError, _array_script_names,
  _build_entry_list, _set_worker_mode, event_log_path, file_sha256,
  get_python_command, logs_dir, str, uuid4, work_id_for_image, .absolute, .append,
  .extend, .join, .quote}` — **no** `mkdir`/`write_text`/`open`/`touch`/`chmod`/
  `unlink`/tempfile.
- **First-level callees.** `_build_entry_list`, `_set_worker_mode`,
  `_array_script_names`, `event_log_path`, `get_python_command`, `work_id_for_image`
  contain no write call. `logs_dir` (`_io_constants.py:1525`), `slurm_scripts_dir`
  (`:1643`) and `phenotypic_cache_dir` (`:925`, "Pure path expression; callers
  ``mkdir`` when they intend to write") are path joins. `file_sha256`'s only
  suspect call is `open` in read mode.
- **Runtime.** `test_build_array_script_spec_writes_nothing` digests the whole
  `output_dir` tree before and after; mutation **M2** (inserting
  `(output_dir / '.leak').write_text(...)` at the top of the builder) makes it
  fail, so the guard is live rather than vacuous.

The reads are as C3 documented: `work_id_for_image` hashes each image *and* the
pipeline JSON, and the `identity_rows` loop hashes each image a second time.

---

## 4. Mutation results (areas 2 and 4)

Six mutations, each applied to the scratch copy and reverted from a saved
original afterwards (final `diff -q` confirmed identical).

| # | Mutation | Result | Reading |
|---|---|---|---|
| M1 | `EXPECTED_PIPELINE_SHA256` made per-call random (a **second** random field) | **caught** — `test_attempt_ids_are_the_only_drift` and `test_generator_and_builder_agree` both fail | the mask cannot silently grow; this is the check the brief asked for |
| M2 | builder writes `<output_dir>/.leak` | **caught** — `test_build_array_script_spec_writes_nothing` | purity guard is live |
| M3 | generator calls a module-private alias instead of the module attribute | **caught** — `test_generator_consumes_the_builder` ("called the builder exactly once", 0 == 1) | consumption guard is structural, as claimed |
| M4 | drop `file_sha256(image_path)` from the identity-row loop (so **every `EXPECTED_INPUT_SHA256S` entry renders empty**) | **SURVIVED** — 6/6 in the C3 file, and 551 passed / 1 unrelated flake across all of `tests/unit/cli` | see Improvement 1 |
| M5 | `ATTEMPT_IDS` gains a bogus extra entry on the 2nd+ call | **SURVIVED** — 6/6 | see Improvement 2 |
| M6 | every `EXPECTED_WORK_IDS` entry replaced with `'CONSTANT-WRONG-WORK-ID'` | **SURVIVED** — 551 passed / 1 unrelated flake across all of `tests/unit/cli` | see Improvement 1 |

C3's three replacement guards do hold under the mutations that target them
(M1–M3). Its own M1–M7 table is accurate as written; M4 and M6 are strictly
sharper versions of its M7, and they are what expose the gap.

**Is the mask narrow (area 4)?** Yes in *location* — the regex
`ATTEMPT_IDS=\(\n.*?\n\)` (`test_build_array_script_spec_is_pure.py:34`) is
anchored on the literal array name, is non-greedy to the first line-initial `)`,
and `subn` asserts exactly one match, so it cannot reach an `EXPECTED_*` array,
an SBATCH directive, or a dispatch arg. M1 confirms this from the other side. It
is *not* narrow in **content**: M5 shows the mask blanks anything inside that
block, including a changed entry count.

---

## 5. Suites and mypy (area 5) — all re-measured, all match

| Target | Expected | Measured |
|---|---|---|
| `tests/unit/cli` + `tests/unit/services` | 552 + 61 = 613 | **613 passed** (311 s) |
| `tests/unit/cli` alone (from the mutation runs) | 552 | **552** (551 + 1 flake) |
| `tests/unit/gui` + `tests/integration/gui` | 1746 / 3 skipped | **1746 passed, 3 skipped** (75 s) |

**mypy — by diff, three trees, all with cold caches.** Baselines were exported
with `git archive` into scratch (no worktree added to the shared repo):

| Tree | Result |
|---|---|
| `7471ae701` (pre-merge branch) | 421 errors / 125 files |
| `9e4159a39` (`upstream/main`) | 417 errors / 124 files |
| `8587bbf5f` (HEAD), cold cache | **417 errors / 124 files** |

Comparing normalised error sets (line/column stripped, deduplicated):

- errors at HEAD present in **neither** parent: **0**
- errors present in **both** parents and absent at HEAD: **0**
- per-file error counts: identical to main for every shared file. The only
  file-level differences are `gui/_operation_registry.py` (2) → `_services/registry.py`
  (2), i.e. Phase 1's promotion carrying its own errors along, and
  `_cli_gui_lifecycle.py` (1), which arrived from main.

Nothing regressed. See Improvement 5 on the reported count.

---

# Improvements (no blockers)

### 1. Main's identity arrays are not pinned by any test — two mutations survive the whole CLI suite
`src/phenotypic/_cli/_cli_slurm_array_scripts.py:383-393`

Emptying every `EXPECTED_INPUT_SHA256S` entry (M4) or filling every
`EXPECTED_WORK_IDS` entry with a constant wrong value (M6) leaves all 552 cases in
`tests/unit/cli` green. `grep -rn 'EXPECTED_WORK_IDS\|EXPECTED_INPUT_SHA256S' tests/`
returns hits only in C3's own file, and only in prose. The worker-side script
compares these arrays to decide whether a task's input still matches what was
planned, so a silent corruption of them is a correctness bug that no test would
report.

This is **pre-existing, inherited from main's `379acee42`** — not introduced by
C3, and out of Task 9's scope. But it is the largest false green in the reviewed
surface, and it is the reason `test_building_a_spec_reads_every_input_image`
cannot carry the weight its name implies:
`work_id_for_image` hashes each image independently, so the test's
`chunk_images <= set(hashed)` assertion stays true even when the identity-row
hash is deleted. Suggested fix: one case asserting the rendered
`EXPECTED_WORK_IDS[i]` / `EXPECTED_INPUT_SHA256S[i]` equal
`work_id_for_image(...)[0]` and `file_sha256(image)` recomputed in the test.

### 2. The `ATTEMPT_IDS` mask hides arbitrary content inside its block
`tests/unit/cli/test_build_array_script_spec_is_pure.py:34,47-51`

M5: making the builder append one bogus entry to `ATTEMPT_IDS` on every call after
the first leaves all six cases green. The mask is correctly narrow in location but
blanks the block's contents wholesale, so entry count and entry shape are
unguarded. Cheap fix: instead of blanking, substitute each entry
positionally (e.g. assert the block has `len(entries)` lines, each a 32-char hex
string, then replace them with a fixed token) so length and shape survive masking.

### 3. `test_building_a_spec_reads_every_input_image` overclaims in its docstring
`tests/unit/cli/test_build_array_script_spec_is_pure.py:163-172`

The docstring says "the identity-row loop hashes the image again, so every chunk
image is opened", but the assertion cannot distinguish the two call sites — M4
proves it. Either narrow the prose to "the builder hashes every chunk image at
least once", or count calls per path (`hashed.count(image) == 2`), which would
also close half of Improvement 1.

### 4. `test_concurrent_process_appends_do_not_lose_records` is flaky under parallel load
`tests/unit/cli/test_cli_terminal_failures.py:115` (a file main added, +337)

Failed in 2 of 3 `-n auto` runs on this 4-core allocation with
`assert process.exitcode == 0` → `None`, i.e. `process.join(timeout=20)` expired
rather than a real assertion failing. Passed in the clean run. Not merge-related,
but it will produce intermittent red on a loaded runner; the 20 s join deserves a
larger budget or a `ci_flaky` marker.

### 5. The reported `416 errors / 123 files` is a warm-cache artifact
Running `uv run --no-sync mypy src/phenotypic` against the repo's existing
`.mypy_cache` reproduces 416/123; the same command with `--cache-dir` pointed at a
fresh directory gives **417/124** — matching main exactly. The extra error is
`detect/nn/_sam3.py:218 Cannot find implementation or library stub for module
named "torch"`, which the warm cache suppressed. The blob for `_sam3.py` is
identical between main and HEAD (`c8e290f8d`), so nothing is wrong with the code;
the *number* in `C3-PROGRESS.md` and in the brief is just measured through a stale
cache. The diff-based conclusion is unaffected. Recommend `--cache-dir` on a
scratch path for any figure quoted in a gate.

### 6. Cosmetics in the ported `tune_spec.py`
- `src/phenotypic/_services/tune_spec.py:1453` imports
  `normalize_metadata_column_reference` inside the function, while line `112`
  already imports `resolve_metadata_csv` from that same module at top level (with
  the import-purity allowlist comment). Main had it at module level. Harmless,
  but the two styles for one module read as an oversight.
- `src/phenotypic/_services/tune_spec.py:1441-1444` — three blank lines before
  `_normalize_setup_metadata_groupby`. `ruff check` passes on all five changed
  files, so this is taste only.
- `src/phenotypic/gui/tune/_setup_authoring.py:18` places
  `_normalize_setup_metadata_groupby` out of alphabetical order in the import
  block and omits it from `__all__` (every call site uses `from ... import`, so
  nothing breaks).

---

## Coverage of the six requested areas

All six were completed. Nothing was skipped or sampled:

1. Merge fidelity — complete, via the `merge-tree` argument plus a hunk-level
   audit of all 9 human-touched files.
2. False greens by mutation — complete; 6 mutations run, 3 caught, 3 survived.
3. Builder write-freedom — complete (static AST + callee audit + live runtime guard).
4. Mask honesty — complete; narrow in location, wide in content (Improvement 2).
5. Suites and mypy — complete; all three counts re-measured, mypy diffed against
   both parents with cold caches.
6. Shim re-export coverage — complete; 1512 files AST-parsed, 3 unresolved names,
   all pre-existing and intentional.

Artifacts: `scratchpad/check_reexports.py`, `scratchpad/purity.py`,
`scratchpad/purity2.py`, `scratchpad/mut_M{1..6}.py`, `scratchpad/mut.sh`,
`scratchpad/mypy_{head_fresh,7471ae701,9e4159a39}.txt`,
`scratchpad/{suite_cli_services,suite_gui,m4_wide,m6_wide}.log`.
