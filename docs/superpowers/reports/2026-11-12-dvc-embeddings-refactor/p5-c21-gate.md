# C21 gate: P5 Task 3, in-place process runs

The C21 implementer reported these results and the orchestrator wrote them
here. Logs are in `/bigdata/exfab/anguy344/gate-results/c21/`.

## Design fork: a second pipeline in place

An in-place run keeps its machine state in `<input>/.phenotypic/`. So a
second in-place run with a different pipeline is refused by the ordinary
resume check, which exits 1 with "Pipeline contents changed" (measured in
job `32171671`).

Spec E7 refused `--restart` in place as "would delete inputs". That is
wrong: `--restart` clears only `.phenotypic/` entries.

**User decision (2026-11-18): allow `--restart` in place.** `--overwrite`
stays refused. The resume refusal names `--restart` in place. Spec E7 and
the plan are amended (`abf70b8a9`).

## Commits

- **`866eb6ac5`**: `_rewrite_store_tables` gains three options. Measure mode
  passes none of them.
  - `expected_root_sha256=` is checked on the root it reads and again just
    before `promote_store`. A mismatch raises `StoreRootChangedError` and
    removes the part.
  - `consolidate=` re-consolidates the part.
  - `clear_tables=False` carries every run folder.
- **`529b5b589`**: in-place process runs.
  - `_cli/_cli_process_in_place.py` holds `in_place_refusal`,
    `process_in_place_core`, `level0_digest`, `in_place_input_digest` and
    `InPlacePixelsChangedError`.
  - `ExecutionConfig.in_place`. On the worker side, a run counts as in place
    only when it writes a store.
  - `store_tree_sha256` is factored out of `file_sha256`, with byte-identical
    output.
  - The processing-config digest gains `in_place: true`.
  - Local and SLURM dispatch both go in place.
  - Equality is lifted from the overlap refusal. The in-place refusals sit
    before the deletion-path refusal.
  - `read_process_input` and `build_process_store_figures` are shared with the
    copy core.
- **`777862384`**: docs.
  - `zarr_storage.md`: what an in-place run does, the refusals, `--restart`
    for a second pipeline, and where to run it (a local copy, never the
    bucket or NAS tree that AutoConvertRaw-GC republishes).
  - Both CLAUDE.md files.

## Calls (accepted)

- **A second root change before the swap** fails the image for this
  invocation (exit 3) with no terminal record, so a plain re-run retries it.
- **A level-0 pixel change** is a terminal scientific failure
  (`InPlacePixelsChangedError`).
- **An `--input` naming one store or file** is refused, so `.phenotypic/`
  never lands inside a store.
- **The worker requires a store export to count as in place**
  (`writes_process_store`). This was the fix for 9 `--layer objmap` reds,
  where a hand-built config with equal paths had taken the in-place work id
  of a `.tif`. The CLI still refuses a flat export with equal paths before
  any worker runs.
- **A pipeline that adds nothing.** With no runnable embedder and no figures,
  it writes nothing and warns. A figures-only pipeline writes the figure run
  and the journal; that is what `clear_tables=False` is for.
- **`upstream` is carried as-is in place.** Carried run entries are not
  re-stamped, and the new entry binds the stored level-0 digest.
- **Continuation in the core.** The core returns before reading the image
  when the store already has the owned folder and every entry is
  `stale: false`.
- **The pre-swap check** reads the live root after consolidation. The window
  between that read and the rename is not covered, and the docstring says so.
- **The input scanner is unchanged.** `.phenotypic/` and `*.part` yield no
  input because of what they contain, not their names. A test pins this.
- **Fixtures.** `process_store_tree` removes the copy run's `.phenotypic/`, as
  an AutoConvertRaw-GC tree has none. Every behaviour test runs in a `toy`
  and a `dino` flavour.

## Verification

- **Red first** (job `32171802`, against `24cb36893`):
  - shared env: `29 failed, 21 passed, 24 skipped`;
  - torch env: `51 failed, 23 passed`.
  - The failures were every new test in `test_process_in_place.py`, plus the
    4 new preflight-ordering cases.
  - The two tests added after the ruling (`--restart` touches no store; a
    second pipeline needs `--restart`) were not seen red.
- **The first focused run** (focus1) had 13 distinct reds, triaged and fixed:
  - 9 objmap and staged-GPU exports: see "the worker requires a store
    export" above;
  - 2 overlap tests: equality is now in place;
  - the root-change test: its test double touched the wrong store.

  The re-run, job `32199255`, gave `104 passed, 25 skipped` in the shared env
  and `129 passed` in torch.
- **The final focused set at `777862384`**, 142 files (jobs
  `32199378`–`32199385`):
  - shared env: 673, 656, 1956 and 717 passed, 0 failed;
  - torch env: 692, 655, 1984 and 748 passed, with 2 reds.
- **The 2 torch reds are load flakes.**
  - `test_concurrent_process_appends_do_not_lose_records` failed only in a
    cold first round in isolation (35 s, against its 20 s join).
  - Same node, interleaved with the pre-C21 `abf70b8a9`: all 6 rounds passed.
  - `test_ordinary_store_build_is_unlocked_but_promotion_is_locked` passed in
    all 9 isolated rounds.
- **Torch skips:** the 24 `[dino]` cases skip without torch, and each has a
  `[toy]` twin.
- **ruff:** clean.
- **mypy:** the touched files have 469 errors before and 469 after. Only the
  pre-existing `_measurement_tables.py` `union-attr` is in a touched file. The
  new module has 0.
