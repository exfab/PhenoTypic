# P5 deep review: process-mode measurers and in-place runs

Reviewer: a fresh `implementation-test-reviewer` (Opus), over
`git diff 3790f9448..777862384 -- src tests`. The review came back as three
messages, and the orchestrator wrote them here.

Repro tests and logs are in `/bigdata/exfab/anguy344/gate-results/rev-p5/`
(`test_rev_p5_repros.py`). Each repro asserts the correct behaviour, so a
failing repro is a confirmed defect.

Baseline, the 6 P5-touched test files (job `32202190`):
- shared env: `108 passed, 30 skipped`;
- torch: `138 passed`.

## Critical

### C1. An input dataset folder named `progress/` is moved into `.phenotypic/`, where `--restart` deletes it

- **Where:**
  - `sdk_/_io_constants.py:1475-1520`, `migrate_legacy_machine_state`, called on
    every run by `load_processing_state`;
  - `machine_state_restart_targets` (`:1670-1677`), which lists the legacy
    root-level `progress/`.
- **Why it happens:** in place, `<output>` is the user's input tree, so a legacy
  root-level name is user data.
- **Repro** (`test_in_place_keeps_an_input_dataset_named_progress`): exit 3, and
  both stores end up at `.phenotypic/progress/…`. The restart target list is
  `['.phenotypic/processing_state.json', 'progress']`.
- **Why the guard misses it:** `_refuse_run_inputs_the_run_would_delete` only
  asks whether `--input` lies inside a target.
- **Copy mode:** the same flaw already exists there, but it costs only derived
  output.

## Important

- **I1. In place over a full run's bundle stores.** It is accepted, and it
  invalidates that run. The repro shows the run going from `complete` to
  `incomplete`.
  - The journal append strips wall-clock fields from *earlier* applications.
  - A later `--mode measure` keeps stale consolidated metadata (found by
    reading).
- **I2. A symlinked input store** is replaced by a real directory, and a
  dangling `.trash` symlink is left (repro). Across filesystems, every chunk is
  copied in full.
- **I3. Legacy integer `gray` stores** fail the level-0 check every time, with
  the wrong remedy. `imread` normalises the store to float32, and the cast back
  truncates it (mechanism repro).
- **I4. The SLURM in-place path has no test.** That path covers the worker's
  in-place derivation, its identity, its expected-input check, its dispatch, and
  the worklist digest. A revert of the worklist digest to `file_sha256` would
  fail every in-place array task, and no test catches it.
- **I5. A crash between `promote_store`'s two renames** leaves the input store
  only as `.<name>.<uuid>.trash`, which the scanner ignores (found by reading;
  the window is microseconds).

## Minor

- **M1.** `_owned_run_is_current` is true for an owned folder with no
  embeddings, and never checks the pipeline's own keys.
- **M2.** Replacing a stale owned folder loses its non-process entries.
  Reachable only on bundle stores (closed by I1).
- **M3.** A corrupt root in the per-image skip check raises outside the
  scientific boundary. It is never terminal, so it is retried forever.
- **M4.** `--durable-writes` is inert in process mode, copy and in place.
  Pre-existing.
- **M5.** The skip warning is logged once per image in the core.
- **M6.** `.part` debris from a crashed in-place run stays in the input tree.
- **M7.** `_refuse_in_place_run` scans without the `--ext` filter, so an
  unrelated file is blamed.

## Test quality

- **The toy embedder ignores pixels.** Without torch, nothing catches an
  embedding computed from the wrong pixels or layer.
- **Two revision tests assert only `== 5`.** Nothing checks that a revision-4
  work id no longer matches.
- **`test_pixel_arrays_are_never_rewritten` checks `rgb/` only.** Folder 1 is
  checked by bytes, not by inode.
- **`test_the_rewrite_consolidates_on_request` never takes the
  drop-the-stale-block branch.**
- **The fixtures lack real data shapes:** uint16 RGB, `figures/` from
  `CalibrateColorRpcc`, a gray store, a bundle store, and a tree that still
  holds a copy run's `.phenotypic/`.

## Checked and correct

- **Refusals:** the order matches the plan, sits above every mutation, and
  holds outside `--skip-validation`.
- **Detector and corrector checks:**
  - `ObjectDetector` is caught in any slot, at any depth.
  - `ImageCorrector` is caught anywhere in `ops`, nested composites included.
  - `PixelNudgingOp` fails on the level-0 digest; the test is real.
- **Transaction:** retry once, then fail unwrapped, leaving no part.
- **Identity:** `in_place_input_digest` and the `in_place` key are correct, and
  every work-id producer goes through them.
- **Records:** they are republished after the write.
- **`writes_process_store`:** it gates both callers.
- **Refusal error type:** `ProcessMeasRefusedError` bypasses the wrapper.
- **Torch OOM:** never terminal.
- **Measure mode:** unchanged.
- **E4 and E7:** implemented as decided, apart from C1.

## Open questions for the user

1. **Q1 (I1).** Accept a full run's bundle stores in place? The options:
   - (a) refuse any store with `image_class` (recommended);
   - (b) allow them, fix the append and re-consolidation, and document that the
     run's proof is invalidated;
   - (c) allow them and republish the owning run's records.
2. **Q2 (C1).** Legacy machine-state names at the input root. The options:
   refuse in place (recommended), or stop treating those names as legacy state
   in place.
3. **Q3 (I3).** Legacy integer gray stores. The options: compare the image
   before and after `apply()` (supports them), or refuse them.
4. **Q4 (I2).** Symlinked stores. The options: refuse (recommended), or follow
   the link and write beside the target.

## User decisions (2026-11-18)

- **Q1:** refuse a full run's bundle stores (root `image_class`) in place.
- **Q2:** refuse in place when a legacy root-level machine-state name exists
  at the input root.
- **Q3:** support legacy integer gray stores. The level-0 check compares the
  exported layer before and after `apply()`.
- **Q4:** refuse a store that is a symlink or resolves outside `--input`.

Spec E7 is amended to match.

## Orchestrator triage

The review-fix cluster, C21r, fixes the following within the spec:

- **C1, I1, I2:** the refusals above.
- **I3:** the before/after compare.
- **I1 independently:** the journal append keeps earlier applications
  byte-for-byte, and `_rewrite_store_tables` re-consolidates whenever the root
  it read carried `consolidated_metadata`.
- **I4:** SLURM in-place tests (the worklist digest, and the worker's
  in-place dispatch).
- **I5:** refuse an orphaned `.trash` that has no live store beside it.
- **M1, M3, M5, M7.**
- **M6:** a startup warning that lists orphaned in-place `.part` debris.
- **Test quality:**
  - a pixel-dependent toy embedder;
  - a behavioural revision test;
  - an inode check for the carried folder;
  - fixture variants (uint16 RGB, a gray store, a bundle store, a tree with a
    copy run's `.phenotypic/` left in it).

Recorded only:
- **M2:** closed by I1.
- **M4:** pre-existing; `--durable-writes` is inert in process mode.
- **The copy-mode `progress/` dataset name:** pre-existing, and costs only
  derived output.
