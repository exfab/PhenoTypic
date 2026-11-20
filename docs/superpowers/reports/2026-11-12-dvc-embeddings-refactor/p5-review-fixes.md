# P5 review fixes (C21r)

The findings are from `review-p5.md`, which also records the user's decisions
of 2026-11-18. The C21 implementer reported this round; the orchestrator wrote
it here. Logs are in `/bigdata/exfab/anguy344/gate-results/c21/`.

## Commits

- **`82549d7d6`**: `_rewrite_store_tables` re-consolidates whenever the root it
  read carried `consolidated_metadata`, so measure mode on a consolidated store
  is covered. An unconsolidated store is written as before, and a test pins
  that.
- **`094bc7140`**: the in-place fixes.
  - **Refusal order:**
    1. `--overwrite`
    2. a non-store export
    3. an `ObjectDetector` anywhere
    4. an `ImageCorrector` in `ops`
    5. a single-store input
    6. a legacy root-level machine-state name (C1, Q2)
    7. an orphaned `.trash` with no live store, refused with its `mv` restore
       (I5)
    8. then, per store:
       - a symlinked store, or one that resolves outside the input (I2, Q4);
       - a non-PhenoTypic store;
       - a bundle store, whose root has `image_class` (I1, Q1);
       - a v1 store;
       - a non-primary layer.
  - **In all modes,** `_refuse_run_inputs_the_run_would_delete` refuses a
    `--restart` whose targets contain a scanned input (C1).
  - **I1:** the journal is append-only. Earlier applications are kept as
    stored.
  - **I3:** the pixel check compares the layer after `apply()` with the layer
    right after the read, so legacy integer gray stores work. The grid
    payloads are re-bound to the stored array's digest.
  - **M1:** `_owned_run_is_current` requires every key the pipeline produces,
    each with `stale` false.
  - **M3:** an unreadable root at the skip check is a per-image failure.
  - **M5:** the skip warning is logged once per process.
  - **M6:** leftover `.part` debris is listed, never deleted.
  - **M7:** the refusal checks only the images the run would process, through
    the `--image-manifest` selection. `--ext` names the overlay format, not an
    input filter.
  - **New tests:** `test_process_in_place_slurm.py` (I4). The toy embedder's
    vector now includes the box's mean `gray`. Also new:
    - a behavioural revision 4→5 work-id test;
    - an inode check for the carried folder;
    - fixture variants: uint16 RGB, gray, legacy integer gray, a bundle
      store, and a tree with a copy run's `.phenotypic/`.
- **`ce694ca47`**: docs. The `zarr_storage.md` how-to (the refusals, restoring
  a `.trash`, the append-only journal, the dtype-independent check, `.part`
  debris) and both CLAUDE.md files.

## Calls (accepted)

- **A leftover copy-run `.phenotypic/`** is another run's recorded state. It
  gets the resume refusal (exit 1) plus the in-place line naming `--restart`.
  With `--restart`, the store gains its next folder.
- **The bundle refusal comes before the v1 check.**
- **The new restart-target scan applies to copy runs too.** It refuses only
  when `--restart` would delete a scanned input.

## Verification

- **Seen red first.** Job `32209404` ran the new tests against the source at
  `9237ca791`.
  - Shared env: `25 failed, 42 passed, 31 skipped`. Torch env: `27 failed, 71
    passed`.
  - The failures covered every refusal, the journal append, the gray stores,
    the trash and part handling, the owned-run check, the corrupt root, the
    once-per-process warning and the re-consolidation.
  - Some tests passed at the old code. These pin coverage rather than a
    defect: the SLURM tests, the revision test, the store-shape tests, the
    copy-run-state test and the grid-vector check.
- **Mutants** (job `32213552`):
  - Reverting the worklist to `file_sha256` fails
    `test_the_worklist_carries_the_in_place_input_digest`.
  - Swapping grid rows and columns fails
    `test_a_toy_grid_table_holds_the_cells_vectors`. The old toy embedder,
    which summed box coordinates, was symmetric under that swap.
  - The byte-reproducibility test compares two runs of the same code, so it
    cannot catch this.
- **Touched and new files** (job `32213538`): shared env `135 passed, 31
  skipped`; torch env `166 passed`.
- **The 143-file focused set** at `ce694ca47` (jobs `32213618`–`32213625`):
  - shared env: 691, 665, 803 and 1874 passed;
  - torch env: 714, 668, 803 and 1930 passed.
  - The only red, in both envs, is
    `test_concurrent_process_appends_do_not_lose_records`. It is the known
    load flake, isolated in C21.
- **mypy:** 469 errors before and after. The touched modules have 0, except
  for the pre-existing `_measurement_tables.py` `union-attr`.
- **ruff:** clean.
