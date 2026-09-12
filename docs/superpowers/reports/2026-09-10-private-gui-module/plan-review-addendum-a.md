# Plan review — Addendum A (Tasks 6–8)

**Reviewer:** `xander-local:plan-reviewer` (Opus) · **Reviewed:** plan Tasks 6–8 and spec Addendum A at `3d57a69f3` (read-only; probes in temp dirs only) · **Date:** 2026-09-10
Saved by the controller: the reviewer has no write tool.

**Verdict:** Tasks 6 and 8 are sound. Task 7 silently breaks two heatmap tests, and its Step 10 count of `3010 passed` is wrong.

## Critical

1. **Task 7 Step 6 / Files — two heatmap tests would error in setup, and nothing in the plan would notice.**
   - `_seed_master_df_in_output` always passes `total_images=len(_IMAGES)` = 2 (`tests/e2e/gui/test_heatmap_tab.py:127`). The `multi-tp-visible` (`:716-746`) and `no-grid-cols` (`:909-925`) frames list only `_IMAGES[0]`.
   - Running the plan's helper verbatim against the real factories raises `AssertionError: master lists 1 images, fixture declared 2`. The default frame, and a re-seed over an already-published run, both resolve `complete`.
   - It is hidden because the module is `ci_flaky` (`:50-51`), so Step 9 and CI deselect it, and Step 8 only probes the default frame.
   - **Fix:** add the file to Files; pass `total_images=df.select(_DATASET_COLUMN, str(IMAGE.IMAGE_NAME)).unique().height`; add both parametrised frames to Step 8.

## Important

2. **Task 7 Step 9 verifies criterion 10 on only 3 of the helper's 9 callers.**
   - Six have module-level `ci_flaky` marks (`test_deliverables_standalone_e2e:57`, `test_filter_offcanvas:38`, `test_heatmap_tab:50`, `test_qc_review_splitter:43`, `test_qc_tab:53`, `test_radial_triage:42`), and `-m "not ci_flaky"` deselects them, so "no new failures" holds vacuously.
   - Their pixel source also changes. Every fixture now gets an 8×8 store; `has_image_source` becomes true once a store exists (`_output_root.py:881`), and `crop_colony` prefers the store to the overlay (`_shared/tiles.py:835-841`). So bboxes 10..310 would be cropped from an 8×8 black store instead of the PNG overlay, and no probe renders that.
   - **Fix:** a Step 9b that runs those six files with ci_flaky included, before (at Task 6's commit) and after Task 7, and diffs the failure sets.
3. **Task 7 Step 10 will report `3006 passed`, not `3010`.** The surface list comes from `git grep` / `git ls-files`, which only see tracked files, and `test_complete_run_fixture.py` stays untracked until Step 11. The same effect is recorded for Task 1 (plan line 583). **Fix:** add the new path to the list, or stage it first; also expect `3 xfailed`.
4. **Task 8 — the post-seeding path has not run in CI since the run-state change.**
   - The last green smoke-capture run (33939157601, `c656cf60`) contains neither `f4ba81004` nor `eadf0fdf5` (`git merge-base --is-ancestor`).
   - With the fix emulated, build → CLI → seeding gives `completion complete mutations_enabled True advisories ()`, so the hub's "Current" check (`capture_gui_tutorial_screenshots.py:1159`) should pass. The hub bind (`:1159`) and the standalone viewer/analysis captures (`:1810`, `:1901`) are still unverified.
   - **Fix:** a step that runs the full script in a throwaway `git worktree` under `/tmp` (sync, run, remove), or record the path as unverified.

## Minor

5. **Task 8 Step 2:** the `if not source_work_ids` guard duplicates `publish_aggregate_snapshot`'s own "No marker-authorized measurements to publish" check (`_cli_completion.py:1107-1109`). Drop it; keep `state is None`, since `.config` needs it.
6. **Task 7 Step 8** does not exercise `test_scatter_tab::scatter_server` (lines 137-150). A fixture cannot be called directly, so copy its body into the probe. The reviewer did so: it resolves `complete`, mutations enabled.
7. **Task 6:** the guard cannot see a `conftest.py` fixture wrapping `page` (`_is_test_file` excludes conftest). No conftest under testpaths does this today.

## Checks out

- **Every shard-assigned module parses:** all 731 pass `ast.parse` on 3.12.11 and 3.11.13. The only warning, `\.` at `tests/unit/test_ome_zarr_invariants.py:269`, did not surface in pytest from a scratch parse test.
- **The browser scan** finds exactly 4 modules, all owned only by `gui-browser`, including `KNOWN_BROWSER_MODULE`.
- **The workflow regex** matches `run-pytest.yml:130-131`, and M4's replace target exists. Blobs are LF, the working tree is CRLF, and `read_text` normalises both.
- **`scatter_server`** writes all three core files; its master has 2 unique (Dataset, ImageName) pairs, which equals `len(_IMAGES)`.
- **No test depends on the old read-only state:**
  - None of the helper's callers asserts a read-only banner or `incomplete` state.
  - The `to_be_disabled` hits belong to run-console, viewer empty-state and lazy-handoff tests on plain sandboxes.
  - `test_analysis_app:43,86` refers to the run button before a model is added.
- **The `_build_sandbox` switch is safe:** `_write_terminal_manifest` is byte-identical to today's helper body (`conftest.py:89-105`), so plain `fake_sandbox` / `live_server` trees do not change.
- **The new gui test is safe on the `gui-browser` shard:** `tests/gui/results_viewer/__init__.py` exists; 5 siblings import `tests._output_layout`; no basename collision; `cache_root` is a sibling of the output, as `_output_root.py:325-328` requires; nothing in it is 3.12-only.
- **Task 8's source set is correct by construction:** the reader compares `source_set_digest` and the count against `_current_success_work_ids` (`_cli_completion.py:833-842`). The mirror, label and QC rewrites touch neither `work_ids` nor the markers, and the publish comes after them.
- **The CI logs match A2 and A3:**
  - In run 34506368188's smoke-capture job, the only fatal error is the `TypeError` at `:443`. The `Metadata_StrainID` traceback comes before `[cli] done` and is non-fatal (DA4).
  - The e2e job's only failures are the two `test_analysis_app` tests (`2 failed, 101 passed, 60 skipped, 36 deselected`).

## Disposition (controller)

All findings were applied to the plan and spec in `fb09ceaf0`:
- C1 → Task 7 Step 7b plus Files plus Step 8 covering all heatmap frames.
- I2 → an overlay-first helper, a fifth guard test, and Step 9b. The before-run found two ci_flaky-hidden failures, which Task 7 now expects to pass.
- I3 → Step 10 stages the new test first (`3011 passed, 16 skipped, 3 xfailed`).
- I4 → Task 8 Step 6.
- Minor 5 → the guard is dropped.
- Minor 6 → Step 8 covers scatter's seeding.
- Minor 7 → already deferred from Task 6's review.
