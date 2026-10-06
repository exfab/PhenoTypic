# Plan review, round 2: pseudo-cropping

- **Reviewed:** spec and plan at `ef7a1d8f`, against round 1 (`plan-review.md`, written at
  `f043931d`) and the code in worktree `worktree-pseudo-cropping`.
- **Scope:** I checked whether each round-1 fix is correct against the code, and looked
  for new problems the fixes introduced. I did not revisit the settled decisions: v4
  for a no-op crop, `PadImage` fill, ROI-sized builder previews, and null offsets plus
  a warning.
- **Reviewer:** independent plan reviewer. This file is my only write.

## Verdict: READY WITH FIXES

Both round-1 blockers are fixed correctly:
- **B1.** The Colony grid's Viv cells now carry `cropOffset`.
- **B2.** The sentinel tests now use `max(READABLE)+1`.

The re-anchoring of Task 9 is correct. The `tiles.py` rework is sound: it computes the
window in ROI coordinates and shifts only the read.

What remains is three tests that fail for reasons unrelated to the feature, and one
logic gap the M5 fix introduced: false-null offsets for images read from a store with
`imread`. All four are cheap to fix.

**Counts:** BLOCKER 0 · MAJOR 4 · MINOR 6 · NIT 5

---

## MAJOR

### R2-M1. Task 8's own test `test_require_readable_store_accepts_v4_refuses_v5` fails once the task is implemented

**Evidence.**
- The test's `_store(tmp_path, 4)` writes `{"phenotypic": {"store_schema_version": 4}}`
  with no `crop_frame` (plan lines 1664-1671 and 1696-1697).
- Task 8 also adds `check_crop_frame_consistency(block)` as the last statement of
  `require_readable_store` (plan line 1853).
- That function raises when `padded != (version == 4)` (lines 1927-1934).
- So a v4 store with no frame is exactly what the new read invariant refuses. Task 8's
  own `test_inconsistent_crop_frames_are_refused` parametrises
  `{"store_schema_version": 4}` as invalid (line 1730).

**What breaks.** Step 4 ("all PASS") fails. An executor may weaken the invariant to
get it green.

**Fix.**
- Have `_store()` accept an optional block. For the v4 case, write a consistent block:
  `crop_frame={**_FRAME, "padded": True}`.
- Keep the v5 case bare. The version check fires before the consistency check.

### R2-M2. `test_builder_intermediates_stay_roi_sized` finds no stores, so it fails before it checks anything (Task 9)

**Evidence.**
- `CropImage` is an `ImageCorrector`.
- `_layers_modified_by` returns all four layers for a corrector
  (`_image_pipeline_core.py:120-121`).
- `apply_with_intermediates` names that snapshot `base_{i:02d}.ome.zarr`
  (`_image_pipeline_core.py:1076-1080`). Only delta stores are named
  `{i:02d}_{key}.ome.zarr`.
- The test globs `*crop*.ome.zarr` (plan line 2073), which matches nothing, so
  `assert stores` fails. This happens with or without the 3f change.
- A plain `base_*` glob would not work either: it also picks up `base_00.ome.zarr`,
  the pre-crop input, which is legitimately canvas-sized.

**Fix.** Assert on every `*.ome.zarr` except `base_00.ome.zarr`. That covers at least
`base_01.ome.zarr`. Also assert the list is non-empty, so the test cannot pass
vacuously.

### R2-M3. The README test hits the generator's no-measurers early return, so it fails after implementation (Task 7)

**Evidence.**
- `_generate_measurements_section` returns "No measurements configured in this
  pipeline." when `not self.pipeline._meas`, before any Object, BBOX or FRAME table is
  built (`_cli/_cli_readme_generator.py:154-157`).
- The test builds `READMEGenerator(..., pipeline=ImagePipeline())`, which has no
  measurers (plan line 1451).
- So the `Frame_OffsetRR` assertion can never hold.

**Fix.** Use `ImagePipeline(meas=[MeasureSize()])`. Optionally also assert that the
FRAME table appears after the BBOX table.

### R2-M4. `imread` of a store produced by a crop pipeline gives false-null offsets and misleading advice (the M5 fix; Tasks 7 and 10)

**Evidence.**
- `_imread_store` copies the store's journal onto the fresh image
  (`_image_io_handler.py:895-899`). It sets no frame, and returns the on-disk
  **canvas**, which is the original frame.
- `journal_records_crop` scans every application in that journal (plan lines
  1482-1493). It finds the earlier `CropImage`, so `_frame_offsets_for_info` returns
  `None`, and every row gets null `Frame_*` plus the warning "Re-run --mode full to
  recover them".
- This is a documented workflow. Root `CLAUDE.md` says process output "is valid input
  to the normal CLI" and "a tree of stores is valid `--input`", with the journal
  retained.
- For such an input the true offset is **0**: the pixels *are* the canvas. The advice
  is also wrong, because the run producing the nulls *is* a `--mode full` run.
- An unpadded `crop_frame` store (an object crop, or a `pad_on_save=False` save) read
  through `imread` has the same problem, even though its offset is recorded in the
  store.

**Fix.** In `_imread_store`, when the block carries a parseable `crop_frame`:
- **padded:** set an identity frame, `CropFrame(canvas_shape, (0, 0))`, with
  `_pad_on_save = False`. The image is the canvas, so offsets are 0 and slicing
  composes correctly.
- **unpadded:** restore the recorded frame, with `_pad_on_save = False`.

Pin both cases with a test: `imread` of a padded store followed by `objects.info()`
should give `0, 0` and no warning.

---

## MINOR

### R2-m1. `ngff_`'s lazy import of `phenotypic._core._crop_frame` loads the whole image stack into light callers (Tasks 8 and 15)

There is **no import cycle**. The import happens at call time, `ngff_` is fully
initialised by then, and `_crop_frame` imports only numpy.

The cost is elsewhere. Importing any `phenotypic._core.*` submodule runs
`phenotypic/_core/__init__.py`, which imports `_image_parts` and `_pipeline_parts`,
the full image stack. The new calls sit on paths that never loaded that stack before:
- `require_readable_store` and `valid_staged_store`, called by the GUI chunk routes
  (`results_viewer/_zarr_routes.py`, `_measurement_routes.py`, `_shared/tiles.py`,
  `builder/_preview_zarr_routes.py`) on their first request;
- `valid_staged_store`, called by `resolve_run_state` →
  `datasets_needing_migration` (`sdk_/_run_state.py:1539` →
  `_io_constants.py:1804-1818`) whenever a tree still holds `.h5` results.

The startup guards (`test_startup_imports`, `test_hub_startup_imports`) will not catch
this, because it happens at request time.

**Fix.** Put `CropFrame` and `crop_frame_to/from_attribute` in a light `sdk_` module,
for example `sdk_/_crop_frame_attr.py`. `_core` already imports from `sdk_`, so have
`_core/_crop_frame.py` re-export or import them from there. `ngff_` then imports a
sibling, not `_core`.

### R2-m2. The display-range ROI window rounds outward, so the zero margin still leaks in at odd boundaries (Task 15)

**Evidence.**
- The image pyramid is a 2× block **mean** (`ngff_.py:140-142`).
- The window takes the floor of the start and the ceiling of the end (plan lines
  3170-3173). Wherever the scaled offset is not an integer, the boundary level pixel
  averages ROI pixels with zeros.
- The shipped spimager crop offsets `(600, 650)` give 650/4 = 162.5 at level 2 and
  650/8 = 81.25 at level 3. At level 3, pixel 81 covers canvas columns 648-655, two of
  which are zero, so it carries only 6/8 of the true value. That drags `lo` down by up
  to about 25%, which is the same washout M2 was about, only smaller.
- The plan's test uses even offsets (50, 60 on a 2-level pyramid), so the scaling is
  exact and the test cannot see this.

**Fix.**
- Round **inward** by default: ceiling for top/left, floor for bottom/right. Fall back
  to outward only if inward would leave the window empty.
- Add a case with odd offsets on a ≥3-level pyramid, for example `top=51, left=61` on
  an image 1100 px or larger.

### R2-m3. Warn-once is per instance, so copies re-warn, and the per-image text defeats warning de-duplication (Task 7)

`_unknown_frame_warned` is an instance attribute. `_set_from_class_instance` does not
copy it, so every `copy()` warns again. Every `apply(inplace=False)` makes a copy, and
refiners, grid finders and probes all call `info()` on copies.

The message embeds the image name. Python's default warning filter de-duplicates by
message text, so nothing collapses. A `--mode measure` over a 6,657-image pre-change
crop tree prints at least one warning per image.

**Fix.** Choose one:
- Use a constant warning text, so the registry de-duplicates it, and send the per-image
  name to `logger.info`.
- Emit one per-run summary from the CLI.

### R2-m4. `journal_records_crop` deep-copies and validates the whole journal on every uncropped `info()` call, and does not catch every error (Task 7)

**Cost.** For any image without a frame, which is every uncropped image, each `info()`
call runs `readonly_operations`. That function runs `_operations` → a full
`validate_provenance_journal` → `deepcopy` and freeze of every operation, parameters
included (`_provenance.py:302-335`). `GridAccessor` alone calls `self.info()` from
about 15 sites, and grid finders and refiners call it again. That is many full journal
copies per image, on the path that is not cropped at all.

**Robustness.** For a v1 journal, `_operations` returns the raw list. A non-mapping
entry then raises `AttributeError` on `.get`, which the plan does not catch (it
catches only `KeyError`/`TypeError`/`ValueError`). That exception would escape from
`info()`.

**Fix.** Walk without copying:
- `journal.get("applications")` → `operations` for v2, or `journal.get("operations")`
  for v1;
- check `isinstance(Mapping)` on each entry;
- do not validate. The docstring already says "must never raise from `info()`".

### R2-m5. The aggregation pin tests the legacy external-Parquet path, not the forward master path (Task 7)

`test_aggregation_of_old_and_new_tables_keeps_frame_nullable` calls
`aggregate_parquet_files`. Forward runs build the master through
`aggregate_embedded_measurement_tables` → `project_embedded_measurement_table`
(`_cli_parquet_agg.py:177, 361-409`). That path first projects each store's table onto
its own descriptor's `measurement_columns`, and only then runs a `diagonal_relaxed`
concat. The spec claim (§4.2) is about that path.

**Fix.** Build two stores with embedded tables, one old (no `Frame_*`) and one new, and
call `aggregate_embedded_measurement_tables`. Alternatively, test the projection plus
concat helper directly.

### R2-m6. `test_uncropped_info_carries_zero_offsets` raises warnings to errors around unrelated work (Task 7)

The test loads the synthetic plate (a PNG imread) *inside*
`warnings.simplefilter("error")`. Any third-party warning raised during that load
fails the test for a reason unrelated to the feature. The round-1 probe already showed
a PNG/EXIF message on this load. It was not a `warnings` warning, but nothing
guarantees the load stays silent.

**Fix.** Load the plate before the `catch_warnings` block and wrap only the
`objects.info()` call. Alternatively, use `pytest.warns` in reverse: assert that no
`UserWarning` matching "unknown" was recorded.

---

## NIT

- **Task 12 file list:** it still says `test_figures_in_store.py:268 (== 3 → >= 3)`,
  but Step 3 now says `== 4`. Fix the bullet.
- **Task 5 Step 2:** it says "The other two pass already". There are now three pins,
  including `test_capture_does_not_mutate_the_callers_stale_input`.
- **Spec §4.2 naming:** the spec still names `_append_frame_offsets(info, image)`. The
  plan implements `append_frame_offsets(info, offsets)` plus
  `ImageDataManager._frame_offsets_for_info()`. Align the names.
- **`FRAME.desc`:** "0 when the image was not cropped" overclaims for journal-less or
  legacy stores, where it really means "no crop recorded". Suggest the wording "0 when
  no crop is recorded".
- **Slicing a frame-less image that was cropped** (for example `objects[i]` on a store
  re-measured from before this change): `_assign_child_frame` falls back to
  `CropFrame(own_shape, (0, 0))`. The child then reports offsets relative to the
  **ROI**, not the original frame, and they are non-null. This is an edge case. Either
  document it, or give no frame when the parent's journal records a crop.

---

## Round-1 findings: verified as fixed

| Round 1 | Status | Evidence |
|---|---|---|
| B1 Colony Viv | **Fixed** | `build_source_spec` gains `cropOffset` from `ngff_.padded_crop_offset(block)`. `_viv_cell_payload` adds it before serialising. `viv_viewer.js` has no strict spec validation, so an extra key is harmless (`dtypeDomain` contrast, `:296-314`). `test_store_source.py:78` pins the key set, and the plan says to add `cropOffset` to it. The objmap-layer spec mutation in `_viv_source_spec` (`_grid.py:526-531`) keeps `cropOffset`, which is correct because the label is padded too. |
| B2 sentinel tests | **Fixed** | They move to `max(READABLE)+1`. The message assertions still hold: `"3"` and `"5"` both appear in the new wording. |
| M1 Task 9 anchor | **Fixed** | All edits are now in `_write_store_part` (`:1243`), with `pad_on_save` threaded through `_save_store`. |
| M2 display range | **Partly fixed** | The window approach is right. Rounding at odd boundaries remains (R2-m2). |
| M3 Review Focus | **Fixed** | Items replaced. The pickle premise was dropped. |
| M4 work-id test | **Fixed** | `rebuilt_without_key` uses `tracker.file_sha256`, which `_cli_failure_tracker` imports and uses itself, and `tracker.processing_configuration_digest`. The relative path matches `_normalized_input_relative_path`. |
| M5 null offsets | **Fixed, with one gap** | The in-memory rule and the store test are right. `imread` is the gap (R2-M4). |
| MINOR-1/2/3 consistency | **Fixed** | `check_crop_frame_consistency` runs in `require_readable_store` and `valid_staged_store` (after the version gate; `ValueError`/`AttributeError` → `False`). The loader checks the canvas against the level-0 extent and checks `roi_shape` after the reads. The `valid_staged_store` extent check compares against `aligned_spatial[0]`, which excludes `original`. |
| MINOR-4 README | **Fixed in code, test broken** | See R2-M3. |
| MINOR-5 aggregation | **Pinned on the wrong function** | See R2-m5. |
| MINOR-6 pickle | **Fixed** | Plain tuples in the dict. `.get` on load. `CropFrame(*stored)` restores. |
| MINOR-7 builder previews | **Fixed in code, test broken** | 3f covers all five writers (two `save2zarr`, three `save_intermediate_zarr`). The test glob is wrong; see R2-M2. `apply_with_intermediates`' only callers are the builder (`_preview_cache.py:386`, `_callbacks.py:6090`). |
| MINOR-8 capture mutation | **Fixed** | The capture reads `_crop_frame` and tests `.fits()` without clearing. The new pin passes: `copy()` copies the stale frame raw, the carry rejects it, and the input is untouched. |
| MINOR-9 real merge site | **Fixed** | `MeasureFeatures.measure(include_meta=True)` is pinned, and `_merged` is rejected. |
| MINOR-10 edge crops | **Fixed** | See the `tiles.py` trace below. |
| MINOR-11 surfaces | **Fixed** | `tests/unit/prefab`, `test_provenance_fencing.py`, `test_cli_provenance_durability.py` and `tests/gui/results_viewer` were added. |
| N1 witness `assert` | **Fixed** | `_check()` raises explicitly. |
| N3 dead JSON branch and cache | **Fixed** | Removed. `pipeline_uses_crop_frame` is now uncached, and the plan states why. |
| N5 original series | **Fixed** | Retained before the crop, then asserted canvas-sized and equal to the plate (`load_layer_zarr(store, "original")` returns `(C, H, W)`, and the test moves the axis). |

### Specific checks requested

- **Import cycle from `ngff_` → `_core._crop_frame`.** There is none, because the
  import runs at call time. There is a cost, though: R2-m1.
- **`_crop_store_layer_window` variable flow, and `_finish_crop`'s bbox frame.**
  Correct.
  - When the store is padded, `(src_height, src_width) = roi_shape` and
    `(d_row, d_col) = offset`.
  - `_crop_window` clamps to the ROI.
  - Only `read_window` is shifted, and `_read_objmap_window` reuses the same shifted
    window, so the labels align with the pixels.
  - `window.paste_offset` and `_finish_crop`'s keep-rectangle both use the unshifted
    ROI-frame `window` with an ROI-frame `bbox`. So a padded crop is byte-identical to
    an unpadded one, edges and contours included, and the per-window `detect_mat`
    normalisation sees no zeros.
  - The unpadded branch is unchanged.
- **`image_display_range` window arithmetic.** It is right for exact scaling, and the
  plan's test hits exactly that case. It is wrong by one boundary pixel when scaling is
  inexact under mean downsampling (R2-m2). `_read_store_level`'s `window` order
  `(top, bottom, left, right)` matches.
- **`_frame_offsets_for_info` warn-once across `copy()`.** The flag is not carried, so
  each copy warns again. See R2-m3.
- **Does `readonly_operations` raise on v1 or legacy journals?** For v1, no: `_operations`
  returns the list without validating. For v2 journals containing `legacy`
  applications, no: they validate. A malformed v2 journal raises `ValueError`, which is
  caught. An `AttributeError` from a non-mapping v1 entry is not caught. See R2-m4.

### Other plan claims verified

- `READMEGenerator(config, pipeline)` exists (`_cli_readme_generator.py:24-39`), and
  `_generate_measurement_table` already handles identity enums (`BBOX`).
- `aggregate_parquet_files(file_paths, path_to_dataset, ...)` exists at that signature
  (`_cli_parquet_agg.py:39-44`).
- The test's `pytest.warns(match="unknown")` matches the warning text ("…offset into the
  original image is unknown…").
- `append_frame_offsets` with positional `pd.array([...], dtype="Int64")` aligns by
  position, whatever the index is.
- `build_source_spec` already holds `block` and `ngff_` in scope
  (`_store_source.py:97-137`).
- `image_display_range` already holds `block` (`tiles.py:591`).
- The `valid_staged_store` cases in the Task 8 test trace correctly:
  - v4 without a frame → refused;
  - v3 claiming padded → refused;
  - garbage → refused;
  - a consistent block with a 10 000² canvas → fails the extent check.
- Task 10's extent check uses `store_level0_shape(path, series["gray"])`, and `gray`
  is always written. Its `roi_shape` check after the reads catches zarr's silent
  truncation. Round 1's probe confirmed that zarr returns `(3, 2, 4)` for an
  out-of-bounds slice.
- Task 14's harness: Stage 1's initial checkpoint store is uncropped v3 and the final
  Stage 1 store is padded v4. `valid_staged_store` passes both: the extent check
  compares the 600×800 aligned extent with the 600×800 canvas.
