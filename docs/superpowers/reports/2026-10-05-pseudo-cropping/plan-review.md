# Plan review: pseudo-cropping

- **Reviewed:** `docs/superpowers/plans/2026-10-05-pseudo-cropping/plan.md` against
  `docs/superpowers/specs/2026-10-05-pseudo-cropping/design.md` and the code at
  `f043931d` (worktree `worktree-pseudo-cropping`).
- **Reviewer:** independent plan reviewer (analysis only; this file is the only write).
- **Date:** 2026-10-05

## Verdict: NOT READY

The in-memory model (Tasks 1–5), the schema (Task 6), the writer and reader shape
(Tasks 9–10) and the work-id key (Task 13) are sound, and most of their code would
work as written. The plan is not ready for three reasons:

1. **The Results Colony tab misplaces every colony on a padded store.** Neither the
   spec nor the plan covers it.
2. **Three existing tests use `STORE_SCHEMA_VERSION + 1` (= 4) as the "future,
   unreadable" version.** They break the moment Task 8 lands, and the plan expects
   those runs to pass.
3. **Task 9 targets the wrong function.** Every edit it lists lives in
   `_write_store_part`, not `_save_store`.

Counts: **BLOCKER 2 · MAJOR 6 · MINOR 11 · NIT 7 · SPEC 6**

---

## BLOCKER

### B1. The Results Colony tab (Viv grid) centres every colony at ROI coordinates on a canvas-sized store. This affects Task 15 and spec §7.

**Evidence.**
- `colony_view/_grid.py:405-410` takes `Bbox_CenterRR/CC` from the measurement
  table as `_centroid_rr/_centroid_cc`.
- `_grid.py:632-641` serialises them as `data-colony-viv-cell`
  `{"centroidRr", "centroidCc", "spec"}`. Here `spec` is
  `build_source_spec(store, …)`, so the cell reads the per-image store.
- `results_viewer/_assets/viv_viewer.js:690-693` documents these values as
  "`centroidRr`/`centroidCc` are STORE pixel coordinates; the view's `target` is
  `[cc, rr, 0]`". They are used at `:383-384` (tile prefetch) and `:561-562` (view
  target).
- `_gui/CLAUDE.md` "Pixel paths": "Results **Colony** | same route, one
  `OrthographicView` per colony".

**What breaks.** After this change the store is canvas-sized and the table stays
ROI-relative. In the mounted Colony tab, every colony's view is off by
`crop_frame.offset`: for the spimager prefab, `(600, 650)` px. The spec's claim
(§7) that "Viv/deck.gl reads store chunks directly … lines up automatically" holds
for the Plate view's image and label layers. It does not hold for the Colony view,
which positions its cameras from table coordinates. Task 15 fixes only the
server-side PNG crop route (`tiles.py`), which is now the QC gallery and overlay
path, not the primary Colony surface.

**Fix.** Choose one:
- Add `cropOffset` (from `ngff_.padded_crop_offset(block)`) to the dict that
  `build_source_spec` returns, and add it in `viv_viewer.js` where the centroids
  are consumed. The source spec already reads the block.
- Add the offset in `_grid.py` before serialising, using the same block that
  `_viv_source_spec` already reads.

Either way, add a test that builds a Colony cell from a padded store and asserts
the serialised centroid equals `Bbox_Center + offset`. Also add
`tests/gui/results_viewer` and the colony-view unit tests to the Phase 5 surface.
This also needs a spec §7 amendment (see SPEC-1).

### B2. Existing tests pin 4 as the "newer, unreadable" store version, so Tasks 8–10 break them. This affects Tasks 8, 9 and 10.

**Evidence.**
- `tests/unit/core/test_image_zarr_roundtrip.py:308-331`
  (`test_load_zarr_raises_on_a_newer_store_schema_version`) and `:334-360`
  (`test_load_layer_zarr_raises_…`) rewrite the store to `STORE_SCHEMA_VERSION + 1`.
  They expect `ValueError("newer PhenoTypic")` and assert that both `"3"` and `"4"`
  appear in the message.
- `tests/unit/sdk_/test_ngff_validity.py:~112-120` writes
  `store_schema_version=ngff_.STORE_SCHEMA_VERSION + 1` and asserts
  `valid_staged_store(store) is False`.

**What breaks.** Once `READABLE_STORE_SCHEMA_VERSIONS = {3, 4}`, version 4 is
readable, so all three tests fail:
- `load_zarr` succeeds instead of raising.
- `valid_staged_store` returns True.

Task 9 Step 4 and Task 10 Step 4 run `test_image_zarr_roundtrip.py` and state
"Expected: all PASS". The executor will hit unexplained red tests and may "fix"
them the wrong way, for example by loosening the gate.

**Fix.** Make this an explicit step in Task 8. Change these tests to use an
unreadable version, `max(READABLE_STORE_SCHEMA_VERSIONS) + 1`, and keep the message
assertions on the new wording. Add the three test files to Task 8 Step 4's run
list. Then sweep `grep -rn "STORE_SCHEMA_VERSION *+ *1" tests` and fix every hit
the same way. The 999 cases in `test_zarr_routes.py:581` and
`test_store_source.py:136` are fine.

---

## MAJOR

### M1. Task 9 edits the wrong function.

**Evidence.**
- `_image_io_handler.py:1151-1241`: `_save_store` only allocates the part and calls
  `self._write_store_part(...)`.
- The `arrays` dict, the `# 1. arrays and chunks` loop, the objmap write, the
  `# 2.` OME loop, `label_shapes`, `build_ome_xml(series_shapes=…)` and the
  `_build_store_attributes(...)` call are all in `_write_store_part`, at
  `:1281-1455`.

**What breaks.** Step 3d says "Directly after the `arrays` dict … is built" in
`_save_store`. No such point exists there. An executor following the text either
stalls or pads in `_save_store` and passes padded arrays nowhere.

**Fix.**
- Add `pad_on_save` to both `_save_store` and `_write_store_part`, and thread it
  through.
- Re-anchor every 3d bullet to `_write_store_part`, with line numbers.

### M2. Server-side colony crops of 16-bit cropped plates lose contrast. This is a missed consumer for Task 15.

**Evidence.** `tiles.py:546-607` (`image_display_range`) takes `min/max` of the
smallest pyramid level of the whole layer. For RGB it feeds `scale_to_uint8`
(`:710-711`) for every non-uint8 crop.

**What breaks.** On a padded store, the zero margin forces `lo = 0`. The project's
own real-store example is 17,290–47,898 (docstring `:558`). Padded, it becomes
0–47,898, so every colony crop of a cropped 16-bit plate renders washed out.

Task 15's equality test cannot catch this: it uses the 8-bit synthetic plate, and
the `arr.dtype == np.uint8` branch at `:708` skips the range entirely. The
per-window `_normalize_to_uint8` used for `detect_mat` crops (`:715`) has the same
problem near the ROI edge.

**Fix.**
- Have `image_display_range` read only the ROI window of the smallest level: the
  offset and `roi_shape` from `crop_frame`, scaled by the level-shape ratio.
  Alternatively, compute it on level 0's ROI window.
- Add a 16-bit padded-vs-unpadded crop equality test.

### M3. Review Focus is missing the two highest-impact risks, and item 3 rests on a false premise. This affects the Review Focus section.

- **Item 3** says "loky/SLURM workers pickle Images". Every CLI worker receives
  paths (`_cli_execution_strategies.py:375`, `_cli_staged_strategy.py:215,352`).
  The frame crosses process boundaries only through the store's `crop_frame`
  attribute. The pickle test is harmless (the probe confirms pickle and deepcopy
  keep both attributes), but the real cross-process risk is the store round-trip
  plus Stage-2/3 `load_zarr`, which other tasks already cover.
- **Missing:**
  - B1 (the Colony Viv grid).
  - B2 (version 4 is reused as the "future" sentinel).
  - M2 (the display range).
  - M5 (crop stores from before the change, remeasured).
  - The no-op crop producing a v4 store (M6).
  - Reader robustness when `canvas_shape` disagrees with the on-disk extent
    (MINOR-1).

**Fix.** Replace item 3 with B1 and add B2, M2 and M5, each with a pinning test.

### M4. The work-id "both producers agree" test cannot detect a worker that ignores the new key. This affects Task 13.

**Evidence.** In `test_both_producers_agree_for_a_crop_pipeline`, if the edit to
`_cli_process_single._worker_work_identity` (passing `pipeline_json=pipeline`) is
forgotten, both sides differ and the test catches it. But if *neither* side passes
it (for example, the edit lands in only one place and is later reverted), the test
still passes. Nothing asserts that the crop key actually reaches either producer
end to end.

**Fix.** Also assert that `selected` differs from
`compute_work_id(...same fields..., pipeline_json=None)`, so the crop
revision is proven present in the producer output, not only "equal on both sides".

### M5. Remeasured or recompiled crop stores from before this change emit `Frame_Offset* = 0`, which the schema defines as "not cropped". This affects spec §4.2/§6.4 and Task 7.

**Evidence.**
- `FRAME` `desc` says "0 when the image was not cropped".
- `--mode measure`/remeasure read through `load_zarr`. A pre-change crop store has
  no `crop_frame`, so `append_frame_offsets(info, None)` writes 0, 0.
- The store's own provenance journal records the `CropImage` op, so the image *was*
  cropped.

**What breaks.** Original-frame coordinates regenerated from those rows are
silently wrong by the crop margin. That is the exact failure the feature exists to
prevent.

The work-id key (Task 13) does force `full` continuation to re-derive. But
`--mode measure` on an old tree re-measures stores; it does not re-apply the
pipeline, so it cannot recover the offset.

**Fix.** Choose one:
- When an image has no frame but its journal contains a `CropImage`/`ImageCropper`
  operation, emit nullable offsets (`pd.NA` with an `Int64` dtype) and warn once.
- Refuse to measure such stores and point at re-running `full`.

Pin the chosen behaviour with a test. This needs a spec decision (SPEC-2).

### M6. A no-op `CropImage()` (every margin `None`) writes a v4 store that older builds refuse, although nothing was cropped. This affects Task 4 and spec §5.2.

**Evidence.**
- `CropImage()` slices `0:H, 0:W`. This yields frame `((H, W), (0, 0))` with
  `_pad_on_save = True`, and the save writes `padded: true` with version 4.
- Review Focus 4 pins this as correct.
- A builder node added with defaults, or a tune spec that leaves margins unset,
  silently produces v4 trees. The work-id key also cold-starts those runs.

**Fix.** Treat an identity frame (`offset == (0, 0)` and
`canvas_shape == shape`) as "no padding needed": write v3 with `padded: false`,
or with no key. Pin it with `test_noop_crop_writes_v3`.

---

## MINOR

### MINOR-1. The reader trusts `crop_frame` against the on-disk extent, and zarr truncates out-of-bounds slices silently. (Task 10)

**Evidence.** The probe shows that `a[(…, slice(8, 20), slice(0, 4))]` on a
10×12 array returns shape `(3, 2, 4)` with no error. A `crop_frame` whose
`canvas_shape` or `roi_shape` disagrees with the stored level-0 extent therefore
yields a truncated image instead of an error. `crop_frame_from_attribute` only
checks self-consistency.

**Fix.** In `_load_from_store`, for padded stores, assert that
`store_level0_shape(gray)[-2:] == canvas_shape` and that every windowed read
returns `roi_shape`. Raise `ValueError` naming `crop_frame` otherwise.

### MINOR-2. `valid_staged_store` does not validate `crop_frame`. (Task 8)

A v4 store with a malformed `crop_frame` passes `valid_staged_store`, which returns
True. Stage 2's `load_zarr` then raises, so the run aborts instead of routing the
image back to Stage 1 the way a malformed store normally is.

**Fix.** In `valid_staged_store`:
- When the key is present, parse it with `crop_frame_from_attribute` and catch
  `ValueError` → `False`.
- Require `version == 4` ⇔ `padded` is true.
- Require `canvas_shape` to equal the aligned extent.

### MINOR-3. Nothing checks that version 4 and `padded` agree. (Task 8)

A v4 store without `crop_frame` would load as an unpadded canvas. A v3 store
claiming `padded: true` would be windowed. Neither should be accepted. Fold this
into `require_readable_store` or the loader.

### MINOR-4. The deliverables README does not document `Frame_*`. (missed consumer)

`_cli/_cli_readme_generator.py:165-185` emits Object, `BBOX` and (for GridImage)
`GRID` tables as the public column documentation (root `CLAUDE.md`: the generator
"emits each member's `desc` as the public column documentation"). Every master now
carries `Frame_OffsetRR/CC` with no README row.

**Fix.** Add `self._generate_measurement_table(FRAME)` after the BBOX table, plus a
README test.

### MINOR-5. The spec's "same column set" claim is false across resumed continuations. (Task 7, spec §4.2)

A non-crop run resumed across the upgrade keeps its digest by design, so reused
stores keep tables without `Frame_*`. The master concatenation uses
`pl.concat(..., how="diagonal_relaxed")` (`_cli_parquet_agg.py:110,409`,
`_cli_finalize_run.py:145`), so those rows get null `Frame_*`, the column is
promoted, and there is no failure.

This is acceptable, but document it. Also add one aggregation test that mixes an
old-schema table with a new one and asserts the column is present and nullable.
Don't assume int64.

### MINOR-6. `save2pickle`/`load_pickle` drop the frame. (missed serializer)

`_image_io_handler.py:2076-2090` writes a fixed dict and `:2147-2211` rebuilds
from arrays. Neither carries `_crop_frame` or `_pad_on_save`.

**Fix.** Either persist both keys (with `.get` on load for old pickles) or document
that pickle files do not carry the frame.

### MINOR-7. The builder previews change behaviour and no test or spec line covers it. (Task 9)

`ImagePipeline.apply_with_intermediates` (`_image_pipeline_core.py:1040-1092`)
writes every post-crop node through `save_intermediate_zarr`/`save2zarr` with
`pad_on_save=None`. Builder previews of every node after a crop therefore become
canvas-sized, and `_preview_cache.py:252-262` reports the canvas `shape`.

That may be desired, but it is a user-visible GUI behaviour change that spec §7
does not list. `tests/unit/gui/builder` and `tests/gui/builder` would catch an
assertion on the ROI shape only at Phase 3 surface time.

**Fix.** Either state it in the spec, or pass `pad_on_save=False` from
`apply_with_intermediates` so previews show the ROI. Pin whichever is chosen.

### MINOR-8. `_valid_crop_frame()` mutates the caller's input image during capture. (Task 5)

The capture `source_frame = logical_input._valid_crop_frame()` runs on the
caller's object, and with `inplace=False` that is the object the user expects
untouched. When the frame is stale, the input loses its frame and flag as a side
effect of a "non-mutating" apply.

**Fix.** At capture, read `logical_input._crop_frame` and test
`.fits(shape)` without clearing. Leave the drop to the result's consumers.

### MINOR-9. The merge-audit tests cannot fail. (Task 7, Step 1b)

None of the refiners merges two `info()` frames. `KeepSectionLargest` merges
`grid.info()` with a two-column `{Object_Label, _pixel_area}` frame. The grid
measurers do not return `Frame_*`. So `test_refiners_run_on_a_cropped_grid` and
`test_grid_measurers_produce_no_suffixed_columns` pass before and after the change.

They are acceptable as regression pins, but the commit message should not present
them as the §4.3 audit. The real audit result is "every merge site is safe". The
one real collision path is `MeasureFeatures.measure(include_meta=True)`
(`abc_/_measure_features.py:451-456`), which merges `info()` with the measurer's
output on `OBJECT.LABEL`. Pin that instead.

The pipeline's own merge uses `suffixes=("", "_merged")`
(`_image_pipeline_core.py:1497`), so `test_pipeline_measurements_carry_offsets_once`
should also reject `_merged`, not only `_x`/`_y`.

### MINOR-10. The `tiles.py` crop test's premise holds only away from ROI edges. (Task 15)

For a colony whose outer boundary touches the ROI edge, the padded window includes
canvas pixels beyond the ROI. `composite_contours` and `find_boundaries` then draw
there, while the unpadded crop clamps, so the byte equality would fail for a
legitimate reason.

**Fix.** Choose `info.head(5)` rows guaranteed interior, for example by filtering
on a bbox margin of at least half the crop size. Otherwise the test is
fixture-luck.

### MINOR-11. The affected surfaces miss known crop users.

- `prefab/_spimager_pipeline.py:18` uses `CropImage(left=650, right=650, top=600,
  bottom=600)`. Add `tests/unit/prefab`.
- `tests/smoke/test_serialization.py` and `tests/smoke/test_operation.py`
  instantiate `CropImage` (Phase 1 runs `tests/smoke`, which is fine).
- `tests/unit/cli/test_cli_provenance_durability.py:88` runs a crop pipeline
  through the worker. Add it to Phase 4 explicitly.
- `tests/integration/cli/test_provenance_fencing.py` uses
  `staged_run_with_provenance`. Add it to Task 14 Step 2.

---

## NIT

- **N1 (witness script).** `crop_frame_invariants.py` checks with `assert`. Under
  `python -O` every check is stripped and the script exits 0. Use explicit
  `if not …: raise AssertionError` or `sys.exit(1)`.
- **N2 (Task 12).** Loosening `test_figures_in_store.py:268` to `>= 3` contradicts
  the plan's own "Do not loosen" rule. Delete the duplicate pin (Task 12 pins `== 4`
  in its own test) or update it to `== 4`.
- **N3 (Task 13).** The embedded-JSON-string branch of `_json_names_crop_class` is
  dead code. `OperationField` serialises `{"class", "params"}` (`sdk_/typing_.py:285-312`),
  and nested pipelines use the `pipeline_operation`/`config` envelope
  (`_serializable_pipeline.py:463-470`), both of which are plain dicts. The
  `lru_cache` is also premature: `work_id_for_image` already reads the whole file
  for `file_sha256` per image. A plain function is simpler.
- **N4 (Task 1).** `rect_origin_from_key` rejects `(r0:r1, c0:c1, :)`, a rectangle
  with an explicit full channel slice. That is harmless, but the docstring calls it
  "a third channel index".
- **N5 (Task 9 test).** `test_original_series_is_unchanged` retains a ROI-sized
  original after the crop, which is the opposite of the CLI's canvas-sized
  snapshot, and checks only presence. Retain before cropping
  (`plate._retain_original()`) and assert the original series' shape equals the
  canvas.
- **N6 (Task 11).** For the colour-space `imsave` (`_color_space_accessor.py:120`),
  "gets the keyword and passes it through" has nothing to pass to: it does not call
  the layer `imsave`. Say "not a layer; unchanged".
- **N7 (spec §10).** The Risks table names a "single `to_canvas_coords` helper"; §7
  names `ngff_.padded_crop_offset`. Align the names.

---

## SPEC

- **SPEC-1 (§7).** "Viv/deck.gl … lines up automatically" is true for the Plate
  layers and false for the Colony grid, which targets per-colony cameras from table
  centroids (B1). Amend §7 to list the Colony grid as a table→store consumer.
- **SPEC-2 (§4.1/§4.2).** "0 when the image was not cropped" cannot be guaranteed
  for stores written before this change (M5). Decide whether old crop stores get
  null offsets, a refusal, or a documented caveat.
- **SPEC-3 (§3.4, PadImage).** A frame kept after `PadImage(mode="edge"|"reflect"|…)`
  or `constant_value != 0` writes **fabricated** pixels at true canvas positions in
  a padded store. This contradicts §1.2: "zero blanks mark the cropped region and
  are never mistaken for an enhanced or processed zone". Either drop the frame for
  any non-zero fill, or accept and document it.
- **SPEC-4 (§5.3).** The round-trip invariant ("equal `_pad_on_save`") is false
  under an explicit `pad_on_save=False` override: the loaded flag becomes
  `padded = False`. The plan's test (`test_unpadded_round_trip_…`) encodes the
  sensible behaviour. Scope the invariant to `pad_on_save=None`.
- **SPEC-5 (§5.2).** Make version 4 ⇔ `padded: true` a *read* invariant, not only a
  write rule (MINOR-3). Also consider not writing v4 for an identity frame (M6).
- **SPEC-6 (§3.5).** The carry-over copies the frame onto any same-shape result
  with no frame. A same-shape op that legitimately produces an unrelated image (a
  registration or replacement op) inherits a frame that no longer describes its
  pixels. This is low risk today; note it as a known limitation beside the rotation
  one.

---

## Verified correct (checked against code; probe where noted)

**Core image model**
- `ImageDataManager._set_from_class_instance` (`_image_data_manager.py:330-368`)
  calls `_set_from_array` first and then copies `_data` wholesale. Copying the frame
  after the `_original` assignment is the right point: the reshape drop fires on the
  destination's *old* shape before the source's frame overwrites it.
- `Image.__init__`'s first `_set_from_array` sees `previous == (0, 2)` from
  `ImageData.clear()`, with the frame `None` (class default). No spurious drop.
- `copy()` is `self.__class__(self)` (`_image_handler.py:694-704`), and `GridImage`
  takes `arr=Image` (`_grid_image_handler.py:59-101`). Both route through
  `_set_from_class_instance`, so the frame is carried.
- `ImageRGB.__setitem__` re-enters `_set_from_array` with the same shape
  (`_rgb_accessor.py:166`). `rotate()` keeps the shape (`_image_handler.py:790-841`).
- `CropImage._operate` (`_image_cropper.py:145-186`) does the following, all
  carrying the frame:
  - slices through `__getitem__` (both `Image` and `GridImageHandler` paths);
  - rebuilds a `GridImage(arr=cropped)`;
  - `set_image`s it onto the copy;
  - sets the name.
- `PadImage._operate` (`_image_padder.py:211-282`) writes `_data` directly before
  the `GridImage` rebuild. The plan's insertion point, before the rebuild, is
  correct because the rebuild copies the frame.
- `ObjectsAccessor.__getitem__` slices with `regionprops.slice`. The probe shows
  `(slice(45, 87, None), slice(654, 696, None))`, which is unit step, so objects
  compose.

**Provenance wrapper**
- The wrapper `_recording_apply` (`_provenance.py:620-741`) has exactly one
  `_carry_logical_image_state` call (`:684`) and the capture block at `:644-645`.
- `ImageOperation.apply` copies when `inplace=False` (`abc_/_image_operation.py:471-492`).
- `ImageCorrector.apply` has no integrity check, so `_RebuildFresh` works.

**Probe results**
- `pickle.dumps/loads` and `copy.deepcopy` of an `Image` keep instance attributes.
- An empty-frame `assign(np.int64(0))` yields int64.
- `load_synth_yeast_plate()` is a 600×800 uint8 `GridImage` with 96 objects.
- zarr v3 `(Ellipsis, slice, slice)` selection works.

**Schema and measurement surfaces**
- `schema/_bbox.py` is the template.
- `IdentityInfo` lives in `schema/_tiers.py`.
- `order_measurement_columns` is at `sdk_/_metadata_helpers.py:111-155`, with the
  info-block test at `:140`. The Task 6 ordering test fails before the change and
  passes after it.
- Identity-kind enums are excluded generically from measurement pickers
  (`plotting/_plot_meas_time_series.py:398-409`).
- `test_classification.py:171-179` is the list to extend.
- Both `info()` paths feed `_get_image_info` (`_image_pipeline_core.py:1269-1272`).
- `GridAccessor.info` calls `grid_finder.measure`, and that call reaches
  `objects.info()` (`abc_/_grid_finder.py:419-420`). `assign` overwrites, so there
  are no duplicate `Frame_*` columns.

**Store versioning and readers**
- `ngff_.py:57` defines `STORE_SCHEMA_VERSION = 3`.
- `PhenotypicAttr` is at `:455`.
- `build_phenotypic_attributes` is at `:522-628`.
- `require_readable_store` is at `:671-704`.
- The `valid_staged_store` version gate is at `:2003-2006`, and its extent check
  excludes `original` (`:2032-2042`), so canvas-padded series agree.
- The browse gate is at `_gui/browse/_tile_routes.py:215-219`.
- The results-viewer routes and tiles use `require_readable_store`.
- `read_ngff_image_spec` (imread) is ungated by design.
- Root-rewriting paths preserve the whole `phenotypic` block, so `crop_frame` and
  the version survive measure re-promotes and provenance upgrades:
  - `replace_image_tables` (`_measurement_tables.py:676-706`);
  - `write_provenance_checkpoint` (`_provenance.py:1062-1094`);
  - `upgrade_store_provenance` (`_cli_migrate_provenance.py:159-187`).

**Writer and loader details (Tasks 9–10)**
- An uncropped store stays bit-identical: with no frame, `_as_written` and
  `_written_shape` are identities and no key is added.
- `_read_store_array` (`:1793-1816`) and `_load_from_store` (`:1671-1790`) take the
  window change as planned. `GridImage._load_from_store` delegates to the base, so
  the frame restore runs for both classes.

**Staged GPU and process mode**
- The staged GPU path reads stores only through `load_zarr`
  (`_cli_staged_workers.py:456, 516, 543`; `_cli_staged_strategy.py:482`) and writes
  through `OutputManager.save_image_store` → `save2zarr` (`_cli_output_manager.py:~1939`).
- The Stage-2 token `objmap_shape` is ROI-sized on both sides.
- `staged_run_with_provenance` (`tests/integration/cli/conftest.py:303-314`) runs
  `CropImage(1, 1, 1, 1)` on the 600×800 plate. The harness methods `store()`,
  `run_stage1/2/3()`, `slot`, `read_measurements()` and `load_stage2_raw` exist with
  the signatures Task 14 uses.
- `write_process_only_layer` (`_cli_process_only.py:156-260`): the zarr branch
  computes levels from `image.shape` (Task 12 replaces that), and the tiff branch
  calls `accessor.imsave(filepath=…)` with no other arguments, so the default
  resolution applies.

**Work-id (Task 13)**
- `compute_work_id` (`_cli_failure_tracker.py:304-329`), `work_id_for_image`
  (`:368-398`) and `_worker_work_identity` (`_cli_process_single.py:123-170`) are the
  only producers.
- `TIFF_DIGEST_AT_81D19EC` and the `_FIXED` fields match
  `tests/unit/cli/test_work_id_raw_revision.py:23-38`.
- `make_config` (`tests/unit/cli/_preflight_support.py:19-43`) accepts
  `pipeline_json`/`input_path`.
- `ImageCropper` is the only alias (`sdk_/_class_aliases.py:15`).
- The JSON tree scan is valid: every nested op serialises as `{"class", …}`.
- A non-crop pipeline's digest is unchanged because the key is added only when
  `pipeline_json` names a crop.

**Accessor `imsave` (Task 11)**
- The three `imsave` sites exist at the cited lines, and `ObjectMap.imsave` calls
  `super().imsave` (`_objmap_accessor.py:612`).

**Test files and imports**
- Every test file, directory and import path the plan names exists:
  - `KeepSectionLargest`, `KeepNearestCenter` and `GridOversizedObjectRemover` are
    exported from `phenotypic.refine`.
  - `MergeWithinSection` is not exported.
  - `MeasureGridLinRegStats`, `MeasureGridSpread`, `MeasureNeighborDist` and
    `MeasureSize` are exported from `phenotypic.measure`.
  - `ImagePipeline.apply_and_measure` exists (`_image_pipeline_core.py:1328`).

**Task 1**
- The unit tests are internally consistent with the given implementation; each
  case was traced by hand.

## Existing-code bugs noticed (not caused by this plan)

- `_image_pipeline_core.py:1489-1495` (`_merge_on_object_labels`) compares
  `df[col_new_df] == df[col_other_df]`, which is the *other* frame against itself.
  The check is always true except where the column holds NaN, which makes it false,
  so any shared column with a NaN becomes `_merged`-suffixed instead of merged on.
  It was clearly meant to compare `new_df[col]` with `df[col]`.
- `CropImage._get_idxes` (`_image_cropper.py:~270-290`) allows `top == bottom_idx`
  (`>` rather than `>=`). That produces an empty image, which later operations
  cannot handle, despite the docstring's claim that "top edge >= bottom edge"
  raises.
