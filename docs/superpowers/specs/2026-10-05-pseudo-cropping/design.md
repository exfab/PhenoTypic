# Pseudo-cropping: CropImage records a crop frame instead of discarding geometry

- **Date:** 2026-10-05
- **Branch / worktree:** `worktree-pseudo-cropping` (`.claude/worktrees/pseudo-cropping`)
- **Status:** design approved in conversation; awaiting written-spec review

## 1. Objective

Today `CropImage` physically shrinks every layer of an `Image` and nothing records
where the surviving region sat in the original frame. Downstream analysis can then no
longer be compared against the original image.

After this change:

1. **Ops are unaffected.** Every operation after a crop still sees only the ROI.
   `image.shape`, `image.rgb[:]`, `image.detect_mat[0:50]` behave exactly as now. On
   an image top-cropped by `t`, `image.rgb[0:50]` *is* rows `t:50+t` of the original
   frame, and that offset is now recorded.
2. **Layers on disk are full-frame.** A cropped image's OME-Zarr store holds `rgb`,
   `gray`, `detect_mat` and the `objmap` label at the **original H×W**, with the ROI
   at its true position and **zeros everywhere outside it**. The zero blanks mark the
   cropped region and are never mistaken for an enhanced or processed zone. The
   real source pixels stay in the existing `"original"` series.
3. **Offsets are in `info()`.** Every `info()` table, and therefore every measurement
   table, carries `Frame_OffsetRR` / `Frame_OffsetCC`. Original-frame coordinates
   are regenerated as `Bbox_* + Frame_Offset*`. **Every existing value is
   unchanged.**

### Non-goals

- Reversible or adjustable ROI (widening a crop to recover outside pixels).
- Live in-memory full-frame arrays. Full frame exists on disk; in memory the arrays
  stay compact.
- Padding per-image figures or deliverable overlay PNGs. They stay ROI-sized.
- A rotation-aware frame (§3.4).
- Any change to existing `Bbox_*`, grid, or centroid values.
- New GUI chrome. No `FEATURES.md` / `WORKFLOWS.md` ledger change.

## 2. Approach (chosen: "C")

Three approaches were considered:

- **A, full-frame ROI view.** Backing arrays stay canvas-sized and accessors
  translate keys. Rejected: it must translate every NumPy index form, and every
  direct `_data.*` user has to be rewritten. There are 7 outside `_core`: the
  padder, denoisers, colour corrector and inoculum detector. A missed site silently
  leaks zeros into an op. It also costs full-frame memory in every op.
- **B, compact ROI + recorded frame.** Memory stays compact, and padding happens
  only at the write boundary.
- **C, B plus composition through slicing.** **Chosen.** Every rectangular
  `Image.__getitem__` composes the frame, so `plate.objects[i]` also knows its
  position on the plate.

Because the outside region is defined as zero, a canvas-sized backing buffer would
carry no information beyond `(H, W, row, col)`. B/C therefore observe the same
semantics as A at a fraction of its risk.

## 3. Data model and in-memory semantics

### 3.1 `CropFrame`

A frozen dataclass in `phenotypic._core` (private module):

```python
@dataclass(frozen=True)
class CropFrame:
    canvas_shape: tuple[int, int]   # original (H, W) the ROI lives in
    offset: tuple[int, int]         # (row, col) of the ROI's top-left in the canvas
```

`ImageDataManager` gains two protected attributes:

| Attribute | Type | Default | Meaning |
|---|---|---|---|
| `_crop_frame` | `CropFrame \| None` | `None` | where this image's pixels sit in the original canvas |
| `_pad_on_save` | `bool` | `False` | whether writers pad to the canvas by default |

`ImageDataManager.clear()` resets both. `_set_from_class_instance` copies both, the
same way it copies `_original`, **after** its `_set_from_array` call. So `copy()`,
`set_image(other_image)` and `Image(other_image)` preserve them.

**Validity invariant.** A frame is valid for an image of 2-D shape `(h, w)` iff
`offset >= (0, 0)` and `offset + (h, w) <= canvas_shape`, element-wise. A single
helper, `_valid_crop_frame() -> CropFrame | None`, returns the frame when valid. When
invalid, it emits a `UserWarning`, clears the frame and `_pad_on_save`, and returns
`None`. **Every consumer (§4, §5) reads the frame through this helper and never
through the attribute directly**, so a stale frame is dropped rather than written
out as a wrong offset.

### 3.2 Composition through slicing

`ImageHandler.__getitem__(key)` (`_image_handler.py:88`) keeps building the subimage
exactly as today, then sets its frame:

- If `key` resolves to a unit-step rectangle `(r0:r1, c0:c1)` (a slice, or a tuple of
  one or two slices with `step in (None, 1)`, resolved with `slice.indices` against
  the parent's 2-D shape):
  - `canvas_shape` = the parent's valid frame's `canvas_shape`, else the parent's
    `shape[:2]`
  - `offset` = the parent's valid frame's `offset` (else `(0, 0)`) `+ (r0, c0)`
- Any other key (an int index, step ≠ 1, an ellipsis with channel indexing, fancy or
  boolean indexing) produces a child with `_crop_frame = None`.
- In every case `_pad_on_save = False`. **Plain slicing never turns padding on.**

So `plate.objects[i]` on a cropped plate carries `offset = plate_offset + bbox_min`
and saves at its own size by default.

### 3.3 `CropImage`

Parameters and validation are unchanged. `_operate` still slices with
`image[top:bottom_idx, left:right_idx]`, which now composes the frame, and still
`set_image`s the result (including the `GridImage` rebuild). That carries the frame
across. It then sets `image._pad_on_save = True`.

- Crop then crop composes: offsets add up, and the canvas stays the first image's
  shape.
- An uncropped image has `_crop_frame = None` and behaves exactly as today
  everywhere.

### 3.4 Other shape-changing paths

| Path | Rule |
|---|---|
| `PadImage._operate` (writes `_data.*` directly) | Frame-aware: `offset -= (pad_top, pad_left)`. If the padded image still satisfies the validity invariant, the frame is kept with the same `_pad_on_save`. Otherwise **drop the frame with a `UserWarning`** and fall back to today's behaviour. An image with no frame is unaffected. |
| `set_image(ndarray)` / `_handle_array_input` with a different 2-D shape | Drop the frame and `_pad_on_save`. Same-shape writes (`rgb[...] = …`, which re-enters `_set_from_array`) keep both. |
| `image.rotate()`, `GridAligner`, `ImageCorrector`/`GridCorrector` rotation | Keep the frame (shape is unchanged). **Documented limitation:** the frame is a translation record. Rotating *before* a crop means offsets are in the rotated canvas's coordinates, not the raw `"original"` series'. The provenance journal already records the rotation. |
| Any other op that reshapes `_data` | Caught by the §3.1 guard at the next consumer: dropped with a warning. |

**PadImage fill inside a kept frame (decided 2026-10-05).** When a kept frame follows
`PadImage(mode="edge"|"reflect"|…)` or `constant_value != 0`, the pad pixels are
part of the analysed image. They are written at their canvas positions in a padded
store, so the store shows exactly what the analysis saw. Zeros still mark only
never-analysed area. This is documented on `PadImage`, not prevented.

### 3.5 Carry-over through the apply wrapper

`_carry_logical_image_state` (`_provenance.py:484`) re-attaches `_original` from
the logical input onto the op's result. It gains one rule. **If the result has no
frame, the logical input has a valid frame, and the result's 2-D shape equals the
input's, copy the input's frame and `_pad_on_save` onto the result.** This covers a
same-shape op that rebuilds its result as a fresh `Image`. A result that already has
a frame (e.g. `CropImage`'s) is left untouched. A result with a changed shape and no
frame gets none.

The capture must not mutate the caller's input. It reads `_crop_frame` and tests
`.fits()` without clearing; the clearing guard runs only on the result's consumers.

**Known limitation.** A same-shape op that legitimately produces an unrelated image
(a registration or replacement op) would inherit a frame that no longer describes
its pixels. No such op exists today; this is noted beside the rotation limitation.

### 3.6 Unchanged

Every accessor, every op body, `image.shape`, and all 7 out-of-`_core` `_data.*`
sites stay as they are. In memory, ops still see a compact ROI.

## 4. `info()` and the measurement schema

There are two `info()` paths, and both feed `ImagePipeline.measure` through
`_get_image_info` (`_image_pipeline_core.py:1270`):

- `ObjectsAccessor.info` → `MeasureBounds().measure(image)`
- `GridAccessor.info` → `grid_finder.measure(image)`. This is a separate path;
  changing `MeasureBounds` alone would miss GridImages.

### 4.1 Schema

A new public `IdentityInfo` enum `phenotypic.schema.FRAME`, category `"Frame"`,
exported from `phenotypic.schema`:

```python
class FRAME(IdentityInfo):
    OFFSET_RR = Entry(
        "OffsetRR",
        "Row offset of the analysed region's top-left pixel within the original "
        "image frame. 0 when the image was not cropped; null when it was cropped "
        "but the offset was not recorded. Original-frame row = "
        "Bbox_*RR + Frame_OffsetRR.",
    )
    OFFSET_CC = Entry(
        "OffsetCC",
        "Column offset of the analysed region's top-left pixel within the original "
        "image frame. 0 when the image was not cropped; null when it was cropped "
        "but the offset was not recorded. Original-frame column = "
        "Bbox_*CC + Frame_OffsetCC.",
    )
```

Per the schema authoring rule, `bio_desc=""` and `image=None` are left for human
authoring.

### 4.2 Behaviour

- One private helper, `_append_frame_offsets(info, image) -> DataFrame`, is called
  at the end of **both** `ObjectsAccessor.info` and `GridAccessor.info`, before
  `insert_metadata`. It reads the frame via `_valid_crop_frame()`.
- The columns are **always emitted**. Their value is:
  - the frame's offset when the image has a valid frame (`int64`);
  - **`0, 0`** when it has no frame and its provenance journal records no
    `CropImage`/`ImageCropper` operation, i.e. it was genuinely never cropped
    (`int64`);
  - **null** (pandas nullable `Int64`, `pd.NA`) when it has no frame **but its
    journal records a crop**. This covers stores written before this change and
    re-measured with `--mode measure`, and frames dropped by the §3.1 guard or by
    `PadImage` overflow. The offset is genuinely unknown, so `0` would silently
    lie. A `UserWarning` is emitted once per image, telling the user to re-run
    `--mode full` to recover the offsets. (Decided 2026-10-05.)
- Cropped and uncropped images concatenate cleanly into `master_measurements.parquet`.
  A *non-crop* run resumed across the upgrade keeps its work id by design, so its
  reused stores' tables lack `Frame_*`. Aggregation (`pl.concat(...,
  how="diagonal_relaxed")`) fills those rows with null, so the column is present
  and nullable in that one case.
- **Every existing `Bbox_*`, grid and centroid value is byte-identical** to today:
  still in ROI coordinates.
- The columns sit in the info block of the canonical order
  (`[metadata] → [measurements] → [Metadata_] → [info]`). The plan verifies
  `order_measurement_columns` places category `Frame` there.

### 4.3 Merge-collision audit (plan task)

Several ops merge `info()` tables with other frames: `refine/_keep_section_largest.py:61`,
`refine/_keep_nearest_center.py:57`, `refine/_grid_oversized_object_remover.py:69`,
`refine/_merge_within_section.py:64`, `measure/_measure_grid_linreg_stats.py:66-68`,
`measure/_measure_neighbor_dist.py:195,208`, `measure/_measure_grid_spread.py:54`,
`measure/_orientation_zones/_operation.py:347`, and `_grid_accessor.py`'s internal
callers. If both sides of a merge carry `Frame_*`, pandas adds `_x`/`_y` suffixes.
The plan enumerates every `.info(` call site mechanically (`grep`), and each one is
either confirmed safe or fixed.

## 5. Persistence

### 5.1 Writing

`save2zarr`, `_save_store`, and `save_intermediate_zarr` gain a keyword argument
`pad_on_save: bool | None = None`. **`None` resolves to `self._pad_on_save`**, and
an explicit `True`/`False` overrides it. The override lives on the save function
itself.

When the resolved value is `True` and `_valid_crop_frame()` returns a frame `F`:

- `rgb`, `gray`, `detect_mat` and the `objmap` label are written at
  `F.canvas_shape`, with the ROI at `F.offset` and **0 everywhere else**. That is 0
  in every channel and dtype, and label 0 means background.
- The pyramid level count is computed from the canvas shape.
- The `"original"` series is unchanged (it is already canvas-sized).
- Peak extra memory is bounded to **one canvas-sized layer at a time**: pad, write,
  release, before the next layer.

When the resolved value is `True` but there is no valid frame, the store is written
unpadded, exactly as today.

### 5.2 Store attributes and versioning

Whenever a valid frame exists, padded or not, the root `attributes.phenotypic`
block gains:

```json
"crop_frame": {"canvas_shape": [H, W], "offset": [r, c], "roi_shape": [h, w], "padded": true}
```

`roi_shape` is the image's own 2-D shape at write time. The reader needs it to slice
a padded layer back to the ROI, because `canvas_shape` and `offset` alone do not
determine the ROI's extent. An unpadded `objects[i]` store therefore still records
where it came from.

**Versioning, scoped to padded stores only.** The read gate is exact equality,
`found != STORE_SCHEMA_VERSION` (= 3, `ngff_.py:697`). A padded store means
something different to a reader that doesn't know about `crop_frame`: such a
reader would load the zero-padded canvas as the image while `Bbox_*` stays in ROI
coordinates.

- A store written with `padded: true` writes `store_schema_version = 4`. **Every
  other store keeps writing 3**, so it stays bit-identical to today's output.
- This build reads `{3, 4}`. `STORE_SCHEMA_VERSION` splits into a written value per
  case plus a `READABLE_STORE_SCHEMA_VERSIONS` set. Every gate (`ngff_.py:697`, the
  validity check at `ngff_.py:2004`, `load_layer_zarr`) accepts the set.
- An older build refuses a padded store with its existing "this build reads 3"
  message rather than misreading it. Existing trees need no `--mode migrate`.
- **Read invariant:** version 4 ⇔ `crop_frame.padded == true`. A v4 store without a
  padded `crop_frame`, or a v3 store claiming `padded: true`, is refused by the
  loader and rejected by `valid_staged_store`. So is a `crop_frame` that does not
  parse, or whose `canvas_shape` disagrees with the stored level-0 extent. The
  staged engine then routes the image back to Stage 1 instead of aborting.
- **A no-op `CropImage()` still writes a padded v4 store** (decided 2026-10-05). The
  rule is "every `CropImage` result saves padded", with no special case for an
  identity frame.
- The existing tests that use `STORE_SCHEMA_VERSION + 1` (= 4) as "a newer,
  unreadable version" are moved to `max(READABLE_STORE_SCHEMA_VERSIONS) + 1`.

### 5.3 Reading

- **`load_zarr`**: if `crop_frame.padded`, slice every layer (rgb, gray, detect_mat,
  objmap) back to the ROI, then restore `_crop_frame` and
  `_pad_on_save = padded`. If not padded, restore the frame only. If there is no
  `crop_frame`, the image has no frame.
  **Round-trip invariant** (for `pad_on_save=None`): `Image.load_zarr(img.save2zarr(p))`
  equals `img` (`__eq__`), and has an equal `_crop_frame` and `_pad_on_save`. With an
  explicit override, the loaded `_pad_on_save` is the `padded` value actually
  written. The loader checks that every windowed read returns `roi_shape` (zarr
  truncates out-of-bounds slices silently) and raises `ValueError` naming
  `crop_frame` otherwise.
- **`save2pickle` / `load_pickle`** persist `_crop_frame` and `_pad_on_save`. They
  read them with `.get`, so old pickle files load without a frame.
- **`load_layer_zarr`** (the GUI tile server's reader) and **`Image.imread(store)`**
  return the **on-disk canvas** unchanged. That is the "full-frame layers on disk"
  view.

### 5.4 Embedded measurement table and figures

The table mechanism is unchanged; it now carries the `Frame_*` columns. Per-image
figures are drawn on the in-memory ROI and are not padded (non-goal).

## 6. CLI

### 6.1 Staged GPU engine

Stage 2 and Stage 3 both read the per-image store through `load_zarr`
(`_cli_staged_workers.py:456,516,543`; `_cli_staged_strategy.py:482`), so they see
the ROI. The retained raw `.npy` stays ROI-sized, `ReplayDetector` lines up, and
Stage 3's re-promotion writes a padded store. No change to the Stage-2 signal
format.

### 6.2 `--mode process`

`write_process_only_layer` (`_cli_process_only.py`) follows the pad rule on both
branches:

- `zarr` (`rgb`/`gray`): `_save_store(..., pad_on_save=None)`, which resolves to
  the image's value. The level count comes from the canvas when padded.
- `tiff` (flat `rgb`/`gray`/`detect_mat` TIFF, `objmap` PNG): the flat branch
  writes through the accessor `imsave` methods. Those methods
  (`AccessorIOHandler.imsave`, `MultiChannelAccessor.imsave`, `ObjectMap.imsave`)
  gain the same `pad_on_save: bool | None = None` keyword with the same
  resolution rule, so every way of saving an image layer honours one rule. Overlay
  writers (`save_overlay`) are not padded (non-goal).

`PROCESS_LAYER_SEMANTICS_REVISION` (`_cli_failure_tracker.py:209`) goes from **3 to
4**, so a process tree from before the change is re-derived.

### 6.3 `full` / `measure` continuation

The work-id payload (`_cli_failure_tracker.py`) gains `"crop_frame_semantics": 1`
**only when the pipeline's operation tree contains a `CropImage`** (detected
tree-wide, like `find_gpu_detectors`). Crop pipelines re-derive their stores instead
of mixing ROI-sized old stores with canvas-sized new ones in one tree. **Every other
pipeline's digest is unchanged**, so no unrelated in-flight continuation
cold-starts. This follows the placement rule the module already documents for
`process_format` and `layer_semantics`.

### 6.4 Unchanged

`recompile`, remeasure and `migrate` read through `load_zarr`. Pre-existing stores
have no `crop_frame`, which means no frame.

## 7. GUI consumers

Viv/deck.gl reads store chunks directly, so the Plate view's image and label layers
show the full canvas and line up automatically. Plate contrast comes from the dtype
domain (`viv_viewer.js` `dtypeDomain`), so padding does not affect it. Code that
maps **table coordinates onto stored pixels** must add the offset:

- **Results Colony grid (Viv)**: `colony_view/_grid.py:_build_cell` serialises
  `Bbox_CenterRR/CC` as `centroidRr/centroidCc`, which `viv_viewer.js` treats as
  store pixel coordinates (camera targets and tile prefetch). `build_source_spec`
  (`_store_source.py`) gains `"cropOffset": [row, col]` from
  `ngff_.padded_crop_offset(block)`, and `_build_cell` adds it to the centroid
  before serialising. The JS is unchanged.
- **Server-side colony crops** (`tiles.py` `_crop_store_layer_window`, behind the QC
  gallery and the `/crops` route): the crop window is computed in **ROI**
  coordinates against the ROI's extent, then shifted into the canvas for the read.
  A crop from a padded store is then byte-identical to one from an unpadded store,
  including colonies at the ROI edge, contours, and per-window `detect_mat`
  normalisation.
- **Display range** (`tiles.py:image_display_range`): reads only the ROI window of
  the smallest pyramid level (offset and `roi_shape` scaled by the level/canvas
  ratio), so the zero margin does not drag `lo` to 0 and wash out 16-bit crops.
- **Builder node previews** (decided 2026-10-05): `apply_with_intermediates` passes
  `pad_on_save=False`, so previews keep showing the cropped region exactly as
  today. Padding applies to run outputs, not previews.

Unchanged consumers (audited):

- **`_gui/results_viewer/_curation_labels.py:483,512`**: centroids are used only as
  identity fingerprints compared against the same table, never placed onto pixels.
- Extent-only consumers (`colony_view/_grid.py:321`,
  `_qc_tab/review/_callbacks.py:1551`) use `Max − Min`, which an offset does not
  change.
- The baked overlay-PNG fallback is drawn on the ROI and has no store, so it is
  never shifted.

One shared helper, `ngff_.padded_crop_offset(block)` (with `ngff_.padded_crop_window`
returning canvas, offset and ROI shape), decides "is this pixel source padded?" in a
single place. It reads the store's own `crop_frame` attribute, so the **store**, not
the measurement table, is the source of truth for the shift. Old masters without
`Frame_*` columns therefore need no special case.

## 8. Testing

Use TDD throughout. Run focused tests per task, the affected surface per phase, and
the full sharded regression once at the end (per root `CLAUDE.md`).

- **Frame semantics:** a plain slice; `CropImage`; crop∘crop; `objects[i]` on a
  cropped plate; `PadImage` inside the canvas (kept) and overflowing it (dropped with
  a warning); rotate (kept); `set_image(ndarray)` reshape (dropped) vs. same-shape
  write (kept); a stale frame dropped by the guard; a non-rectangular key gives no
  frame; carry-over through the apply wrapper for a same-shape rebuilt result.
- **`info()`:** `Frame_*` present on both paths, `0` when there is no frame. **`Bbox_*`
  byte-identical to `main`** for a crop pipeline (golden comparison). The audited
  merge sites produce no `_x`/`_y` columns.
- **Store:** round-trip identity; zeros outside the ROI on disk; ROI bytes equal; v4
  only when padded, v3 otherwise, and an unpadded store is bit-identical to today's;
  v3 stores remain readable; a gate restricted to `{3}` rejects v4; the `pad_on_save`
  override in both directions; `imread`/`load_layer_zarr` return the canvas.
- **CLI:** the process revision bumps; the work-id digest is **pinned unchanged** for
  a pipeline without a crop and changes for a crop pipeline; a staged-GPU crop
  round-trip with a fake `GpuDetector`; `--mode process` tiff and zarr outputs are
  canvas-sized.
- **GUI:** a tile crop with an offset on a padded store and none on the overlay
  fallback; old masters without `Frame_*`.
- **Logic validation:**
  `docs/superpowers/logic_validation_scripts/2026-10-05-pseudo-cropping/crop_frame_invariants.py`
  (stdlib + numpy, does not import `phenotypic`). It re-derives offset composition,
  the pad/slice identity, original-coordinate regeneration, and the pad-overflow
  rule.

## 9. Documentation

`CropImage` and `PadImage` docstrings; the `crop_and_pad` how-to;
`docs/source/how_to/pages/zarr_storage.md`; `src/phenotypic/_core/CLAUDE.md` (frame
attributes, the composition rule, the §3.1 guard); `src/phenotypic/schema/CLAUDE.md`
if it lists categories; the root `CLAUDE.md` store-layout notes (padded stores, v4);
the `working-with-ome-zarr` skill.

## 10. Risks

| Risk | Mitigation |
|---|---|
| A geometry-changing op not covered by §3.4 leaves a stale frame | §3.1 guard at every consumer drops it with a warning; never a wrong offset |
| `Frame_*` merge collisions in refine/measure ops | §4.3 mechanical audit of every `.info(` call site |
| An older build misreads a padded store | v4 on padded stores only (§5.2) |
| A mixed ROI/canvas store tree after upgrade | Crop-only `crop_frame_semantics` digest key (§6.3); process revision bump (§6.2) |
| A GUI tile crop or Colony view misplaced on a padded store | Single `ngff_.padded_crop_offset` helper read from the store; overlay fallback explicitly not offset (§7) |
| A deliverables README missing the new columns | `_cli_readme_generator.py` emits a `FRAME` table after `BBOX` |
| Rotate-then-crop offsets misread as raw-original coordinates | Documented limitation (§3.4); rotation is in provenance |
